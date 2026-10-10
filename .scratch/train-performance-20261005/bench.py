"""临时训练性能实验；只读正式 checkpoint，所有结果留在本目录。"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/home/tml104/Mortal')
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'mortal'))
import torch
from torch import nn
from precompute_pipeline.common.settings import load_settings
from precompute_pipeline.train import train as training
from precompute_pipeline.train.reader import DatasetBatchReader


def forward(batch, models, settings, args):
    """保留生产损失；可单独测量移走设备端 Python 判断及 BF16。"""
    obs, actions, masks, steps, rewards, ranks = batch
    count = len(obs)
    rows = torch.arange(count, device='cuda')
    if args.check_mask and not masks[rows, actions].all():
        raise RuntimeError('invalid action')
    target = (settings.loss.gamma ** steps * rewards).to(torch.float32)
    dtype = torch.float16 if args.precision == 'fp16' else torch.bfloat16
    with torch.autocast('cuda', dtype=dtype):
        phi = models[0](obs)
        q_all = models[1](phi, masks)
        q = q_all[rows, actions]
        dqn_loss = .5 * nn.functional.mse_loss(q, target)
        cql_loss = q_all.logsumexp(-1).mean() - q.mean()
        logits, = models[2](phi)
        rank_loss = nn.functional.cross_entropy(logits, ranks)
        loss = dqn_loss + settings.loss.cql_weight * cql_loss + settings.loss.rank_weight * rank_loss
    accuracy = (logits.argmax(-1) == ranks).float().mean()
    metrics = torch.stack((loss.detach(), dqn_loss.detach(), cql_loss.detach(),
                           rank_loss.detach(), accuracy.detach(), q.detach().mean(), target.detach().mean())).double()
    return count, loss, metrics


def main(args):
    """使用同一已保存模型和样本顺序，预热后以同步边界计时。"""
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.mark_eval:
        original_metrics = training._forward_metrics
        def marked_metrics(*values, **keywords):
            torch.compiler.cudagraph_mark_step_begin()
            return original_metrics(*values, **keywords)
        training._forward_metrics = marked_metrics
    base = load_settings(ROOT / 'mortal/precompute_pipeline/config.toml').training
    settings = replace(base, batch_size=args.batch, enable_compile=args.compile != 'none',
                       compile_dynamic=args.compile == 'dynamic')
    torch.backends.cudnn.benchmark = True
    state = torch.load(settings.output_dir / 'latest.pth', weights_only=True, map_location='cpu')
    seed = state['main_seed']
    cursor = state['epoch_samples_consumed']
    epoch = state['current_epoch']
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    raw, models = training._build_models(replace(settings, enable_compile=False), torch.device('cuda'))
    for model, key in zip(raw, ('mortal', 'dqn', 'aux_net')):
        model.load_state_dict(state[key])
    reference = None
    if args.layout:
        from layout import convert_brain, build_optimizer
        probe_reader = DatasetBatchReader(settings.dataset_progress, split='train', batch_size=32, device='cuda', seed=seed, epoch=epoch, prefetch=False)
        probe_iterator = probe_reader.iter_from(cursor)
        probe = next(probe_iterator)[0]
        probe_iterator.close()
        old_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        raw[0].eval()
        with torch.inference_mode():
            reference = raw[0](probe).clone()
        raw = (convert_brain(raw[0]), raw[1], raw[2])
        raw[0].eval()
        with torch.inference_mode():
            converted = raw[0](probe)
            layout_difference = (reference-converted).abs().max().item()
            torch.testing.assert_close(reference, converted, atol=1e-4, rtol=1e-3)
        torch.backends.cudnn.allow_tf32 = old_tf32
        del probe, reference, converted
        raw[0].train()
        optimizer = build_optimizer(raw, settings)
    else:
        layout_difference = None
        optimizer = training._build_optimizer(raw, settings)
    models = raw
    if args.compile != 'none':
        models = tuple(torch.compile(m, dynamic=args.compile=='dynamic', mode='reduce-overhead' if args.compile=='reduce' else 'default') for m in raw)
    optimizer.load_state_dict(state['optimizer'])
    if args.layout:
        for parameter, values in optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in values and parameter.ndim == 4 and values[key].ndim == 3:
                    values[key] = values[key].unsqueeze(2).contiguous(memory_format=torch.channels_last)
    if args.fused:
        for group in optimizer.param_groups:
            group['fused'] = True
            group['foreach'] = False
        for parameter, state_values in optimizer.state.items():
            if 'step' in state_values:
                state_values['step'] = state_values['step'].to(parameter.device)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
    if args.precision == 'fp16':
        scaler.load_state_dict(state['scaler'])
    del state
    gpu_decoder = None
    if args.gpu_packed:
        from gpu_reader import PackedDeviceDecoder
        from precompute_pipeline.train import reader as reader_module
        gpu_decoder = PackedDeviceDecoder()
        reader_module.decode_chunk_torch = gpu_decoder
    reader = DatasetBatchReader(settings.dataset_progress, split='train', batch_size=args.batch,
                                device='cuda', seed=seed, epoch=epoch, prefetch=True)
    iterator = reader.iter_from(cursor)
    resident = next(iterator)
    if not resident[2][torch.arange(len(resident[0]), device='cuda'), resident[1]].all():
        raise RuntimeError('invalid fixture')
    if args.source == 'reader':
        resident = None
    metrics_sum = torch.zeros(7, dtype=torch.float64, device='cuda')
    iteration = 0

    def step(override=None):
        nonlocal iteration
        batch = override if override is not None else (next(iterator) if args.source == 'reader' else resident)
        if args.compile != 'none':
            torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        count, loss, metrics = forward(batch, models, settings, args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metrics_sum.add_(metrics * count)
        iteration += 1
        if args.sync_metrics or iteration % 100 == 0:
            metrics_sum.tolist()
        if iteration % 100 == 0:
            metrics_sum.zero_()
        return count

    init_start = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - init_start
    tracked_parameter = next(raw[1].parameters())
    parameter_before = tracked_parameter.detach().clone()
    optimizer_step_before = int(optimizer.state[tracked_parameter]['step'].item())
    torch.cuda.reset_peak_memory_stats()
    (WORK / f'{args.name}-timing.json').write_text(json.dumps({'phase':'timing','started_at':time.time()}))
    start = time.perf_counter()
    samples = sum(step() for _ in range(args.steps))
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    (WORK / f'{args.name}-timing.json').write_text(json.dumps({'phase':'finished','finished_at':time.time()}))
    result = {'status': 'ok', 'name': args.name, 'batch': args.batch, 'source': args.source,
              'sync_metrics': args.sync_metrics, 'check_mask': args.check_mask,
              'compile': args.compile, 'fused': args.fused, 'precision': args.precision, 'layout': args.layout, 'layout_probe_max_abs_diff': layout_difference,
              'torch_threads': torch.get_num_threads(), 'warmup_seconds': warm_seconds,
              'steps': args.steps, 'seconds': duration, 'samples_per_second': samples / duration,
              'ms_per_batch': duration / args.steps * 1000,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
              'metrics_finite': bool(torch.isfinite(metrics_sum).all().item()),
              'optimizer_updates': int(optimizer.state[tracked_parameter]['step'].item()) - optimizer_step_before,
              'parameter_max_abs_change': (tracked_parameter.detach()-parameter_before).abs().max().item()}
    if args.validate_cycle:
        cycle_step_start = int(optimizer.state[tracked_parameter]['step'].item())
        evaluations = []
        for cycle in range(args.validation_cycles):
            torch.cuda.synchronize()
            evaluation_start = time.perf_counter()
            evaluation = training._evaluate(settings=settings, execution_models=models,
                device=torch.device('cuda'), max_samples=131072)
            torch.cuda.synchronize()
            evaluations.append({'seconds': time.perf_counter()-evaluation_start, **evaluation})
            print(json.dumps({'validation_cycle':cycle+1,'seconds':evaluations[-1]['seconds']}), flush=True)
            for _ in range(3):
                step()
        tail = tuple(value[:216] for value in next(iterator))
        step(tail)
        torch.cuda.synchronize()
        result['validation_cycles'] = evaluations
        result['updates_after_validation_and_tail'] = int(optimizer.state[tracked_parameter]['step'].item()) - cycle_step_start
        result['tail_batch_samples'] = len(tail[0])
        result['cycle_peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
        (WORK / f'{args.name}-profile.txt').write_text(profile.key_averages().table(
            sort_by='self_device_time_total', row_limit=45))
        profile.export_chrome_trace(str(WORK / f'{args.name}-trace.json'))
    if gpu_decoder is not None:
        result['gpu_packed'] = True
        result['decoder_exact_match'] = gpu_decoder.equality_checked
        result['chunks_uploaded'] = gpu_decoder.chunks_uploaded
    iterator.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--source', choices=('resident', 'reader'), default='resident')
    parser.add_argument('--compile', choices=('none', 'static', 'dynamic', 'reduce'), default='none')
    parser.add_argument('--precision', choices=('fp16', 'bf16'), default='fp16')
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--sync-metrics', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--check-mask', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--layout', action='store_true')
    parser.add_argument('--validate-cycle', action='store_true')
    parser.add_argument('--gpu-packed', action='store_true')
    parser.add_argument('--mark-eval', action='store_true')
    parser.add_argument('--validation-cycles', type=int, default=2)
    args = parser.parse_args()
    try:
        result = main(args)
    except Exception as error:
        result = {'status': 'error', 'name': args.name, 'error': str(error),
                  'traceback': traceback.format_exc()}
    (WORK / f'{args.name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
