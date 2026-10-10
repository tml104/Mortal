"""一次性预计算网格测量；仅临时输出，不改变正式数据与配置。"""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from unittest.mock import patch
from contextlib import nullcontext


REPOSITORY = Path('/home/tml104/Mortal')
WORK = Path('/tmp/mortal-precompute-grid-20260930')


def select_files(count):
    """从当前游标之后选择固定牌谱，供每个新进程使用同一输入。"""
    state = json.loads((REPOSITORY / 'data/precomputed_chunks/progress.json').read_text())
    root = REPOSITORY / 'data/grp/train'
    cursor = state['source_cursor']['relative_path']
    chosen = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name < cursor.split('/')[0]:
            continue
        for filename in sorted(folder.glob('*.json.gz')):
            if filename.relative_to(root).as_posix() > cursor:
                chosen.append(str(filename))
                if len(chosen) == count:
                    return chosen
    raise RuntimeError('not enough source logs')


def run_candidate(args):
    """以实际生成/编码/提交路径测量一组线程和批次参数。"""
    os.environ['RAYON_NUM_THREADS'] = str(args.rayon)
    sys.path.insert(0, str(REPOSITORY / 'mortal'))
    import numpy as np
    import torch
    from precompute_pipeline.common import codec, files as file_ops
    from precompute_pipeline.common.settings import load_settings
    from precompute_pipeline.precompute import precompute, storage

    # 实验变体只改变该临时子进程里的函数，不写入仓库源码。
    if args.codec_mode != 'original':
        import inspect
        source = inspect.getsource(codec._normalize_batch)
        source = source.replace(
            'scalar_binary = obs[:, SCALAR_BINARY_PLANES, :,]\n    scalar_binary = scalar_binary[:, :, 0]',
            'scalar_binary = obs[:, SCALAR_BINARY_PLANES, 0]',
        ).replace(
            'scalar_float = obs[:, SCALAR_FLOAT_PLANES, :,]',
            'scalar_float = obs[:, SCALAR_FLOAT_PLANES, 0]',
        ).replace('scalar_float[:, :, 0],', 'scalar_float,')
        if args.codec_mode in ('contiguous', 'contiguous-checked'):
            source = source.replace('tile_binary = obs[:, TILE_BINARY_PLANES, :]',
                                    'tile_binary = np.take(obs, TILE_BINARY_PLANES, axis=1).astype(np.uint8)')
            source = source.replace('obs[:, TILE_FLOAT_PLANES, :],',
                                    'np.take(obs, TILE_FLOAT_PLANES, axis=1),')
        exec(compile(source, '<temporary codec experiment>', 'exec'), codec.__dict__)
    if args.codec_mode in ('trusted', 'contiguous'):
        def validated_input(value):
            obs = np.asarray(value)
            if obs.dtype != np.float32 or obs.ndim != 3 or obs.shape[1:] != (1012,34):
                raise ValueError('unexpected observation dtype/shape')
            return np.ascontiguousarray(obs)
        codec.validate_observation_semantics = validated_input

    selected = json.loads(Path(args.manifest).read_text())[:args.count]
    original = load_settings(REPOSITORY / 'mortal/precompute_pipeline/config.toml').precompute
    timings = dict(rust=0.0, grp=0.0, generation=0.0, encode=0.0,
                   validation=0.0, write_chunk=0.0, write_progress=0.0, buffer_copy=0.0)

    def timed(key, function):
        def wrapped(*positional, **keywords):
            start = time.perf_counter()
            try:
                return function(*positional, **keywords)
            finally:
                timings[key] += time.perf_counter() - start
        return wrapped

    with (
        tempfile.TemporaryDirectory(prefix='precompute-grid-') as temporary,
        tempfile.TemporaryDirectory(prefix='precompute-grid-', dir=args.disk_parent)
        if args.disk_parent else nullcontext(None) as disk_temporary,
    ):
        root = Path(temporary)
        for split in ('train', 'val'):
            (root / 'input' / split).mkdir(parents=True)
        for index, filename in enumerate(selected):
            (root / 'input/train' / f'{index:06}.json.gz').symlink_to(filename)
        settings = dataclasses.replace(
            original,
            train_dir=root / 'input/train', val_dir=root / 'input/val',
            logical_root=root / 'logical', progress_file=root / 'logical/progress.json',
            c_root=root / 'c', d_root=Path(disk_temporary) / 'd' if disk_temporary else root / 'd',
            c_budget_bytes=2**30, d_budget_bytes=2**30,
            rayon_threads=args.rayon, torch_threads=args.torch,
            file_batch_size=args.batch,
        )
        generator = precompute._build_generator(settings)
        warmup_start = time.perf_counter()
        generator.load_files(selected[:4])
        warmup_seconds = time.perf_counter() - warmup_start
        gc.collect()
        real_loader = generator.loader

        class TimedLoader:
            def load_gz_log_files(self, filenames):
                return timed('rust', real_loader.load_gz_log_files)(filenames)

        generator.loader = TimedLoader()
        generator._predict_rank_matrix = timed('grp', generator._predict_rank_matrix)
        generator.load_files = timed('generation', generator.load_files)
        cpu_before = time.process_time()
        with (
            patch.object(precompute, 'encode_chunk', timed('encode', precompute.encode_chunk)),
            patch.object(codec, 'validate_observation_semantics',
                         timed('validation', codec.validate_observation_semantics)),
            patch.object(storage, 'atomic_write_bytes', timed('write_chunk', file_ops.atomic_write_bytes)),
            patch.object(storage, 'atomic_write_json', timed('write_progress', file_ops.atomic_write_json)),
            patch.object(precompute.SampleBuffer, 'append', timed('buffer_copy', precompute.SampleBuffer.append)),
        ):
            start = time.perf_counter()
            state = precompute.run_precompute(settings, generator=generator)
            elapsed = time.perf_counter() - start
        cpu_seconds = time.process_time() - cpu_before
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        # 只在计时结束后校验紧凑字段；奖励单独比较，辨认浮点末位差异。
        fingerprints = []
        all_rewards = []
        for chunk_file in sorted((settings.d_root / 'train').glob('chunk_*.mpc.zst')):
            packed = codec.decode_packed_chunk(chunk_file.read_bytes())
            digest = hashlib.sha256()
            for field in dataclasses.fields(packed):
                value = getattr(packed, field.name)
                if field.name == 'rewards':
                    all_rewards.append(value.copy())
                elif isinstance(value, np.ndarray):
                    digest.update(value.tobytes(order='C'))
                else:
                    digest.update(str(value).encode())
            fingerprints.append(digest.hexdigest())
        rewards = np.concatenate(all_rewards)
        baseline_file = WORK / f'baseline-{len(selected)}.npz'
        if args.baseline:
            np.savez(baseline_file, rewards=rewards, fingerprints=np.asarray(fingerprints))
        reference = np.load(baseline_file)
        observations_equal = np.array_equal(reference['fingerprints'], fingerprints)
        rewards_equal = np.array_equal(reference['rewards'], rewards)
        max_reward_difference = float(np.max(np.abs(reference['rewards']-rewards)))
        record = {
            'rayon':args.rayon,'torch':args.torch,'file_batch_size':args.batch,
            'files':len(selected),'samples':state['splits']['train']['sample_count'],
            'chunks':state['splits']['train']['chunk_count'],
            'seconds':elapsed,'files_per_second':len(selected)/elapsed,
            'samples_per_second':state['splits']['train']['sample_count']/elapsed,
            'cpu_percent':cpu_seconds/elapsed/(os.cpu_count() or 1)*100,
            'peak_rss_gib':peak_rss,'warmup_seconds':warmup_seconds,
            'timings':timings,'output_bytes':sum(state['volume_bytes'].values()),
            'nonreward_fields_equal':bool(observations_equal),
            'rewards_bitwise_equal':bool(rewards_equal),
            'max_reward_difference':max_reward_difference,
            'torch_threads_effective':torch.get_num_threads(),
            'disk_parent':args.disk_parent or '/tmp',
            'codec_mode':args.codec_mode,
        }
        with (WORK / 'results.jsonl').open('a') as output:
            output.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if not observations_equal:
            raise RuntimeError('nonreward data differs from baseline')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--manifest', default=str(WORK / 'manifest.json'))
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--rayon', type=int, default=4)
    parser.add_argument('--torch', type=int, default=4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--disk-parent')
    parser.add_argument('--codec-mode', choices=['original','scalar-first','trusted','contiguous','contiguous-checked'], default='original')
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    if args.prepare:
        chosen = select_files(256)
        Path(args.manifest).write_text(json.dumps(chosen, indent=2)+'\n')
        print(json.dumps({'selected':len(chosen),'first':chosen[0],'last':chosen[-1]}))
    else:
        run_candidate(args)


if __name__ == '__main__':
    main()
