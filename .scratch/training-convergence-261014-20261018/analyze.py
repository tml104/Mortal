"""只读分析 261014 训练日志，保存指标快照、统计与验证 loss 趋势图。"""

from datetime import datetime
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/home/tml104/Mortal')
WORK = Path(__file__).resolve().parent
RUN = ROOT / 'artifacts/mortal_precomputed_optimize_261014'
os.environ['MPLCONFIGDIR'] = '/tmp/mortal-convergence-261014-mpl'
sys.path.insert(0, '/tmp/mortal-convergence-plot-20261010')

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.dates import DateFormatter

ZONE = ZoneInfo('Asia/Shanghai')
METRICS = ('loss', 'dqn_loss', 'cql_loss', 'rank_loss', 'rank_accuracy', 'q_mean', 'q_target_mean')


def stamp(timestamp):
    """图表及报告统一采用北京时间。"""
    return datetime.fromtimestamp(float(timestamp), ZONE).isoformat()


def stats(values):
    """描述时间序列波动；分位区间不是置信区间。"""
    values = np.asarray(values)
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), max=float(values.max()),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))


def bins(x, y, width, start=None):
    """以固定宽度分箱，避免把最后的单次验证波动当成趋势。"""
    start = math.floor(float(x.min()) / width) * width if start is None else start
    result = []
    for low in np.arange(start, float(x.max()) + width / 100, width):
        mask = (x >= low) & (x < low + width)
        if mask.any():
            result.append([float(x[mask].mean()), float(y[mask].mean()),
                           float(np.quantile(y[mask], .1)), float(np.quantile(y[mask], .9)), int(mask.sum())])
    return np.asarray(result)


def main():
    """不改训练状态；TensorBoard 恢复标记由官方读取器处理。"""
    torch.set_num_threads(2)
    checkpoint_path = RUN / 'latest.pth'
    with checkpoint_path.open('rb') as stream:
        file_info = os.fstat(stream.fileno())
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
    dataset = json.loads((ROOT / 'data/precomputed_chunks/progress.json').read_text())
    total = dataset['splits']['train']['sample_count']
    batch = checkpoint['contract']['batch_size']
    batches_per_epoch = math.ceil(total / batch)

    def epochs(steps):
        return np.floor_divide(steps, batches_per_epoch) + np.minimum(np.remainder(steps, batches_per_epoch) * batch, total) / total

    accumulator = EventAccumulator(str(RUN / 'tensorboard'), size_guidance={'scalars': 0})
    accumulator.Reload()
    series = {tag: np.asarray([[event.step, event.wall_time, event.value] for event in accumulator.Scalars(tag)])
              for tag in accumulator.Tags()['scalars']}
    np.savez_compressed(WORK / 'metrics.npz', **series)
    last = series['train/loss'][-1]
    saved_step = checkpoint['batches_completed']
    last_epoch = float(epochs(last[0]))
    full = []
    for i, row in enumerate(series['validation/epoch/loss']):
        full.append({'epoch': i + 1, 'step': int(row[0]), 'time': stamp(row[1]),
                     **{name: float(series[f'validation/epoch/{name}'][i, 2]) for name in METRICS}})
    nonfinite = {tag: int((~np.isfinite(rows[:, 2])).sum()) for tag, rows in series.items()}
    nonfinite = {tag: count for tag, count in nonfinite.items() if count}
    model_nonfinite = [f'{group}.{name}' for group in ('mortal', 'dqn', 'aux_net')
                       for name, tensor in checkpoint[group].items()
                       if not torch.isfinite(tensor).all().item()]
    lr = sorted(set(map(float, series['train/learning_rate'][:, 2])))

    epoch_stats, quarter_stats, hourly = {}, {}, []
    for ep in range(1, math.ceil(last_epoch) + 1):
        epoch_stats[str(ep)] = {'complete': ep <= checkpoint['completed_epochs'], 'metrics': {}}
        for prefix in ('train', 'validation/periodic'):
            for name in METRICS:
                tag = f'{prefix}/{name}'; rows = series[tag]; x = epochs(rows[:, 0])
                values = rows[(x > ep - 1) & (x <= ep), 2]
                if len(values): epoch_stats[str(ep)]['metrics'][tag] = stats(values)
    for low in np.arange(5.5, last_epoch, .25):
        high = min(low + .25, last_epoch)
        quarter_stats[f'{low:.2f}-{high:.4f}'] = {}
        for prefix in ('train', 'validation/periodic'):
            rows = series[f'{prefix}/loss']; x = epochs(rows[:, 0])
            values = rows[(x > low) & (x <= high), 2]
            if len(values): quarter_stats[f'{low:.2f}-{high:.4f}'][prefix] = stats(values)
    end = last[1]
    for offset in range(24, 0, -1):
        low, high = end - offset * 3600, end - (offset - 1) * 3600
        entry = {'start': stamp(low), 'end': stamp(high), 'metrics': {}}
        for prefix in ('train', 'validation/periodic'):
            for name in METRICS:
                tag = f'{prefix}/{name}'; rows = series[tag]
                values = rows[(rows[:, 1] > low) & (rows[:, 1] <= high), 2]
                if len(values): entry['metrics'][tag] = stats(values)
        hourly.append(entry)

    recent_comparison = {}
    for prefix in ('train', 'validation/periodic'):
        rows = series[f'{prefix}/loss']
        recent_comparison[prefix] = {}
        for label, begin, finish in (('first6h_of_last24h', 24, 18), ('last6h', 6, 0), ('last24h', 24, 0)):
            selected = rows[(rows[:, 1] > end - begin * 3600) & (rows[:, 1] <= end - finish * 3600)]
            recent_comparison[prefix][label] = stats(selected[:, 2])

    logs = []
    validation_records = {}
    full_validation_intervals = []
    full_validation_start = None
    for path in sorted(RUN.glob('log-*.log')):
        for line in path.read_text().splitlines():
            fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
            if fields.get('event') == 'validation_end':
                validation_records[(fields['kind'], int(fields['batches']))] = fields
            if fields.get('kind') == 'epoch' and fields.get('event') in ('validation_start', 'validation_end'):
                timestamp = datetime.strptime(line[1:16], '%y%m%d-%H:%M:%S').replace(tzinfo=ZONE).timestamp()
                if fields['event'] == 'validation_start':
                    full_validation_start = timestamp
                elif full_validation_start is not None:
                    full_validation_intervals.append([full_validation_start, timestamp])
            if fields.get('event') in ('start', 'resume', 'epoch_complete', 'error') or 'KeyboardInterrupt' in line:
                logs.append(line)
    cross_checks = 0
    for kind in ('periodic', 'epoch'):
        for row in series[f'validation/{kind}/loss']:
            record = validation_records[(kind, int(row[0]))]
            assert math.isclose(float(record['loss']), row[2], rel_tol=1e-6)
            cross_checks += 1
    assert not nonfinite and not model_nonfinite
    assert math.isclose(epochs(saved_step), checkpoint['samples_seen'] / total, abs_tol=1e-10)
    assert math.isclose(series['validation/periodic/loss'][-1, 2], checkpoint['last_validation']['loss'], rel_tol=1e-6)

    report = {
        'run': str(RUN), 'as_of': stamp(end), 'checkpoint_time': stamp(file_info.st_mtime),
        'test_set_present': any(tag.startswith('test/') for tag in series),
        'train_samples_per_epoch': total, 'full_val_samples': dataset['splits']['val']['sample_count'],
        'periodic_val_samples': checkpoint['contract']['validation_samples'],
        'checkpoint': {key: checkpoint[key] for key in ('completed_epochs', 'current_epoch', 'epoch_samples_consumed',
                       'samples_seen', 'batches_completed', 'last_validation', 'contract', 'scaler')},
        'saved_progress_epochs': float(epochs(saved_step)), 'last_logged_progress_epochs': last_epoch,
        'last_logged_batch': int(last[0]), 'uncommitted_logged_batches': int(last[0] - saved_step),
        'learning_rates': lr, 'nonfinite_scalars': nonfinite, 'nonfinite_model_tensors': model_nonfinite,
        'scalar_counts': {tag: len(rows) for tag, rows in series.items()},
        'validation_log_cross_checks': cross_checks, 'full_validation': full,
        'epoch_stats': epoch_stats, 'quarter_epoch_stats': quarter_stats, 'hourly_last24': hourly,
        'recent_comparison': recent_comparison, 'full_validation_intervals': full_validation_intervals,
        'log_events': logs,
    }
    (WORK / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (WORK / 'full_validation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(full[0])); writer.writeheader(); writer.writerows(full)

    font_path = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'train': '#2471a3', 'validation/periodic': '#d97924'}
    labels = {'train': '训练集（日志窗口平均）', 'validation/periodic': '固定验证子集（131,072 样本）'}
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax in axes[0]:
        for prefix in colors:
            rows = series[f'{prefix}/loss']; x = epochs(rows[:, 0]); y = rows[:, 2]
            binned = bins(x, y, .05)
            ax.plot(x, y, color=colors[prefix], alpha=.07, linewidth=.3, rasterized=True)
            ax.fill_between(binned[:, 0], binned[:, 2], binned[:, 3], color=colors[prefix], alpha=.12)
            ax.plot(binned[:, 0], binned[:, 1], color=colors[prefix], label=labels[prefix], linewidth=1.5)
        ax.plot([row['epoch'] for row in full], [row['loss'] for row in full], 'D--', color='#26834b', label='全量验证集（epoch 结束）', markersize=5)
        ax.set(xlabel='累计训练轮数（按样本数折算）', ylabel='总 loss'); ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0, 0].set(title='从头训练至当前：快速下降后逐步放缓', xlim=(0, last_epoch + .05))
    axes[0, 1].set(title='放大后期趋势：第 2 个 epoch 以后', xlim=(2, last_epoch + .05), ylim=(2.82, 3.20))
    for ax, prefix in zip(axes[1], ('train', 'validation/periodic')):
        rows = series[f'{prefix}/loss']; recent = rows[rows[:, 1] >= end - 24 * 3600]
        times, values = [], []
        for i, row in enumerate(recent):
            moment = datetime.fromtimestamp(row[1], ZONE)
            if i and row[1] - recent[i - 1, 1] > 300:
                times.append(moment); values.append(float('nan'))
            times.append(moment); values.append(row[2])
        ax.plot(times, values, color=colors[prefix], alpha=.22, linewidth=.5, label='原始记录')
        for low, high in full_validation_intervals:
            if high >= end - 24 * 3600:
                ax.axvspan(datetime.fromtimestamp(low, ZONE), datetime.fromtimestamp(high, ZONE), color='#666666', alpha=.09)
        binned = bins(recent[:, 1], recent[:, 2], 3600, start=end - 24 * 3600)
        moments = [datetime.fromtimestamp(t, ZONE) for t in binned[:, 0]]
        ax.plot(moments, binned[:, 1], 'o-', color=colors[prefix], markersize=3, linewidth=1.7, label='每小时均值')
        ax.fill_between(moments, binned[:, 2], binned[:, 3], color=colors[prefix], alpha=.13, label='小时内 P10–P90 波动')
        ax.set(title='最近 24 小时：' + labels[prefix], xlabel='北京时间', ylabel='总 loss')
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M', tz=ZONE)); ax.tick_params(axis='x', rotation=15)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    figure.suptitle(f'mortal_precomputed_optimize_261014｜学习率 1e-5｜截至 {stamp(end)[:19].replace("T", " ")}', fontsize=13)
    figure.text(.5, .012, '没有独立测试集 loss。彩色阴影表示时间波动而非置信区间；灰区为全量验证、暂停训练。上图趋势按 0.05 epoch 分箱。', ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .045, 1, .95))
    for suffix in ('png', 'svg'): figure.savefig(WORK / f'loss_trends.{suffix}', dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    specs = [('dqn_loss', 1, 'DQN loss（权重 1）'), ('cql_loss', 5, '5 × CQL loss'),
             ('rank_loss', .2, '0.2 × 顺位分类 loss'), ('rank_accuracy', 100, '顺位分类准确率（%）')]
    for ax, (metric, weight, title) in zip(axes.flat, specs):
        for prefix in colors:
            rows = series[f'{prefix}/{metric}']; x = epochs(rows[:, 0]); mask = x >= 1
            binned = bins(x[mask], rows[mask, 2] * weight, .05)
            ax.plot(binned[:, 0], binned[:, 1], color=colors[prefix], label=labels[prefix])
        ax.plot([row['epoch'] for row in full], [row[metric] * weight for row in full], 'D--', color='#26834b', label='全量验证集', markersize=5)
        ax.set(title=title, xlabel='累计训练轮数', xlim=(1, last_epoch + .05)); ax.grid(alpha=.2); ax.legend(fontsize=8)
    figure.suptitle('损失分解与辅助任务｜总 loss = DQN + 5 × CQL + 0.2 × 顺位分类 loss', fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ('png', 'svg'): figure.savefig(WORK / f'validation_components.{suffix}', dpi=170)
    plt.close(figure)
    print(json.dumps({key: report[key] for key in ('as_of', 'saved_progress_epochs', 'last_logged_progress_epochs',
          'uncommitted_logged_batches', 'full_validation', 'quarter_epoch_stats', 'nonfinite_scalars',
          'validation_log_cross_checks')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
