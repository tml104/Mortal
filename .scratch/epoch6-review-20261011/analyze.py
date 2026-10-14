"""读取冻结的第六轮状态，按样本进度对齐新旧日志并绘制趋势。"""

from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo


ROOT = Path('/home/tml104/Mortal')
WORK = Path(__file__).resolve().parent
PLOT_DEPS = Path('/tmp/mortal-convergence-plot-20261010')
if PLOT_DEPS.exists():
    sys.path.insert(0, str(PLOT_DEPS))

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


ZONE = ZoneInfo('Asia/Shanghai')
NAMES = ('loss', 'dqn_loss', 'cql_loss', 'rank_loss', 'rank_accuracy', 'q_mean', 'q_target_mean')
TOTAL = 838796504


def read_metrics(directory):
    """由 TensorBoard 的恢复标记剔除未提交、随后重跑的旧记录。"""
    accumulator = EventAccumulator(str(directory), size_guidance={'scalars': 0})
    accumulator.Reload()
    return {
        tag: [{'step': event.step, 'value': event.value, 'wall_time': event.wall_time}
              for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get('scalars', [])
    }


def statistics(rows):
    values = np.asarray([row['value'] for row in rows])
    return {
        'count': len(values), 'mean': float(values.mean()), 'std': float(values.std()),
        'minimum': float(values.min()), 'maximum': float(values.max()),
        'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95)),
    }


def main():
    os.chdir(ROOT)
    torch.set_num_threads(2)
    checkpoint_path = ROOT / 'artifacts/mortal_precomputed_optimize/latest.pth'
    with checkpoint_path.open('rb') as stream:
        metadata = os.fstat(stream.fileno())
        checkpoint = torch.load(stream, weights_only=True, map_location='cpu')
    if checkpoint['completed_epochs'] != 6 or checkpoint['last_validation']['kind'] != 'epoch':
        raise RuntimeError('分析必须使用第六轮完整保存点')
    old_checkpoint = torch.load(ROOT / 'artifacts/mortal_precomputed_v2/latest.pth', weights_only=True, map_location='cpu')
    sources = [ROOT / 'artifacts/mortal_precomputed_v2/tensorboard', checkpoint_path.parent / 'tensorboard']
    old, new = (read_metrics(directory) for directory in sources)
    migration_step = old_checkpoint['batches_completed']
    migration_samples = old_checkpoint['samples_seen']
    full = []
    for optimized, data in ((False, old), (True, new)):
        for index, event in enumerate(data['validation/epoch/loss']):
            full.append({
                'epoch': index + (4 if optimized else 1), 'step': event['step'],
                'time': datetime.fromtimestamp(event['wall_time'], ZONE).isoformat(),
                **{name: data[f'validation/epoch/{name}'][index]['value'] for name in NAMES},
            })
    anchors = [(migration_step, migration_samples)]
    anchors.extend((row['step'], row['epoch'] * TOTAL) for row in full if row['epoch'] >= 4)

    def progress(step, optimized):
        if optimized:
            batch, samples = max((anchor for anchor in anchors if anchor[0] <= step), key=lambda anchor: anchor[0])
            return (samples + (step - batch) * 2048) / TOTAL
        epochs, batches = divmod(step, math.ceil(TOTAL / 1024))
        return epochs + min(batches * 1024, TOTAL) / TOTAL

    series = {}
    for prefix in ('train', 'validation/periodic'):
        for name in NAMES:
            tag = f'{prefix}/{name}'
            series[tag] = [
                {**event, 'epoch': progress(event['step'], optimized)}
                for optimized, data in ((False, old), (True, new))
                for event in data.get(tag, [])
                if optimized or event['step'] <= migration_step
            ]
    nonfinite = {str(source): {tag: sum(not math.isfinite(row['value']) for row in rows)
                              for tag, rows in data.items()
                              if any(not math.isfinite(row['value']) for row in rows)}
                 for source, data in zip(sources, (old, new), strict=True)}
    end = metadata.st_mtime
    recent = []
    for begin, finish in ((12, 10), (10, 8), (8, 6), (6, 4), (4, 2), (2, 0)):
        values = {}
        for tag, rows in series.items():
            selected = [row for row in rows if end - begin * 3600 <= row['wall_time'] < end - finish * 3600]
            if selected:
                values[tag] = statistics(selected)
        recent.append({'hours_before_checkpoint': [begin, finish], 'metrics': values})
    epoch_windows = {}
    for epoch in (4, 5, 6):
        epoch_windows[str(epoch)] = {
            tag: statistics(selected)
            for tag, rows in series.items()
            if (selected := [row for row in rows if epoch - 1 <= row['epoch'] < epoch])
        }
    log_events = []
    for path in sorted(checkpoint_path.parent.glob('log-*.log')):
        for line in path.read_text().splitlines():
            fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
            if fields.get('event') in {'resume', 'epoch_complete'} or 'KeyboardInterrupt' in line or 'RuntimeError' in line:
                log_events.append({'file': str(path), 'line': line})
    old_steps = {float(value['step']) for value in old_checkpoint['optimizer']['state'].values() if 'step' in value}
    new_steps = {float(value['step']) for value in checkpoint['optimizer']['state'].values() if 'step' in value}
    report = {
        'checkpoint_time': datetime.fromtimestamp(end, ZONE).isoformat(),
        'checkpoint': {key: checkpoint[key] for key in (
            'completed_epochs', 'current_epoch', 'epoch_samples_consumed', 'samples_seen',
            'batches_completed', 'last_validation', 'contract', 'scaler',
        )},
        'sources': list(map(str, sources)), 'full_validation': full,
        'recent_windows': recent, 'epoch_windows': epoch_windows, 'nonfinite': nonfinite,
        'old_optimizer_steps': sorted(old_steps), 'new_optimizer_steps': sorted(new_steps),
        'learning_rates': sorted({row['value'] for row in new['train/learning_rate']}),
        'migration_epoch': migration_samples / TOTAL, 'log_events': log_events,
    }
    (WORK / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (WORK / 'metrics.json').write_text(json.dumps({'old': old, 'new': new}) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for tag, label, color in (
        ('train/loss', 'Training (0.05-epoch means)', '#2471a3'),
        ('validation/periodic/loss', 'Fixed-subset validation (bin means)', '#d9822b'),
    ):
        rows = series[tag]
        x = np.asarray([row['epoch'] for row in rows]); y = np.asarray([row['value'] for row in rows])
        means = [(float(x[mask].mean()), float(y[mask].mean()))
                 for low in np.arange(.25, 6, .05)
                 if (mask := (x >= low) & (x < low + .05)).any()]
        axes[0].plot(*zip(*means), label=label, color=color, linewidth=1.8)
    axes[0].plot([row['epoch'] for row in full], [row['loss'] for row in full], 'D--',
                 color='#26734d', label='Full validation (epoch end)')
    axes[0].axvline(migration_samples / TOTAL, linestyle=':', color='#7653a6', label='Conv2d + batch 2048 migration')
    axes[0].set(xlabel='Cumulative training samples / samples per epoch', ylabel='Weighted total loss',
                title='Long-term trend after first 0.25 epoch', xlim=(.25, 6.1))
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    selected = [row for row in series['validation/periodic/loss'] if end - 12 * 3600 <= row['wall_time'] <= end]
    axes[1].plot([datetime.fromtimestamp(row['wall_time'], ZONE) for row in selected],
                 [row['value'] for row in selected], color='#c2c7cc', linewidth=.7, label='Each fixed-subset validation')
    axes[1].plot([datetime.fromtimestamp(end - sum(window['hours_before_checkpoint']) * 1800, ZONE) for window in recent],
                 [window['metrics']['validation/periodic/loss']['mean'] for window in recent],
                 'o-', color='#c45c26', linewidth=2, label='Two-hour means')
    axes[1].xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M', tz=ZONE))
    axes[1].set(xlabel='Asia/Shanghai', ylabel='Fixed-subset total loss', title='Last 12 hours before epoch 6 checkpoint')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    figure.suptitle('Mortal convergence review | epoch 6 saved: ' + report['checkpoint_time'][:19].replace('T', ' '))
    figure.text(.5, .015, 'Training, fixed-subset and full validation use different sample sets. Batch-size migration is a confounding change.',
                ha='center', fontsize=8, color='#555555')
    figure.tight_layout(rect=(0, .055, 1, .92)); figure.savefig(WORK / 'convergence.png', dpi=170)
    plt.close(figure)
    print(json.dumps({'checkpoint': report['checkpoint'], 'full_validation': full, 'recent_windows': recent,
                      'epoch_windows': epoch_windows, 'nonfinite': nonfinite}, ensure_ascii=False))


if __name__ == '__main__':
    main()
