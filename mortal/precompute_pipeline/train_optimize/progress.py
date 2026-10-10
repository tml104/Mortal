"""只用于终端展示的单调时钟统计，不参与训练恢复。"""

import time


class TrainingProgress:
    """本次增量与指标窗口分开计时，验证/保存期间暂停训练速度。"""

    def __init__(self, settings, *, clock=time.monotonic, started_at=None):
        self.settings, self.clock = settings, clock
        self.started = clock() if started_at is None else started_at
        self.run_samples = 0
        self.recent_samples = 0
        self.recent_seconds = 0.0
        self.active_since = None
        self.saved_batch = 0

    def emit(self, event, progress, **fields):
        elapsed = self.clock() - self.started
        base = {
            'event': event, 'epoch': min(progress['current_epoch'] + 1, self.settings.num_epochs),
            'target_epochs': self.settings.num_epochs, 'samples': progress['samples_seen'],
            'batches': progress['batches_completed'], 'saved_batch': self.saved_batch,
            'elapsed_s': f'{elapsed:.2f}',
        }
        base.update(fields)
        print(' '.join(f'{key}={value}' for key, value in base.items()), flush=True)

    def resume_training(self):
        if self.active_since is None:
            self.active_since = self.clock()

    def pause_training(self):
        if self.active_since is not None:
            self.recent_seconds += self.clock() - self.active_since
            self.active_since = None

    def processed(self, count):
        self.run_samples += count
        self.recent_samples += count

    def training_progress(self, progress, total, values=None, *, reset=True):
        """ETA 仅涵盖当前 epoch 的剩余数据处理。"""
        now = self.clock()
        seconds = self.recent_seconds + (now - self.active_since if self.active_since is not None else 0)
        rate = self.recent_samples / seconds if self.recent_samples and seconds > 0 else 0
        consumed = progress['epoch_samples_consumed']
        fields = {
            'epoch_samples': f'{consumed}/{total}', 'percent': f'{100 * consumed / total:.2f}' if total else '100.00',
            'run_samples_s': f'{self.run_samples / (now - self.started):.2f}' if now > self.started else 'unknown',
            'train_samples_s': f'{rate:.2f}' if rate else 'unknown',
            'eta_s': f'{(total - consumed) / rate:.1f}' if rate else 'unknown',
            'metric_kind': 'window',
        }
        fields.update({name: f'{value:.6g}' for name, value in (values or {}).items()})
        self.emit('train', progress, **fields)
        if reset:
            self.recent_samples, self.recent_seconds = 0, 0.0
            if self.active_since is not None:
                self.active_since = self.clock()

    def validation_progress(self, progress, kind, count, total, started):
        elapsed = self.clock() - started
        rate = count / elapsed if count and elapsed > 0 else 0
        self.emit(
            'validation', progress, kind=kind, validation_samples=f'{count}/{total}',
            percent=f'{100 * count / total:.2f}' if total else '100.00',
            validation_samples_s=f'{rate:.2f}' if rate else 'unknown',
            eta_s=f'{(total - count) / rate:.1f}' if rate else 'unknown',
        )
