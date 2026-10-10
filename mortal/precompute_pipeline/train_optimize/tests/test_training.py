"""在公开训练入口检验 CUDA 优化执行、尾批、保存和恢复。"""

from dataclasses import replace
from pathlib import Path
import contextlib
import io
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

MORTAL_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.train.tests.test_reader import build_dataset
from precompute_pipeline.train.tests.test_train import training_settings
from precompute_pipeline.common.optimized_checkpoint import load_checkpoint
from precompute_pipeline.train_optimize.settings import OptimizedTrainingSettings
from precompute_pipeline.train_optimize.train import run_training


def small_settings(root, progress):
    old = training_settings(root, progress)
    return OptimizedTrainingSettings(
        dataset_progress=progress, output_dir=root / 'artifacts/optimized',
        batch_size=4, num_epochs=2, model=old.model,
        log_every_batches=2, save_every_batches=2, validation_samples=2,
        # 数值恢复使用 FP32，避免极短构造任务的正常 AMP 溢出造成随机全跳步。
        # 生产默认 FP16 由真实数据集成和 180 批验收另外覆盖。
        torch_threads=1, enable_amp=False,
    )


@unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA 验证固定优化组合')
class OptimizedTrainingTests(unittest.TestCase):
    def test_swallowed_log_failure_before_commit_does_not_save_progress(self):
        from torch.utils.tensorboard import SummaryWriter
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                settings = small_settings(root, build_dataset(root))
            original_open, add_scalar = Path.open, SummaryWriter.add_scalar
            output, errors = io.StringIO(), io.StringIO()
            logger = logging.getLogger('mortal.tests.validation_sink_failure')
            handler = logging.StreamHandler(errors)
            logger.addHandler(handler)
            logger.propagate = False
            class FailedLog:
                def __init__(self, real):
                    self.real = real
                def __getattr__(self, name):
                    return getattr(self.real, name)
                def write(self, value):
                    if b'validation library warning' in value:
                        raise OSError('validation logging failed')
                    return self.real.write(value)
            def open_file(path, *args, **kwargs):
                real = original_open(path, *args, **kwargs)
                return FailedLog(real) if path.name.startswith('log-') and path.suffix == '.log' else real
            def record_scalar(writer, tag, *args, **kwargs):
                result = add_scalar(writer, tag, *args, **kwargs)
                if tag == 'validation/periodic/loss':
                    try:
                        logger.warning('validation library warning')
                    except OSError:
                        pass  # 模拟外部 SDK 吞掉诊断输出错误，训练仍必须拒绝提交。
                return result
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch.object(Path, 'open', open_file), patch.object(SummaryWriter, 'add_scalar', record_scalar):
                    with self.assertRaisesRegex(OSError, 'validation logging failed'):
                        run_training(settings, max_batches=2)
                self.assertEqual(load_checkpoint(settings.output_dir / 'latest.pth')['batches_completed'], 0)
                self.assertIn('event=validation_end', output.getvalue())
                checkpoints = [line for line in output.getvalue().splitlines() if 'event=checkpoint' in line]
                self.assertEqual(len(checkpoints), 1)
            finally:
                logger.removeHandler(handler)
                logger.propagate = True
                handler.close()

    def test_log_failure_during_updates_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                settings = small_settings(root, build_dataset(root))
            original_open = Path.open
            class FailedLog:
                def __init__(self, real):
                    self.real = real
                def __getattr__(self, name):
                    return getattr(self.real, name)
                def write(self, text):
                    if b'event=train ' in text:
                        raise OSError('training log unavailable')
                    return self.real.write(text)
            def open_file(path, *args, **kwargs):
                real = original_open(path, *args, **kwargs)
                if path.name.startswith('log-') and path.suffix == '.log':
                    return FailedLog(real)
                return real
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch.object(Path, 'open', open_file):
                with self.assertRaisesRegex(OSError, 'training log unavailable'):
                    run_training(settings, max_batches=2)
            state = load_checkpoint(settings.output_dir / 'latest.pth')
            self.assertEqual(state['batches_completed'], 0)
            self.assertIn('training log unavailable', errors.getvalue())
            self.assertNotIn('event=validation_start', output.getvalue())
            self.assertNotIn('event=complete', output.getvalue())

    def test_controlled_clock_excludes_save_validation_and_old_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                settings = replace(small_settings(root, build_dataset(root)), prefetch=False)
            class Clock:
                now = 0.0
                train_reads = 0
                def __call__(self):
                    return self.now
            clock = Clock()
            read_bytes, torch_save = Path.read_bytes, torch.save
            def timed_read(path):
                if path.name.endswith('.mpc.zst'):
                    clock.now += 10
                    if 'train' in path.parts:
                        clock.train_reads += 1
                return read_bytes(path)
            def timed_save(*args, **kwargs):
                clock.now += 1000
                return torch_save(*args, **kwargs)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(Path, 'read_bytes', timed_read), patch('torch.save', timed_save):
                run_training(settings, max_batches=2, clock=clock)
            clock.now, clock.train_reads = 0.0, 0
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch.object(Path, 'read_bytes', timed_read), patch('torch.save', timed_save):
                run_training(settings, max_batches=2, clock=clock)
            line = next(line for line in output.getvalue().splitlines() if 'event=train ' in line and 'batches=4 ' in line)
            fields = dict(item.split('=', 1) for item in line.split('] ', 1)[1].split())
            self.assertEqual(fields['samples'], '13')
            self.assertAlmostEqual(float(fields['train_samples_s']), 5 / (clock.train_reads * 10), delta=0.005)
            # 此日志前完成一次 epoch 验证(10s)和一次保存(1000s)。
            self.assertAlmostEqual(float(fields['elapsed_s']), 1010 + clock.train_reads * 10, delta=0.01)
            self.assertAlmostEqual(float(fields['run_samples_s']), 5 / float(fields['elapsed_s']), delta=0.005)
            unknown = io.StringIO()
            with contextlib.redirect_stdout(unknown):
                run_training(replace(settings, output_dir=root / 'artifacts/unknown'), max_batches=1, clock=lambda: 0.0)
            self.assertIn('train_samples_s=unknown eta_s=unknown', unknown.getvalue())

    def test_validation_and_save_failure_keep_last_committed_checkpoint(self):
        for failure in ('validation', 'save'):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with contextlib.redirect_stdout(io.StringIO()):
                    settings = small_settings(root, build_dataset(root))
                if failure == 'validation':
                    from precompute_pipeline.common.dataset import iter_chunk_records
                    next(iter(iter_chunk_records(settings.dataset_progress, 'val'))).path.unlink()
                real_replace = os.replace
                writes = 0
                def fail_second_checkpoint(source, destination):
                    nonlocal writes
                    if Path(destination) == settings.output_dir / 'latest.pth':
                        writes += 1
                        if failure == 'save' and writes == 2:
                            raise OSError('simulated save failure')
                    return real_replace(source, destination)
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(io.StringIO()), patch('os.replace', side_effect=fail_second_checkpoint):
                    with self.assertRaises((OSError, FileNotFoundError)):
                        run_training(settings, max_batches=2)
                state = load_checkpoint(settings.output_dir / 'latest.pth')
                self.assertEqual(state['batches_completed'], 0)
                checkpoint_lines = [line for line in stream.getvalue().splitlines() if 'event=checkpoint' in line]
                self.assertEqual(len(checkpoint_lines), 1)
                self.assertNotIn('event=complete', stream.getvalue())

    def test_progress_reports_processed_and_saved_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                settings = small_settings(root, build_dataset(root))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                run_training(settings, max_batches=2)
            output = stream.getvalue()
            for event in ('start', 'prepare', 'first_batch_start', 'first_batch_end', 'train', 'validation_start', 'validation_end', 'checkpoint', 'paused'):
                self.assertIn(f'event={event}', output)
            self.assertIn('run_samples_s=', output)
            self.assertIn('train_samples_s=', output)
            self.assertIn('saved_batch=2', output)
            state = load_checkpoint(settings.output_dir / 'latest.pth')
            self.assertNotIn('run_samples_s', state)
            self.assertNotIn('elapsed_s', state)

    def test_full_batches_tail_and_full_validation_commit(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(temporary)
            progress = build_dataset(root)
            settings = small_settings(root, progress)
            settings.output_dir.mkdir(parents=True)
            history = settings.output_dir / 'log-260101-000000.log'
            history.write_text('[260101-00:00:00] historical run\n')
            obsolete = settings.output_dir / 'old-output.txt'
            obsolete.write_text('obsolete')
            fake_log_directory = settings.output_dir / 'log-260102-000000.log'
            fake_log_directory.mkdir()
            result = run_training(settings)
            self.assertEqual(history.read_text(), '[260101-00:00:00] historical run\n')
            self.assertFalse(obsolete.exists())
            self.assertFalse(fake_log_directory.exists())
            current_logs = [path for path in settings.output_dir.glob('log-*.log') if path != history]
            self.assertEqual(len(current_logs), 1)
            text = current_logs[0].read_text()
            for event in ('start', 'prepare', 'train', 'validation_end', 'checkpoint', 'complete'):
                self.assertIn(f'event={event}', text)
            self.assertTrue(all(line.startswith('[') for line in text.splitlines()))
            self.assertEqual(result['samples_seen'], 18)
            self.assertEqual(result['batches_completed'], 6)
            self.assertEqual(result['completed_epochs'], 2)
            state = load_checkpoint(settings.output_dir / 'latest.pth')
            self.assertEqual(state['last_validation']['kind'], 'epoch')
            self.assertEqual(state['last_validation']['sample_count'], 2)
            self.assertGreater(len(state['optimizer']['state']), 0)
            for name, value in state['mortal'].items():
                if name.endswith('num_batches_tracked'):
                    self.assertEqual(value.item(), 6)
            self.assertTrue(list((settings.output_dir / 'tensorboard').glob('events.out.tfevents.*')))

    def test_interrupted_run_restores_optimizer_rng_and_sample_suffix(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            settings = replace(small_settings(root, build_dataset(root)), log_every_batches=4)
            with patch('secrets.randbits', return_value=424242):
                partial = run_training(settings, max_batches=1)
            self.assertEqual(partial['completed_epochs'], 0)
            self.assertEqual(partial['epoch_samples_consumed'], 4)
            self.assertEqual(load_checkpoint(settings.output_dir / 'latest.pth')['metric_samples'], 4)
            resumed = run_training(settings)
            reference = replace(settings, output_dir=root / 'artifacts/reference')
            with patch('secrets.randbits', return_value=424242):
                run_training(reference)
            left = load_checkpoint(settings.output_dir / 'latest.pth')
            right = load_checkpoint(reference.output_dir / 'latest.pth')
            self.assertEqual(resumed['samples_seen'], 18)
            self.assertEqual(left['main_seed'], 424242)
            for key in ('mortal', 'dqn', 'aux_net'):
                for name, value in left[key].items():
                    torch.testing.assert_close(value, right[key][name], rtol=1e-4, atol=2e-5)
            self.assertEqual(left['scaler'], right['scaler'])
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events = EventAccumulator(str(settings.output_dir / 'tensorboard')).Reload()
            self.assertEqual([entry.step for entry in events.Scalars('train/loss')], [1, 4, 6])
            before = (settings.output_dir / 'latest.pth').read_bytes()
            with self.assertRaisesRegex(RuntimeError, '不兼容'):
                run_training(replace(settings, batch_size=8))
            self.assertEqual((settings.output_dir / 'latest.pth').read_bytes(), before)
            self.assertEqual(run_training(settings)['batches_completed'], 6)
            corrupt = dict(left)
            del corrupt['cuda_rng_state_all']
            torch.save(corrupt, settings.output_dir / 'latest.pth')
            with self.assertRaisesRegex(RuntimeError, 'cuda_rng_state_all'):
                run_training(settings)


if __name__ == '__main__':
    unittest.main()
