"""通过公开训练入口观察文件和终端，不绑定日志辅助实现。"""

import contextlib
import builtins
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
import logging
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch

MORTAL_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.dataset import load_dataset_state
from precompute_pipeline.common.optimized_checkpoint import (
    SCHEMA_VERSION, MODEL_FORMAT, model_metadata, training_contract,
)
from precompute_pipeline.common.optimized_model import build_models, build_optimizer
from precompute_pipeline.train.tests.test_reader import build_dataset
from precompute_pipeline.train_optimize.tests.test_training import small_settings
from precompute_pipeline.train_optimize.train import run_training


START = datetime(2026, 10, 9, 23, 59, 59, tzinfo=timezone(timedelta(hours=8))).timestamp()


def completed_training(root):
    """真实新格式的已完成小模型状态，使日志测试不需要 GPU。"""
    with contextlib.redirect_stdout(io.StringIO()):
        settings = replace(small_settings(root, build_dataset(root)), num_epochs=1)
    models = build_models(settings.model)
    state = {
        'schema_version': SCHEMA_VERSION, 'model_format': MODEL_FORMAT,
        'model_spec': model_metadata(settings.model),
        'contract': training_contract(settings, load_dataset_state(settings.dataset_progress)['generation_id']),
        **dict(zip(('mortal', 'dqn', 'aux_net'), (model.state_dict() for model in models))),
        'optimizer': build_optimizer(models, settings.optimizer).state_dict(),
        'scaler': {}, 'main_seed': 71, 'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': [torch.tensor([71] + [0] * 15, dtype=torch.uint8)],
        'completed_epochs': 1, 'current_epoch': 1, 'epoch_samples_consumed': 0,
        'samples_seen': 9, 'batches_completed': 3, 'last_validation': None,
        'metric_sums': [0.0] * 7, 'metric_samples': 0,
    }
    settings.output_dir.mkdir(parents=True)
    torch.save(state, settings.output_dir / 'latest.pth')
    return settings


class RunLoggingTests(unittest.TestCase):
    def test_handler_created_during_training_is_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            output, errors = io.StringIO(), io.StringIO()
            logger = logging.getLogger('mortal.tests.created_during_training')
            logger.propagate = False
            original_open = builtins.open
            handlers = []
            def observed_open(file, *args, **kwargs):
                if isinstance(file, (str, Path)) and Path(file) == settings.dataset_progress and not handlers:
                    handler = logging.StreamHandler()
                    handlers.append(handler)
                    logger.addHandler(handler)
                    logger.warning('created inside')
                return original_open(file, *args, **kwargs)
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch('builtins.open', observed_open):
                    run_training(settings, wall_clock=lambda: START)
                    self.assertIs(handlers[0].stream, errors)
                    logger.warning('outside session')
                    run_training(settings, wall_clock=lambda: START + 1)
                self.assertIn('outside session', errors.getvalue())
                self.assertNotIn('outside session', (settings.output_dir / 'log-261009-235959.log').read_text())
            finally:
                for handler in handlers:
                    logger.removeHandler(handler)
                    handler.close()
                logger.propagate = True

    def test_python_logging_cannot_swallow_file_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            output, errors = io.StringIO(), io.StringIO()
            logger = logging.getLogger('mortal.tests.log_failure')
            handler = logging.StreamHandler(errors)
            logger.addHandler(handler)
            logger.propagate = False
            original_open, original_path_open = builtins.open, Path.open
            continued = []
            class FailedFile:
                def __init__(self, real):
                    self.real = real
                def __getattr__(self, name):
                    return getattr(self.real, name)
                def write(self, value):
                    raise OSError('logging sink failed')
            def path_open(path, *args, **kwargs):
                real = original_path_open(path, *args, **kwargs)
                return FailedFile(real) if path.name == 'log-261009-235959.log' else real
            def observed_open(file, *args, **kwargs):
                if isinstance(file, (str, Path)) and Path(file) == settings.dataset_progress:
                    logger.warning('external warning')
                    continued.append(True)
                return original_open(file, *args, **kwargs)
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch('builtins.open', observed_open), patch.object(Path, 'open', path_open), patch.object(logging, 'raiseExceptions', False):
                    with self.assertRaisesRegex(OSError, 'logging sink failed'):
                        run_training(settings, wall_clock=lambda: START)
                self.assertEqual(continued, [])
            finally:
                logger.removeHandler(handler)
                logger.propagate = True
                handler.close()

    def test_close_error_does_not_mask_or_repeat_training_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            settings.dataset_progress.write_text('{invalid json')
            original_open = Path.open
            class CloseFailure:
                def __init__(self, real):
                    self.real = real
                def __getattr__(self, name):
                    return getattr(self.real, name)
                def close(self):
                    self.real.close()
                    raise OSError('simulated close failure')
            def open_file(path, *args, **kwargs):
                real = original_open(path, *args, **kwargs)
                return CloseFailure(real) if path.name == 'log-261009-235959.log' else real
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors), patch.object(Path, 'open', open_file):
                with self.assertRaises(json.JSONDecodeError):
                    run_training(settings, wall_clock=lambda: START)
            self.assertEqual(errors.getvalue().count('json.decoder.JSONDecodeError:'), 1)
            self.assertIn('simulated close failure', errors.getvalue())

    def test_existing_python_logging_stderr_handler_is_captured_and_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            output, errors = io.StringIO(), io.StringIO()
            logger = logging.getLogger('mortal.tests.before_training')
            original_level, original_propagate = logger.level, logger.propagate
            handler = logging.StreamHandler(errors)
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            original_open = builtins.open
            def observed_open(file, *args, **kwargs):
                if isinstance(file, (str, Path)) and Path(file) == settings.dataset_progress:
                    logger.warning('bound before run\nsecond line')
                return original_open(file, *args, **kwargs)
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch('builtins.open', observed_open):
                    run_training(settings, wall_clock=lambda: START)
                text = (settings.output_dir / 'log-261009-235959.log').read_text()
                self.assertIn('[261009-23:59:59] bound before run\n[261009-23:59:59] second line\n', text)
                self.assertIs(handler.stream, errors)
                logger.warning('outside training')
                self.assertNotIn('outside training', (settings.output_dir / 'log-261009-235959.log').read_text())
            finally:
                logger.removeHandler(handler)
                logger.setLevel(original_level)
                logger.propagate = original_propagate
                handler.close()

    def test_partial_history_line_and_interleaved_streams_remain_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            path = settings.output_dir / 'log-261009-235959.log'
            path.write_text('[261009-23:59:58] unfinished old run')
            output, errors = io.StringIO(), io.StringIO()
            original_open = builtins.open
            def observed_open(file, *args, **kwargs):
                if isinstance(file, (str, Path)) and Path(file) == settings.dataset_progress:
                    sys.stdout.write('stdout fragment')
                    sys.stdout.flush()
                    sys.stderr.write('stderr fragment')
                    raise RuntimeError('stream boundary failure')
                return original_open(file, *args, **kwargs)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch('builtins.open', observed_open):
                with self.assertRaisesRegex(RuntimeError, 'stream boundary failure'):
                    run_training(settings, wall_clock=lambda: START)
            text = path.read_text()
            self.assertTrue(text.startswith('[261009-23:59:58] unfinished old run\n[261009-23:59:59] stdout fragment\n'))
            self.assertIn('[261009-23:59:59] stderr fragment\n[261009-23:59:59] Traceback (most recent call last):', text)
            self.assertTrue(text.endswith('\n'))
            self.assertEqual(text.count('[261009-23:59:59]'), len(text.splitlines()) - 1)

    def test_cli_missing_config_fails_on_terminal_without_log_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run([
                sys.executable, str(MORTAL_DIR / 'precompute_pipeline/train_optimize/train.py'),
                '--config', str(root / 'missing.toml'),
            ], cwd=root, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FileNotFoundError', result.stderr)
            self.assertTrue(all(line.startswith('[') for line in result.stderr.splitlines()))
            self.assertEqual(list(root.iterdir()), [])

    def test_file_creation_write_and_flush_failures_abort_and_restore_streams(self):
        for failure in ('create', 'write', 'flush'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                settings = completed_training(Path(temporary))
                checkpoint = (settings.output_dir / 'latest.pth').read_bytes()
                original_open = Path.open
                opened = []
                class FaultyFile:
                    def __init__(self, real):
                        self.real = real
                    def __getattr__(self, name):
                        return getattr(self.real, name)
                    def write(self, value):
                        if failure == 'write':
                            raise OSError('simulated log write failure')
                        return self.real.write(value)
                    def flush(self):
                        if failure == 'flush':
                            raise OSError('simulated log flush failure')
                        return self.real.flush()
                def open_file(path, *args, **kwargs):
                    if path.name == 'log-261009-235959.log':
                        if failure == 'create':
                            raise PermissionError('simulated log create failure')
                        real = original_open(path, *args, **kwargs)
                        opened.append(real)
                        return FaultyFile(real)
                    return original_open(path, *args, **kwargs)
                output, errors = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch.object(Path, 'open', open_file):
                    with self.assertRaisesRegex(OSError, f'simulated log {failure} failure'):
                        run_training(settings, wall_clock=lambda: START)
                    self.assertIs(sys.stdout, output)
                    self.assertIs(sys.stderr, errors)
                self.assertIn(f'simulated log {failure} failure', errors.getvalue())
                self.assertNotIn('event=complete', output.getvalue())
                self.assertTrue(all(file.closed for file in opened))
                self.assertEqual((settings.output_dir / 'latest.pth').read_bytes(), checkpoint)

    def test_cli_failure_has_one_timestamped_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = completed_training(root)
            checkpoint = settings.output_dir / 'latest.pth'
            torch.save({'schema_version': 1}, checkpoint)
            before = checkpoint.read_bytes()
            config = root / 'config.toml'
            config.write_text(
                f'[training]\ndataset_progress="{settings.dataset_progress}"\n'
                f'output_dir="{settings.output_dir}"\n'
            )
            result = subprocess.run([
                sys.executable, str(MORTAL_DIR / 'precompute_pipeline/train_optimize/train.py'),
                '--config', str(config),
            ], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stderr.count('Traceback (most recent call last):'), 1)
            self.assertIn('RuntimeError', result.stderr)
            log = next(settings.output_dir.glob('log-*.log')).read_text()
            self.assertEqual(log, result.stderr)
            self.assertTrue(all(line.startswith('[') for line in log.splitlines()))
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_exception_is_recorded_before_streams_close_and_still_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            checkpoint = (settings.output_dir / 'latest.pth').read_bytes()
            original = settings.dataset_progress.read_text()
            settings.dataset_progress.write_text('{invalid json')
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                with self.assertRaises(json.JSONDecodeError):
                    run_training(settings, wall_clock=lambda: START)
                self.assertIs(sys.stdout, output)
                self.assertIs(sys.stderr, errors)
            path = settings.output_dir / 'log-261009-235959.log'
            text = path.read_text()
            self.assertEqual(text.count('Traceback (most recent call last):'), 1)
            self.assertIn('JSONDecodeError', text)
            self.assertEqual(text, errors.getvalue())
            self.assertTrue(all(line.startswith('[261009-23:59:59] ') for line in text.splitlines()))
            self.assertEqual((settings.output_dir / 'latest.pth').read_bytes(), checkpoint)
            settings.dataset_progress.write_text(original)
            with contextlib.redirect_stdout(io.StringIO()):
                run_training(settings, wall_clock=lambda: START + 1)
            self.assertEqual(path.read_text(), text)

    def test_python_streams_fragments_warning_and_midnight(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            output, errors = io.StringIO(), io.StringIO()
            original_open = builtins.open
            now = [START]
            def observed_open(file, *args, **kwargs):
                if isinstance(file, (str, Path)) and Path(file) == settings.dataset_progress:
                    now[0] = START + 3
                    sys.stdout.write('普通片段')
                    sys.stdout.flush()
                    current = settings.output_dir / 'log-261009-235959.log'
                    self.assertTrue(current.read_text().endswith('[261010-00:00:02] 普通片段'))
                    sys.stdout.write(' joined\n\n')
                    sys.stderr.writelines(['stderr first\n', 'stderr second\n'])
                    with warnings.catch_warnings():
                        warnings.simplefilter('always')
                        warnings.warn('warning first\nwarning second', RuntimeWarning)
                return original_open(file, *args, **kwargs)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch('builtins.open', observed_open):
                run_training(settings, wall_clock=lambda: now[0])
                self.assertIs(sys.stdout, output)
                self.assertIs(sys.stderr, errors)
            content = (settings.output_dir / 'log-261009-235959.log').read_text()
            self.assertIn('[261010-00:00:02] 普通片段 joined\n[261010-00:00:02] \n', content)
            self.assertIn('[261010-00:00:02] warning second', content)
            for line in content.splitlines():
                self.assertTrue(line.startswith('[261010-00:00:02] '), line)
            for line in output.getvalue().splitlines() + errors.getvalue().splitlines():
                self.assertIn(line, content.splitlines())

    def test_same_second_appends_and_later_start_uses_new_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            checkpoint = (settings.output_dir / 'latest.pth').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                run_training(settings, wall_clock=lambda: START)
                run_training(settings, wall_clock=lambda: START)
                run_training(settings, wall_clock=lambda: START + 2)
            files = sorted(settings.output_dir.glob('log-*.log'))
            self.assertEqual([file.name for file in files], ['log-261009-235959.log', 'log-261010-000001.log'])
            self.assertEqual(files[0].read_text().count('event=resume'), 2)
            self.assertEqual(files[0].read_text().count('event=complete'), 2)
            self.assertEqual((settings.output_dir / 'latest.pth').read_bytes(), checkpoint)

    def test_completed_run_writes_timestamped_file_and_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = completed_training(Path(temporary))
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                result = run_training(settings, wall_clock=lambda: START)
            path = settings.output_dir / 'log-261009-235959.log'
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(), output.getvalue())
            self.assertEqual(errors.getvalue(), '')
            self.assertEqual(result['batches_completed'], 3)
            self.assertIn('event=resume', path.read_text())
            self.assertIn('event=complete', path.read_text())
            for line in path.read_text().splitlines():
                self.assertTrue(line.startswith('[261009-23:59:59] '), line)


if __name__ == '__main__':
    unittest.main()
