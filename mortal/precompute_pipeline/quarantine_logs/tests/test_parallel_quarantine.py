import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.settings import load_settings
from precompute_pipeline.quarantine_logs.quarantine_logs import (
    ValidationResult,
    run_quarantine,
)


def make_settings(root, *, workers=3, max_in_flight=3, commit_every=1):
    train = root / 'train'
    val = root / 'val'
    quarantine = root / 'quarantine'
    train.mkdir()
    val.mkdir()
    config = root / 'pipeline.toml'
    config.write_text(
        '\n'.join((
            '[quarantine]',
            f'train_dir = "{train}"',
            f'val_dir = "{val}"',
            f'quarantine_dir = "{quarantine}"',
            f'report_file = "{quarantine / "report.jsonl"}"',
            f'progress_file = "{quarantine / "progress.json"}"',
            f'commit_every = {commit_every}',
            f'worker_processes = {workers}',
            'rayon_threads_per_worker = 1',
            f'max_in_flight = {max_in_flight}',
            '',
        )),
        encoding='utf-8',
    )
    return load_settings(config).quarantine


def touch_logs(root, names):
    for name in names:
        filename = root / 'train' / name
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.touch()


def result(reason=None, *, gzip=0.001, python=0.002, libriichi=0.003):
    return ValidationResult(
        reason=reason,
        gzip_seconds=gzip,
        python_seconds=python,
        libriichi_seconds=libriichi,
    )


def thread_executor(settings):
    return ThreadPoolExecutor(max_workers=settings.worker_processes)


class TrackingExecutor:
    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()
        self.submitted = 0

    def submit(self, function, *args):
        with self.lock:
            self.submitted += 1
        return self.pool.submit(function, *args)

    def shutdown(self, *, wait, cancel_futures=False):
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class ParallelQuarantineTests(unittest.TestCase):
    def test_out_of_order_results_commit_files_reports_and_cursors_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = make_settings(root)
            names = ['a.json.gz', 'b.json.gz', 'c.json.gz']
            touch_logs(root, names)
            delays = {
                'a.json.gz': 0.06,
                'b.json.gz': 0.03,
                'c.json.gz': 0.0,
            }
            output = []

            def validate(filename):
                time.sleep(delays[filename.name])
                return result(f'bad {filename.name}')

            summary = run_quarantine(
                settings,
                executor_factory=thread_executor,
                validate_file=validate,
                output=output.append,
            )

            self.assertEqual(summary.checked, 3)
            self.assertEqual(summary.quarantined, 3)
            report = [
                json.loads(line)
                for line in settings.report_file.read_text(encoding='utf-8').splitlines()
            ]
            self.assertEqual(
                [record['source'] for record in report],
                [f'train/{name}' for name in names],
            )
            cursor_lines = [
                line for line in output
                if 'split=train' in line and 'cursor=' in line
            ]
            self.assertEqual(
                [line.split('cursor=', 1)[1].split(' ', 1)[0] for line in cursor_lines],
                names,
            )

    def test_completed_results_waiting_for_slow_prefix_fill_the_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = make_settings(root, workers=3, max_in_flight=3)
            names = [f'{letter}.json.gz' for letter in 'abcdef']
            touch_logs(root, names)
            release_first = threading.Event()
            two_fast_finished = threading.Event()
            fast_count = 0
            fast_lock = threading.Lock()
            executor = TrackingExecutor(workers=3)
            failure = []

            def validate(filename):
                nonlocal fast_count
                if filename.name == 'a.json.gz':
                    if not release_first.wait(timeout=5):
                        raise RuntimeError('test did not release first file')
                else:
                    with fast_lock:
                        fast_count += 1
                        if fast_count >= 2:
                            two_fast_finished.set()
                return result()

            def run():
                try:
                    run_quarantine(
                        settings,
                        executor_factory=lambda unused: executor,
                        validate_file=validate,
                        output=lambda unused: None,
                    )
                except BaseException as exc:
                    failure.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(two_fast_finished.wait(timeout=5))
            time.sleep(0.05)
            with executor.lock:
                self.assertEqual(executor.submitted, 3)
            release_first.set()
            thread.join(timeout=5)

            self.assertFalse(thread.is_alive())
            self.assertEqual(failure, [])
            with executor.lock:
                self.assertEqual(executor.submitted, 6)

    def test_unknown_failure_keeps_read_ahead_files_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = make_settings(root, workers=4, max_in_flight=4)
            names = ['a.json.gz', 'b.json.gz', 'c.json.gz', 'd.json.gz']
            touch_logs(root, names)

            def failing_validate(filename):
                if filename.name == 'a.json.gz':
                    time.sleep(0.01)
                    return result()
                if filename.name == 'b.json.gz':
                    time.sleep(0.08)
                    raise RuntimeError('unknown worker failure')
                time.sleep(0.02)
                return result(f'bad {filename.name}')

            with self.assertRaisesRegex(RuntimeError, 'unknown worker failure'):
                run_quarantine(
                    settings,
                    executor_factory=thread_executor,
                    validate_file=failing_validate,
                    output=lambda unused: None,
                )

            progress = json.loads(settings.progress_file.read_text(encoding='utf-8'))
            self.assertEqual(progress['status'], 'incomplete')
            self.assertEqual(progress['split'], 'train')
            self.assertEqual(progress['last_relative_path'], 'a.json.gz')
            self.assertTrue((root / 'train/b.json.gz').exists())
            self.assertTrue((root / 'train/c.json.gz').exists())
            self.assertTrue((root / 'train/d.json.gz').exists())
            self.assertFalse(settings.report_file.exists())

            resumed = []

            def resumed_validate(filename):
                resumed.append(filename.name)
                return result()

            summary = run_quarantine(
                settings,
                executor_factory=thread_executor,
                validate_file=resumed_validate,
                output=lambda unused: None,
            )

            self.assertEqual(summary.checked, 3)
            self.assertEqual(sorted(resumed), names[1:])

    def test_terminal_progress_does_not_change_progress_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = make_settings(
                root, workers=2, max_in_flight=2, commit_every=2,
            )
            touch_logs(root, ['a.json.gz', 'b.json.gz'])
            output = []

            run_quarantine(
                settings,
                executor_factory=thread_executor,
                validate_file=lambda unused: result('bad'),
                output=output.append,
            )

            train_lines = [line for line in output if 'split=train' in line]
            self.assertEqual(len(train_lines), 1)
            line = train_lines[0]
            for field in (
                'processed=2/2',
                '100.00%',
                'quarantined=2',
                'rate=',
                'eta=',
                'cursor=b.json.gz',
                'gzip=',
                'python=',
                'libriichi=',
            ):
                self.assertIn(field, line)
            progress = json.loads(settings.progress_file.read_text(encoding='utf-8'))
            self.assertEqual(
                set(progress),
                {'schema_version', 'status', 'split', 'last_relative_path'},
            )

    def test_user_interrupt_does_not_advance_past_last_committed_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = make_settings(root, workers=1, max_in_flight=1)
            touch_logs(root, ['a.json.gz', 'b.json.gz', 'c.json.gz'])

            def validate(filename):
                if filename.name == 'a.json.gz':
                    return result('bad a')
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                run_quarantine(
                    settings,
                    executor_factory=thread_executor,
                    validate_file=validate,
                    output=lambda unused: None,
                )

            progress = json.loads(settings.progress_file.read_text(encoding='utf-8'))
            self.assertEqual(progress['last_relative_path'], 'a.json.gz')
            self.assertTrue((root / 'quarantine/train/a.json.gz').exists())
            self.assertTrue((root / 'train/b.json.gz').exists())
            self.assertTrue((root / 'train/c.json.gz').exists())


if __name__ == '__main__':
    unittest.main()
