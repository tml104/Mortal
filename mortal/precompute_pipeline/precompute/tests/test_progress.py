import io
from dataclasses import replace
import json
from contextlib import redirect_stdout
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.files import atomic_write_json
from precompute_pipeline.precompute.precompute import run_precompute
from precompute_pipeline.precompute.tests.test_full_dataset_generation import (
    FakeGenerator,
    make_batch,
)
from precompute_pipeline.precompute.tests.test_precompute_resume import build_settings


def log_records(output):
    """仅解析用户可见的预计算日志字段，忽略空格排版。"""
    return [
        dict(word.split('=', 1) for word in shlex.split(line)[1:])
        for line in output.splitlines() if line.startswith('[precompute] ')
    ]


class ObservedOutput(io.StringIO):
    """在每次终端刷新时同时观察当时已经落盘的进度。"""

    def __init__(self, progress_file):
        super().__init__()
        self.progress_file = progress_file
        self.snapshots = []

    def flush(self):
        records = log_records(self.getvalue())
        if records:
            state = None
            if self.progress_file.exists():
                state = json.loads(self.progress_file.read_text())
            self.snapshots.append((records[-1], state))
        super().flush()


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PrecomputeProgressTests(unittest.TestCase):
    def test_default_logs_only_committed_chunks_and_flushes_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = build_settings(Path(temporary))
            output = ObservedOutput(settings.progress_file)
            generator = FakeGenerator({
                'train.json.gz': make_batch([1, 2, 3, 4, 5]),
                'val.json.gz': make_batch([101]),
            })
            with redirect_stdout(output):
                run_precompute(settings, generator=generator)

            records = log_records(output.getvalue())
            self.assertEqual(records[0]['event'], 'start')
            self.assertEqual(records[-1]['event'], 'complete')
            self.assertEqual(
                [r['split'] for r in records if r['event'] == 'scanning'],
                ['train', 'val'],
            )
            chunks = [r for r in records if r['event'] == 'chunk_committed']
            self.assertEqual(
                [(r['split'], int(r['chunks']), int(r['samples'])) for r in chunks],
                [('train', 1, 2), ('train', 2, 4), ('train', 3, 5), ('val', 1, 1)],
            )
            committed = [
                (r, state) for r, state in output.snapshots
                if r['event'] == 'chunk_committed'
            ]
            self.assertEqual(len(committed), 4)
            for record, state in committed:
                split = record['split']
                self.assertEqual(int(record['chunks']), state['splits'][split]['chunk_count'])
                self.assertEqual(int(record['samples']), state['splits'][split]['sample_count'])
                self.assertIn('c_gib', record)
                self.assertIn('d_gib', record)
                self.assertIn('cursor', record)
                self.assertIn('offset', record)

    def test_custom_interval_still_reports_tail_empty_split_and_completed_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = replace(build_settings(Path(temporary)), log_every_chunks=2)
            (settings.val_dir / 'val.json.gz').unlink()
            generator = FakeGenerator({'train.json.gz': make_batch([1, 2, 3, 4, 5])})
            output = io.StringIO()
            with redirect_stdout(output):
                state = run_precompute(settings, generator=generator)
            records = log_records(output.getvalue())
            commits = [r for r in records if r['event'] == 'chunk_committed']
            self.assertEqual([(r['split'], r['chunks']) for r in commits], [('train', '2')])
            finished = [r for r in records if r['event'] == 'split_complete']
            self.assertEqual(
                [(r['split'], r['chunks'], r['samples']) for r in finished],
                [('train', '3', '5'), ('val', '0', '0')],
            )
            previous = settings.progress_file.read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                resumed = run_precompute(settings, generator=FakeGenerator({}))
            self.assertEqual([r['event'] for r in log_records(output.getvalue())], ['already_complete'])
            self.assertEqual(resumed, state)
            self.assertEqual(settings.progress_file.read_bytes(), previous)

    def test_failed_chunk_or_progress_write_never_reports_success(self):
        for failure in ('chunk', 'progress'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                settings = build_settings(Path(temporary))
                generator = FakeGenerator({
                    'train.json.gz': make_batch([1, 2, 3, 4, 5]),
                    'val.json.gz': make_batch([101]),
                })
                output = io.StringIO()
                if failure == 'chunk':
                    injection = patch(
                        'precompute_pipeline.precompute.storage.atomic_write_bytes',
                        side_effect=OSError('disk failure'),
                    )
                    expected_commits = []
                else:
                    def fail_second(filename, state):
                        if state['current_split'] == 'train' and state['splits']['train']['chunk_count'] == 2:
                            raise OSError('progress failure')
                        atomic_write_json(filename, state)
                    injection = patch(
                        'precompute_pipeline.precompute.storage.atomic_write_json',
                        side_effect=fail_second,
                    )
                    expected_commits = ['1']
                with injection, redirect_stdout(output), self.assertRaises(OSError):
                    run_precompute(settings, generator=generator)
                records = log_records(output.getvalue())
                self.assertEqual(
                    [r['chunks'] for r in records if r['event'] == 'chunk_committed'],
                    expected_commits,
                )
                self.assertNotIn('complete', [r['event'] for r in records])
                state = json.loads(settings.progress_file.read_text())
                self.assertEqual(state['splits']['train']['chunk_count'], len(expected_commits))

    def test_file_progress_counts_only_complete_committed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = build_settings(Path(temporary))
            (settings.val_dir / 'val.json.gz').unlink()
            clock = ManualClock()
            batches = {
                'train.json.gz': make_batch([1, 2, 3, 4, 5]),
            }

            class TimedGenerator(FakeGenerator):
                def load_files(self, filenames):
                    clock.now += 10.0
                    return super().load_files(filenames)

            output = io.StringIO()
            with redirect_stdout(output):
                run_precompute(settings, generator=TimedGenerator(batches), clock=clock)
            chunks = [
                r for r in log_records(output.getvalue()) if r['event'] == 'chunk_committed'
            ]
            self.assertEqual([r['files'] for r in chunks], ['0/1', '0/1', '1/1'])
            self.assertEqual([float(r['percent']) for r in chunks], [0.0, 0.0, 100.0])
            self.assertEqual([float(r['run_samples_s']) for r in chunks], [0.2, 0.4, 0.5])
            self.assertEqual([r['eta_s'] for r in chunks], ['unknown', 'unknown', '0.0'])

    def test_resume_distinguishes_middle_and_end_of_source_file(self):
        for first_size in (2, 3):
            with self.subTest(first_size=first_size), tempfile.TemporaryDirectory() as temporary:
                settings = build_settings(Path(temporary))
                (settings.train_dir / 'train.json.gz').unlink()
                (settings.val_dir / 'val.json.gz').unlink()
                for name in ('a.json.gz', 'b.json.gz'):
                    (settings.train_dir / name).touch()
                batches = {
                    'a.json.gz': make_batch(list(range(1, first_size + 1))),
                    'b.json.gz': make_batch([10, 11, 12]),
                }

                def fail_second(filename, state):
                    if state['current_split'] == 'train' and state['splits']['train']['chunk_count'] == 2:
                        raise OSError('stop after first saved chunk')
                    atomic_write_json(filename, state)

                with patch(
                    'precompute_pipeline.precompute.storage.atomic_write_json',
                    side_effect=fail_second,
                ), redirect_stdout(io.StringIO()), self.assertRaises(OSError):
                    run_precompute(settings, generator=FakeGenerator(batches))
                previous = json.loads(settings.progress_file.read_text())
                self.assertEqual(previous['source_cursor'], {
                    'relative_path': 'a.json.gz', 'sample_offset': 2,
                })
                clock = ManualClock()

                class TimedGenerator(FakeGenerator):
                    def load_files(self, filenames):
                        clock.now += 10.0
                        return super().load_files(filenames)

                output = io.StringIO()
                with redirect_stdout(output):
                    final = run_precompute(
                        replace(settings, rayon_threads=1, torch_threads=1),
                        generator=TimedGenerator(batches),
                        clock=clock,
                    )
                records = log_records(output.getvalue())
                self.assertEqual(records[0]['event'], 'resume')
                self.assertEqual(records[0]['samples'], '2')
                self.assertEqual(float(records[0]['run_samples_s']), 0.0)
                first = next(r for r in records if r['event'] == 'chunk_committed')
                self.assertEqual(first['files'], '1/2')
                self.assertEqual(float(first['run_samples_s']), 0.1)
                self.assertEqual(float(first['run_files_s']), 0.0 if first_size == 2 else 0.05)
                self.assertEqual(first['eta_s'], 'unknown' if first_size == 2 else '20.0')
                self.assertEqual(final['generation_id'], previous['generation_id'])
                self.assertEqual(final['splits']['train']['sample_count'], first_size + 3)
                self.assertEqual(set(final), set(previous))
                self.assertEqual(set(final['splits']['train']), set(previous['splits']['train']))

    def test_multiple_files_and_tail_report_current_split_eta(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = build_settings(Path(temporary))
            (settings.train_dir / 'train.json.gz').unlink()
            for name in ('a.json.gz', 'b.json.gz', 'c.json.gz'):
                (settings.train_dir / name).touch()
            clock = ManualClock()

            class TimedGenerator(FakeGenerator):
                def load_files(self, filenames):
                    clock.now += 10.0
                    return super().load_files(filenames)

            batches = {
                'a.json.gz': make_batch([1, 2, 3]),
                'b.json.gz': make_batch([4, 5, 6]),
                'c.json.gz': make_batch([]),
                'val.json.gz': make_batch([101, 102, 103]),
            }
            output = io.StringIO()
            with redirect_stdout(output):
                run_precompute(settings, generator=TimedGenerator(batches), clock=clock)
            records = log_records(output.getvalue())
            chunks = [r for r in records if r['event'] == 'chunk_committed']
            self.assertEqual([r['files'] for r in chunks], ['0/3', '1/3', '2/3', '0/1', '1/1'])
            self.assertEqual(chunks[1]['eta_s'], '40.0')
            train_end = next(r for r in records if r['event'] == 'split_complete')
            self.assertEqual(train_end['files'], '3/3')
            self.assertEqual(train_end['eta_s'], '0.0')
            # 全局本次文件速度为 3/40，而 ETA 只估算 val 剩余的一份。
            self.assertEqual(chunks[3]['eta_s'], '13.3')
            self.assertEqual(records[-1]['eta_s'], '0.0')

    def test_five_stage_times_exclude_nested_backend_time_and_reset_on_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = build_settings(Path(temporary))
            (settings.val_dir / 'val.json.gz').unlink()
            clock = ManualClock()

            class MeasuredGenerator(FakeGenerator):
                def load_files(self, filenames):
                    with self.timings.measure('rust'):
                        clock.now += 2.0
                    with self.timings.measure('grp'):
                        clock.now += 3.0
                    clock.now += 5.0
                    return super().load_files(filenames)

            generator = MeasuredGenerator({'train.json.gz': make_batch([1, 2])})
            output = io.StringIO()
            with redirect_stdout(output):
                run_precompute(settings, generator=generator, clock=clock)
            chunk = next(r for r in log_records(output.getvalue()) if r['event'] == 'chunk_committed')
            self.assertEqual(
                [float(chunk[key]) for key in ('rust_s', 'grp_s', 'python_s', 'encode_s', 'commit_s')],
                [2.0, 3.0, 5.0, 0.0, 0.0],
            )
            output = io.StringIO()
            with redirect_stdout(output):
                run_precompute(settings, generator=generator, clock=clock)
            restart = log_records(output.getvalue())[0]
            self.assertEqual(restart['event'], 'already_complete')
            self.assertEqual(
                [float(restart[key]) for key in ('rust_s', 'grp_s', 'python_s', 'encode_s', 'commit_s')],
                [0.0] * 5,
            )

    def test_resume_after_tail_commit_does_not_regenerate_or_count_old_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = build_settings(Path(temporary))
            (settings.val_dir / 'val.json.gz').unlink()

            def fail_split_switch(filename, state):
                if state['current_split'] == 'val':
                    raise OSError('split transition interrupted')
                atomic_write_json(filename, state)

            with patch(
                'precompute_pipeline.precompute.storage.atomic_write_json',
                side_effect=fail_split_switch,
            ), redirect_stdout(io.StringIO()), self.assertRaises(OSError):
                run_precompute(settings, generator=FakeGenerator({
                    'train.json.gz': make_batch([1, 2, 3]),
                }))
            previous = json.loads(settings.progress_file.read_text())
            self.assertTrue(previous['source_cursor']['eof'])
            output = io.StringIO()
            with redirect_stdout(output):
                state = run_precompute(settings, generator=FakeGenerator({}), clock=ManualClock())
            records = log_records(output.getvalue())
            self.assertEqual([r for r in records if r['event'] == 'chunk_committed'], [])
            end = next(r for r in records if r['event'] == 'split_complete')
            self.assertEqual(end['files'], '1/1')
            self.assertEqual(float(end['run_files_s']), 0.0)
            self.assertEqual(float(end['run_samples_s']), 0.0)
            self.assertEqual(state['generation_id'], previous['generation_id'])
            self.assertEqual(state['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
