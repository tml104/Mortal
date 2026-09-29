import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.settings import load_settings
from precompute_pipeline.quarantine_logs.log_validation import (
    InvalidLogData,
    validate_protocol_text,
)
from precompute_pipeline.quarantine_logs.quarantine_logs import run_quarantine


def minimal_events():
    return [
        {'type': 'start_game', 'names': ['a', 'b', 'c', 'd']},
        {
            'type': 'start_kyoku',
            'bakaze': 'E',
            'dora_marker': '1m',
            'kyoku': 1,
            'honba': 0,
            'kyotaku': 0,
            'oya': 0,
            'scores': [25000, 25000, 25000, 25000],
            'tehais': [['?'] * 13 for _ in range(4)],
        },
        {'type': 'ryukyoku', 'deltas': [0, 0, 0, 0]},
        {'type': 'end_kyoku'},
        {'type': 'end_game'},
    ]


def render(events):
    return ''.join(json.dumps(event) + '\n' for event in events)


def write_gzip(filename, events):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        f.write(render(events))


def write_quarantine_config(
    root,
    *,
    train_dir=None,
    val_dir=None,
    quarantine_dir=None,
    commit_every=100,
):
    """为筛选工作流测试写入统一的单 worker 配置。"""
    train_dir = root / 'train' if train_dir is None else train_dir
    val_dir = root / 'val' if val_dir is None else val_dir
    quarantine_dir = (
        root / 'quarantine' if quarantine_dir is None else quarantine_dir
    )
    config_file = root / 'pipeline.toml'
    config_file.write_text(
        '\n'.join((
            '[quarantine]',
            f'train_dir = "{train_dir}"',
            f'val_dir = "{val_dir}"',
            f'quarantine_dir = "{quarantine_dir}"',
            f'report_file = "{quarantine_dir / "report.jsonl"}"',
            f'progress_file = "{quarantine_dir / "progress.json"}"',
            f'commit_every = {commit_every}',
            'worker_processes = 1',
            'rayon_threads_per_worker = 1',
            'max_in_flight = 1',
            '',
        )),
        encoding='utf-8',
    )
    return config_file


class ProtocolOnlyValidator:
    def validate_raw_log(self, raw_log):
        return None


class ProtocolValidationTests(unittest.TestCase):
    def test_accepts_complete_game_with_multiple_hora(self):
        events = minimal_events()
        events[2:3] = [
            {'type': 'hora', 'actor': 1, 'target': 0, 'deltas': [-8000, 8000, 0, 0]},
            {'type': 'hora', 'actor': 2, 'target': 0, 'deltas': [-8000, 0, 8000, 0]},
        ]

        parsed = validate_protocol_text(render(events))

        self.assertEqual([event['type'] for event in parsed[-4:]], [
            'hora', 'hora', 'end_kyoku', 'end_game',
        ])

    def test_rejects_kyoku_without_result(self):
        events = minimal_events()
        del events[2]

        with self.assertRaisesRegex(InvalidLogData, '结算'):
            validate_protocol_text(render(events))

    def test_rejects_missing_required_event_field(self):
        events = minimal_events()
        del events[1]['scores']

        with self.assertRaisesRegex(InvalidLogData, 'scores'):
            validate_protocol_text(render(events))

    def test_rejects_unhashable_type_and_tile_as_input_errors(self):
        events = minimal_events()
        events[1]['dora_marker'] = []
        with self.assertRaises(InvalidLogData):
            validate_protocol_text(render(events))

        events = minimal_events()
        events[2]['type'] = {}
        with self.assertRaises(InvalidLogData):
            validate_protocol_text(render(events))


class QuarantineWorkflowTests(unittest.TestCase):
    def test_importing_quarantine_entry_does_not_load_training_runtime(self):
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(MORTAL_DIR)
        completed = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import sys; '
                    'import precompute_pipeline.quarantine_logs.quarantine_logs; '
                    'assert "libriichi" not in sys.modules; '
                    'assert "torch" not in sys.modules'
                ),
            ],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_single_spawn_worker_keeps_libriichi_out_of_parent_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'train/corrupt.json.gz'
            source.parent.mkdir(parents=True)
            source.write_bytes(
                b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff'
                b'\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00'
            )
            config_file = write_quarantine_config(root)

            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(MORTAL_DIR)
            completed = subprocess.run(
                [
                    sys.executable,
                    '-c',
                    (
                        'import sys; '
                        'from precompute_pipeline.common.settings import load_settings; '
                        'from precompute_pipeline.quarantine_logs.quarantine_logs '
                        'import run_quarantine; '
                        'assert "libriichi" not in sys.modules; '
                        f'summary = run_quarantine(load_settings({str(config_file)!r})'
                        '.quarantine, output=lambda unused: None); '
                        'assert summary.quarantined == 1; '
                        'assert "libriichi" not in sys.modules'
                    ),
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_moves_only_invalid_log_and_records_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / 'data/train'
            val = root / 'data/val'
            quarantine = root / 'data/quarantine'
            valid = train / 'season/valid.json.gz'
            invalid = val / 'season/invalid.json.gz'
            write_gzip(valid, minimal_events())
            write_gzip(invalid, minimal_events()[:-1])
            config_file = write_quarantine_config(
                root,
                train_dir=train,
                val_dir=val,
                quarantine_dir=quarantine,
            )

            summary = run_quarantine(
                load_settings(config_file).quarantine,
                sample_validator=ProtocolOnlyValidator(),
            )

            target = quarantine / 'val/season/invalid.json.gz'
            self.assertTrue(valid.exists())
            self.assertFalse(invalid.exists())
            self.assertTrue(target.exists())
            self.assertEqual(summary.checked, 2)
            self.assertEqual(summary.quarantined, 1)
            report = [
                json.loads(line)
                for line in (quarantine / 'report.jsonl').read_text().splitlines()
            ]
            self.assertEqual(report[0]['source'], 'val/season/invalid.json.gz')
            self.assertIn('end_game', report[0]['reason'])

    def test_does_not_overwrite_existing_quarantine_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            val = root / 'val'
            source = val / 'same.json.gz'
            target = root / 'quarantine/val/same.json.gz'
            write_gzip(source, minimal_events()[:-1])
            write_gzip(target, minimal_events())
            config_file = write_quarantine_config(root, val_dir=val)

            with self.assertRaises(FileExistsError):
                run_quarantine(
                    load_settings(config_file).quarantine,
                    sample_validator=ProtocolOnlyValidator(),
                )

            self.assertTrue(source.exists())

    def test_quarantines_corrupt_deflate_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'train/corrupt.json.gz'
            source.parent.mkdir(parents=True)
            source.write_bytes(
                b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff'
                b'\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00'
            )
            config_file = write_quarantine_config(root)

            summary = run_quarantine(
                load_settings(config_file).quarantine,
                sample_validator=ProtocolOnlyValidator(),
            )

            self.assertEqual(summary.quarantined, 1)
            self.assertTrue((root / 'quarantine/train/corrupt.json.gz').exists())


if __name__ == '__main__':
    unittest.main()
