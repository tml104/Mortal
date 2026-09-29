import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.codec import decode_chunk_numpy
from precompute_pipeline.common.dataset import iter_chunk_records, load_dataset_state
from precompute_pipeline.common.files import atomic_write_json
from precompute_pipeline.common.settings import PrecomputeSettings, RewardSettings
from precompute_pipeline.precompute.precompute import run_precompute
from precompute_pipeline.precompute.tests.test_full_dataset_generation import (
    FakeGenerator,
    make_batch,
)


def build_settings(root):
    train_dir = root / 'source/train'
    val_dir = root / 'source/val'
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    (train_dir / 'train.json.gz').touch()
    (val_dir / 'val.json.gz').touch()
    return PrecomputeSettings(
        train_dir=train_dir,
        val_dir=val_dir,
        logical_root=root / 'logical',
        progress_file=root / 'logical/progress.json',
        c_root=root / 'disk-c/MortalChunks',
        d_root=root / 'disk-d/MortalChunks',
        c_budget_bytes=10_000_000,
        d_budget_bytes=10_000_000,
        chunk_size=2,
        file_batch_size=1,
        compression_level=1,
        rayon_threads=2,
        torch_threads=2,
        reward=RewardSettings(
            state_file=root / 'unused.pth',
            d_model=128,
            nhead=4,
            num_layers=4,
            max_seq_len=60,
            pts=(6.0, 4.0, 2.0, 0.0),
        ),
    )


class PrecomputeResumeTests(unittest.TestCase):
    def test_recovers_from_published_chunk_without_progress_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = build_settings(root)
            batches = {
                'train.json.gz': make_batch([1, 2, 3, 4, 5]),
                'val.json.gz': make_batch([101]),
            }
            real_atomic_write = atomic_write_json

            def fail_second_train_commit(filename, state):
                if (
                    state['current_split'] == 'train'
                    and state['splits']['train']['chunk_count'] == 2
                ):
                    raise RuntimeError('simulated progress failure')
                real_atomic_write(filename, state)

            with patch(
                'precompute_pipeline.precompute.storage.atomic_write_json',
                side_effect=fail_second_train_commit,
            ):
                with self.assertRaisesRegex(RuntimeError, 'progress failure'):
                    run_precompute(settings, generator=FakeGenerator(batches))

            committed = load_dataset_state(settings.progress_file, require_complete=False)
            generation_id = committed['generation_id']
            self.assertEqual(committed['splits']['train']['chunk_count'], 1)
            orphan = settings.d_root / 'train/chunk_00000001.mpc.zst'
            self.assertTrue(orphan.exists())

            state = run_precompute(settings, generator=FakeGenerator(batches))

            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['generation_id'], generation_id)
            markers = []
            for record in iter_chunk_records(settings.progress_file, 'train'):
                markers.extend(decode_chunk_numpy(record.path.read_bytes()).obs[:, 7, 0].tolist())
            self.assertEqual(markers, [1.0, 2.0, 3.0, 4.0, 5.0])
            self.assertEqual(state['splits']['train']['sample_count'], 5)
            self.assertEqual(state['splits']['train']['chunk_count'], 3)

    def test_corrupt_progress_fails_without_cleaning_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = build_settings(root)
            settings.progress_file.parent.mkdir(parents=True)
            settings.progress_file.write_text('{broken', encoding='utf-8')
            chunk = settings.d_root / 'train/chunk_00000000.mpc.zst'
            chunk.parent.mkdir(parents=True)
            chunk.write_bytes(b'keep')

            with self.assertRaises(json.JSONDecodeError):
                run_precompute(settings, generator=FakeGenerator({}))

            self.assertEqual(chunk.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
