import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.settings import PrecomputeSettings, RewardSettings
from precompute_pipeline.precompute.precompute import run_precompute
from precompute_pipeline.precompute.tests.test_full_dataset_generation import (
    FakeGenerator,
    make_batch,
)
from precompute_pipeline.train.reader import DatasetBatchReader


def build_dataset(root):
    train_dir = root / 'source/train'
    val_dir = root / 'source/val'
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    batches = {}
    for name, markers in (
        ('a.json.gz', [1, 2, 3]),
        ('b.json.gz', [4, 5, 6, 7]),
        ('c.json.gz', [8, 9]),
    ):
        (train_dir / name).touch()
        batches[name] = make_batch(markers)
    (val_dir / 'v.json.gz').touch()
    batches['v.json.gz'] = make_batch([101, 102])
    settings = PrecomputeSettings(
        train_dir=train_dir,
        val_dir=val_dir,
        logical_root=root / 'logical',
        progress_file=root / 'logical/progress.json',
        c_root=root / 'disk-c/MortalChunks',
        d_root=root / 'disk-d/MortalChunks',
        c_budget_bytes=10_000_000,
        d_budget_bytes=10_000_000,
        chunk_size=4,
        file_batch_size=2,
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
    run_precompute(settings, generator=FakeGenerator(batches))
    return settings.progress_file


def marker_sequence(batches):
    if not batches:
        return []
    return np.concatenate([batch[0][:, 7, 0].cpu().numpy() for batch in batches]).tolist()


class DatasetBatchReaderTests(unittest.TestCase):
    def test_repeats_same_two_level_shuffle_and_keeps_final_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            progress = build_dataset(Path(temporary))
            first = list(DatasetBatchReader(
                progress,
                split='train',
                batch_size=3,
                device='cpu',
                seed=1234,
                epoch=0,
                prefetch=True,
            ))
            second = list(DatasetBatchReader(
                progress,
                split='train',
                batch_size=3,
                device='cpu',
                seed=1234,
                epoch=0,
                prefetch=False,
            ))
            other_epoch = list(DatasetBatchReader(
                progress,
                split='train',
                batch_size=3,
                device='cpu',
                seed=1234,
                epoch=1,
                prefetch=False,
            ))

            self.assertEqual([len(batch[0]) for batch in first], [3, 3, 3])
            self.assertEqual(marker_sequence(first), marker_sequence(second))
            self.assertCountEqual(marker_sequence(first), list(range(1, 10)))
            self.assertNotEqual(marker_sequence(first), marker_sequence(other_epoch))
            for batch in first:
                rows = np.arange(len(batch[1]))
                self.assertTrue(batch[2][rows, batch[1]].all())

    def test_resume_position_matches_suffix_without_reading_prior_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            progress = build_dataset(Path(temporary))
            reader = DatasetBatchReader(
                progress,
                split='train',
                batch_size=4,
                device='cpu',
                seed=987,
                epoch=0,
                prefetch=True,
            )
            complete = marker_sequence(list(reader))
            suffix = marker_sequence(list(reader.iter_from(5)))

            self.assertEqual(suffix, complete[5:])
            self.assertEqual([len(batch[0]) for batch in reader.iter_from(5)], [4])
            self.assertEqual(list(reader.iter_from(9)), [])
            with self.assertRaises(ValueError):
                list(reader.iter_from(10))


if __name__ == '__main__':
    unittest.main()
