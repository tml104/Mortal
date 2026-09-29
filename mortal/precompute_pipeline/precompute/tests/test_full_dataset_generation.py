import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.codec import SampleBatch, decode_chunk_numpy, encode_chunk
from precompute_pipeline.common.dataset import iter_chunk_records, load_dataset_state
from precompute_pipeline.common.settings import PrecomputeSettings, RewardSettings
from precompute_pipeline.precompute.precompute import run_precompute


def make_batch(markers):
    markers = np.asarray(markers, dtype=np.float32)
    sample_count = len(markers)
    obs = np.zeros((sample_count, 1012, 34), dtype=np.float32)
    obs[:, 7, :] = markers[:, None]
    actions = np.arange(sample_count, dtype=np.uint8) % 46
    masks = np.zeros((sample_count, 46), dtype=np.bool_)
    masks[np.arange(sample_count), actions] = True
    return SampleBatch(
        obs=obs,
        actions=actions,
        masks=masks,
        steps_to_done=np.arange(sample_count, dtype=np.uint16),
        rewards=markers.astype(np.float64),
        ranks=np.arange(sample_count, dtype=np.uint8) % 4,
    )


class FakeGenerator:
    def __init__(self, batches):
        self.batches = batches
        self.calls = []

    def load_files(self, filenames):
        self.calls.append([Path(filename).name for filename in filenames])
        return [self.batches[Path(filename).name] for filename in filenames]


class FullDatasetGenerationTests(unittest.TestCase):
    def test_generates_ordered_fixed_chunks_across_two_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / 'source/train'
            val_dir = root / 'source/val'
            for directory, names in (
                (train_dir, ('a.json.gz', 'b.json.gz', 'c.json.gz')),
                (val_dir, ('v.json.gz',)),
            ):
                directory.mkdir(parents=True)
                for name in names:
                    (directory / name).touch()
            batches = {
                'a.json.gz': make_batch([1, 2, 3]),
                'b.json.gz': make_batch([4, 5, 6, 7]),
                'c.json.gz': make_batch([8, 9]),
                'v.json.gz': make_batch([101, 102]),
            }
            first_chunk_size = len(encode_chunk(make_batch([1, 2, 3, 4])))
            c_root = root / 'disk-c/MortalChunks'
            d_root = root / 'disk-d/MortalChunks'
            (d_root / 'train').mkdir(parents=True)
            (d_root / 'train/chunk_00000000.mpc.zst').write_bytes(b'old')
            (d_root / 'train/keep.txt').write_text('keep', encoding='utf-8')
            settings = PrecomputeSettings(
                train_dir=train_dir,
                val_dir=val_dir,
                logical_root=root / 'logical',
                progress_file=root / 'logical/progress.json',
                c_root=c_root,
                d_root=d_root,
                c_budget_bytes=10_000_000,
                d_budget_bytes=first_chunk_size,
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

            progress = run_precompute(settings, generator=FakeGenerator(batches))

            self.assertEqual(progress['status'], 'complete')
            state = load_dataset_state(settings.progress_file)
            self.assertEqual(state['generation_id'], progress['generation_id'])
            self.assertEqual(state['splits']['train']['sample_count'], 9)
            self.assertEqual(state['splits']['train']['chunk_count'], 3)
            self.assertEqual(state['splits']['train']['tail_sample_count'], 1)
            self.assertEqual(state['splits']['val']['sample_count'], 2)
            self.assertEqual(state['splits']['val']['chunk_count'], 1)
            self.assertEqual(state['splits']['val']['tail_sample_count'], 2)

            train_records = iter_chunk_records(settings.progress_file, 'train')
            self.assertEqual([record.volume for record in train_records], ['d', 'c', 'c'])
            observed = []
            for record in train_records:
                batch = decode_chunk_numpy(record.path.read_bytes())
                observed.extend(batch.obs[:, 7, 0].tolist())
            self.assertEqual(observed, list(range(1, 10)))

            val_records = iter_chunk_records(settings.progress_file, 'val')
            self.assertEqual([record.volume for record in val_records], ['c'])
            val_batch = decode_chunk_numpy(val_records[0].path.read_bytes())
            self.assertEqual(val_batch.obs[:, 7, 0].tolist(), [101.0, 102.0])
            self.assertTrue((settings.logical_root / 'train/c').is_symlink())
            self.assertTrue((settings.logical_root / 'train/d').is_symlink())
            self.assertTrue((settings.logical_root / 'val/c').is_symlink())
            self.assertEqual((d_root / 'train/keep.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
