import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MORTAL_DIR = REPOSITORY_ROOT / 'mortal'
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.codec import decode_chunk_numpy, encode_chunk
from precompute_pipeline.precompute.samples import SampleGenerator, build_shared_targets


class SharedTargetTests(unittest.TestCase):
    def test_builds_kyoku_rewards_and_post_kyoku_ranks(self):
        features = np.asarray([
            [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
            [1, 0, 0, 2.0, 3.0, 2.5, 2.5],
        ], dtype=np.float64)
        boundary_ranks = np.asarray([
            [0, 1, 2, 3],
            [3, 0, 1, 2],
        ])
        rank_matrix = np.zeros((2, 4, 4), dtype=np.float64)
        for boundary, ranks in enumerate(boundary_ranks):
            rank_matrix[boundary, np.arange(4), ranks] = 1.0

        rewards, ranks = build_shared_targets(
            features,
            rank_by_player=np.asarray([3, 0, 1, 2]),
            final_scores=np.asarray([10000, 40000, 30000, 20000]),
            rank_matrix=rank_matrix,
            pts=np.asarray([6.0, 4.0, 2.0, 0.0]),
        )

        np.testing.assert_array_equal(rewards, [
            [-6.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        np.testing.assert_array_equal(ranks, [
            [0, 1, 2, 3],
            [3, 0, 1, 2],
            [3, 0, 1, 2],
        ])


class RealLogSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state_file = REPOSITORY_ROOT / 'artifacts/grp_transformer.pth'
        files = sorted((REPOSITORY_ROOT / 'data/grp/val').rglob('*.json.gz'))
        if not state_file.exists() or not files:
            raise unittest.SkipTest('需要本地 GRP 权重和至少一份 val 牌谱')
        cls.state_file = state_file
        cls.log_file = files[0]

    def test_real_log_samples_survive_lossless_chunk_roundtrip(self):
        generator = SampleGenerator(
            state_file=self.state_file,
            d_model=128,
            nhead=4,
            num_layers=4,
            max_seq_len=60,
            pts=(6.0, 4.0, 2.0, 0.0),
        )

        expected = generator.load_file(self.log_file)
        actual = decode_chunk_numpy(encode_chunk(expected))

        self.assertGreater(len(expected.actions), 0)
        self.assertEqual(expected.obs.shape[1:], (1012, 34))
        self.assertTrue(expected.masks[np.arange(len(expected.actions)), expected.actions].all())
        np.testing.assert_array_equal(actual.obs, expected.obs)
        np.testing.assert_array_equal(actual.actions, expected.actions)
        np.testing.assert_array_equal(actual.masks, expected.masks)
        np.testing.assert_array_equal(actual.steps_to_done, expected.steps_to_done)
        np.testing.assert_array_equal(actual.rewards, expected.rewards)
        np.testing.assert_array_equal(actual.ranks, expected.ranks)


if __name__ == '__main__':
    unittest.main()
