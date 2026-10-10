import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import zstandard as zstd


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.codec import (
    SCALAR_BINARY_PLANES,
    SCALAR_FLOAT_PLANES,
    TILE_BINARY_PLANES,
    TILE_FLOAT_PLANES,
    SampleBatch,
    decode_chunk_numpy,
    decode_packed_chunk,
    encode_chunk,
)


def sample_batch(sample_count=3):
    rng = np.random.default_rng(20260928)
    obs = np.zeros((sample_count, 1012, 34), dtype=np.float32)
    scalar_binary = rng.integers(
        0, 2, size=(sample_count, len(SCALAR_BINARY_PLANES)), dtype=np.uint8,
    )
    obs[:, SCALAR_BINARY_PLANES, :] = scalar_binary[:, :, None]
    obs[:, TILE_BINARY_PLANES, :] = rng.integers(
        0,
        2,
        size=(sample_count, len(TILE_BINARY_PLANES), 34),
        dtype=np.uint8,
    )
    scalar_float = rng.normal(size=(sample_count, len(SCALAR_FLOAT_PLANES))).astype(np.float32)
    obs[:, SCALAR_FLOAT_PLANES, :] = scalar_float[:, :, None]
    obs[:, TILE_FLOAT_PLANES, :] = rng.normal(
        size=(sample_count, len(TILE_FLOAT_PLANES), 34),
    ).astype(np.float32)
    actions = np.arange(sample_count, dtype=np.int64) % 46
    masks = np.zeros((sample_count, 46), dtype=np.bool_)
    masks[np.arange(sample_count), actions] = True
    masks[:, 45] = True
    return SampleBatch(
        obs=obs,
        actions=actions,
        masks=masks,
        steps_to_done=np.arange(sample_count, dtype=np.int64),
        rewards=np.linspace(-1.5, 2.5, sample_count, dtype=np.float64),
        ranks=np.arange(sample_count, dtype=np.int64) % 4,
    )


class ChunkCodecTests(unittest.TestCase):
    def test_importing_codec_does_not_load_torch(self):
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(MORTAL_DIR)
        completed = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import sys; '
                    'import precompute_pipeline.common.codec; '
                    'assert "torch" not in sys.modules'
                ),
            ],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_fixed_plane_partition_covers_v4_observation(self):
        categories = (
            SCALAR_BINARY_PLANES,
            TILE_BINARY_PLANES,
            SCALAR_FLOAT_PLANES,
            TILE_FLOAT_PLANES,
        )

        self.assertEqual([len(value) for value in categories], [417, 514, 19, 62])
        flattened = [plane for category in categories for plane in category]
        self.assertEqual(len(set(flattened)), 1012)
        self.assertEqual(sorted(flattened), list(range(1012)))

    def test_roundtrip_preserves_all_six_fields_exactly(self):
        expected = sample_batch()

        encoded = encode_chunk(expected, compression_level=1)
        packed = decode_packed_chunk(encoded)
        actual = decode_chunk_numpy(packed)

        self.assertEqual(packed.sample_count, 3)
        self.assertEqual(encoded[:4], b'\x28\xb5\x2f\xfd')
        np.testing.assert_array_equal(actual.obs, expected.obs)
        np.testing.assert_array_equal(actual.actions, expected.actions)
        np.testing.assert_array_equal(actual.masks, expected.masks)
        np.testing.assert_array_equal(actual.steps_to_done, expected.steps_to_done)
        np.testing.assert_array_equal(actual.rewards, expected.rewards)
        np.testing.assert_array_equal(actual.ranks, expected.ranks)


    def test_encoding_preserves_pre_optimization_payload(self):
        # 来自实现前 codec 的三条固定样本参考；与压缩器版本无关。
        payload = zstd.ZstdDecompressor().decompress(encode_chunk(sample_batch(3)))
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            '471cc9b25489b4f17733360e5a51759b2a1a7a15290bbbe061e3f8fbf41cefc5',
        )

    def test_keeps_required_structure_label_and_reward_checks(self):
        mutations = {
            'obs dtype': lambda batch: setattr(batch, 'obs', batch.obs.astype(np.float64)),
            'obs shape': lambda batch: setattr(batch, 'obs', batch.obs[:, :, :33]),
            'reward': lambda batch: batch.rewards.__setitem__(0, np.inf),
            'action': lambda batch: batch.actions.__setitem__(0, 46),
            'steps': lambda batch: batch.steps_to_done.__setitem__(0, -1),
            'rank': lambda batch: batch.ranks.__setitem__(0, 4),
            'length': lambda batch: setattr(batch, 'ranks', batch.ranks[:1]),
        }
        for name, mutate in mutations.items():
            with self.subTest(field=name):
                batch = sample_batch()
                mutate(batch)
                with self.assertRaises(ValueError):
                    encode_chunk(batch)

    def test_rejects_action_not_allowed_by_mask(self):
        batch = sample_batch()
        batch.masks[np.arange(3), batch.actions] = False

        with self.assertRaisesRegex(ValueError, 'mask'):
            encode_chunk(batch)

    def test_rejects_truncated_chunk(self):
        encoded = encode_chunk(sample_batch())

        with self.assertRaises(ValueError):
            decode_packed_chunk(encoded[:-3])


if __name__ == '__main__':
    unittest.main()
