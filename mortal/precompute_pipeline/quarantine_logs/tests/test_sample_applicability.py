import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.settings import load_settings
from precompute_pipeline.quarantine_logs.log_validation import InvalidLogData
from precompute_pipeline.quarantine_logs.quarantine_logs import (
    ValidationResult,
    run_quarantine,
)
from precompute_pipeline.quarantine_logs.sample_applicability import validate_gameplays
from precompute_pipeline.quarantine_logs.tests.test_quarantine_logs import (
    minimal_events,
    write_quarantine_config,
    write_gzip,
)


class FakeGrp:
    def __init__(self, kyoku_count=1):
        self.feature = np.asarray([
            [index, 0, 0, 2.5, 2.5, 2.5, 2.5]
            for index in range(kyoku_count)
        ], dtype=np.float64)

    def take_feature(self):
        return self.feature

    def take_rank_by_player(self):
        return bytes([0, 1, 2, 3])

    def take_final_scores(self):
        return [40000, 30000, 20000, 10000]


class FakeGame:
    def __init__(self, *, sample_count=1, kyoku_count=1, legal=True, player_id=0):
        self.sample_count = sample_count
        self.kyoku_count = kyoku_count
        self.legal = legal
        self.player_id = player_id

    def take_obs(self):
        return [np.zeros((1012, 34), dtype=np.float32) for _ in range(self.sample_count)]

    def take_actions(self):
        return [0] * self.sample_count

    def take_masks(self):
        masks = []
        for _ in range(self.sample_count):
            mask = np.zeros(46, dtype=np.bool_)
            mask[0] = self.legal
            masks.append(mask)
        return masks

    def take_at_kyoku(self):
        return bytes([0] * self.sample_count)

    def take_dones(self):
        return [True] if self.sample_count == 0 else [True] * self.sample_count

    def take_apply_gamma(self):
        return [True] * self.sample_count

    def take_grp(self):
        return FakeGrp(self.kyoku_count)

    def take_player_id(self):
        return self.player_id


class SampleApplicabilityTests(unittest.TestCase):
    def test_allows_empty_player_view_when_another_player_has_sample(self):
        result = validate_gameplays([
            FakeGame(sample_count=0, player_id=0),
            FakeGame(sample_count=1, player_id=1),
        ], max_seq_len=60)

        self.assertEqual(result.sample_count, 1)
        self.assertEqual(result.nonempty_players, 1)

    def test_rejects_log_with_no_samples_for_any_player(self):
        with self.assertRaisesRegex(InvalidLogData, '四家合计零样本'):
            validate_gameplays(
                [FakeGame(sample_count=0, player_id=player) for player in range(4)],
                max_seq_len=60,
            )


    def test_screening_keeps_full_observation_semantic_checks(self):
        cases = ((4, 1, 1.0), (0, 0, 0.5), (961, 0, np.nan), (961, 0, np.inf))
        for plane, tile, value in cases:
            with self.subTest(plane=plane, value=value):
                observation = np.zeros((1012, 34), dtype=np.float32)
                observation[plane, tile] = value
                game = FakeGame()
                game.take_obs = lambda: [observation]
                with self.assertRaisesRegex(InvalidLogData, 'observation'):
                    validate_gameplays([game], max_seq_len=60)

    def test_rejects_action_not_allowed_by_mask(self):
        with self.assertRaisesRegex(InvalidLogData, 'mask'):
            validate_gameplays([FakeGame(legal=False)], max_seq_len=60)

    def test_rejects_log_beyond_reward_model_capacity(self):
        with self.assertRaisesRegex(InvalidLogData, '60'):
            validate_gameplays([FakeGame(kyoku_count=61)], max_seq_len=60)


class ScanResumeTests(unittest.TestCase):
    def test_resume_uses_relative_path_after_quarantined_cursor_is_gone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / 'train'
            for name in ('a.json.gz', 'b.json.gz', 'c.json.gz'):
                write_gzip(train / name, minimal_events())
            config_file = write_quarantine_config(
                root, train_dir=train, commit_every=1,
            )
            settings = load_settings(config_file).quarantine
            first_calls = []

            def first_pass(filename):
                first_calls.append(filename.name)
                if filename.name == 'a.json.gz':
                    return ValidationResult('bad a', 0.0, 0.0, 0.0)
                raise RuntimeError('simulated interruption')

            executor_factory = lambda unused: ThreadPoolExecutor(max_workers=1)
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                run_quarantine(
                    settings,
                    executor_factory=executor_factory,
                    validate_file=first_pass,
                    output=lambda unused: None,
                )

            second_calls = []

            def second_pass(filename):
                second_calls.append(filename.name)
                return ValidationResult(None, 0.0, 0.0, 0.0)

            summary = run_quarantine(
                settings,
                executor_factory=executor_factory,
                validate_file=second_pass,
                output=lambda unused: None,
            )

            self.assertEqual(first_calls, ['a.json.gz', 'b.json.gz'])
            self.assertEqual(second_calls, ['b.json.gz', 'c.json.gz'])
            self.assertEqual(summary.checked, 2)
            self.assertTrue((root / 'quarantine/train/a.json.gz').exists())

            final_calls = []
            summary = run_quarantine(
                settings,
                executor_factory=executor_factory,
                validate_file=lambda filename: final_calls.append(filename),
                output=lambda unused: None,
            )
            self.assertEqual(final_calls, [])
            self.assertEqual(summary.checked, 0)


if __name__ == '__main__':
    unittest.main()
