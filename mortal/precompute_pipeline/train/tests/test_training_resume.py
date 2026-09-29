import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.train.tests.test_reader import build_dataset
from precompute_pipeline.train.tests.test_train import training_settings
from precompute_pipeline.train.train import run_training


def with_changes(settings, **changes):
    return settings.__class__(**{**settings.__dict__, **changes})


def assert_model_states_equal(test_case, left, right):
    for model_key in ('mortal', 'dqn', 'aux_net'):
        test_case.assertEqual(left[model_key].keys(), right[model_key].keys())
        for name in left[model_key]:
            torch.testing.assert_close(
                left[model_key][name],
                right[model_key][name],
                rtol=0,
                atol=0,
            )


class TrainingResumeTests(unittest.TestCase):
    def test_interrupted_training_matches_uninterrupted_and_can_add_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress_file = build_dataset(root)
            interrupted = training_settings(root, progress_file)
            uninterrupted = with_changes(
                interrupted,
                output_dir=root / 'artifacts/uninterrupted',
            )

            with patch(
                'precompute_pipeline.train.train.secrets.randbits',
                return_value=424242,
            ):
                partial = run_training(interrupted, max_batches=2)
            self.assertEqual(partial['completed_epochs'], 0)
            self.assertEqual(partial['epoch_samples_consumed'], 8)

            from precompute_pipeline.train import train as train_module
            real_writer = train_module.SummaryWriter
            purge_steps = []

            def capture_writer(*args, **kwargs):
                purge_steps.append(kwargs.get('purge_step'))
                return real_writer(*args, **kwargs)

            with patch.object(train_module, 'SummaryWriter', side_effect=capture_writer):
                resumed = run_training(interrupted)
            self.assertEqual(purge_steps, [3])
            with patch(
                'precompute_pipeline.train.train.secrets.randbits',
                return_value=424242,
            ):
                run_training(uninterrupted)

            self.assertEqual(resumed['completed_epochs'], 1)
            self.assertEqual(resumed['samples_seen'], 9)
            resumed_state = torch.load(
                interrupted.output_dir / 'latest.pth',
                weights_only=True,
                map_location='cpu',
            )
            uninterrupted_state = torch.load(
                uninterrupted.output_dir / 'latest.pth',
                weights_only=True,
                map_location='cpu',
            )
            assert_model_states_equal(self, resumed_state, uninterrupted_state)

            extended = with_changes(interrupted, num_epochs=2)
            result = run_training(extended)
            self.assertEqual(result['completed_epochs'], 2)
            self.assertEqual(result['samples_seen'], 18)
            extended_state = torch.load(
                interrupted.output_dir / 'latest.pth',
                weights_only=True,
                map_location='cpu',
            )
            self.assertEqual(extended_state['main_seed'], 424242)

            unchanged = run_training(extended)
            self.assertEqual(unchanged['completed_epochs'], 2)
            self.assertEqual(unchanged['samples_seen'], 18)

    def test_incompatible_resume_configuration_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress_file = build_dataset(root)
            settings = training_settings(root, progress_file)
            with patch(
                'precompute_pipeline.train.train.secrets.randbits',
                return_value=123,
            ):
                run_training(settings, max_batches=2)
            checkpoint = settings.output_dir / 'latest.pth'
            before = checkpoint.read_bytes()

            incompatible = with_changes(settings, batch_size=2)
            with self.assertRaisesRegex(RuntimeError, '不兼容'):
                run_training(incompatible)

            self.assertEqual(checkpoint.read_bytes(), before)

            incompatible_validation = with_changes(settings, validation_seed=1)
            with self.assertRaisesRegex(RuntimeError, '不兼容'):
                run_training(incompatible_validation)
            self.assertEqual(checkpoint.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
