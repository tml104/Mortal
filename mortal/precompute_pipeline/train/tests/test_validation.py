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


class ValidationCommitTests(unittest.TestCase):
    def test_runs_periodic_subset_then_full_validation_before_saves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress_file = build_dataset(root)
            settings = training_settings(root, progress_file)
            from precompute_pipeline.train import train as train_module
            real_evaluate = train_module._evaluate
            real_save = train_module.atomic_torch_save
            validation_limits = []
            commits = []

            def observe_evaluate(*args, **kwargs):
                validation_limits.append(kwargs['max_samples'])
                return real_evaluate(*args, **kwargs)

            def observe_save(state, filename):
                commits.append({
                    'samples_seen': state['samples_seen'],
                    'completed_epochs': state['completed_epochs'],
                    'validation_kind': (
                        state['last_validation']['kind']
                        if state.get('last_validation') is not None
                        else None
                    ),
                })
                real_save(state, filename)

            with (
                patch.object(train_module, '_evaluate', side_effect=observe_evaluate),
                patch.object(train_module, 'atomic_torch_save', side_effect=observe_save),
            ):
                run_training(settings)

            self.assertEqual(validation_limits, [settings.validation_samples, None])
            self.assertEqual(commits, [
                {'samples_seen': 0, 'completed_epochs': 0, 'validation_kind': None},
                {'samples_seen': 8, 'completed_epochs': 0, 'validation_kind': 'periodic'},
                {'samples_seen': 9, 'completed_epochs': 1, 'validation_kind': 'epoch'},
            ])
            final = torch.load(
                settings.output_dir / 'latest.pth',
                weights_only=True,
                map_location='cpu',
            )
            self.assertEqual(final['last_validation']['kind'], 'epoch')
            self.assertEqual(final['last_validation']['sample_count'], 2)

    def test_validation_failure_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress_file = build_dataset(root)
            settings = training_settings(root, progress_file)
            settings = settings.__class__(
                **{**settings.__dict__, 'save_every_batches': 1},
            )
            from precompute_pipeline.train import train as train_module

            with patch.object(
                train_module,
                '_evaluate',
                side_effect=RuntimeError('simulated validation failure'),
            ):
                with self.assertRaisesRegex(RuntimeError, 'validation failure'):
                    run_training(settings)

            checkpoint = torch.load(
                settings.output_dir / 'latest.pth',
                weights_only=True,
                map_location='cpu',
            )
            self.assertEqual(checkpoint['samples_seen'], 0)
            self.assertEqual(checkpoint['batches_completed'], 0)
            self.assertIsNone(checkpoint['last_validation'])


if __name__ == '__main__':
    unittest.main()
