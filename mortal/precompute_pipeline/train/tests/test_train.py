import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.settings import (
    LossSettings,
    ModelSettings,
    OptimizerSettings,
    TrainingSettings,
)
from precompute_pipeline.train.tests.test_reader import build_dataset
from precompute_pipeline.train.train import run_training


def training_settings(root, progress_file):
    return TrainingSettings(
        dataset_progress=progress_file,
        output_dir=root / 'artifacts/training-output',
        device='cpu',
        batch_size=4,
        num_epochs=1,
        enable_amp=False,
        enable_compile=False,
        compile_dynamic=True,
        enable_cudnn_benchmark=False,
        prefetch=False,
        log_every_batches=1,
        save_every_batches=2,
        validation_samples=4,
        validation_seed=0,
        model=ModelSettings(conv_channels=16, num_blocks=1, freeze_bn=False),
        loss=LossSettings(gamma=1.0, cql_weight=5.0, rank_weight=0.2),
        optimizer=OptimizerSettings(
            learning_rate=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.1,
        ),
    )


class TrainingFromChunksTests(unittest.TestCase):
    def test_trains_full_epoch_including_tail_and_saves_initial_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress_file = build_dataset(root)
            settings = training_settings(root, progress_file)
            saved = []

            from precompute_pipeline.train import train as train_module
            real_save = train_module.atomic_torch_save

            def capture(state, filename):
                saved.append({
                    'samples_seen': state['samples_seen'],
                    'batches_completed': state['batches_completed'],
                    'optimizer_state_count': len(state['optimizer']['state']),
                })
                real_save(state, filename)

            with patch.object(train_module, 'atomic_torch_save', side_effect=capture):
                result = run_training(settings)

            self.assertEqual(saved[0], {
                'samples_seen': 0,
                'batches_completed': 0,
                'optimizer_state_count': 0,
            })
            self.assertEqual(result['completed_epochs'], 1)
            self.assertEqual(result['samples_seen'], 9)
            self.assertEqual(result['batches_completed'], 3)
            self.assertEqual(result['epoch_samples_consumed'], 0)
            self.assertGreater(saved[-1]['optimizer_state_count'], 0)
            checkpoint = torch.load(
                settings.output_dir / 'latest.pth',
                weights_only=True,
                map_location='cpu',
            )
            self.assertEqual(checkpoint['samples_seen'], 9)
            self.assertIn('mortal', checkpoint)
            self.assertIn('dqn', checkpoint)
            self.assertIn('aux_net', checkpoint)
            self.assertIn('scaler', checkpoint)
            event_files = list((settings.output_dir / 'tensorboard').glob('events.out.tfevents.*'))
            self.assertTrue(event_files)

    def test_rejects_output_outside_artifacts_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress_file = build_dataset(root)
            settings = training_settings(root, progress_file)
            unsafe = root / 'unsafe-output'
            unsafe.mkdir()
            marker = unsafe / 'keep.txt'
            marker.write_text('keep', encoding='utf-8')
            settings = settings.__class__(**{
                **settings.__dict__,
                'output_dir': unsafe,
            })

            with self.assertRaisesRegex(ValueError, 'artifacts'):
                run_training(settings)

            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_rejects_output_containing_protected_config_or_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress_file = build_dataset(root)
            settings = training_settings(root, progress_file)
            output = settings.output_dir
            output.mkdir(parents=True)
            config_file = output / 'pipeline.toml'
            config_file.write_text('protected', encoding='utf-8')
            settings = settings.__class__(**{
                **settings.__dict__,
                'protected_paths': (config_file,),
            })

            with self.assertRaisesRegex(ValueError, '受保护'):
                run_training(settings)

            self.assertEqual(config_file.read_text(encoding='utf-8'), 'protected')


if __name__ == '__main__':
    unittest.main()
