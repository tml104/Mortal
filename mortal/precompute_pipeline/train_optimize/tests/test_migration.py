"""通过完整迁移操作核对模型与优化历史，而不是私有映射函数。"""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

MORTAL_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORTAL_DIR))

from model import Brain, DQN, AuxNet
from precompute_pipeline.train.tests.test_reader import build_dataset
from precompute_pipeline.train.tests.test_train import training_settings
from precompute_pipeline.train.train import _build_optimizer, _training_contract
from precompute_pipeline.common.dataset import load_dataset_state
from precompute_pipeline.common.optimized_checkpoint import load_checkpoint, load_policy
from precompute_pipeline.common.optimized_model import build_models, build_optimizer
from precompute_pipeline.train_optimize.settings import OptimizedTrainingSettings
from precompute_pipeline.train_optimize.migrate import run_migration


def source_state(root, progress):
    """构造含非零 BN/AdamW 历史和轮内游标的旧格式真实状态。"""
    settings = training_settings(root, progress)
    torch.manual_seed(731)
    models = (Brain(version=4, conv_channels=16, num_blocks=1), DQN(version=4), AuxNet((4,)))
    optimizer = _build_optimizer(models, settings)
    obs = torch.randn(3, 1012, 34)
    masks = torch.ones(3, 46, dtype=torch.bool)
    phi = models[0](obs)
    loss = models[1](phi, masks).square().mean() + models[2](phi)[0].square().mean()
    loss.backward()
    optimizer.step()
    state = {
        'schema_version': 1,
        'contract': _training_contract(settings, load_dataset_state(progress)['generation_id']),
        'mortal': models[0].state_dict(), 'dqn': models[1].state_dict(),
        'aux_net': models[2].state_dict(), 'optimizer': optimizer.state_dict(),
        'scaler': {}, 'main_seed': 731, 'completed_epochs': 0, 'current_epoch': 0,
        'epoch_samples_consumed': 5, 'samples_seen': 5, 'batches_completed': 123456,
        'torch_rng_state': torch.get_rng_state(), 'metric_sums': [2.0] * 7,
        # 固定 Philox seed=731、offset=0，CPU 对照无需实际访问 GPU。
        'cuda_rng_state_all': [torch.tensor([219, 2] + [0] * 14, dtype=torch.uint8)],
        'metric_samples': 5, 'last_validation': {'kind': 'periodic', 'at_batch': 123000},
    }
    return settings, models, state, obs, masks


class MigrationTests(unittest.TestCase):
    def test_migration_preserves_policy_optimizer_and_sample_position(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = build_dataset(root)
            old, models, state, obs, masks = source_state(root, progress)
            source = root / 'old.pth'
            torch.save(state, source)
            before = source.read_bytes()
            settings = OptimizedTrainingSettings(
                dataset_progress=progress, output_dir=root / 'artifacts/new',
                model=old.model, loss=old.loss, optimizer=old.optimizer,
                enable_amp=old.enable_amp, validation_samples=old.validation_samples,
            )
            result = run_migration(settings, source)
            new = load_checkpoint(result)
            brain, dqn, metadata = load_policy(result, device='cpu')
            models[0].eval()
            models[1].eval()
            torch.testing.assert_close(brain(obs), models[0](obs), rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(dqn(brain(obs), masks), models[1](models[0](obs), masks), rtol=1e-4, atol=1e-5)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(new['batches_completed'], 123456)
            self.assertEqual(new['epoch_samples_consumed'], 5)
            self.assertEqual(new['metric_samples'], 0)
            self.assertEqual(new['metric_sums'], [0.0] * 7)
            self.assertEqual(new['last_validation'], state['last_validation'])
            self.assertEqual(new['contract']['batch_size'], 2048)
            self.assertEqual(metadata['conv_channels'], 16)
            torch.testing.assert_close(new['torch_rng_state'], state['torch_rng_state'], rtol=0, atol=0)
            torch.testing.assert_close(new['cuda_rng_state_all'][0], state['cuda_rng_state_all'][0], rtol=0, atol=0)
            for parameter_id, values in state['optimizer']['state'].items():
                migrated = new['optimizer']['state'][parameter_id]
                for name, value in values.items():
                    actual = migrated[name]
                    if value.ndim == 3:
                        actual = actual.squeeze(2)
                    torch.testing.assert_close(actual, value, rtol=0, atol=0)
            for name, value in state['mortal'].items():
                actual = new['mortal'][name]
                if value.ndim == 3:
                    actual = actual.squeeze(2)
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            new_models = build_models(settings.model)
            for key, model in zip(('mortal', 'dqn', 'aux_net'), new_models):
                model.load_state_dict(new[key])
            old_optimizer = _build_optimizer(models, old)
            old_optimizer.load_state_dict(state['optimizer'])
            new_optimizer = build_optimizer(new_models, settings.optimizer)
            new_optimizer.load_state_dict(new['optimizer'])
            for group, optimizer in ((models, old_optimizer), (new_models, new_optimizer)):
                for model in group:
                    model.train()
                optimizer.zero_grad(set_to_none=True)
                phi = group[0](obs)
                loss = group[1](phi, masks).square().mean() + group[2](phi)[0].square().mean()
                loss.backward()
            for old_model, new_model in zip(models, new_models):
                for (name, left), (_, right) in zip(old_model.named_parameters(), new_model.named_parameters()):
                    grad = right.grad.squeeze(2) if right.ndim == 4 else right.grad
                    torch.testing.assert_close(grad, left.grad, rtol=5e-3, atol=1e-5)
            old_optimizer.step()
            new_optimizer.step()
            for old_model, new_model in zip(models, new_models):
                for left, right in zip(old_model.parameters(), new_model.parameters()):
                    actual = right.squeeze(2) if right.ndim == 4 else right
                    torch.testing.assert_close(actual, left, rtol=1e-4, atol=2e-6)
            bn = next(module for module in new_models[0].modules() if isinstance(module, torch.nn.BatchNorm2d))
            counter = bn.num_batches_tracked.clone()
            new_models[0].freeze_bn(True)
            new_models[0](obs)
            torch.testing.assert_close(bn.num_batches_tracked, counter)
            new_models[0].reset_running_stats()
            self.assertEqual(bn.num_batches_tracked.item(), 0)
            with self.assertRaisesRegex(ValueError, '非空'):
                run_migration(settings, source)


if __name__ == '__main__':
    unittest.main()
