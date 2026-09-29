import gzip
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MORTAL_DIR = REPOSITORY_ROOT / 'mortal'
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.dataset import load_dataset_state
from precompute_pipeline.precompute.precompute import main as precompute_main
from precompute_pipeline.quarantine_logs.quarantine_logs import main as quarantine_main
from precompute_pipeline.train.train import main as train_main


@unittest.skipUnless(
    os.environ.get('MORTAL_RUN_INTEGRATION') == '1',
    '设置 MORTAL_RUN_INTEGRATION=1 执行真实牌谱贯通测试',
)
class PipelineIntegrationTests(unittest.TestCase):
    def test_three_entries_run_on_isolated_real_log_copies(self):
        source_logs = sorted((REPOSITORY_ROOT / 'data/grp/val').rglob('*.json.gz'))
        reward = REPOSITORY_ROOT / 'artifacts/grp_transformer.pth'
        if len(source_logs) < 2 or not reward.exists():
            raise unittest.SkipTest('需要两份本地牌谱和 GRP Transformer 权重')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / 'source/train'
            val_dir = root / 'source/val'
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            shutil.copy2(source_logs[0], train_dir / 'train.json.gz')
            shutil.copy2(source_logs[1], val_dir / 'val.json.gz')
            with gzip.open(train_dir / 'broken.json.gz', 'wt', encoding='utf-8') as f:
                f.write('{"type":"start_game"}\n')
            config = root / 'pipeline.toml'
            config.write_text(
                '\n'.join((
                    '[quarantine]',
                    f'train_dir = "{train_dir}"',
                    f'val_dir = "{val_dir}"',
                    f'quarantine_dir = "{root / "quarantine"}"',
                    f'report_file = "{root / "quarantine/report.jsonl"}"',
                    f'progress_file = "{root / "quarantine/progress.json"}"',
                    'commit_every = 1',
                    '',
                    '[precompute]',
                    f'train_dir = "{train_dir}"',
                    f'val_dir = "{val_dir}"',
                    f'logical_root = "{root / "chunks"}"',
                    f'progress_file = "{root / "chunks/progress.json"}"',
                    f'c_root = "{root / "disk-c"}"',
                    f'd_root = "{root / "disk-d"}"',
                    'c_budget_gib = 1',
                    'd_budget_gib = 1',
                    'chunk_size = 256',
                    'file_batch_size = 1',
                    'compression_level = 1',
                    'rayon_threads = 2',
                    'torch_threads = 2',
                    '',
                    '[reward]',
                    f'state_file = "{reward}"',
                    'd_model = 128',
                    'nhead = 4',
                    'num_layers = 4',
                    'max_seq_len = 60',
                    'pts = [6.0, 4.0, 2.0, 0.0]',
                    '',
                    '[training]',
                    f'dataset_progress = "{root / "chunks/progress.json"}"',
                    f'output_dir = "{root / "artifacts/training"}"',
                    'device = "cpu"',
                    'batch_size = 128',
                    'num_epochs = 1',
                    'enable_amp = false',
                    'enable_compile = false',
                    'compile_dynamic = true',
                    'enable_cudnn_benchmark = false',
                    'prefetch = true',
                    'log_every_batches = 2',
                    'save_every_batches = 3',
                    'validation_samples = 256',
                    'validation_seed = 0',
                    '',
                    '[model]',
                    'conv_channels = 16',
                    'num_blocks = 1',
                    'freeze_bn = false',
                    '',
                    '[loss]',
                    'gamma = 1.0',
                    'cql_weight = 5.0',
                    'rank_weight = 0.2',
                    '',
                    '[optimizer]',
                    'learning_rate = 1e-4',
                    'betas = [0.9, 0.999]',
                    'eps = 1e-8',
                    'weight_decay = 0.1',
                    '',
                )),
                encoding='utf-8',
            )

            quarantine_main(['--config', str(config)])
            self.assertTrue((root / 'quarantine/train/broken.json.gz').exists())
            precompute_main(['--config', str(config)])
            dataset = load_dataset_state(root / 'chunks/progress.json')
            self.assertGreater(dataset['splits']['train']['sample_count'], 0)
            self.assertGreater(dataset['splits']['val']['sample_count'], 0)
            train_main(['--config', str(config)])
            checkpoint = torch.load(
                root / 'artifacts/training/latest.pth',
                weights_only=True,
                map_location='cpu',
            )
            self.assertEqual(checkpoint['completed_epochs'], 1)
            self.assertEqual(checkpoint['last_validation']['kind'], 'epoch')


if __name__ == '__main__':
    unittest.main()
