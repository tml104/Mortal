"""公开读取边界保持旧样本后缀，配置独立于筛选/奖励入口。"""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

MORTAL_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.train.tests.test_reader import build_dataset, marker_sequence
from precompute_pipeline.train.reader import DatasetBatchReader as OriginalReader
from precompute_pipeline.train_optimize.reader import DatasetBatchReader
from precompute_pipeline.train_optimize.settings import load_settings
from precompute_pipeline.train_optimize.settings import OptimizedTrainingSettings
from precompute_pipeline.train_optimize.train import run_training


class ReaderAndSettingsTests(unittest.TestCase):
    def test_unavailable_training_mode_does_not_clear_output(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            progress = build_dataset(root)
            output = root / 'artifacts/new'
            output.mkdir(parents=True)
            marker = output / 'keep.txt'
            marker.write_text('keep')
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                run_training(OptimizedTrainingSettings(progress, output, device='cpu'))
            self.assertEqual(marker.read_text(), 'keep')

    def test_new_batches_preserve_old_shuffle_suffix(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            progress = build_dataset(Path(temporary))
            arguments = dict(split='train', device='cpu', seed=912, epoch=1)
            expected = marker_sequence(list(OriginalReader(progress, batch_size=3, **arguments)))
            for prefetch in (True, False):
                reader = DatasetBatchReader(progress, batch_size=4, prefetch=prefetch, **arguments)
                for consumed in (0, 3, 4, 5, 8, 9):
                    self.assertEqual(marker_sequence(list(reader.iter_from(consumed))), expected[consumed:])
                with self.assertRaises(ValueError):
                    list(reader.iter_from(10))

    def test_training_config_defaults_without_unrelated_sections(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'training.toml'
            path.write_text('[training]\ndataset_progress="./data/progress.json"\noutput_dir="./artifacts/new"\n')
            settings = load_settings(path)
            self.assertEqual(settings.batch_size, 2048)
            self.assertEqual(settings.compile_threads, 2)
            self.assertEqual(settings.num_epochs, 30)
            self.assertEqual(settings.output_dir, Path('artifacts/new').resolve())
            path.write_text(path.read_text() + 'compile_threads=0\n')
            with self.assertRaisesRegex(ValueError, 'compile_threads'):
                load_settings(path)


if __name__ == '__main__':
    unittest.main()
