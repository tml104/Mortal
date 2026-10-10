import sys
import re
import tempfile
import unittest
from pathlib import Path


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.files import iter_dataset_files
from precompute_pipeline.common.settings import load_settings


class SettingsAndFilesTests(unittest.TestCase):
    def test_loads_quarantine_paths_relative_to_current_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_file = root / 'pipeline.toml'
            config_file.write_text(
                '\n'.join((
                    '[quarantine]',
                    'train_dir = "input/train"',
                    'val_dir = "input/val"',
                    'quarantine_dir = "output/quarantine"',
                    'report_file = "output/quarantine/report.jsonl"',
                    '',
                )),
                encoding='utf-8',
            )

            settings = load_settings(config_file)

            working_dir = Path.cwd()
            self.assertEqual(settings.quarantine.train_dir, working_dir / 'input/train')
            self.assertEqual(settings.quarantine.val_dir, working_dir / 'input/val')
            self.assertEqual(
                settings.quarantine.quarantine_dir,
                working_dir / 'output/quarantine',
            )
            self.assertEqual(settings.quarantine.worker_processes, 8)
            self.assertEqual(settings.quarantine.rayon_threads_per_worker, 2)
            self.assertEqual(settings.quarantine.max_in_flight, 16)

    def test_rejects_invalid_quarantine_parallelism(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_file = root / 'pipeline.toml'
            base = (
                '[quarantine]\n'
                'train_dir = "train"\n'
                'val_dir = "val"\n'
                'quarantine_dir = "quarantine"\n'
            )
            invalid_values = (
                'worker_processes = 0\n',
                'rayon_threads_per_worker = -1\n',
                'max_in_flight = 0\n',
                'worker_processes = 3\nmax_in_flight = 2\n',
            )

            for invalid in invalid_values:
                with self.subTest(config=invalid):
                    config_file.write_text(base + invalid, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_settings(config_file)


    def test_precompute_log_interval_defaults_and_overrides(self):
        template = (MORTAL_DIR / 'precompute_pipeline/config.example.toml').read_text()
        template = template.replace('log_every_chunks = 1\n', '')
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / 'pipeline.toml'
            filename.write_text(template)
            self.assertEqual(load_settings(filename).precompute.log_every_chunks, 1)
            for raw, expected in (('3', 3), ('0', None), ('-1', None), ('true', None), ('1.5', None)):
                with self.subTest(value=raw):
                    filename.write_text(template.replace('[precompute]', f'[precompute]\nlog_every_chunks = {raw}'))
                    if expected is None:
                        with self.assertRaisesRegex(ValueError, 'log_every_chunks'):
                            load_settings(filename)
                    else:
                        self.assertEqual(load_settings(filename).precompute.log_every_chunks, expected)


    def test_precompute_thread_defaults_and_explicit_overrides(self):
        template = (MORTAL_DIR / 'precompute_pipeline/config.example.toml').read_text()
        template = re.sub(r'^(rayon_threads|torch_threads|file_batch_size) = .*\n', '', template, flags=re.M)
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / 'pipeline.toml'
            filename.write_text(template)
            settings = load_settings(filename).precompute
            self.assertEqual(
                (settings.rayon_threads, settings.torch_threads, settings.file_batch_size),
                (16, 1, 4),
            )
            filename.write_text(template.replace(
                '[precompute]',
                '[precompute]\nrayon_threads = 2\ntorch_threads = 3\nfile_batch_size = 1',
            ))
            settings = load_settings(filename).precompute
            self.assertEqual(
                (settings.rayon_threads, settings.torch_threads, settings.file_batch_size),
                (2, 3, 1),
            )

    def test_orders_dataset_files_by_relative_posix_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'b').mkdir()
            (root / 'a').mkdir()
            (root / 'b/z.json.gz').touch()
            (root / 'a/y.json.gz').touch()
            (root / 'a/ignore.json').touch()

            files = iter_dataset_files(root)

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in files],
                ['a/y.json.gz', 'b/z.json.gz'],
            )


if __name__ == '__main__':
    unittest.main()
