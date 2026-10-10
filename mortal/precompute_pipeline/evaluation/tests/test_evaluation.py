"""在 Rust arena 边界控制排名，核对评测报告与完整批次提交。"""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

MORTAL_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.inference.tests.test_inference import make_checkpoint
from precompute_pipeline.common.policy import PolicySettings
from precompute_pipeline.evaluation.evaluate import EvaluationSettings, run_evaluation


class EvaluationTests(unittest.TestCase):
    def test_seed_batches_and_statistics_are_persisted(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            policy = PolicySettings(make_checkpoint(root), device='cpu')
            settings = EvaluationSettings(policy, policy, root / 'results', games=20, games_per_batch=8, torch_threads=1)
            history = root / 'results/previous/result.json'
            history.parent.mkdir(parents=True)
            history.write_text('keep historical result')
            calls = []
            responses = iter(([2, 3, 2, 1], [1, 2, 3, 2], [1, 1, 1, 1]))

            class Arena:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def py_vs_py(self, **kwargs):
                    calls.append((kwargs['seed_start'], kwargs['seed_count'], self.kwargs['log_dir']))
                    return next(responses)

            first_path = run_evaluation(settings, arena_factory=Arena)
            report = json.loads(first_path.read_text())
            self.assertEqual(calls, [((10000, 8192), 2, None), ((10002, 8192), 2, None), ((10004, 8192), 1, None)])
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['rankings'], [4, 6, 6, 4])
            self.assertEqual(report['games_completed'], 20)
            self.assertEqual(report['average_rank'], 2.5)
            self.assertEqual(report['average_rank_points'], 4.5)
            self.assertFalse(report['challenger']['enable_amp'])
            self.assertEqual(history.read_text(), 'keep historical result')
            responses = iter(([2, 3, 2, 1], [1, 2, 3, 2], [1, 1, 1, 1]))
            second_path = run_evaluation(settings, arena_factory=Arena)
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(json.loads(first_path.read_text()), report)

    def test_failed_batch_leaves_previous_complete_batch_report(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            policy = PolicySettings(make_checkpoint(root), device='cpu')
            settings = EvaluationSettings(policy, policy, root / 'results', games=8, games_per_batch=4, torch_threads=1)
            class Arena:
                count = 0
                def __init__(self, **kwargs):
                    pass
                def py_vs_py(self, **kwargs):
                    Arena.count += 1
                    if Arena.count == 2:
                        raise RuntimeError('arena failed')
                    return [1, 1, 1, 1]
            with self.assertRaisesRegex(RuntimeError, 'arena failed'):
                run_evaluation(settings, arena_factory=Arena)
            report = json.loads(next((root / 'results').glob('*/result.json')).read_text())
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['games_completed'], 4)
            self.assertEqual(report['rankings'], [1, 1, 1, 1])
            with self.assertRaises(ValueError):
                EvaluationSettings(policy, policy, root, games=5)


if __name__ == '__main__':
    unittest.main()
