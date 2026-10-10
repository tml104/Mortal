"""手动启用的有限真实数据验收：生产形状、恢复、协议和 Rust 对局。"""

from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

import numpy as np
import torch

MORTAL_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.codec import decode_packed_chunk, decode_chunk_numpy, encode_chunk
from precompute_pipeline.common.dataset import iter_chunk_records, CHUNK_FILENAME
from precompute_pipeline.common.optimized_checkpoint import load_checkpoint
from precompute_pipeline.common.policy import PolicySettings
from precompute_pipeline.train_optimize.settings import OptimizedTrainingSettings
from precompute_pipeline.train_optimize.train import run_training
from precompute_pipeline.inference.inference import run_inference
from precompute_pipeline.evaluation.evaluate import EvaluationSettings, run_evaluation


def make_real_subset(root, source_progress):
    """仅截取真实样本到独立数据集，原 chunks 只读且精度不变。"""
    state = {'schema_version': 1, 'generation_id': uuid.uuid4().hex,
             'status': 'complete', 'chunk_size': 2048, 'splits': {}}
    for split in ('train', 'val'):
        record = next(iter(iter_chunk_records(source_progress, split)))
        packed = decode_packed_chunk(record.path.read_bytes())
        directory = root / split / 'c'
        directory.mkdir(parents=True)
        for index, (start, end) in enumerate(((0, 2048), (2048, 4096), (4096, 4103))):
            batch = decode_chunk_numpy(packed, np.arange(start, end))
            (directory / CHUNK_FILENAME.format(chunk_id=index)).write_bytes(encode_chunk(batch))
        state['splits'][split] = {
            'chunk_count': 3, 'sample_count': 4103, 'tail_sample_count': 7,
            'ranges': [{'start': 0, 'end': 3, 'volume': 'c'}],
        }
    progress = root / 'progress.json'
    progress.write_text(json.dumps(state))
    return progress


@unittest.skipUnless(os.environ.get('MORTAL_RUN_OPTIMIZED_INTEGRATION') == '1', '手动启用有限真实 CUDA/对局验收')
class RealPipelineTests(unittest.TestCase):
    def test_production_shapes_resume_mjai_and_two_real_match_batches(self):
        if not torch.cuda.is_available():
            self.skipTest('需要 CUDA')
        source = Path('data/precomputed_chunks/progress.json')
        if not source.exists():
            self.skipTest('需要已完成真实 chunks')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = make_real_subset(root / 'chunks', source)
            settings = OptimizedTrainingSettings(
                progress, root / 'artifacts/training', num_epochs=3,
                save_every_batches=2, log_every_batches=2, validation_samples=2051,
                torch_threads=8,
            )
            partial = run_training(settings, max_batches=2)
            self.assertEqual(partial['epoch_samples_consumed'], 4096)
            self.assertEqual(partial['completed_epochs'], 0)
            result = run_training(settings)
            self.assertEqual(result['completed_epochs'], 3)
            self.assertEqual(result['samples_seen'], 12309)
            self.assertEqual(result['batches_completed'], 9)
            checkpoint = settings.output_dir / 'latest.pth'
            state = load_checkpoint(checkpoint)
            self.assertEqual(state['last_validation']['sample_count'], 4103)
            for name, value in state['mortal'].items():
                if name.endswith('num_batches_tracked'):
                    self.assertEqual(value.item(), 9)
            fixture = MORTAL_DIR / 'precompute_pipeline/inference/tests/fixtures/start_kyoku.jsonl'
            for device, amp in (('cpu', False), ('cuda:0', True)):
                output = io.StringIO()
                run_inference(PolicySettings(checkpoint, device=device, enable_amp=amp), 0,
                              io.StringIO(fixture.read_text()), output)
                self.assertTrue(output.getvalue())
                json.loads(output.getvalue().splitlines()[0])
            policy = PolicySettings(checkpoint, device='cuda:0', enable_amp=True)
            evaluation = EvaluationSettings(policy, policy, root / 'evaluations', games=8,
                                            games_per_batch=4, torch_threads=8, rayon_threads=4,
                                            save_logs=True)
            report_file = run_evaluation(evaluation)
            report = json.loads(report_file.read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['games_completed'], 8)
            self.assertEqual(len(list((report_file.parent / 'logs').glob('*.json.gz'))), 8)
            report_path = os.environ.get('MORTAL_OPTIMIZED_ACCEPTANCE_REPORT')
            if report_path:
                Path(report_path).write_text(json.dumps({
                    'model': state['model_spec'], 'batch_size': 2048, 'completed_epochs': 3,
                    'samples_seen': result['samples_seen'], 'batches_completed': 9,
                    'fixed_subset_samples': 2051, 'full_validation_samples': 4103,
                    'bn_updates': 9, 'cpu_mjai': 'passed', 'gpu_amp_mjai': 'passed',
                    'match': report, 'note': 'isolated dataset and model artifacts removed by TemporaryDirectory',
                }, indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
