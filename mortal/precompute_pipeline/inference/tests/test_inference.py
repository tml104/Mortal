"""普通 mjai 公共入口的协议输出与无数据依赖回归。"""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

MORTAL_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.settings import ModelSettings
from precompute_pipeline.common.optimized_model import build_models
from precompute_pipeline.common.optimized_checkpoint import SCHEMA_VERSION, MODEL_FORMAT, model_metadata
from precompute_pipeline.common.policy import PolicySettings
from precompute_pipeline.inference.inference import run_inference


def make_checkpoint(root):
    """固定模型来源，不需要训练数据或额外导出步骤。"""
    settings = ModelSettings(16, 1, False)
    torch.manual_seed(81)
    brain, dqn, _ = build_models(settings)
    filename = root / 'new.pth'
    torch.save({
        'schema_version': SCHEMA_VERSION, 'model_format': MODEL_FORMAT,
        'model_spec': model_metadata(settings), 'mortal': brain.state_dict(), 'dqn': dqn.state_dict(),
    }, filename)
    return filename


class InferenceTests(unittest.TestCase):
    def test_real_mjai_prefix_produces_only_protocol_json(self):
        torch.set_num_threads(1)
        fixture = Path(__file__).parent / 'fixtures/start_kyoku.jsonl'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            output = io.StringIO()
            run_inference(PolicySettings(checkpoint, device='cpu'), 0, io.StringIO(fixture.read_text()), output)
            responses = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len(responses), 1)
            self.assertEqual(responses[0]['type'], 'dahai')
            self.assertEqual(responses[0]['actor'], 0)
            events = [json.loads(line) for line in fixture.read_text().splitlines()]
            self.assertIn(responses[0]['pai'], events[1]['tehais'][0] + [events[2]['pai']])
            with self.assertRaises(ValueError):
                run_inference(PolicySettings(checkpoint, device='cpu'), 4, io.StringIO(), output)
            torch.save({'schema_version': 1}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, 'Conv2d'):
                run_inference(PolicySettings(checkpoint, device='cpu'), 0, io.StringIO(), output)


if __name__ == '__main__':
    unittest.main()
