"""新推理和对局评测共享的模型设备配置与 MortalEngine 构造。"""

from dataclasses import dataclass
from pathlib import Path

import torch

from engine import MortalEngine
from precompute_pipeline.common.optimized_checkpoint import load_policy
from precompute_pipeline.common.runtime import boolean, path_value


@dataclass(frozen=True)
class PolicySettings:
    checkpoint: Path
    device: str = 'cpu'
    enable_amp: bool = False
    enable_quick_eval: bool = True
    enable_rule_based_agari_guard: bool = True
    name: str = 'MortalConv2d'


def policy_settings(section, *, default_device='cpu', default_amp=False):
    return PolicySettings(
        checkpoint=path_value(section, 'checkpoint'), device=section.get('device', default_device),
        enable_amp=boolean(section, 'enable_amp', default_amp),
        enable_quick_eval=boolean(section, 'enable_quick_eval', True),
        enable_rule_based_agari_guard=boolean(section, 'enable_rule_based_agari_guard', True),
        name=section.get('name', 'MortalConv2d'),
    )


def create_engine(settings):
    """CPU 强制 FP32；GPU 可选 FP16 AMP，均采用普通推理前向。"""
    device = torch.device(settings.device)
    if device.type not in {'cuda', 'cpu'}:
        raise ValueError('推理设备只支持 cpu 或 cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'配置设备不可用: {device}')
    brain, dqn, metadata = load_policy(settings.checkpoint, device=device)
    engine = MortalEngine(
        brain, dqn, is_oracle=False, version=4, device=device,
        enable_amp=settings.enable_amp and device.type == 'cuda',
        enable_quick_eval=settings.enable_quick_eval,
        enable_rule_based_agari_guard=settings.enable_rule_based_agari_guard,
        name=settings.name,
    )
    return engine, metadata
