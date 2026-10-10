"""保留 v4 参数键和特征顺序的原生二维主干。"""

import torch
from torch import nn

from model import Brain, ChannelAttention, DQN, AuxNet


class SpatialChannelAttention(nn.Module):
    """高度为 1 的空间归约保持原一维通道注意力含义。"""

    def __init__(self, shared_mlp):
        super().__init__()
        self.shared_mlp = shared_mlp

    def forward(self, x):
        weight = (
            self.shared_mlp(x.mean((-2, -1)))
            + self.shared_mlp(x.amax((-2, -1)))
        ).sigmoid()
        return x * weight[:, :, None, None]


def _replace_spatial_modules(parent):
    """原位替换空间算子，保留参数键与模块遍历次序供 AdamW 迁移。"""
    for name, child in list(parent.named_children()):
        if isinstance(child, nn.Conv1d):
            replacement = nn.Conv2d(
                child.in_channels, child.out_channels,
                kernel_size=(1, child.kernel_size[0]),
                stride=(1, child.stride[0]), padding=(0, child.padding[0]),
                dilation=(1, child.dilation[0]), groups=child.groups,
                bias=child.bias is not None,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight.unsqueeze(2))
                if child.bias is not None:
                    replacement.bias.copy_(child.bias)
        elif isinstance(child, nn.BatchNorm1d):
            replacement = nn.BatchNorm2d(
                child.num_features, eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats,
            )
            replacement.load_state_dict(child.state_dict())
        elif isinstance(child, ChannelAttention):
            replacement = SpatialChannelAttention(child.shared_mlp)
        else:
            _replace_spatial_modules(child)
            continue
        parent.add_module(name, replacement)


class BrainConv2d(Brain):
    """外部仍接收 (B,1012,34)，内部以 channels_last 二维布局执行。"""

    def __init__(self, *, conv_channels, num_blocks):
        super().__init__(version=4, conv_channels=conv_channels, num_blocks=num_blocks)
        _replace_spatial_modules(self.encoder)
        self.to(memory_format=torch.channels_last)

    def forward(self, obs):
        return super().forward(obs.unsqueeze(2).contiguous(memory_format=torch.channels_last))

    def train(self, mode=True):
        """二维 BN 同样遵守 Brain 的冻结约定。"""
        nn.Module.train(self, mode)
        if self._freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def reset_running_stats(self):
        """重置二维 BN 的统计而不重新初始化已学习参数。"""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()


def build_models(model_settings, device='cpu'):
    """训练与迁移共享相同参数注册次序。"""
    brain = BrainConv2d(
        conv_channels=model_settings.conv_channels, num_blocks=model_settings.num_blocks,
    ).to(device)
    brain.freeze_bn(model_settings.freeze_bn)
    return brain, DQN(version=4).to(device), AuxNet((4,)).to(device)


def build_optimizer(models, settings):
    """Conv2d/Linear weight 衰减，其余参数不衰减，沿用旧分组顺序。"""
    decay, no_decay = [], []
    for model in models:
        for module in model.modules():
            for name, parameter in module.named_parameters(recurse=False):
                target = decay if isinstance(module, (nn.Linear, nn.Conv2d)) and name == 'weight' else no_decay
                target.append(parameter)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': settings.weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=settings.learning_rate, betas=settings.betas, eps=settings.eps, weight_decay=0.0,
    )
