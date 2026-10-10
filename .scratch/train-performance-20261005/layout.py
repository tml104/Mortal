"""临时等价二维表示：保持 H=1，验证 channels_last 对一维卷积链路的影响。"""
import torch
from torch import nn
from model import ChannelAttention


class PlaneAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.shared_mlp = original.shared_mlp

    def forward(self, value):
        weight = (self.shared_mlp(value.mean((-2, -1))) +
                  self.shared_mlp(value.amax((-2, -1)))).sigmoid()
        return value * weight[:, :, None, None]


class SpatialBrain(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.inner = original

    def forward(self, observation):
        value = observation.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        return self.inner(value)


def convert_brain(brain):
    """卷积核扩一个长度为 1 的维度；归一化及注意力沿相同逻辑元素归约。"""
    def convert(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv1d):
                replacement = nn.Conv2d(child.in_channels, child.out_channels,
                    (1, child.kernel_size[0]), stride=(1, child.stride[0]),
                    padding=(0, child.padding[0]), dilation=(1, child.dilation[0]),
                    groups=child.groups, bias=child.bias is not None,
                    device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    replacement.weight.copy_(child.weight.unsqueeze(2))
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
            elif isinstance(child, nn.BatchNorm1d):
                replacement = nn.BatchNorm2d(child.num_features, eps=child.eps,
                    momentum=child.momentum, affine=child.affine,
                    track_running_stats=child.track_running_stats,
                    device=child.weight.device, dtype=child.weight.dtype)
                replacement.load_state_dict(child.state_dict())
            elif isinstance(child, ChannelAttention):
                replacement = PlaneAttention(child)
            else:
                convert(child)
                continue
            replacement.train(child.training)
            setattr(parent, name, replacement)
    convert(brain)
    return SpatialBrain(brain).to(memory_format=torch.channels_last)


def build_optimizer(models, settings):
    """保持原权重衰减分组，补入临时替换的 Conv2d 参数。"""
    decay, no_decay = [], []
    for model in models:
        for module in model.modules():
            for name, parameter in module.named_parameters(recurse=False):
                target = decay if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and name == 'weight' else no_decay
                target.append(parameter)
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': settings.optimizer.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}],
        lr=settings.optimizer.learning_rate, betas=settings.optimizer.betas,
        eps=settings.optimizer.eps, weight_decay=0.0)
