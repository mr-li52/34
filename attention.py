"""
注意力模块实现（PyTorch 版）。

实现一个简单的 Squeeze-and-Excitation (SE) 通道注意力模块，
对 MobileNetV2 输出的特征图进行通道维度的加权，从而突出对分类更有用的通道。
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        reduced_channels = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


