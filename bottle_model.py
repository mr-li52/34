"""
模型构建模块（PyTorch 版）。

这里使用 torchvision 的 MobileNetV2 作为特征提取骨干，并在其输出特征图上叠加：
- 通道注意力模块（Squeeze-and-Excitation，见 attention.SEBlock）；
- 自定义的两层 MLP 分类头（含 BatchNormalization + Dropout)。
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from models.attention import SEBlock


class BottleNet(nn.Module):
    """
    基于 MobileNetV2 + SE 注意力 + 自定义 MLP 头的瓶子分类网络。
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # 使用 ImageNet 预训练权重的 MobileNetV2 作为特征提取器
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        backbone = mobilenet_v2(weights=weights)

        # 去掉原始分类头，只保留特征提取部分
        self.features = backbone.features  # 输出通道一般为 1280
        last_channels = backbone.last_channel

        # 通道注意力模块（SE）
        self.se = SEBlock(channels=last_channels, ratio=16)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 自定义两层 MLP 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.se(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

