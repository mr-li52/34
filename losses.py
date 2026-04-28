"""
损失函数相关工具（PyTorch 版）。
"""

import torch
import torch.nn as nn

from config.bottle_config import LABEL_SMOOTHING


def classification_loss(num_classes: int) -> nn.Module:
    """
    返回用于多类别分类的损失函数。

    这里使用带 label smoothing 的交叉熵损失：
    - 当 LABEL_SMOOTHING=0.0 时，退化为普通交叉熵；
    - 当 LABEL_SMOOTHING>0 时，可以一定程度缓解过拟合、提升泛化。
    """
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


