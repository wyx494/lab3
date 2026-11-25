"""构建全部模型并做一次前向/反向，快速验证模型定义无误。

用法: python -m scripts.smoke_test
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model, get_target_layer  # noqa: E402

CASES = [
    ("simplecnn", 32, False),
    ("se_cnn", 32, False),
    ("resnet18_cifar", 32, False),
    ("resnet18", 96, False),
    ("mobilenet_v3_small", 96, False),
]


def main() -> None:
    for name, size, pretrained in CASES:
        model = build_model(name, num_classes=10, pretrained=pretrained)
        x = torch.randn(2, 3, size, size)
        logits = model(x)
        logits.sum().backward()
        n = sum(p.numel() for p in model.parameters())
        assert logits.shape == (2, 10), logits.shape
        assert get_target_layer(model) is not None
        print(f"{name:22s} input={size} params={n/1e6:6.2f}M  OK")


if __name__ == "__main__":
    main()
