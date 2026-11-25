"""模型定义：自定义 CNN（含 SE 注意力变体）、CIFAR 版 ResNet-18、ImageNet 预训练模型。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力：全局池化 → 两层 MLP → 通道重标定。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x.mean(dim=(2, 3)).view(b, c)).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """两层 3x3 卷积 + BN + ReLU，池化后通道翻倍。"""

    def __init__(self, in_ch: int, out_ch: int, use_se: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.se(self.conv(x)))


class SimpleCNN(nn.Module):
    """面向 32x32 输入的自定义 CNN：3 个卷积块 + 全局平均池化 + Dropout 分类头。"""

    def __init__(self, num_classes: int = 10, widths: tuple[int, int, int] = (64, 128, 256),
                 dropout: float = 0.5, use_se: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, widths[0], use_se),
            ConvBlock(widths[0], widths[1], use_se),
            ConvBlock(widths[1], widths[2], use_se),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[2], num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @property
    def target_layer(self) -> nn.Module:
        """Grad-CAM 的默认目标层：最后一个卷积块。"""
        return self.features[-1]


class BasicBlock(nn.Module):
    """ResNet 基本残差块（CIFAR 版，步长只在第一个卷积上生效）。"""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18CIFAR(nn.Module):
    """适配 32x32 输入的 ResNet-18：3x3 主干、无 maxpool、无 7x7 大核。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        layers += [BasicBlock(out_ch, out_ch, 1) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

    @property
    def target_layer(self) -> nn.Module:
        return self.layer4


def build_model(name: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """模型工厂，便于对比实验统一调用。"""
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes, dropout=0.5)
    if name == "se_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=0.5, use_se=True)
    if name == "resnet18_cifar":
        return ResNet18CIFAR(num_classes=num_classes, dropout=0.1)
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"未知模型: {name}")


def get_target_layer(model: nn.Module) -> nn.Module:
    """返回用于 Grad-CAM 的最后一个卷积层（按模型类型查表，避免污染模型结构）。"""
    if isinstance(model, SimpleCNN):
        return model.features[-1]
    if isinstance(model, ResNet18CIFAR):
        return model.layer4
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("无法自动确定 Grad-CAM 目标层，请在配置中显式指定")
