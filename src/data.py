"""数据加载与预处理：CIFAR-10 训练/验证划分、数据增强、DataLoader 构建。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# CIFAR-10 训练集统计量（逐通道均值/标准差），用于归一化
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
# ImageNet 统计量：加载预训练权重微调时需要与预训练一致
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUG_CHOICES = ("none", "basic", "strong")


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_train: int
    num_val: int
    num_test: int


def build_transforms(
    input_size: int = 32,
    aug: str = "basic",
    pretrained_norm: bool = False,
):
    """构造训练/评估两套变换。

    aug:
        none   —— 仅 ToTensor + Normalize（用于增强消融实验）
        basic  —— 随机裁剪 + 水平翻转（经典 CIFAR 增强）
        strong —— 在 basic 上叠加颜色抖动、RandAugment、随机擦除
    """
    if aug not in AUG_CHOICES:
        raise ValueError(f"未知的增强策略: {aug}")
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained_norm else (CIFAR_MEAN, CIFAR_STD)

    if input_size == 32:
        resize = []
        crop = transforms.RandomCrop(32, padding=4)
    else:
        resize = [transforms.Resize(input_size)]
        crop = transforms.RandomResizedCrop(input_size, scale=(0.55, 1.0))

    # 测试/验证：只做确定性的缩放与归一化
    test_tf = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if aug == "none":
        train_ops = resize + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    else:
        train_ops = resize + [crop, transforms.RandomHorizontalFlip(p=0.5)]
        if aug == "strong":
            train_ops += [
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.RandAugment(num_ops=2, magnitude=9),
            ]
        train_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if aug == "strong":
            train_ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    train_tf = transforms.Compose(train_ops)
    return train_tf, test_tf


def get_datasets(data_root: str, train_tf, test_tf):
    """返回训练集（带增强）与测试集。

    优先读取本地 ImageFolder 目录 ``data/cifar10/{train,test}/<类别>/*.png``
    （离线环境用 ``scripts/prepare_data.py`` 预先解压好）；
    若不存在则回退到 torchvision 的标准 CIFAR-10 自动下载。
    """
    folder_root = os.path.join(data_root, "cifar10")
    if os.path.isdir(os.path.join(folder_root, "train")):
        train_full = datasets.ImageFolder(os.path.join(folder_root, "train"), transform=train_tf)
        test_set = datasets.ImageFolder(os.path.join(folder_root, "test"), transform=test_tf)
    else:
        train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    return train_full, test_set


def get_train_view(data_root: str, transform):
    """获取不带数据增强的训练集视图（供验证集使用同一批索引）。"""
    folder_root = os.path.join(data_root, "cifar10")
    if os.path.isdir(os.path.join(folder_root, "train")):
        return datasets.ImageFolder(os.path.join(folder_root, "train"), transform=transform)
    return datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)


def split_train_val(
    dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """按类别分层划分训练/验证集（每类各取 val_ratio 作验证）。"""
    targets = torch.tensor(dataset.targets)
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    for c in range(10):
        idx = torch.nonzero(targets == c, as_tuple=False).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_dataloaders(
    data_root: str = "data",
    input_size: int = 32,
    aug: str = "basic",
    pretrained_norm: bool = False,
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DataBundle:
    train_tf, test_tf = build_transforms(input_size, aug, pretrained_norm)
    train_full, test_set = get_datasets(data_root, train_tf, test_tf)
    train_set, val_set = split_train_val(train_full, val_ratio, seed)
    # 验证集不应带数据增强：单独包一层只做确定性变换的子集视图
    val_view = get_train_view(data_root, test_tf)
    val_set = Subset(val_view, val_set.indices)

    pin = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **common)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_train=len(train_set),
        num_val=len(val_set),
        num_test=len(test_set),
    )
