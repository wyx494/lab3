"""实验配置：统一的 ExperimentConfig 与命名预设（presets）。

每个预设对应报告中一组对比实验，可用 `python -m src.train --exp <name>` 直接复现。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentConfig:
    exp: str = "cnn_baseline"          # 实验名（同时作为输出目录名）
    model: str = "simplecnn"           # simplecnn / se_cnn / resnet18_cifar / resnet18 / mobilenet_v3_small
    pretrained: bool = False           # 是否加载 ImageNet 预训练权重
    data_root: str = "data"
    out_dir: str = "results"
    input_size: int = 32               # 预训练模型需要插值放大到 96/224
    aug: str = "basic"                 # none / basic / strong
    epochs: int = 40
    batch_size: int = 128
    lr: float = 0.05
    weight_decay: float = 5e-4
    optimizer: str = "sgd"             # sgd / adamw
    scheduler: str = "cosine"          # cosine / none
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    dropout: float = 0.5
    ema: bool = True
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    patience: int = 12
    val_ratio: float = 0.1
    num_workers: int = 4
    seed: int = 42
    amp: bool = True

    @property
    def run_dir(self) -> str:
        return f"{self.out_dir}/{self.exp}"

    def to_dict(self) -> dict:
        return asdict(self)


# 预设：覆盖基础要求（自定义 CNN、增强、正则化）+ 加分项（注意力、架构对比、预训练微调、增强/优化器消融）
PRESETS: dict[str, dict] = {
    "cnn_baseline": dict(
        model="simplecnn", aug="basic", epochs=40, lr=0.05, optimizer="sgd",
    ),
    "cnn_noaug": dict(
        model="simplecnn", aug="none", epochs=40, lr=0.05, optimizer="sgd",
    ),
    "cnn_strongaug": dict(
        model="simplecnn", aug="strong", epochs=40, lr=0.05, optimizer="sgd",
    ),
    "cnn_adamw": dict(
        model="simplecnn", aug="basic", epochs=40, lr=1e-3, optimizer="adamw",
        weight_decay=1e-4,
    ),
    "se_cnn": dict(
        model="se_cnn", aug="strong", epochs=40, lr=0.05, optimizer="sgd",
    ),
    "resnet18_cifar": dict(
        model="resnet18_cifar", aug="strong", epochs=60, lr=0.1, optimizer="sgd",
        batch_size=128, patience=12,
    ),
    "resnet18_pretrained": dict(
        model="resnet18", pretrained=True, input_size=96, aug="strong", epochs=12,
        lr=0.01, optimizer="sgd", weight_decay=5e-4, batch_size=128, patience=5,
    ),
    "mobilenet_pretrained": dict(
        model="mobilenet_v3_small", pretrained=True, input_size=96, aug="strong",
        epochs=15, lr=0.01, optimizer="sgd", batch_size=128, patience=6,
    ),
}


def get_config(name: str, **overrides) -> ExperimentConfig:
    """按预设名构造配置，并允许命令行覆盖任意字段。"""
    if name not in PRESETS:
        raise KeyError(f"未知实验预设 {name}，可选: {', '.join(PRESETS)}")
    cfg = ExperimentConfig(exp=name, **PRESETS[name])
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
