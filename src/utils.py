"""通用工具：随机种子、指标统计、绘图（含中文字体配置）。"""
from __future__ import annotations

import json
import os
import random

import matplotlib
import numpy as np
import torch

# 把 matplotlib 缓存放到项目内，避免在受限环境下写用户目录失败
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CIFAR10_CLASSES_ZH = [
    "飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车",
]
CIFAR10_CLASSES_EN = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_chinese_font() -> None:
    """优先使用 Windows 自带中文字体，避免图中出现方块。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "DengXian", "SimSun", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()


def savefig(fig, path: str, dpi: int = 200) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, path: str, title: str = "训练过程") -> None:
    """绘制 loss / accuracy 双曲线（训练集 vs 验证集）。"""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

    axes[0].plot(epochs, history["train_loss"], label="训练损失", lw=1.8)
    axes[0].plot(epochs, history["val_loss"], label="验证损失", lw=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("损失曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="训练准确率", lw=1.8)
    axes[1].plot(epochs, history["val_acc"], label="验证准确率", lw=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率 (%)")
    axes[1].set_title("准确率曲线")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    savefig(fig, path)


def plot_confusion_matrix(cm: np.ndarray, path: str, names: list[str]) -> None:
    """绘制带数值标注的混淆矩阵（行归一化颜色 + 原始计数）。"""
    n = len(names)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6.5,
                color="white" if norm[i, j] > 0.55 else "#333333",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="行归一化比例")
    ax.set_title("测试集混淆矩阵")
    fig.tight_layout()
    savefig(fig, path)


def plot_misclassified(
    images: np.ndarray,
    true_idx: np.ndarray,
    pred_idx: np.ndarray,
    conf: np.ndarray,
    path: str,
    names: list[str],
    cols: int = 4,
) -> None:
    """展示误分类样本：真实类别 / 预测类别 / 置信度。"""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.75))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        ax.set_title(
            f"真:{names[true_idx[i]]}  预:{names[pred_idx[i]]}\n置信度 {conf[i]*100:.0f}%",
            fontsize=7.0,
            color="#c0392b",
            linespacing=1.25,
        )
    fig.subplots_adjust(hspace=0.32, wspace=0.06, top=0.93, bottom=0.02)
    savefig(fig, path)


def classification_report(cm: np.ndarray, names: list[str]) -> dict:
    """由混淆矩阵计算每类 precision / recall / f1。"""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    pred_sum = cm.sum(axis=0).astype(float)
    prec = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        names[i]: {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "accuracy": round(float(rec[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(names))
    }
