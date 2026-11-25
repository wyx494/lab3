"""汇总所有实验指标，并生成报告所需的对比图与表格数据。

用法:
    python -m scripts.make_summary --results results --out results/summary
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# 先固定 matplotlib 缓存目录，再导入 pyplot（受限环境下用户目录不可写）
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_json, save_json, savefig, setup_chinese_font  # noqa: E402

setup_chinese_font()

# 报告中展示顺序与中文名
DISPLAY = {
    "cnn_baseline": "自定义CNN",
    "cnn_noaug": "CNN(无增强)",
    "cnn_strongaug": "CNN(强增强)",
    "cnn_adamw": "CNN(AdamW)",
    "se_cnn": "SE-CNN",
    "resnet18_cifar": "ResNet-18(从零)",
    "resnet18_pretrained": "ResNet-18(预训练)",
    "mobilenet_pretrained": "MobileNetV3-S(预训练)",
}


def collect(results_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "metrics.json"))):
        m = load_json(path)
        if m.get("exp") in DISPLAY:
            m["label"] = DISPLAY[m["exp"]]
            rows.append(m)
    return rows


def plot_val_curves(results_dir: str, rows: list[dict], out: str) -> None:
    """各主要模型的验证准确率曲线对比。"""
    picks = ["cnn_baseline", "se_cnn", "resnet18_cifar", "resnet18_pretrained", "mobilenet_pretrained"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for r in rows:
        if r["exp"] not in picks:
            continue
        hist_path = os.path.join(results_dir, r["exp"], "history.json")
        if not os.path.exists(hist_path):
            continue
        h = load_json(hist_path)
        ax.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], lw=1.7, label=DISPLAY[r["exp"]])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("验证准确率 (%)")
    ax.set_title("不同模型的验证准确率曲线")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    savefig(fig, out)


def plot_ablation(rows: list[dict], out: str) -> None:
    """增强策略 / 优化器消融柱状图（同一骨干 SimpleCNN）。"""
    index = {r["exp"]: r for r in rows}

    def acc(exp):
        return index[exp]["test_acc"] if exp in index else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    # 左：数据增强消融
    names = ["无增强", "基础增强", "强增强"]
    vals = [acc("cnn_noaug"), acc("cnn_baseline"), acc("cnn_strongaug")]
    bars = axes[0].bar(names, vals, color=["#95a5a6", "#3498db", "#e67e22"])
    axes[0].set_title("数据增强消融 (SimpleCNN)")
    axes[0].set_ylabel("测试准确率 (%)")
    axes[0].set_ylim(min(v for v in vals if not np.isnan(v)) - 1.2, 94.5)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)

    # 右：优化器消融
    names2 = ["SGD+Momentum", "AdamW"]
    vals2 = [acc("cnn_baseline"), acc("cnn_adamw")]
    bars2 = axes[1].bar(names2, vals2, color=["#3498db", "#9b59b6"])
    axes[1].set_title("优化器消融 (SimpleCNN)")
    axes[1].set_ylim(min(v for v in vals2 if not np.isnan(v)) - 1.2, 94.5)
    for b, v in zip(bars2, vals2):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, out)


def plot_tradeoff(rows: list[dict], out: str) -> None:
    """参数量-准确率权衡图。"""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for r in rows:
        ax.scatter(r["params_m"], r["test_acc"], s=42, label=r["label"])
        ax.annotate(r["label"], (r["params_m"], r["test_acc"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6.8)
    ax.set_xlabel("参数量 (M)")
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("参数量与准确率的权衡")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    savefig(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = collect(args.results)
    if not rows:
        raise SystemExit("未找到任何 metrics.json，请先运行训练")
    save_json(rows, os.path.join(args.out, "results_table.json"))
    plot_val_curves(args.results, rows, os.path.join(args.out, "curves_models.png"))
    plot_ablation(rows, os.path.join(args.out, "ablation.png"))
    plot_tradeoff(rows, os.path.join(args.out, "tradeoff.png"))

    print(f"{'实验':24s}{'模型':22s}{'增强':7s}{'参数(M)':>8s}{'测试acc':>9s}{'耗时(min)':>10s}")
    for r in sorted(rows, key=lambda x: -x["test_acc"]):
        print(f"{r['exp']:24s}{r['label']:22s}{r['aug']:7s}{r['params_m']:8.2f}"
              f"{r['test_acc']:9.2f}{r['train_time_min']:10.1f}")


if __name__ == "__main__":
    main()
