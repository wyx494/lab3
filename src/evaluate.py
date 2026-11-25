"""测试集评估：总体/每类准确率、混淆矩阵、误分类样本导出。

示例:
    python -m src.evaluate --run results/cnn_baseline
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .data import build_dataloaders
from .engine import predict
from .models import build_model
from .utils import (CIFAR10_CLASSES_ZH, classification_report, get_device,
                    load_json, plot_confusion_matrix, plot_misclassified,
                    save_json, set_seed)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616])


def denormalize(tensor: torch.Tensor, pretrained: bool) -> np.ndarray:
    """反归一化到 [0,1]，用于可视化。"""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained else (CIFAR_MEAN, CIFAR_STD)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * std + mean, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="训练输出目录，如 results/cnn_baseline")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--num-misclassified", type=int, default=12)
    args = ap.parse_args()

    cfg = load_json(os.path.join(args.run, "config.json"))
    set_seed(cfg["seed"])
    device = get_device()

    bundles = build_dataloaders(
        data_root=args.data_root, input_size=cfg["input_size"], aug="none",
        pretrained_norm=cfg["pretrained"], batch_size=256, num_workers=cfg["num_workers"],
        val_ratio=cfg["val_ratio"], seed=cfg["seed"],
    )
    model = build_model(cfg["model"], num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pt"),
                                     map_location=device, weights_only=True))
    model.to(device)

    logits, targets = predict(model, bundles.test_loader, device)
    probs = torch.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)

    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(targets.numpy(), preds.numpy()):
        cm[t, p] += 1

    report = classification_report(cm, CIFAR10_CLASSES_ZH)
    overall = float(np.trace(cm) / cm.sum() * 100)
    print(f"[{cfg['exp']}] 测试集总体准确率: {overall:.2f}%")

    # 找最容易混淆的类别对（非对角线计数最大的组合）
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(i, j, int(off[i, j])) for i in range(10) for j in range(10) if i != j]
    pairs.sort(key=lambda x: -x[2])
    top_pairs = [
        {"true": CIFAR10_CLASSES_ZH[i], "pred": CIFAR10_CLASSES_ZH[j], "count": c}
        for i, j, c in pairs[:5]
    ]
    print("  最易混淆类别对:", top_pairs)

    plot_confusion_matrix(cm, os.path.join(args.run, "confusion_matrix.png"), CIFAR10_CLASSES_ZH)

    # 误分类样本：按真实类别轮转，每类取 2 个最高置信度错误，保证覆盖面
    wrong = torch.nonzero(preds != targets, as_tuple=False).flatten()
    selected: list[int] = []
    for c in range(10):
        idx = wrong[targets[wrong] == c]
        if len(idx) == 0:
            continue
        idx = idx[conf[idx].argsort(descending=True)][:2]
        selected.extend(idx.tolist())
    order = torch.tensor(selected)[: args.num_misclassified]
    imgs = [denormalize(bundles.test_loader.dataset[i][0], cfg["pretrained"]) for i in order]
    plot_misclassified(
        np.stack(imgs), targets[order].numpy(), preds[order].numpy(), conf[order].numpy(),
        os.path.join(args.run, "misclassified.png"), CIFAR10_CLASSES_ZH,
    )

    save_json(
        {
            "exp": cfg["exp"],
            "test_acc": round(overall, 3),
            "per_class": report,
            "confusion_matrix": cm.tolist(),
            "top_confusions": top_pairs,
        },
        os.path.join(args.run, "eval_metrics.json"),
    )
    print(f"  结果已保存到 {args.run}/")


if __name__ == "__main__":
    main()
