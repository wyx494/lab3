"""Grad-CAM 可解释性分析：定位模型决策时关注的图像区域。

实现参考 Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017。

用法:
    python -m src.gradcam --run results/resnet18_cifar
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import build_dataloaders
from .evaluate import denormalize
from .models import build_model, get_target_layer
from .utils import CIFAR10_CLASSES_ZH, get_device, load_json, savefig, set_seed


class GradCAM:
    """对目标层的前向激活与反向梯度加权求和，得到类别激活热力图。"""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.acts = out.detach()

    def _save_gradient(self, module, grad_inp, grad_out):
        self.grads = grad_out[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None):
        """返回 (归一化热力图 HxW, 预测类别, 置信度)。"""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        prob = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
        target = pred if class_idx is None else int(class_idx)
        logits[0, target].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)           # 通道重要性
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # 加权求和 + ReLU
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred, float(prob[0, pred].item())


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把热力图以半透明方式叠加到原图上。"""
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def pick_samples(dataset, cam_engine, device, num_samples: int, only_wrong: bool,
                 max_try_per_class: int = 40):
    """按类别轮转挑选样本，保证热力图覆盖不同类别（默认每类一张）。"""
    targets = np.asarray(dataset.targets)
    picks = []
    # 类别均匀抽样（例如 6 张时取 10 类里等间隔的 6 类），避免样本集中在同一类
    classes = np.unique(targets)
    if num_samples < len(classes):
        classes = classes[np.linspace(0, len(classes) - 1, num_samples).astype(int)]
    for c in classes:
        if len(picks) >= num_samples:
            break
        idxs = np.nonzero(targets == c)[0][:max_try_per_class]
        for i in idxs:
            img, label = dataset[int(i)]
            x = img.unsqueeze(0).to(device)
            cam, pred, prob = cam_engine.generate(x)
            if only_wrong and pred == c:
                continue
            picks.append((img, cam, pred, c, prob))
            break
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="训练输出目录")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--only-wrong", action="store_true", help="只可视化误分类样本")
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

    cam_engine = GradCAM(model, get_target_layer(model))
    picks = pick_samples(bundles.test_loader.dataset, cam_engine, device,
                         args.num_samples, args.only_wrong)
    images = [denormalize(p[0], cfg["pretrained"]) for p in picks]
    cams = [p[1] for p in picks]
    titles = [f"真:{CIFAR10_CLASSES_ZH[p[3]]} 预:{CIFAR10_CLASSES_ZH[p[2]]} {p[4]*100:.0f}%"
              for p in picks]

    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.9, 4.2))
    for k in range(n):
        axes[0, k].imshow(images[k])
        axes[0, k].set_title(titles[k], fontsize=8)
        axes[0, k].axis("off")
        axes[1, k].imshow(overlay(images[k], cams[k]))
        axes[1, k].axis("off")
    axes[0, 0].set_ylabel("原图")
    axes[1, 0].set_ylabel("Grad-CAM")
    fig.suptitle("Grad-CAM：模型关注区域（红=高响应）", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.run, "gradcam.png")
    savefig(fig, out)
    print("已保存:", out, "| 样本类别:", [CIFAR10_CLASSES_ZH[p[3]] for p in picks])


if __name__ == "__main__":
    main()
