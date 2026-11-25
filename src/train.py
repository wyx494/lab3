"""训练入口。

示例:
    python -m src.train --exp cnn_baseline
    python -m src.train --exp resnet18_cifar --epochs 20 --batch-size 256
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from . import config as config_mod
from .data import build_dataloaders
from .engine import (EarlyStopping, ModelEMA, build_criterion, build_optimizer,
                     build_scheduler, evaluate, train_one_epoch)
from .models import build_model
from .utils import count_parameters, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 图像分类训练脚本")
    p.add_argument("--exp", default="cnn_baseline", help="预设实验名")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--aug", default=None, choices=["none", "basic", "strong"])
    p.add_argument("--optimizer", default=None, choices=["sgd", "adamw"])
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-ema", action="store_true")
    return p.parse_args()


def run(cfg) -> dict:
    set_seed(cfg.seed)
    device = get_device()
    os.makedirs(cfg.run_dir, exist_ok=True)
    print(f"[{cfg.exp}] 设备={device} 模型={cfg.model} 增强={cfg.aug} epochs={cfg.epochs}")

    bundles = build_dataloaders(
        data_root=cfg.data_root, input_size=cfg.input_size, aug=cfg.aug,
        pretrained_norm=cfg.pretrained, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, val_ratio=cfg.val_ratio, seed=cfg.seed,
    )
    print(f"  训练/验证/测试样本数: {bundles.num_train}/{bundles.num_val}/{bundles.num_test}")

    model = build_model(cfg.model, num_classes=10, pretrained=cfg.pretrained).to(device)
    n_params = count_parameters(model)
    print(f"  可训练参数量: {n_params/1e6:.2f} M")

    criterion = build_criterion(cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs, cfg.scheduler)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema else None
    stopper = EarlyStopping(patience=cfg.patience, path=os.path.join(cfg.run_dir, "best.pt"))

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "time")}
    # EMA 在训练早期尚未收敛，前若干个 epoch 用原始权重做验证更可靠
    ema_start_epoch = max(3, cfg.epochs // 5)
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        stats = train_one_epoch(model, bundles.train_loader, criterion, optimizer,
                                device, scaler, cfg.grad_clip, ema)
        if scheduler is not None:
            scheduler.step()

        use_ema = ema is not None and epoch >= ema_start_epoch
        eval_model = ema.module if use_ema else model
        val_stats = evaluate(eval_model, bundles.val_loader, criterion, device)
        improved = stopper.step(val_stats.acc, eval_model, epoch)

        history["train_loss"].append(stats.loss)
        history["train_acc"].append(stats.acc)
        history["val_loss"].append(val_stats.loss)
        history["val_acc"].append(val_stats.acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["time"].append(stats.time)
        print(f"  epoch {epoch:3d}/{cfg.epochs} | train loss {stats.loss:.4f} acc {stats.acc:5.2f}% "
              f"| val loss {val_stats.loss:.4f} acc {val_stats.acc:5.2f}% "
              f"| {stats.time:.1f}s {'*' if improved else ''}", flush=True)
        if stopper.should_stop:
            print(f"  早停触发：验证准确率连续 {cfg.patience} 轮未提升")
            break

    train_time = time.time() - t_start
    best_path = os.path.join(cfg.run_dir, "best.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_stats = evaluate(model, bundles.test_loader, criterion, device)
    print(f"  [测试集] loss {test_stats.loss:.4f} acc {test_stats.acc:.2f}% "
          f"| 最优验证 acc {stopper.best:.2f}% @ epoch {stopper.best_epoch} "
          f"| 总耗时 {train_time/60:.1f} min")

    metrics = {
        "exp": cfg.exp,
        "model": cfg.model,
        "pretrained": cfg.pretrained,
        "aug": cfg.aug,
        "epochs_run": len(history["train_loss"]),
        "params_m": round(n_params / 1e6, 3),
        "best_val_acc": round(stopper.best, 3),
        "best_epoch": stopper.best_epoch,
        "test_acc": round(test_stats.acc, 3),
        "test_loss": round(test_stats.loss, 4),
        "train_time_min": round(train_time / 60, 2),
        "device": str(device),
    }
    save_json(cfg.to_dict(), os.path.join(cfg.run_dir, "config.json"))
    save_json(history, os.path.join(cfg.run_dir, "history.json"))
    save_json(metrics, os.path.join(cfg.run_dir, "metrics.json"))
    return metrics


def main() -> None:
    args = parse_args()
    overrides = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, model=args.model,
        aug=args.aug, optimizer=args.optimizer, input_size=args.input_size,
        data_root=args.data_root, out_dir=args.out_dir, num_workers=args.workers,
        seed=args.seed,
    )
    cfg = config_mod.get_config(args.exp, **overrides)
    if args.no_ema:
        cfg.ema = False
    run(cfg)


if __name__ == "__main__":
    main()
