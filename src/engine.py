"""训练与评估核心：单轮训练、验证、早停、学习率调度。"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


@dataclass
class EpochStats:
    loss: float
    acc: float
    time: float


class AverageMeter:
    """滑动累计平均，用于统计 loss / accuracy。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def build_criterion(label_smoothing: float = 0.1) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def build_optimizer(model: nn.Module, name: str = "sgd", lr: float = 0.05,
                    weight_decay: float = 5e-4, momentum: float = 0.9):
    """SGD(momentum) 与 AdamW 两种优化器，便于对比。"""
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"未知优化器: {name}")


def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 3, name: str = "cosine"):
    """线性 warmup + 余弦退火，是当前 CNN 训练最常用的稳定组合。"""
    if name == "none":
        return None
    warmup = max(1, min(warmup_epochs, epochs // 5))
    sched1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    sched2 = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=1e-5)
    return SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[warmup])


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    grad_clip: float = 5.0, ema=None) -> EpochStats:
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    t0 = time.time()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = targets.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update((logits.argmax(1) == targets).float().mean().item(), bs)
    return EpochStats(loss_meter.avg, acc_meter.avg * 100.0, time.time() - t0)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> EpochStats:
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = targets.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update((logits.argmax(1) == targets).float().mean().item(), bs)
    return EpochStats(loss_meter.avg, acc_meter.avg * 100.0, 0.0)


@torch.no_grad()
def predict(model, loader, device):
    """返回 (logits, targets)，用于混淆矩阵与错误分析。"""
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        all_logits.append(logits.float().cpu())
        all_targets.append(targets)
    return torch.cat(all_logits), torch.cat(all_targets)


class ModelEMA:
    """权重指数滑动平均（EMA），通常能带来 0.2~0.5 个百分点的稳定收益。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if v.dtype.is_floating_point:
                ema_v.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                ema_v.copy_(v)


class EarlyStopping:
    """按验证准确率早停，并保存最优权重。"""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4, path: str = "best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best = -float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, metric: float, model: nn.Module, epoch: int) -> bool:
        """返回本 epoch 是否刷新了最优指标。"""
        if metric > self.best + self.min_delta:
            self.best = metric
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
