"""准备 CIFAR-10 数据。

两种使用方式（二选一）：
1) 已下载 cifar10.tgz（fast.ai 镜像）放到 data/ 下，本脚本自动解压为
   data/cifar10/{train,test}/<类别>/*.png，训练时由 ImageFolder 读取；
2) 直接运行训练脚本，torchvision 会自动下载官方 cifar-10-python.tar.gz。

用法: python scripts/prepare_data.py [--root data]
"""
from __future__ import annotations

import argparse
import os
import tarfile

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def extract(tgz: str, root: str) -> str:
    """解压 fast.ai 版 cifar10.tgz，返回数据集根目录。"""
    target = os.path.join(root, "cifar10")
    if os.path.isdir(os.path.join(target, "train")):
        print("已解压:", target)
        return target
    print(f"解压 {tgz} -> {target} ...")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(root)
    # 压缩包内顶层目录可能叫 cifar10/，统一到 root/cifar10
    for cand in (os.path.join(root, "cifar10"), os.path.join(root, "cifar-10")):
        if os.path.isdir(os.path.join(cand, "train")):
            return cand
    raise RuntimeError("解压后未找到 train/ 目录，请检查压缩包")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    args = ap.parse_args()
    tgz = os.path.join(args.root, "cifar10.tgz")
    if not os.path.exists(tgz):
        print("未找到 data/cifar10.tgz，将由 torchvision 在首次训练时自动下载官方数据")
        return
    root = extract(tgz, args.root)
    for split in ("train", "test"):
        n = sum(len(files) for _, _, files in os.walk(os.path.join(root, split)))
        print(f"  {split}: {n} 张图像")
    print("类别:", ", ".join(CLASSES))


if __name__ == "__main__":
    main()
