"""离线数据准备：把 CIFAR-10 parquet 转成 ImageFolder 目录结构。

data/cifar10/train/<class>/*.png, data/cifar10/test/<class>/*.png

用法: python tools/parquet_to_imagefolder.py --train data/train.parquet --test data/test.parquet
"""
from __future__ import annotations

import argparse
import io
import os

import pyarrow.parquet as pq
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def convert(parquet_path: str, split: str, root: str) -> int:
    out_root = os.path.join(root, "cifar10", split)
    for c in CLASSES:
        os.makedirs(os.path.join(out_root, c), exist_ok=True)
    table = pq.read_table(parquet_path, columns=["img", "label"])
    imgs = table.column("img").to_pylist()
    labels = table.column("label").to_pylist()
    counters = {i: 0 for i in range(len(CLASSES))}
    for item, label in zip(imgs, labels):
        data = item["bytes"] if isinstance(item, dict) else item
        counters[label] += 1
        path = os.path.join(out_root, CLASSES[label], f"{counters[label]:05d}.png")
        Image.open(io.BytesIO(data)).save(path)
    return len(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.parquet")
    ap.add_argument("--test", default="data/test.parquet")
    ap.add_argument("--root", default="data")
    args = ap.parse_args()
    n_train = convert(args.train, "train", args.root)
    n_test = convert(args.test, "test", args.root)
    print(f"train={n_train}, test={n_test}")


if __name__ == "__main__":
    main()
