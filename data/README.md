# 数据目录说明

本目录用于存放 CIFAR-10 数据集，**不需要提交到代码仓库**。

训练脚本 `src/data.py` 会按以下顺序寻找数据：

1. `data/cifar10/{train,test}/<类别>/*.png`（ImageFolder 结构，离线环境推荐）；
2. 若不存在，则自动调用 `torchvision.datasets.CIFAR10(download=True)` 下载官方
   `cifar-10-python.tar.gz` 并解压到 `data/cifar-10-batches-py/`。

## 方式一：自动下载（联网环境）

直接运行训练命令即可，torchvision 会自动完成下载与校验：

```bash
python -m src.train --exp cnn_baseline
```

## 方式二：离线准备

把 CIFAR-10 的 png 版本压缩包（如 fast.ai 的 `cifar10.tgz`）放到本目录，然后执行：

```bash
python scripts/prepare_data.py --root data
```

脚本会解压出 `data/cifar10/train/*` 与 `data/cifar10/test/*` 共 60 000 张图像。
