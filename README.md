# 当代人工智能作业三 —— CIFAR-10 图像分类

> 代码仓库：<https://github.com/wyx494/lab3>

本项目基于 PyTorch 独立完成 CIFAR-10 十类图像分类任务，包含**自定义 CNN 设计、训练与评估、
可视化分析、错误分析、消融实验与 Grad-CAM 可解释性分析**。所有实验均可用一条命令复现。

## 1. 目录结构

```
code_lab3/
├── README.md                 # 本文件：环境、运行说明、结果汇总
├── requirements.txt          # 依赖清单
├── src/
│   ├── config.py             # 实验配置与 8 组预设（presets）
│   ├── data.py               # 数据加载、分层划分、数据增强
│   ├── models.py             # SimpleCNN / SE-CNN / ResNet-18 / MobileNetV3
│   ├── engine.py             # 训练循环、验证、EMA、早停、学习率调度
│   ├── train.py              # 训练入口（命令行）
│   ├── evaluate.py           # 测试集评估：混淆矩阵、每类准确率、错分样本
│   ├── gradcam.py            # Grad-CAM 可解释性可视化
│   └── utils.py              # 随机种子、指标、绘图工具
├── scripts/
│   ├── prepare_data.py       # 数据解压/准备
│   ├── make_summary.py       # 汇总所有实验，生成对比图表
│   ├── smoke_test.py         # 模型结构快速自检（前向 + 反向）
│   └── run_all.py            # 一键跑完全部实验
├── tools/                    # 离线下载/格式转换辅助脚本（非训练必需）
│   ├── fetch_cifar10.py      # 多线程分段下载 CIFAR-10
│   ├── fetch_file.py         # 通用多线程下载
│   └── parquet_to_imagefolder.py  # parquet → ImageFolder 图片目录
├── report/
│   ├── make_report.py        # 由实验结果自动生成实验报告（docx）
│   └── 实验报告.docx / .pdf   # 最终报告（≤5 页）
├── results/                  # 每次实验的输出（日志、图表、指标）
└── data/                     # 数据集目录（首次运行自动准备）
```

## 2. 环境准备

- Python 3.10+（开发环境为 Python 3.12）
- PyTorch 2.5.1 + torchvision 0.20.1（CUDA 12.4 版本，GPU 训练；CPU 亦可运行但较慢）
- 其余依赖见 `requirements.txt`

```bash
# 建议使用虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> 若需 GPU 版 PyTorch 且官方源较慢，可使用镜像：
> `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --find-links https://mirrors.aliyun.com/pytorch-wheels/cu124/`
> 仅 CPU 环境可直接 `pip install torch==2.5.1 torchvision==0.20.1`。

## 3. 数据准备

CIFAR-10 共 60,000 张 32×32 彩色图像（50,000 训练 / 10,000 测试，10 类）。

```bash
python scripts/prepare_data.py     # 若 data/cifar10.tgz 存在则解压到 data/cifar10/
```

若本地无数据，首次运行训练脚本时 `torchvision` 会自动下载官方 `cifar-10-python.tar.gz`。
训练集按类别**分层划分**为 45,000 训练 / 5,000 验证，测试集 10,000 张只用于最终评估。

## 4. 运行方式

```bash
# 单组实验（8 个预设见 src/config.py）
python -m src.train --exp cnn_baseline

# 常用命令行覆盖
python -m src.train --exp resnet18_cifar --epochs 30 --batch-size 256

# 测试集评估：混淆矩阵、每类准确率、误分类样本
python -m src.evaluate --run results/cnn_baseline

# Grad-CAM 可视化
python -m src.gradcam --run results/resnet18_cifar

# 汇总全部实验并生成对比图表
python -m scripts.make_summary

# 一键跑完全部实验（耗时较长）
python -m scripts.run_all
```

每组实验在 `results/<exp>/` 下产出：

| 文件 | 说明 |
| --- | --- |
| `best.pt` | 验证集最优权重 |
| `config.json` / `history.json` / `metrics.json` | 配置、逐轮曲线数据、最终指标 |
| `confusion_matrix.png` | 混淆矩阵（`evaluate` 生成） |
| `misclassified.png` | 高置信度误分类样本 |
| `gradcam.png` | Grad-CAM 热力图 |

## 5. 实验设置

| 预设 | 模型 | 增强 | 优化器 | Epochs | 输入 |
| --- | --- | --- | --- | --- | --- |
| `cnn_baseline` | SimpleCNN | basic | SGD | 40 | 32 |
| `cnn_noaug` | SimpleCNN | none | SGD | 40 | 32 |
| `cnn_strongaug` | SimpleCNN | strong | SGD | 40 | 32 |
| `cnn_adamw` | SimpleCNN | basic | AdamW | 40 | 32 |
| `se_cnn` | SE-CNN（通道注意力） | strong | SGD | 40 | 32 |
| `resnet18_cifar` | CIFAR 版 ResNet-18 | strong | SGD | 60 | 32 |
| `resnet18_pretrained` | ResNet-18 ImageNet 预训练 | strong | SGD | 12 | 96 |
| `mobilenet_pretrained` | MobileNetV3-Small 预训练 | strong | SGD | 15 | 96 |

统一训练技巧：Label Smoothing(0.1)、线性 warmup + 余弦退火、梯度裁剪、权重 EMA、
按验证准确率早停并保存最优权重、混合精度训练（AMP）。

## 6. 结果汇总

在 RTX 4070 Laptop（8 GB）上完成全部 8 组实验，测试集准确率如下（验证集用于选模型，测试集只评估一次）：

| 实验预设 | 模型 | 增强 | 参数量(M) | Epoch | 验证 acc | 测试 acc | 耗时(min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `resnet18_cifar` | ResNet-18（从零） | strong | 11.17 | 60 | 95.16% | **94.90%** | 20.4 |
| `resnet18_pretrained` | ResNet-18（ImageNet 预训练） | strong | 11.18 | 12 | 95.00% | 94.26% | 10.1 |
| `cnn_strongaug` | SimpleCNN（强增强） | strong | 1.15 | 40 | 91.84% | 91.30% | 13.3 |
| `se_cnn` | SE-CNN（通道注意力） | strong | 1.16 | 40 | 91.88% | 91.20% | 13.8 |
| `cnn_baseline` | SimpleCNN（基础增强） | basic | 1.15 | 40 | 92.34% | 91.07% | 7.1 |
| `cnn_adamw` | SimpleCNN（AdamW） | basic | 1.15 | 40 | 91.60% | 90.60% | 9.9 |
| `mobilenet_pretrained` | MobileNetV3-Small（预训练） | strong | 1.53 | 15 | 91.20% | 90.47% | 8.6 |
| `cnn_noaug` | SimpleCNN（无增强） | none | 1.15 | 40 | 87.76% | 87.26% | 6.1 |

> 说明：8 组实验的完整日志、曲线数据与图表均在 `results/<exp>/` 下；
> 汇总图表见 `results/summary/`（`curves_models.png`、`ablation.png`、`tradeoff.png`）。

## 7. 分析结论

1. **数据增强是收益最大的单项改进**：无增强 87.26% → 基础增强（随机裁剪 + 水平翻转）91.07%，
   提升约 3.8 个百分点；继续加码到强增强仅再提升 0.2 个百分点，而单轮耗时接近翻倍，边际收益递减。
2. **深度残差结构有效**：从零训练的 ResNet-18（11.2 M 参数）达到 94.90%，
   比 1.15 M 的浅层 CNN 高 3.6 个百分点。
3. **迁移学习胜在效率**：ImageNet 预训练的 ResNet-18 仅微调 12 个 epoch 就达到 94.26%，
   训练时间约为从零训练的一半；但若给足轮数，原生 32×32 分辨率从零训练仍可略胜一筹。
4. **优化器与注意力机制收益有限**：SGD+Momentum 优于 AdamW（91.07% vs 90.60%）；
   SE 通道注意力在通道数仅 64~256 的浅层网络上与基线持平（91.20% vs 91.30%），
   参数只增加 0.01 M，说明该任务上瓶颈不在通道选择。
5. **误差集中在语义相近类别**：最佳模型的混淆矩阵中，猫↔狗（70 + 55 例）、
   汽车↔卡车（22 + 21 例）、鸟→鹿（19 例）是主要错误来源；Grad-CAM 显示模型对主体轮廓关注良好，
   错误样本常聚焦于背景或局部纹理。

## 8. 复现说明

- 所有随机种子固定为 42（`src/utils.py: set_seed`），数据划分与训练过程可复现；
- 单卡 RTX 4070 Laptop（8 GB）即可完成全部 8 组实验，总训练耗时约 1.5 小时；
- 训练日志逐轮打印 loss/accuracy，并写入 `history.json`，便于绘制曲线；
- 由于体积原因，`results/` 中未附带模型权重（`best.pt`）；若需重新评估某组实验，
  先运行 `python -m src.train --exp <name>` 生成权重，再运行 `python -m src.evaluate --run results/<name>`。
