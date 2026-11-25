"""CIFAR-10 图像分类实验代码包。

在导入 matplotlib 之前统一定位缓存目录，避免在受限环境下写用户目录失败。
"""
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
warnings.filterwarnings("ignore", category=UserWarning)
