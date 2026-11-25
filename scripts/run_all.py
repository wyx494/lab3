"""一键运行全部实验、评估与可解释性分析。

用法:
    python -m scripts.run_all              # 跑全部实验
    python -m scripts.run_all --skip-existing   # 跳过已有 metrics.json 的实验（断点续跑）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPS = [
    "cnn_noaug",
    "cnn_baseline",
    "cnn_strongaug",
    "cnn_adamw",
    "se_cnn",
    "resnet18_cifar",
    "resnet18_pretrained",
    "mobilenet_pretrained",
]


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    ret = subprocess.call(cmd, cwd=ROOT)
    if ret != 0:
        raise SystemExit(f"命令失败({ret}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true", help="跳过已有结果的实验")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--only", default=None, help="只跑指定实验，逗号分隔，如 cnn_adamw,se_cnn")
    ap.add_argument("--no-extra", action="store_true", help="不跑 Grad-CAM 与汇总")
    args = ap.parse_args()

    exps = EXPS if not args.only else [e.strip() for e in args.only.split(",") if e.strip()]
    py = sys.executable
    for exp in exps:
        metrics = os.path.join(ROOT, "results", exp, "metrics.json")
        if args.skip_existing and os.path.exists(metrics):
            print(f"[skip] {exp} 已有结果", flush=True)
        else:
            run([py, "-m", "src.train", "--exp", exp, "--data-root", args.data_root])
        run([py, "-m", "src.evaluate", "--run", f"results/{exp}", "--data-root", args.data_root])
    if args.no_extra:
        return
    for exp in ("cnn_baseline", "resnet18_cifar", "resnet18_pretrained"):
        run([py, "-m", "src.gradcam", "--run", f"results/{exp}", "--data-root", args.data_root])
    run([py, "-m", "scripts.make_summary"])


if __name__ == "__main__":
    main()
