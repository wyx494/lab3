"""通用多线程分段下载器（应对单连接限速的网络环境）。

用法: python tools/fetch_file.py <url> <dest> [--threads 16] [--md5 <hex>]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.request

from fetch_cifar10 import UA, md5sum, ranged_download


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--md5", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.dest)), exist_ok=True)
    if os.path.exists(args.dest):
        if args.md5 and md5sum(args.dest) == args.md5:
            print("已存在且校验通过:", args.dest)
            return 0
        if not args.md5:
            print("文件已存在:", args.dest, os.path.getsize(args.dest))
            return 0
    ok = ranged_download(args.url, args.dest, threads=args.threads)
    if ok and args.md5:
        got = md5sum(args.dest)
        print("md5:", got)
        return 0 if got == args.md5 else 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
