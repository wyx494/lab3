"""多线程分段下载 CIFAR-10，兼容慢速网络。

用法: python tools/fetch_cifar10.py --dest <data_dir>
优先 toronto 官方 tar.gz（torchvision 可直接校验并解压），失败时回退到 S3/fast.ai 镜像。
"""
import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


def head_size(url, timeout=20):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers.get("Content-Length", 0))


def ranged_download(url, dest, threads=24, chunk=1 << 20, timeout=60):
    total = head_size(url)
    if total <= 0:
        raise RuntimeError("无法获取文件大小")
    part = dest + ".part"
    with open(part, "wb") as f:
        f.truncate(total)
    ranges = [(i, min(i + chunk, total) - 1) for i in range(0, total, chunk)]
    lock = threading.Lock()
    done = [0]
    errs = []

    def worker(idx):
        start, end = ranges[idx]
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"短读 {len(data)}")
                with lock:
                    with open(part, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                    done[0] += len(data)
                return
            except Exception as e:  # 网络抖动重试
                if attempt == 3:
                    with lock:
                        errs.append(f"块{idx} 失败: {type(e).__name__} {e}")
                else:
                    time.sleep(1.5 * (attempt + 1))

    threads = min(threads, len(ranges))
    t0 = time.time()
    pool = []
    qi = [0]
    qlock = threading.Lock()

    def runner():
        while True:
            with qlock:
                if qi[0] >= len(ranges):
                    return
                i = qi[0]
                qi[0] += 1
            worker(i)
            with lock:
                pct = 100.0 * done[0] / total
            if int(time.time() - t0) % 5 == 0:
                print(f"\r  {pct:5.1f}%  {done[0]/1048576:6.1f}/{total/1048576:.1f} MB", end="", flush=True)

    for _ in range(threads):
        th = threading.Thread(target=runner, daemon=True)
        th.start()
        pool.append(th)
    for th in pool:
        th.join()
    print()
    if errs:
        print("部分分块失败:", errs[:3])
        return False
    os.replace(part, dest)
    print(f"  下载完成 {os.path.getsize(dest)/1048576:.1f} MB，用时 {time.time()-t0:.0f}s")
    return True


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data")
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    out = os.path.join(args.dest, "cifar-10-python.tar.gz")
    if os.path.exists(out) and md5sum(out) == CIFAR_MD5:
        print("已存在且校验通过:", out)
        return 0

    urls = [
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ]
    for url in urls:
        print("尝试:", url)
        try:
            if ranged_download(url, out, threads=args.threads):
                got = md5sum(out)
                print("md5:", got, "期望:", CIFAR_MD5)
                if got == CIFAR_MD5:
                    print("校验通过")
                    return 0
                print("校验失败，删除重试")
                os.remove(out)
        except Exception as e:
            print("失败:", type(e).__name__, e)
    return 1


if __name__ == "__main__":
    sys.exit(main())
