"""
Fetch the NASA C-MAPSS turbofan degradation dataset (FD001) — the "silent drift"
substrate. 100 engines run to failure; gradual multivariate sensor degradation
with NO discrete error events. Open mirror of the NASA Prognostics Data Repository.
"""
import os, urllib.request

URL = ("https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/"
       "master/CMAPSSData/train_FD001.txt")
OUT = "/tmp/cmapss/"


def main():
    os.makedirs(OUT, exist_ok=True)
    p = OUT + "train_FD001.txt"
    if os.path.exists(p) and os.path.getsize(p) > 0:
        print("cached", p); return
    open(p, "wb").write(urllib.request.urlopen(URL, timeout=90).read())
    print(f"downloaded {os.path.getsize(p)} bytes -> {p}")


if __name__ == "__main__":
    main()
