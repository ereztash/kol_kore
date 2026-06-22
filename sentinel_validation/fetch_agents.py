"""
Fetch the agent-trajectory dataset (substrate B): SWE-bench Verified evaluation
trajectories with pass/fail reward labels.

Source: togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories
(Hugging Face, ~150 MB, 500 trajectories, reward in {0,1}). Downloads the parquet
shards to /tmp/agent/.
"""
import os, time, urllib.request, concurrent.futures as cf

BASE = ("https://huggingface.co/datasets/togethercomputer/"
        "CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories/"
        "resolve/refs%2Fconvert%2Fparquet/trajectory/train/")
OUT = "/tmp/agent/"
N_SHARDS = 32


def grab(fn):
    p = OUT + fn
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return "cached"
    for a in range(4):
        try:
            open(p, "wb").write(urllib.request.urlopen(BASE + fn, timeout=120).read())
            return "ok"
        except Exception:
            if a == 3:
                return "FAIL"
            time.sleep(2 * (a + 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    files = [f"{i:04d}.parquet" for i in range(N_SHARDS)]
    t = time.time()
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(grab, files))
    print(f"{res.count('ok')} new, {res.count('cached')} cached, {res.count('FAIL')} fail "
          f"in {time.time()-t:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
