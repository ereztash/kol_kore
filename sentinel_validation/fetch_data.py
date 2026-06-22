"""
Fetch the PhysioNet/CinC Challenge 2015 training set (open access) and build the
label map used by validate_sentinel_physionet.py.

Downloads ~400 MB of waveform records to /tmp/p2015/training/ and writes
/tmp/labels.json = {record: {type, result, n, fs}}.
"""
import os, json, time, urllib.request, concurrent.futures as cf

BASE = "https://physionet.org/files/challenge-2015/1.0.0/training/"
OUT  = "/tmp/p2015/training/"
os.makedirs(OUT, exist_ok=True)


def fetch(url, timeout=90, retries=4):
    for a in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 * (a + 1))


def grab(rec):
    for ext in (".hea", ".mat"):
        p = OUT + rec + ext
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        open(p, "wb").write(fetch(BASE + rec + ext))
    return rec


def main():
    records = fetch(BASE + "RECORDS").decode().split()
    print(f"downloading {len(records)} records ...")
    t = time.time()
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, _ in enumerate(ex.map(grab, records), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(records)}  ({time.time()-t:.0f}s)")

    labels = {}
    for rec in records:
        head = open(OUT + rec + ".hea").read().splitlines()
        comments = [x[1:] for x in head if x.startswith("#")]
        first = head[0].split()
        labels[rec] = dict(type=comments[0] if comments else "?",
                           result=comments[1] if len(comments) > 1 else "?",
                           fs=int(first[2]), n=int(first[3]))
    json.dump(labels, open("/tmp/labels.json", "w"))
    n_true = sum(v["result"] == "True alarm" for v in labels.values())
    print(f"done: {len(labels)} records, {n_true} true / {len(labels)-n_true} false")
    print("wrote /tmp/labels.json")


if __name__ == "__main__":
    main()
