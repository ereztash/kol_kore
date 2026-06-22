"""
STRENGTHEN S1 — real-data evidence for the "joint sees what marginals miss"
principle (de-synthesises Meta-B), on C-MAPSS.

Meta-B showed a joint monitor beats single dashboards -- but on data we built.
Here we ask the same question on REAL failures: do cases exist where the JOINT
detector (Mahalanobis over all sensors) fires while NO single-sensor redline has
fired yet, at a MATCHED total false-alarm rate, on held-out engines?

Fair calibration: the OR-of-all-single-sensors is given the SAME total FAR budget
as the joint monitor (each single sensor calibrated at alpha / n_sensors), so the
comparison is honest, not a strawman.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

DATA = "/tmp/cmapss/train_FD001.txt"
OUT = os.path.dirname(os.path.abspath(__file__))
BASELINE = 20

def load_raw():
    d = np.loadtxt(DATA)
    units = d[:, 0].astype(int); sensors = d[:, 5:]
    keep = sensors.std(0) > 1e-6
    S = sensors[:, keep]
    healthy = np.vstack([S[units == u][:BASELINE] for u in np.unique(units)])
    mu, sd = healthy.mean(0), healthy.std(0) + 1e-9
    Z = np.abs((S - mu) / sd)                      # per-sensor deviation from healthy
    Zh = (healthy - mu) / sd
    cov = np.cov(Zh.T) + 1e-3 * np.eye(Zh.shape[1])
    P = np.linalg.pinv(cov)
    Zsign = (S - mu) / sd
    H = np.sqrt(np.einsum('ij,jk,ik->i', Zsign, P, Zsign))   # joint Mahalanobis
    engines = {}
    for u in np.unique(units):
        m = units == u
        engines[int(u)] = dict(life=int(d[m, 1].max()), H=H[m], Zabs=Z[m])
    return engines, Z.shape[1]

def first_after(sig, theta, t0=BASELINE):
    i = np.where(sig[t0:] > theta)[0]
    return (t0 + i[0]) if len(i) else None

def evaluate(engines, n_sensors, alpha=0.01, n_splits=30):
    ids = list(engines.keys())
    rng = np.random.default_rng(0)
    res = dict(joint_lead=[], or_lead=[], joint_beats_or=[], joint_before_any_single=[])
    for _ in range(n_splits):
        perm = rng.permutation(ids); cut = len(ids) // 2
        train, test = perm[:cut], perm[cut:]
        # joint threshold at FAR alpha; each single at alpha/n so OR-FAR ~ alpha
        th_joint = float(np.quantile([engines[u]['H'][:BASELINE].max() for u in train], 1 - alpha))
        a_single = alpha / n_sensors
        th_single = np.array([
            float(np.quantile([engines[u]['Zabs'][:BASELINE, s].max() for u in train], 1 - a_single))
            for s in range(n_sensors)])
        jl, ol, beat, jbefore = [], [], 0, 0
        for u in test:
            e = engines[u]; life = e['life']
            tj = first_after(e['H'], th_joint)
            # OR of singles: earliest cross of ANY single sensor
            singles = [first_after(e['Zabs'][:, s], th_single[s]) for s in range(n_sensors)]
            singles = [t for t in singles if t is not None]
            to = min(singles) if singles else None
            if tj is not None:
                jl.append(life - tj)
                if to is None or tj < to:
                    jbefore += 1                   # joint fired before ANY single
            ol.append((life - to) if to is not None else 0)
            if tj is not None and (to is None or tj <= to):
                beat += 1
        res['joint_lead'].append(np.median(jl) if jl else 0)
        res['or_lead'].append(np.median(ol) if ol else 0)
        res['joint_beats_or'].append(beat / len(test))
        res['joint_before_any_single'].append(jbefore / len(test))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}

def main():
    print("STRENGTHEN S1: real 'joint sees what marginals miss' on C-MAPSS\n")
    engines, n = load_raw()
    print(f"engines: {len(engines)}   sensors: {n}\n")
    for alpha in (0.005, 0.01, 0.05):
        r = evaluate(engines, n, alpha)
        print(f"--- matched total FAR = {alpha*100:.1f}% (held-out, 30 splits) ---")
        print(f"   joint median lead        : {r['joint_lead'][0]:6.1f} cycles")
        print(f"   OR-of-all-singles lead   : {r['or_lead'][0]:6.1f} cycles")
        print(f"   joint fires >= as early  : {r['joint_beats_or'][0]*100:4.0f}% of engines")
        print(f"   joint fires BEFORE any single (compound-like) : {r['joint_before_any_single'][0]*100:4.0f}% of engines\n")
    json.dump({str(a): evaluate(engines, n, a) for a in (0.005, 0.01, 0.05)},
              open(os.path.join(OUT, "results_S1.json"), "w"), indent=2)
    print(f"results -> {OUT}/results_S1.json")

if __name__ == "__main__":
    main()
