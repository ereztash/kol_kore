"""
META-TEST A — ablation + baselines (C-MAPSS).

Question the headline result must survive: is Sentinel's G(t) actually better than
standard, off-the-shelf detectors, and does its specific innovation -- the
persistence gate (1 - c) -- add anything? We beat a "raw redline" earlier, but that
is a strawman. Here every detector is calibrated to the SAME per-engine healthy
false-alarm rate on a TRAIN split and evaluated (lead time before failure) on a
disjoint TEST split, averaged over many random splits.

Detectors (all on the same Mahalanobis health index H(t)):
  raw        : instantaneous H            (strawman)
  ews_var    : rolling variance of H      (Scheffer/Dakos critical slowing down)
  ews_ac1    : rolling lag-1 autocorr     (Scheffer/Dakos)
  cusum      : CUSUM changepoint on H     (classic SPC baseline)
  G_nopersist: cumulative KL, NO gate     (ablation: remove (1-c))
  G_persist  : cumulative KL with (1-c)   (the proposal's G)
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
np.random.seed(0)

from validate_sentinel_cmapss import load, BASELINE
OUT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
#  Detectors: H(t) -> signal s(t)
# --------------------------------------------------------------------------- #
def det_raw(H):
    return H.copy()

def _rolling(H, w, fn):
    return np.array([fn(H[max(0, i - w):i + 1]) for i in range(len(H))])

def det_ews_var(H, w=15):
    return _rolling(H, w, np.var)

def det_ews_ac1(H, w=15):
    def ac1(x):
        if len(x) < 3 or np.std(x[:-1]) < 1e-9 or np.std(x[1:]) < 1e-9:
            return 0.0
        r = np.corrcoef(x[:-1], x[1:])[0, 1]
        return float(r) if np.isfinite(r) else 0.0
    return _rolling(H, w, ac1)

def det_cusum(H, slack=0.5):
    mu, sd = H[:BASELINE].mean(), H[:BASELINE].std() + 1e-9
    z = (H - mu) / sd
    S, out = 0.0, []
    for zi in z:
        S = max(0.0, S + zi - slack); out.append(S)
    return np.array(out)

def _G(H, gate, base_end=BASELINE, W=10, nb=12):
    base = H[:base_end]
    bins = np.linspace(base.min(), np.percentile(H, 99) + 1e-6, nb)
    pb, _ = np.histogram(base, bins=bins); pb = pb + 1e-6; pb /= pb.sum()
    G = np.zeros(len(H)); prev = 0.0; acc = 0.0
    for t in range(len(H)):
        w = H[max(0, t - W):t + 1]
        pw, _ = np.histogram(w, bins=bins); pw = pw + 1e-6; pw /= pw.sum()
        kl = float(np.sum(pw * np.log(pw / pb)))
        c = (1.0 if kl < prev else 0.0) if gate else 0.0
        acc += kl * (1.0 - c); prev = kl; G[t] = acc
    return G

def det_G_nopersist(H):
    return _G(H, gate=False)

def det_G_persist(H):
    return _G(H, gate=True)

DETECTORS = {
    'raw': det_raw, 'ews_var': det_ews_var, 'ews_ac1': det_ews_ac1,
    'cusum': det_cusum, 'G_nopersist': det_G_nopersist, 'G_persist': det_G_persist,
}

# --------------------------------------------------------------------------- #
#  Held-out, matched-FAR lead-time protocol
# --------------------------------------------------------------------------- #
def evaluate(engines, alpha=0.01, n_splits=30):
    ids = list(engines.keys())
    # precompute each detector's signal once per engine
    sig = {name: {u: fn(engines[u]['H']) for u in ids} for name, fn in DETECTORS.items()}
    out = {name: dict(leads=[], far=[]) for name in DETECTORS}
    rng = np.random.default_rng(0)
    for _ in range(n_splits):
        perm = rng.permutation(ids); cut = len(ids) // 2
        train, test = perm[:cut], perm[cut:]
        for name in DETECTORS:
            # calibrate theta on TRAIN healthy windows to per-engine FAR = alpha
            hmax = [sig[name][u][:BASELINE].max() for u in train]
            theta = float(np.quantile(hmax, 1 - alpha))
            leads, fa = [], 0
            for u in test:
                s = sig[name][u]; life = engines[u]['life']
                if (s[:BASELINE] > theta).any():
                    fa += 1
                idx = np.where(s[BASELINE:] > theta)[0]
                if len(idx):
                    leads.append(life - (BASELINE + idx[0]))
            out[name]['leads'].append(np.median(leads) if leads else 0.0)
            out[name]['far'].append(fa / len(test))
    res = {}
    for name in DETECTORS:
        res[name] = dict(median_lead=float(np.mean(out[name]['leads'])),
                         lead_sd=float(np.std(out[name]['leads'])),
                         test_far=float(np.mean(out[name]['far'])))
    return res

def main():
    print("META-A: ablation + baselines on C-MAPSS (held-out, matched-FAR lead time)\n")
    engines = load()
    table = {}
    for alpha in (0.005, 0.01, 0.05):
        res = evaluate(engines, alpha=alpha)
        table[alpha] = res
        print(f"--- target FAR = {alpha*100:.1f}%  (median held-out lead before failure, cycles) ---")
        for name, r in sorted(res.items(), key=lambda kv: -kv[1]['median_lead']):
            print(f"   {name:12s}  lead = {r['median_lead']:6.1f} ± {r['lead_sd']:4.1f}   (test FAR {r['test_far']*100:4.1f}%)")
        print()

    # verdict
    a = table[0.01]
    gp, gn = a['G_persist']['median_lead'], a['G_nopersist']['median_lead']
    best_base = max(a['cusum']['median_lead'], a['ews_var']['median_lead'], a['ews_ac1']['median_lead'])
    print("VERDICT @1% FAR:")
    print(f"  persistence gate (1-c):  G_persist {gp:.1f} vs G_nopersist {gn:.1f}  "
          f"-> {'helps' if gp>gn else 'no help / hurts'} ({gp-gn:+.1f} cycles)")
    print(f"  vs best standard baseline ({best_base:.1f}):  "
          f"G_persist {'beats' if gp>best_base else 'does NOT beat'} it ({gp-best_base:+.1f} cycles)")

    json.dump(table, open(os.path.join(OUT, "results_meta_A.json"), "w"), indent=2)
    print(f"\nresults -> {OUT}/results_meta_A.json")

if __name__ == "__main__":
    main()
