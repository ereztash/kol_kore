"""
STRENGTHEN S2 — real-data evidence for the persistence / "uncorrected deviation"
mechanism (de-synthesises Meta-C), on the SWE-bench agent traces.

The proposal's specific claim is the (1 - c) gate: it is the UNCORRECTED deviation
that drives failure, not deviation per se. Meta-C imposed a habituation model; here
we test the claim directly on real traces: does the structure of uncorrected
deviation predict failure BEYOND the raw error/deviation count?

uncorrected run = longest consecutive stretch of error/retry steps with no
recovering progress (a clean, distinct tool_call) in between -- i.e. the agent is
stuck normalizing its own deviation.
"""
import os, json, glob, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")

from validate_sentinel_agents import to_events, DATA_GLOB
OUT = os.path.dirname(os.path.abspath(__file__))

def features(ev):
    n = len(ev)
    dev = [e in ('error', 'retry') for e in ev]
    total_dev = sum(dev)
    # longest run of consecutive deviations with no recovering 'tool_call'/'node' inside
    max_run = run = 0
    for e in ev:
        if e in ('error', 'retry'):
            run += 1; max_run = max(max_run, run)
        else:
            run = 0
    # fraction of deviations that are 'uncorrected' = not followed within 2 steps
    # by a clean distinct tool_call (a recovery)
    unc = 0
    for i, e in enumerate(ev):
        if e in ('error', 'retry'):
            recovered = any(ev[j] == 'tool_call' for j in range(i + 1, min(n, i + 3)))
            if not recovered:
                unc += 1
    return dict(n=n, total_dev=total_dev, dev_rate=total_dev / n,
                max_uncorrected_run=max_run, uncorrected_count=unc,
                uncorrected_frac=unc / max(total_dev, 1))

def main():
    print("STRENGTHEN S2: real persistence / uncorrected-deviation on agent traces\n")
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA_GLOB))], ignore_index=True)
    rows = []
    for _, r in df.iterrows():
        try:
            ev = to_events(json.loads(r['messages']))
        except Exception:
            continue
        if len(ev) < 12:
            continue
        f = features(ev); f['y'] = int(r['reward'] == 0)   # y=1 FAIL
        rows.append(f)
    R = pd.DataFrame(rows); y = R['y'].values
    print(f"trajectories: {len(R)}   fails: {int(y.sum())}\n")

    print("single-feature AUC (FAIL vs PASS):")
    for c in ['total_dev', 'dev_rate', 'max_uncorrected_run', 'uncorrected_count', 'uncorrected_frac']:
        a = roc_auc_score(y, R[c]); print(f"   {c:20s}: {max(a, 1-a):.3f}")

    # does uncorrected structure ADD over raw deviation count? (nested CV-AUC)
    def cvauc(cols):
        X = StandardScaler().fit_transform(R[cols].values)
        sk = StratifiedKFold(5, shuffle=True, random_state=0); a = []
        for tr, te in sk.split(X, y):
            clf = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X[tr], y[tr])
            a.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        return np.mean(a), np.std(a)
    base = cvauc(['total_dev'])
    plus = cvauc(['total_dev', 'max_uncorrected_run', 'uncorrected_count'])
    print(f"\nincremental value of the (1-c) structure (5-fold CV-AUC):")
    print(f"   raw deviation count only        : {base[0]:.3f} ± {base[1]:.3f}")
    print(f"   + uncorrected-deviation features: {plus[0]:.3f} ± {plus[1]:.3f}   (Δ {plus[0]-base[0]:+.3f})")

    # matched-deviation test: within strata of similar total_dev, does uncorrected separate?
    R['dev_bin'] = pd.qcut(R['total_dev'], 5, duplicates='drop')
    aucs = []
    for _, g in R.groupby('dev_bin'):
        if g['y'].nunique() == 2 and len(g) > 30:
            a = roc_auc_score(g['y'], g['max_uncorrected_run']); aucs.append(max(a, 1-a))
    print(f"\nMATCHED on total deviation (within-stratum AUC of uncorrected-run): "
          f"{np.mean(aucs):.3f}  -> {'adds signal' if np.mean(aucs)>0.55 else 'no independent signal'}")

    json.dump(dict(base_auc=base, plus_auc=plus,
                   single={c: float(max(roc_auc_score(y, R[c]), 1-roc_auc_score(y, R[c])))
                           for c in ['total_dev', 'max_uncorrected_run', 'uncorrected_count']},
                   matched_stratum_auc=float(np.mean(aucs))),
              open(os.path.join(OUT, "results_S2.json"), "w"), indent=2)
    print(f"\nresults -> {OUT}/results_S2.json")

if __name__ == "__main__":
    main()
