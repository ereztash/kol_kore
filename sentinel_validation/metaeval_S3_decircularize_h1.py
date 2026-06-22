"""
STRENGTHEN S3 — de-circularize H1 on a REAL human-operator substrate.

The earlier H1 tests were partly circular (J derived from the same signal as the
label). Here load is EXTERNALLY manipulated and independent of the outcome:
the Worker-HP mental-arithmetic dataset (Zenodo 10688787) -- 46 subjects, each
performing 3 timed arithmetic tests at a counterbalanced difficulty (E=easy,
N=normal), with a performance SCORE per test.

  load     = difficulty level (E=1, N=2)        -- externally set, outcome-independent
  capacity = the subject's own ability          -- estimated leave-one-out (no leakage)
  failure  = poor performance (bottom-third score)
  J        = load / capacity

Non-circular questions:
  (1) does external load reduce performance?              (load is real)
  (2) do LOW-capacity operators degrade MORE under load?  (the load/capacity coupling)
  (3) does P_fail(J=load/capacity) beat load-alone and capacity-alone?  (the ratio matters)
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))
XLSX = "/tmp/demo.xlsx"

def load_long():
    df = pd.read_excel(XLSX, sheet_name='Subjects Experiment 2 ', header=1)
    rows = []
    for _, r in df.iterrows():
        sid = r['Subject ID']
        if not isinstance(sid, str):
            continue
        for i in range(1, 4):
            score = r.get(f'Test {i}'); diff = r.get(f'Test {i}.1')
            if pd.notna(score) and isinstance(diff, str):
                rows.append(dict(subject=sid, score=float(score),
                                 load=2.0 if diff.strip().upper() == 'N' else 1.0,
                                 level=diff.strip().upper()))
    return pd.DataFrame(rows)

def main():
    print("STRENGTHEN S3: de-circularized load/capacity test on real human data\n")
    d = load_long()
    print(f"observations: {len(d)}  subjects: {d.subject.nunique()}  "
          f"(Easy={int((d['level']=='E').sum())}, Normal={int((d['level']=='N').sum())})\n")

    # (1) external load effect on performance (paired within subject)
    piv = d.pivot_table(index='subject', columns='level', values='score', aggfunc='mean').dropna()
    t, p = stats.wilcoxon(piv['E'], piv['N'])
    print("(1) external load reduces performance:")
    print(f"    mean score  Easy={piv['E'].mean():.2f}  Normal={piv['N'].mean():.2f}  "
          f"Δ={piv['N'].mean()-piv['E'].mean():+.2f}  (Wilcoxon p={p:.3f})")

    # capacity = leave-one-out subject mean (no leakage), normalized to global mean
    gmean = d.score.mean()
    cap = {}
    for s, g in d.groupby('subject'):
        cap[s] = g.score.mean()
    d['cap_loo'] = [ (d[(d.subject==r.subject)].score.sum()-r.score) /
                     max(len(d[d.subject==r.subject])-1,1) / gmean
                     for r in d.itertuples() ]

    # (2) load x capacity coupling: do low-capacity subjects drop MORE Easy->Normal?
    capser = pd.Series(cap)
    med = capser.median()
    drops = {}
    for grp, subs in (('low_capacity', capser[capser <= med].index),
                      ('high_capacity', capser[capser > med].index)):
        sub = piv.loc[piv.index.intersection(subs)]
        drops[grp] = float((sub['E'] - sub['N']).mean())   # performance drop under load
    print("\n(2) load×capacity coupling (performance drop Easy→Normal):")
    print(f"    low-capacity operators drop  : {drops['low_capacity']:+.2f}")
    print(f"    high-capacity operators drop : {drops['high_capacity']:+.2f}")
    print(f"    -> {'low-capacity degrade MORE under load (supports load/capacity)' if drops['low_capacity']>drops['high_capacity'] else 'no coupling'}")

    # (3) failure = bottom-third score; does J=load/capacity beat load & capacity alone?
    thr = np.percentile(d.score, 33)
    d['fail'] = (d.score <= thr).astype(int)
    d['J'] = d.load / d.cap_loo
    groups = d.subject.values
    def loso_auc(cols):
        X = d[cols].values; y = d['fail'].values
        logo = LeaveOneGroupOut(); pred = np.zeros(len(y))
        for tr, te in logo.split(X, y, groups):
            if y[tr].sum() in (0, len(tr)):       # need both classes in train
                pred[te] = y[tr].mean(); continue
            clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, pred)
    auc_load = loso_auc(['load']); auc_cap = loso_auc(['cap_loo']); auc_J = loso_auc(['J'])
    auc_both = loso_auc(['load', 'cap_loo'])
    print("\n(3) failure prediction (leave-one-subject-out AUC):")
    print(f"    load alone        : {auc_load:.3f}")
    print(f"    capacity alone    : {auc_cap:.3f}")
    print(f"    J = load/capacity : {auc_J:.3f}")
    print(f"    load + capacity   : {auc_both:.3f}")
    verdict = ("J beats both parts -> the RATIO carries non-circular signal"
               if auc_J >= max(auc_load, auc_cap) else
               "J does NOT beat its parts -> ratio adds nothing here")
    print(f"    -> {verdict}")

    json.dump(dict(n=len(d), load_effect=dict(easy=float(piv['E'].mean()),
              normal=float(piv['N'].mean()), p=float(p)),
              coupling=drops,
              auc=dict(load=auc_load, capacity=auc_cap, J=auc_J, both=auc_both)),
              open(os.path.join(OUT, "results_S3.json"), "w"), indent=2)
    print(f"\nresults -> {OUT}/results_S3.json")

if __name__ == "__main__":
    main()
