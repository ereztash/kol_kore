"""
MAXIMIZE (agent substrate) — find the highest DEFENSIBLE number, held-out.

Pre-registered protocol (fixed BEFORE looking at results):
  * Outer 5-fold stratified CV; everything (features, model) fit on train folds
    only, scored on the held-out fold. Report mean +/- std test AUC AND the
    train-test gap (to expose overfitting).
  * Models compared at fixed, modest settings (no test-based tuning):
      - logistic on the ORIGINAL feature set (reproduces the 0.73 baseline)
      - logistic on the RICH feature set
      - gradient boosting on the RICH feature set
  * Feature-group ablation (held-out): which groups carry signal.
  * The OPERATIONAL metric: online early-warning. Using only the first h-fraction
    of each trajectory (causal prefix), held-out AUC(h) -- how early can we flag a
    doomed run at all, and at what cost.

No metric is chosen after the fact; the honest result is reported, gains or not.
"""
import os, json, glob, warnings, re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
np.random.seed(0)

from validate_sentinel_agents import DATA_GLOB, ERR, ALPH
OUT = os.path.dirname(os.path.abspath(__file__))

TOOLS = ['think', 'str_replace_editor', 'execute_bash', 'finish', 'view', 'search', 'other']
def tool_bucket(name):
    if not name: return 'other'
    n = name.lower()
    for t in ['think', 'finish']:
        if t in n: return t
    if 'replace' in n or 'edit' in n or 'editor' in n: return 'str_replace_editor'
    if 'bash' in n or 'exec' in n or 'command' in n: return 'execute_bash'
    if 'view' in n or 'read' in n or 'open' in n or 'cat' in n: return 'view'
    if 'search' in n or 'grep' in n or 'find' in n: return 'search'
    return 'other'

def parse(msgs):
    """Return per-step records: (lex_symbol, tool_bucket, is_error)."""
    seq = [m for m in msgs if m.get('role') in ('assistant', 'tool')]
    out, prev = [], None
    for i, m in enumerate(seq):
        if m.get('role') != 'assistant': continue
        tc = m.get('tool_calls'); tool = tc[0].get('function', {}).get('name') if tc else None
        obs = seq[i+1].get('content') if i+1 < len(seq) and seq[i+1].get('role') == 'tool' else ''
        is_err = bool(ERR.search(str(obs)))
        if is_err: sym = 'error'
        elif tool in (None, 'think'): sym = 'node'
        elif tool == prev: sym = 'retry'
        else: sym = 'tool_call'
        out.append((sym, tool_bucket(tool), is_err)); prev = tool
    return out

def G_div(syms):
    idx = {s: i for i, s in enumerate(ALPH)}; x = np.array([idx[s] for s in syms])
    be = max(5, int(len(x) * 0.25))
    pb = np.bincount(x[:be], minlength=4) + 1e-6; pb /= pb.sum()
    G = np.zeros(len(x)); prev = acc = 0.0
    for t in range(len(x)):
        w = x[max(0, t-10):t+1]; pw = np.bincount(w, minlength=4)+1e-6; pw /= pw.sum()
        kl = float(np.sum(pw*np.log(pw/pb))); c = 1.0 if kl < prev else 0.0
        acc += kl*(1-c); prev = kl; G[t] = acc
    return G

def ent(counts):
    p = np.array(counts, float); s = p.sum()
    if s == 0: return 0.0
    p = p[p > 0]/s; return float(-(p*np.log(p)).sum())

def featurize(recs, prefix=1.0):
    n_full = len(recs); n = max(4, int(round(n_full*prefix)))
    r = recs[:n]
    syms = [x[0] for x in r]; tools = [x[1] for x in r]; errs = [x[2] for x in r]
    c = Counter(syms); tc = Counter(tools)
    G = G_div(syms)
    third = max(1, n//3)
    er = np.array([1.0 if e else 0.0 for e in errs])
    dev = np.array([1.0 if s in ('error', 'retry') else 0.0 for s in syms])
    # repetition / loops
    maxrun_tool = maxrun_err = run_t = run_e = 0; prevt = None
    for t, e in zip(tools, errs):
        run_t = run_t+1 if t == prevt else 1; maxrun_tool = max(maxrun_tool, run_t); prevt = t
        run_e = run_e+1 if e else 0; maxrun_err = max(maxrun_err, run_e)
    f = {}
    # group: budget
    f['g_n_steps'] = n; f['g_J_budget'] = n/100.0
    # group: lexicon
    for s in ALPH: f[f'lex_{s}'] = c[s]/n
    f['lex_entropy'] = ent([c[s] for s in ALPH])
    # group: tool mix
    for tb in TOOLS: f[f'tool_{tb}'] = tc[tb]/n
    f['tool_entropy'] = ent([tc[tb] for tb in TOOLS])
    # group: dynamics
    f['dyn_err_early'] = er[:third].mean(); f['dyn_err_late'] = er[-third:].mean()
    f['dyn_err_trend'] = float(np.polyfit(np.arange(n), er, 1)[0])
    f['dyn_dev_late'] = dev[-third:].mean()
    f['dyn_G_final'] = float(G[-1]); f['dyn_G_slope'] = float((G[-1]-G[n//2])/(n/2))
    # group: repetition/recovery
    f['rep_max_tool_run'] = maxrun_tool; f['rep_max_err_run'] = maxrun_err
    recov = sum(1 for i, s in enumerate(syms) if s in ('error', 'retry')
                and any(syms[j] == 'tool_call' for j in range(i+1, min(n, i+3))))
    f['rep_recovery_frac'] = recov/max(c['error']+c['retry'], 1)
    return f

GROUPS = {'budget': 'g_', 'lexicon': 'lex_', 'tool_mix': 'tool_', 'dynamics': 'dyn_', 'repetition': 'rep_'}
ORIG = ['g_n_steps', 'g_J_budget', 'lex_error', 'lex_retry', 'lex_node',
        'tool_entropy', 'dyn_G_final', 'dyn_G_slope', 'dyn_err_early', 'dyn_dev_late']

def cv(X, y, model='logit', folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    tr_a, te_a = [], []
    for tr, te in skf.split(X, y):
        if model == 'logit':
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
        else:
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                                 learning_rate=0.05, l2_regularization=1.0,
                                                 random_state=0)
        clf.fit(X[tr], y[tr])
        te_a.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        tr_a.append(roc_auc_score(y[tr], clf.predict_proba(X[tr])[:, 1]))
    return float(np.mean(te_a)), float(np.std(te_a)), float(np.mean(tr_a))

def main():
    print("MAXIMIZE agent substrate — held-out, pre-registered\n")
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA_GLOB))], ignore_index=True)
    recs_all, y = [], []
    for _, rr in df.iterrows():
        try: rec = parse(json.loads(rr['messages']))
        except Exception: continue
        if len(rec) < 12: continue
        recs_all.append(rec); y.append(int(rr['reward'] == 0))
    y = np.array(y)
    feats = [featurize(r, 1.0) for r in recs_all]
    F = pd.DataFrame(feats).fillna(0.0); cols = list(F.columns)
    print(f"trajectories: {len(F)}  fails: {int(y.sum())}  features: {len(cols)}\n")

    def sub(names): return F[names].values
    print("=== held-out 5-fold ROC-AUC (test | train-gap) ===")
    te, sd, tr = cv(sub(ORIG), y, 'logit')
    print(f"  baseline (orig feats, logistic) : {te:.3f} ± {sd:.3f}   (train {tr:.3f})")
    te2, sd2, tr2 = cv(sub(cols), y, 'logit')
    print(f"  rich feats, logistic            : {te2:.3f} ± {sd2:.3f}   (train {tr2:.3f})")
    te3, sd3, tr3 = cv(sub(cols), y, 'gbm')
    print(f"  rich feats, gradient boosting   : {te3:.3f} ± {sd3:.3f}   (train {tr3:.3f})")

    print("\n=== feature-group ablation (rich GBM, drop one group) ===")
    base = te3
    for gname, pref in GROUPS.items():
        keep = [c for c in cols if not c.startswith(pref)]
        t, _, _ = cv(sub(keep), y, 'gbm')
        print(f"  without {gname:11s}: {t:.3f}   (Δ {t-base:+.3f})")

    print("\n=== OPERATIONAL metric: online early-warning (causal prefix) ===")
    print("  using only first h-fraction of each trajectory, held-out GBM AUC:")
    ew = {}
    for h in (0.25, 0.5, 0.75, 1.0):
        Fh = pd.DataFrame([featurize(r, h) for r in recs_all]).fillna(0.0)[cols].values
        t, s, _ = cv(Fh, y, 'gbm'); ew[h] = t
        print(f"     h={h:.2f}:  AUC = {t:.3f} ± {s:.3f}")

    json.dump(dict(n=len(F), n_fail=int(y.sum()),
                   auc=dict(baseline=te, rich_logit=te2, rich_gbm=te3),
                   early_warning={str(k): v for k, v in ew.items()}),
              open(os.path.join(OUT, "results_maximize_agent.json"), "w"), indent=2)
    print(f"\nresults -> {OUT}/results_maximize_agent.json")
    print(f"\nHONEST DELTA: agent fail/success {te:.3f} -> {max(te2,te3):.3f} held-out "
          f"({max(te2,te3)-te:+.3f}). Early-warning at h=0.5: {ew[0.5]:.3f}.")

if __name__ == "__main__":
    main()
