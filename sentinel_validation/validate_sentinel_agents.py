"""
Sentinel — substrate B (AI-agent layer). Validation of the SAME failure model on
REAL LLM-agent execution traces with pass/fail labels.

This is the agent counterpart of validate_sentinel_physionet.py and tests the
document's milestone M4 substrate ("ניטור סוכני AI ... על הלקסיקון
{node, tool_call, retry, error}"). The proposal's whole premise (H3) is that ONE
model — the logistic J transition and the cumulative-divergence G — applies to
both a human operator and an agent layer. The ICU pilot exercised the operator
substrate; this exercises the agent substrate, on the SAME equations.

Dataset
-------
SWE-bench Verified evaluation trajectories (OpenHands agent), 500 runs labelled
reward in {0 (fail), 1 (pass)}; 29-100 steps each. Each assistant step is a tool
call; each tool turn is its observation. We project every step onto the document's
lexicon {node, tool_call, retry, error}.

Mechanisms (identical maths to the ICU pilot)
  J(t)  = budget utilisation = steps_used / budget   (the agent's load / capacity)
  G(t)  = Σ D_KL( window event-dist || baseline event-dist ) · (1 - c)
  raw monitor = cumulative explicit-error count crossing a threshold
"""
import os, re, json, glob, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(0)

DATA_GLOB = "/tmp/agent/*.parquet"
OUT  = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(OUT, "figs")
BUDGET = 100                       # OpenHands step cap = agent "capacity"
ALPH = ['node', 'tool_call', 'retry', 'error']

# explicit-error signatures in tool observations
ERR = re.compile(
    r"(traceback \(most recent call last\)|^\s*\w*error:|exception|no such file|"
    r"command not found|syntaxerror|importerror|nameerror|attributeerror|typeerror|"
    r"valueerror|keyerror|assertionerror|no replacement was performed|"
    r"did not appear verbatim|multiple occurrences|exit code [1-9]|fatal:|failed)",
    re.I | re.M)

# --------------------------------------------------------------------------- #
#  Trace -> event stream over {node, tool_call, retry, error}
# --------------------------------------------------------------------------- #
def to_events(msgs):
    seq = [m for m in msgs if m.get('role') in ('assistant', 'tool')]
    ev, prev = [], None
    for i, m in enumerate(seq):
        if m.get('role') != 'assistant':
            continue
        tc = m.get('tool_calls')
        tool = tc[0].get('function', {}).get('name') if tc else None
        obs = seq[i + 1].get('content') if i + 1 < len(seq) and seq[i + 1].get('role') == 'tool' else ''
        if ERR.search(str(obs)):
            sym = 'error'
        elif tool in (None, 'think'):
            sym = 'node'
        elif tool == prev:
            sym = 'retry'
        else:
            sym = 'tool_call'
        ev.append(sym); prev = tool
    return ev

# --------------------------------------------------------------------------- #
#  Sentinel constructs (same maths as the ICU pilot)
# --------------------------------------------------------------------------- #
def G_divergence(ev, base_frac=0.25, W=10):
    idx = {s: i for i, s in enumerate(ALPH)}
    x = np.array([idx[e] for e in ev])
    be = max(5, int(len(x) * base_frac))
    pb = np.bincount(x[:be], minlength=4) + 1e-6; pb = pb / pb.sum()
    G = np.zeros(len(x)); prev = 0.0; acc = 0.0
    for t in range(len(x)):
        w = x[max(0, t - W):t + 1]
        pw = np.bincount(w, minlength=4) + 1e-6; pw = pw / pw.sum()
        kl = float(np.sum(pw * np.log(pw / pb)))
        c = 1.0 if kl < prev else 0.0
        acc += kl * (1.0 - c); prev = kl; G[t] = acc
    return G

def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def featurize(ev):
    n = len(ev); G = G_divergence(ev)
    cnt = {s: ev.count(s) for s in ALPH}
    err = np.array([1.0 if e == 'error' else 0.0 for e in ev])
    dev = np.array([1.0 if e in ('error', 'retry') else 0.0 for e in ev])
    q = max(2, n // 4)
    feat = dict(
        n_steps=n, J_budget=n / BUDGET,
        err_rate=cnt['error'] / n, retry_rate=cnt['retry'] / n, node_rate=cnt['node'] / n,
        action_entropy=entropy([cnt[s] / n for s in ALPH]),
        G_final=float(G[-1]), G_slope=float((G[-1] - G[n // 2]) / (n / 2)),
        dev_early=float(dev[:q].mean()), dev_late=float(dev[-q:].mean()),
        dev_drift=float(dev[-q:].mean() - dev[:q].mean()),
    )
    return feat, G, err

def load():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(DATA_GLOB))],
                   ignore_index=True)
    rows, series, skip = [], [], 0
    for _, r in df.iterrows():
        try:
            ev = to_events(json.loads(r['messages']))
        except Exception:
            skip += 1; continue
        if len(ev) < 12:
            skip += 1; continue
        feat, G, err = featurize(ev)
        feat['y'] = int(r['reward'] == 0)          # y = 1  -> FAIL
        rows.append(feat)
        series.append(dict(y=feat['y'], G=G, err=err, n=len(ev)))
    return rows, series, skip

# --------------------------------------------------------------------------- #
FEATS = ['n_steps', 'J_budget', 'err_rate', 'retry_rate', 'node_rate',
         'action_entropy', 'G_final', 'G_slope', 'dev_early', 'dev_late', 'dev_drift']

def sanitize(X):
    X = np.array(X, float)
    for j in range(X.shape[1]):
        col = X[:, j]; bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.median(col[~bad]) if (~bad).any() else 0.0
    return X

def cv_auc(X, y, n=5):
    X = sanitize(X)
    skf = StratifiedKFold(n, shuffle=True, random_state=0); a = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight='balanced'))
        clf.fit(X[tr], y[tr]); a.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(a)), float(np.std(a))

def test1(rows):
    y = np.array([r['y'] for r in rows])
    X = sanitize([[r[f] for f in FEATS] for r in rows])
    m, s = cv_auc(X, y)
    singles = {f: float(max(roc_auc_score(y, X[:, i]), 1 - roc_auc_score(y, X[:, i])))
               for i, f in enumerate(FEATS)}
    return dict(auc=(m, s), n=len(y), n_fail=int(y.sum()),
                single=dict(sorted(singles.items(), key=lambda kv: -kv[1])))

def _logistic(J, k, J0):
    return 1.0 / (1.0 + np.exp(-k * (J - J0)))

def test2(rows):
    """H1 on the agent: P_fail(J), J = budget utilisation = steps / budget."""
    J = np.array([r['J_budget'] for r in rows]); y = np.array([r['y'] for r in rows])
    (k, J0), _ = curve_fit(_logistic, J, y, p0=[6, 0.7], maxfev=20000,
                           bounds=([0.1, 0.0], [50, 1.5]))
    auc = roc_auc_score(y, _logistic(J, k, J0))
    edges = np.linspace(J.min(), J.max(), 9); cen, rate = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (J >= lo) & (J < hi)
        if msk.sum() >= 5:
            cen.append((lo + hi) / 2); rate.append(float(y[msk].mean()))
    frac_cap_fail = float(np.mean([r['n_steps'] >= 98 for r in rows if r['y'] == 1]))
    return dict(k=float(k), J_star=float(J0), slope=float(k / 4), auc=float(auc),
                n=len(y), n_fail=int(y.sum()), frac_fail_at_cap=frac_cap_fail,
                bins=(cen, rate), J=J, y=y)

def _first_at(mask, t0):
    idx = np.where(mask[t0:])[0]
    return (t0 + idx[0]) if len(idx) else None

def test3(series, fp_grid=(0.05, 0.10, 0.20, 0.40)):
    """H2 on the agent: does the divergence monitor G(t) raise its alarm before a
    raw cumulative-error monitor, on FAILING trajectories? Both thresholds are
    calibrated on the SUCCESS population (disjoint from the evaluation set)."""
    succ = [s for s in series if s['y'] == 0]
    fail = [s for s in series if s['y'] == 1]
    base = 3                                       # skip first few steps (warm-up)
    Gmax_succ = [float(s['G'][base:].max()) for s in succ]
    Emax_succ = [float(np.cumsum(s['err'])[base:].max()) for s in succ]

    def eval_at(thG, thE):
        leads, gfire, examples = [], [], []
        for s in fail:
            G = s['G']; cumE = np.cumsum(s['err'])
            g_idx = _first_at(G > thG, base)
            e_idx = _first_at(cumE > thE, base)
            if e_idx is None:                      # raw monitor never fires -> skip
                continue
            gfire.append(g_idx is not None)
            if g_idx is not None:
                leads.append(float(e_idx - g_idx))
                if len(examples) < 6:
                    examples.append((s, e_idx, g_idx, thG))
        leads = np.array(leads)
        return dict(g_sens=float(np.mean(gfire)) if gfire else None,
                    n=len(leads),
                    median_lead=float(np.median(leads)) if len(leads) else None,
                    frac_lead=float((leads > 0).mean()) if len(leads) else None,
                    leads=leads, examples=examples)

    table, chosen = [], None
    for fp in fp_grid:
        thG = float(np.percentile(Gmax_succ, 100 * (1 - fp)))
        thE = float(np.percentile(Emax_succ, 100 * (1 - fp)))
        res = eval_at(thG, thE)
        table.append(dict(fp=fp, thG=thG, thE=thE, g_sens=res['g_sens'],
                          median_lead=res['median_lead'], frac_lead=res['frac_lead']))
        if fp == 0.20:
            chosen = res
    return dict(table=table, op=next(r for r in table if r['fp'] == 0.20),
                examples=chosen['examples'] if chosen else [],
                leads=chosen['leads'] if chosen else np.array([]))

# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_roc(rows, t1):
    y = np.array([r['y'] for r in rows]); X = sanitize([[r[f] for f in FEATS] for r in rows])
    skf = StratifiedKFold(5, shuffle=True, random_state=0); base = np.linspace(0, 1, 100); tprs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
        clf.fit(X[tr], y[tr]); fpr, tpr, _ = roc_curve(y[te], clf.predict_proba(X[te])[:, 1])
        t = np.interp(base, fpr, tpr); t[0] = 0; tprs.append(t)
    plt.figure(figsize=(5, 5))
    plt.plot(base, np.mean(tprs, 0), lw=2, color='purple',
             label=f"Sentinel J+G  AUC={t1['auc'][0]:.3f}±{t1['auc'][1]:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=.4)
    plt.xlabel("false-alarm rate"); plt.ylabel("failure detection")
    plt.title("Agent T1 — FAIL vs PASS trajectory\n(500 SWE-bench agent runs, 5-fold CV)")
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "agent_t1_roc.png"), dpi=130); plt.close()

def fig_logistic(t2):
    J, y = t2['J'], t2['y']; k, J0 = t2['k'], t2['J_star']
    xs = np.linspace(J.min(), J.max(), 200)
    plt.figure(figsize=(6, 4.3))
    plt.scatter(J, y + np.random.uniform(-.03, .03, len(J)), s=10, alpha=.3, color='mediumpurple')
    cen, rate = t2['bins']
    plt.plot(cen, rate, 'o', color='darkorange', ms=8, label="empirical FAIL-rate / bin")
    plt.plot(xs, _logistic(xs, k, J0), 'r-', lw=2,
             label=f"fit  k={k:.1f}, J*={J0:.2f}, AUC={t2['auc']:.2f}")
    plt.axvline(J0, ls=':', color='red', alpha=.6); plt.text(J0, .5, "  J*", color='red')
    plt.xlabel("J = budget utilisation (steps / 100)")
    plt.ylabel("P(FAIL)")
    plt.title("Agent T2 — logistic phase transition on the agent substrate")
    plt.legend(loc='center left', fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "agent_t2_logistic.png"), dpi=130); plt.close()

def fig_tradeoff(t3):
    tab = t3['table']
    fp = [100 * r['fp'] for r in tab]
    sens = [100 * (r['g_sens'] or 0) for r in tab]
    lead = [r['median_lead'] if r['median_lead'] is not None else 0 for r in tab]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.plot(fp, sens, 'o-', color='purple', label="G sensitivity on FAILS")
    ax1.set_xlabel("false-trigger rate on PASS trajectories (%)")
    ax1.set_ylabel("G detected FAILS (%)", color='purple'); ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(fp, lead, 's--', color='crimson', alpha=.7, label="median lead (steps)")
    ax2.axhline(0, color='gray', lw=.8)
    ax2.set_ylabel("median lead vs raw-error monitor (steps)", color='crimson')
    ax1.set_title("Agent T3 — divergence monitor vs raw-error monitor\n"
                  "(lead > 0 ⇒ G fires earlier; cleaner drift than the ICU substrate)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right'); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "agent_t3_tradeoff.png"), dpi=130); plt.close()

def fig_example(t3):
    if not t3['examples']:
        return
    s, e_idx, g_idx, thG = t3['examples'][0]
    G = s['G']; cumE = np.cumsum(s['err']); t = np.arange(len(G))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(t, G, color='purple', label="G(t) cumulative divergence")
    ax1.axhline(thG, ls=':', color='purple', alpha=.6, label="G alarm threshold")
    ax1.set_xlabel("agent step"); ax1.set_ylabel("G(t)", color='purple')
    ax2 = ax1.twinx()
    ax2.plot(t, cumE, color='gray', label="cumulative explicit errors (raw monitor)")
    ax2.set_ylabel("cumulative errors", color='gray')
    if g_idx is not None:
        ax1.axvline(g_idx, color='purple', lw=1.5); ax1.text(g_idx, G.max() * .9, " G fires", color='purple')
    ax1.axvline(e_idx, color='gray', lw=1.5); ax1.text(e_idx, 0, " raw fires", color='gray')
    lead = e_idx - (g_idx if g_idx is not None else e_idx)
    rel = "G precedes raw" if lead > 0 else "G lags raw"
    ax1.set_title(f"Agent T3 example (FAIL): step_raw={e_idx}, step_G={g_idx}  ({rel} by {abs(lead):.0f})")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "agent_t3_example.png"), dpi=130); plt.close()

# --------------------------------------------------------------------------- #
def main():
    print("Loading agent trajectories ...")
    rows, series, skip = load()
    y = np.array([r['y'] for r in rows])
    print(f"  usable: {len(rows)}   skipped: {skip}   fails: {int(y.sum())}  passes: {int((y==0).sum())}")

    t1 = test1(rows); t2 = test2(rows); t3 = test3(series)

    print("\n=== Agent T1  FAIL-vs-PASS separation (H1/H2) ===")
    print(f"  CV-AUC = {t1['auc'][0]:.3f} ± {t1['auc'][1]:.3f}   (n={t1['n']}, fails={t1['n_fail']})")
    print("  top single features:")
    for f, a in list(t1['single'].items())[:5]:
        print(f"      {f:14s} AUC={a:.3f}")
    print("\n=== Agent T2  logistic phase transition (H1) ===")
    print(f"  J = budget utilisation;  k={t2['k']:.2f}  J*={t2['J_star']:.3f}  max-gain k/4={t2['slope']:.2f}")
    print(f"  AUC(J)={t2['auc']:.3f}   (n={t2['n']}, fails={t2['n_fail']}, "
          f"{100*t2['frac_fail_at_cap']:.0f}% of fails hit the step-cap)")
    print("\n=== Agent T3  G(t) drift vs raw-error monitor lead (H2) ===")
    print("  false-trigger | G-sensitivity | median lead(steps) | %G-leads")
    for r in t3['table']:
        sens = f"{100*r['g_sens']:.0f}%" if r['g_sens'] is not None else "  -"
        lead = f"{r['median_lead']:+.0f}" if r['median_lead'] is not None else "  -"
        frac = f"{100*r['frac_lead']:.0f}%" if r['frac_lead'] is not None else "  -"
        print(f"      {100*r['fp']:4.0f}%     |     {sens:>4s}      |      {lead:>5s}        |   {frac:>4s}")

    fig_roc(rows, t1); fig_logistic(t2); fig_tradeoff(t3); fig_example(t3)
    print(f"\nfigures -> {FIGS}/")
    out = dict(dataset="SWE-bench Verified eval trajectories (500, reward 0/1)",
               n_usable=len(rows), n_fail=int(y.sum()),
               test1=dict(auc=t1['auc'], single=t1['single'], n=t1['n'], n_fail=t1['n_fail']),
               test2=dict(k=t2['k'], J_star=t2['J_star'], slope=t2['slope'], auc=t2['auc'],
                          n=t2['n'], n_fail=t2['n_fail'], frac_fail_at_cap=t2['frac_fail_at_cap']),
               test3=dict(table=t3['table'], op=t3['op']))
    json.dump(out, open(os.path.join(OUT, "results_agents.json"), "w"), indent=2)
    print(f"results -> {OUT}/results_agents.json")

if __name__ == "__main__":
    main()
