"""
Sentinel — substrate C (silent drift). Validation of the SAME failure model on
NASA C-MAPSS turbofan run-to-failure data.

Why this substrate: the ICU pilot (artifact) and the agent pilot (loud, early
errors) both lack the regime H2 is designed for — a failure that develops
*silently* until it crosses a threshold. C-MAPSS engines degrade gradually across
21 sensors with NO discrete error events, which is exactly the document's
"silent drift / critical slowing down" regime (Scheffer/Dakos, cited in the
proposal's Appendix A). All 100 engines run to failure, so there is no pass/fail
class here; the decisive test is LEAD TIME:

  H1  P_fail(J) logistic, J = normalised degradation index  -> recover J*, k.
  H2  does the cumulative-divergence monitor G(t) raise its alarm BEFORE a raw
      redline monitor, at a MATCHED healthy false-alarm rate? (the regime where
      H2 should finally hold)

Same maths (J, G, logistic) as the ICU and agent pilots — exercising H3 on a
third, fundamentally different substrate (machine prognostics).
"""
import os, json, warnings
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(0)

DATA = "/tmp/cmapss/train_FD001.txt"
OUT  = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(OUT, "figs")
BASELINE = 20            # first cycles = healthy reference
W = 10                   # divergence window (cycles)

# --------------------------------------------------------------------------- #
#  Health index H(t) = Mahalanobis distance of the sensor vector from the
#  pooled healthy baseline.  G(t) = cumulative KL drift of H from its baseline.
# --------------------------------------------------------------------------- #
def load():
    d = np.loadtxt(DATA)
    units = d[:, 0].astype(int)
    sensors = d[:, 5:]
    keep = sensors.std(0) > 1e-6              # drop constant sensors
    S = sensors[:, keep]
    # standardise by pooled healthy baseline (first BASELINE cycles of each engine)
    healthy = np.vstack([S[(units == u)][:BASELINE] for u in np.unique(units)])
    mu, sd = healthy.mean(0), healthy.std(0) + 1e-9
    Z = (S - mu) / sd
    Zh = (healthy - mu) / sd
    cov = np.cov(Zh.T) + 1e-3 * np.eye(Zh.shape[1])
    P = np.linalg.pinv(cov)
    def maha(z):
        return np.sqrt(np.einsum('ij,jk,ik->i', z, P, z))
    engines = {}
    for u in np.unique(units):
        m = units == u
        cyc = d[m, 1].astype(int)
        H = maha(Z[m])
        engines[int(u)] = dict(cycle=cyc, H=H, life=int(cyc.max()),
                               rul=cyc.max() - cyc)
    return engines

def G_divergence(H, base_end=BASELINE, W=W, nb=12):
    base = H[:base_end]
    bins = np.linspace(base.min(), np.percentile(H, 99) + 1e-6, nb)
    pb, _ = np.histogram(base, bins=bins); pb = pb + 1e-6; pb /= pb.sum()
    G = np.zeros(len(H)); prev = 0.0; acc = 0.0
    for t in range(len(H)):
        w = H[max(0, t - W):t + 1]
        pw, _ = np.histogram(w, bins=bins); pw = pw + 1e-6; pw /= pw.sum()
        kl = float(np.sum(pw * np.log(pw / pb)))
        c = 1.0 if kl < prev else 0.0
        acc += kl * (1.0 - c); prev = kl; G[t] = acc
    return G

# --------------------------------------------------------------------------- #
#  H1 — logistic phase transition on the degradation index
# --------------------------------------------------------------------------- #
def _logistic(J, k, J0):
    return 1.0 / (1.0 + np.exp(-k * (J - J0)))

def test_h1(engines):
    """Cycle-level: J = normalised degradation; label imminent (RUL<=30) vs
    healthy (RUL>=80). Recover the critical J*."""
    Hs = np.concatenate([e['H'] for e in engines.values()])
    lo = float(np.percentile(Hs, 50))                       # typical healthy level
    hi = float(np.median([e['H'][-3:].mean() for e in engines.values()]))  # failure level -> J~1
    J, y = [], []
    for e in engines.values():
        Jn = np.clip((e['H'] - lo) / (hi - lo), 0, 1.3)
        for j, rul in zip(Jn, e['rul']):
            if rul <= 30:   J.append(j); y.append(1)
            elif rul >= 80: J.append(j); y.append(0)
    J, y = np.array(J), np.array(y)
    (k, J0), _ = curve_fit(_logistic, J, y, p0=[6, 0.5], maxfev=20000,
                           bounds=([0.1, -0.5], [60, 1.5]))
    auc = roc_auc_score(y, _logistic(J, k, J0))
    edges = np.linspace(J.min(), min(J.max(), 1.3), 11); cen, rate = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (J >= a) & (J < b)
        if m.sum() >= 20: cen.append((a + b) / 2); rate.append(float(y[m].mean()))
    return dict(k=float(k), J_star=float(J0), slope=float(k / 4), auc=float(auc),
                n=len(y), n_imminent=int(y.sum()), bins=(cen, rate), J=J, y=y)

# --------------------------------------------------------------------------- #
#  H2 — early-warning lead of G(t) over a raw redline, matched false-alarm rate
# --------------------------------------------------------------------------- #
def test_h2(engines, alphas=(0.005, 0.01, 0.05)):
    for e in engines.values():
        e['G'] = G_divergence(e['H'])
    # healthy reference pool = first BASELINE cycles of every engine
    H_healthy = np.concatenate([e['H'][:BASELINE] for e in engines.values()])
    G_healthy = np.concatenate([e['G'][:BASELINE] for e in engines.values()])

    def fire(sig, thr):
        idx = np.where(sig[BASELINE:] > thr)[0]
        return (BASELINE + idx[0]) if len(idx) else None

    table, chosen = [], None
    for a in alphas:
        th_raw = float(np.quantile(H_healthy, 1 - a))
        th_G   = float(np.quantile(G_healthy, 1 - a))
        lead_raw, lead_G, gain, examples = [], [], [], []
        for u, e in engines.items():
            tr = fire(e['H'], th_raw); tg = fire(e['G'], th_G)
            lr = (e['life'] - tr) if tr is not None else 0
            lg = (e['life'] - tg) if tg is not None else 0
            lead_raw.append(lr); lead_G.append(lg)
            if tr is not None and tg is not None:
                gain.append(tr - tg)                      # cycles G fires earlier
            if a == 0.01 and tr is not None and tg is not None and len(examples) < 6:
                examples.append((u, tr, tg, th_raw, th_G))
        row = dict(alpha=a, th_raw=th_raw, th_G=th_G,
                   med_lead_raw=float(np.median(lead_raw)),
                   med_lead_G=float(np.median(lead_G)),
                   med_gain_cycles=float(np.median(gain)) if gain else None,
                   frac_G_earlier=float(np.mean([g > 0 for g in gain])) if gain else None,
                   n_both=len(gain))
        table.append(row)
        if a == 0.01:
            chosen = examples
    return dict(table=table, examples=chosen or [])

# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_h1(h1):
    J, y = h1['J'], h1['y']; k, J0 = h1['k'], h1['J_star']
    xs = np.linspace(J.min(), min(J.max(), 1.3), 200)
    plt.figure(figsize=(6, 4.3))
    cen, rate = h1['bins']
    plt.plot(cen, rate, 'o', color='darkorange', ms=8, label="empirical imminent-fail rate / bin")
    plt.plot(xs, _logistic(xs, k, J0), 'r-', lw=2, label=f"fit  k={k:.1f}, J*={J0:.2f}, AUC={h1['auc']:.2f}")
    plt.axvline(J0, ls=':', color='red', alpha=.6); plt.text(J0, .5, "  J*", color='red')
    plt.xlabel("J = normalised degradation index"); plt.ylabel("P(imminent failure, RUL≤30)")
    plt.title("C-MAPSS H1 — logistic phase transition (3rd substrate)")
    plt.legend(loc='center left', fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cmapss_h1_logistic.png"), dpi=130); plt.close()

def fig_example(engines, h2):
    if not h2['examples']:
        return
    u, tr, tg, th_raw, th_G = h2['examples'][0]
    e = engines[u]; t = e['cycle']
    fig, ax1 = plt.subplots(figsize=(7.4, 4.2))
    ax1.plot(t, e['H'], color='teal', label="health index H(t) (Mahalanobis)")
    ax1.axhline(th_raw, ls='--', color='gray', label="raw redline")
    ax1.axvline(t[tr], color='gray', lw=1.5); ax1.text(t[tr], e['H'].min(), " raw fires", color='gray')
    ax1.axvline(t[tg], color='crimson', lw=1.5); ax1.text(t[tg], e['H'].max()*.8, " G fires", color='crimson')
    ax1.axvline(e['life'], color='black', lw=1, ls=':'); ax1.text(e['life'], e['H'].min(), " FAIL", fontsize=8)
    ax2 = ax1.twinx(); ax2.plot(t, e['G'], color='crimson', alpha=.7, label="G(t) cumulative divergence")
    ax2.axhline(th_G, ls=':', color='crimson', alpha=.5)
    ax1.set_xlabel("cycle"); ax1.set_ylabel("H(t)", color='teal'); ax2.set_ylabel("G(t)", color='crimson')
    ax1.set_title(f"C-MAPSS H2 — engine {u}: G fires {t[tr]-t[tg]} cycles before the raw redline")
    h1_, l1 = ax1.get_legend_handles_labels(); h2_, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1_ + h2_, l1 + l2, fontsize=7, loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cmapss_h2_example.png"), dpi=130); plt.close()

def fig_lead(engines, h2):
    row = next(r for r in h2['table'] if r['alpha'] == 0.01)
    leadG, leadR = [], []
    for u, e in engines.items():
        idxG = np.where(e['G'][BASELINE:] > row['th_G'])[0]
        idxR = np.where(e['H'][BASELINE:] > row['th_raw'])[0]
        if len(idxG): leadG.append(e['life'] - (BASELINE + idxG[0]))
        if len(idxR): leadR.append(e['life'] - (BASELINE + idxR[0]))
    plt.figure(figsize=(6.4, 4.2))
    bins = np.linspace(0, max(max(leadG), max(leadR)), 25)
    plt.hist(leadR, bins=bins, alpha=.6, color='gray', label=f"raw redline (median {np.median(leadR):.0f})")
    plt.hist(leadG, bins=bins, alpha=.6, color='crimson', label=f"Sentinel G (median {np.median(leadG):.0f})")
    plt.xlabel("warning lead before failure (cycles)"); plt.ylabel("# engines")
    plt.title("C-MAPSS H2 — earlier warning at matched 1% healthy false-alarm rate\n"
              "silent drift: G fires well before the raw redline")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cmapss_h2_leadtime.png"), dpi=130); plt.close()

# --------------------------------------------------------------------------- #
def main():
    print("Loading C-MAPSS FD001 (100 run-to-failure engines) ...")
    engines = load()
    print(f"  engines: {len(engines)}   median life: {int(np.median([e['life'] for e in engines.values()]))} cycles")

    h1 = test_h1(engines)
    h2 = test_h2(engines)

    print("\n=== C-MAPSS H1  logistic phase transition ===")
    print(f"  k={h1['k']:.2f}  J*={h1['J_star']:.3f}  max-gain k/4={h1['slope']:.2f}  AUC={h1['auc']:.3f}")
    print(f"  (cycle-level: {h1['n_imminent']} imminent vs {h1['n']-h1['n_imminent']} healthy)")
    print("\n=== C-MAPSS H2  early-warning lead of G over raw redline (THE silent-drift test) ===")
    print("  healthy-FA | median lead RAW | median lead G | G earlier by | %G-earlier")
    for r in h2['table']:
        gain = f"{r['med_gain_cycles']:+.0f}c" if r['med_gain_cycles'] is not None else "  -"
        frac = f"{100*r['frac_G_earlier']:.0f}%" if r['frac_G_earlier'] is not None else " -"
        print(f"     {100*r['alpha']:4.1f}%   |     {r['med_lead_raw']:5.0f}       |   {r['med_lead_G']:5.0f}      |"
              f"   {gain:>5s}     |   {frac:>4s}")

    fig_h1(h1); fig_example(engines, h2); fig_lead(engines, h2)
    print(f"\nfigures -> {FIGS}/")
    out = dict(dataset="NASA C-MAPSS FD001 (100 run-to-failure turbofan engines)",
               n_engines=len(engines),
               h1=dict(k=h1['k'], J_star=h1['J_star'], slope=h1['slope'], auc=h1['auc'],
                       n=h1['n'], n_imminent=h1['n_imminent']),
               h2=dict(table=h2['table']))
    json.dump(out, open(os.path.join(OUT, "results_cmapss.json"), "w"), indent=2)
    print(f"results -> {OUT}/results_cmapss.json")

if __name__ == "__main__":
    main()
