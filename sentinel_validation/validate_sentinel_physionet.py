"""
Sentinel — empirical validation of the failure model on a REAL labelled dataset.
================================================================================

Hypothesis under test (from "Sentinel — Dual-Substrate Deviation under One
Failure Model", MAFAT proposal). The proposal validates its mechanism only on
*synthetic* data sampled from a known logistic (milestone M2, "proof of the
recovery method"). This script replaces that synthetic step with REAL data and
asks whether the three core constructs survive contact with it:

  H1  Logistic phase transition.  P_fail(J) = 1 / (1 + exp(-k (J - J*))), with
      the steepest slope (maximum sensitivity) at a recoverable critical point J*.
  H2  Silent drift / early warning.  G(t) = Σ D_KL(behaviour(t) || baseline)·(1-c(t))
      fires DURING the drift, before a raw threshold ("error") monitor.
  H3  One model, two readouts.  The same J / G machinery is applied to a stream;
      here we exercise the *human-operator-style* substrate (a monitored ICU
      patient generating alarms). The agent substrate is the documented next step.

Dataset
-------
PhysioNet/Computing in Cardiology Challenge 2015 — "Reducing False Arrhythmia
Alarms in the ICU". 750 training records, each a multi-channel waveform ending
at a life-threatening arrhythmia alarm (alarm fixed at t = 300 s). Every alarm
is expert-adjudicated TRUE or FALSE. This is the document's #1 operational gap
(alarm fatigue / inattentional blindness) with clean ground truth.

This is the substrate-A pilot of the proposal's milestone M2 on real data.
"""
import os, json, warnings, collections
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, stft
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

DATA_DIR = "/tmp/p2015/training"
OUT      = os.path.dirname(os.path.abspath(__file__))
FIGS     = os.path.join(OUT, "figs")
ALARM_S  = 300            # alarm occurs at 300 s in every record
ECG_ORDER = ['II', 'V', 'I', 'III', 'MCL', 'MCL1', 'aVR', 'aVL', 'aVF']
# Clinical thresholds of the Challenge alarm definitions (bpm)
THRESH = {'Tachycardia': ('hi', 140), 'Bradycardia': ('lo', 40)}

# --------------------------------------------------------------------------- #
#  Signal -> behaviour stream b(t) = heart-rate at 1 Hz over the pre-alarm window
# --------------------------------------------------------------------------- #
def _bandpass(x, fs, lo, hi):
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, x)

def _pick_channel(sig, names):
    for nm in ECG_ORDER:
        if nm in names and np.isfinite(sig[:, names.index(nm)]).mean() > 0.5:
            return names.index(nm), 'ECG'
    for nm in ['ABP', 'PLETH']:
        if nm in names and np.isfinite(sig[:, names.index(nm)]).mean() > 0.5:
            return names.index(nm), nm
    return None, None

def _median_filt(x, k=9):
    k = k | 1
    pad = k // 2
    xp = np.pad(x, pad, mode='edge')
    return np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)

def heart_rate(sig, names, fs, dur=ALARM_S):
    """Robust rate via STFT dominant frequency of the rectified ECG envelope,
    smoothed with a rolling median. Returns HR at 1 Hz on [0, dur) or None."""
    j, chan = _pick_channel(sig, names)
    if j is None:
        return None, None
    n = int(dur * fs)
    x = sig[:n, j].astype(float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    lo, hi = (5, 18) if chan == 'ECG' else (0.7, 5)
    env = np.abs(_bandpass(x, fs, lo, hi))
    f, t, Z = stft(env, fs=fs, nperseg=int(4 * fs), noverlap=int(3 * fs))
    band = (f >= 0.5) & (f <= 3.6)                      # 30..216 bpm
    dom = f[band][np.argmax(np.abs(Z[band, :]), axis=0)]
    hr = 60.0 * dom
    grid = np.arange(0, dur, 1.0)
    hr = np.interp(grid, t, hr)
    return _median_filt(np.clip(hr, 15, 240), 9), chan

# --------------------------------------------------------------------------- #
#  Sentinel constructs:  J(t),  G(t),  raw monitor,  EWS features
# --------------------------------------------------------------------------- #
HR_BINS = np.linspace(20, 200, 19)

def G_divergence(b, base_end=60, W=20, bins=HR_BINS):
    """Cumulative KL divergence of a trailing window from the healthy baseline,
    with the document's persistence factor (1 - c): drift only accumulates while
    it is *not* being corrected (c=1 when the divergence is shrinking)."""
    base = b[:base_end]
    pbase, _ = np.histogram(base, bins=bins); pbase = pbase + 1e-6; pbase /= pbase.sum()
    G = np.zeros(len(b)); prev = 0.0; acc = 0.0
    for t in range(len(b)):
        w = b[max(0, t - W):t + 1]
        pw, _ = np.histogram(w, bins=bins); pw = pw + 1e-6; pw /= pw.sum()
        kl = float(np.sum(pw * np.log(pw / pbase)))
        c = 1.0 if kl < prev else 0.0
        acc += kl * (1.0 - c); prev = kl; G[t] = acc
    return G

def J_approach(b, atype):
    """Normalised approach-to-threshold in [0, ~2]; J>=1 means at the clinical
    threshold. Defined for the rate-based alarm types."""
    if atype == 'Tachycardia':
        return np.clip(b / 140.0, 0, 2)
    if atype in ('Bradycardia', 'Asystole'):
        return np.clip(40.0 / np.maximum(b, 1), 0, 2)
    return np.clip(b / 140.0, 0, 2)               # fallback proxy

def rolling(x, w, fn):
    return np.array([fn(x[max(0, i - w):i + 1]) for i in range(len(x))])

def ac1(x):
    if len(x) < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:       # constant sub-window -> undefined
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

def features(hr, atype):
    """Per-record feature vector. EWS = critical-slowing-down indicators
    (variance & lag-1 autocorrelation and their trends, Scheffer/Dakos) plus the
    cumulative divergence G — exactly the constructs the proposal cites."""
    G = G_divergence(hr)
    var_roll = rolling(hr, 30, np.var)
    ac_roll  = rolling(hr, 30, ac1)
    tt = np.arange(len(hr))
    def slope(y):  return float(np.polyfit(tt, y, 1)[0])
    J = J_approach(hr, atype)
    feat = dict(
        hr_mean=float(hr.mean()), hr_std=float(hr.std()),
        hr_range=float(np.percentile(hr, 95) - np.percentile(hr, 5)),
        hr_trend=slope(hr),
        hr_delta=float(hr[-30:].mean() - hr[:60].mean()),
        ac1=float(ac1(hr)),
        ac1_trend=slope(ac_roll),
        var_trend=slope(var_roll),
        G_final=float(G[-1]),
        G_slope=float((G[-1] - G[len(G) // 2]) / (len(G) / 2)),
        J_peak=float(np.percentile(J[-120:], 95)),
    )
    return feat, G, J

# --------------------------------------------------------------------------- #
#  Load + process every record
# --------------------------------------------------------------------------- #
def process_all():
    labels = json.load(open("/tmp/labels.json"))
    rows, series, skipped = [], {}, 0
    for r, meta in labels.items():
        path = os.path.join(DATA_DIR, r)
        if not os.path.exists(path + ".hea"):
            skipped += 1; continue
        try:
            rec = wfdb.rdrecord(path)
            hr, chan = heart_rate(rec.p_signal, rec.sig_name, rec.fs)
            if hr is None:
                skipped += 1; continue
            feat, G, J = features(hr, meta['type'])
        except Exception:
            skipped += 1; continue
        feat.update(record=r, atype=meta['type'], chan=chan,
                    y=1 if meta['result'] == 'True alarm' else 0)
        rows.append(feat)
        series[r] = dict(hr=hr, G=G, J=J, atype=meta['type'], y=feat['y'])
    return rows, series, skipped

# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #
FEATS = ['hr_mean', 'hr_std', 'hr_range', 'hr_trend', 'hr_delta',
         'ac1', 'ac1_trend', 'var_trend', 'G_final', 'G_slope', 'J_peak']

def sanitize(X):
    """Replace any residual non-finite feature value with its column median."""
    X = np.array(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]; bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.median(col[~bad]) if (~bad).any() else 0.0
    return X

def cv_auc(X, y, n=5):
    X = sanitize(X)
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight='balanced'))
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))

def test1_separation(rows):
    """H1/H2 headline: does the instability/divergence signal separate TRUE from
    FALSE alarms? Reports CV-AUC for the full feature set and single features."""
    import numpy as np
    y = np.array([r['y'] for r in rows])
    X = sanitize([[r[f] for f in FEATS] for r in rows])
    mean, sd = cv_auc(X, y)
    singles = {}
    for i, f in enumerate(FEATS):
        a = roc_auc_score(y, X[:, i])
        singles[f] = float(max(a, 1 - a))           # direction-agnostic
    # subset: rate-based alarm types only (where HR proxy is physiological)
    rate_idx = [i for i, r in enumerate(rows)
                if r['atype'] in ('Tachycardia', 'Bradycardia', 'Asystole')]
    rmean, rsd = cv_auc(X[rate_idx], y[rate_idx])
    return dict(auc_all=(mean, sd), auc_rate_types=(rmean, rsd),
                single_feature_auc=dict(sorted(singles.items(),
                                               key=lambda kv: -kv[1])),
                n=len(y), n_true=int(y.sum()))

def _logistic(J, k, J0):
    return 1.0 / (1.0 + np.exp(-k * (J - J0)))

def test2_logistic(rows):
    """H1: fit the proposal's 1-parameter logistic P(true | J_peak) on the
    rate-threshold alarm types and recover the critical point J* and slope k."""
    sub = [r for r in rows if r['atype'] in ('Tachycardia', 'Bradycardia')]
    J = np.array([r['J_peak'] for r in sub]); y = np.array([r['y'] for r in sub])
    p0 = [8.0, 1.0]
    (k, J0), _ = curve_fit(_logistic, J, y, p0=p0, maxfev=20000,
                           bounds=([0.1, 0.0], [50, 2.0]))
    auc = roc_auc_score(y, _logistic(J, k, J0))
    # empirical true-rate in J bins (for the figure / goodness check)
    edges = np.linspace(J.min(), J.max(), 9)
    cen, rate = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (J >= lo) & (J < hi)
        if m.sum() >= 4:
            cen.append((lo + hi) / 2); rate.append(float(y[m].mean()))
    return dict(k=float(k), J_star=float(J0), auc=float(auc),
                n=len(y), n_true=int(y.sum()),
                slope_at_Jstar=float(k / 4),          # max local gain of a logistic
                bins=(cen, rate), J=J, y=y)

def _first_cross(mask, t0=60):
    """First index >= t0 where boolean mask is True, else None."""
    idx = np.where(mask[t0:])[0]
    return (t0 + idx[0]) if len(idx) else None

def test3_leadtime(series, fp_grid=(0.05, 0.10, 0.20, 0.40)):
    """H2: does the cumulative-divergence monitor G(t) raise its alarm BEFORE a
    raw clinical-threshold ("error") monitor, on the TRUE rate-threshold alarms?

    For each target false-trigger rate, theta_G is calibrated on the FALSE-alarm
    population (max post-baseline G), then sensitivity and lead are measured on
    the disjoint TRUE-alarm set. Reporting the whole trade-off avoids cherry-
    picking a threshold and exposes the artifact confound honestly."""
    rate = lambda s: s['atype'] in ('Tachycardia', 'Bradycardia')
    fmax = [float(s['G'][60:].max()) for r, s in series.items() if rate(s) and s['y'] == 0]
    trues = [(r, s) for r, s in series.items() if rate(s) and s['y'] == 1]

    def eval_at(theta):
        leads, g_fired, examples = [], [], []
        for r, s in trues:
            hr, G = s['hr'], s['G']
            d, thr = THRESH[s['atype']]
            raw_idx = _first_cross((hr > thr) if d == 'hi' else (hr < thr))
            if raw_idx is None:
                continue
            g_idx = _first_cross(G > theta)
            g_fired.append(g_idx is not None)
            if g_idx is not None:
                leads.append(float(raw_idx - g_idx))
                if len(examples) < 6:
                    examples.append((r, raw_idx, g_idx, float(theta)))
        leads = np.array(leads)
        return dict(n_eval=len(g_fired),
                    g_sensitivity=float(np.mean(g_fired)) if g_fired else None,
                    n=len(leads),
                    median_lead_s=float(np.median(leads)) if len(leads) else None,
                    frac_G_leads=float((leads > 0).mean()) if len(leads) else None,
                    leads=leads, examples=examples)

    table = []
    chosen = None
    for fp in fp_grid:
        theta = float(np.percentile(fmax, 100 * (1 - fp)))
        actual_fp = float(np.mean([m > theta for m in fmax]))
        res = eval_at(theta)
        row = dict(fp_target=fp, theta_G=theta, fp_actual=actual_fp,
                   g_sensitivity=res['g_sensitivity'],
                   median_lead_s=res['median_lead_s'],
                   frac_G_leads=res['frac_G_leads'])
        table.append(row)
        if fp == 0.20:                               # operating point for the figure
            chosen = res
    return dict(table=table, examples=chosen['examples'] if chosen else [],
                leads=chosen['leads'] if chosen else np.array([]),
                op_point=next(r for r in table if r['fp_target'] == 0.20))

# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_roc(rows, t1):
    y = np.array([r['y'] for r in rows])
    X = sanitize([[r[f] for f in FEATS] for r in rows])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    plt.figure(figsize=(5, 5))
    tprs, base = [], np.linspace(0, 1, 100)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight='balanced'))
        clf.fit(X[tr], y[tr])
        fpr, tpr, _ = roc_curve(y[te], clf.predict_proba(X[te])[:, 1])
        tprs.append(np.interp(base, fpr, tpr)); tprs[-1][0] = 0
    m = np.mean(tprs, 0)
    plt.plot(base, m, lw=2, label=f"Sentinel EWS+G  AUC={t1['auc_all'][0]:.3f}±{t1['auc_all'][1]:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=.4)
    plt.xlabel("False-alarm rate"); plt.ylabel("True-alarm detection")
    plt.title("T1 — TRUE vs FALSE arrhythmia alarm\n(750 ICU records, 5-fold CV)")
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "t1_roc_true_vs_false.png"), dpi=130); plt.close()

def fig_logistic(t2):
    J, y = t2['J'], t2['y']; k, J0 = t2['k'], t2['J_star']
    xs = np.linspace(J.min(), J.max(), 200)
    plt.figure(figsize=(6, 4.3))
    plt.scatter(J, y + np.random.uniform(-.03, .03, len(J)), s=10, alpha=.35,
                color='steelblue', label="records (jittered)")
    cen, rate = t2['bins']
    plt.plot(cen, rate, 'o', color='darkorange', ms=8, label="empirical TRUE-rate / bin")
    plt.plot(xs, _logistic(xs, k, J0), 'r-', lw=2,
             label=f"fit  P=1/(1+e^-k(J-J*))\nk={k:.1f},  J*={J0:.2f},  AUC={t2['auc']:.2f}")
    plt.axvline(J0, ls=':', color='red', alpha=.6)
    plt.text(J0, .5, "  J*", color='red')
    plt.xlabel("J_peak  (approach to clinical threshold; 1.0 = at threshold)")
    plt.ylabel("P(TRUE alarm)")
    plt.title("T2 — logistic phase transition recovered on real data\n(Tachycardia + Bradycardia)")
    plt.legend(loc='center left', fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "t2_pfail_logistic.png"), dpi=130); plt.close()

def fig_example(series, t3):
    if not t3['examples']:
        return
    r, raw_idx, g_idx, theta_G = t3['examples'][0]
    s = series[r]; hr, G = s['hr'], s['G']; t = np.arange(len(hr))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(t, hr, color='steelblue', label="heart rate")
    thr = THRESH[s['atype']][1]
    ax1.axhline(thr, ls='--', color='gray', label=f"clinical threshold ({thr})")
    ax1.set_xlabel("time (s)   — alarm fires at t=300"); ax1.set_ylabel("HR (bpm)", color='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(t, G, color='crimson', alpha=.8, label="G(t) cumulative divergence")
    ax2.axhline(theta_G, ls=':', color='crimson', alpha=.6, label="G alarm threshold")
    ax2.set_ylabel("G(t)", color='crimson')
    if g_idx is not None:
        ax1.axvline(g_idx, color='crimson', lw=1.5)
        ax1.text(g_idx, hr.max(), " G fires", color='crimson', fontsize=9)
    ax1.axvline(raw_idx, color='gray', lw=1.5)
    ax1.text(raw_idx, hr.min(), " raw monitor", color='gray', fontsize=9)
    lead = raw_idx - (g_idx if g_idx is not None else raw_idx)
    rel = "G precedes raw" if lead > 0 else "G lags raw"
    ax1.set_title(f"T3 example — {r} ({s['atype']} TRUE): "
                  f"t_raw={raw_idx}s, t_G={g_idx}s  ({rel} by {abs(lead):.0f}s)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "t3_example_trace.png"), dpi=130); plt.close()

def fig_leadhist(t3):
    """Trade-off curve: G sensitivity on TRUE alarms vs false-trigger rate, with
    the artifact confound made explicit."""
    tab = t3['table']
    fp = [100 * r['fp_actual'] for r in tab]
    sens = [100 * (r['g_sensitivity'] or 0) for r in tab]
    leads = [r['median_lead_s'] if r['median_lead_s'] is not None else 0 for r in tab]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.plot(fp, sens, 'o-', color='seagreen', label="G sensitivity on TRUE alarms")
    ax1.set_xlabel("false-trigger rate on FALSE alarms (%)")
    ax1.set_ylabel("G detected TRUE alarms (%)", color='seagreen')
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(fp, leads, 's--', color='crimson', alpha=.7, label="median lead (s)")
    ax2.axhline(0, color='gray', lw=.8)
    ax2.set_ylabel("median lead vs raw monitor (s)", color='crimson')
    ax1.set_title("T3 — divergence monitor trade-off (Tachy+Brady TRUE alarms)\n"
                  "artifact-driven FALSE alarms inflate G ⇒ weak standalone early warning")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right'); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "t3_leadtime_tradeoff.png"), dpi=130); plt.close()

# --------------------------------------------------------------------------- #
def main():
    print("Processing 750 ICU alarm records ...")
    rows, series, skipped = process_all()
    print(f"  usable records: {len(rows)}   skipped: {skipped}")
    by = collections.Counter((r['atype'], r['y']) for r in rows)
    print("  class balance:", dict(collections.Counter(r['y'] for r in rows)))

    t1 = test1_separation(rows)
    t2 = test2_logistic(rows)
    t3 = test3_leadtime(series)

    print("\n=== T1  TRUE-vs-FALSE separation (H1/H2) ===")
    print(f"  CV-AUC all types       : {t1['auc_all'][0]:.3f} ± {t1['auc_all'][1]:.3f}  (n={t1['n']}, true={t1['n_true']})")
    print(f"  CV-AUC rate-based types: {t1['auc_rate_types'][0]:.3f} ± {t1['auc_rate_types'][1]:.3f}")
    print("  top single features    :")
    for f, a in list(t1['single_feature_auc'].items())[:5]:
        print(f"      {f:11s} AUC={a:.3f}")
    print("\n=== T2  logistic phase transition (H1) ===")
    print(f"  recovered  k = {t2['k']:.2f},  J* = {t2['J_star']:.3f},  max-gain k/4 = {t2['slope_at_Jstar']:.2f}")
    print(f"  AUC(J_peak) = {t2['auc']:.3f}   (n={t2['n']}, true={t2['n_true']})")
    print("\n=== T3  early-warning lead of G over raw monitor (H2) ===")
    print("  false-trigger | G-sensitivity | median lead | %G-leads   (calibrated on FALSE alarms)")
    for row in t3['table']:
        sens = f"{100*row['g_sensitivity']:.0f}%" if row['g_sensitivity'] is not None else "  -"
        lead = f"{row['median_lead_s']:+.0f}s" if row['median_lead_s'] is not None else "   -"
        frac = f"{100*row['frac_G_leads']:.0f}%" if row['frac_G_leads'] is not None else "  -"
        print(f"     {100*row['fp_actual']:4.0f}%     |     {sens:>4s}      |    {lead:>5s}    |   {frac:>4s}")
    print("  -> pure-divergence alarm is confounded by artifact-driven FALSE alarms (see report).")

    fig_roc(rows, t1); fig_logistic(t2); fig_example(series, t3); fig_leadhist(t3)
    print(f"\nfigures written to {FIGS}/")

    out = dict(
        dataset="PhysioNet/CinC Challenge 2015 (training, 750 records)",
        n_usable=len(rows), n_skipped=skipped,
        test1_separation={k: v for k, v in t1.items()},
        test2_logistic=dict(k=t2['k'], J_star=t2['J_star'], auc=t2['auc'],
                            slope_at_Jstar=t2['slope_at_Jstar'],
                            n=t2['n'], n_true=t2['n_true']),
        test3_leadtime=dict(tradeoff_table=t3['table'], op_point=t3['op_point']),
    )
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(f"results written to {OUT}/results.json")

if __name__ == "__main__":
    main()
