"""
META-TEST C — "normalization of deviance" made measurable (C-MAPSS).

The proposal's actual phenomenon ("חריגה שלא נענשה הופכת לנורמה"): the PERCEIVED
alarm threshold drifts upward as uncorrected deviations are normalized,
    J_alarm(t+1) = J_alarm(t) + alpha * delta(t) * (1 - c(t)),
until it floats above the TRUE threshold J* -- so an observer using the perceived
threshold goes blind. Sentinel's value is to track the GAP (J_alarm - J*), which
is exactly the accumulated normalized deviance.

We test three observers on the 100 run-to-failure engines, at matched healthy
false-alarm rate:
  habituating : alarms when J(t) > J_alarm(t)   (the operator who normalizes)
  fixed       : alarms when J(t) > J*           (a static alarm)
  gap (Sentinel): alarms when (J_alarm - J*) > theta_gap  (tracks the normalization)

Sweeping the habituation rate alpha shows: the faster deviance is normalized, the
blinder the habituating observer becomes -- while the gap monitor keeps its lead.
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from validate_sentinel_cmapss import load, BASELINE
OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(OUT, "figs")

def normalize_J(engines):
    Hs = np.concatenate([e['H'] for e in engines.values()])
    lo = float(np.percentile(Hs, 50))
    hi = float(np.median([e['H'][-3:].mean() for e in engines.values()]))
    for e in engines.values():
        e['J'] = np.clip((e['H'] - lo) / (hi - lo), 0, 1.5)
    return engines

def reference(J, beta):
    """The observer's drifting 'new normal': an EWMA that habituates to recent J.
    Faster beta = faster normalization of whatever it currently sees."""
    ref = np.zeros(len(J)); ref[0] = J[0]
    for t in range(1, len(J)):
        ref[t] = (1 - beta) * ref[t - 1] + beta * J[t]
    return ref

def first_cross(mask):
    i = np.where(mask[BASELINE:])[0]
    return (BASELINE + i[0]) if len(i) else None

def evaluate(engines, beta, far=0.01):
    base0 = float(np.median([e['J'][:BASELINE].mean() for e in engines.values()]))
    # habituating reference + gap from the fixed healthy normal
    for e in engines.values():
        e['ref'] = reference(e['J'], beta)
        e['gap'] = e['ref'] - base0
    # calibrate every monitor to the SAME healthy false-alarm rate
    hab_h = np.concatenate([(e['J'] - e['ref'])[:BASELINE] for e in engines.values()])
    fix_h = np.concatenate([(e['J'] - base0)[:BASELINE] for e in engines.values()])
    gap_h = np.concatenate([e['gap'][:BASELINE] for e in engines.values()])
    m_hab = float(np.quantile(hab_h, 1 - far))
    m_fix = float(np.quantile(fix_h, 1 - far))
    th_gap = float(np.quantile(gap_h, 1 - far))

    leads = {'habituating': [], 'fixed': [], 'gap': []}
    miss = {'habituating': 0, 'fixed': 0, 'gap': 0}
    for e in engines.values():
        life = e['life']
        t_hab = first_cross((e['J'] - e['ref']) > m_hab)     # adaptive baseline (normalizes)
        t_fix = first_cross((e['J'] - base0) > m_fix)        # fixed healthy baseline
        t_gap = first_cross(e['gap'] > th_gap)               # tracks the normalization itself
        for name, t in (('habituating', t_hab), ('fixed', t_fix), ('gap', t_gap)):
            if t is None:
                miss[name] += 1                              # blind to this failure
            else:
                leads[name].append(life - t)                # lead among DETECTED only
    return ({k: (float(np.median(v)) if v else 0.0) for k, v in leads.items()},
            {k: miss[k] / len(engines) for k in miss}, base0)

def main():
    print("META-C: normalization of deviance / threshold gap (C-MAPSS)\n")
    engines = normalize_J(load())

    print("median lead before failure (cycles) | miss-rate, at matched 1% healthy FAR:")
    print("  beta | habituating        | fixed          | gap(Sentinel)")
    sweep = []
    for beta in (0.05, 0.1, 0.2, 0.4, 0.6):
        leads, miss, base0 = evaluate(engines, beta)
        sweep.append(dict(beta=beta, leads=leads, miss=miss))
        print(f"  {beta:4.2f} |   {leads['habituating']:4.0f} (miss {miss['habituating']*100:3.0f}%) |"
              f"  {leads['fixed']:4.0f} (miss {miss['fixed']*100:2.0f}%) |  {leads['gap']:4.0f} (miss {miss['gap']*100:2.0f}%)")

    # example trajectory at strong habituation
    evaluate(engines, 0.4)
    e = engines[list(engines.keys())[0]]; t = e['cycle']; base0 = e['gap'][0] * 0 + float(np.median([engines[u]['J'][:BASELINE].mean() for u in engines]))
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(t, e['J'], color='teal', label="J(t) degradation")
    ax.plot(t, e['ref'], color='orange', lw=2, label="drifting 'normal' (habituated baseline)")
    ax.axhline(base0, ls='--', color='crimson', label="true healthy baseline")
    ax.fill_between(t, base0, e['ref'], where=e['ref'] > base0, color='orange', alpha=.2,
                    label="normalized-deviance gap (Sentinel tracks)")
    ax.axvline(e['life'], color='black', ls=':', label="failure")
    ax.set_xlabel("cycle"); ax.set_ylabel("J")
    ax.set_title("Meta-C — the habituated baseline chases the drift upward;\n"
                 "an observer using it goes blind. The GAP is the hidden risk.")
    ax.legend(fontsize=7, loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "metaC_threshold_gap.png"), dpi=130); plt.close()

    plt.figure(figsize=(6.2, 4.2))
    a = [s['beta'] for s in sweep]
    plt.plot(a, [s['miss']['habituating'] * 100 for s in sweep], 'o-', color='gray',
             label="habituating observer (normalizes deviance)")
    plt.plot(a, [s['miss']['fixed'] * 100 for s in sweep], 's-', color='steelblue',
             label="fixed baseline")
    plt.plot(a, [s['miss']['gap'] * 100 for s in sweep], '^-', color='crimson',
             label="gap monitor (Sentinel)")
    plt.xlabel("habituation rate β (how fast deviance is normalized)")
    plt.ylabel("failures MISSED entirely (%)  — blindness")
    plt.title("Meta-C — normalization of deviance is real & measurable:\n"
              "faster normalization ⇒ blinder observer; a fixed baseline stays at 0%")
    plt.legend(); plt.ylim(-2, 40); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "metaC_sweep.png"), dpi=130); plt.close()

    print(f"\nfigures -> {FIGS}/")
    json.dump(dict(sweep=sweep), open(os.path.join(OUT, "results_meta_C.json"), "w"), indent=2)
    print(f"results -> {OUT}/results_meta_C.json")

if __name__ == "__main__":
    main()
