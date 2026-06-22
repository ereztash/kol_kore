"""
META-TEST B — the joint "compound failure window" (the proposal's actual
differentiator).

The proposal's unique claim is NOT a logistic risk curve or a drift detector
(both exist) -- it is monitoring operator AND agent under ONE model, so the
dangerous case "both elevated SIMULTANEOUSLY" is caught even though each substrate
alone looks ordinary. We isolate exactly that.

Design (controlled simulation; no public coupled human+agent dataset exists):
the MARGINAL distribution of each substrate is made IDENTICAL between normal and
compound episodes -- in both, each axis has one elevated burst. The ONLY
difference is timing: in normal episodes the two bursts are INDEPENDENT, in
compound episodes they are SYNCHRONIZED. Therefore any single-substrate monitor is
provably blind (its detection rate on compound = its false-alarm rate), and only a
joint monitor can exploit the simultaneity. Every monitor is calibrated to the
SAME false-alarm rate on normal episodes.
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(OUT, "figs")
T = 200
BURST_H, BURST_D = 0.34, 16          # burst height above 0.2 baseline, duration

def _ar1(n, rng, s=0.03):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + s * rng.standard_normal()
    return x

def episode(kind, jitter=0, rng=None):
    rng = rng or np.random.default_rng()
    op = 0.2 + _ar1(T, rng); ag = 0.2 + _ar1(T, rng)
    def burst(x, t0):
        x[t0:t0 + BURST_D] += BURST_H
        return x
    if kind == 'normal':
        burst(op, rng.integers(20, T - BURST_D - 20))
        burst(ag, rng.integers(20, T - BURST_D - 20))      # independent timing
    elif kind == 'compound':
        t0 = rng.integers(40, T - BURST_D - 40)
        burst(op, t0)
        burst(ag, t0 + rng.integers(-jitter, jitter + 1) if jitter else t0)  # synchronized
    op = np.clip(op, 0, 1.2); ag = np.clip(ag, 0, 1.2)
    both = (op >= 0.40) & (ag >= 0.40)                      # joint elevation
    fail_time, run = None, 0
    for t in range(T):
        run = run + 1 if both[t] else 0
        if run >= 8:
            fail_time = t; break
    return dict(kind=kind, op=op, ag=ag, fail=fail_time)

def make_dataset(n, jitter=0, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n):
        kind = 'normal' if rng.random() < 0.5 else 'compound'
        eps.append(episode(kind, jitter, rng))
    return eps

def calibrate_eval(eps, far=0.05):
    normal = [e for e in eps if e['kind'] == 'normal']
    th_op = float(np.quantile([e['op'].max() for e in normal], 1 - far))
    th_ag = float(np.quantile([e['ag'].max() for e in normal], 1 - far))
    th_jt = float(np.quantile([(e['op'] * e['ag']).max() for e in normal], 1 - far))
    # raise single thresholds so OR also has FAR = far (OR of two ~independent monitors)
    th_or = float(np.quantile([max(e['op'].max(), e['ag'].max()) for e in normal], 1 - far))

    def t_op(e):  i = np.where(e['op'] > th_op)[0];                 return int(i[0]) if len(i) else None
    def t_ag(e):  i = np.where(e['ag'] > th_ag)[0];                 return int(i[0]) if len(i) else None
    def t_or(e):
        i = np.where((e['op'] > th_or) | (e['ag'] > th_or))[0];     return int(i[0]) if len(i) else None
    def t_jt(e):  i = np.where(e['op'] * e['ag'] > th_jt)[0];       return int(i[0]) if len(i) else None
    fns = dict(op_only=t_op, agent_only=t_ag, OR=t_or, joint=t_jt)

    far_meas = {n: float(np.mean([f(e) is not None for e in normal])) for n, f in fns.items()}
    comp = [e for e in eps if e['kind'] == 'compound' and e['fail'] is not None]
    res = {}
    for n, f in fns.items():
        det, leads = 0, []
        for e in comp:
            ft = f(e)
            if ft is not None and ft <= e['fail']:
                det += 1; leads.append(e['fail'] - ft)
        res[n] = dict(detect=det / len(comp) if comp else 0.0,
                      median_lead=float(np.median(leads)) if leads else 0.0)
    return res, far_meas, (th_op, th_jt), comp

def fig_example(comp, th):
    e = comp[0]; t = np.arange(T)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(t, e['op'], color='steelblue', label="J_op (operator)")
    ax.plot(t, e['ag'], color='seagreen', label="J_ag (agent)")
    ax.plot(t, e['op'] * e['ag'], color='crimson', lw=2, label="J_op·J_ag (joint)")
    ax.axhline(th[0], ls='--', color='gray', label="single-substrate alarm line")
    ax.axhline(th[1], ls=':', color='crimson', label="joint alarm line")
    if e['fail']:
        ax.axvline(e['fail'], color='black', ls=':', label="compound failure")
    ax.set_title("Meta-B — identical marginals, synchronized bursts:\n"
                 "neither single axis crosses its line; the JOINT monitor fires")
    ax.set_xlabel("time"); ax.set_ylabel("J"); ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "metaB_compound_example.png"), dpi=130); plt.close()

def fig_sweep(sweep):
    j = [s['jitter'] for s in sweep]
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(j, [s['joint'] * 100 for s in sweep], 's-', color='crimson', label="joint Sentinel monitor")
    plt.plot(j, [s['OR'] * 100 for s in sweep], 'o-', color='gray', label="OR of two dashboards")
    plt.xlabel("burst de-synchronization (± steps jitter)")
    plt.ylabel("compound-failure detection (%)")
    plt.title("Meta-B — joint monitor's power IS the simultaneity it captures\n"
              "(matched 5% false-alarm rate; singles ≈ chance throughout)")
    plt.legend(); plt.ylim(0, 100); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "metaB_sweep.png"), dpi=130); plt.close()

def main():
    print("META-B: joint compound-failure window (controlled simulation)\n")
    eps = make_dataset(6000, jitter=0)
    res, far_meas, th, comp = calibrate_eval(eps)
    print(f"compound failing episodes: {len(comp)}")
    print("false-alarm rate on NORMAL (calibrated equal):")
    print("  " + "  ".join(f"{k}={v*100:.1f}%" for k, v in far_meas.items()))
    print("\nCOMPOUND detection & lead at matched FAR:")
    for n, r in res.items():
        print(f"   {n:10s}  detect={r['detect']*100:5.1f}%   median lead={r['median_lead']:5.1f}")
    print("\n(single monitors are provably blind: detection ≈ their false-alarm rate,")
    print(" because the marginal of each axis is identical in normal vs compound.)")

    sweep = []
    for jit in (0, 8, 16, 30, 60):
        e2 = make_dataset(4000, jitter=jit, seed=jit + 1)
        r2, _, _, _ = calibrate_eval(e2)
        sweep.append(dict(jitter=jit, joint=r2['joint']['detect'], OR=r2['OR']['detect']))
        print(f"  jitter ±{jit:2d}: joint detect={r2['joint']['detect']*100:4.0f}%  OR detect={r2['OR']['detect']*100:4.0f}%")

    fig_example(comp, th); fig_sweep(sweep)
    print(f"\nfigures -> {FIGS}/")
    json.dump(dict(far=far_meas, compound=res, sweep=sweep),
              open(os.path.join(OUT, "results_meta_B.json"), "w"), indent=2)
    print(f"results -> {OUT}/results_meta_B.json")

if __name__ == "__main__":
    main()
