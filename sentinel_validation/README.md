# Sentinel — empirical validation pilot

Validates the failure model from the *Sentinel — Dual-Substrate Deviation under
One Failure Model* (MAFAT) proposal on a **real, labelled** dataset instead of
the proposal's synthetic data (milestone M2).

**Dataset:** PhysioNet/CinC Challenge 2015 — *Reducing False Arrhythmia Alarms
in the ICU* (750 expert-labelled true/false alarms). This is the proposal's #1
operational gap — alarm fatigue.

See **[`REPORT.md`](REPORT.md)** for the full writeup and verdict on H1/H2/H3.

## What it tests
- **T1** — does the instability/divergence signal separate TRUE vs FALSE alarms? (5-fold CV AUC)
- **T2** — recover the logistic phase transition `P_fail(J)=1/(1+e^{-k(J-J*)})`; estimate `J*`, `k`.
- **T3** — does the cumulative-divergence monitor `G(t)` precede a raw threshold monitor? (lead-time trade-off)

## Headline result
- **H1 holds:** logistic transition recovered, `J*≈0.62` (proposal calibrated `0.65`); rate-type AUC `0.83`.
- **H2 does not replicate cleanly:** artifact-driven false alarms inflate cumulative `G`, so the
  pure-divergence early warning is confounded on real physiological data — exactly the synthetic→field gap.

## Reproduce
```bash
pip install numpy scipy pandas scikit-learn matplotlib wfdb

# 1. fetch the dataset (~400 MB) to /tmp/p2015/training/ and write /tmp/labels.json
python3 fetch_data.py

# 2. run the validation
python3 validate_sentinel_physionet.py
```
Outputs: `results.json` and `figs/*.png`.

> The raw waveform data is **not** committed (open-access on PhysioNet; ~400 MB).
> `fetch_data.py` downloads it and extracts the labels.
