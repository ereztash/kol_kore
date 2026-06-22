# Sentinel — empirical validation pilot

Validates the failure model from the *Sentinel — Dual-Substrate Deviation under
One Failure Model* (MAFAT) proposal on **real, labelled** data instead of the
proposal's synthetic data — on **both** substrates the proposal targets.

See **[`REPORT.md`](REPORT.md)** for the full writeup and verdict on H1/H2/H3.

## Two substrates, the same equations

| | Substrate A — human operator | Substrate B — AI agent |
|---|---|---|
| Dataset | PhysioNet/CinC 2015 ICU alarms (750, true/false) | SWE-bench Verified agent runs (500, pass/fail) |
| `J` (load/capacity) | approach to clinical HR threshold | budget utilisation (steps / 100) |
| `G(t)` | KL drift of HR distribution | KL drift of `{node,tool_call,retry,error}` dist |
| script | `validate_sentinel_physionet.py` | `validate_sentinel_agents.py` |

## What each pipeline tests
- **T1** — does the instability/divergence signal separate the two classes? (5-fold CV AUC)
- **T2** — recover the logistic `P_fail(J)=1/(1+e^{-k(J-J*)})`; estimate `J*`, `k`.
- **T3** — does the divergence monitor `G(t)` precede a raw error monitor? (lead-time trade-off)

## Headline result
- **H1 holds on both substrates:** logistic phase transition recovered, `J*=0.62` (ICU)
  and `J*=0.82` (agent) — both inside the proposal's stated range 0.4–0.8; AUC 0.83 / 0.73.
- **H3 supported:** the *same* `J`/`G`/logistic pipeline recovers failure on ECG waveforms
  *and* agent traces — fundamentally different substrates.
- **H2 does not give early-warning lead** on either public dataset — ICU because artifact-driven
  false alarms inflate cumulative `G`; agent because SWE-bench errors are loud and early (no
  "silent drift"). This pinpoints the synthetic→field gap the funded work must close.

## Reproduce
```bash
pip install -r requirements.txt

# Substrate A — ICU alarms (~400 MB)
python3 fetch_data.py && python3 validate_sentinel_physionet.py

# Substrate B — agent traces (~150 MB)
python3 fetch_agents.py && python3 validate_sentinel_agents.py
```
Outputs: `results.json`, `results_agents.json`, and `figs/*.png`.

> Raw data is **not** committed (both open-access; ~550 MB total). The `fetch_*.py`
> scripts download it.
