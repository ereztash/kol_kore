# Sentinel — empirical validation pilot

Validates the failure model from the *Sentinel — Dual-Substrate Deviation under
One Failure Model* (MAFAT) proposal on **real, labelled** data instead of the
proposal's synthetic data — across **three** fundamentally different substrates.

See **[`REPORT.md`](REPORT.md)** for the three-substrate writeup (H1/H2/H3), and
**[`META_REPORT.md`](META_REPORT.md)** for the meta-tests that probe whether we
measured the proposal's *actual novelty* (ablation vs standard baselines; the joint
compound-failure window; normalization of deviance) rather than its commodity parts.

## Three substrates, the same equations (`J`, `G`, logistic)

| | A — human operator | B — AI agent | C — machine (silent drift) |
|---|---|---|---|
| Dataset | PhysioNet/CinC 2015 ICU alarms (750) | SWE-bench Verified agent runs (500) | NASA C-MAPSS FD001 (100 engines) |
| `J` (load/capacity) | approach to HR threshold | budget utilisation (steps/100) | normalised degradation index |
| failure regime | noisy (artifact) | noisy (loud errors) | **silent drift** |
| script | `validate_sentinel_physionet.py` | `validate_sentinel_agents.py` | `validate_sentinel_cmapss.py` |

## Verdict
- **H1 (logistic phase transition) — holds on all three** (AUC 0.83 / 0.73 / 0.97).
  But `J*` is substrate-specific (0.62 / 0.82 / 0.32), **not** a universal constant.
- **H2 (G as early warning) — regime-dependent, and this is the key finding.**
  On genuinely *silent* drift (C-MAPSS) `G` warns a median **+78 cycles before** a raw
  redline (90% of engines). On noisy substrates (ICU artifact, agent loud-errors) it
  gives no lead. The proposal's H2 is correct **in its intended regime**.
- **H3 (one model, many substrates) — supported:** the same `J`/`G`/logistic pipeline
  recovers failure on ECG waveforms, agent traces, and turbofan sensors.

## Reproduce
```bash
pip install -r requirements.txt

python3 fetch_data.py    && python3 validate_sentinel_physionet.py   # A: ICU   (~400 MB)
python3 fetch_agents.py  && python3 validate_sentinel_agents.py      # B: agent (~150 MB)
python3 fetch_cmapss.py  && python3 validate_sentinel_cmapss.py      # C: C-MAPSS (~3 MB)
```
Outputs: `results*.json` and `figs/*.png`. Raw data is open-access and **not** committed.
