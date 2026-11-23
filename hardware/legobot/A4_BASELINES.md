# A4: baselines on the LEGO robot

SOFTWARE_PLAN A4, 2026-11-22. The LEGO robot (`legobot_config.py`) on its derived scenario (A3,
`legobot_scenario.py`), with the current-mode DC motor model. These are the reference points the
LEGO tuning runs (E1, E2) are compared against. Re-score with `python hardware/legobot/legobot_baselines.py`.

## Scores

None of these are tuned by search. Mean cost per run (lower is better), falls in brackets.

| Baseline | Gains | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev worst | Stable latency (nominal, linear) |
|---|---|---|---|---|---|---|
| Hand-tuned PID | kp 6.0, ki 2.0, kd 0.5, kv_p 0.3 | 1.797 (0) | 2.071 (0) | 2.050 (0) | 4.16 | 0–67 ms |
| Model-based PID | 3.636, 3.456, 0.382, 0.355 | 1.317 (0) | 1.498 (0) | 1.479 (0) | 2.73 | 0–80 ms |
| Cost-matched LQR | −4.325, −6.876, −0.919, 0, 0.130, 0.128 | 1.548 (0) | 1.992 (0) | 1.903 (0) | 5.23 | — |

"Robust" uses the robustness box of `legobot_config.py` (DATASHEET 2 spreads, latency 5–25 ms) and
config.py's chaos penalty; the chaos term is 2.3e-4 for all three. Costs are not comparable with the
Pololu robot's (different scenario and cost limits). Fall rates need the 3,000-scenario fall-rate set;
0 of 150 here only rules out frequent falls.

## Model-based PID

`pid_design` with config.py's design choices (inner poles at 2× the unstable pole, damping 1,
integral pole at 0.1×, outer bandwidth 1/3 of the RHP zero), from the LEGO robot's linear model.
Nothing changed.

## Hand-tuned PID: how the gains were picked

config.py's hand-tuned gains (4.0, 0.5, 0.15, 0.1) are linearly unstable on this robot at 10 ms,
and score 12.0 plain. Picked by hand instead, scoring each trial on the plain scenario and on the 10
smoke seeds (5000–5009, robust):

| Trial | kp, ki, kd, kv_p | Plain | Smoke robust (falls of 30) | Note |
|---|---|---|---|---|
| 0 | 4.0, 0.5, 0.15, 0.1 | 11.99 | 14.72 (0) | config.py's; unstable at 10 ms |
| 1 | 8.4, 1.05, 0.32, 0.24 | 24.79 | 565 (9) | config.py's × 2.1 (ratio of critical pitch gains, 0.627/0.298); kv_p × (1.3/0.26) × (12°/25°) |
| 2 | 8.4, 1.05, 0.32, 0.1 | 9.28 | 13.51 (0) | unstable at 10 ms |
| 3 | 6.0, 1.05, 0.32, 0.24 | 2.79 | 8.26 (0) | lower kp stops the oscillation |
| 4 | 8.4, 1.05, 0.5, 0.24 | 2.68 | 5.47 (0) | so does more kd |
| 5 | 8.4, 0, 0.32, 0.24 | 24.84 | 564 (9) | ki isn't the cause |
| 6 | 6.0, 1.05, 0.5, 0.24 | 2.57 | 2.78 (0) | both |
| 7 | 5.0, 1.05, 0.4, 0.24 | 2.58 | 2.79 (0) | unstable at 10 ms |
| 8 | 6.0, 1.05, 0.5, 0.3 | 1.94 | 2.15 (0) | more kv_p for the slower goal |
| 9 | 6.0, 2.0, 0.5, 0.24 | 2.36 | 2.52 (0) | |
| 10 | 6.0, 1.0, 0.5, 0.4 | 1.59 | 3.91 (0) | better plain, worse robust: stopped raising kv_p |
| **11** | **6.0, 2.0, 0.5, 0.3** | **1.80** | **1.98 (0)** | **chosen**: round numbers, best robust |

Eleven trials, round numbers only, no search. The smoke seeds are the first 10 of the dev seeds, so
the dev score of the hand-tuned set is slightly optimistic; the model-based and LQR designs saw no
seeds at all.

## Tests

`pololu_only` skips the ten tests that check config.py's Pololu numbers or Pololu-specific gains
(including the module-level pitch-only gains in test_conditions and the 0.3 m/s steps in
test_velocity, which still fail with LEGO values, so they stay skipped). Five tests that had assumed
zero Coulomb friction now zero it themselves and run on both robots.
