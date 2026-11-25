# PR #15 audit, prompt 2 of 2: for a fresh window

Paste everything below the line into a fresh agent window started at the repository root, on branch
`claude/legobot-tick-timing-574623`, **after** the E1 window has committed its recovery work
(`pr15-handoff-1-e1-window.md`).

---

You are fixing the findings of an independent audit of PR #15 (branch `claude/legobot-tick-timing-574623`). The
engine, config and robot-code changes in that PR are sound and stay as they are. What needs fixing is how Phase E1's
results are recorded and interpreted, plus a few small gaps. Do not start E2.

## Rules

- Python: always `C:\venvs\balancebot\Scripts\python.exe` (Bash: `/c/venvs/balancebot/Scripts/python.exe`).
  Run from the repo root.
- Read first: `CLAUDE.md`, `hardware/legobot/E1_NOTES.md`, `docs/ml-baselines-plan.md` (sections 2.5, 3.3, 13),
  `hardware/legobot/SOFTWARE_PLAN.md`, `hardware/legobot/A4_BASELINES.md` (last section),
  `docs/legobot-review/review-v2.md`.
- Seed tiers (`config.py` `RobustnessSettings`): training 1–50; smoke 5000–5009; dev 5000–5049. The validation
  set (3000–3099) and the fall-rate set (6000–6999) are **frozen**: never score them, and never pass `--final`.
  Retired, never reuse: 100–109, 1000–1099, 2000–2099. Fresh seeds start above 6,000,000.
- NumPy float64 for every number you report. Under pulse-density PWM plus backlash, a run's fall is chaotic:
  compare candidates with paired per-scenario bootstrap CIs and McNemar tests, never on 150-scenario fall counts
  alone.
- After any change to the simulator, the tuner or a config, run both suites, `python -m pytest` and
  `ROBOT=legobot python -m pytest`, and the robot tests,
  `python -m pytest hardware/legobot/robot/test_control.py hardware/legobot/robot/measure/test_fit.py`.
  All three passed at `fae53bb` (663 / 534 / 43).
- The Pololu robot must stay bit-identical to `d252bff`: same `CONFIG.snapshot()`, same objectives, same saved
  JSON keys (apart from L7 below).
- One commit per finding. Don't push, merge or open a PR without asking the user.

## Step 0: check the E1 window's recovery landed

The E1 window was asked to commit, under `hardware/legobot/`, the scripts behind
`results/legobot/e1_ablations/ablation_summary.json`, `results/legobot/e1_learning_scales_check.json`,
`results/legobot/comparison/e1_staged_gains/` and the seeds-5050–5149 confirmation, plus per-variant records,
the 5050–5149 per-scenario results, and corrected provenance in the staged gain files. Check `git log` and the
files. If the scripts aren't there, stop and tell the user.

Then confirm the provenance fix: `compare_controllers.py --gain-runs` on
`e1_staged_gains/cma_lat10_15/seed0/cma.json` must print "trained under a different config:
robustness.latency_range_s". If it doesn't, fix the file (or the check) and add a test in
`optimization/tests/test_compare_controllers.py`.

## L1 (major): E1's tuned PIDs don't beat the untuned model-based PID out of sample

The audit re-scored on 300 fresh 60 s runs (seeds 5050–5149, robust, NumPy float64, gains from E1_NOTES'
Step 1 table):

| Controller | Mean | Falls | Median |
|---|---:|---:|---:|
| model-based (1.9797, 5.236, 0.1509, 0.5153) | 60.1 | 5/300 | 27.80 |
| CMA-mid s0 best (2.0618, 7.4220, 0.1457, 0.4307) | 72.2 | 7/300 | 27.59 |
| CMA-warm (mean) s1 (2.0173, 3.9543, 0.1641, 0.5388) | 89.5 | 9/300 | 31.71 |

E1_NOTES' own fresh-seed table agrees. CMA-mid's edge (1.046 [1.006, 1.084]) holds on the selection seeds
5000–5049 only.

Fix:
1. In `E1_NOTES.md` sections 1, 3 and 5, state plainly that no Step 1 candidate beats the model-based PID out of
   sample (typical cost equal within noise, falls no better).
2. Remove the recommendation to freeze a ki ≈ 6–7 reference PID (section 4 table, last row; section 5 item 5).
   The reference PID choice stays open for E2, with the model-based PID as a finalist.
3. Add an E2 protocol item to section 5: the robust objective is a mean dominated by about 5–13 chaotic falls in
   150 scenarios, so the optimizer largely fits fall luck. Propose (don't run) the options: more training seeds,
   and/or a median or trimmed-mean objective with falls counted separately; always re-score the model-based PID
   beside the finalists.
4. Correct `SOFTWARE_PLAN.md`'s E1 row and `E1_HANDOFF.md` wherever they imply E1 produced a better PID.

## L3 (major): seeds 5050–5149 are an unregistered tier

`E1_NOTES.md`, `E1_HANDOFF.md:58` and `A4_BASELINES.md:520` use and recommend 5050–5149 as an "extended dev"
slice. It isn't in `RobustnessSettings`, so the overlap check doesn't cover it, and it conflicts with the plan's
single fresh-seed protocol (`docs/ml-baselines-plan.md` 3.3). It doesn't touch the frozen sets.

Ask the user which option to take; A is simpler:
- **A. Register it.** Add `dev_extended_seeds: Tuple[int, ...] = tuple(range(5050, 5150))` to
  `RobustnessSettings` / `RobustnessConfig` in `config.py`, include it in the overlap check, add it to the tier
  table in `docs/ml-baselines-plan.md` 2.5 (use: confirming dev selections on scenarios they weren't picked on;
  never for selection), and add it to `NOT_SCORING` like the other held-back sets. Add a test that it overlaps no
  other set. Pololu's snapshot must stay identical.
- **B. Retire it.** Add 5050–5149 to `retired_seeds` and the plan as spent, and point the guidance in
  `E1_HANDOFF.md` at fresh seeds above 6,000,000.

## L4 (major): the LEGO `LEARNING` scales are stale now

`python hardware/legobot/legobot_learning_scales.py` (the plan's own procedure: hand-tuned reference, nominal
robot, 150 dev runs of 60 s) at `fae53bb`:

| Quantity | max | p99 | p90 | configured scale (table says p99) |
|---|---:|---:|---:|---|
| int_v | 0.731 | 0.694 | 0.656 | 0.4 (0.39) |
| int_theta | 0.347 | 0.330 | 0.311 | 0.2 (0.18) |
| pid_int_theta | 0.092 | 0.054 | 0.036 | 0.06 |
| theta_dot_meas | 4.28 | 2.53 | 1.77 | 3.4 |
| v_meas | 0.815 | 0.529 | 0.351 | 0.4 |

At `d252bff` it was already off (int_v p99 0.469), and the hand-tuned PID fell 4/150 on the nominal robot there; at
`fae53bb` it falls 0/150, so the measuring protocol holds again.

Fix:
1. Re-run the script; update the table in `legobot_config.py` (around lines 331–345) and the `LEARNING` values by
   the rules the table states (peak for the integrators and setpoint, p99 where it says p99).
2. Note in `docs/ml-baselines-plan.md` 13 that the scales were re-measured on 2026-11-24 after the Round-2 fixes,
   still with the hand-tuned reference, and are re-measured again once E2 picks its reference PID.
3. Run both suites; `optimization/tests/test_legobot_learning.py` may pin values.

## L5 (minor): ablation selection bias and overstated PDM wording

Each Step 2 variant picks between its CMA-ES mean and best sample on dev 5000–5049; the `full_model` control
picks on the full model with those same seeds, which are then part of the cross-evaluation (`dev150_60s` and 150
of `dev450`). One optimizer seed per variant. The large effects (IMU, encoder, backlash) survive; PDM (p = 0.017)
and free tick phase (p = 0.057) don't clear the Bonferroni level the notes themselves state (0.0045).

Fix:
1. Add the selection-bias caveat to `E1_NOTES.md` section 2.
2. Change `SOFTWARE_PLAN.md:234` so PDM and tick phase read "kept on physical grounds; ablation evidence
   suggestive", matching E1_NOTES.
3. Add to E1_NOTES section 5's E2 plan: re-run the PDM and tick-phase ablations with 2–3 tuning seeds, picking on
   training seeds (or a held-out slice) and evaluating on scenarios not used for picking.

## Minor (cheap; after the above)

- **L6, gearbox play.** The 2.58° in `bench_data/20261123_073624_backlash_fit.json` comes from the whole 15 s
  record (including the 1–2 s slip); 3–15 s alone gives 2.67° (`robot.measure.backlash.analyse` on that window).
  The half-centroid estimate is biased toward the middle, and in 3–15 s the counts span −3…+1 (true span over 3°,
  so at least ±1.5°). Correct the wording in `DATASHEET.md` 8.7, `PHYSICS.md` 10.3, the `ROBOT` comment in
  `legobot_config.py` and `review-v2.md`. Propose to the user, don't change the config: nominal 1.5° with a range of
  about 1.2–2.0°, or re-measure with the rotor braked during Bench C.
- **L7, Pololu JSON.** `optimization/tuning/reporting.py:51` writes `mean_gains` into Pololu's `cma_es.json` too.
  Gate it to LEGO (as the `cma-es (mean)` row already is), or document it.
- **L8, review items marked done that aren't.**
  - `legobot_scenario.py:211`: `_Play.step` still feeds `drive()` the motor-side (oracle) speed; pass the encoder
    shaft speed, as H5 asked.
  - Add H5's test that `caught(..., dt=0.0025)` holds torque 0 for `round(latency / dt)` steps, and H4's
    timing-distribution test in `simulation/tests/test_slow_pwm.py`.
  - H8.1: at least a printed note about the fixed tick phase in `run_tuning.py` and `run_batch_tuning.py --plain`.
  - `hardware/legobot/pwm_pulse_check.py:71,85`: strict `> 0.5` / `< -0.5`, like `pdm_update`.
  - Re-run `python hardware/legobot/legobot_scenario.py` (about 30 min) after the `_Play.step` change; if the
    catch bounds move from 0.0377 / 0.3049 / 0.423, tell the user before touching the pushes.
- **L9, robot code.** Seed `Safety.vin` from the start-up vin (`balance.py` has `notes["vin at start"]`), so the
  fixed 17 rad/s fallback isn't used until the first in-loop reading. In `sim_hw.SimWorld`, assert `pwm_tick`
  equals `HAT_LINE_PERIOD` within a tolerance, or pace on `pwm_tick`. Fix the `measure/backlash.py` docstring,
  which says both "NOT coast" and "coasts".
- **L10.** In `E1_NOTES.md` Step 1's table, label the "Plain 60 s (30 runs)" column as `run_tuning.py`'s scoring on
  seeds 100–109 (retired; existing `run_batch_tuning.py` behaviour).

## Done when

- Step 0 checked; L1, L3, L4, L5 fixed, each in its own commit; the minors where cheap.
- Three test suites pass; Pololu snapshot, objectives and saved-JSON keys unchanged against `d252bff`.
- No file in `results/legobot/` scores a seed in 3000–3099 or 6000–6999.
- A short summary for the user: each finding, what changed, and anything left undone and why.
