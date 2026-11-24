# E1 handoff: development tuning on the LEGO robot

State on 2026-11-24, branch `claude/legobot-tick-timing-574623` (post-Round-2 external review, ready for E1):
- **Round-1 & Round-2 review fixes are in:** Round-1 findings (`docs/legobot-review/triage.md`, including the slow-PWM
  `drive()` fix) and Round-2 reconciled findings (`docs/legobot-review/review-v2.md`: `H1`–`H13`, `I1`–`I3`).
- **Pulse ticks at a free phase (bench B1, 2026-11-23):** the lock model (ticks locked to the loop, 7.5–15 ms)
  was reverted after B1 found the HAT's ticks keep no fixed phase. Each scenario draws the tick phase
  (`pwm_tick_phase`, a 0–10 ms wait) on top of a 7.5–12.5 ms sample-to-HAT latency. E1 numbers made before
  this aren't comparable with ones after it.
- **Backlash landed & reconciled (B5 + Round-2 `H3`/`I1`):** all three engines model gearbox play (`±1.3°` nominal,
  randomized `±1.1–1.6°` after excluding unbraked `set 0` rotor slip) and coupling play (`±1.5°` nominal, randomized
  `±0–3.0°`). Re-derived pushes are **`0.73 N` × 50 ms** at **`5.5 s` and `7.5 s`** (`jf = 0.03772 N s`).
- **Search & scoring safeguards (`H1`/`H2`):** `SEARCH.ki_max = 10.0` (so `model_based_gains()` at `ki = 5.24`
  sits inside the search space near the midpoint), `ROBUSTNESS.sensitivity_weight = 0.0` on LEGO (disables the
  chaotic twin penalty, halving GPU wall clock; objective = `mean_cost`), and `run_batch_tuning.py` supports
  `--cma-x0` warm-starts and scores `cma-es (mean)` (`xfavorite`).
- **Model-based PID:** made linearly stable (integral ratio `0.3`, `ki = 5.24`).
- **Tests:** they run clean under both robots in `C:\venvs\balancebot`, torch included.
- **Baselines (`legobot_baselines.py`, post-`H9`/`H11`):** hand-tuned plain `20.5` (`0/9`), train **115.7** (`8` of 150 fell),
  dev **172.8** (`13`); model-based plain `10.8` (`0/9`), train **93.4** (`7`), dev **83.3** (`6`) (`A4_BASELINES.md`, last section).
- **Phase E1 outcome (`E1_NOTES.md`, 2026-11-24):** completed. Step 2 ablations ranked the LEGO models for E2, while no Step 1 tuned PID beat the untuned `model-based` PID out of sample on `5050–5149` (`300 × 60 s`: falls `5–13/300` vs `5/300` for `model-based`; typical cost within a few percent, with the `CMA-ES` best samples' `1.5–5%` edge not recoverable by dev selection). The reference PID choice remains open for E2, with `model-based` kept as a finalist alongside objective options that reduce chaotic fall-luck dominance.

## What gates what

| Item | Gates |
|---|---|
| Bench B1 (tick offsets after a power cycle) and B2 (duty 0: coast or brake), in BENCH_CHECKS.md | E1 conclusions. *Both done 2026-11-23.* B1 failed: the ticks keep no fixed phase, so the lock model was reverted (free tick phase drawn per scenario, latency 7.5–12.5 ms; pushes 0.67 N initially, now 0.73 N after Round-2 `H3`; triage G6). B2: duty 0 coasts, as modelled. E1 is unblocked |
| The backlash port (part 2 below) | E2. **Done / Landed** in all three engines (`balance_sim`, `batch_sim`, `torch_sim.py`; PHYSICS.md 10.3): play is drawn per scenario (`±1.1–1.6°` gearbox, `±0–3.0°` coupling after Round-2 `H3`/`I1`) and pushes are re-derived with it (`0.73 N` at `5.5 s` and `7.5 s`). Baselines fall in 4–9% of robust scenarios (`4.7–8.7%`; `A4_BASELINES.md`, last section) |
| Bench B3–B4 and C1–C5 | E2 (final tuning) |

---

## Part 1: prompt for the E1 window (paste into a fresh window in this repository's root)

I'm starting phase E1 of my LEGO balancing-robot project: development tuning of the cascaded PID on the
LEGO robot's simulation, to find which of the robot's models matter. This window owns E1. The bench
checks run in another window (the backlash port in part 2 is already landed in all three engines).

**Read first:**
1. `CLAUDE.md`: the venv and the test commands.
2. `hardware/legobot/E1_HANDOFF.md`: this file, for what gates what.
3. `hardware/legobot/SOFTWARE_PLAN.md`: phase E, and the "External review" paragraph.
4. `hardware/legobot/A4_BASELINES.md`: the last two sections, for the baselines and the engine-noise note.
5. `legobot_config.py`.

**Rules:**
- **Python:** use `C:\venvs\balancebot` (torch with CUDA). Run the LEGO searches in float64:
  `python optimization/run_batch_tuning.py --robot legobot --backend torch`, with no `--float32` (the script
  warns why).
- **Seeds:** develop on the training (1–50) and dev (5000–5049) seeds only. The validation (3000–3099) and
  fall-rate (6000–6999) sets are frozen. Don't pass `--final`.
- **Speed over accuracy while developing:** start with short budgets, benchmark the full robust run's wall
  clock on this GPU, then scale up.
- **No LQR on this robot.**
- **Noise:** under pulse-density PWM + backlash a run is bit-chaotic. A single flipped fall moves a 150-scenario mean
  by ~11, and single-tier 150-scenario scores carry **`~±30` objective (`±3–4` falls)** of chaotic variance (and NumPy
  vs torch can disagree on borderline falls). Do not treat single-tier differences within `±30` / `±4` falls as signal:
  compare candidates and ablations with **paired per-scenario bootstrap CIs / McNemar tests** (or larger dev seed slices
  when close, e.g. `5050–5149`, or trace the falls).

**Steps:**
1. Benchmark, then run the default robust tuning (DE and CMA-ES, trying `--cma-x0 model-based` as well as midpoint)
   in float64. Compare with A4's current baselines:
   hand-tuned plain `20.5` (`0/9`), train **115.7** (`8` of 150 fell), dev **172.8** (`13`); model-based plain `10.8` (`0/9`),
   train **93.4** (`7`), dev **83.3** (`6`)
   (`A4_BASELINES.md`, last section: `±1.1–1.6°` gearbox play, `±0–3.0°` coupling play, `0.73 N` pushes at `5.5 s`
   and `7.5 s`, free tick phase, `sensitivity_weight = 0.0`, `ki_max = 10.0`, post-`H9`/`H11`).
2. Ablations, to find which models matter. Re-tune or re-score with one model switched to its idealized
   form at a time, and see what changes the tuned gains or their cost:
   - **gearbox and coupling backlash**, against rigid `0` play (and gearbox-only vs coupling-only), plus a sensitivity
     check on the wider **`coupling_backlash` `0–4.35°`** and **`gearbox_backlash` `±1.0–2.0°`** bounds (`review-v2.md`
     `I1`/`J5`, testing whether the unadjusted `11.3°` visual span / CSV dwell-swing range changes the tuned gains);
   - the IMU and complementary filter, against measured pitch with noise;
   - the encoder, against continuous velocity;
   - the pulse ticks, against averaged slow PWM, and the tick phase's spread, against one fixed phase;
   - the feedforward constants' mismatch spreads;
   - the pitch offset spread;
   - **the latency range:** `(0.010, 0.015)` vs `(0.0075, 0.0125)` (covering the `15.0–17.5 ms` hardware loop-delay band; `review-v2.md` `H4`).

   Keep each ablation small (dev seeds, modest budgets).
3. Check the tuned gains on a wider latency sweep, as an evaluation only.
4. Record everything in a new `hardware/legobot/E1_NOTES.md`: commands, budgets, wall clock, results,
   conclusions. Note which models matter for E2.
5. If bench B3 or B4 changes the model, as recorded in `docs/legobot-review/triage.md`, stop and re-plan.
6. Ask me before committing. Don't push.

---

## Part 2: brief for the backlash port (SOFTWARE_PLAN B5 — **Done / Landed**)

> **Status: Done / Landed.** The 3-DOF step-event backlash model is already implemented and tested in all three
> engines (`balance_sim`, `batch_sim`, `torch_sim.py`; see `PHYSICS.md` 10.3 and `SOFTWARE_PLAN.md` B5), with
> Round-2 reconciled play ranges (`±1.1–1.6°` gearbox, `±0–3.0°` coupling) and `0.73 N` pushes (`review-v2.md`
> `H3`/`I1`). The original handoff brief is kept below for historical reference.

**Why:** the gearbox play measures ±2.6–3° at the encoder, plus up to ±3° of LEGO coupling play the encoder
can't see (DATASHEET 8.7). A reference-engine-only experiment (PHYSICS.md 10.3) found that ±1.5° raised the
baselines' cost 19–129% and added a fall. So the model has to be ported before E2.

**The problem with the old experiment:** it modelled the tooth contact as a stiff 200 Hz spring-damper. That
needs a 0.1 ms step, 25× finer than tuning's 2.5 ms, so the batched engines would run 25× slower. Port a
model that keeps dt = 2.5 ms instead. The likely design, to confirm against the old experiment's results on
the nominal robot:
- The motor's output side gets its own angle (a 5th and 6th state: angle and speed of the motor side
  relative to the wheel, or the gap position).
- Inside the dead zone ±b, the motor side moves under motor torque and motor friction, and the wheel is
  uncoupled.
- When the gap closes, an inelastic engagement: momentum exchange, restitution ≈ 0. Then the two are locked
  until the transmitted torque reverses sign.
- Events are detected at step granularity, with identical arithmetic in all three engines.

**Done when:**
- The reference engine reproduces the old experiment's cost increases within reason.
- The batched NumPy engine matches the reference bit for bit in float64, and torch matches to rounding
  (`test_equivalence.py`, `test_torch.py`).
- Config fields exist for the play: per side, randomized in robust scoring within the measured range, about
  ±3–6° total.
- Where the encoder sits relative to the play is a decision; DATASHEET 8.7 has the evidence.
- PHYSICS.md 10.3 and SOFTWARE_PLAN B5 are updated.

**Rules:** the same as part 1 (venv, both robots' test suites, all three engines). Ask before committing.
