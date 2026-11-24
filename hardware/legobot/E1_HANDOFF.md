# E1 handoff: development tuning on the LEGO robot (and the backlash port)

State on 2026-11-23, branch `legobot-config` (merged with `main`, PR #13 included):
- **Review fixes are in:** the review's findings (`docs/legobot-review/triage.md`), with the slow-PWM
  `drive()` fix.
- **Pulse ticks at a free phase (bench B1, 2026-11-23):** the lock model (ticks locked to the loop, 7.5–15 ms)
  was reverted after B1 found the HAT's ticks keep no fixed phase. Each scenario now draws the tick phase
  (`pwm_tick_phase`, a 0–10 ms wait) on top of a 7.5–12.5 ms sample-to-HAT latency, and the scenario was
  re-derived with the longer reactions (A4_BASELINES.md, last section). E1 numbers made before this aren't
  comparable with ones after it.
- **Model-based PID:** made linearly stable (integral ratio 0.3).
- **Tests:** they run clean under both robots in `C:\venvs\balancebot`, torch included.
- **Baselines:** the reference points are A4_BASELINES.md's last table.

## What gates what

| Item | Gates |
|---|---|
| Bench B1 (tick offsets after a power cycle) and B2 (duty 0: coast or brake), in BENCH_CHECKS.md | E1 conclusions. *Both done 2026-11-23.* B1 failed: the ticks keep no fixed phase, so the lock model was reverted (free tick phase drawn per scenario, latency 7.5–12.5 ms, pushes 0.67 N; triage G6, branch `claude/legobot-tick-timing-574623`). B2: duty 0 coasts, as modelled. E1 can start once that branch is merged |
| The backlash port (part 2 below) | E2. *Done 2026-11-23 on branch `claude/legobot-backlash`* (PHYSICS.md 10.3): the play is drawn per scenario and the pushes are re-derived with it (0.70 N, was 1.02; 0.67 N after B1). The baselines fall in 3–11% of robust scenarios, every fall catchable (A4_BASELINES.md, last section). E1 results without it are not comparable with E2's |
| Bench B3–B4 and C1–C5 | E2 (final tuning) |

---

## Part 1: prompt for the E1 window (paste into a fresh window in this repository's root)

I'm starting phase E1 of my LEGO balancing-robot project: development tuning of the cascaded PID on the
LEGO robot's simulation, to find which of the robot's models matter. This window owns E1. The bench
checks run in another window, and so may the backlash port (part 2 of `hardware/legobot/E1_HANDOFF.md`).

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
- **Noise:** under pulse-density PWM a run is bit-chaotic. A flipped fall moves a 150-scenario mean by ~11,
  and the NumPy and torch engines can disagree on a borderline fall. Don't read differences smaller than
  that between candidates without tracing the falls (dev seeds, or the retired 1000–1099 set).

**Steps:**
1. Benchmark, then run the default robust tuning (DE and CMA-ES) in float64. Compare with A4's baselines:
   hand-tuned train 170.9 (12 of 150 fell), dev 226.5 (17); model-based train 73.0 (5), dev 120.9 (9)
   (A4_BASELINES.md, last section: play modelled, free tick phase).
2. Ablations, to find which models matter. Re-tune or re-score with one model switched to its idealized
   form at a time, and see what changes the tuned gains or their cost:
   - the IMU and complementary filter, against measured pitch with noise;
   - the encoder, against continuous velocity;
   - the pulse ticks, against averaged slow PWM, and the tick phase's spread, against one fixed phase;
   - the feedforward constants' mismatch spreads;
   - the pitch offset spread;
   - the latency range.

   Keep each ablation small (dev seeds, modest budgets).
3. Check the tuned gains on a wider latency sweep, as an evaluation only.
4. Record everything in a new `hardware/legobot/E1_NOTES.md`: commands, budgets, wall clock, results,
   conclusions. Note which models matter for E2.
5. If bench B1 or B2 changes the model, as recorded in `docs/legobot-review/triage.md`, stop and re-plan.
6. Ask me before committing. Don't push.

---

## Part 2: brief for the backlash port (SOFTWARE_PLAN B5; its own window, can run alongside E1)

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
