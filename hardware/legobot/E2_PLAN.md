# E2 plan: final tuning on the measured LEGO robot, and the full L policy

Drafted 2026-11-24, against `main` at `89f660c` (bench checks, the ±4.0° total-play cap, 0.71 N pushes,
the policy-on-robot code). Follows `SOFTWARE_PLAN.md` phase E, `E1_NOTES.md` section 5, and
`POLICY_ON_ROBOT_PLAN.md` section 4. **Decisions D1–D6 (section 3) accepted as recommended, 2026-11-24.**

## 1. What E2 produces

1. **A frozen reference PID** for the LEGO robot: a tuned finalist, or the model-based PID if no tuned
   candidate beats it out of sample (E1's outcome).
2. **A full-profile L policy** (the 10-weight linear policy from `docs/ml-baselines-plan.md` 4.1), trained
   on the measured config with the scales re-measured from the frozen reference. It is exported as a bundle
   for the Pi, so `POLICY_ON_ROBOT_PLAN.md` P7 step 4 has a real policy to run.
3. **One final scoring run** on the frozen validation and fall-rate tiers, for every LEGO controller in the
   paper, once.

## 2. Starting point

- **Config:** `legobot_config.py` on `main`. Pushes 0.71 N × 50 ms at 5.5 s and 7.5 s; gearbox play
  ±1.1–3.4° and coupling play ±0–2.5°, with the total capped at ±4.0°; coulomb friction 0.030–0.060 N m;
  η 0.739; latency 7.5–12.5 ms plus the free tick phase.
- **Baselines** (`A4_BASELINES.md`, last section; robust, falls in brackets):

  | | Train (1–50) | Dev (5000–5049) |
  |---|---|---|
  | Model-based PID | 72.45 (5) | 86.47 (6) |
  | Hand-tuned PID | 222.24 (17) | 197.33 (15) |

- **E1:** no tuned PID beat model-based out of sample. The 150-scenario mean objective is dominated by 5–13
  chaotic falls (about 1,000–2,000 each), so the searches mostly fit fall luck on seeds 1–50.
- **Policies:** the four bundles in `results/legobot/policies/` (`l_pid`, `quick_l`, `a_warm_zero`,
  `b1_zero`) match `main`'s config today. They carry the provisional scales and the hand-tuned reference,
  so after E2.3–E2.4 the Pi's config check (P2b) will refuse them until they are re-exported.
- **Robot code:** P1–P6 and P8 are done (PR #20). P7 steps 1–3 need only L-PID. Step 4 waits on this plan's L.

## 3. Decisions (accepted 2026-11-24)

| # | Decision | Adopted | Why |
|---|---|---|---|
| D1 | How to stop fitting fall luck | **More training seeds, same cost:** `--seeds 150` (450 scenarios) for the PID searches and `--train-seeds 150` for L. A trimmed or median objective is the fallback, only if dev still shows fall-dominated selection | Same score for every method (fairness contract item 4); it cuts fall-count noise by about √3 without changing what is optimized. `first_train_seeds` already steps around the retired 100–109 |
| D2 | Where to confirm dev picks | **A fresh slice:** 6,000,000–6,000,099 (300 scenarios, 60 s), registered as a tier before use | 5050–5149 already informed E1's "model-based stays a finalist" conclusion, so it's no longer clean for E2's choice |
| D3 | Rule for the reference PID | Adopt a tuned finalist only if, on the D2 slice, its paired median-cost ratio to model-based has a 95% CI below 1 **and** McNemar doesn't show more falls. Otherwise model-based | Written down before the runs, so the choice can't follow the results |
| D4 | Latency range | Keep 7.5–12.5 ms. P3d measures the Pi's real read-to-command delay with the policy; revisit only if that falls outside the range | E1's 10–15 ms result didn't hold out of sample (`E1_NOTES.md` 5.2) |
| D5 | Which LEGO controllers the final run covers | **Reference PID, model-based, hand-tuned and L.** A, A-warm and B1 on LEGO only if they are frozen by then; otherwise they are out of the LEGO comparison | The frozen tiers run once, on frozen controllers only. A controller added later can't be scored on them without breaking that rule |
| D6 | L's precision and run seeds | **Train in float64**, 10 run seeds, on the cluster if available. On the laptop alone: 3 run seeds, disclosed | LEGO float32 is off by 2.6% (triage G5), past the 1% flag (fairness item 8). Float64 is about 2.7× slower (section 5) |

## 4. Steps

Rules as in E1: the venv, float64 for LEGO searches, train and dev seeds only until E2.8, no LQR, and
paired bootstrap / McNemar comparisons (a single 150-scenario tier carries about ±30 objective, ±3–4 falls,
of chaotic noise). Ask before committing; don't push.

| # | Step | Size | Done when |
|---|---|---|---|
| E2.0 | **Prep.** Branch off `main`. Register the D2 tier (`RobustnessSettings`, `compare_controllers.py --tier`, which today has smoke/dev/validation/fall_rate only). Benchmark float64 throughput at D1's seed count: one PID search with `--cma-iters 2 --de-iters 2`, and L with `--stop-after 5` | S | Wall-clock numbers fill in section 5 |
| E2.1 | **PID searches, 10 run seeds.** `python optimization/run_batch_tuning.py --robot legobot --backend torch --no-show --seeds 150 --seed k --cma-x0 midpoint`, k = 0–9 (the reference procedure: DE plus CMA-ES). Also 10 seeds with `--cma-x0 model-based --de-iters 2 --de-pop 6`, as in E1 | M (compute) | 20 runs under `results/legobot/batch_tuning/` |
| E2.2 | **Pick the finalists on dev** (5000–5049, 60 s, float64): each run's `cma-es (mean)` and best, chosen by objective. **Confirm on the D2 slice** with model-based alongside, using D3's rule | S | A table in `E2_NOTES.md`; the reference chosen by D3 |
| E2.3 | **Freeze the reference.** Write it to `results/legobot/reference_pid/comparison.json` in `read_pid_comparison`'s layout, and set `LEARNING.reference_pid_file` and `reference_pid_entry`. If it's model-based, the same file names the model-based gains, so `reference_is_provisional` turns False either way | S | `compare_controllers.py` stops labelling LEGO ratios provisional |
| E2.4 | **Re-measure the scales:** `python hardware/legobot/legobot_learning_scales.py`, then update `LEARNING` and its table. **Re-run gate (b)** (L-PID against the frozen reference). A reference that never falls makes the plain gate readable again (ml-baselines-plan 13) | S | Scales updated; gate within 1% where both stay up |
| E2.5 | **L development** (1–2 run seeds): `ROBOT=legobot python optimization/run_neuroevolution.py --method linear --profile full --float64 --train-seeds 150 --run-seed 0`. Watch the dev curve. Check the float32-vs-float64 gap on one run, to confirm D6 | S–M | One full L run's dev score against the reference and model-based |
| E2.6 | **L full:** run seeds 0–9 with the same command. Choose the run by float64 dev objective, never by validation (fairness item 7) | M (compute) | 10 runs; the selected run named |
| E2.7 | **Export and deploy:** `python optimization/export_policy.py <selected run> --name l_e2` (golden vectors and envelope included). Re-export the synthetic bundles under the new scales and reference: `--synthetic l_pid`, `a_warm_zero`, `b1_zero`. Then `deploy.sh` | S | Bundles committed; the Pi's self-test and config check pass |
| E2.8 | **Final, once** (D5's controllers, `--final`): `compare_controllers.py --tier validation`, then `--tier fall_rate`, then `--latency-sweep`. Report every L run's validation score, not only the selected one's. No per-scenario inspection before this | S | Headline LEGO numbers |
| E2.9 | **Record** commands, budgets, wall clock and results in `hardware/legobot/E2_NOTES.md`. Update `SOFTWARE_PLAN.md`'s E2 row and `ml-baselines-plan.md` section 13 | S | — |

## 5. Compute (laptop, RTX 5070 Ti, float64; replace with E2.0's benchmark)

| Job | Basis | Estimate |
|---|---|---|
| One PID search, 450 scenarios | E1: 21 min at 150 scenarios, at a 33 W GPU limit | about 1 h |
| E2.1, 20 searches | | about 13–20 h (overnight runs, or the cluster) |
| One L run, full profile | 4.2 s/gen in float32 at 150 scenarios (ml-baselines-plan 13), × ~2.7 for float64, × 3 for 450 scenarios. The 1.5 G robot-second cap bounds it at about 650 generations | about 4–6 h to the cap; less if the dev score stagnates |
| E2.6, 10 L runs | | about 40–60 h: the cluster, or 3 seeds on the laptop (D6) |
| E2.8 | 300 + 3,000 scenarios × 60 s × 4–5 controllers, NumPy float64 | a few hours |

## 6. Hardware track, in parallel

These don't wait on E2:

- **P7 steps 1–3 now**, with the current `l_pid` bundle (hand-tuned gains): self-test and timing on the Pi
  (P3a/P3b), a `--no-drive` run replayed on the laptop (P4b), stand checks, then L-PID on the floor
  against the hand-tuned PID. This proves the robot runs the policy pipeline correctly before any trained
  policy exists.
- **P3d's measured read-to-command latency** feeds D4. Measure it before E2.5, so L trains on the right
  latency range.

After E2.7:

- Re-run the step-3 comparison with the re-exported `l_pid` (now the frozen reference's L-PID).
- **P7 step 4:** `l_e2` on the floor (stand still, small pushes, then the scenario), each run scored with
  `score_run.py` (P8) and replayed with `replay_policy.py`.
- E3 (sim-to-real) uses the frozen reference PID and `l_e2`.

```
E2.0 → E2.1 → E2.2 → E2.3 → E2.4 → E2.5 → E2.6 → E2.7 → E2.8 → E2.9
                                  ↑                    ↓
P7 steps 1–3 (now) ── P3d latency ┘           P7 step 4, E3
```
