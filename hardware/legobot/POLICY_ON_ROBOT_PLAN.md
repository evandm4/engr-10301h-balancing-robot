# Plan: running the learned policies on the LEGO robot

What the robot code (`hardware/legobot/robot/`) needs before it can run the policies from
`docs/ml-baselines-plan.md` (L, A, A-warm, B1) on hardware. Written 2026-11-24, before any LEGO
policy has been trained beyond a 30-generation quick L run. Follows SOFTWARE_PLAN.md's phase D and
feeds its E3/E4.

## 1. Where things stand

- **The loop already has the right seam.** `Brain.act` calls `controller.update(state, dt)` with
  `state = [x, v, theta, theta_dot]` in the simulator's conventions, and the loop calls `reset()` at
  arming and only updates while armed (`balance.py` run). `balance_sim.PolicyController` has the same
  interface, its own integrators (leaky, freeze on saturation) and command history, and is the
  reference implementation the batched engines are tested against bit for bit. So the robot can run
  **the reference class itself**, not a port, as it already does for the PID. That is the main
  design decision: no re-implementation of the observation vector, so it can't drift from training.
- **Inputs are already identical to the PID's:** the complementary filter's pitch, the bias-corrected
  gyro, encoder velocity r(ω̄ + gyro), and the goal. The policy adds only internal state.
- **Cost per tick (measured on the laptop today, reference class):** L 5 µs, A-sized MLP (481
  params) 44 µs, B1 (513) 55 µs. Pure Python on the Pi 3B is roughly 10–20× slower, so expect about
  0.5–1 ms for an MLP, against 0.06 ms for today's controller and the loop's 1.2 ms
  `COMPUTE_LEAD_S`. That fits the 5 ms tick, but it adds latency between the IMU read and the
  command, and the lead decides where the write lands (section 3, P3). Must be measured on the Pi.
- **What's missing:** loading a policy file in `control.build_controller`, getting it and its
  training config onto the Pi, checking the deployed config matches the one it was trained under,
  timing, logging the policy's internals, and tests.

## 2. Problems found while planning

1. **`--goal still` would be wrong for B1.** `build_controller` sets `controller.goal = 0.0` after
   building. For a residual policy that changes the policy's goal but not its base PID's
   (`PolicyController.base` keeps the scenario profile), so the PID part would track the scenario
   while the network sees 0. Fix: pass the goal when building, not after.
2. **The `.npz` doesn't say which config it was trained under.** It holds the spec and weights;
   the run's `snapshot.json` holds the config. The control period, filter τ, actuator model,
   feedforward gain and torque cap must match the deployed `legobot_config.py`, or the robot runs a
   policy in a loop it never saw. `ProjectConfig.scoring_differences` already does this comparison
   for `compare_controllers.py`.
3. **Training scales will change.** `legobot_config.LEARNING`'s scales are provisional until E2's
   tuned PID (ml-baselines-plan 13). A policy carries its own scales in the `.npz`, so it stays
   self-consistent, but the config check above will flag a policy trained before the re-measure.
4. **Policies have no lean cap.** The cascaded PID caps its pitch setpoint at 25°; L, A and A-warm
   have no such limit (B1 keeps its base PID's). The 0.70 rad tilt trip still applies. Worth
   knowing on the first floor runs, not a blocker.
5. **Bit-identity across machines isn't guaranteed.** NumPy's `tanh` on the Pi (aarch64) can differ
   from the laptop's (x86 SIMD) in the last bit. Checks between the two use a tolerance (1e-12
   N m), not equality.

## 3. The plan

Sizes: S under a day, M a few days. P1–P6 need no trained LEGO policy: they are built and tested with
synthetic policies whose behaviour is known exactly (PID-equivalent L; A-warm or B1 with a zero
output layer, which equal L-PID and the PID) and the existing quick-run L.

### P1. Policy bundle (laptop side)

| # | Task | Size | Done when |
|---|---|---|---|
| P1a | `optimization/export_policy.py <run folder> [--name]`: writes `results/legobot/policies/<name>/` with `final.npz`, the run's `snapshot.json` and `summary.json`, and a `manifest.json` (kind, n_params, multiply-adds per tick, sha256 of the `.npz`, source run path, git commit, the run's dev score) | S | A bundle for the quick L run and for synthetic policies |
| P1b | **Golden vectors** in the bundle (`golden.npz`): one dev-seed scenario through the reference simulator with this policy, recording each tick's state given to `update()` and the torque returned (about 400 ticks: the 0.30 rad start, a push, the cruise) | S | Re-running the vectors through a fresh `PolicyController` on the laptop matches exactly |
| P1c | **Input envelope** (`envelope.json`): per raw observation, the 0.1/99.9 percentiles and max |value| over the policy's own 60 s dev runs (reference engine, a handful of dev seeds × 3 tilts) | S | Envelope present; used by P4 and P5 |

Bundles are a few kB, so commit them under `results/legobot/policies/` (results are tracked; the
paper needs to name exactly what ran).

### P2. Loading on the robot

| # | Task | Size | Done when |
|---|---|---|---|
| P2a | `control.load_policy_bundle(path)`: a bundle folder or a bare `.npz` (bare only with `--allow-unchecked`). Uses `balance_sim.load_policy`, checks the sha256 against the manifest, refuses a spec kind it doesn't know (a future GRU) | S | Corrupt, wrong-observation-list and unknown-kind files are refused with a clear message |
| P2b | **Config check:** `cfg.scoring_differences(snapshot["config"])` against the deployed `legobot_config`; any difference refuses to start unless `--allow-config-mismatch`, which is logged in the summary. Also refuse if the `.npz`'s `max_torque` differs from the config's | S | Test: a snapshot with a different `control_dt_s` or feedforward gain is refused |
| P2c | `build_controller`: `--controller` takes a bundle path or `policy:<name>` (looked up in `~/legobot/policies/`). Builds with `cfg.build_policy_controller`, with the goal passed in (problem 1). The feedforward gain comes from the snapshot's config; `--ff-gain` is refused for policies (it would change the plant the policy was trained on) | S | `--goal still` gives goal 0 to both the network and B1's base PID |
| P2d | **Startup self-test:** run the bundle's golden vectors through the controller before gyro calibration; refuse to arm if any torque differs by more than 1e-9 N m. Reset the controller afterwards. Also times the forward pass (P3) | S | Self-test passes on the Pi; deliberately perturbed weights fail it |
| P2e | `deploy.sh` also copies `results/legobot/policies/` to `~/legobot/policies/`. Every log's meta records the bundle name, sha256 and kind next to `DEPLOYED` | S | A log names exactly which policy ran |

### P3. Timing on the Pi

| # | Task | Size | Done when |
|---|---|---|---|
| P3a | Measure the per-tick cost of L, an A-sized MLP and B1 on the Pi (the self-test's timing, plus a 10 s `--no-drive` run each at SCHED_FIFO): mean, p99, worst, and IMU-read-to-write | S | Numbers recorded in DATASHEET 6 |
| P3b | **Compute lead per controller.** `COMPUTE_LEAD_S` (1.2 ms) places the tick start so the write lands at `LINE_WRITE_PHASE_S`. Set the lead from the self-test: the PID's measured lead plus the policy's p99 forward time over the PID's | S | Logged `cmd_phase` stays at the target (sd < 0.1 ms) with each policy |
| P3c | **Only if the MLP's p99 is over about 1 ms:** speed up `PolicyController.forward` without changing its arithmetic: weights converted to Python float lists once at construction (the laptop test today: 44 → 18 µs), same summation order, `np.tanh` kept. It lives in `balance_sim`, so run both robots' test suites, including `test_policy.py`, `test_equivalence.py` and `test_torch.py` | S | Bit-identical torques; equivalence tests pass |
| P3d | Check the measured IMU-read-to-command latency against the simulator's latency range (7.5–15 ms). An extra ~1 ms is inside it; if a policy pushes the total outside it, that goes into the paper as a hardware condition the policy wasn't trained on | S | Recorded |

### P4. Logging and replay

| # | Task | Size | Done when |
|---|---|---|---|
| P4a | For policy runs, extra log columns: the raw observation vector (11, or 13 for B1), pre-clip output, saturated flag, B1's residual torque | S | Columns present; `animate.py --csv` still replays the run |
| P4b | `robot/replay_policy.py <log>` (runs on the laptop): feeds the logged `state` rows through a fresh controller from the same bundle and compares torques (tolerance 1e-12). Proves after the fact that the robot ran what was trained, tick by tick | S | A sim_hw run replays exactly; a real run replays within tolerance |
| P4c | Summary lines: fraction of ticks with each input outside its envelope (P1c), with the worst excursion; fraction of ticks saturated or integrators frozen | S | Shown in `_summary.txt` |

### P5. Safety additions

All of D3's trips stay as they are. New:

| # | Task | Size | Done when |
|---|---|---|---|
| P5a | **Out-of-envelope monitor:** log by default (P4c). Optional trip, `--envelope-trip k`: coast if any input stays past k × its envelope's max |value| for 0.1 s. Off by default until the first floor runs show how often real inputs leave the envelope | S | Trip tested with `--inject` (a new `ood` case that adds an offset to one input) |
| P5b | Bench runs on the stand use `--bench` (duty cap 0.3), as the PID's did | — | — |

### P6. Tests (laptop, `test_control.py`; both robots' suites)

- **L-PID through the robot code ≈ the PID through the robot code:** closed loop with `sim_hw`
  (`--sim --sim-noise`), PID-equivalent weights of the hand-tuned gains. Close, not equal (leaky,
  frozen integrators against the PID's anti-windup; ml-baselines-plan 13's gate (b)).
- **B1 with a zero output layer through the robot code equals the PID through the robot code**, bit
  for bit (same seeds), with `--goal still` and `--goal scenario`. This one catches problem 1.
- **A-warm with a zero output layer equals L** through the robot code, bit for bit.
- Closed loop balances and tracks the scenario with the quick-run L or a synthetic L, as the existing
  `test_closed_loop_*` tests do for the PID.
- Bundle round trip; config-mismatch refusal; self-test failure on perturbed weights; envelope trip.
- `test_every_robot_module_compiles` covers the new modules.

### P7. Hardware bring-up (needs P1–P5 and a deployed bundle)

Order, each step before the next:

1. **On the Pi, no drive:** self-test, timing (P3a), a `--no-drive` run held by hand, replayed on
   the laptop (P4b).
2. **On the stand** (`--bench`): signs and saturation behave like the PID's (on a fixed stand
   every velocity loop runs to the cap; this checks wiring, not balance).
3. **L-PID on the floor first.** It should behave like the hand-tuned PID. If it does, the input
   pipeline is right; if not, the problem is in the robot code, not in learning. This is the
   hardware version of the gate test.
4. **Then trained policies:** L, then A / A-warm, then B1. For each: stand still, small pushes, then
   the scenario (E4's progression).

### P8. Comparison runs (E4 hooks; small)

- `--label` per controller; a run sheet that alternates controllers in a fixed order within one
  battery charge, so battery level isn't confounded with the controller.
- Score real runs with the simulator's own cost code (`balance_sim.metrics`) from the run CSV, so
  hardware numbers use the same cost as the paper's simulation numbers. If E3 builds this scorer for
  the PID, reuse it.

## 4. Order and dependencies

```
P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8
                         ↑
E2 (tuned PID) → re-measure scales → train L, A, A-warm, B1 on LEGO → export (P1) → P7 step 4
```

- P1–P6 can start now: everything is tested with synthetic policies and the quick L run.
- P7 steps 1–3 need only L-PID, so they can run before any real LEGO policy exists.
- P7 step 4 waits on trained LEGO policies, which wait on E2 (ml-baselines-plan 13).

## 5. Decisions recommended, not yet made

| Decision | Recommendation | Reason |
|---|---|---|
| Port the policy or import it | Import `balance_sim.PolicyController` | Same reasoning as the PID (SOFTWARE_PLAN 1, item 4): one implementation, no drift; the golden vectors prove it on the Pi |
| Speed up the forward pass | Only if the Pi's p99 is over ~1 ms (P3c) | The reference class probably fits; changing shared simulator code costs a full equivalence re-check |
| Out-of-envelope trip | Log only at first; turn on if the floor runs show excursions before falls | A trip that fires on normal handling would block testing |
| Commit bundles | Yes, under `results/legobot/policies/` | A few kB each; the paper must name what ran |
| Config mismatch | Refuse by default, override flag logged | A policy run under a different loop than it was trained in isn't a fair test |
| GRU / PPO policies (later phases) | Out of scope; the loader refuses unknown kinds | They need hidden state the reference class doesn't have yet |
