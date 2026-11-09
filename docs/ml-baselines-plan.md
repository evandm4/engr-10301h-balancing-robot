# Implementation plan: learned-controller baselines against the ML-tuned cascaded PID

Status: REVISION 4, after three adversarial reviews. Phase 0 is built; L is next.

Phase 0 progress (2026-11-09): built and tested; only the throughput numbers below remain to be
folded into budgets.
- Policies: `balance_sim/policy.py` (reference) and `batch_sim/policy.py` (batched) for L, A, A-warm
  and B1 (residual: `residual_gains` in the spec; zero output is the reference PID bit for bit, one
  robot, batched and under robust scoring). NumPy matches the reference bit for bit; the GPU matches
  NumPy to about 2e-15 in float64 (full 20 s runs, with the baddbmm forward pass). Gate (a) passes at rtol 1e-9; gate (b) measures +0.68% (training)
  and -0.66% (60 s dev).
- Residual input scales measured on 15 dev scenarios: PID setpoint peaks at 0.44 rad (scale 0.3), the
  PID's pitch integral at 0.07 rad s (scale 0.05).
- `stability.LinearController` takes several leaky integrators (merged when they share a leak, so
  unused or redundant ones add no mode); `LinearController.linear_policy` is L's exact linearization.
  The recorded PID and LQR margins reproduce exactly. L-PID: spectral radius 0.9947 (PID 0.9951),
  stable from 0 to 37.25 ms (PID: the same).
- `PolicySearch` (resumable CMA-ES; ES state and NumPy's RNG saved), `run_neuroevolution.py` (quick
  and full profiles, log-spaced checkpoints plus the PID's 58.3 M budget, float64 dev scores at 60 s
  and 20 s, stagnation stop, float32-vs-float64 check, resume, `scripts/slurm_neuroevolution.sh`).
  A resumed run matches an uninterrupted one exactly.
- `compare_controllers.py`: tiers with frozen ones behind `--final`, per-scenario costs and fall
  times, crossed bootstrap, order-statistic interval, Clopper-Pearson, McNemar, the latency sweep
  with the 4-fall rule. `run_lqr_comparison.py` is unchanged (it does not call the shared code yet).
- TF32: cuDNN's turned off and matmul's checked in `TorchSimulator`.
- Benchmark (`benchmark_policies.py`, RTX 5070 Ti laptop on AC, float32, full-length runs). With the
  first GPU forward pass (elementwise, about 600 small kernels per tick for A;
  `results/benchmarks/policies_20261109-130318.json`): A-sized MLP at population 1,332, 32 s per
  generation; B1 39 s. Switched the GPU path to batched matrix products (`baddbmm`, as 3.1 allows;
  NumPy keeps the fixed-order loop and its bit-identity with the reference; GPU float64 still matches
  NumPy at rtol 1e-9). Now (`results/benchmarks/policies_20261109-131647.json`):

  | Policy | Population | s per generation | hours for the full cap |
  |---|---|---|---|
  | L | 256 | 3.4 (the GPU's per-batch floor) | 0.9 (1.5 G robot-seconds) |
  | A-sized (A-warm, 492 params) | 1,332 | 16.4 | 3.4 (6 G) |
  | B1 | 1,332 | 21.0 | 4.4 (6 G) |

  max_batch 400k is no better (17.7 and 19.7 s), so the default stays at 200k. 10 runs each of A,
  A-warm and B1 to the full cap: about 110 GPU-hours, less where dev stagnation stops runs early.
  The 4.2 estimate (11 to 15 s) holds within about 10%.

Dates: drafted 2026-11-08, revised 2026-11-09 (three times). Branch: `claude/ml-alternatives-cma-es-pid-773b0d`.

**Current implementation scope (decided 2026-11-09): everything up to PPO.** That is Phase 0 on
the CMA-ES path, the linear policy (L), neuroevolution (A, and A-warm), the residual policy with
CMA-ES (B1), 10 re-runs of the reference PID, and the final comparison on the validation and
fall-rate tiers. PPO (C), the residual with PPO (B2), the GRU (E) and differentiable simulation (D)
stay designed here for a later phase but are not built now.

## 0. Change log

### Revision 4 (third review: blocker B1, major M-a to M-d, minor m-a to m-g)

The third review simulated the gate test instead of reasoning about it (scripts were in the
session scratchpad). Its measurements are quoted where they set a value.

| Change | Section | Cause |
|---|---|---|
| Integrators freeze while the previous command was saturated (anti-windup); T_leak fixed at 10 s in advance | 3.1 | B1 |
| Gate split: exactness against a "PID with L's integrators" reference, plus a disclosed gap to the real PID with a bound stated in advance | 3.7 | B1, Q2 |
| PID-equivalent L is its own row in the results; learned policies are read against it | 1, 8 | M-a |
| Rationale for the leak corrected: 60 s runs push raw integrals out of the training range (pitch-bias drift), not friction in 6 s | 3.1 | M-b |
| Bootstrap with scenarios crossed with runs; order-statistic interval for the median of 10 runs; per-run McNemar; dev score defined | 3.4 | M-c |
| Latency sweep: 2.5 ms grid, 15-45 ms; a count rule; named scenario subset; per-run metric | 3.4 | M-d |
| Generation is 8 M robot-seconds; cap about 750 generations | 2.6, 4.2 | m-a |
| Full vs sep-CMA threshold explicit at 600 parameters; compared once | 3.3 | m-b |
| L uses population 256; one captured graph per candidate count; whole-candidate padding replaces the 1,024-row rounding | 3.2, 4.1 | m-c |
| A-warm: per-block initial step sizes, scored start point, L's budget charged | 4.3, 5 | m-d |
| L drops `goal - v_meas` (linearly dependent inputs) | 3.1 | m-e |
| Linearization reports the spectral radius over the coupled modes only | 3.5 | m-f |
| "Twice as many scenario runs"; PID re-run seeds named, seed 0 as a reproducibility check | 2.6, 2.7 | m-g |

### Revision 3 (second review: M1-M4 major, m1-m13 minor)

| Change | Section | Cause |
|---|---|---|
| Integrator inputs redesigned (leaky); gate test that L with PID-equivalent weights matches the PID | 3.1, 3.7 | M1 |
| A-warm variant: L plus a zero-initialized MLP | 4.3 | M1, Q4 |
| PPO env captured as CUDA graphs, staggered rollout groups, robot-second cap, re-derived hours | 6 | M2, Q2 |
| Latency sweep moved to the fall-rate tier with intervals and a defined metric | 3.4 | M3 |
| Scope cut to everything up to PPO; gates instead of an open-ended list | top, 10 | M4, decision 2026-11-09 |
| Log-spaced checkpoints up to each method's cap; budget ratio corrected to about 100x | 2.6, 1 | m1 |
| Neuroevolution time re-estimated (about 3 h per run); population sized to whole chunks | 4.2 | m2 |
| Dev also scored at 20 s, for the overfitting gap only | 2.5, 4.2 | m3 |
| PPO exploration noise defined in N m and capped | 6.3 | m4 |
| Linearization: exact for L; MLPs at the state actually reached; zero-slope integrator modes dropped | 3.5 | m5 |
| Seed mechanics: smoke-in-dev exception, fixed-buffer fresh-seed path, one fresh-seed protocol | 2.5, 3.3 | m6 |
| "Run seeds" vs "noise seeds" naming | 2 | m7 |
| Existing controllers' falls traced on dev or the retired 1000-1099 set | 2.5 | m8 |
| Reproducibility across machines: one platform per method's final runs, one validation machine, resume test | 9 | m9 |
| TF32 off for cuDNN too | 3.2 | m10 |
| Reward equivalence at rtol 1e-12, not bit for bit | 6.1 | m11 |
| Two-level bootstrap; named headline estimator; chaos twins disclosed in the ES budget | 3.4, 2.6 | m12 |
| GRU gate equations and truncated sequences | 7 | m13 |
| Robust control (H-infinity, mu-synthesis) named as the classical alternative | 1.1 | discussion 2026-11-09 |

### Revision 2 (first review: R1-R20, plus decisions of 2026-11-09)

Reference PID and its 58.3 M robot-second budget (R1); generalized `LinearController` (R2); the
evaluator's policy path (R3); hard-clip output (R4); linear control (R5); fall-rate tier (R6);
synchronized PPO episodes (R7, R8); reward normalization, privileged critic, low noise (R9); budget
curves (R10, R13); neuroevolution budgets (R11); overfitting checks (R12); test tolerances (R14);
gamma horizon, observation scales, seed overlap, latency-sweep path, fall-time logging, tuning-effort
disclosure (R15-R20); last 4 commands as inputs; 10 run seeds; GRU back in the plan; one pipeline
for laptop and cluster.

## 1. Why, and the question being answered

Every controller in the project so far is a fixed control law (cascaded PID, LQR) whose handful of
gains or weights an optimizer picks. That is black-box optimization, not machine learning in the
sense a reader expects. This plan adds controllers where a neural network is trained:

| # | Method | Keeps PID? | What is trained | Trained by | Phase |
|---|---|---|---|---|---|
| L-PID | PID-equivalent linear policy (reference row) | Its weights are the PID's | Nothing: weights computed from the reference PID's gains | none | now |
| L | Linear policy (control) | No | 10 weights: observations to torque | CMA-ES | now |
| A | Neuroevolution | No | MLP: observations to torque | CMA-ES | now |
| A-warm | Neuroevolution from L | No | L plus a zero-initialized MLP | CMA-ES | now |
| B1 | Residual policy | Yes (frozen reference PID) | MLP correction added to the PID torque | CMA-ES | now |
| C | Deep RL | No | MLP actor-critic | PPO | later |
| B2 | Residual policy | Yes | as B1 | PPO | later |
| E | Recurrent policy | No | GRU actor-critic | PPO | post-semester |
| D | Differentiable simulation | Either | PID gains, or the MLP from A | Backprop through the simulator | post-semester |

Imitation learning (copying LQR or PID) is deliberately left out. It can only approach its teacher
on the teacher's cost, and the residual policy covers the one thing imitation would add, which is
starting from a competent controller.

Research question, consistent with the paper's thesis (`paper/results-planning-notes.md`): does a
learned controller beat the ML-tuned PID on robustness (fall rate on held-out robots, latency
tolerance, sensitivity) and on typical cost (median per-scenario ratio), and at what cost in
compute, interpretability and analyzability?

How the linear control reads (the key to interpreting A). The comparisons are made against
**L-PID**, not the PID itself: the policies' integrators (leaky, with freeze-on-saturation anti-windup)
differ from the PID's, and that change alone moves the cost (the third review measured +0.68% on
training and -0.66% on 60 s dev at T_leak = 10 s). L-PID vs PID isolates the integrator change; L vs
L-PID isolates search and extra inputs.

| Result | Meaning |
|---|---|
| L about equal to L-PID, A worse | The optimizer is the bottleneck, not representation |
| A or A-warm better than L | Nonlinearity helps on this task |
| L better than L-PID | The extra inputs (command history) or a better linear law help, even without a network |
| All about equal to L-PID | The PID structure is near the best this information allows |

Every outcome is reportable. "A network trained with about 100x the simulation budget does not beat
4 well-tuned gains" is a legitimate result for this plant.

### 1.1 The classical alternative not built here

The thesis is about robustness to uncertain robot parameters and latency. Classical control's own
answer to that is **robust control**: H-infinity design (minimize the worst-case amplification
from disturbances to errors across all frequencies) and mu-synthesis (guarantee stability and
performance for a stated range of parameter uncertainty, such as body mass +-15%). It is not in
this plan, but the paper names it in related work and limitations. An H-infinity baseline is a
candidate for a later phase (python-control has `hinfsyn`, which needs `slycot`).

### 1.2 Naming

- **Noise seeds** pick a scenario's sensor noise and randomized robot (training 1-50, and the tiers
  in 2.5).
- **Run seeds** pick an optimizer's initialization and sampling (10 per method).

Config fields, CLI flags and the paper use these two names and nothing else.

## 2. The fairness contract (applies to every method)

Anything that breaks one of these is disclosed in the paper as an ablation, never as the main
comparison.

1. **Same plant and task.** The same `batch_sim` physics, actuator model, friction, pushes, goal
   profile, 5 ms controller tick, 2.5 ms physics step, torque limit and fall angle. Training on the
   20 s scenario, validation on the 60 s one.
2. **Same uncertainty.** Robust scoring as in `RobustnessConfig`: 50 training noise seeds x 3 start
   tilts, one randomized robot and one random latency (5 to 20 ms) per scenario. The deployed
   controller is not told its robot or latency. (A PPO critic may see them, 6.3, because it is never
   deployed.)
3. **Same information.** The policy sees what the PID sees: measured velocity, pitch and pitch rate
   (same noise and bias tables) and the current goal velocity. No true state, no wheel position, no
   preview of future goals. It may also see what the controller computes itself: its error
   integrals and its own last 4 torque commands (the LQR already uses commands in flight). The
   observation list is fixed in `config.py` and recorded in every run's snapshot.
4. **Same score.** The existing cost (`pitch_reference = "upright"`, Bryson weights, fall penalty
   with the remaining-time term). Training objectives may differ (PPO's normalized reward), but every
   method is scored by the same code on the same cost.
5. **Evaluation tiers** (also saved as a project memory so later sessions follow them):

   | Tier | Noise seeds (x 3 tilts) | Run length | Use |
   |---|---|---|---|
   | Smoke | 5000-5009 (30; the first 10 dev seeds) | 20 s | unit tests, quick checks after each change |
   | Development | 5000-5049 (150) | 60 s, plus 20 s for the overfitting gap only | every hyperparameter, architecture and run-selection decision |
   | Validation | 2000-2099 (300), frozen | 60 s | headline typical-cost numbers, once, on final controllers |
   | Fall-rate | 6000-6999 (3,000), frozen | 60 s | fall rates and the latency sweep, once, at the end |

   - Training stays 1-50. Retired or reserved, never reused: 100-109, 1000-1099, and >= 1,000,000
     (fresh seeds for the fresh-scenario ablations).
   - Smoke inside dev is the one intended overlap; every other pair of sets must be disjoint, and the
     overlap check enforces exactly that.
   - Small tiers can't detect fall-rate changes, so no fall-robustness conclusion comes from smoke or
     dev.
   - Nobody inspects per-scenario validation or fall-rate results until every controller in the
     comparison is frozen. The open item in the paper notes (tracing why the existing controllers
     fall) is done on dev or on the retired 1000-1099 set instead.
   - The fall-rate tier is also run on the existing PID and LQR controllers.
6. **Compute reported as curves.** The unit is robot-seconds of simulation, counted as nominal
   scenario-seconds attempted (what the evaluator charges: it runs the full length unless the whole
   batch has fallen). Evolution-strategy budgets include the 150 chaos-penalty twin runs per
   candidate, which PPO doesn't need; the paper says so, because per robot-second PPO gets twice as
   many scenario runs. Dev evaluations are reported separately and not charged.
   - Checkpoints, log-spaced up to each method's cap: 1, 3, 10, 30, 100, 300 M, 1, 3, 6 G
     robot-seconds and the final budget. One A generation (1,332 x 300 x 20 s) is already about
     8 M, so A's curve effectively starts at its first generation.
   - Also scored at the reference PID's budget: CMA-ES (4,800 evaluations) plus DE (4,920),
     9,720 x 300 simulations x 20 s = **58.3 M robot-seconds** (28.8 M for CMA-ES alone).
7. **Repeated training: 10 run seeds per method**, including 10 re-runs of the reference PID's exact
   procedure (CMA-ES plus DE, best by training objective), with run seeds 0-9. Run seed 0 on the
   reference platform should reproduce the reference gains (the reference run was recorded with
   `git_dirty: true`), which doubles as a reproducibility check. Every run's validation score is
   reported, and runs are chosen by float64 dev score (the objective, chaos term included, matching
   training), never by validation. During development, one or two run
   seeds and quick settings (section 9).
8. **Same validation precision.** Final validation in float64 NumPy, on one designated machine
   (section 9). Networks trained in float32 are exported, re-scored in float64 on training and dev
   scenarios, and flagged if the two disagree by more than about 1%.
9. **Disclosed tuning effort.** The networks get dev-set searches (size, sigma0, population, ...);
   the PID got fixed settings (population 120, CMA-ES plus DE). The paper reports the number of dev
   evaluations per method.

## 3. Shared infrastructure (Phase 0)

### 3.1 The policy as a controller

- `simulation/balance_sim/controllers.py`: `NeuralPolicyController` (one robot, NumPy), the reference
  implementation, as `CascadedPIDController` is today. `LinearPolicyController` is the same class
  with no hidden layers. `ResidualPolicyController` wraps a `CascadedPIDController` and a
  `NeuralPolicyController`.
- `simulation/batch_sim/policy.py`: pure functions through the `xp` shim, like
  `batch_sim/controller.py`.
- **Forward pass in a fixed order:** an explicit loop of elementwise multiply-adds over the inputs,
  not `@`, shared by `balance_sim` and NumPy `batch_sim`. That restores bit-identity between those
  two and keeps BLAS (whose summation order varies by library, thread count and CPU) out of the
  float64 validation. The GPU path uses `torch.baddbmm` and is tested by tolerance.
- **Output: a hard clip**, `clip(scale * z, -max_torque, max_torque)`, so the network can command
  exactly full torque. The push and the 0.30 rad start tilt are sized at the edge of what full
  torque can catch. The residual uses `clip(pid + authority * max_torque * tanh(z), +-max_torque)`.
- **Observation vector, 11 inputs**, each divided by a fixed scale from `config.py` (no running
  normalization):
  `[v_meas, theta_meas, theta_dot_meas, goal, goal - v_meas, I_v, I_theta, last 4 torque commands]`.
  - **The integrators are leaky and freeze on saturation:**
    `I <- I + dt * (e - I / T_leak)`, skipped on any tick where the previous tick's pre-clip output
    exceeded max_torque in magnitude (conditional-integration anti-windup; it needs only one forward
    pass per tick and works for any policy). **T_leak = 10 s, fixed in advance**, never selected on
    any policy's dev performance. A wide safety clip remains, and the fraction of ticks each
    integrator is clipped or frozen is logged.
  - Why the leak: with raw integrals and the reference PID's closed loop, the velocity integral
    reaches at most 0.43 m in 20 s training but 1.01 m by 60 s, growing about 0.015 m/s at
    standstill (the 0.01 rad pitch bias divided by kv). A network would see up to 2.3x its training
    range in validation. With T_leak = 10 s and the freeze, the maximum is 0.28 m in both (third
    review, measured).
  - Why the freeze: the PID's own integral has anti-windup (`limited_integral`), and the 0.30 rad
    start tilt saturates the torque in every scenario. Without anti-windup, L with the PID-equivalent
    weights misses the PID by 1.7-5% (by T_leak); with it, by +0.68% (training) and -0.66% (60 s
    dev) at T_leak = 10 s.
  - The PID integrates `theta_meas - setpoint`. When nothing saturates, that equals
    `ki * I_theta - ki * kv * I_v` in these inputs, and leaky integration is linear, so a shared T_leak
    keeps the combination exact. That is why L can contain "the PID with L's integrator semantics"
    exactly (gate test (a), 3.7), and the real PID approximately (gate test (b)).
  - Scales: the Bryson limits in `CostConfig` for velocity (0.2 m/s), pitch (0.3 rad) and torque
    (0.1 N m); for pitch rate (which has no Bryson limit) the peak rate of the PID's recovery from
    the 0.30 rad start, computed once and written to `config.py`; integrator scales from their
    steady-state ranges at T_leak.
  - Timing, matched to the PID: integrators update before use. "Torque commands" are the clipped
    commands sent to the latency queue, newest first, zero before the first tick. The last 4 cover
    the 20 ms maximum training latency.
- Sizes: `hidden = (16, 16)` by default: 11x16 + 16 = 192, 16x16 + 16 = 272, 16 + 1 = 17, so **481
  parameters**. A `(32, 32)` variant has 1,473.
- **L drops `goal - v_meas`**, because together with `v_meas` and `goal` it is linearly dependent
  (CMA-ES would drift along the null direction, growing weights that cancel). L has 10 inputs, no
  bias (the controller has no reason for a constant torque, and the PID has none): **10 weights**.
  The MLPs keep all 11 inputs; their nonlinearity makes the redundant input useful as a feature.
- Residual inputs: the 11 above plus the PID's setpoint and its pitch integral (13), so B1 has
  13x16 + 16 + 272 + 17 = **513 parameters**.
- Each saved policy (`.npz`) records its weights, observation list, scales, T_leak, max_torque,
  output rule, and (residual) the base PID gains. Actor weights only.

### 3.2 Batched simulation of many different networks

- `SimSpec.controller` gains `"policy"` and `"residual"`.
- Weights live once per candidate, never once per simulation: a `(P, ...)` array in NumPy, a fixed
  `(P_max, ...)` buffer on the GPU. The batch is laid out as `(P, S)` with S = n_total = 300 (150
  scenarios plus their 150 twins). Shapes stay fixed, so CUDA-graph capture still works.
- Padding: for policies, the existing rounding of torch batches up to a multiple of 1,024 rows is
  replaced by padding with whole dummy candidates. Each distinct candidate count gets its own
  captured graph (the simulator cache holds 3). Populations are chosen as whole multiples of the
  per-chunk candidate count (`max_batch // n_total`: 666 at the default 200k, so 1,332 is two
  chunks), and Phase 0 benchmarks `max_batch` = 400k, where 1,332 candidates fit in one chunk and
  BATCHED.md shows 5.3 rather than 8.1 microseconds per simulation.
- Both latency paths are supported: per-simulation delay (robust scoring) and the global
  `slot_per_step` path (`latency_range=None`), which the latency sweep uses.
- TF32 off everywhere it applies: assert `torch.get_float32_matmul_precision() == "highest"`,
  `torch.backends.cuda.matmul.allow_tf32 is False`, and set
  `torch.backends.cudnn.allow_tf32 = False` (it defaults to True in the project's torch 2.11).

### 3.3 Evaluator and search

- `BatchEvaluator` gets a policy path: `evaluate_policies(params (P, n_params), seeds)` sends the
  `(P_chunk, n_params)` block plus S to `_simulate_chunk`, bypassing `decode_units` and the unit cube.
  (The unit-cube clip is correct for PID gains, which are bounded and positive. Network weights are
  unbounded and signed, so they need their own path.) The chaos-penalty twins work unchanged.
- `cma_es_policy_search(evaluator, x0, sigma0, popsize, diagonal, budget)`:
  - no bounds; full CMA up to 600 parameters (so A's 481 and B1's 513 both use it), sep-CMA
    (`CMA_diagonal`) above, the threshold a config value. Full vs sep-CMA is compared once on quick
    dev settings, since at population 1,332 covariance learning is slow (c_mu about 0.003) and full
    CMA's advantage is uncertain;
  - optional per-block initial step sizes (`CMA_stds`), used by A-warm;
  - the start point x0 is scored at generation 0 and kept as a "best ever" candidate, so a warm
    start can't end worse than it began;
  - stops at a robot-second cap or when the dev score has stagnated for a configured number of
    checkpoints, whichever comes first;
  - returns both the best candidate ever seen and the CMA distribution mean (`es.result.xfavorite`),
    both dev-scored, since in hundreds of dimensions the mean often generalizes better;
  - saves and restores the ES state and NumPy's global RNG state (`cma` draws from it), so a resumed
    run continues exactly.
- `RobustnessSettings` / `RobustnessConfig` gain `smoke_seeds`, `dev_seeds`, `fall_rate_seeds` and the
  reserved ranges, with the overlap rule of 2.5.
- **Fresh-scenario path** for the ablations: new noise tables and robots copied into fixed buffers
  each generation, instead of the per-seed-tuple caches `BatchEvaluator` keeps today (which never
  evict and would re-capture graphs every generation). **One fresh-seed protocol for every method**
  (A, B1, the PID, later C), drawn from the reserved range >= 1,000,000, so "does more data help?" is
  answered the same way for all of them. Not built in the current scope unless time allows; designed
  so it slots in.

### 3.4 One comparison harness

`optimization/compare_controllers.py` loads saved controllers (PID gains JSON, LQR gains JSON,
policy `.npz`) and runs, in float64 NumPy on the designated validation machine:

- validation (300) and fall-rate (3,000) at 60 s;
- **the latency sweep**, at 60 s, on a named subset of the fall-rate tier: noise seeds 6000-6332 x
  3 tilts (999 scenarios), on a 2.5 ms grid from 15 to 45 ms (13 latencies). About 1 to 2 h of
  float64 CPU per controller. Pre-registered metric: **the first latency with 4 or more falls**
  (the count at which a Clopper-Pearson upper bound passes 1% on about 1,000 scenarios). Reported per
  run, with the median over runs, and the full fall-count curve with intervals is always shown,
  because curves can be non-monotone (the existing weight-tuned LQR has 1 fall at 20 ms and none at
  25 ms). The paper states that latencies above 20 ms are outside both the training range and the
  networks' 4-command history; that is the point of the test, but it must be said;
- **statistics:**
  - Headline estimator: the median, over a method's 10 runs, of each run's median per-scenario
    ratio against the reference PID.
  - Its bootstrap interval treats scenarios as **crossed** with runs: each replicate resamples run
    indices and **one** scenario index vector shared by every run (resampling scenarios separately
    per run would shrink the shared scenario-difficulty component and give intervals that are too
    narrow).
  - Also a distribution-free interval for the median over 10 runs: the 2nd to 9th order statistics,
    about 98% coverage.
  - Fall rates: Clopper-Pearson intervals; McNemar tests per run against the reference PID on shared
    scenarios, reported as the median and range over runs.
  - Every run's numbers are also listed;
- per-scenario costs and fall times for every controller, so each fall can be attributed to the
  start tilt, a push (5.5 or 7.5 s), braking, or drift during standstill.

`run_lqr_comparison.py` keeps working and calls into the shared code.

### 3.5 Linearized analysis

- Generalize `LinearController` (`stability.py`) to a vector integral state: `integral_gain` of length
  n_I, `integrand` of shape (n_I x 3), `loop_matrix` base 3 + n_I. Leaky integrators add their leak
  term to the integrator rows. Keep `cascaded_pid` and `lqr` as special cases, with regression tests
  that their margins don't change.
- **L is linear, so its linearization is exact** and needs no operating point. Its stable latency
  range is reported alongside the PID's and LQR's. This is required in the current scope.
- **For MLPs (A, A-warm, B1), optional:** linearize at the state the network actually settles into in
  a noise-free 60 s run on the nominal robot, including the clip's slope. Label the result as local.
- Leaky integrators the policy doesn't use leave an eigenvalue of 1 - dt/T_leak (0.9995 at 10 s).
  That doesn't break `stable_latency_range`, but it would dominate the reported spectral radius
  (the PID's own is 0.9951), so the spectral radius is reported over the coupled modes only.
- Test: the linearized loop's response matches a small-amplitude, noise-free, unsaturated simulation.

### 3.6 Config

A `LearningConfig` section in `config.py`: observation list, scales and T_leak, network sizes, output
rule, residual authority, seed tiers, CMA-ES policy settings, budgets, caps and checkpoints, PPO
settings (for later), and run profiles (section 9). Everything goes into `snapshot()`, plus the
hardware (GPU model, CPU model and SIMD features, CUDA and torch versions, OS).

### 3.7 Phase 0 tests

- **Gate test (must pass before L or A runs), in two parts:**
  - (a) Exactness: L with the PID-equivalent weights matches a reference implementation of "the
    cascaded PID with L's integrator semantics" (leaky, freeze-on-saturation) to rtol 1e-9 in
    float64.
  - (b) Disclosed gap: that variant's training objective and 60 s dev score are within 1% of the
    real reference PID's (bound stated now; the third review measured +0.68% and -0.66%). The gap is
    reported in the paper.
- `balance_sim` and NumPy `batch_sim` policies agree bit for bit in float64 (fixed-order forward
  pass). NumPy against torch: one-tick outputs to rounding, full runs on trained networks at rtol
  1e-9, and the existing "every disagreement is chaos" check (`test_torch.py`) for random networks.
- Different networks in one batch give the same result as each network alone, including a short last
  chunk.
- The residual with zero output weights and bias reproduces the PID bit for bit in float64.
- The network can command exactly +-max_torque.
- Exported then reloaded parameters give identical torques; the `.npz` holds only actor weights and
  the 11-input list.
- The latency-sweep path (global latency) works for policies.
- Seed sets obey the overlap rule.
- A resumed CMA-ES run matches an uninterrupted one.

## 4. Methods L, A, A-warm

### 4.1 Linear policy (first milestone)

CMA-ES over the 10 weights with the policy path, from a random start (the PID-equivalent weights are
used only by the gate test and the L-PID row, so L's result is an honest search result). Population
256: small populations sit on the GPU's fixed per-batch cost (about 3 s per generation below about
100k simulations), so CMA's default of about 11 would take roughly 50 minutes for 1,000 generations.
It checks the whole Phase 0 pipeline and gives the reference point for reading A.

### 4.2 Neuroevolution (A)

- `optimization/run_neuroevolution.py`: CMA-ES or sep-CMA-ES over the MLP parameters with the robust
  evaluator on the GPU.
- Initialization: Xavier-scaled hidden weights, small output layer, drawn from the run seed.
- Signal from a random start: almost every candidate falls within 0.3 to 1 s, but the fall penalty's
  remaining-time term ranks them by survival time, and CMA-ES uses only ranks, so there is a coarse
  signal. Fallbacks if runs plateau, each disclosed: a short-horizon curriculum (5 s, no pushes, then
  the full scenario); a smaller network; a larger population.
- **Budget, stated so it isn't read as a finding:** at the reference PID's 58.3 M robot-seconds A gets
  about 9,700 candidates, roughly 7 generations at population 1,332. Expect a near-random
  matched-budget score. A full run is capped by robot-seconds (about 6 G, about 750 generations of
  8 M each) or dev stagnation.
- **Wall-clock estimate:** the reference PID's CMA-ES ran 40 generations of 36,000 simulations in
  124 s (3.1 s per generation, about 1.3x the September benchmark floor, because the physics is now
  heavier). Scaling to 1,332 x 300 simulations plus the MLP gives about 11 to 15 s per generation, so
  **about 2.5 to 3 h per full run** on AC power, 2 to 3x longer on battery. Benchmarked in Phase 0 before
  budgets are fixed.
- Overfitting checks (481 to 1,473 weights can memorize 150 scenarios where 4 gains can't): log the
  training-vs-dev gap at each checkpoint, with dev scored at **20 s** for this purpose so the gap
  isn't confused with the extra 40 s of standstill in the 60 s runs.
- Outputs: `results/neuroevolution/<timestamp>/` with policies at each checkpoint, history CSV and
  config snapshot.

### 4.3 A-warm

The same MLP, with a linear path from inputs to output added, initialized to L's best weights, and
the MLP's output layer initialized to zero. It starts exactly at L-best (not at the PID). It answers
"does nonlinearity help, starting from a competent linear controller?" without confusing that with
how hard it is to search 481 weights from scratch. Details:

- Per-block initial step sizes (`CMA_stds`): small for the linear path, larger for the MLP, so the
  first generation doesn't scatter far from L.
- The start point is scored at generation 0 and kept as a candidate.
- L's robot-seconds are charged to A-warm's budget.
- Disclosed as a warm start; A from scratch remains the pure test.

## 5. Method B1: residual policy with CMA-ES

- Base: the reference PID, loaded from `results/lqr_comparison/20261108-225341/comparison.json` (the
  same gains every ratio in the paper is measured against), recorded in the run snapshot.
- Torque: `clip(PID + authority * max_torque * tanh(NN(obs)), +-max_torque)`, authority 0.25 and 0.5
  as an ablation. Output weights and bias start at zero, so training starts exactly at the PID, and
  that start point is scored at generation 0 and kept as a candidate.
- Trained with A's code. B3 (optional): CMA-ES on PID gains and residual jointly.
- Log the residual's share of total torque over validation runs. A residual near zero means the PID
  was already near-optimal for this cost.
- Likely needs fewer generations than A (it starts competent); budget set from dev curves.

## 6. Later phase: PPO (C) and residual PPO (B2)

Designed now so Phase 0 doesn't block it; built after the current scope, with a go/no-go gate (10).

### 6.1 Environment

- `optimization/rl/env.py`: `BalanceVecEnv`, built from the `batch_sim` pieces (`rk4_step`,
  `actuator_torque`, the `Schedule`, noise tables, latency ring, `sample_robots`).
- **Captured as CUDA graphs** (env tick plus the actor's sampling, as `TorchSimulator` already does
  for its loop; random sampling works inside graphs). Without graphs each tick costs about 5 to 10 ms
  in kernel launches regardless of the number of environments, which at 1,200 environments would mean
  about 20 h per run seed. With graphs, about 0.6 to 0.8 ms per tick.
- **Staggered rollout groups:** the environments are split into K groups (for example 8), each a
  separate env instance with its own global step counter, started 4,000/K ticks apart. Updates come
  every 4,000/K ticks, on truncated segments bootstrapped by the time-aware critic, and every update
  still covers every phase of the schedule. 2 to 4 replicas per scenario. Memory stays under about
  1 GB.
- **Equivalence test:** the PID run through the env gives a summed raw reward equal to minus
  `BatchEvaluator`'s cost on the same scenarios, at rtol 1e-12 (the evaluator sums the three cost
  integrals separately and weights them at the end, so bit-identity isn't possible unless the env
  does the same; it may, to make the test exact). Required cases: odd delays (3, 5, 7 physics steps);
  a fall on the first physics step of a tick; the pre-step cost sample on the fall step;
  `fall_time = (k+1)*dt` from each group's step index.

### 6.2 Reward

- Raw per tick: `-(pitch^2 dt + effort_weight * torque^2 dt + velocity_weight * (v - goal)^2 dt)`
  over the tick's physics steps while upright, minus `fall_penalty * (1 + remaining)` on a fall.
- Training uses running return normalization and a training-only fall-penalty scale (a dev
  hyperparameter): a typical run costs about 1.6 (about 4e-4 per tick) while a fall costs 1,000 to
  2,000. Scoring uses the real cost.
- gamma's horizon is 1/(1 - gamma) ticks: 0.995 is 1 s, 0.999 is 5 s, 0.9995 is 10 s. Dev grid: those
  three.
- The chaos penalty can't be a reward; it is reported at validation for every method.

### 6.3 Training

- Single-file PPO, `optimization/rl/ppo.py`, following CleanRL's continuous-action implementation.
- Actor: Gaussian on the unsquashed output, clipped in the env, exported as `clip(mean)`.
  **Exploration noise is defined in N m:** initial sigma at most 0.02 N m. (At 0.076 N m, noise alone
  would add about 1.0 to a typical 1.6 cost through the effort term.) The noise-induced cost is
  logged.
- **Privileged critic** (never deployed): the actor's full observation, the true state (wheel
  position dropped or scaled), robot parameters, latency, time remaining, and explicit phase
  features (current goal and its slope, time to and since each push and goal change). The export test
  confirms none of it reaches the actor.
- Budget: capped in robot-seconds in `LearningConfig`. Estimated about 2 h per run seed with graphs
  at 58 M robot-seconds; with a dev grid and 10 run seeds, roughly 30 to 50 GPU-hours for C, and
  similar for B2. Re-measured in the PPO phase before committing.
- Headline on the fixed 150 training scenarios; fresh-scenario ablation through the shared protocol
  (3.3).

## 7. Post-semester phases

- **E: GRU policy.** A policy with memory can infer each robot's properties and its latency as it
  runs, which no fixed-gain controller can; it speaks directly to the robustness thesis. Trained with
  recurrent PPO on C's environment, using truncated sequences (64 to 256 ticks) with stored hidden
  states and a burn-in, not whole 4,000-tick episodes. The NumPy and `balance_sim` GRUs must copy
  PyTorch's gate equations exactly (the reset gate applies after the hidden-state matmul), with a
  test. cuDNN TF32 off (3.2). Several times C's compute; a cluster job. Not trained with CMA-ES
  (a hidden-16 GRU has about 1.4k parameters, and A vs C and B1 vs B2 already compare the
  optimizers on identical architectures).
- **D: differentiable simulation.** Backpropagate the cost through the simulator to the 4 PID gains
  (the direct comparison with CMA-ES) or to A's MLP. Needs a functional rollout separate from
  `TorchSimulator`, a smooth stand-in for the fall penalty, and truncated windows with gradient
  clipping. It complements PPO rather than replacing it (Suh et al., ICML 2022).
- **An H-infinity baseline** (1.1), if the robustness comparison needs a classical robust design.

## 8. Results and paper

- Comparison table: L, A, A-warm, B1 (later C, B2) against the existing controllers, with the
  headline ratio and its two-level CI, mean, fall rate on 3,000 with its interval, sensitivity,
  latency-sweep metric with intervals, linearized stable latency (L exactly; MLPs if done),
  parameters, training robot-seconds, wall-clock, and all 10 runs.
- Learning curves on a log robot-seconds axis, from the checkpoint dev scores.
- Per-scenario ratio distributions, the latency sweep, one example run with the residual torque.
- Inference cost per tick (multiply-adds) and whether it fits a microcontroller at 200 Hz.
- Related work and limitations include robust control (1.1).
- New `ML_BASELINES.md`; updates to `README.md` (layout), `ARCHITECTURE.md` (status).

## 9. One pipeline for laptop and cluster

One code path; only configuration differs between machines.

- **Run profiles** in `LearningConfig`: `quick` (smoke seeds, short budgets, one run seed, float32) is
  the default for development and tests; `full` (dev and checkpoint evaluation, full budgets, 10 run
  seeds) is for final runs. Selected by `--profile`, every value overridable.
- **Device and speedups detected, never required:** CUDA if available, else CPU NumPy. `torch.compile`
  where Triton is available (Linux); CUDA graphs on both.
- **Cluster-friendly:** checkpoint and resume (with the RNG state, 3.3), one run seed per process so
  sweeps map onto job arrays, no interactive plots on headless machines, and a thin SLURM script in
  `scripts/` that only calls the normal entry points.
- **Reproducibility across machines:**
  - `torch.compile` and different GPUs change float32 rounding, and chaos then makes CMA runs
    diverge. So **all 10 final runs of a method use one platform configuration**, recorded in the
    snapshot.
  - **All final float64 validation runs on one designated machine**, with its CPU model and SIMD
    features recorded (SIMD transcendental functions differ between AVX2 and AVX-512 CPUs). The
    fixed-order forward pass (3.1) removes BLAS from that path.
- Everything must run on the Windows laptop. Tests use the quick profile.

## 10. Order of work and gates

Current scope:

1. **Phase 0** (CMA-ES path): policy controllers, evaluator path, generalized `LinearController`, seed
   tiers, harness, profiles, tests. Gate: all Phase 0 tests pass, including the L-equals-PID gate,
   and the throughput benchmark is done.
2. **L.** Gate: the pipeline runs end to end on smoke and dev.
3. **A and A-warm.** Dev grid on quick settings, then 10 run seeds.
4. **B1.** Dev grid, then 10 run seeds.
5. **10 re-runs of the reference PID.**
6. **Final comparison:** validation and fall-rate tiers, latency sweep, statistics, figures. Once, on
   frozen controllers.

Later phase (go/no-go set when the current scope is done): the PPO env with graphs and the
equivalence test, then C, then B2. If C isn't learning by its go/no-go date, it is reported as
attempted, with what was tried.

Post-semester: E, D, the H-infinity baseline.

Provisional dates, aligned with the every-other-Friday progress reviews (final dates once the
paper's deadline is fixed):

- **2026-11-20:** Phase 0 and L done (gate passed, pipeline end to end).
- **2026-12-04:** A, A-warm and B1 trained on 10 run seeds; PID re-runs done.
- After that: the final comparison, then the PPO go/no-go.

## 11. Open questions

The third review's answers to the revision 3 questions were adopted (leak plus freeze with a fixed
T_leak; the split gate; A-warm with per-block step sizes; the count rule for the latency metric).
Remaining:

1. Freeze-on-saturation uses the previous tick's saturation flag. Is one tick of lag acceptable at
   the 0.30 rad start, or should the frozen check use the current tick (two forward passes)? The
   third review measured almost no difference (0.68% vs 0.72% on training).
2. Should the residual's (B1) PID part keep the real PID's integrator, or switch to L's semantics
   for consistency with L-PID? The plan keeps the real PID so B1 starts exactly at the reference.
