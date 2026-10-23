# LQR, and how it compares with the cascaded PID

The cascaded PID is one controller structure among many. This adds the textbook model-based alternative, a linear-quadratic regulator (LQR), to the base robot, and compares the two, hand-designed and ML-tuned, on one common score.

```
simulation/balance_sim/planar2d.py     linearize(): the linear model (PHYSICS.md 3.1), with friction
simulation/balance_sim/lqr.py          LQRWeights, discrete_model, lqr_gain, LQRController
simulation/balance_sim/stability.py    exact stability of the sampled, delayed linear loop, for either controller
simulation/batch_sim/controller.py     lqr_update: the same controller for a batch (NumPy and GPU)
optimization/tuning/search_space.py    what the tuner searches: LQR weights, or LQR gains directly
optimization/run_lqr_comparison.py     the comparison; output in results/lqr_comparison/
simulation/tests/test_lqr.py           65 tests
config.py (LQRConfig)                  search bounds; the cost-matched weights come from CostConfig
```

## Why LQR is the natural comparison

In its linear range (no saturation, no lean cap), the cascaded PID is already state feedback. With $e = \theta + k_v(v - v_{\text{goal}})$:

```math
\tau = k_p\theta + k_d\dot\theta + k_pk_v\,(v - v_{\text{goal}}) + k_i\!\int\!\theta\,dt + k_ik_v\!\int\!(v - v_{\text{goal}})\,dt .
```

That is five feedback gains set by four numbers, in a fixed pattern. An LQR drops the pattern: it feeds back the full state with whatever gains minimize a quadratic cost on the linear model. The project's cost already has that form (PHYSICS.md 10.5), so the LQR can be designed for the same cost the tuner scores. The comparison then asks what the PID's structure costs, and what machine learning adds on top of a model-based design.

## The design (`lqr.py`)

**Model.** `linearize()` gives $\dot s = As + B\tau$ about upright (PHYSICS.md 3.1; a test checks it against finite differences of `Planar2D.derivatives`). The axle position is dropped, since nothing depends on it, leaving $[v - v_{\text{goal}}, \theta, \dot\theta]$, plus the integral of velocity error when it is weighted.

**Cost.** Per second, with pitch from upright:

```math
\theta^2 + w_v(v - v_{\text{goal}})^2 + w_r\dot\theta^2 + w_i\Bigl(\int(v - v_{\text{goal}})\Bigr)^2 + w_u\tau^2 .
```

The **cost-matched** weights are the cost function's own: $w_v = 1$ (`velocity_weight`), $w_u = 0.01$ (`effort_weight`), no rate or integral term. On the linear, noise-free, unsaturated robot, this is the optimal controller for the project's cost.

**Timing, built in exactly.** The controller runs every 5 ms and holds its command for one period, so the model is discretized with a zero-order hold. The 10 ms latency means the two commands issued in the last two ticks are still on their way to the motor. They become part of the state, and the gain feeds them back too. A latency that isn't a whole number of periods is handled exactly as well (PHYSICS.md 7.4's split period). `test_discrete_model_matches_integrating_the_linear_robot` checks one period against direct integration of the linear robot. The result has a clean interpretation, which `test_delay_aware_gain_acts_on_the_predicted_state` verifies: the delay-aware gain is the delay-free LQR applied to the state predicted two ticks ahead.

**The controller.** `LQRController` computes $\tau = -Kz$ on the measured state, clips it to the torque limit, and remembers the clipped commands it sent. It commands no lean, so its pitch reference is upright. The batched `lqr_update` adds the terms in the same order, so in float64 the NumPy batch reproduces the reference simulator's trajectories bit for bit (`test_batched_trajectories_match_reference_exactly`). The GPU matches to rounding for well-behaved gains, as for the PID.

**What the design doesn't know:** sensor noise, torque saturation, the motor's back-EMF limit, large angles, and the robot-to-robot spread. The simulations test those.

## Loop stability (`stability.py`)

Both controllers are linear in their unsaturated range, so the sampled, delayed loop is a matrix $z_{k+1} = Mz_k$. It is stable when every eigenvalue of $M$ is inside the unit circle. `stable_latency_range` finds the latencies over which that holds. This generalizes PHYSICS.md 7.4's analysis to either controller and reproduces its numbers: the hand-tuned cascade goes unstable at 11.5 ms and the 2026-09-30 tuned cascade at 31.0 ms.

## How the tuner searches an LQR

The batched evaluator takes `controller="lqr"` and a `to_gains` map from search-space points to gains, so all the existing optimizers and robust scoring work unchanged.

- **Weights** (`lqr_weight_space`, `LQRWeightDesigner`). Four weights, log scale, pitch fixed at 1 (only ratios matter): the same dimension as the PID's four gains. Every candidate is an LQR design for the nominal robot, so every candidate stabilizes the nominal linear loop.
- **Gains directly** (`lqr_gain_space`, `LQRGainScaling`). The cost-matched design's five gains, each scaled by a factor in [1/20, 20]. This drops the weights' parameterization to see whether it limits the LQR. Two safeguards:
  - **Gains that don't stabilize the nominal linear loop are pulled back** toward the cost-matched design, by bisection, to the last stable point. Every LQR design has that guarantee, and this keeps it while still giving the optimizer a gradient.
  - **The integral gain isn't searched.** An earlier version searched it, and CMA-ES found gains that scored 0.908 on the training scenarios, against about 0.94 for the ML-tuned PID. They had an unstable closed-loop mode (eigenvalue 1.0009, a 6 s time constant) that a 12 s scenario with a 0.5 m integral cap never exposes. On a 30 s run at constant speed, those gains held 0.44 m/s against a goal of 0.50 with the cap and fell at 29 s without it. The search had exploited the scenario's finite length. This is a general hazard of black-box tuning, and a reason to check any tuned controller's linear stability.

## The comparison (`run_lqr_comparison.py`)

| Controller | What it is |
|---|---|
| PID, hand-tuned | config.py's `hand_tuned_gains` |
| PID, ML-tuned | the 4 gains, searched |
| LQR, cost-matched | Q, R = the cost's weights, designed for 10 ms. No tuning, no simulation |
| LQR, latency ignored | the same weights, designed as if there were no latency (the textbook design) |
| LQR, ML-tuned weights | the 4 weights, searched |
| LQR, ML-tuned gains | the 5 gains, searched around the cost-matched design, kept nominally stable |

**One score for everyone.** The project's default cost measures pitch error from the lean the PID commands. The LQR commands no lean, so that would score the two kinds of controller against different references. Every controller here is scored, and every search minimizes, the configured cost with pitch measured from upright (`pitch_reference="upright"`, the convention `run_pivot_tuning.py` already uses). So the ML-tuned PID here is tuned for that score, and its numbers aren't directly comparable with `results/batch_tuning/`.

**Same treatment for the three searches.** Robust scoring from config.py (50 seeds × 3 tilts, randomized robots, chaos penalty: 300 simulations per candidate). CMA-ES and differential evolution each run at the same budget, about 4,800 candidates, and the better result is kept. Validation then uses 100 unseen seeds with unseen robots (300 scenarios) in float64.

### Results

Run `results/lqr_comparison/20261014-165757/` (GPU, float32 searches, float64 validation; `lqr_comparison.png`, `example_runs.png`, `comparison.json`). Validation: 300 unseen scenarios. "vs ML PID" is the paired difference on the same 300 scenarios (same noise, same robot), with a 95% interval.

| Controller | Validation cost | vs ML PID | Better on | Effort term | Stable latency (linear) | Cost at 0 / 20 / 25 / 30 ms (falls of 60) |
|---|---:|---:|---:|---:|---|---|
| PID, hand-tuned | 1.836 | +95% | 0/300 | 0.0016 | 0–11.25 ms | 1.69 / 11.5 / 100 (30) / 182 (60) |
| **PID, ML-tuned** | **0.940** | | | 0.0002 | 0–24.0 ms | 0.92 / 1.02 / 3.08 / 15.9 (3) |
| LQR, cost-matched | 0.957 | +1.8% ± 0.05% | 0/300 | 0.0009 | 7.5–12.25 ms | 1.12 / 1.23 / 3.29 / 5.45 |
| LQR, latency ignored | 1.797 | +91% | 0/300 | 0.0141 | unstable at 10 ms (0–4.75 ms) | 0.93 / 4.30 / 44 (13) / 139 (45) |
| LQR, ML-tuned weights | 0.955 | +1.6% ± 0.04% | 0/300 | 0.0004 | 0–18.5 ms | 0.94 / 1.11 / 2.73 / 5.42 |
| LQR, ML-tuned gains | 0.951 | +1.2% ± 0.03% | 0/300 | 0.0003 | 0–22.5 ms | 0.93 / 1.20 / 6.89 (1) / 103 (33) |

No controller fell in validation. Velocity tracking is about 90% of every tuned controller's cost.

What the searches found:
- **ML-tuned PID:** kp 1.10, ki 0.63, kd 0.106, kv_p 0.325. CMA-ES and DE both reach a training objective of 0.9401.
- **ML-tuned weights:** w_velocity 1.21 and w_torque 0.40, 40 times the cost's 0.01. The rate and integral weights went to the bottom of their ranges, which in practice means switched off.
- **ML-tuned gains:** every gain scaled to about 0.27–0.29 of the cost-matched design, and the feedback on the commands in flight scaled to about 0.08. DE found this; CMA-ES stalled at 0.972, worse than the design it started from.

### Findings

1. **The ML-tuned PID wins, narrowly but consistently.** Every LQR variant is 1.2–1.8% worse, on every one of the 300 paired scenarios. The margins are small because about 90% of the cost is velocity tracking, which physics limits: the non-minimum-phase start and the torque and lean limits (PHYSICS.md 8.2).
2. **A textbook LQR, designed from the model and the cost alone, comes within 1.8% of the ML-tuned PID.** It needs no simulation and no tuning, against about 3 million simulations for the PID search, and it halves the hand-tuned PID's cost. Model-based design is a strong baseline, and in nominal conditions ML adds little on top of it.
3. **Modeling the latency is essential.** The same design without the 10 ms latency is unstable at 10 ms. Torque saturation turns the instability into a bang-bang oscillation between the torque limits, so it doesn't fall, but its cost nearly doubles and it spends 64 times the ML-tuned PID's effort. At 0 ms, where its assumption holds, it scores 0.93, as good as any controller.
4. **ML tuning barely improves the LQR's score but greatly improves its robustness.** Both searches softened the design: the torque weight went up 40×, or the gains went down about 3.5×. That cut torque effort by 2–3×, less sensor noise passed through to the motor (visible in `example_runs.png`), and the latency range over which the linear loop is stable widened from 7.5–12.25 ms to 0–18.5 or 0–22.5 ms. The cost-matched design is optimal for a noise-free, unsaturated linear robot with exactly known latency. The realistic robot (noise, saturation, back-EMF, ±15–25% parameter spread) wants a gentler controller. Robust scoring found that without being told about latency at all.
5. **The two kinds of controller fail differently under the wrong latency.** The ML-tuned PID stays best up to 20 ms (cost 1.02), then starts falling at 30 ms. The cost-matched and weight-tuned LQRs degrade sooner, but at 30 ms they still haven't fallen (cost about 5.4, sustained oscillation). The cost-matched LQR is also worse when the latency is shorter than designed (1.12 at 0 ms, 1.22 at 5 ms): it compensates for delay that isn't there.
6. **The PID's edge is not the LQR's weight parameterization.** Searching the LQR gains directly still leaves linear state feedback 1.2% behind. The likely sources are the PID's nonlinear pieces (the 25° lean cap, anti-windup) and its integral of pitch error, which the LQR state lacks. That is a hypothesis, not yet tested.
7. **Two methodological lessons:**
   - An unconstrained gain search exploited the finite scenario. That is why the direct search keeps the nominal loop stable and leaves the integral gain out.
   - One optimizer can quietly fail. CMA-ES stalled on the direct gain search, and running DE as well caught it.

### Caveats and next steps

- **One scenario.** The scores come from a single goal profile of velocity steps. Smooth goal profiles (to-do item 2) and deliberately chosen weights (PHYSICS.md 10.5, to-do item 3) would make the cost measure balance quality more and physics limits less, and might separate the controllers further.
- **Different scoring from earlier runs.** The ML-tuned PID here is tuned for pitch measured from upright, so its gains differ from those in `results/batch_tuning/`.
- **Latency is fixed at 10 ms during tuning.** Randomizing it (PHYSICS.md 10.1) would reward the differences in finding 5.
- **The LQR knows nothing about noise.** A Kalman filter in front of it (LQG) is the natural next permutation. So is an LQR with the PID's lean cap, which would test finding 6.
- **Shared code with the pivot model.** `pivot_sim.lqr_gain` uses the same delay-aware scheme and could reuse `balance_sim.discrete_model`. It was left unchanged here.

### Reproducing

```
python optimization/run_lqr_comparison.py --backend torch --float32          the full run, ~7 min on the GPU
python optimization/run_lqr_comparison.py --replot results/lqr_comparison/<run>   redraw the figures only
python simulation/animate.py --csv results/lqr_comparison/<run>/trajectories/PID_ML-tuned.csv results/lqr_comparison/<run>/trajectories/LQR_cost-matched.csv
python -m pytest simulation/tests/test_lqr.py
```
