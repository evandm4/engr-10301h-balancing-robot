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

The **cost-matched** weights are the cost function's own: $w_v$ (`velocity_weight`), $w_u$ (`effort_weight`), no rate or integral term. On the linear, noise-free, unsaturated robot, this is the optimal controller for the project's cost. Everything below was produced with $w_v = 1$, $w_u = 0.01$ and a step-shaped goal. `config.py` now has $w_v = 2.25$, $w_u = 9$ (Bryson's rule, derived from acceptable-value limits) and a smooth goal; see the note at the start of Results.

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
| PID, model-based | textbook sequential cascade design from the linear model (`pid_design.py`, PHYSICS.md 6.5). No tuning, no simulation |
| PID, ML-tuned | the 4 gains, searched |
| LQR, cost-matched | Q, R = the cost's weights, designed for 10 ms. No tuning, no simulation |
| LQR, latency ignored | the same weights, designed as if there were no latency (the textbook design) |
| LQR, ML-tuned weights | the 4 weights, searched |
| LQR, ML-tuned gains | the 5 gains, searched around the cost-matched design, kept nominally stable |

**One score for everyone.** The project's default cost measures pitch error from the lean the PID commands. The LQR commands no lean, so that would score the two kinds of controller against different references. Every controller here is scored, and every search minimizes, the configured cost with pitch measured from upright (`pitch_reference="upright"`, the convention `run_pivot_tuning.py` already uses). So the ML-tuned PID here is tuned for that score, and its numbers aren't directly comparable with `results/batch_tuning/`.

**Same treatment for the three searches.** Robust scoring from config.py (50 seeds × 3 tilts, randomized robots, chaos penalty: 300 simulations per candidate). CMA-ES and differential evolution each run at the same budget, about 4,800 candidates, and the better result is kept. Validation then uses 100 unseen seeds with unseen robots (300 scenarios) in float64.

### Results

**Current: fresh held-out validation set (2026-11-09).** Run `results/lqr_comparison/20261108-225341/`. The searches are identical to the run below (same training seeds, same results), but validation now uses noise seeds 2000–2099 and the robots they draw. The earlier validation set (seeds 1000–1099) was used while sizing the 20 s scenario's push and blend times, so it was retired; the new set was fixed after the scenario, search bounds and model-based design were frozen and is used for nothing else.

| Controller | Validation cost | vs ML PID (median per scenario) | Falls (of 300) | Worst run | Effort term | Stable latency (linear) |
|---|---:|---:|---:|---:|---:|---|
| PID, hand-tuned | 39.04 | 6.23x | 4 | 1918 | 5.277 | 0-15.75 ms |
| PID, model-based | 9.484 | 1.36x | 1 | 1878 | 1.745 | 0-23.75 ms |
| **PID, ML-tuned** | **2.221** | | **0** | 4.82 | 1.562 | 0-37.25 ms |
| LQR, cost-matched | 28.28 | 4.21x | 3 | 1903 | 2.159 | 0-44.25 ms |
| LQR, latency ignored | 6.871 | 3.09x | 0 | 15.15 | 2.112 | 0-36.50 ms |
| LQR, ML-tuned weights | 29.78 | 1.64x | 4 | 1988 | 2.048 | 0-42.25 ms |
| LQR, ML-tuned gains | 2.982 | 1.34x | 0 | 6.41 | 2.190 | 0-35.00 ms |

- **The conclusions survive the new set.** The ML-tuned PID is still the only PID that never falls; the median ratios move by at most 0.14 (cost-matched LQR) and the ranking is unchanged.
- **The model-based PID** has a median 1.36x the ML-tuned PID's and falls once; the tuned LQR gains are at 1.34x with no falls.
- **Falls:** hand-tuned PID 4, weight-tuned LQR 4, cost-matched LQR 3, model-based PID 1; the ML-tuned PID, tuned LQR gains and latency-ignoring LQR none.
- **Latency sweeps** (60 scenarios per point): the ML-tuned PID stays within 6% of its 10 ms cost up to 20 ms (1.55 against 1.47), reaches 2.24 at 30 ms and falls 4 times at 40 ms. The model-based PID falls twice at 20 ms and 37 times at 30 ms. The latency-ignoring LQR never falls up to 40 ms (10.1). The cost-matched LQR falls in one scenario at every latency, including zero.

**Earlier: 20 s scenario, retired validation set (2026-11-08, later).** Run `results/lqr_comparison/20261108-220417/`, which adds the model-based PID; every other row reproduces the first run on this scenario (`20261108-182234/`) exactly. Same protocol as the run below, on the redesigned scenario (`ScenarioConfig`: start tilts up to 0.30 rad, a 1.6 s speed-up to a 1.3 m/s cruise, two 2.9 N pushes, braking and reversing; 20 s training runs, 60 s validation runs). Same robot, cost and robust scoring.

| Controller | Validation cost | vs ML PID (median per scenario) | Falls (of 300) | Worst run | Effort term | Stable latency (linear) |
|---|---:|---:|---:|---:|---:|---|
| PID, hand-tuned | 30.02 | 6.18x | 2 | 1901 | 5.986 | 0-15.75 ms |
| PID, model-based | 15.34 | 1.36x | 2 | 1874 | 1.763 | 0-23.75 ms |
| **PID, ML-tuned** | **2.261** | | **0** | 4.99 | 1.600 | 0-37.25 ms |
| LQR, cost-matched | 22.32 | 4.35x | 2 | 1900 | 2.260 | 0-44.25 ms |
| LQR, latency ignored | 7.077 | 3.15x | 0 | 14.90 | 2.197 | 0-36.50 ms |
| LQR, ML-tuned weights | 48.92 | 1.66x | 7 | 1948 | 2.066 | 0-42.25 ms |
| LQR, ML-tuned gains | 3.050 | 1.35x | 0 | 6.44 | 2.244 | 0-35.00 ms |

- **The PID still wins, by more.** The tuned LQR gains are 35% behind (18% on the 12 s scenario), the tuned weights 66%, the cost-matched LQR 4.3x.
- **A textbook PID design is close on typical runs but not robust.** The model-based PID, designed from the linear model with no tuning and no simulation (PHYSICS.md 6.5), has a median cost per scenario 1.36x the ML-tuned PID's: level with the tuned LQR gains, and far ahead of the hand-tuned PID (6.2x) and the cost-matched LQR (4.3x). But it falls on 2 of the 300 robots, and its linear loop tolerates only 23.75 ms of latency against the tuned PID's 37.25 ms (in the sweep it falls twice at 20 ms and 25 times at 30 ms). What the ML tuning adds over a competent model-based design is mostly robustness: no falls and about 1.6 times the delay margin.
- **Falls now separate the controllers.** Four of the seven fall at least once on the 300 unseen robots: the hand-tuned PID (2), the model-based PID (2), the cost-matched LQR (2) and the weight-tuned LQR (7). The ML-tuned PID, the tuned LQR gains and the latency-ignoring LQR never fall. A single fall costs 1000-2000, so mean costs with falls are dominated by them; the median ratio is the fairer comparison of the rest.
- **The PID search confirms the batch tuner.** It lands on kp 1.635, ki 1.767, kd 0.299, kv_p 0.620, the same gains `run_batch_tuning.py` found on this scenario (`results/batch_tuning/20261108-153813/`).
- **The weight search can't avoid falls.** Its training objective (38.6) includes falls, and its one free parameter (`w_integral`, settled at 6.21) is not enough: with the pitch, velocity and pitch-rate weights held at 0 (the 2026-11-06 normalization below), no design it can reach survives every push. Re-opening those weights (`LQRConfig.w_*_fixed = None`) is the next thing to try.
- **Ignoring the latency in the design helps again** (7.08 against 22.3 for the delay-aware design, and no falls).
- **Tuned LQR gain scales:** 3.08 (velocity), 0.861 (pitch), 1.71 (pitch rate), against 3.02, 0.757 and 1.59 on the 12 s scenario.
- **Latency sweeps** (60 scenarios per point): the tuned PID stays within 5% of its 10 ms cost up to 20 ms (1.58 against 1.50), reaches 2.61 at 30 ms, and falls 3 times at 40 ms. The tuned LQR gains hold to 20 ms, fall once at 30 ms and 32 times at 40 ms. The cost-matched LQR is flat to 15 ms, then falls once from 20 ms. The latency-ignoring LQR is the most delay-tolerant here: no falls up to 40 ms (cost 10.9). The hand-tuned PID falls twice at 20 ms and 50 times at 25 ms.

**Earlier: datasheet robot, 12 s scenario (2026-11-08).** Run `results/lqr_comparison/20261108-133118/`, same protocol as the 2026-11-06 run below. The robot is now config.py's datasheet values (DATASHEET.md): the motors' reflected rotor inertia in the dynamics, a gearbox efficiency of 0.78, 0.34 N m of torque, a 0.01 rad pitch-estimate offset, and robust scoring with Coulomb friction 0.02–0.08 N m, viscous 0–0.0015 N m s/rad, rotor inertia ±50% and gear efficiency ±10%. No controller fell in validation, and no search ended on a bound.

| Controller | Validation cost | vs ML PID (median per scenario) | Worst run | Effort term | Stable latency (linear) |
|---|---:|---:|---:|---:|---|
| PID, hand-tuned | 5.052 | 7.05x | 34.52 | 1.375 | 0-15.75 ms |
| **PID, ML-tuned** | **0.5832** | | 1.134 | 0.418 | 0-40.50 ms |
| LQR, cost-matched | 2.238 | 3.71x | 4.827 | 0.449 | 0-44.25 ms |
| LQR, latency ignored | 1.664 | 2.80x | 3.357 | 0.431 | 0-36.50 ms |
| LQR, ML-tuned weights | 0.7936 | 1.39x | 1.461 | 0.547 | 0-38.25 ms |
| LQR, ML-tuned gains | 0.6897 | 1.18x | 1.349 | 0.486 | 0-40.25 ms |

- **The ranking is unchanged:** the ML-tuned PID wins, the tuned LQR gains are 18% behind (1.18x, as before), the tuned weights 39% (1.44x before), and the cost-matched LQR 3.7x (3.70x before).
- **Every cost is about 3.5 times higher, and effort is now the largest term** (0.42 of the tuned PID's 0.58). Holding the robot against the Coulomb friction robust scoring now draws (0.02–0.08 N m, against 0–0.02) costs torque all the time it moves: a steady 0.05 N m over 12 s alone is $9 	imes 0.05^2 	imes 12 = 0.27$. On the nominal, frictionless robot the same PID scores 0.255 (PHYSICS.md 8.2).
- **The tuned PID got stiffer and leans on its integrator even more:** kp 1.57, ki 2.10, kd 0.285, kv_p 0.645 (from 0.66, 0.50, 0.112, 0.80). The larger friction is a larger load for the integrator to carry (PHYSICS.md 9.4), and the rotor inertia makes the robot slower, which allows more gain before the delay bites (PHYSICS.md 7.4).
- **Tuned LQR parameters:** the integral weight settled at 15.2 per unit of torque weight (4.6 before), inside its range. The gain search scaled the cost-matched gains by 3.02 (velocity), 0.757 (pitch) and 1.59 (pitch rate). DE found that design; CMA-ES stalled at 0.902, as it did in the first run (Old table).
- **Ignoring the latency in the design helps again** (1.66 against 2.24 for the delay-aware design), as on 2026-10-24.
- **Latency sweeps** (60 scenarios per point): the tuned PID holds within 5% of its 10 ms cost up to 25 ms (0.61), reaches 0.78 at 30 ms and 2.26 at 40 ms with no falls (it fell 12 times at 40 ms before). The tuned LQR gains and weights fall once and 3 times at 40 ms. The cost-matched LQR is flat out to 30 ms (2.23) and survives 40 ms (2.84). The hand-tuned PID falls once at 20 ms and 15 times at 25 ms.

**Earlier: friction and latency randomized (2026-11-06), robot before the datasheet update.** Run `results/lqr_comparison/20261106-154350/`, same protocol as below (GPU float32 searches, 300 unseen scenarios in float64). Robust scoring now also draws each scenario's viscous friction (0–0.001 N m s/rad), Coulomb friction (0–0.02 N m) and latency (5–20 ms in whole 2.5 ms steps). Every controller is still designed or tuned for the nominal robot's 10 ms. The weight search is torque-normalized, with only the integral weight searched (see the corner finding below). No controller fell in validation, and no search ended on a bound.

| Controller | Validation cost | vs ML PID (median per scenario) | Worst run | Effort term | Stable latency (linear) |
|---|---:|---:|---:|---:|---|
| PID, hand-tuned | 15.34 | 98.5x | 38.36 | 7.384 | 0-11.25 ms |
| **PID, ML-tuned** | **0.1687** | | 0.264 | 0.0608 | 0-35.75 ms |
| LQR, cost-matched | 0.6268 | 3.70x | 1.025 | 0.0652 | 0-30.75 ms |
| LQR, latency ignored | 0.9287 | 2.99x | 15.72 | 0.3481 | 0-21.0 ms |
| LQR, ML-tuned weights | 0.2410 | 1.44x | 0.340 | 0.0940 | 0-31.75 ms |
| LQR, ML-tuned gains | 0.1989 | 1.18x | 0.314 | 0.0737 | 0-34.25 ms |

- **Integral action is what friction asks for.** The tuned PID's pitch integral gain went from 0.106 to 0.498 (kp 0.66, kd 0.112, kv_p 0.80). A friction torque is a load the integrator has to carry, which is PHYSICS.md 9.4's mechanism. Its cost rose 29%, from 0.130 to 0.169.
- **The LQR gain search fell from 2.2% to 18% behind** (0.199 against 0.169). The search holds the integral gain at 0 (see "How the tuner searches an LQR"), so nothing in that design can carry a steady load. The weight search, whose designs do have integral action, is 44% behind, limited by the LQR's pitch gain (see the corner finding).
- **The hand-tuned PID is now 5x worse than before** (15.3 against 3.1). Its linear loop is unstable above 11.25 ms, and 4 of the 7 latencies a scenario can draw (12.5–20 ms) are above that. It doesn't fall, but it oscillates (sweep: 2.5 at 10 ms, 11.9 at 12.5 ms, 157 with 8 falls at 25 ms).
- **Modeling the latency matters again.** The latency-ignoring design now costs 0.93, against 0.63 for the delay-aware one (on 2026-10-24 it was slightly ahead). In the sweep it degrades past 15 ms (2.8 at 20 ms), while the delay-aware design holds 0.63 up to 25 ms.
- **Latency sweeps** (60 scenarios per point, friction still randomized): all four tuned or designed controllers hold within 10% of their 10 ms cost up to 20 ms. At 30 ms the tuned PID is at 0.78, the tuned LQR gains at 1.34, the tuned weights at 0.99 and the cost-matched LQR at 2.70. At 40 ms only the cost-matched LQR survives every scenario (21.9, no falls); the tuned PID falls 12 times out of 60.

**Earlier: re-run on the Bryson cost, fixed 10 ms latency, no friction (2026-10-24).** The table and findings further down were produced with the earlier cost ($w_v = 1$, $w_u = 0.01$, step-shaped goal, $k_{v,p} \le 0.4$) and are kept as a record. `config.py` now has $w_v = 2.25$, $w_u = 9$ (Bryson's rule), a smooth goal, $k_{v,p} \le 0.9$, wider LQR search ranges, and the pitch-rate weight and the gains on the commands in flight held at off (`LQRConfig.w_pitch_rate_fixed`, `direct_commands_scale`), so the weight search has 3 parameters and the gain search has 3. Run `results/lqr_comparison/20261024-142418/` (GPU, same protocol: 300 unseen scenarios, no controller fell):

| Controller | Validation cost | vs ML PID (median per scenario) | Worst run | Effort term | Stable latency (linear) |
|---|---:|---:|---:|---:|---|
| PID, hand-tuned | 3.121 | 16.6x | 13.99 | 1.090 | 0-11.25 ms |
| **PID, ML-tuned** | **0.1304** | | 0.161 | 0.0297 | 0-39.75 ms |
| LQR, cost-matched | 0.3895 | 3.02x | 0.451 | 0.0312 | 0-30.75 ms |
| LQR, latency ignored | 0.3724 | 2.89x | 0.423 | 0.0336 | 0-21.0 ms |
| LQR, ML-tuned weights | 0.1857 | 1.44x | 0.207 | 0.0522 | 0-33.25 ms |
| LQR, ML-tuned gains | 0.1333 | 1.02x | 0.163 | 0.0309 | 0-40.0 ms |

- **ML-tuned PID:** kp 0.510, ki 0.106, kd 0.097, kv_p 0.822, inside its range. A separate batch-tuning run with the cap at 2.0 also settled at 0.82 for the same cost.
- **The ML-tuned LQR gains come within 2.2% of the ML-tuned PID** (0.1333 against 0.1304), against 1.2% behind in the old table. The untuned cost-matched LQR is 3x worse, not within 1.8%: with a smooth goal and a heavy torque weight, a design that only regulates (no goal feed-forward) gives up a lot of tracking, and the gain search recovers nearly all of it.
- **Modeling the latency no longer matters.** The latency-ignoring design (0.3724) beats the delay-aware one (0.3895) and is stable at 10 ms (to 21 ms). Old finding 3 was a property of the light torque weight; at $w_u = 9$ the design is gentle enough to tolerate the delay.
- **Tuning still widens the stable-latency range:** 39.75 ms for the tuned PID, 40.0 ms for the tuned LQR gains and 33.25 ms for the tuned weights, against 30.75 ms for the cost-matched LQR.
- **Effort is a real term** (0.03 for the tuned PID and LQR gains, against 0.0002-0.0009 before).
- **The gain search's corner was the command feedback doing nothing.** In the earlier runs both command-feedback scales sat at the floor of their log range (0.05, then 0.01 when it was widened). With those gains held at exactly zero, the cost is unchanged (0.1333 against 0.1334) and the other three scales settle inside their range (velocity 1.03, pitch 0.37, pitch rate 0.69). The same holds for the pitch-rate weight, held at 0: the weight search's cost is identical (0.1857). So the in-flight command feedback and the pitch-rate term were not helping the tuned designs here; no edge effect remains in that row.
- **The weight search's corner, resolved (2026-11-06).** It came from the normalization, not the ranges. With the pitch weight fixed at 1, a design with *no* pitch weight can only be approached by sending every other weight to infinity. The search's corner ($w_v \to 0$, $w_u \to \infty$, $w_i/w_u \approx 3$) is exactly that design: per unit of torque weight it is $Q = \mathrm{diag}(10^{-6}, 0.01, 0, 3.08)$ (velocity, pitch, rate, integral) with $R = 1$. A scan of the robust score in that normalization (GPU, 300 scenarios per point) shows:
  - **Pitch and velocity weights:** the score falls monotonically as they go to 0, and is flat to $10^{-5}$ below $10^{-4}$, so the limit is exactly 0.
  - **Integral weight:** it then has an interior optimum. At fixed latency $q_i \approx 3.2$ (score 0.1837, against 0.1848 at the old corner); with friction and latency randomized, $q_i \approx 4.6$.
  - **Negative pitch weight:** letting it go negative (an indefinite $Q$, so no longer a standard LQR) keeps improving the score (0.1780 at $-0.05$).

  So the binding constraint is that $Q$ be positive semidefinite. Every such design keeps a pitch gain of about 1.27, while the tuned direct gains use 0.50. That is why no weight search, however wide, closes the gap to the direct gain search (0.186 against 0.133).

  What changed: `lqr_weight_space` now fixes the torque weight at 1, the one weight that must stay positive. `LQRConfig` holds the pitch, velocity and pitch-rate weights at 0, so only $w_i$ is searched, and its optimum is interior. Setting `w_pitch_fixed`, `w_velocity_fixed` or `w_pitch_rate_fixed` to `None` searches them again.
- **The weight search's corner (as recorded on 2026-10-24)** persisted with those two parameters held off: $w_v$ at its floor (1e-4) and $w_u$ at its ceiling (100), with $w_i$ at 308. Almost no velocity weight and very heavy torque and integral weights is a degenerate design. The search's cost improved once, when the ranges were widened (0.2157 to 0.1857), and did not change when the pitch-rate weight was held off. I have not tested whether widening further, or a different parameterization, would help; read this row as a limit of the weight search, not as the best LQR.
- The PID (0.1305, 0.1304, 0.1304) and the LQR with tuned gains (0.1339, 0.1334, 0.1333) kept their positions and nearly identical costs across the three runs of this section, so that comparison is stable.

Old table (earlier cost):

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
- **Latency was fixed at 10 ms during tuning** in these runs. *Done 2026-11-06:* latency and friction are now randomized (see the current results at the top of this section).
- **Give the direct gain search integral action, safely.** With friction it is now 18% behind the PID, for lack of an integral term. Searching the integral gain again needs a guard against the slow runaway described above, for example requiring the nominal loop's spectral radius to stay below about 0.999, or running longer validation scenarios.
- **The LQR knows nothing about noise.** A Kalman filter in front of it (LQG) is the natural next permutation. So is an LQR with the PID's lean cap, which would test finding 6.
- **Shared code with the pivot model.** `pivot_sim.lqr_gain` uses the same delay-aware scheme and could reuse `balance_sim.discrete_model`. It was left unchanged here.

### Reproducing

```
python optimization/run_lqr_comparison.py --backend torch --float32          the full run, ~7 min on the GPU
python optimization/run_lqr_comparison.py --replot results/lqr_comparison/<run>   redraw the figures only
python simulation/animate.py --csv results/lqr_comparison/<run>/trajectories/PID_ML-tuned.csv results/lqr_comparison/<run>/trajectories/LQR_cost-matched.csv
python -m pytest simulation/tests/test_lqr.py
```
