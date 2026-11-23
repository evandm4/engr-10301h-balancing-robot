# A4: baselines on the LEGO robot

SOFTWARE_PLAN A4, 2026-11-22. The LEGO robot (`legobot_config.py`) on its derived scenario (A3,
`legobot_scenario.py`), with the voltage-mode driver since B1, the simulated IMU since B2 and the encoder model since B3. These are the reference points the
LEGO tuning runs (E1, E2) are compared against. Re-score with `python hardware/legobot/legobot_baselines.py`.

## Scores with the measured body (current)

Body measured (DATASHEET 1, as built): mass 0.362 kg (440 g total minus the wheels), COM 0.060 m above
the axle, pitch inertia 8.8e-4 kg m^2; spreads ±1%, ±8%, ±15%. The sections below used 0.44 kg (the
total mass, entered by mistake as the body mass) or the 0.55 kg estimate, a 0.09 m COM and 0.0035 kg m^2.
Scenario re-derived (legobot_scenario.py): lean cap 15°, cruise 0.27 m/s, pushes 0.63 N (97% of the
0.0325 N s catchable on every corner), tilts 0.03 / −0.09 / 0.18 rad, pitch_limit_rad 0.18, velocity_limit
0.04 m/s, torque_limit 0.09 N m (weights 20.25 and 4.0). Costs are not comparable with the sections below.

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev chaos |
|---|---|---|---|---|
| Hand-tuned PID (6, 2, 0.5, 0.3) | 4.404 (0) | 3.949 (0) | 3.696 (0) | 0.048 |
| Model-based PID (2.80, 2.94, 0.27, 0.52) | 2.133 (0) | 172.4 (14) | 159.1 (13) | 14 |
| Cost-matched LQR | 3.010 (0) | 186.4 (15) | 220.9 (18) | 47 |

## Scores with the pitch offset drawn per robot (B6; body estimated, superseded)

Robust scoring now draws each scenario's pitch offset (the accelerometer's) uniformly in ±0.015 rad
instead of +0.015 rad every time; plain runs keep +0.015. Everything else as below.

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev chaos |
|---|---|---|---|---|
| Hand-tuned PID | 2.405 (0) | 1.766 (0) | 1.784 (0) | 4.5e-3 |
| Model-based PID | 2.136 (0) | 220.3 (18) | 159.8 (13) | 24 |
| Cost-matched LQR | 2.039 (0) | 14.94 (1) | 39.33 (3) | 0.11 |

The hand-tuned PID's cost fell (its mean offset is now 0, and its integral handles either sign); the
model-based PID falls more (13 of 150, from 9), so the sign of the offset matters to it.

## Scores with 0.44 kg body mass (superseded)

Body mass set to 0.44 kg (the robot's total mass, entered by mistake for the body; was the 0.55 kg estimate). The scenario was re-derived
for it: lean cap 13° (was 12°), pushes 0.85 N (was 0.90 N; 98% of the 0.0434 N s catchable on every
corner), gentle stop 1.2 s. The model-based design follows the robot; the hand-tuned gains don't.

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev chaos |
|---|---|---|---|---|
| Hand-tuned PID (6, 2, 0.5, 0.3) | 2.405 (0) | 2.716 (0) | 2.668 (0) | 4.5e-3 |
| Model-based PID (3.44, 3.09, 0.38, 0.34) | 2.136 (0) | 111.4 (9) | 109.0 (9) | 0.068 |
| Cost-matched LQR | 2.039 (0) | 3.185 (0) | 27.42 (2) | 0.15 |

### Feedforward fixes tried (dev seeds, 0.55 kg body, before the mass update)

All 150 development scenarios in the reference engine (it reproduces the batched scores exactly).
Mean cost (falls of 150):

| Robot code | Hand-tuned PID | Model-based PID | Cost-matched LQR |
|---|---|---|---|
| Feedforward as configured (gain 1) | 2.64 (0) | 99.4 (8) | 39.5 (3) |
| Velocity from the encoder plus the gyro averaged over the same 10 ms | 2.62 (0) | 99.2 (8) | 38.5 (3) |
| Feedforward gain 0.8 / 0.9 | 13.1 (1) / 3.08 (0) | 269 (23) / 158 (13) | 39.9 (3) / 39.7 (3) |
| Feedforward gain 1.05 / 1.1 / 1.2 / 1.3 | 2.48 / 2.35 / 2.16 / 2.05 (0) | 74.9 (6) / 48.2 (4) / 39.0 (3) / 132 (11) | 39.4 (3) / 39.4 (3) / 49.4 (4) / 112 (9) |
| Oracle: feedforward given the true wheel speed at the tick | 2.58 (0) | 50.7 (4) | 26.2 (2) |
| Oracle speed and the robot's exact constants | 2.53 (0) | 26.2 (2) | 1.98 (0) |

The stale wheel speed in the feedforward is half of the problem and the constants' mismatch the other
half. Leaving some back-EMF uncancelled (gain < 1) is worse: the drag that remains feeds the velocity
loop's runaway (PHYSICS.md 9.4). Over-cancelling a little offsets the stale speed, but the best gain
depends on the controller (hand-tuned keeps improving to 1.3, model-based is best near 1.2, the LQR at
1.0–1.1). So `ActuatorConfig.voltage_feedforward_gain` (default 1.0) is a robot-code setting to tune
together with each controller's gains in E1. Syncing the gyro to the encoder interval changed nothing
and was not kept.

## Scores with the encoder model (B3, superseded by the mass update)

Since B3 the velocity the controller (and the feedforward) sees comes from the motors' encoders as the
Build HAT reports them: 1° counts sampled at 100 Hz, 5 ms in transit, held, differenced 10 ms apart.
IMU and voltage-mode driver as before; gains and scenario unchanged.

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev chaos |
|---|---|---|---|---|
| Hand-tuned PID | 2.381 (0) | 2.692 (0) | 2.645 (0) | 4.4e-3 |
| Model-based PID | 1.920 (0) | 74.98 (6) | 99.39 (8) | 12 |
| Cost-matched LQR | 1.946 (0) | 3.068 (0) | 39.52 (3) | 0.12 |

Dev seeds, same robots, one change at a time (mean cost, falls of 150):

| Baseline | As scored | Robot code's constants exact | Encoder off (white velocity noise) | Current-mode driver |
|---|---|---|---|---|
| Hand-tuned PID | 2.64 (0) | 2.59 (0) | 2.61 (0) | 2.52 (0) |
| Model-based PID | 99.4 (8) | 74.6 (6) | 26.5 (2) | 2.01 (0) |
| Cost-matched LQR | 39.5 (3) | 2.84 (0) | 1.77 (0) | 2.36 (0) |

The new falls need the voltage-mode driver: with current mode the encoder costs nothing in falls. The
feedforward cancels back-EMF with a wheel speed that is 5–15 ms old (held and in transit), which leaves a
delayed drag of c_m (ω − ω̂) on the motors. The model-based PID's falls all come at 3.4–4.0 s, just after
the climb to cruise speed, mostly from the 0.14 rad tilt at 20–25 ms latency, and its chaos term (12)
says the outcome flips with a 1e-4 rad nudge: the loop is near an instability there. The hand-tuned PID
is unaffected. Robot-code options to try later (none modeled yet): a smaller feedforward gain, or
predicting the wheel speed forward over its age from the commanded torque.

Plain-scenario sensitivity (reference engine, 9 runs, no falls in any case):

| Velocity sensing | Hand-tuned | Model-based | LQR |
|---|---|---|---|
| White noise 0.025 m/s (B2) | 2.336 | 1.823 | 1.511 |
| Encoder, 5 ms transit (configured) | 2.381 | 1.920 | 1.946 |
| Encoder, no transit delay | 2.354 | 1.864 | 1.708 |
| Encoder, 10 ms transit | 2.406 | 1.990 | 2.293 |
| Encoder differenced 5 ms apart | 2.644 | 2.003 | 2.960 |

## Scores with the simulated IMU (B2, superseded by B3)

Since B2 the controller sees the robot code's complementary-filter estimate (τ = 0.5 s) from a simulated
gyro and accelerometer (10 cm above the axle), instead of true pitch plus white noise. Voltage-mode
driver as in B1; gains and scenario unchanged.

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev worst |
|---|---|---|---|---|
| Hand-tuned PID | 2.336 (0) | 2.659 (0) | 2.615 (0) | 4.09 |
| Model-based PID | 1.823 (0) | 14.27 (1) | 26.46 (2) | 1826 |
| Cost-matched LQR | 1.511 (0) | 1.774 (0) | 1.767 (0) | 3.57 |

The plain costs rose 7–52% (the PIDs most): the accelerometer reads acceleration as tilt, so the
estimate lags and leans during every speed change. The LQR's falls went from 8 to 0 and the model-based
PID's from 3 to 2: the old model's 0.005 rad of white pitch noise per tick was harsher than a filtered
estimate (the accelerometer's 0.0135 rad per sample leaves about 1e-3 rad after the filter).

Sensitivity on the plain scenario (mean cost, no falls in any case; reference engine):

| Sensor | Hand-tuned | Model-based | LQR |
|---|---|---|---|
| Noise model (B1) | 1.584 | 1.196 | 1.408 |
| IMU, accel noise 0.0135, τ 0.5 s, d 0.10 m (configured) | 2.336 | 1.823 | 1.511 |
| Accel noise at the datasheet's 0.00135 | 2.330 | 1.817 | 1.509 |
| τ = 1.0 s | 2.908 | 2.125 | 1.647 |
| τ = 0.25 s | 2.109 | 2.068 | 1.528 |
| IMU at d = 0.05 m | 2.291 | 1.807 | 1.511 |

## Scores with the voltage-mode driver (B1, superseded by B2)

Since B1 the LEGO robot uses the voltage-mode driver with back-EMF feedforward, and the scenario was
re-derived for it (pushes 0.90 N, tilts up to 0.14 rad, pitch_limit_rad 0.14; legobot_config.py).
Gains unchanged.

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev worst |
|---|---|---|---|---|
| Hand-tuned PID | 1.584 (0) | 14.81 (1) | 1.858 (0) | 7.10 |
| Model-based PID | 1.196 (0) | 14.77 (1) | 36.54 (3) | 1881 |
| Cost-matched LQR | 1.408 (0) | 97.08 (8) | 100.0 (8) | 1958 |

What makes them fall, on the dev seeds (same robots, latencies and noise):

| Baseline | Robot code's constants nominal (as scored) | Constants exact for each robot | Current-mode driver |
|---|---|---|---|
| Hand-tuned PID | 1.858 (0/150) | 1.776 (0/150) | 1.753 (0/150) |
| Model-based PID | 36.54 (3/150) | 1.465 (0/150) | 1.253 (0/150) |
| Cost-matched LQR | 100.0 (8/150) | 35.86 (3/150) | 1.616 (0/150) |

The falls are robots whose true R is 14–17% above the robot code's value and/or whose k is 9–14%
below it (the torque delivered is about 0.8 of the command and the feedforward over-cancels the
back-EMF), with 1.3–1.5x rotor inertia and 17.5–25 ms latency, mostly from the 0.14 rad tilt. The
LQR also falls with exact constants: the feedforward uses a wheel speed sampled one latency earlier,
which acts as a delayed drag (c_m = 0.051 N m s/rad) that its design doesn't know about. Both are
effects the real robot has. Measuring R and k on the bench (C3) narrows the first; tuning on the
robust scenario (E1) has to handle what remains, including R rising as the winding warms.

## Scores with the current-mode driver (A4, superseded)

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

## Measured motor (C4, 2026-11-23)

After `legobot_config` took the bench-measured motor (DATASHEET 8.7: k 0.348, a = ηk/R 0.0597 N m/V,
rotor inertia 5.45e-4, friction 0.042 N m + 0.0017 N m s/rad, max_torque 0.90; tighter spreads), with the
scenario and cost **unchanged** (still the 0.27 m/s scenario), `python hardware/legobot/legobot_baselines.py`:

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev worst | Dev chaos |
|---|---|---|---|---|---|
| Hand-tuned PID (6, 2, 0.5, 0.3) | 24.23 (0) | 24.88 (0) | 24.90 (0) | 50.1 | 0.38 |
| Model-based PID (1.55, 1.75, 0.14, 0.52), re-designed | 2.624 (0) | 3.128 (0) | 3.036 (0) | 8.16 | 0.028 |
| Cost-matched LQR | 29.77 (0) | 76.38 (4) | 63.56 (3) | 1845 | 49 |

The robot is much lighter on the wheel side (2J/r² 0.92 kg, was 2.05) and stronger, so the old hand-picked
gains no longer suit it. The hand-tuned PID still never falls, but it costs 10× more. The model-based design
now has no falls. Not yet redone: the scenario (A3 re-derivation with these constants gives a 0.51 m/s
cruise and flags two inconsistencies: SOFTWARE_PLAN C4 status) and new hand-tuned gains.

## Slow-PWM actuator (B1b, 2026-11-23)

The same measured robot and unchanged scenario, with the actuator model switched from "voltage" to
"slow_pwm" (`SlowPwmActuator`: torque = |d| a (sgn(d) V − kω), the robot code sending the inverse's duty):

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev worst | Dev chaos |
|---|---|---|---|---|---|
| Hand-tuned PID (6, 2, 0.5, 0.3) | 27.64 (0) | 28.17 (0) | 28.16 (0) | 53.0 | 0.26 |
| Model-based PID (1.55, 1.75, 0.14, 0.52) | 2.246 (0) | 2.072 (0) | 2.054 (0) | 3.49 | 2.4e-3 |
| Cost-matched LQR | 28.05 (0) | 25.10 (0) | 24.83 (0) | 42.5 | 0.63 |

No baseline falls. With constant errors the slow-PWM driver's torque error scales with the command and adds no
drag at zero command, so the robust scores are steadier: the model-based PID's worst dev run falls from 8.2
to 3.5, and the LQR goes from 3 falls to none. The hand-tuned gains still need redoing for the measured robot.

## Slow PWM with pulse-density modulation (B1b, 2026-11-23)

`legobot_config` now models the HAT's measured pulse-density modulation (`pwm_tick_s` = 10 ms; DATASHEET 8.7),
with the ticks starting half a control period after the control samples. Same robot and scenario:

| Baseline | Plain (seeds 1–3, 9 runs) | Robust, train seeds 1–50 (150) | Robust, dev seeds 5000–5049 (150) | Dev worst | Dev chaos |
|---|---|---|---|---|---|
| Hand-tuned PID (6, 2, 0.5, 0.3) | 39.94 (0) | 72.63 (3) | 98.41 (6) | 1812 | 62 |
| Model-based PID (1.55, 1.75, 0.14, 0.52) | 3.755 (0) | 3.625 (0) | 3.587 (0) | 4.84 | 0.13 |
| Cost-matched LQR | 241.2 (1) | 244.3 (18) | 304.2 (23) | 1860 | 180 |

The model-based PID barely notices the pulses: its cost goes from 2.05 to 3.6, with no falls. The LQR and the
hand-tuned PID now fall. The LQR is also very sensitive to the **phase** of the HAT's ticks relative to the
control loop, which is not synchronized on the robot. Nine plain runs, with the ticks starting 0 / 2.5 / 5 /
7.5 ms after a control sample (reference engine), cost (falls):

| Baseline | Averaged | 0 ms | 2.5 ms (configured) | 5 ms | 7.5 ms |
|---|---|---|---|---|---|
| Model-based PID | 2.2 (0) | 3.8 (0) | 3.8 (0) | 3.7 (0) | 3.7 (0) |
| Cost-matched LQR | 28.1 (0) | 445 (2) | 241 (1) | 29.1 (0) | 26.5 (0) |
| Hand-tuned PID | 27.6 (0) | 38.4 (0) | 39.9 (0) | 32.5 (0) | 31.9 (0) |

So the tick phase should be drawn per scenario in robust scoring (SOFTWARE_PLAN B1b, next step) before
any final tuning. Until then, gains tuned with this config are tuned for one phase.

### With the tick phase drawn per scenario (2026-11-23)

`ROBUSTNESS.randomize_range` now draws `pwm_tick_phase` uniformly over a whole tick for each scenario. The
HAT's ticks are locked to its own clock, not the robot's loop (DATASHEET 8.7, `tick_lock`). Plain runs keep
0.25.

| Baseline | Plain (9) | Robust train (150) | Robust dev (150) | Dev worst | Dev chaos |
|---|---|---|---|---|---|
| Hand-tuned PID | 39.94 (0) | 61.07 (2) | 92.19 (5) | 1822 | 81 |
| Model-based PID | 3.755 (0) | 3.695 (0) | 3.677 (0) | 5.02 | 0.16 |
| Cost-matched LQR | 241.2 (1) | 402.2 (31) | 368.5 (28) | 1859 | 230 |

Robust scores now average over the phase instead of resting on one value. The model-based PID is
unaffected, and the LQR falls in about a fifth of the scenarios. That is the design's fragility under
pulse-density PWM, not luck with the phase.
