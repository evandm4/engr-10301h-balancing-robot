# Reviewer prompt: LEGO robot conversion (for Gemini)

Upload `legobot-review-bundle.xml` (built from `repomix.config.json` in this folder), then paste everything
below the line.

---

You are an adversarial code reviewer. The attached file is a repomix bundle of a research codebase: every
file is wrapped in a `<file path="...">` tag, with line numbers. Your job is to find what is wrong in the
work that converts a simulated balancing robot into a model of a real LEGO robot, plus the code that runs
on that robot. Find it before the expensive tuning runs start, because those runs will produce gains that
go onto real hardware. Assume there are bugs and find them. A review that finds nothing is only useful if
it shows exactly what was checked.

You cannot run code. Work by reading and by tracing values through the code by hand (units, signs, array
shapes, call order). Where a finding can only be settled by running something, say so, and give the exact
test or command that would settle it. Do not write a rewrite of the code; describe fixes precisely enough
to implement.

## Context

This is an undergraduate honors research project (Purdue ENGR 10301H, fall 2026). It started as a
simulation of a two-wheeled self-balancing robot. A cascaded PID (pitch loop inside a velocity loop: gains
kp, ki, kd on pitch and kv_p on velocity) has its gains tuned by CMA-ES against a robust objective. That
objective scores every candidate over 50 noise seeds × 3 start tilts, with robot parameters, latency and
other effects randomized per scenario, plus a chaos penalty. The result is checked on held-out seeds. The
paper's thesis is that ML tuning mainly buys robustness, not typical-case cost.

The project is now moving to a physical robot so the tuned gains can be tested on hardware and compared
with the simulator. The robot:
- two LEGO SPIKE Large Angular Motors (45602) driving 69 mm wheels directly;
- a Raspberry Pi 3B with a Raspberry Pi Build HAT, which drives the motors by PWM duty with no current
  sensing and streams encoder speed and position at 100 Hz over serial;
- an ICM-20948 IMU (gyro and accelerometer) on I²C;
- a 2S LiPo battery;
- a body of 0.362 kg with its centre of mass 60 mm above the axle.

The Build HAT runs in "slow PWM" mode (`pwmparams 1 0.01`). Bench tests found it behaves as
τ = |d|·a·(sgn(d)·V − k·ω) per motor, delivered as pulse-density modulation: 10 ms full-voltage pulses on a
tick grid locked to the HAT's 100 Hz data stream. The robot code is Python on the Pi, and it imports the
simulator's own controller, filter, encoder and actuator classes, so the same code runs in simulation and
on hardware. The robot's feedforward turns a torque command into a duty by calling the simulator
actuator's `drive()`, which is meant to be the exact inverse of the simulator's `torque()`.

There are three simulation engines:
- the reference engine (`simulation/balance_sim`);
- a batched NumPy engine (`simulation/batch_sim`), claimed bit-identical to the reference in float64;
- a PyTorch engine (`batch_sim/torch_sim.py`), used for the large tuning runs.

Every model change is supposed to exist in all three.

**Status.** These are done:
- the LEGO config and robot selection (A1, A2);
- the scenario derivation (A3) and baselines (A4);
- the voltage-mode and slow-PWM actuator models (B1, B1b);
- the simulated IMU with a complementary filter (B2) and the encoder model (B3);
- latency folded into one delay (B4) and the per-scenario pitch offset (B6);
- bench measurements that replaced the estimated motor and sensor constants (C3, C4);
- the robot code: loop, safety trips, calibration, logging, and locking the loop to the HAT's clock
  (D1 to D5).

**Next comes phase E:** large CMA-ES tuning runs on the LEGO config (development seeds first, then final
runs and a frozen validation set touched once), then the first real balancing tests. `SOFTWARE_PLAN.md`
has the full plan.

**Decisions already made, not to re-argue** unless you find a concrete correctness consequence:
- The LQR is dropped for the LEGO robot. Ignore all LQR code paths.
- Cascaded PID only.
- Cruise speed is 0.40 m/s, with speed changes softened to half of the Pololu robot's lean shares.
- Slow-PWM mode on the HAT.
- The robot code is Python on the Pi.
- The earlier simulated robot ("Pololu", `config.py`'s default) stays as it is. The LEGO robot is a
  separate config (`legobot_config.py`), selected by `ROBOT=legobot` or `--robot legobot`.

**Known open items, not findings:**
- The balance-point calibration and the battery discharge log are not yet measured.
- The slow-PWM pulse ripple at low duty needs re-measuring.
- The gearbox backlash model (B5) exists in the reference engine only and still has to be ported to the
  batched engines.

If one of these makes something else wrong, report that consequence.

**Seed tiers:**
| Tier | Seeds | Use |
|---|---|---|
| Training | 1–50 | tuning |
| Smoke | 5000–5009 | quick checks |
| Development | 5000–5049 | development decisions |
| Validation | 3000–3099, frozen | final numbers |
| Fall rate | 6000–6999, frozen | final fall rates |

Retired seed sets must never be reused.

**Documentation convention.** The markdown files are lab notebooks: new sections are added and old ones
marked "superseded", and the newest section is the current state. Don't report superseded sections.
**Do** report a doc that calls itself current but disagrees with the code, or a number in the code that
disagrees with the measurement it cites, when that could mislead a decision.

## What the bundle contains

**Included:**
- `config.py` and `legobot_config.py`;
- everything under `hardware/legobot/`: the datasheet, software plan, baselines and handoff notes;
  `legobot_scenario.py`, `legobot_baselines.py` and `pwm_pulse_check.py`; the robot code in `robot/`; the
  measurement scripts in `robot/measure/`;
- the bench fit summaries (`bench_data/*_fit.json`), but **not** the raw bench CSVs, which are too large.
  Don't flag the missing CSVs; judge the fits by their code and by the summaries;
- all of `simulation/balance_sim/` and `simulation/batch_sim/`, with the simulation tests relevant to the
  conversion;
- the batch tuning code, which the phase E runs will use;
- `PHYSICS.md`, `ARCHITECTURE.md` and `README.md`.

**Left out on purpose:**
- the results folders, the archive and the pivot-robot variant;
- the LQR comparison and the neural-network baseline work. `policy.py` files are present only because the
  simulator imports them, so don't review them.

If you need a file that isn't there, name it and say what you would check in it.

## Read first

1. `hardware/legobot/SOFTWARE_PLAN.md`: what was meant to be built, and the "as built" notes.
2. `hardware/legobot/DATASHEET.md`: sections 2, 3.3, 3.4, 5, 6, 7 and 8.7.
3. `legobot_config.py`, then `config.py` (`load_config`, `ProjectConfig`, `ActuatorConfig`,
   `RobustnessSettings`, `ScenarioConfig`, `CostConfig`).
4. `simulation/balance_sim/actuators.py` (`VoltageModeActuator`, `SlowPwmActuator`), `sensors.py`,
   `simulator.py` and `controllers.py`.
5. `simulation/batch_sim/physics.py`, `simulate.py`, `schedule.py`, `robot.py`, `torch_sim.py`.
6. `hardware/legobot/robot/control.py`, `balance.py`, `buildhat_serial.py`, `imu.py`, `calibrate.py`,
   `loop_config.py`, `sim_hw.py`, `test_control.py`.
7. `hardware/legobot/legobot_scenario.py` and `A4_BASELINES.md`.
8. `optimization/batch_tuning/evaluation.py`, `robust.py`, `optimization/tuning/search_space.py`.
9. The tests: `simulation/tests/test_slow_pwm.py`, `test_equivalence.py`, `test_torch.py`,
   `test_imu.py`, `test_encoder.py`, `test_voltage_mode.py`, `optimization/tests/test_config.py`,
   `test_robust.py`.

Open the code before accepting any claim a doc makes about it.

## What to attack

1. **Sim-to-real parity of the shared code path.** Trace one control tick on the robot (`control.py`
   `Brain.sense` → `Brain.act` → `Feedforward.duties` → the HAT command string) next to the same tick in
   the simulator. Check:
   - **Units:** HAT speed and position units against rad and rad/s; degrees against radians.
   - **Sign conventions:** the mirrored motor, `PORT_SIGN`, the IMU axis mapping, the sign of pitch,
     velocity and torque.
   - **Wheel speed:** count difference over one 10 ms HAT period, and v = r(ω̄ + gyro): does this match
     the simulator's `Encoder` and its definition of relative wheel speed?
   - **Estimator inputs:** the complementary filter's τ and rate, gyro bias, pitch zero.
   - **Torque split** τ/2 per wheel against the simulator's total torque, and the duty cap against the
     simulator's clip.
   - **HAT number formatting:** does writing the duty with limited decimals quantize it?

   Anything the simulator models differently from what the robot does is a finding.
2. **`drive()` as the inverse of `torque()`.** For d = τ / (a·(V ∓ k·ω̂)): how is the sign branch chosen
   near ω ≈ 0 and when τ and ω have opposite signs? What happens as k·ω̂ approaches V (division blow-up,
   sign flip)? Does clipping to ±1 then lose the inverse property in a way the controller doesn't see?
   - Is ω̂ (measured, delayed, quantized) used on the robot exactly as in the simulator?
   - Is V the measured vin on the robot and something consistent in the simulator?
   - Is friction handled the same way on both sides?
   - Check the per-robot mismatch draws between the controller's constants and the true constants.
3. **The slow-PWM and pulse-density model** (`SlowPwmActuator`, `pdm_update`, `pwm_tick`,
   `pwm_tick_phase`, `pwm_steps_per_tick`, `pwm_tick_offset`). Does the pulse-density accumulator produce
   the measured behaviour (10 ms full-voltage pulses at about 100 × duty per second)? Check the tick
   timing relative to samples and latency, and whether the delayed duty is quantized at the right moment.
   Check what happens at d = 0 (coast against brake) and on a sign change. Do the three engines implement
   this identically, including the phase draw per scenario?
4. **Three-engine equivalence.** Look for any feature that one engine has and another lacks or does
   differently: slow PWM, ticks, the encoder hold and quantization, the IMU and filter state, the
   per-scenario pitch bias, latency, friction, backlash. Then check whether `test_equivalence.py` and
   `test_torch.py` actually exercise each LEGO path with non-trivial settings (randomization on, nonzero
   bias, slow PWM on, several tick phases), or whether they pass with features switched off. Can the
   float32 torch path flip fall detection or pulse decisions compared with float64 often enough to bias
   the tuning?
5. **Robot selection and config leakage.**
   - Can any script, test or tuning run silently use the Pololu config when LEGO was asked for? Look at
     import-time `from config import CONFIG` bindings, when the environment variable is read, and default
     arguments.
   - Are results kept apart (`results/legobot/...`), and does the snapshot record the robot?
   - `legobot_config.py` reuses some of `config.py`'s sections as "not about the robot". Check that each
     is really robot-independent: gain search ranges, integral limits, controller limits, chaos penalty,
     and everything scaled in N·m or m/s for a robot with about a seventh of the torque.
6. **The scenario derivation** (`legobot_scenario.py`).
   - The catch test claims a full-torque response is a bound no controller can beat, so a fall is the
     controller's fault. Does that hold under the actuator model the tuning actually uses (slow PWM with
     pulse-density modulation and ticks)? Or does the derivation use a different drive model, such as
     `voltage_drive`, so the bound doesn't apply?
   - Does its 2^11-corner box match the randomization the tuner actually draws, including draws added
     later (tick phase, feedforward mismatch, pitch bias, latency range)?
   - Does the worst run without a fall really stay under the fall-penalty bound?
   - Check the cost limits (0.30 rad, 0.06 m/s, 0.09 N·m) and that the goal profile's leans stay under
     the cap.
7. **Randomization against reality.** The spreads were "shrunk to measurement uncertainty." Is that too
   narrow for sim-to-real transfer? Look at:
   - the left/right motor differences, and the friction estimates from two methods that disagree by
     about 35%;
   - the battery voltage over a discharge, from 7.6 V nominal down to the 6.6 V cutoff: is V drawn
     across that range?
   - the latency range against the measured loop timing, IMU and encoder delay differences, and the
     IMU's mount height and filter lag.

   Name any real-world effect that a tuned controller could depend on but the hardware won't provide.
8. **Robot runtime safety and real-time behaviour** (`balance.py`, `control.Safety`, `LineClock`, `Pacer`,
   `buildhat_serial.py`, `run_safe.sh`, `kill9_test.sh`).
   - Can the watchdog's raw serial write, which bypasses the driver's lock, interleave with a normal
     command and produce a malformed line the HAT misreads?
   - Is there a path where the process dies (exception, `kill -9`, Pi brown-out) and the HAT keeps
     driving the last duty?
   - Check the stall-trip thresholds against real stall behaviour, the arming conditions, line-clock lock
     loss and re-lock, and the pacer's re-phasing.
   - Check exception handling in the reader thread, file I/O inside the real-time loop, and the startup
     race with SIGINT.
   - List any hazard to the robot or a person that no trip covers.
9. **The measurement scripts and fits** (`robot/measure/fit.py`, `motor_steps.py`, `lever.py`,
   `pwm_probe.py`, `tick_lock.py`, `imu_still.py`, `latency.py`, `backlash.py`). Check whether a fit
   ignores a term that biases a constant (Coulomb friction in a spin-up fit, back-EMF in a stall fit) or
   gets its uncertainty wrong. Do the values in `legobot_config.py` match the `_fit.json` summaries they
   cite?
10. **Tests that don't test.** Find tests that are tautological (they check a formula against itself, as
    a `drive()`/`torque()` round trip through shared code might), tolerances loose enough to hide a sign
    or unit error, or claims in the docs that no test covers. List the missing tests that matter most.
11. **Readiness for phase E.** What would waste or invalidate an expensive tuning run? For example:
    - the optimizer exploiting a simulator artifact (tick phase, kd amplifying pulse noise, the integrator
      against the pitch bias);
    - search bounds made for the Pololu robot;
    - a cost term that doesn't scale to this robot;
    - seed overlap or leakage (the hand-tuned gains were picked on the smoke seeds, which are part of the
      development set);
    - nondeterminism, or a gap in what the snapshot records.

## How to report

- **Start with a verdict paragraph:** ready for phase E, ready after the listed fixes, or not ready, and
  why.
- **Then the findings, most severe first.** Number them G1, G2, ... so they can be triaged. For each:
  - **Severity:**
    - *blocker:* wrong gains or unsafe hardware behaviour;
    - *major:* a biased result or a real sim-to-real gap;
    - *minor:* a real but small consequence.
  - **Area:** one of the 11 above.
  - **Location:** `path:line` from the bundle's line numbers, for every place involved.
  - **The claim being tested,** quoted or paraphrased from the code or docs.
  - **What is wrong.**
  - **A concrete failure scenario:** specific inputs or state, and the wrong output or behaviour.
  - **A specific fix,** and the test that would have caught it.
  - **Basis:** verified by tracing the code, or judgment. Give a confidence (high, medium or low).
- **Then "checked and sound":** a short list of the claims you traced and found correct, one line each,
  so the coverage is visible.
- **Then "questions for the author":** what you couldn't decide from the bundle.
- **End with the five changes you would make before any phase E run,** in order.

Don't pad. Leave out style nits, naming, and anything you can't tie to a real consequence for the
simulation results or the hardware. Don't report the same root cause twice; list the places it shows up
under one finding.
