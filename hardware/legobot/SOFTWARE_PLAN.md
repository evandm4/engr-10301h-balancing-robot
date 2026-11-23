# Software plan for the LEGO build

What has to change in the simulation, tuner and new robot code before the LEGO robot (see
[DATASHEET.md](DATASHEET.md)) can run tuned gains. Started from a seven-item list written for the Pololu build
(PHYSICS.md 9–10) and reconciled with this build's datasheet. 2026-11-22.

## 1. What changes from the earlier list

| Earlier item | For the LEGO build | Why |
|---|---|---|
| 1. Motor driver: DRV8874 current loop, or back-EMF feedforward, or `VoltageModeActuator` | **No current loop is possible.** Use back-EMF feedforward on the robot, and simulate the voltage-mode driver *with* that feedforward and its errors | The Build HAT has no current sensing; its command is PWM duty (DATASHEET 3.3). Feedforward needs R and k_e (uncertain) and motor speed (100 Hz, a few ms old), so its errors must be in the model. Uncompensated drag is 0.051 N m s/rad, 3.5× the Pololu motor's (DATASHEET 5.5) |
| 2. Simulate accelerometer, gyro and filter | **Kept, and extended** with the IMU's height above the axle and the IMU's filter lag | DATASHEET 3.4 and 7.9: the sim hands the controller true pitch plus noise |
| 3. Encoders: 48 CPR, differenced at 200 Hz | **360 counts per output turn, updated at 100 Hz, ±3° slack**; difference 10 ms apart | DATASHEET 3.1, 5.4, 7.4. Differencing at 200 Hz would read zero half the time |
| 4. Firmware with the cascade or LQR, gains from JSON, τ/2 per wheel, 2 A limit | **Robot code is Python on the Pi**, so it can import the simulator's own controller classes. No current limit exists; the torque cap and a duty cap replace it | Same code on both sides removes a whole class of sim-to-real bugs (units, anti-windup, lean cap) |
| 5. Safety and startup | **Kept**, plus stall detection (no current limit to protect the motors), LiPo cutoff from the HAT's `vin`, and coasting on a Build HAT "Motor power fault" | DATASHEET 3.3, 3.6 |
| 6. Calibration and sign checks | **Kept**, plus IMU axis mapping, accelerometer offset, and `plimit 1` / `bias 0` at startup | DATASHEET 3.4, 7.2 |
| 7. Logging and sim-to-real tools | **Kept**; measurement scripts follow DATASHEET section 8 | |
| (not listed) | **A separate LEGO configuration**, with config.py's Pololu robot preserved | Your requirement: both configs kept |
| (not listed) | **A LEGO scenario and cost limits.** The current scenario cruises at 1.3 m/s; this robot tops out at 0.54–0.67 m/s with no load | DATASHEET 7.3. Tuning on the current scenario would score every robot as falling |
| (not listed) | **Latency range 5–25 ms and the motor's 3 ms L/R** | DATASHEET 6, 7.6 |

## 2. Constraints from the codebase

- **Three engines.** Every physics or sensor change goes into the reference simulator (`simulation/balance_sim`),
  the NumPy batched engine (`simulation/batch_sim`, which must stay bit-identical to the reference in float64;
  `simulation/tests/test_equivalence.py`) and the PyTorch engine (`batch_sim/torch_sim.py`, `test_torch.py`). That
  triples the cost of each model change, so each change is first tried in the reference engine with a quick
  experiment (as `physics_checks.py` does) and ported only if it matters.
- **One config, imported everywhere.** About 30 files do `from config import CONFIG`. A second robot needs a
  selection mechanism, not a second copy of each script.
- **Gains are only comparable under one robot, scenario and cost** (CostConfig's note). LEGO results start a new
  series; the Pololu results stay valid for the Pololu robot.
- **Seed discipline** (your evaluation-seed tiers): develop on smoke and dev seeds; the frozen validation set is
  touched once, at the end, for the LEGO robot's final numbers.

## 3. The plan

Five phases. Phases A and B need no hardware; phase C needs the assembled robot and can run alongside B. Sizes
are rough: S is under a day, M a few days, L about a week.

### Phase A: LEGO configuration and scenario (no hardware)

| # | Task | Size | Done when |
|---|---|---|---|
| A1 | `legobot_config.py` at the root, built like `pivot_config.py`: a `ProjectConfig` whose robot, actuator, sensor and timing sections carry DATASHEET section 2's values and spreads. config.py keeps the Pololu robot unchanged | S | Both configs build; `snapshot()` records which robot |
| A2 | Robot selection: config.py exposes `load_config(name)`; scripts take `--robot {pololu,legobot}` (default pololu), or an environment variable `ROBOT=legobot` read once in config.py, so scripts work without edits. Results go to `results/<robot>/...` | S–M | Every run script and `animate.py` work for both robots; tests run against both |
| A3 | Re-derive the scenario and cost limits for the LEGO motors, the way ScenarioConfig's docstring did for the Pololu motors: top speed with headroom on the weakest randomized motor, the blend times from the lean the acceleration needs, push sizes from the largest catchable impulse, start tilts, `velocity_limit` and `torque_limit`, and a re-check of the fall-penalty bound | M | Every element of the LEGO scenario is tied to a number in DATASHEET; a full-torque response catches every push on every robot corner |
| A4 | Hand-tuned and model-based baselines for the LEGO robot (`pid_design` already designs from the linear model) | S | Baseline scores on the LEGO scenario |

### Phase B: Model changes (no hardware)

Order is by expected effect. Each starts in the reference engine with a short experiment, then is ported.

| # | Task | Size | Done when |
|---|---|---|---|
| B1 | **Voltage-mode actuator with feedforward.** In `actuators.py`: duty = clip((R̂ i_cmd + k̂_e ω̂)/(plimit V), ±1), applied voltage = duty × plimit × V, current from the true R and k_e. ω̂ is the motor speed the robot code sees (B3), R̂ and k̂_e are the robot code's constants. Options: feedforward off (plain voltage mode), and a dead-band jump if the bench shows one (current firmware dropped `bias`; `pwmparams` may set a minimum PWM instead, to be checked in C1). Robustness draws the mismatch R̂/R and k̂_e/k_e per robot (±17%, ±15%, the datasheet spreads). Replaces the current-mode `DCMotorActuator` for the LEGO robot; the Pololu robot keeps its model | M (reference) + M (port) | Reproduces PHYSICS.md 9.3's numbers with feedforward off on the Pololu robot; equivalence and torch tests pass |
| B2 | **Simulated IMU and filter.** Sensor model outputs gyro rate (noise, bias, DLPF modeled as a first-order lag at the chosen bandwidth) and accelerometer tilt from ẍ, θ̈ and the mount height d (plus noise and offset). A complementary filter, the same code the robot will run (section 4), turns them into the pitch estimate the controller sees. ẍ and θ̈ come from the dynamics at the sample time | M (reference) + M–L (port: the filter adds state to every simulation) | With noise off and the robot still, the estimate converges to the truth; during the scenario's accelerations the error has the size DATASHEET 3.4 predicts |
| B3 | **Encoder model.** Wheel angle quantized to 1°, updated every 10 ms and held between updates, read with a configurable delay; velocity from counts 10 ms apart. Gearbox slack as a ±3° position error, then (B5) as a real dead zone if needed | S–M | Velocity noise matches DATASHEET 5.4 (0.025 m/s at 69 mm) |
| B4 | **Timing.** Latency range 5–25 ms; the motor's L/R (3 ms) either folded into latency or as a first-order current lag in the actuator. Decide by experiment in the reference engine | S | Decision recorded in PHYSICS.md |
| B1b | *Done 2026-11-23:* `SlowPwmActuator` in the reference engine, `slow_pwm` in `batch_sim.physics` (NumPy and torch share it), config model `"slow_pwm"`; equivalence, torch and unit tests (`test_slow_pwm.py`) pass; matches the bench (DATASHEET 8.7). Measured since: slow PWM is pulse-density modulation, with 10 ms full-voltage pulses at about 100 × duty per second (`pulse_imu`). In the reference engine it causes no falls, but raises the model-based PID's cost 65% (`pwm_pulse_check.py --model pdm`, DATASHEET 8.7). *Ported 2026-11-23:* `SlowPwmActuator.pwm_tick` / `ActuatorConfig.pwm_tick_s` (10 ms in `legobot_config`); the loop quantizes the delayed duty once per tick (`pdm_update`) in all three engines, with ticks starting half a control period after the samples; bit-identical across engines (tests). The model-based PID is unaffected (3.6, no falls), but the LQR and hand-tuned PID now fall, and the LQR swings between 26 and 445 with the tick phase (A4_BASELINES.md). *Tick phase, 2026-11-23:* the ticks are locked to the HAT's data stream (`tick_lock`), so D2 can lock the robot loop to it and choose the offset for latency. The simulation draws `pwm_tick_phase` per scenario in robust scoring (`legobot_config`), so tuned gains don't rely on one phase. **Model the HAT's slow-PWM mode** (chosen in C3, DATASHEET 8.7): τ = \|d\| a (sgn(d) V − kω), with damping proportional to duty, so the error from feedforward constants scales with the command instead of adding drag. Put it in the three engines as an actuator model; the robot's feedforward inverts it as d = τ / (a (V ∓ kω̂)). Interface agreed with the phase D session: the new actuator class's `drive(command, measured_wheel_speed)` returns the duty (the inverse above, clipped to ±1), and `torque()` applies the model, so the robot and the simulator share one formula. `robot/control.py`'s `Feedforward` builds its own actuator from `cfg.actuator.build_motor_params()` with V = the measured vin, and will construct the new class | M | The reference engine reproduces the C3 slow-PWM spin-ups and the lever readings (`bench_data/`); equivalence and torch tests pass |
| B5 | Optional, only if B1–B4 tuning shows sensitivity: gearbox backlash as a dead zone, and different sensor delays for IMU and encoder | M | — |
|  | *Measured 2026-11-23 (DATASHEET 8.7):* gearbox ±2.6–3° at the encoder, plus up to ±3° of LEGO coupling play the encoder can't see. That is above the ±1.5° that raised the baselines' cost 19–129% in the B5 test (PHYSICS.md 10.3), so **port the backlash model**, with the play between motor and wheel drawn in about ±3–6° total | | |
| B6 | Low priority, before the final validation runs. **Pitch offset drawn per robot.** `pitch_bias` is one positive constant in every run (0.015 rad in `legobot_config.py`), so the tuner sees one offset size and one sign. Draw it per scenario, uniform in ±b (b = 0.015 rad, the residual after balance-point calibration, DATASHEET 8.4). This also covers a fore-aft COM offset e: that is exactly a pitch offset of atan(e/l), about 0.017 rad per mm at the measured l = 60 mm, with a negligible second-order change in COM distance. The batched engines need the bias as a per-scenario array instead of a scalar in the spec | S (reference) + S (port) | Equivalence and torch tests pass with a nonzero per-scenario bias; PHYSICS.md 10's "pitch offset per robot" gap is closed |
| B7 | Update PHYSICS.md (9.3, 9.5, 10) and ARCHITECTURE.md for what was built | S | — |

### Phase C: Bench tools and measurements (needs the assembled robot; parallel with B)

All of this runs on the Pi. The first two are also the robot code's drivers (phase D).

| # | Task | Size | Done when |
|---|---|---|---|
| C1 | **Build HAT driver** over raw serial with pyserial (`/dev/serial0`, 115200). Firmware load by the `buildhat` library in a short separate process, which exits and leaves the firmware running (DATASHEET 3.5); then `version` check, `port_plimit` and `pwmparams` per port (current firmware; `bias` is gone), a combi mode with speed, position and absolute position for both ports, a background reader that timestamps each data line, abbreviated motor commands (`po 0;pwm;set .421;po 1;pwm;set -.421`; time the exact string) | M | Both motors commanded and read at 100 Hz; timestamps logged |
| C2 | **IMU driver** for the ICM-20948 over I²C at 400 kHz (raise the Pi's default 100 kHz): gyro and accelerometer at a fixed rate, DLPF set, axis mapping applied | S–M | 200 Hz reads with timing logged |
| C3 | **Measurement scripts for DATASHEET section 8**: motor rail vs `vin`; stall current (R); no-load speed and current vs duty (k_e, friction); string-and-mass torque (efficiency, stall); spin-up fit (output inertia, with and without wheel); backlash; gyro noise and bias with motors off and on; accelerometer offset; balance point; battery discharge log; loop timing (latency). Each writes a CSV and prints the fitted constant with its uncertainty | M–L | Every Est./Pyb value in DATASHEET section 2 replaced by a measurement or confirmed |
| C4 | Fill in `legobot_config.py` from C3 and shrink the robustness spreads to the measurement uncertainty; update DATASHEET | S | — |

**C3/C4 status, 2026-11-23.** The scripts are in `robot/measure/` (DATASHEET 8.7).
- **Done:** the hands-free runs (duty map, spin-up and coast-down in three PWM modes, latency, IMU noise)
  and the lever-on-scale torque test in two modes.
- **Chosen:** slow PWM (`pwmparams 1 0.01`), which behaves as τ = |d| a (sgn(d) V − kω). The library
  mode's gain doubles at 0.65 duty, and fast PWM has a 0.4 dead band.
- **C4, done:** every motor constant (k, a, rotor inertia, both friction terms, max_torque) and the sensor
  noise in `legobot_config.py` are measured, and their spreads are shrunk (DATASHEET 2).
- **Still open:**
  - the balance point (`robot.calibrate balance`) and `discharge` (backlash measured: port B5);
  - B1b is done; still open is the slow-PWM pulse ripple at low duty (re-run `pwm_probe` with the robot secured);
  - **re-deriving the scenario (A3) for the measured robot.** `legobot_scenario.py` now gives a
    0.51 m/s cruise (was 0.27), a 0.35 rad catchable start tilt (was 0.18) and smaller forward pushes
    (0.014 N s at the higher speed). It also flags two inconsistencies to resolve first: the goal profile's
    0.292 rad lean exceeds the 0.262 rad lean cap, and the worst run without a fall costs 1532, above the
    1000 fall penalty;
  - new hand-tuned gains (A4_BASELINES.md, "Measured motor").

### Phase D: Robot code (needs C1, C2; uses B's filter)

| # | Task | Size | Done when |
|---|---|---|---|
| D1 | **Control loop** at 200 Hz: read IMU → filter (the B2 code) → read the latest motor data (C1) → controller → torque split τ/2 per wheel → feedforward to duty (B1's formula, the same constants as the sim) → command. Imports the controller classes from `simulation/balance_sim` so anti-windup, lean cap and units are identical; gains loaded from a results JSON | M | Runs on the bench with the wheels off the ground; loop timing logged |
| D2 | **Real-time behaviour** on Linux: a fixed-period loop on `time.perf_counter`, elevated scheduling priority, Bluetooth off so the serial link has the full UART; log every overrun | S | Jitter and overruns measured; latency figure fed back into B4 |
| D3 | **Safety and startup:** arm only when held upright and still; cut to coast past a tilt limit; stall detection (high duty, no motion for a set time → coast); duty cap; LiPo cutoff from `vin` (about 6.6 V under load); coast on "Motor power fault"; coast if the loop or serial link stalls; a physical off switch | M | Each trip tested deliberately on the bench |
| D4 | **Calibration at startup:** gyro bias (robot still for 2 s), stored accelerometer offset and balance-point pitch zero, motor and encoder sign check (one motor is mirrored) | S | — |
| D5 | **Logging** at 200 Hz in the format `balance_sim.save_trajectory` writes, so `animate.py --csv` replays real runs, plus raw sensor channels and loop timing | S | A real run replays in animate.py |

**Phase D as built (2026-11-23).** Code in `robot/`: `control.py` (the hardware-free core), `balance.py` (the
loop, real-time setup, watchdog, logs), `calibrate.py` (D4), `loop_config.py` (limits), `sim_hw.py` (HAT, IMU
and clock stand-ins backed by the simulator), `test_control.py`. `deploy.sh` copies `robot/`, `config.py`,
`legobot_config.py` and `simulation/balance_sim/` to the Pi's `~/legobot`, with a `DEPLOYED` stamp that every
log records.

- **Shared with the simulator, not copied:** the robot imports `config.load_config("legobot")`. The estimator is
  `ComplementaryFilter` with the config's τ, run at the 200 Hz control period. Wheel speed is the count
  difference over one 10 ms HAT period, as `Encoder` computes it, and v = r(ω̄ + gyro). The controller is built by
  `ProjectConfig.build_controller` or `build_lqr_controller`, so the torque cap, lean cap and integral limits
  match the sim. Duty = `VoltageModeActuator.drive(τ, ω_wheel)` / vin, per wheel, using the HAT's measured vin.
- **Controllers:** `--controller hand | model | lqr | <gains JSON>` (tuning results' `gains`, or `K` for LQR).
  An optional `feedforward_gain` in the JSON is applied too. The LQR design needs scipy, which the Pi doesn't
  have, so export it first: `--controller lqr --export-gains lqr.json` on the laptop. `--goal still | scenario`.
- **Trips (each one coasts both motors and ends the run):**
  - tilt past 0.70 rad;
  - a wheel stalled (|duty| ≥ 0.5, or 0.8 × the cap on the bench, at under 1 rad/s for 1 s);
  - a wheel turning fast against its duty (a sign error);
  - 3 vin replies below 6.6 V, or no vin reply for 2 s;
  - "Motor power fault";
  - motor data older than 50 ms;
  - the serial reader thread dying;
  - a tick more than 25 ms late;
  - the loop going silent for 100 ms: a watchdog thread coasts the motors with a raw write that bypasses the
    driver's lock.

  Arming requires |pitch| < 0.05 rad and rate < 0.3 rad/s for 0.5 s, within 30 s. `--inject loop|hang|stale|vin|fault`
  triggers one trip on purpose. Tilt and stall are tested by hand.
- **Startup:** the gyro bias comes from 2 s at rest, and startup refuses to continue if the robot moved. The
  pitch zero (balance point, which absorbs the accelerometer offset and the mounting angle) and the sign-checked
  `PORT_SIGN` are read from `calibration.json`, written by `robot.calibrate balance|signs`.
- **Real time:** sleep to 0.4 ms before each deadline, then spin. If a period is missed, the loop re-phases
  instead of catching up. GC is frozen and disabled during the run, and memory is locked. The loop runs at
  SCHED_FIFO 50, pinned to core 3; the watchdog runs at 51 and the reader at 49. The Pi's
  `/etc/security/limits.d/legobot-rt.conf` (added 2026-11-23) gives `user` rtprio 60 and unlimited memlock, so no
  root is needed. Without that file, the loop falls back to normal priority and records it in the summary.
- **Checked against the simulator:** 16 tests pass. Closed loop against the simulator (`--sim --sim-noise
  --goal scenario`), the hand-tuned, model-based and LQR controllers all finish the 20 s scenario, cruising within
  0.07 m/s of 0.27 m/s. Plain voltage mode (`--ff-gain 0`) falls and trips on tilt, as the sim predicts (A4). On
  the Pi 3B, sense + safety + controller + feedforward take 0.06 ms per tick (p99 0.08 ms).
- **Bench, 2026-11-23** (robot upside down on the stand, wheels off the ground, `--bench` duty cap 0.3,
  `--pitch-zero` set to the stand's tilt; logs `~/legobot/logs/20261123_06{29,30,3}*`):
  - **D1:** the loop arms and drives both motors. Positive duty turns both wheels forward, and both encoders read
    forward. With a 0.03 rad pitch offset the wheels hold the 0.3 cap. On a fixed stand the velocity loop has no
    pendulum to balance against, so it runs away to the cap. At θ ≈ 0 the duty stays inside the HAT's dead band
    and the wheels don't move. Duty 0.3 gives about 8 rad/s, above the 5.4 rad/s that duty × vin / k predicts:
    the HAT's duty-to-voltage map is not linear (C3's finding; see B1).
  - **D2 (10 s runs at SCHED_FIFO):** period sd 0.002–0.022 ms, worst lateness 0.02–0.70 ms, 0 overruns. At
    normal priority before (DATASHEET 6) it was sd 0.05–0.13 ms and 2–4 overruns, worst 8.3 ms. From the IMU read
    to the command written takes 1.3 ms mean, 1.8 ms p99, worst 3.6 ms. The newest motor reading is 0–9 ms old at
    each tick. Command to first encoder motion is 15–17.5 ms (C3, duty 1.0); that includes the motor's mechanical
    response.
  - **D3 (injected, wheels spinning at 0.3):** each trip fired with the right cause: loop stall (one 50 ms tick),
    watchdog (loop blocked 0.5 s; it fired after 103 ms, and both wheels read 0 at the first tick after the hang,
    where without it the HAT keeps driving), stale motor data, battery low, and motor power fault. After every one,
    both wheels were at 0 deg/s 1 s after the stop.
  - **D5:** a bench `_run.csv` replays in `simulation/animate.py --robot legobot --csv`.
- **Slow PWM (B1b), 2026-11-23:**
  - **Feedforward:** the robot sends `SlowPwmActuator.drive()`'s duty directly, d = (τ/2)/(a(V ∓ kω)) per wheel,
    with V = vin. `"voltage"` still works (volts / vin).
  - **Duty check (no-drive run):** θ = −0.031 rad gave τ = −0.183 N m, and the loop sent d = −0.201 =
    −0.0915/(0.0597 × 7.63), as the formula gives.
  - **On the stand (cap 0.3, pwmparams 1 0.01):** both wheels turned with the duty's sign and settled at
    14–17.5 rad/s, against about 18.5 from the measured a, k and friction. The library mode gave about 8 rad/s
    at the same duty. Timing was unchanged: period sd 0.003 ms, 0 overruns.
  - **Fixes:**
    - SIGINT is restored at startup: run_safe.sh's background start leaves it ignored until `install_safety`
      runs (C3's finding). A SIGINT at 1.5 s or 2.5 s into startup now exits within 0.34 s.
    - The gyro-bias rejection limit was raised to 0.10 rad/s. This chip reads 0.044–0.047 at rest (spec
      ±0.087); motion is still caught by the sd check.
  - **Calibration:** the pitch zero in `~/legobot/calibration.json` (−0.010 ± 0.010 rad) came from C3's
    release test, not `calibrate balance`.
- **Locked to the HAT's line grid (D2), 2026-11-23.** In slow PWM the HAT's pulse ticks are locked to its
  100 Hz data lines (C3, `measure/tick_lock.py`), so the loop now ticks on the HAT's clock.
  - **How:** `control.LineClock` estimates the line grid from the left port's arrival times (period from a
    least-squares fit over 256 lines, smoothed across updates; phase from the lower 10% envelope, since arrivals
    only ever come late). The driver keeps the last 512 arrivals. `control.Pacer` puts two ticks per line period
    at a fixed offset. The grid is re-estimated every 20 ticks, after that tick's command is out. Without a
    lock the loop falls back to the Pi's clock; `--no-line-lock` forces that.
  - **Lock on the bench:** every tick locked. The data age at each tick of the pair stayed fixed (3.3 and
    8.1 ms, sd 1.2) over 10 s, with no drift. The line period is 9.9997–9.9999 ms by the Pi's clock
    (−14 to −21 ppm).
  - **Choosing the phase (`robot.lock_sweep`):** one wheel stepped to duty 1.0, with the first write at 0–9 ms
    after the line, 8 trials per offset for the left wheel and 6 for the right, then a sawtooth fit to the
    delay to motion. The left motor's tick (the cliff) is at 8.3 ms (90% 7.2–8.3), with a base delay of
    13.3 ms. The right motor's is about 2 ms later, at 10.3 ms (90% 10.2–11.3), with a base of 12.7 ms.
    `LINE_WRITE_PHASE_S` = 6.5 ms, before the earlier cliff with margin. Each command then waits about 1.8 ms
    (left) and 3.8 ms (right) for its tick, against 0–10 ms each, 5 ms on average, at a random phase.
  - **Measured in the loop:** writes start at 6.33 ms (sd 0.06 ms); 0 overruns.
  - **Tests:** 21 pass. They cover the tracker on synthetic arrivals with jitter, dropped lines, late lines and
    two-line reads (period within 60 ppm, phase within 0.3 ms), the pacer, and a locked closed loop against the
    simulator. In the simulator the HAT's lines run on their own 9.9912 ms clock with arrival jitter.
- **Still to do on hardware (needs hands):** the tilt trip (tilt past 40°), the stall trip (hold a wheel with
  `--bench` and a pitch offset), and `robot.calibrate signs --wheels-off --save` (asks whether each wheel turned
  forward).

### Phase E: Retune and compare

| # | Task | Size | Done when |
|---|---|---|---|
| E1 | Development tuning on the LEGO config with B1–B4, on dev seeds, to find which models matter | M | — |
| E2 | Final tuning on the measured config (C4), full rigor, then the frozen validation set once | M (mostly compute) | Validated LEGO gains |
| E3 | **Sim-to-real comparison:** replay a logged run's measured start state and commands through the simulator in short windows (about 0.5 s, re-initialized from the log each window, since an unstable system diverges in open loop), and compare closed-loop runs statistically (pitch and velocity spectra, push recovery) | M | A sim-vs-real figure for the paper |
| E4 | First balancing tests with progressively harder tasks: stand still, small pushes, then the scenario | — | — |

## 4. Order and dependencies

```
A1 → A2 → A3 → A4 ─┐
                   ├→ B1 → B2 → B3 → B4 ─┐    (B6 any time before E2's final tuning)
C1, C2 → C3 → C4 ──┼─────────────────────┼→ E1 → E2 → E3 → E4
                   └→ D1 → D2 → D3 → D4 → D5 ─┘
```

- **Start with A1–A3**: nothing else is meaningful until the sim describes this robot and asks it for things it
  can do.
- **B1 is the highest-risk item**, as it was for the Pololu build, but here it is certain to apply: the Build HAT
  has only voltage mode.
- **C3 early pays off**: rotor inertia (±50%) and resistance (3.73 vs 5.14 Ω) are the widest spreads, and both
  change the tuned gains. Measuring them before E2 avoids tuning for a robot that doesn't exist.
- **D can start as soon as C1 and C2 work**; D1 reuses B2's filter, so B2 should land first or D1 starts with a
  simple filter and swaps it in.

## 5. Decisions recommended, not yet made

| Decision | Recommendation | Reason |
|---|---|---|
| Feedforward or plain voltage mode on the robot | Feedforward, simulated with its errors | Plain voltage mode was the case that failed for the Pololu robot, with 3.5× less drag |
| How to select the robot | `--robot` flag plus a `ROBOT` environment variable in config.py | Leaves every existing command working for the Pololu robot |
| Build HAT library or raw serial | **Decided 2026-11-22:** library for firmware load only, then raw serial with pyserial | Latency is visible and controllable; the protocol is short (DATASHEET 3.3) |
| Filter | Complementary filter first; Kalman only if the error budget needs it | Simple to run identically in sim and on the robot |
| Fold L/R into latency or model it | Decide in B4 by experiment | 3 ms against a 10 ms nominal latency may not matter |
