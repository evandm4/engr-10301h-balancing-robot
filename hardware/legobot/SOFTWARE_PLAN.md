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
| B1 | **Voltage-mode actuator with feedforward.** In `actuators.py`: duty = clip((R̂ i_cmd + k̂_e ω̂)/(plimit V), ±1), applied voltage = duty × plimit × V, current from the true R and k_e. ω̂ is the motor speed the robot code sees (B3), R̂ and k̂_e are the robot code's constants. Options: feedforward off (plain voltage mode), and a `bias` dead-band jump. Robustness draws the mismatch R̂/R and k̂_e/k_e per robot (±17%, ±15%, the datasheet spreads). Replaces the current-mode `DCMotorActuator` for the LEGO robot; the Pololu robot keeps its model | M (reference) + M (port) | Reproduces PHYSICS.md 9.3's numbers with feedforward off on the Pololu robot; equivalence and torch tests pass |
| B2 | **Simulated IMU and filter.** Sensor model outputs gyro rate (noise, bias, DLPF modeled as a first-order lag at the chosen bandwidth) and accelerometer tilt from ẍ, θ̈ and the mount height d (plus noise and offset). A complementary filter, the same code the robot will run (section 4), turns them into the pitch estimate the controller sees. ẍ and θ̈ come from the dynamics at the sample time | M (reference) + M–L (port: the filter adds state to every simulation) | With noise off and the robot still, the estimate converges to the truth; during the scenario's accelerations the error has the size DATASHEET 3.4 predicts |
| B3 | **Encoder model.** Wheel angle quantized to 1°, updated every 10 ms and held between updates, read with a configurable delay; velocity from counts 10 ms apart. Gearbox slack as a ±3° position error, then (B5) as a real dead zone if needed | S–M | Velocity noise matches DATASHEET 5.4 (0.025 m/s at 69 mm) |
| B4 | **Timing.** Latency range 5–25 ms; the motor's L/R (3 ms) either folded into latency or as a first-order current lag in the actuator. Decide by experiment in the reference engine | S | Decision recorded in PHYSICS.md |
| B5 | Optional, only if B1–B4 tuning shows sensitivity: gearbox backlash as a dead zone, and different sensor delays for IMU and encoder | M | — |
| B6 | Update PHYSICS.md (9.3, 9.5, 10) and ARCHITECTURE.md for what was built | S | — |

### Phase C: Bench tools and measurements (needs the assembled robot; parallel with B)

All of this runs on the Pi. The first two are also the robot code's drivers (phase D).

| # | Task | Size | Done when |
|---|---|---|---|
| C1 | **Build HAT driver** over raw serial (`/dev/serial0`, 115200): firmware load at boot, `plimit 1`, `bias 0`, a combi mode with speed, position and absolute position for both ports, a background reader that timestamps each data line, abbreviated motor commands (`po 0;set .421;po 1;set -.421`, 2.5 ms). The `buildhat` Python library is simpler but adds overhead we can't see; measure both | M | Both motors commanded and read at 100 Hz; timestamps logged |
| C2 | **IMU driver** for the ICM-20948 over I²C at 400 kHz (raise the Pi's default 100 kHz): gyro and accelerometer at a fixed rate, DLPF set, axis mapping applied | S–M | 200 Hz reads with timing logged |
| C3 | **Measurement scripts for DATASHEET section 8**: motor rail vs `vin`; stall current (R); no-load speed and current vs duty (k_e, friction); string-and-mass torque (efficiency, stall); spin-up fit (output inertia, with and without wheel); backlash; gyro noise and bias with motors off and on; accelerometer offset; balance point; battery discharge log; loop timing (latency). Each writes a CSV and prints the fitted constant with its uncertainty | M–L | Every Est./Pyb value in DATASHEET section 2 replaced by a measurement or confirmed |
| C4 | Fill in `legobot_config.py` from C3 and shrink the robustness spreads to the measurement uncertainty; update DATASHEET | S | — |

### Phase D: Robot code (needs C1, C2; uses B's filter)

| # | Task | Size | Done when |
|---|---|---|---|
| D1 | **Control loop** at 200 Hz: read IMU → filter (the B2 code) → read the latest motor data (C1) → controller → torque split τ/2 per wheel → feedforward to duty (B1's formula, the same constants as the sim) → command. Imports the controller classes from `simulation/balance_sim` so anti-windup, lean cap and units are identical; gains loaded from a results JSON | M | Runs on the bench with the wheels off the ground; loop timing logged |
| D2 | **Real-time behaviour** on Linux: a fixed-period loop on `time.perf_counter`, elevated scheduling priority, Bluetooth off so the serial link has the full UART; log every overrun | S | Jitter and overruns measured; latency figure fed back into B4 |
| D3 | **Safety and startup:** arm only when held upright and still; cut to coast past a tilt limit; stall detection (high duty, no motion for a set time → coast); duty cap; LiPo cutoff from `vin` (about 6.6 V under load); coast on "Motor power fault"; coast if the loop or serial link stalls; a physical off switch | M | Each trip tested deliberately on the bench |
| D4 | **Calibration at startup:** gyro bias (robot still for 2 s), stored accelerometer offset and balance-point pitch zero, motor and encoder sign check (one motor is mirrored) | S | — |
| D5 | **Logging** at 200 Hz in the format `balance_sim.save_trajectory` writes, so `animate.py --csv` replays real runs, plus raw sensor channels and loop timing | S | A real run replays in animate.py |

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
                   ├→ B1 → B2 → B3 → B4 ─┐
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
| Build HAT library or raw serial | Raw serial, after measuring both | Latency is visible and controllable; the protocol is short (DATASHEET 3.3) |
| Filter | Complementary filter first; Kalman only if the error budget needs it | Simple to run identically in sim and on the robot |
| Fold L/R into latency or model it | Decide in B4 by experiment | 3 ms against a 10 ms nominal latency may not matter |
