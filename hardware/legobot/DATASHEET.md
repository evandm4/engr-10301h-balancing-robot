# Datasheet: LEGO SPIKE Prime build ("legobot")

The physical robot planned as the project's reference build: two LEGO SPIKE Prime angular motors driving the
wheels directly, a Raspberry Pi 3B with the Raspberry Pi **Build HAT** (motor drive and motor sensors) and the
**SparkFun Auto pHAT** (ICM-20948 IMU), on a LEGO chassis. This file gives, for every physical constant the
simulation needs, the value for this build, where it comes from, how uncertain it is, and how to measure it. It also
lists where the robot differs from what the current model can represent. Compiled 2026-11-22.

The Pololu reference build in [DATASHEET.md](../../DATASHEET.md) stays as it is; `config.py` is unchanged. Section
2 is the proposed value set for a separate LEGO configuration.

Tags, as in the main datasheet:

| Tag | Meaning |
|---|---|
| **DS** | Manufacturer datasheet, fact sheet or official documentation (Sources, section 9) |
| **Pyb** | Measured by the Pybricks authors and published in their firmware source (section 3.1) |
| **Derived** | Calculated here from DS/Pyb numbers; [legobot_numbers.py](legobot_numbers.py) prints every one |
| **Est.** | Engineering estimate, no source. Measure it (section 8) |

## Contents

1. [Build and recommendations](#1-build-and-recommendations)
2. [Proposed simulation values](#2-proposed-simulation-values)
3. [Component data](#3-component-data)
4. [Mass properties](#4-mass-properties)
5. [Derivations](#5-derivations)
6. [Latency budget](#6-latency-budget)
7. [Where the current model doesn't match this robot](#7-where-the-current-model-doesnt-match-this-robot)
8. [What to measure, and how](#8-what-to-measure-and-how)
9. [Sources](#9-sources)

---

## 1. Build and recommendations

All parts are owned.

| Function | Part | Notes |
|---|---|---|
| Wheel motors (×2) | **LEGO Technic Large Angular Motor 45602**, direct drive | Medium Angular Motor 45603 is the fallback; section 3.1 compares them |
| Wheels (×2) | **69 mm LEGO wheels** (planned); Ø88 × 14.34 mm (49295) is a modest upgrade, Ø56 mm (39367) a downgrade | Sections 3.2, 5.6 |
| Computer | Raspberry Pi 3B | Runs the controller (Python) |
| Motor interface | Raspberry Pi Build HAT | RP2040 firmware drives the four LPF2 ports; plain-text serial link to the Pi |
| IMU | SparkFun Auto pHAT (ROB-16328), on-board TDK ICM-20948 | Only the IMU is used; its motor driver stays idle |
| Battery | **2S LiPo** (chosen, for mass) | Section 3.6 |

**Recommendations, with the reason for each:**

1. **Large motors, not Medium.** They have about 3× the torque at the same no-load speed (0.85 vs 0.28 N m stall
   for the pair at 7.2 V, Pybricks data). The Medium motor's speed falls to 4.6 rad/s under 0.1 N m (Pyb), so it
   has no margin for recovering from a push.
2. **The planned 69 mm wheels are adequate; bigger is somewhat better, not necessary.** The motors are slow (about
   16.6 rad/s with no load at 7.2 V), and their output-side inertia is large, so a bigger wheel helps in two ways:
   more top speed, and less of that inertia felt by the body. Against 69 mm, an 88 mm wheel gives 27% more top speed
   (0.73 against 0.57 m/s at 7.2 V) and about 12% more acceleration per unit torque (section 5.6). Worth using if you
   already have 88 mm wheels; not worth buying before the spin-up test (8.2), because the best size depends on the
   motors' inertia, which is only an estimate so far.
3. **2S LiPo (decided).** Lightest option, and the best electrical fit: the Build HAT wants 8 V ±10% (7.2–8.8 V),
   the LEGO motors are characterized at 7.2 V, and a LiPo holds 7.4–8.4 V for most of its discharge with little sag.
   Fresh alkaline AAs exceed 8.8 V and sag badly under load.
4. **In the controller, set `plimit 1` and `bias 0` on the Build HAT** (or simulate their effect; section 7.2). The
   HAT's defaults scale and offset every motor command.

## 2. Proposed simulation values

How to represent this robot with the existing `config.py` classes. "Spread" is the `RobustnessSettings.randomize`
relative spread (or `randomize_range`) to use **until the quantity is measured**; section 8 says what each
measurement brings it down to. Values marked Est. are placeholders to replace with your measurements.

### 2.1 `RobotConfig`

| Field | Value | Spread until measured | Tag | Notes |
|---|---|---|---|---|
| `body_mass` | 0.55 kg | ±0.15 | Est. | Everything except the wheels: motors (their housings ride with the body), Pi, both HATs, battery, LEGO frame. Section 4. A gram scale makes the spread ±0.01 |
| `body_com_height` | 0.09 m | ±0.15 | Est. (design choice) | Set by where the battery sits. Measured to ±5 mm, the spread becomes about ±0.06 |
| `body_inertia` | 0.0035 kg m² | ±0.30 | Est. | Pitch inertia about the body's own COM. Section 4. A pendulum test brings it to about ±0.05 |
| `wheel_mass` | 0.078 kg | — | Measured | Pair: 39 g per wheel including its axle |
| `wheel_radius` | 0.0345 m | — | From diameter | Half the 69 mm diameter (0.044 m for the 88 mm wheel). Approximation for now; the loaded rolling radius (roll ten turns, divide the distance by 20π) can replace it later |
| `wheel_inertia` | 6.0e-5 kg m² | ±0.20 | Measured mass / Est. | Pair, 0.65 m r² each (tire at the rim), as in the main datasheet. Slightly high, since the axle's few grams sit at the centre. The motor's output-side inertia is separate (`rotor_inertia` below) |
| `max_torque` | 0.50 N m | — | DS | 2 × the 0.25 N m stall torque on LEGO's fact sheet. Pybricks' data implies 0.85 N m (section 5.1), so 0.50 is the conservative published figure |
| `wheel_friction` | 0 | range 0–0.001 N m s/rad | Est. | No source separates viscous from Coulomb drag for these motors |
| `coulomb_friction` | 0.046 N m | range 0.03–0.08 N m | Pyb/Derived | Pybricks' friction torque, 0.023 N m per motor. LEGO's no-load current implies 0.024–0.031 per motor (5.2) |
| `gravity` | 9.81 m/s² | — | — | Unchanged |

### 2.2 `ActuatorConfig` (model `"dc_motor"`)

The internal gear ratio of the SPIKE motors is not published, and nothing here needs it. Every published and
measured number is at the **output shaft**, so the motor is described as a lumped output-side motor with
`gear_ratio = 1`. `DCMotorActuator` then gives exactly the measured behaviour: torque = n η k i, back-EMF = k ω.

| Field | Value | Spread until measured | Tag | Notes |
|---|---|---|---|---|
| `gear_ratio` | 1.0 | — | — | Lumped: constants below are output-referred |
| `torque_constant` | 0.411 V s/rad | ±0.15 | Pyb | Output back-EMF constant k_e. LEGO's sheet gives 0.355 (−14%) |
| `gear_efficiency` | 0.565 | ±0.10 | Pyb/Derived | Net torque per amp over k_e: 0.232 / 0.411. LEGO's sheet gives 0.50 |
| `resistance` | 4.4 Ω | ±0.17 | Pyb/DS | Midpoint of Pybricks' 3.73 Ω and the 5.14 Ω implied by LEGO's stall current, so the spread covers both. Includes driver and wiring. One stall-current measurement settles it (8.2) |
| `supply_voltage` | 7.6 V × `plimit` | ±0.105 | DS | 2S LiPo, 6.8–8.4 V, **times the HAT's power limit** (7.2). With `plimit 1`: 7.6 V |
| `current_limit` | 2.5 A | — | — | The Build HAT has no current control, so this is set high enough never to bind (stall at 8.4 V and 3.73 Ω is 2.25 A) |
| `n_motors` | 2 | — | — | |
| `rotor_inertia` | 1.22e-3 kg m² | ±0.50 | Pyb (estimate) | Per motor, **output side** (with `gear_ratio = 1`, reflected = 2 × 1.22e-3 = 2.44e-3). Pybricks derived it from an *estimated* acceleration; a spin-up test measures it (8.2). It is by far the largest inertia on the wheel side |

### 2.3 `SensorConfig` (control rate 200 Hz; see 7.3 for the encoder's 100 Hz)

| Field | Value | Tag | Notes |
|---|---|---|---|
| `pitch_rate_std` | 0.02 rad/s (keep) | Derived/Est. | ICM-20948 at a 51 Hz filter: 2.2e-3 rad/s (5.4). Keep the conservative 0.02 until the chassis measurement (8.4); LEGO frames rattle |
| `pitch_rate_bias` | 0.01 rad/s (keep) | DS/Est. | Uncalibrated offset is up to ±5 °/s (0.087 rad/s): **calibrate at every boot**. What remains is thermal drift (±0.05 °/s/°C) |
| `pitch_std` | 0.005 rad (keep) | Derived | Datasheet-level is 1.2–1.6e-4 rad after a complementary filter (5.4) |
| `pitch_bias` | 0.015 rad | Est. | Raised from 0.01: the accelerometer's board-level offset is up to ±50 mg (0.05 rad) uncalibrated, and a LEGO mount adds play. Calibrate the balance point (8.4) |
| `position_std` | 0.0010 m | DS/Derived | ±3° accuracy including gearbox slack, at r = 34.5 mm (5.4). One count is 0.60 mm |
| `velocity_std` | 0.025 m/s | Derived | Counts differenced 10 ms apart (the sensor's 100 Hz update) |

### 2.4 `TimingConfig` and robustness latency

| Field | Value | Tag | Notes |
|---|---|---|---|
| `latency_s` | 0.010 s (keep) | Derived/Est. | Budget is about 6–15 ms (section 6), slower than the Pololu build's 1.5–6 ms |
| `latency_range_s` | (0.005, 0.025) | Derived/Est. | Shifted up to cover the budget with margin; 2.5 ms is not reachable on this hardware |
| `control_dt_s` | 0.005 s | — | Possible for the IMU path; motor data only refreshes every 10 ms (7.3) |

## 3. Component data

### 3.1 Motors

**LEGO fact sheets** (DS; all data at a 7.2 V supply, ±15% on speed and current):

| Quantity | Large Angular 45602 | Medium Angular 45603 |
|---|---|---|
| Voltage range | 5–9 V | 5–9 V |
| No load | 175 rpm (18.3 rad/s), 135 mA | 185 rpm (19.4 rad/s), 110 mA |
| Maximum efficiency | 8 N cm at 135 rpm, 430 mA | 3.5 N cm at 135 rpm, 280 mA |
| Stall | 25 N cm, 1400 mA | 18 N cm, 800 mA |
| Rotation sensor | 360 counts/rev on the output; accuracy ≤ ±3° (sensor tolerance plus gearbox slack); update rate 100 Hz | same |
| Sensor outputs | speed (% of maximum design speed), relative position (°), absolute position (±180°) | same |
| Connector | LPF2, 250 mm lead | same |
| Mass, internal gear ratio, inertia | not published | not published |

**Pybricks measurements** (Pyb). The Pybricks firmware ships a model of each motor, generated by
`pbio/doc/control/motor_data.py` from "experiments by Pybricks authors": at 7.2 V, a no-load point and a point under
a 0.0999 N m load (Large: 16.6 rad/s at 0.10 A, then 12.7 rad/s at 0.53 A; Medium: 16.6 rad/s at 0.09 A, then
4.6 rad/s at 0.51 A), an estimated no-load acceleration and an estimated inductance. The same formulas give, per
motor at the output shaft:

| Constant | Large | Medium |
|---|---|---|
| Resistance R | 3.73 Ω | 10.73 Ω |
| Back-EMF constant k_e | 0.411 V s/rad | 0.376 V s/rad |
| Net torque per amp k_t | 0.232 N m/A | 0.238 N m/A |
| Effective efficiency k_t / k_e | 0.565 | 0.634 |
| Coulomb friction | 0.023 N m | 0.021 N m |
| Output-side inertia (rotor through the gears) | 1.22e-3 kg m² (estimate) | 3.0e-4 kg m² (estimate) |
| Inductance, L/R | 12 mH, 3.2 ms (estimate) | 24 mH, 2.2 ms (estimate) |
| Stall at 7.2 V | 1.93 A, 0.43 N m | 0.67 A, 0.14 N m |
| Firmware rated max speed | 1000 °/s (17.5 rad/s) | 1000 °/s |

Pybricks treats the SPIKE Large and Medium as identical to the Technic Large and Medium angular motors.

**The two sources disagree on the Large motor's stall** (0.25 N m at 1.4 A from LEGO; 0.43 N m at 1.9 A from
extrapolating Pybricks' points). LEGO's stall figure may be a protected or rated value rather than a measured
extrapolation. Their no-load speeds agree within LEGO's ±15%. The proposed values (section 2) use Pybricks' k_e and
efficiency, a resistance between the two, and LEGO's stall torque as the torque cap. Section 8.2 resolves it.

### 3.2 Wheels

| Part | Diameter × width | Mass | Tag |
|---|---|---|---|
| **Planned wheel** | 69 mm | 39 g each, with its axle | Measured |
| 49295 | 88 × 14.34 mm | not published; about 25–35 g each (Est.) | DS (size) |
| 39367 | 56 × 14 mm | not published; about 12–18 g each (Est.) | DS (size) |

All mount on a Technic axle in the motor's crosshole output. A wheel on a short axle is a cantilever; support the
outer end of the axle if the chassis allows, so the wheel doesn't toe in under load.

### 3.3 Raspberry Pi Build HAT

| Quantity | Value | Tag |
|---|---|---|
| Input | +8 V ±10% (7.2–8.8 V), 48 W, DC 5521 centre-positive barrel (5.5 × 2.1 mm) | DS |
| Official supply | 8 V, 6 A | DS |
| Powers the Pi | Yes, from the barrel input (no USB supply needed) | DS |
| Motor supply on each port | "about +7.2V", one dedicated driver IC per port; latched "Motor power fault" on overload. Whether it is regulated or follows the input is not stated | DS |
| Drive | PWM of either polarity. **Direct PWM mode (default): setpoint −1 to +1 directly sets the output power.** No current sensing or current control | DS |
| On-board PID | Optional, per port, on a motor's speed or position; not usable for torque control | DS |
| `plimit` | Global limit on motor drive power, all ports. **Default 0.1** at the serial level; must be raised. (.NET library default 0.7) | DS |
| `bias` | Added to positive and subtracted from negative drive values, to overcome static friction. .NET default 0.3; the protocol's own SPIKE example uses 0.4 | DS |
| Pi link | UART, 115200 baud 8N1, no flow control, plain text; commands can be abbreviated and chained with `;` | DS |
| Motor data | Streamed after `select`/`combi`; the SPIKE motors' modes include SPEED (%), POS (°), APOS (°); a combi of modes 1, 2, 3 gives all three in one line | DS |
| Pins used | GPIO0/1 (ID EEPROM), GPIO4 (reset), GPIO14/15 (Tx/Rx), GPIO16/17 (RTS/CTS, unused) | DS |
| Stacking | Components on the bottom, 9 mm spacers. For GPIO pass-through, an optional tall header and 15 mm spacers | DS |
| Pi support | Any 40-pin Pi (so the 3B), Raspberry Pi OS Bookworm (not yet Trixie) | DS |
| Driver IC part number, per-port current limit, PWM frequency, mass | Not published | — |

### 3.4 SparkFun Auto pHAT and its ICM-20948

| Quantity | Value | Tag |
|---|---|---|
| Interface | I²C (Qwiic), up to 400 kHz | DS |
| Gyro range / sensitivity | ±250, 500, 1000, 2000 °/s; 131 LSB/(°/s) at ±250 | DS |
| Gyro noise density | 0.015 °/s/√Hz | DS |
| Gyro zero-rate offset | ±5 °/s initial (25 °C); ±0.05 °/s/°C over temperature | DS |
| Gyro filter (DLPF 3 dB / noise bandwidth) | cfg 2: 119.5 / 154.3 Hz; cfg 3: 51.2 / 73.3 Hz; cfg 4: 23.9 / 35.9 Hz | DS |
| Gyro output rate | up to 1.125 kHz with the DLPF (9 kHz bypassed) | DS |
| Accelerometer noise density | 230 µg/√Hz | DS |
| Accelerometer zero-g offset | ±25 mg component, ±50 mg board-level; ±0.80 mg/°C | DS |
| Accelerometer filter | cfg 3: 50.4 / 34.4 Hz; cfg 4: 23.9 / 17.0 Hz | DS |
| Power | From the Pi's header; its USB-C input can also back-power the Pi's 5 V rail | DS |
| Stacking | "Designed for stacking, full header support" | DS |

Compared with the IMUs in the main datasheet, the ICM-20948's gyro is 3× noisier than an MPU-6050 and 5× noisier
than an ICM-42688-P, and still about 10× quieter than the sim's 0.02 rad/s.

**Stacking with the Build HAT.** The Build HAT's pins (GPIO 0, 1, 4, 14–17) and the Auto pHAT's I²C bus (GPIO 2/3)
do not overlap (the pHAT's full pin list was not readable, so confirm on its schematic). Either order works:
Pi → Auto pHAT (pass-through header) → Build HAT, or Pi → Build HAT (tall header, 15 mm spacers) → Auto pHAT.
**Do not connect the Auto pHAT's USB-C while the Build HAT powers the Pi**; both would drive the 5 V rail.

**Orientation.** Any orientation works if the firmware knows it, but the simple choice is to mount the board
square to the chassis, with one of the IMU's axes parallel to the wheel axle. Then pitch rate is that one gyro
axis, and tilt is atan2 of the two accelerometer axes perpendicular to the axle; no rotation matrix is needed.
LEGO makes squareness easy. Find which axis is which, and its sign, from the silkscreen or by rotating the robot
about the axle and watching which gyro channel responds. Small errors don't matter:

- A board yawed by δ (gyro axis not quite parallel to the axle) reads pitch rate × cos δ, 0.06% low at 2°, and
  picks up sin δ of any turning rate, which is zero when driving straight.
- A board pitched by δ on its mount is a constant pitch offset, removed by the balance-point calibration (8.4).
  What remains after calibration is `pitch_bias`.
- Don't use the magnetometer: the motors' magnets and currents sit a few centimetres away.

**Offset from the axle.** The gyro reads the same pitch rate anywhere on a rigid body, so its offset needs no
correction. The accelerometer, at height d above the axle, reads gravity plus its own acceleration: the axle's
acceleration ẍ, a tangential d θ̈ and a centripetal d θ̇² (along the body). Tilt from the accelerometer is off by
about (ẍ + d θ̈)/g; the centripetal term is along the measuring direction and only second order. Rough sizes for
d = 0.10 m with a complementary filter of τ = 0.5 s (Est.):

- The d θ̈ term averages out, because θ̇ stays bounded; a pitch-rate swing of 1 rad/s leaves a brief error of about
  (d/g)(Δθ̇/τ) ≈ 0.02 rad.
- The ẍ term, which doesn't depend on where the IMU is, is bigger: 1 m/s² held for 0.5 s gives about 0.06 rad.

Both can be subtracted in firmware (ẍ from the encoders, θ̈ by differentiating the gyro, d measured with a ruler).
Mounting the IMU lower shrinks the d terms; the HAT stack sets its height, so putting the Pi stack low and the
battery high is the way to get both a low IMU and a high COM. **The simulation models none of this yet** (7.9).
Mount the board rigidly: LEGO joints have play.

### 3.5 Raspberry Pi 3B

Runs Linux, so the control loop has scheduling jitter (section 6). On the Pi 3B the full UART is normally assigned to
Bluetooth, and `/dev/serial0` maps to the mini UART; the Build HAT works on either at 115200 baud.

### 3.6 Power options

| | 2S LiPo | 6×AA NiMH | 6×AA alkaline |
|---|---|---|---|
| Full | 8.4 V | about 8.4 V off the charger, settling to 7.2–7.8 V | 9.0–9.6 V (**above the HAT's 8.8 V**) |
| Most of the discharge | 7.4–7.8 V | about 7.2 V, lower under load | falls steadily from 9 V to 6 V |
| End | 6.4–6.8 V under load (cutoff) | about 6.0 V | about 6 V |
| Pack resistance (Est.) | tens of mΩ | 0.1–0.3 Ω | 0.9–1.8 Ω |
| Sag at a 3 A peak (both motors stalling plus the Pi) (Est.) | 0.1–0.2 V | 0.3–0.9 V | 3–5 V |
| Mass | 1300 mAh: 70–90 g; 2200 mAh: 130 g | about 160–180 g with holder | about 140–160 g with holder |

The Build HAT's range is 7.2–8.8 V. Every option spends some of its discharge below 7.2 V under load; how the HAT
and the Pi's 5 V supply behave there is not documented. Test it before relying on it (8.5). A heavy pack is also the
easiest way to set the COM height: it is the densest single part.

For the simulation, `supply_voltage` is what the motor sees at 100% duty. Section 7.2 covers the `plimit` scaling
and whether the motor rail follows the battery.

## 4. Mass properties

No masses are published for the motors, HATs or LEGO wheels, so this is a budget to fill with the gram scale. The
estimates only set the starting values in section 2.

| Part | Mass (g), estimate | Height above axle | Tag |
|---|---|---|---|
| 2 × Large Angular Motor (housings ride with the body) | 110–140 | 0 (on the axle) | Est. |
| Raspberry Pi 3B | 42–50 | design | Est. |
| Build HAT + spacers | 20–30 | with the Pi | Est. |
| Auto pHAT + header | 10–15 | with the Pi | Est. |
| Battery (LiPo 1300–2200 mAh) | 80–135 | design (sets the COM) | DS (section 3.6) |
| LEGO frame, axles, pins, wiring | 100–250 | spread over the tower | Est. |
| **Body total** | **about 0.45–0.65 kg** | | |

Pitch inertia about the body's COM: a LEGO tower of height h with mass spread along it has a radius of gyration of
roughly 0.25–0.35 h about its own centre, so a 0.20 m tower of 0.55 kg gives about 0.002–0.0045 kg m² (Est.).
Concentrating the battery at the top raises both the COM and this inertia.

For balancing, the important number is the comparison in 5.3: the motors' output-side inertia is equivalent to
2.05 kg of extra mass at the wheel (69 mm wheels), several times the whole body.

## 5. Derivations

Every derived number is printed by [legobot_numbers.py](legobot_numbers.py)
(`python hardware/legobot/legobot_numbers.py`).

### 5.1 Motor constants

From LEGO's no-load and stall points (linear motor, V = 7.2 V):
R = V / I_stall = 5.14 Ω; k_e = (V − I₀R)/ω₀ = 0.355 V s/rad; k_t = τ_stall / I_stall = 0.179 N m/A.
The linear model misses the sheet's own maximum-efficiency point by about 12% in speed and 25% in current, within
the ±15% tolerance on speed but not on current. Motor-plus-gearbox efficiency at that point is 37%.

From Pybricks' points (their `make_model`): k_t = Δτ/Δi = 0.232 N m/A; friction τ_s = k_t i₀ = 0.023 N m;
R/k_e from the slope of speed against torque; then k_e = 0.411 V s/rad and R = 3.73 Ω. In SI k_e and k_t would be
equal for a lossless motor; the ratio, 0.565, is the gearbox and motor loss beyond the Coulomb friction, and is
what `gear_efficiency` represents.

### 5.2 Friction

At no load all motor torque goes into friction: k_t i₀ = 0.024 N m per motor (LEGO sheet constants) or 0.031 N m
(Pybricks k_t with LEGO's no-load current); Pybricks' own τ_s is 0.023 N m. For the pair, 0.046–0.063 N m, inside
the proposed 0.03–0.08 N m range. How that splits into viscous and Coulomb parts is unknown (8.3).

### 5.3 Output-side inertia

Pybricks estimated a no-load acceleration of 800 °/s in 0.04 s (Large). With the torque available at zero speed,
J = (k_t V/R − τ_s)/a = 1.22e-3 kg m² per motor; the mechanical time constant J R/(k_t k_e) is then 48 ms. Seen as
translating mass at the wheel, 2J/r² is **2.05 kg with the 69 mm wheel**, 1.26 kg with the 88 mm wheel and
3.11 kg with the 56 mm wheel. The Pololu build's equivalent is 0.34 kg. Because it rests on an estimated acceleration, the proposed spread is ±50%
until the spin-up test (8.2).

### 5.4 Sensor noise

σ = noise density × √(noise bandwidth), with the ICM-20948's tabulated noise bandwidths:

- Gyro: 3.3e-3 rad/s (cfg 2), 2.2e-3 rad/s (cfg 3), 1.6e-3 rad/s (cfg 4).
- Accelerometer tilt, raw, at cfg 3: 1.35e-3 rad. Through a complementary filter with τ = 0.5–1 s:
  1.2–1.6e-4 rad.
- Uncalibrated offsets: accelerometer ±50 mg = 0.05 rad of tilt; gyro ±5 °/s = 0.087 rad/s. Both must be calibrated.
- Encoder at r = 34.5 mm: one count = 2π r/360 = 0.60 mm, σ = 0.17 mm from quantization alone. The ±3° accuracy
  figure (including gearbox slack), treated as uniform, gives σ = 1.04 mm. Velocity from counts 10 ms apart:
  σ = (q/Δt)/√6 = 0.025 m/s; 5 ms apart it would be 0.049 m/s, but the data only change every 10 ms.

### 5.5 Back-EMF drag in voltage mode

With PWM duty proportional to the torque command and no back-EMF compensation, the motors add a viscous drag
c_m = n k_t k_e / R = **0.051 N m s/rad** (Large, Pybricks constants). That is 3.5× the Pololu build's 0.0146,
where PHYSICS.md 9.3 already found it the largest unmodeled effect. Section 7.1.

### 5.6 Wheel size

A bigger wheel trades ground force for speed, and also divides the motors' output inertia, as felt at the axle, by
r². Ignoring the pendulum coupling, the axle's acceleration per newton-metre of wheel torque is
a/τ = 1/(r M + J/r), with M the body and wheel mass (0.63 kg) and J the motors' and wheels' inertia. It peaks at
r* = √(J/M).

| Motor inertia (vs Pybricks' estimate) | Best diameter | 56 mm | **69 mm** | 88 mm |
|---|---|---|---|---|
| ×0.5 | 90 mm | 15.8 | **17.0** | 17.6 m/s² per N m |
| ×1 | 126 mm | 9.4 | **10.6** | 11.9 |
| ×1.5 | 154 mm | 6.7 | **7.7** | 8.9 |

Speed headroom matters too, because a voltage-limited motor loses torque as it speeds up. Share of the stall torque
still available at 7.2 V:

| Ground speed | 56 mm | **69 mm** | 88 mm |
|---|---|---|---|
| 0.2 m/s | 57% | **65%** | 73% |
| 0.3 m/s | 35% | **48%** | 59% |
| 0.4 m/s | 14% | **30%** | 45% |
| No-load top speed (6.8–8.4 V) | 0.44–0.55 m/s | **0.54–0.67 m/s** | 0.69–0.86 m/s |

So the 88 mm wheel is better on both counts, by 4–16% in acceleration depending on the true inertia and by 27% in
speed, but the 69 mm wheel loses nothing essential. The spin-up test (8.2) measures J and so says how much a bigger
wheel would gain.

## 6. Latency budget

From the moment the IMU senses a change to the moment motor current responds:

| Stage | Delay | Tag |
|---|---|---|
| IMU low-pass filter group delay | about 1–4 ms (gyro DLPF cfg 2–3) | Est. |
| I²C read of gyro and accelerometer (about 14 bytes) | 0.4 ms at 400 kHz; about 1.5 ms at the Pi's default 100 kHz | Derived |
| Estimator and controller in Python on a Pi 3B | 0.5–2 ms | Est. |
| Linux scheduling | usually under 1 ms; occasional spikes of several ms | Est. |
| Serial command for both motors | 2.5 ms abbreviated (`po 0;set .421;po 1;set -.421`), 3.6 ms in full | Derived |
| HAT firmware parse and PWM update | under 1 ms | Est. |
| Current rise, L/R | about 3 ms | Pyb (estimate) |
| **Total** | **about 6–15 ms** | |

The encoder path is slower still: motor data refresh at 100 Hz (0–10 ms old when read), and each motor's data line
takes about 2 ms to send at 115200 baud. Two motors at 100 Hz use about 40% of the HAT-to-Pi direction.

The sim's 10 ms nominal latency fits this budget. Its current randomization (2.5–20 ms) reaches below what this
hardware can do; 5–25 ms covers the budget with margin.

## 7. Where the current model doesn't match this robot

In order of how much each changes the robot the sim sees. None of these require new hardware; several can be
handled by firmware choices, and the rest are modeling decisions for later.

1. **The Build HAT is a voltage-mode driver.** `DCMotorActuator` assumes the delivered torque equals the command
   until the voltage or current limit binds, which requires current control or exact back-EMF compensation
   (PHYSICS.md 9.3). The Build HAT has neither: its setpoint is PWM duty. Compensating in the controller,
   V = R τ_c/(n η k) + k ω̂, needs ω̂ from the motor's own sensor (100 Hz, a few ms old) and R, which is uncertain by
   ±17% and rises as the winding warms. Uncompensated, the drag is 0.051 N m s/rad (5.5). The tool to check either
   case already exists: `VoltageModeActuator` in `simulation/examples/physics_checks.py`.
2. **`plimit` and `bias` change the actuator.** The HAT multiplies the drive by `plimit` (default 0.1 at the serial
   level, 0.7 in the .NET library) and adds `bias` (a jump of 0.3–0.4 of full scale at zero command). The sim
   models neither. Set `plimit 1` and `bias 0` and the existing model applies, with `supply_voltage` equal to the
   motor rail; otherwise `supply_voltage` must be scaled by `plimit` and the bias is an unmodeled dead-band jump.
   Also unknown: whether the motor rail ("about +7.2V") follows the battery or is regulated (8.5).
3. **The scenario asks for speeds this robot can't reach.** `ScenarioConfig` cruises at 1.3 m/s, sized from the
   Pololu motor's corner speed. With 69 mm wheels the LEGO robot's no-load top speed is 0.54–0.67 m/s across the
   battery's range (0.69–0.86 m/s with 88 mm wheels), and the usable speed under load is lower: at 0.3 m/s half the
   stall torque is already gone (5.6). A LEGO configuration
   needs the scenario's speeds, accelerations, push sizes and the cost's `velocity_limit` re-derived from these
   motors, the way the current ones were derived from the Pololu motor.
4. **Motor data arrive at 100 Hz.** At a 200 Hz control rate, every other encoder reading repeats. The sim's
   sensor model samples fresh state every control step. The fix in firmware is to difference positions 10 ms apart
   (as `velocity_std` above assumes); the model's sample-and-hold and sensor delay are not represented.
5. **Gearbox slack.** The ±3° accuracy includes slack. The sim has no backlash; `position_std` stands in for its
   effect on the measurement, but not for the dead zone in torque transmission, which can cause small limit cycles
   while standing still.
6. **Inductance.** L/R is about 3 ms (Pybricks estimate), against 0.2–0.5 ms for the Pololu motor. The model omits
   inductance; folding it into the latency (section 6 does) is the simple approximation.
7. **The two motor data sources disagree on stall torque** (3.1). The cap and spreads in section 2 cover both until
   measured.
8. **LEGO structure.** Joint play and frame flex add vibration on the IMU and compliance between body and wheels;
   none of it is modeled. Brace the tower and mount the IMU rigidly.
9. **The IMU is idealized.** The controller receives true pitch plus white noise and a constant offset
   (`SensorNoise`); there is no simulated accelerometer, gyro or complementary filter, so neither the axle's
   acceleration nor the IMU's height above the axle affects the pitch estimate (3.4 gives their size). PHYSICS.md 9.5
   names the d θ̈ and d θ̇² terms, and PHYSICS.md 10.2 lists the fix: simulate the accelerometer from ẍ, θ̈ and the
   mount height, plus the gyro, and run the firmware's own filter.

## 8. What to measure, and how

With a gram scale, multimeter, phone slow-motion video, string, tape and a ruler, plus the Build HAT itself as a
data logger. Do 8.1 and 8.2 first: they set the values with the widest spreads.

### 8.1 Body mass properties

| Quantity | Method |
|---|---|
| Masses | Weigh every subassembly to fill section 4, and the finished body without wheels. Scale resolution makes `body_mass` exact for this purpose |
| COM height | Lay the body on its side and balance it across a thin straight edge (a ruler's edge) parallel to the axle; the balance line passes through the COM. Repeat from the other side and average. Or hang it from a string at two different points and photograph the plumb line each time; the lines cross at the COM. Both give about ±3–5 mm |
| Pitch inertia (compound pendulum) | Hang the body from a pin through a hole on the axle line and let it swing below; time 20 small swings (under 10°) with slow-motion video. I_pivot = m g l T²/(4π²), with l the measured pivot-to-COM distance; I_com = I_pivot − m l². Keep the wheels off, or subtract their share |
| Pitch inertia (bifilar pendulum, cross-check) | Hang the body with its pitch axis vertical (on its side) from two parallel strings of length L, a distance d apart, symmetric about the COM. Twist it slightly and time 20 oscillations. I_com = m g d² T² / (16 π² L). Measures inertia about the COM directly, with no subtraction |

For the compound pendulum, m l² is larger than I_com on this robot, so the subtraction amplifies timing error about
4–5× (a 0.1% error in T becomes about 0.5% in I_com); an error in l mostly cancels (3 mm moves I_com about 1%).
Timing 20 periods at 240 frames/s gives T to better than 0.1%, so either method reaches a few percent. Agreement
between the two is the check.

### 8.2 Motor constants, using the Build HAT as the logger

Run each with `plimit 1`, `bias 0`, one motor at a time, wheel attached unless stated. Read `vin` each time.

| Quantity | Method |
|---|---|
| Motor rail voltage | Multimeter across the motor at full duty with the wheel lifted, against `vin`. Says whether `supply_voltage` follows the battery (7.2 of section 7) |
| R | Hold the wheel stalled at a low duty (0.2–0.3) for a second or two; measure the current with the multimeter in series with the battery and subtract the idle current (Pi and HATs). R = duty × V_rail / I. Low duty keeps the winding cool. Settles the 3.73 vs 5.14 Ω question |
| k_e | No-load speed (from POS differences or SPEED) at duties 0.3, 0.5, 0.7, 1.0, with current at each. k_e = (V − I R)/ω; the slope of V against ω is the cleanest estimate |
| Net torque per amp (efficiency) | Wind a string on a LEGO pulley or the wheel (radius r), hang a known mass (weighed water bottle) and find the current that holds it still or lifts it slowly: k_t = m g r / ΔI. η = k_t / k_e |
| Output-side inertia | Spin-up: step the duty from 0 to 0.5, log position at 100 Hz, fit ω(t) = ω∞(1 − e^(−t/τ)). J = τ k_t k_e / R. Repeat with the wheel removed to get the motor alone; the difference is the wheel |
| Friction | No-load current at several speeds (from the k_e runs): τ_f = k_t I₀, fit τ_c + b ω. Coast-down from full speed gives a check |
| Backlash | Hold the motor housing, rock the wheel by hand, and read the position change |
| Stall torque | Same string-and-mass rig, increase the mass until it stalls at full duty (briefly) |

### 8.3 Friction split

The free-run fit in 8.2 separates Coulomb (constant) from viscous (proportional to speed), as PHYSICS.md 9.4
describes. It replaces both `randomize_range` entries.

### 8.4 IMU

| Quantity | Method |
|---|---|
| Gyro noise and bias | Log 30 s with the robot still, motors off; then held still with the motors running at a few speeds. σ and mean of each |
| Pitch offset (balance point) | Find the angle at which the robot balances on its wheels with the motors coasting (hold and release gently, or log the angle where it reverses direction); compare with the IMU's reading there |
| Accelerometer offset | Read the tilt on a level surface, then rotate the robot 180° about the vertical and read again; half the difference is the offset |

### 8.5 Power

Log `vin` and the Pi's under-voltage flag (`vcgencmd get_throttled`) through a full battery discharge with the motors
cycling. Gives the real voltage range for `supply_voltage`, and shows whether the HAT or the Pi drops out below
7.2 V.

### 8.6 Latency and traction

| Quantity | Method |
|---|---|
| Software latency | Timestamp each control step's IMU read, controller output and serial write in the loop; log the distribution. Add the DLPF and L/R terms of section 6 |
| End-to-end check | Command a duty step and timestamp it; find when the position stream first moves. Includes the motor's mechanical response, so compare with the spin-up fit |
| Traction μ | Robot on a board, motors off (the gearbox holds the wheels), tilt until it slides: μ = tan(angle) |

## 9. Sources

- LEGO Education SPIKE Prime technical fact sheets:
  [Large Angular Motor 45602](https://cdn.shopify.com/s/files/1/0539/2056/3365/files/SPIKE_Prime_Tech_fact_sheet_Large_Angular_Motor_45602.pdf),
  [Medium Angular Motor 45603](https://cdn.shopify.com/s/files/1/0539/2056/3365/files/SPIKE_Prime_Tech_fact_sheet_Medium_Angular_Motor_45603.pdf),
  [index of technical specifications](https://education.lego.com/en-us/product-resources/spike-prime/downloads/technical-specifications)
- Pybricks firmware:
  [motor_data.py](https://github.com/pybricks/pybricks-micropython/blob/master/lib/pbio/doc/control/motor_data.py),
  [motor_model.py](https://github.com/pybricks/pybricks-micropython/blob/master/lib/pbio/doc/control/motor_model.py),
  [servo_settings.c](https://github.com/pybricks/pybricks-micropython/blob/master/lib/pbio/src/motor/servo_settings.c),
  [motor documentation](https://docs.pybricks.com/en/latest/pupdevices/motor.html)
- Raspberry Pi Build HAT:
  [documentation](https://www.raspberrypi.com/documentation/accessories/build-hat.html),
  [serial protocol (RP-008140-DS)](https://datasheets.raspberrypi.com/build-hat/build-hat-serial-protocol.pdf),
  [power supply](https://raspberrypi.com/products/build-hat-power-supply)
- SparkFun Auto pHAT: [product page](https://www.sparkfun.com/sparkfun-auto-phat-for-raspberry-pi.html),
  [hookup guide](https://learn.sparkfun.com/tutorials/sparkfun-auto-phat-hookup-guide/all) (not readable by the
  fetch tool; pin list unconfirmed)
- TDK InvenSense [ICM-20948 datasheet DS-000189 v1.3](https://cdn.sparkfun.com/assets/7/f/e/c/d/DS-000189-ICM-20948-v1.3.pdf)
- LEGO wheels: [49295 (88 × 14.34 mm)](https://www.brickowl.com/catalog/lego-wheel-88-x-14-34-with-medium-azure-tire-49295/comesin),
  [39367 (56 mm)](https://www.brickowl.com/catalog/lego-wheel-56-with-black-tire-39367/comesin)
- Battery figures not listed above are from [DATASHEET.md](../../DATASHEET.md) section 3.4 or are marked Est.
