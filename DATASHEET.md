# Datasheet: physical components and robot geometry

Realistic values for every physical constant in `config.py` and `pivot_config.py`. They come from a reference
build made of real, purchasable parts. For each constant, this file gives the simulated value, the realistic
value or range, and where that number comes from. Compiled 2026-11-08.

**Applied 2026-11-08.** `config.py` now uses the realistic values, and the model gained the three missing pieces
(rotor inertia, gearbox efficiency, a pitch-estimate offset); section 7 lists exactly what changed. In the tables
below, the "Sim" column is the value *before* that change.

Every number carries one of these tags:

| Tag | Meaning |
|---|---|
| **DS** | Manufacturer datasheet or product page (linked in Sources) |
| **Derived** | Calculated from DS numbers here; the calculation is in section 5 and in the script it cites |
| **Est.** | Engineering estimate with no published source. Measure it once the part exists (section 8) |

Verdict column: ✓ the sim value is realistic · ~ plausible but worth revisiting · ✗ off by enough to matter.

## Contents

1. [Reference build (bill of materials)](#1-reference-build-bill-of-materials)
2. [Constant-by-constant comparison](#2-constant-by-constant-comparison)
3. [Component datasheets](#3-component-datasheets)
4. [Robot geometry and mass properties](#4-robot-geometry-and-mass-properties)
5. [Derivations](#5-derivations)
6. [Latency budget](#6-latency-budget)
7. [Findings and suggested config changes](#7-findings-and-suggested-config-changes)
8. [What to measure on the real robot](#8-what-to-measure-on-the-real-robot)
9. [Sources](#9-sources)

---

## 1. Reference build (bill of materials)

The parts are chosen to match the simulated robot's scale: a 2S LiPo, a gear ratio near 20:1, 70 mm wheels and
roughly 0.4 N m of total torque. The simulation does not depend on these exact parts. They are the closest real
match, so their datasheets can stand in for the robot's.

| Function | Part | Why this one |
|---|---|---|
| Wheel motors (×2) | **Pololu 20.4:1 Metal Gearmotor 25Dx65L mm HP 6V with 48 CPR encoder** (item 4803) | Its back-EMF constant (5.3 mV s/rad) and gear ratio match the sim's 0.005 and 20 almost exactly. Run from 2S, its no-load wheel speed (65 rad/s) is close to the sim's 74 rad/s. The encoder is on the motor shaft, so it measures relative wheel speed directly (PHYSICS.md 9.3). |
| Wheels (×2) | **Pololu Wheel 70×8 mm** (item 1425) + **Pololu universal hub, 4 mm** (item 1081) | Radius 35 mm, as in the sim. The wheel's bore is a 3 mm D-hole, so the 4 mm motor shaft needs the hub. Alternative: Pololu 80×10 mm Multi-Hub wheel (item 3690), which has a 4 mm D collet but a 40 mm radius. |
| Motor drivers (×2) | **Pololu DRV8874 single brushed DC motor driver carrier** (item 4035) | 2.1 A continuous and 6 A peak, an adjustable current limit, and a current-sense output (about 1.1 V/A). With that output the firmware can close the current loop that the sim's "current-mode" actuator assumes (PHYSICS.md 9.3). A TB6612FNG (1 A continuous per channel) is undersized for a 2 A limit. |
| Battery | **2S 7.4 V 1300 mAh LiPo** (for example, Gens Ace 30C, 79 g, 69×35×17 mm) | The sim's 7.4 V nominal supply. A 2200 mAh pack (about 130–135 g) is the heavier option. |
| IMU | **TDK ICM-42688-P** (current part) or **MPU-6050** (common legacy part) | Both datasheets are in section 3.5. The ICM-42688-P has about 2× lower gyro noise and 6× lower accelerometer noise. |
| Pivot servo (pivot robot only) | **TowerPro MG996R** | `ServoConfig` was already written for this class of servo. |
| Pivot angle sensor (pivot robot only) | **ams AS5600** 12-bit magnetic encoder on the pivot shaft | The MG996R has no position output, so the pivot angle needs its own sensor. |

## 2. Constant-by-constant comparison

### 2.1 Body and wheels (`RobotConfig`)

| Field | Sim | Realistic | Tag | Verdict | Notes |
|---|---|---|---|---|---|
| `body_mass` | 0.80 kg | 0.53–0.67 kg (reference builds); 0.5–0.9 kg is the realistic span | Derived/Est. | ~ | It includes the motors (196 g). See section 4. 0.8 kg is the heavy end, but reachable with a larger battery or a printed chassis. |
| `body_com_height` | 0.10 m | 0.068 m (compact build) to 0.106 m (tall build) | Derived | ✓ | A design choice. Both motors sit on the axle, so the COM rises only as high as the battery is placed. |
| `body_inertia` | 0.0033 kg m² | 0.0031 (compact) to 0.0072 (tall) | Derived | ~ | At a 0.10 m COM, the realistic value is about **2× the sim's**. The sim's 0.0033 implies a 64 mm radius of gyration. A tower tall enough to put the COM at 0.10 m has about 100 mm. |
| `wheel_mass` | 0.10 kg (pair) | **0.034 kg** (pair: 2 × (14 g wheel + 3.2 g hub)) | DS | ✗ | About 3× too heavy, but it matters little next to the rotor inertia below. |
| `wheel_radius` | 0.035 m | 0.035 m (70 mm wheel) or 0.040 m (80 mm) | DS | ✓ | |
| `wheel_inertia` | 6.1e-5 kg m² | Wheels alone: **2.2e-5** (pair). Reflected rotor inertia: **2.7e-4 to 6.7e-4** extra | Derived/Est. | ✗ | The sim's value is a solid 100 g disk. The real wheels are lighter, but each motor's rotor, seen through 20.4:1, has 4–11× the inertia of the wheels themselves. It belongs in the mass matrix of PHYSICS.md 10.3, not folded into `wheel_inertia`. See 5.3. |
| `max_torque` | 0.40 N m | 0.34 N m at a 2.0 A limit; 0.43 N m at 2.5 A (both after gearbox losses) | Derived | ~ | 0.40 N m needs 2.35 A per motor once the gearbox's about 78% efficiency is included. Gearbox rating: 0.39 N m continuous and 0.78 N m intermittent per motor, so either is safe. |
| `wheel_friction` (viscous) | 0, randomized 0–0.001 | **0.0004–0.0013** N m s/rad (total, output side) | Derived | ~ | Comes from the 0.42 A free-run current. Section 5.2 shows how it splits between viscous and Coulomb. |
| `coulomb_friction` | 0, randomized 0–0.02 | **0.03–0.07** N m (total, output side) | Derived | ✗ | The free-run current implies 0.091 N m of total drag at free-run speed. The sim's range is 2–4× too small at the top. |
| `coulomb_friction_speed` | 1.0 rad/s | — | — | — | A numerical smoothing parameter, not a physical constant. |
| `gravity` | 9.81 m/s² | 9.8015 m/s² (West Lafayette, 40.42° N, 187 m) | Derived | ✓ | 0.09% difference. Not worth changing. |

### 2.2 Wheel motors (`ActuatorConfig`), Pololu 25D HP 6V at 20.4:1

| Field | Sim | Realistic | Tag | Verdict | Notes |
|---|---|---|---|---|---|
| `gear_ratio` | 20.0 | 20.4 | DS | ✓ | |
| `torque_constant` | 0.005 N m/A | 0.0053 V s/rad from speed; 0.0049 N m/A from stall | Derived | ✓ | The two estimates agree within 8%. |
| `resistance` | 3.0 Ω | **1.0 Ω** winding (6 V / 6.0 A stall), plus about 0.25 Ω for the driver, wiring and battery: about **1.25 Ω** in total | Derived/Est. | ✗ | 2.4× too high. 3.0 Ω is the 6V *LP* winding, which has about half the speed and a third of the torque. |
| `supply_voltage` | 7.4 V, randomized ±10% (6.66–8.14 V) | 2S LiPo: **8.4 V full**, 7.4 V nominal, 6.4–6.8 V at cutoff under load | DS | ~ | The comment in config.py says "8.1 V (fully charged)". A 2S pack is 8.4 V fully charged. A better distribution: 7.6 V ± 0.8 V. |
| `current_limit` | 2.0 A per motor | 1.5 A continuous (Pololu: ≤25% of stall), 2.1 A continuous for the DRV8874, 3–4 A for short peaks | DS | ✓ | 2.0 A is a reasonable firmware limit. |
| `n_motors` | 2 | 2 | — | ✓ | |
| *(not modeled)* gearbox efficiency | 1.0 | about **0.78** at stall; 42% motor+gearbox at the peak-efficiency point | Derived/DS | ✗ | PHYSICS.md 9.6 assumes a lossless gearbox. |
| *(not modeled)* rotor inertia | 0 | 3–8e-7 kg m² per rotor | Est. | ✗ | See 5.3. |
| *(not modeled)* inductance | 0 | about 0.2–0.5 mH, so L/R ≈ 0.2–0.5 ms ≪ 5 ms | Est. | ✓ | Ignoring it is justified (PHYSICS.md 9.6). |

Robustness spreads (`RobustnessSettings.randomize`): `torque_constant` ±10% ✓ (Pololu quotes ±20% on no-load speed
and ±50% on no-load current, but those are catalogue tolerance bands, not part-to-part spread). `resistance` ±15% ✓:
copper rises 0.39%/°C, so a winding 40 °C hot is 16% higher.

### 2.3 Sensors (`SensorConfig`, per sample at the 200 Hz control rate)

| Field | Sim | Datasheet-derived | Tag | Verdict | Notes |
|---|---|---|---|---|---|
| `pitch_rate_std` | 0.02 rad/s | ICM-42688-P: **6.1e-4**; MPU-6050: **1.1e-3** (98 Hz filter) | Derived | ~ | The sim is 20–30× noisier than the chips. Motor and gear vibration on a real chassis add noise the datasheet doesn't include, so a conservative value is defensible. Measure it on the chassis with the motors running. |
| `pitch_rate_bias` | 0.01 rad/s | Uncalibrated: ICM-42688-P ±8.7e-3 (±0.5 °/s); MPU-6050 ±0.35 (±20 °/s). After calibration at power-on, about 1–2e-3 left | DS/Est. | ✓ | 0.01 represents a calibrated gyro that drifts with temperature. Calibrating at boot is mandatory with the MPU-6050. |
| `pitch_std` | 0.005 rad (white) | White part after a complementary filter: **2e-4 to 3e-4**. Constant offset from mounting and balance point: **0.009–0.017** (0.5–1°) | Derived/Est. | ~ | The real error is not white noise. It is a constant offset plus slow errors whenever the robot accelerates (PHYSICS.md 9.5). Adding a constant pitch offset would be more realistic than a larger white σ. |
| `position_std` | 5e-4 m | Encoder quantization: **6.5e-5 m**. Gearbox backlash of 0.5–2° adds 0.3–1.2 mm of play | Derived/Est. | ✓ | 979.62 counts per wheel revolution means 0.224 mm per count. The sim's value is a fair stand-in for backlash. |
| `velocity_std` | 0.02 m/s | **0.018 m/s** (difference of two quantized positions, 5 ms apart) | Derived | ✓ | Matches. |

### 2.4 Timing (`TimingConfig`, `RobustnessSettings.latency_range_s`)

| Field | Sim | Realistic | Tag | Verdict |
|---|---|---|---|---|
| `latency_s` | 10 ms (randomized 5–20 ms) | **about 1.5–6 ms** (section 6) | Derived/Est. | ✓ (conservative) |
| `control_dt_s` | 5 ms (200 Hz) | Any 32-bit MCU can run 1 kHz. The IMU outputs data at 1 kHz | DS | ✓ |

### 2.5 Pivot robot (`pivot_config.py`)

| Field | Sim | Realistic | Tag | Verdict | Notes |
|---|---|---|---|---|---|
| `stall_torque` | 1.0 N m | 1.08 N m at 6 V (11 kg cm); 0.92 at 4.8 V | DS | ✓ | Already derated a little. |
| `no_load_time_per_60deg_s` | 0.15 s | 0.15 s at 6 V; 0.17–0.19 s at 4.8 V | DS | ✓ | |
| `update_period_s` | 20 ms | 20 ms (analog servo, 50 Hz) | DS | ✓ | |
| `angle_limit_deg` | ±60° | Mechanical travel of about ±60° (120° total at the standard 1–2 ms pulse) | Est. | ✓ | |
| `pivot_angle_std_deg` | 0.5° | AS5600 quantization: 0.088° per count, so σ = 0.025° | Derived | ✓ (conservative) | |
| `upper_mass` | 0.25 kg | MG996R alone is 55 g (TowerPro also lists 80 g, probably with accessories) | DS | ✓ | 0.25 kg leaves about 170 g for the bracket and payload. |
| other body splits | — | Follow from section 4 | — | ✓ | They keep the base robot's 0.80 kg and 0.10 m, so they inherit its verdicts. |

---

## 3. Component datasheets

### 3.1 Pololu 25D metal gearmotors (all windings at rated voltage)

Values are from the Pololu category tables. Stall values are, in Pololu's words, "a theoretical extrapolation of
performance at loads far from stall."

| Winding | Rated V | Bare motor free-run | Bare stall torque | Stall current | Free-run current (no encoder / encoder) | 20.4:1 free-run | 20.4:1 stall torque |
|---|---|---|---|---|---|---|---|
| **6V HP** | 6 | 10,000 rpm | 0.3 kg cm | 6.0 A | 420 / 500 mA | 480 rpm | 4.8 kg cm (0.471 N m) |
| 6V LP | 6 | 6,200 rpm | 0.15 kg cm | 2.0 A | 100 / 120 mA | 290 rpm | 2.5 kg cm (0.245 N m) |
| 12V HP | 12 | 10,000 rpm | 0.4 kg cm | 5.0 A | 250 / 300 mA | 500 rpm | 7.4 kg cm (0.726 N m) |
| 12V MP | 12 | 8,200 rpm | 0.17 kg cm | 1.8 A | 80 / 100 mA | 380 rpm | 3.2 kg cm (0.314 N m) |
| 12V LP | 12 | 5,600 rpm | 0.14 kg cm | 0.9 A | 50 / 60 mA | 280 rpm | 2.4 kg cm (0.235 N m) |

**Item 4803 (20.4:1, HP 6V, 48 CPR encoder):** 98 g; 25 mm diameter × 65 mm long; 4 mm D-shaft, 12.5 mm long;
480 rpm ±20% and 0.50 A ±50% at no load (6 V); peak efficiency 42% at 390 rpm, 0.92 kg cm, 1.5 A; encoder 48 CPR
on the motor shaft = **979.62 counts per output revolution**. Pololu's load guidance: ≤4 kg cm (0.39 N m)
continuous and ≤8 kg cm (0.78 N m) intermittent; "25% or less of the stall current" for continuous operation.
Not published: winding resistance, torque constant, rotor inertia, inductance, backlash.

Derived constants per winding (section 5.1), for comparing alternatives at 7.6 V with 0.25 Ω of extra series
resistance:

| Winding | R (Ω) | k_e (mV s/rad) | k_t from stall (mN m/A) | Gearbox η at stall | No-load wheel speed at 7.6 V | Top speed at r = 35 mm |
|---|---|---|---|---|---|---|
| **6V HP** | 1.00 | 5.33 | 4.90 | 0.78 | 65 rad/s | 2.28 m/s |
| 6V LP | 3.00 | 8.78 | 7.35 | 0.82 | 41 rad/s | 1.42 m/s |
| 12V HP | 2.40 | 10.9 | 7.85 | 0.91 | 31 rad/s | 1.09 m/s |
| 12V MP | 6.67 | 13.4 | 9.26 | 0.92 | 26 rad/s | 0.91 m/s |

The 12V windings on 2S cannot reach the sim's speeds: the goal profile peaks at 0.71 m/s, and the robot needs
headroom above that to recover. The 6V LP is the sim's current resistance, but its torque at a 2 A limit is barely
0.4 N m with nothing to spare. **6V HP is the match.** Its rated voltage is 6 V, so a 2S pack (up to 8.4 V) runs it
up to 40% over rating. That is common practice with a current limit, but it shortens brush life.

### 3.2 Wheels and hubs (Pololu)

| Item | Size | Mass each | Bore | Notes |
|---|---|---|---|---|
| 1420 | 60 × 8 mm | ≈11 g (0.4 oz) | 3 mm D press-fit | silicone tire |
| **1425** | **70 × 8 mm** | **≈14 g (0.5 oz)** | 3 mm D press-fit | accepts the universal mounting hubs |
| 1430 | 80 × 10 mm | ≈20 g (0.7 oz) | 3 mm D press-fit | |
| 1435 | 90 × 10 mm | ≈23 g (0.8 oz) | 3 mm D press-fit | |
| 3690 / 3691 | 80 × 10 mm Multi-Hub | 23 g | collets for 3/4 mm, D or round | direct fit on the 25D's 4 mm D-shaft |
| **1081** (hub) | 19 mm dia × 5 mm, 4 mm bore | **3.2 g** | #4-40 holes | 1997 is the M3 version |

### 3.3 Motor drivers

| Part (Pololu carrier) | Motor V | Continuous / peak per channel | Current sense / limit |
|---|---|---|---|
| **DRV8874** (4035, single) | 4.5–37 V | 2.1 A / 6 A | IPROPI about 1.1 V/A; adjustable current regulation |
| DRV8876 (4036, single) | 4.5–37 V | 1.3 A / 3.5 A | about 2.5 V/A; adjustable |
| TB9051FTG (2997, single) | 4.5–28 V | 2.6 A / 5 A | about 500 mV/A |
| TB67H420FTG (2999) | 10–47 V | 1.7 A dual, 3.4 A single | VREF chopping limit. **10 V minimum: too high for 2S** |
| MAX14870 dual shield (2519) | 4.5–36 V | 1.7 A / 2.5 A | optional sense resistor |
| TB6612FNG (713, dual) | 4.5–13.5 V | 1 A / 3 A | none |
| DRV8833 (2130, dual) | 2.7–10.8 V | 1.2 A / 2 A | optional sense resistors |
| DRV8835 (2135, dual) | 0–11 V | 1.2 A / 1.5 A | none |

The driver's on-resistance is in series with the winding. I could not confirm the DRV8874's from the datasheet; a
sibling part (DRV8873) is 75 mΩ per switch at 25 °C, about 0.15 Ω for the high and low switch in series and more
when hot. The 0.25 Ω used above for driver, wiring, connector and battery together is an estimate.

### 3.4 Battery: 2S LiPo

| Quantity | Value |
|---|---|
| Full charge | 4.20 V/cell, so **8.4 V** |
| Nominal | 3.7 V/cell, so 7.4 V |
| Storage | about 3.8–3.85 V/cell, so 7.6–7.7 V |
| Minimum (cutoff) | 3.0 V/cell absolute; 3.2–3.4 V/cell under load in practice, so 6.4–6.8 V |
| Mass, 1300 mAh | 69–88 g depending on C rating (Gens Ace 30C: 79 g, 69 × 35 × 17 mm) |
| Mass, 2200 mAh | 130–135 g (103 × 34 × 19 mm) |
| Internal resistance | not found for these packs; tens of mΩ for a 2S pack of this size (Est.). Sag at 4 A is about 0.1–0.2 V |

### 3.5 IMUs

| Quantity | MPU-6050 | ICM-42688-P |
|---|---|---|
| Gyro noise density | 0.005 °/s/√Hz | 0.0028 °/s/√Hz |
| Gyro RMS noise | 0.05 °/s (datasheet, at its test bandwidth) | — |
| Gyro zero-rate offset, initial, 25 °C | ±20 °/s (quoted from PS-MPU-6000A) | ±0.5 °/s |
| Accelerometer noise density | 400 µg/√Hz | 65 µg/√Hz (X, Y), 70 µg/√Hz (Z) |
| Gyro output rate | 8 kHz with the low-pass filter off, 1 kHz with it on | up to 32 kHz |
| Low-pass filter delay (gyro) | 0.98 ms at 256 Hz; 1.9 ms at 188 Hz; 2.8 ms at 98 Hz; 4.8 ms at 42 Hz | configurable |

### 3.6 Encoder

Pololu 48 CPR quadrature magnetic encoder on the motor shaft: 48 × 20.4 = **979.62 counts per wheel revolution**,
so **0.224 mm** of travel per count at r = 35 mm (0.257 mm at 40 mm).

### 3.7 Servo: TowerPro MG996R

| Quantity | 4.8 V | 6.0 V |
|---|---|---|
| Stall torque | 9.4 kg cm (0.92 N m) | 11 kg cm (1.08 N m) |
| Speed | 0.17–0.19 s/60° | 0.15 s/60° (6.98 rad/s) |
| Operating voltage | 4.8–6.6 V | |
| Current | 170 mA no-load, 1.4 A stall (TowerPro) | |
| Mass | 55 g (spec list; 80 g elsewhere on the same page) | |
| Size | 40.7 × 19.7 × 42.9 mm | |
| Dead band | 1 µs | |

---

## 4. Robot geometry and mass properties

Two reference body layouts built from the BOM, computed with `datasheet_numbers.py` (section 5). Axle at height z = 0; the
pitch axis runs along the axle. Box inertias use m(a² + c²)/12 about the part's own centre, then the parallel-axis
theorem.

**Compact build** (three 120 × 70 × 3 mm acrylic decks at z = 20, 100 and 180 mm; four 190 mm M4 rods; battery on
the top deck):

| Part | Mass (g) | z (mm) | Tag |
|---|---|---|---|
| 2 × 25D HP gearmotor with encoder | 196 | 0 (on the axle) | DS |
| 2 × motor bracket + screws | 20 | 0 | Est. |
| Bottom deck | 30 | 20 | Derived (acrylic, 1190 kg/m³) |
| 2 × driver carrier + wiring | 12 | 25 | Est. |
| Middle deck | 30 | 100 | Derived |
| MCU + IMU + 5 V regulator + perfboard | 35 | 108 | Est. |
| Top deck | 30 | 180 | Derived |
| 2S 1300 mAh LiPo | 79 | 190 | DS |
| 4 × M4 steel rod, 190 mm | 56 | 85 | Derived |
| Wiring, switch, connectors, fasteners | 40 | 80 | Est. |
| **Body total** | **528 g** | **COM 68 mm** | **I_com = 0.0031 kg m²** |

**Tall build** (decks at 20, 130 and 240 mm; 250 mm rods; 2S 2200 mAh pack, 135 g, on top; 60 g of printed side
panels): **672 g, COM 106 mm, I_com = 0.0072 kg m²**.

What this means for the sim's body:
- **The motors are about 30% of the body mass, and they sit on the axle.** The COM height is set almost entirely
  by where the battery goes. Moving the 135 g pack from the top deck to the bottom lowers the COM by about 45 mm.
- A COM at 0.10 m needs a tower about 0.25 m tall. That tower's inertia is about 0.007 kg m², twice the sim's
  0.0033. The sim's mass and inertia together describe a body whose mass is more concentrated near the COM than a
  battery-on-top tower can be.
- A lower COM makes the robot fall faster: the inverted-pendulum time constant scales as √l. The compact build is
  the harder one to balance.

Wheels: the pair is 2 × (14 + 3.2) = 34 g. Inertia per 70 mm wheel is between a disk (8.6e-6) and a ring (1.7e-5)
kg m², because the silicone tire sits at the rim. Taking 0.65 m r² gives 1.1e-5 per wheel and **2.2e-5 for the
pair** (Est.). The hub adds 1.4e-7, which is negligible.

---

## 5. Derivations

Every derived number in this file is printed by
[simulation/examples/datasheet_numbers.py](simulation/examples/datasheet_numbers.py)
(`python simulation/examples/datasheet_numbers.py`). Its inputs are the datasheet values and estimates quoted here.

### 5.1 Motor constants from catalogue numbers

For the bare 6V HP motor (6 V, 10,000 rpm free-run, 0.42 A free-run, 6.0 A and 0.3 kg cm at stall):

```math
R = \frac{V}{I_{\text{stall}}} = 1.00\ \Omega,\qquad
k_e = \frac{V - I_0 R}{\omega_0} = \frac{6 - 0.42}{1047} = 5.33\ \text{mV s/rad},\qquad
k_t = \frac{\tau_{\text{stall}}}{I_{\text{stall}}} = \frac{0.0294}{6.0} = 4.90\ \text{mN m/A}.
```

In SI units k_e and k_t should be equal; they agree to 8%, which is within the catalogue's extrapolation error.
Gearbox efficiency at stall is the geared stall torque over the ideal value:
η = 0.471 / (0.0294 × 20.4) = **0.78**. Output torque per amp, losses included: 78.5 mN m/A per motor. At
2.0 A per motor, the pair gives 0.34 N m.

### 5.2 Friction from free-run current

At free run, all of the motor's torque goes into internal friction (brushes, bearings, gear mesh). Seen at the
wheels, for both motors together:

```math
\tau_0 = n\,N\,k\,I_0 = 2 \times 20.4 \times 0.00533 \times 0.42 = 0.091\ \text{N m at } \omega_0 = 51\ \text{rad/s}.
```

Splitting it into Coulomb plus viscous, with a fraction α Coulomb, gives τ_c = α τ_0 and b = (1 − α) τ_0 / ω_0:

| Coulomb fraction α | τ_c (N m) | b (N m s/rad) |
|---|---|---|
| 0.3 | 0.027 | 0.0013 |
| 0.5 | 0.046 | 0.0009 |
| 0.8 | 0.073 | 0.0004 |

The split is unknown, so the realistic ranges are τ_c ≈ 0.03–0.07 N m and b ≈ 0.0004–0.0013 N m s/rad. Small
gearmotors are usually Coulomb-dominated at low speed. PHYSICS.md 9.4 has the measurement that pins this down.
Gearbox losses under load (the 78% efficiency) come on top of this no-load drag.

**Consequence for the tuned gains:** the viscous range sits under the current gains' runaway threshold
b < r k_p k_v = 0.018 (PHYSICS.md 9.4). But a real Coulomb load of 0.05 N m is about 2.5× the largest value the
robustness test draws today.

### 5.3 Reflected rotor inertia

Pololu does not publish rotor inertia. Estimate: a 370-class armature (about 10–20 g of laminations and copper,
radius 8–9 mm) treated as a solid cylinder has J_r = ½ m r² ≈ **3.2e-7 to 8.1e-7 kg m²**. Check: the mechanical
time constant J_r R / (k_t k_e) comes out at 12–31 ms, the usual range for small brushed motors.

Seen at the wheel, through the gearbox, for both motors (PHYSICS.md 10.3):

```math
J = n N^2 J_r = 2 \times 20.4^2 \times J_r \approx 2.7\times10^{-4}\ \text{to}\ 6.7\times10^{-4}\ \text{kg m}^2 .
```

That is **4–11× the sim's entire `wheel_inertia`** (6.1e-5) and 12–30× the real wheels' 2.2e-5. As an equivalent
translating mass, J/r² is 0.22–0.55 kg, comparable to the body itself. It slows how fast the wheels can be
accelerated, which is exactly what balancing relies on. The rotor spins relative to the body, so the term enters
the mass matrix in the three places that PHYSICS.md 10.3 lists.

### 5.4 Sensor noise at the control rate

Datasheets give white-noise densities. A per-sample σ needs the bandwidth of the filter in front of the sample:
σ = ND × √(ENBW), with ENBW ≈ (π/2) f₋₃dB for the IMU's internal low-pass filter.

- Gyro at a 98 Hz filter: MPU-6050 0.062 °/s = 1.1e-3 rad/s; ICM-42688-P 0.035 °/s = 6.1e-4 rad/s.
- Accelerometer tilt, raw, at a 44 Hz filter: MPU-6050 3.3e-3 rad; ICM-42688-P 5.8e-4 rad.
- Complementary filter with time constant τ = 0.5–1 s: the accelerometer passes only below 1/(2πτ), so its
  share of the pitch noise is 2e-4 to 2.8e-4 rad (MPU-6050) or 3.5e-5 to 4.9e-5 rad (ICM-42688-P). The gyro's
  integrated noise adds 2e-5 to 6e-5 rad. A gyro bias b_g becomes a pitch offset of b_g τ: 0.005–0.010 rad for
  the sim's 0.01 rad/s.
- Encoder: one count is q = 2πr / 979.62 = 0.224 mm. Uniform quantization gives σ_x = q/√12 = 0.065 mm.
  Differencing two counts 5 ms apart gives σ_v = (q/dt)/√6 = 0.018 m/s.

### 5.5 Local gravity

International gravity formula at 40.42° N, with a free-air correction for 187 m of elevation: g = 9.8015 m/s².

---

## 6. Latency budget

From the moment the IMU senses the robot to the moment the motor current responds:

| Stage | Delay | Tag |
|---|---|---|
| IMU low-pass filter group delay | 1.9–2.8 ms (MPU-6050 at 188–98 Hz); about 1–3 ms (ICM-42688-P, depends on settings) | DS / Est. |
| Sensor read over I²C (about 20 bytes at 400 kHz) or SPI | 0.05–0.5 ms | Derived |
| Estimator and controller computation (32-bit MCU) | < 0.5 ms | Est. |
| PWM update (20 kHz carrier) | about 0.05 ms | Derived |
| Current rise, L/R | 0.2–0.5 ms | Est. |
| **Total** | **about 1.5–6 ms** | |

The sim's 10 ms nominal and 5–20 ms robustness range are conservative. Any extra software filtering on the gyro
or encoder velocity would add to this; a first-order filter at f_c adds about 1/(2π f_c) of delay.

---

## 7. Findings and the config changes made

In order of how much they change the robot the sim sees. **All applied on 2026-11-08** (the "Applied" line under
each). Each one changes every score, so every run in `results/` predates the current robot and needs re-running
(CostConfig's note on comparability applies). On the current robot the 2026-11-06 tuned PID gains score about 55
on validation with 1 fall in 30 (0.140 before); the hand-tuned gains are unchanged at about 2.1. Turning the rotor
inertia off alone brings the tuned gains back to about 4: it is the largest single change.

1. **Rotor inertia is missing and is the largest inertia on the wheel side.** It is 2.7–6.7e-4 kg m² against the
   modelled 6.1e-5 (5.3). Implement the mass-matrix change in PHYSICS.md 10.3, with
   `rotor_inertia ≈ 5e-7` per motor and a wide randomization spread (±50%) until it is measured.
   *Applied:* `RobotParams.rotor_inertia` in both engines and the pivot model; `ActuatorConfig.rotor_inertia`
   = 5e-7 per motor, reflected as 2 × 20.4² × 5e-7 = 4.2e-4 kg m²; randomized ±50%.
2. **Winding resistance is 1.0 Ω, not 3.0** (about 1.25 Ω including the driver). In current mode this only moves
   the voltage limit (more torque at speed). In **voltage mode**, the back-EMF drag
   c_m = nN²k²/R rises from 0.0067 to **0.019 N m s/rad**, which is above the tuned gains' runaway threshold of
   0.018 (PHYSICS.md 9.3–9.4). With this motor, closed-loop current control (the DRV8874's current sense) or exact
   back-EMF compensation is required, not optional.
   *Applied:* `resistance` 1.25 Ω, with `torque_constant` 0.0053 and `gear_ratio` 20.4.
3. **Friction ranges are too narrow.** Suggested `randomize_range`: `coulomb_friction` 0.02–0.08 N m,
   `wheel_friction` 0–0.0015 N m s/rad (5.2). *Applied.*
4. **Gearbox efficiency of about 0.78** is unmodelled. Either add an efficiency factor on output torque
   (PHYSICS.md 10.3) or drop the effective torque: at the 2 A limit the robot has 0.34 N m, not 0.40.
   *Applied, both:* `MotorParams.gear_efficiency` 0.78 (randomized ±10%) and `max_torque` 0.34 N m.
5. **Body inertia is likely low.** For a 0.10 m COM, 0.006–0.007 kg m² is realistic (section 4). Raise the
   nominal value or widen its spread (currently ±25%). *Applied:* `body_inertia` 0.0065, spread ±25% kept.
6. **Wheel mass** 0.10 kg → 0.034 kg (pair), and **wheel_inertia** 6.1e-5 → 2.2e-5, *once rotor inertia is
   modelled separately*. Lowering them before item 1 is done would make the robot unrealistically nimble.
   *Applied,* together with item 1.
7. **Battery voltage:** nominal 7.6 V and ±0.8 V (6.8–8.4 V). Fix the config.py comment ("8.1 V fully charged"
   should be 8.4 V). *Applied:* `supply_voltage` 7.6 V, spread ±10.5%.
8. **IMU model:** add a constant pitch offset (σ ≈ 0.01 rad across robots) for mounting error and balance-point
   offset. The sim's white `pitch_std` and `pitch_rate_std` can come down by 10× or more if the chassis measurement
   (section 8) confirms datasheet-level noise. *Applied:* `SensorNoise.pitch_bias`, 0.01 rad in `SensorConfig`
   (a fixed offset for every run, like the gyro bias; not yet drawn per robot). The white noise levels are kept
   until measured.
9. **Already realistic:** `gear_ratio`, `torque_constant`, `current_limit`, `wheel_radius`, `body_com_height`
   (as a design choice), `velocity_std`, `position_std`, `latency_s` (conservative), `gravity`, and every servo
   parameter.

## 8. What to measure on the real robot

The constants above that are estimates, and the cheapest way to pin each one down:

| Quantity | Method |
|---|---|
| Body mass, COM height | Scale; balance the body on a knife edge parallel to the axle |
| Body inertia | Hang it as a pendulum from the axle and time the swing: I_com = M g l T² / (4π²) − M l² |
| Motor R | Hold the shaft stalled briefly at low voltage; R = V/I (then subtract the leads) |
| k_e | Spin one motor with the other as a generator, or measure no-load speed and current at several voltages |
| Friction τ_c and b | Free-run current at several PWM levels: τ_f = nNk_t i₀, fit τ_c + bω (PHYSICS.md 9.4) |
| Rotor + wheel inertia | Spin-down test: cut power at a known speed and fit the deceleration with the friction model above |
| Gearbox backlash | Lock the motor shaft and rock the wheel; read the play with a dial or the encoder on the wheel side |
| Gyro noise and bias | Log the gyro with the robot still, motors off and then motors running; take σ and the mean |
| Pitch offset | Find the angle where the robot balances with no torque; compare it with the IMU's zero |
| Latency | Toggle a pin at the start of the control step and scope it against the motor current |
| Traction μ | Tilt a plank with the robot on it, wheels locked, until it slides: μ = tan(angle) |

## 9. Sources

- Pololu 25D metal gearmotors: [category](https://www.pololu.com/category/115/25d-metal-gearmotors),
  [6V HP](https://www.pololu.com/category/183/6v-high-power-hp-25d-mm-gearmotors),
  [6V LP](https://www.pololu.com/category/182/6v-low-power-lp-25d-mm-gearmotors),
  [12V HP](https://www.pololu.com/category/186/12v-high-power-hp-25d-mm-gearmotors),
  [12V MP](https://www.pololu.com/category/185/12v-medium-power-mp-25d-mm-gearmotors),
  [12V LP](https://www.pololu.com/category/184/12v-low-power-lp-25d-mm-gearmotors),
  [item 4803 specs](https://www.pololu.com/product/4803/specs),
  [25D datasheet PDF](https://www.pololu.com/file/0J1829/pololu-25d-metal-gearmotors-rev-2-0.pdf) (not read)
- Pololu wheels and hubs: [70×8 mm](https://www.pololu.com/product/1425/specs),
  [60×8](https://www.pololu.com/product/1420/specs), [80×10](https://www.pololu.com/product/1430/specs),
  [90×10](https://www.pololu.com/product/1435/specs), [80×10 Multi-Hub](https://www.pololu.com/product/3690/specs),
  [4 mm hub 1081](https://www.pololu.com/product/1081/specs), [M3 hub 1997](https://www.pololu.com/product/1997/specs)
- Motor drivers: [DRV8874](https://www.pololu.com/product/4035), [DRV8876](https://www.pololu.com/product/4036),
  [TB9051FTG](https://www.pololu.com/product/2997), [TB67H420FTG](https://www.pololu.com/product/2999),
  [MAX14870](https://www.pololu.com/product/2519), [TB6612FNG](https://www.pololu.com/product/713),
  [DRV8833](https://www.pololu.com/product/2130), [DRV8835](https://www.pololu.com/product/2135),
  [DRV8873 electrical characteristics](https://ti.com/document-viewer/DRV8873/datasheet/electrical-characteristics-a-drv8873-q1-electcharact-1487092951-electchar-sections)
- Servo: [TowerPro MG996R](https://www.towerpro.com.tw/product/mg996r/)
- IMUs: [TDK ICM-42688-P](https://invensense.tdk.com/products/icm-42688-p/),
  [ICM-42688-P ZRO discussion](https://adm.invensense.tdk.com/zero-rate-offset-icm-42688-p),
  [TDK MPU-6050 listing](https://www.datasheets.com/tdk/mpu-6050),
  [MPU-6050 noise discussion (Pololu forum)](https://forum.pololu.com/t/mpu6050-evaluation-rate-noise-density-rms-noise/8721),
  [MPU-6050 gyro bias discussion](https://adm.invensense.tdk.com/mpu6050-gyro-bias),
  [MPU-6050 low-pass filter table (I2Cdev source)](https://os.mbed.com/users/amq/code/MPU6050/docs/tip/MPU6050_8cpp_source.html),
  [mpu6050-dmp filter config](https://docs.rs/crate/mpu6050-dmp/latest/source/src/config.rs)
- Batteries: [Gens Ace 2S 1300 mAh](https://www.gensace.de/gens-ace-soaring-1300mah-7-4v-30c-2s1p-lipo-battery-pack-with-t3-plug.html),
  [Gens Ace 2S 1300 mAh 45C](https://www.amainhobbies.com/gens-ace-2s-lipo-battery-45c-7.4v-1300mah-gea2s130045d/p1336212),
  [Gens Ace 2S 1300 mAh (Botland)](https://botland.com.pl/en/battery-li-half/4616-gens-ace-1300mah-74v-6928493332108.html),
  [2S 2200 mAh](https://smalladdictionsrc.com/gens-ace-2s-lipo-battery-60c-7-4v-2200mah-w-xt-60-connector),
  [2S LiPo voltages](https://batteryequivalents.com/lipo-2s-battery-101-all-about-lipo-2s-batteries.html),
  [ESC cutoff defaults](https://www.teamtekin.com/news/?p=4754)
- Motor inductance reference (RS-380PH, measured 165 µH):
  [simulation slides](https://www.slideshare.net/slideshow/the-simulation-of-dc-motor-control-circuit/118793032)
