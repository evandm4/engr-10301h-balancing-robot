"""
=============================================================================
 LEGO ROBOT CONFIGURATION ("legobot")
=============================================================================
The LEGO SPIKE build: two Large Angular Motors (45602) driving 69 mm wheels
directly, a Raspberry Pi 3B with the Build HAT, the SparkFun Auto pHAT's
ICM-20948 IMU, and a 2S LiPo. hardware/legobot/DATASHEET.md gives the source
of every value (section 2) and the spread to randomize until it is measured
(section 8 says how).

Built from config.py's section classes, the way pivot_config.py builds on
config.py, but as a complete ProjectConfig of its own: every section whose
values depend on the robot is written out here, and config.py's Pololu robot
is not touched. Sections that are not about the robot (controller limits,
search ranges, LQR search, the robustness test's seeds and chaos penalty)
are config.py's, and are named below where they are reused on purpose. The
controller section differs only in its lean cap (A3).

Select it with ROBOT=legobot or --robot legobot (config.py: load_config).

Values tagged Est. or Pyb in DATASHEET 2 are placeholders until the bench
measurements (SOFTWARE_PLAN C3) replace them; the robustness spreads cover
them until then.
"""

from dataclasses import replace

from config import (
    ActuatorConfig,
    ControllerConfig,
    CostConfig,
    ProjectConfig,
    RobotConfig,
    RobustnessSettings,
    ScenarioConfig,
    SensorConfig,
    TimingConfig,
)

# ----------------------------------------------------------------- ROBOT ---
# DATASHEET 2.1. The motor housings ride with the body; the motors' output-side
# inertia is ActuatorConfig.rotor_inertia below, not wheel_inertia.
ROBOT = RobotConfig(
    body_mass=0.44,                # kg, measured 2026-11-22 (gram scale): motors, Pi, both HATs, battery, frame
    body_com_height=0.09,          # m, Est. (design choice: where the battery sits)
    body_inertia=0.0035,           # kg m^2 about the body's own COM, Est. (section 4)
    wheel_mass=0.078,              # kg, both wheels, measured: 39 g each with axle
    wheel_radius=0.0345,           # m, half the 69 mm diameter
    wheel_inertia=6.0e-5,          # kg m^2, both wheels, 0.65 m r^2 each (tire at the rim)
    max_torque=0.50,               # N m total: 2 x LEGO's 0.25 N m stall (Pybricks implies 0.85)
    wheel_friction=0.0,            # N m s/rad, Est.: no source separates viscous drag
    coulomb_friction=0.046,        # N m, both motors: Pybricks' 0.023 each (LEGO's no-load current: 0.024-0.031)
    coulomb_friction_speed=1.0,    # rad/s, as config.py (PHYSICS.md 9.4)
    gravity=9.81,
)

# -------------------------------------------------------------- ACTUATOR ---
# DATASHEET 2.2: a lumped output-side motor (gear_ratio = 1). Every published and
# measured number is at the output shaft, so the motor model gives torque = n eta k i
# and back-EMF = k w directly. Requires the Build HAT at plimit 1 and bias 0
# (DATASHEET 7.2). The Build HAT is a voltage-mode driver (PWM duty, no current
# sensing), so the model is "voltage" with back-EMF feedforward (SOFTWARE_PLAN B1):
# the robot code computes the drive voltage from the nominal constants below and the
# measured wheel speed; robust scoring randomizes the true ones. Without feedforward
# every baseline falls in every run (drag 0.051 N m s/rad and a 4.7 V ceiling).
ACTUATOR = ActuatorConfig(
    model="voltage",
    voltage_feedforward=True,
    gear_ratio=1.0,                # lumped: constants below are output-referred
    torque_constant=0.411,         # V s/rad, output k_e (Pybricks; LEGO's sheet: 0.355)
    resistance=4.4,                # ohm incl. driver and wiring: midpoint of 3.73 (Pybricks) and 5.14 (LEGO stall)
    supply_voltage=7.6,            # V: 2S LiPo (6.8-8.4) x plimit 1
    current_limit=2.5,             # A: no current control on the HAT; set never to bind (stall <= 2.25 A)
    n_motors=2,
    gear_efficiency=0.565,         # net torque per amp / k_e: 0.232 / 0.411 (LEGO's sheet: 0.50)
    rotor_inertia=1.22e-3,         # kg m^2 per motor, output side (Pybricks estimate); reflected 2.44e-3
)

# ---------------------------------------------------------------- SENSOR ---
# DATASHEET 2.3 and 3.4. The ICM-20948's gyro and accelerometer are simulated and fused by
# the robot code's complementary filter (model "imu", SOFTWARE_PLAN B2), so the pitch the
# controller sees carries the accelerometer's acceleration error (low-passed) and the gyro
# bias turns into an offset of bias x tau. In this model pitch_std and pitch_bias are the
# accelerometer tilt's. Noise kept at 10x the datasheet for both sensors (gyro 0.02 rad/s
# against 2.2e-3; accelerometer tilt 0.0135 rad against 1.35e-3 at DLPF cfg 3) until
# measured on the chassis (8.4): LEGO frames rattle. Velocity comes from the motors' own
# encoders as the Build HAT reports them (SOFTWARE_PLAN B3): 360 counts per turn, sampled at
# 100 Hz, about 5 ms in transit (two 2 ms data lines; DATASHEET 6), held between samples, and
# differenced 10 ms apart: 0.025 m/s of quantization noise at r = 34.5 mm, averaged over
# speeds (DATASHEET 5.4). Differencing 5 ms apart raised every baseline's cost (B3).
SENSOR = SensorConfig(
    model="imu",
    pitch_std=0.0135,              # rad, accelerometer tilt per sample (raw, before the filter)
    pitch_bias=0.015,              # rad, accelerometer offset after calibration, and LEGO mount play
    pitch_rate_std=0.02,           # rad/s, gyro
    pitch_rate_bias=0.01,          # rad/s, gyro, after calibrating at boot
    position_std=0.0010,           # m: unused with the encoder (kept for the noise model)
    velocity_std=0.025,            # m/s: unused with the encoder (kept for the noise model)
    encoder_counts=360,
    encoder_period_s=0.010,
    encoder_delay_s=0.005,
    imu_height=0.10,               # m above the axle, Est. (the HAT stack's height; DATASHEET 3.4)
    filter_time_constant=0.5,      # s: 1 s raised the baselines' cost 15-25% more, 0.25 s was mixed (B2)
)

# ---------------------------------------------------------------- TIMING ---
# DATASHEET 2.4 and 6: budget about 6-15 ms; the 3 ms L/R is folded into it.
TIMING = TimingConfig(
    latency_s=0.010,
    dt_s=0.0025,
    demo_dt_s=0.001,
    control_dt_s=0.005,            # 200 Hz (motor data refresh every 10 ms; DATASHEET 7.4)
    fall_angle_deg=60.0,
)

# ------------------------------------------------------------ ROBUSTNESS ---
# DATASHEET 2: spreads until measured. Seeds, chaos penalty and robot_seed are
# config.py's. The validation set 3000-3099 is untouched for this robot (its robots
# are drawn from these spreads, not config.py's), so it is this series' frozen set.
ROBUSTNESS = replace(
    RobustnessSettings(),
    randomize=(
        ("body_mass", 0.01),           # measured (gram scale; DATASHEET 2.1)
        ("body_com_height", 0.15),
        ("body_inertia", 0.30),
        ("wheel_inertia", 0.20),
        ("rotor_inertia", 0.50),       # Pybricks derived it from an estimated acceleration
        ("torque_constant", 0.15),     # covers LEGO's 0.355 (-14%)
        ("resistance", 0.17),          # covers 3.73-5.14 ohm
        ("supply_voltage", 0.105),     # 6.8-8.4 V
        ("gear_efficiency", 0.10),     # covers LEGO's 0.50 (-11%) nearly
    ),
    randomize_range=(
        ("wheel_friction", 0.0, 0.001),       # N m s/rad
        ("coulomb_friction", 0.03, 0.08),     # N m, both motors
    ),
    latency_range_s=(0.005, 0.025),           # 2.5 ms is not reachable on this hardware
)

# ------------------------------------------- CONTROLLER, SCENARIO, COST ---
# SOFTWARE_PLAN A3. Every number below is printed, with its derivation, by
#     python hardware/legobot/legobot_scenario.py          (--check re-derives config.py's)
# which repeats config.py's ScenarioConfig/CostConfig method on this robot. "Corner" means
# every corner of ROBUSTNESS above (2048 robots); latency 25 ms, the top of its range.
#
# Lean cap and top speed. config.py fixed the lean cap at 25 deg and took the top speed as
# the highest at which every randomized motor still has the torque that leaning at the cap
# needs. On this robot that speed is 0.017 m/s at 25 deg: the output-side rotor inertia
# (2.05 kg of equivalent mass at r = 34.5 mm) doubles the torque per unit acceleration and
# the motors are slow. So the cap and the speed are chosen together: the cap that maximizes
# top speed x lean-cap acceleration on the limiting corner (the weakest robot's peak power),
# 13 deg (0.227 rad: 1.50 m/s^2, 0.136 N m nominal), and its top speed, 0.263 m/s -> 0.26.
# (12 deg before the body mass was measured: 0.55 -> 0.44 kg, 2026-11-22.)
#
# Baselines (A4). The model-based design (pid_design) keeps config.py's design choices; on
# this robot it gives (3.64, 3.46, 0.38, 0.36), stable for latencies 0-80 ms. config.py's
# hand-tuned gains are linearly unstable here at 10 ms, so the LEGO robot has its own,
# picked by hand in 11 trials (hardware/legobot/A4_BASELINES.md): config.py's scaled by the
# ratio of critical pitch gains (2.1), then kp down and kd up until the 20 s scenario stopped
# oscillating, kv_p up for the slower goal, ki up; stable for latencies 0-67 ms.
CONTROLLER = replace(
    ControllerConfig(),
    max_lean_deg=13.0,
    hand_tuned_gains=(6.0, 2.0, 0.5, 0.3),     # kp, ki, kd, kv_p
)

# The Pololu scenario's proportions and start times, sized from this robot with its
# voltage-mode driver (the robot code's constants nominal, the true ones at the corner)
# and the measured 0.44 kg body:
#   t = 0-2 s        start tilt, goal 0.
#   t = 2-2.75 s     0 -> 0.26 m/s in 0.75 s: peak 0.76 m/s^2, a 0.117 rad lean (Pololu's
#                    0.25 / 0.30 of pitch_limit_rad); 51% of the lean-cap acceleration.
#   t = 2.75-9.5 s   cruise at 0.26 m/s, 41% of the nominal no-load speed (Pololu 53%). Every
#                    corner still has, after friction, its own lean-cap torque here.
#   t = 5.5, 7.5 s   pushes, forward then backward: 0.85 N for 50 ms = 0.0425 N s, 98% of the
#                    largest forward push a full-torque response catches at 0.26 m/s on every
#                    corner (0.0434 N s: light, tall, high rotor inertia, weakest motor at high
#                    speed, most friction; 0.129 nominal at 10 ms). 50 ms is under the 136 ms
#                    fastest unstable time constant (153 ms nominal). Backward: 0.17 catchable,
#                    easy (braking is helped by back-EMF), as on the Pololu robot.
#   t = 9.5-10.9 s   0.26 -> -0.10 m/s in 1.4 s: 0.56 m/s^2, 0.086 rad (Pololu's share).
#   t = 10.9-14 s    cruise backward at -0.10 m/s (Pololu's 0.5/1.3 of top speed); 2.9 rad/s
#                    at the wheels, so Coulomb friction has fully flipped.
#   t = 14-15.2 s    -0.10 -> 0 in 1.2 s: 0.18 m/s^2, 0.028 rad.
#   t = 15.2-20 s    standing still.
# Start tilts 0.0233, -0.07, 0.14 rad: Pololu's shares (1/6, -1/2, 1) of pitch_limit_rad,
# whose 0.14 is 77% of the 0.184 rad a full-torque response catches on every corner (heavy,
# tall, high rotor inertia, weakest motor; 0.344 nominal at 10 ms), as 0.30 was of 0.39.
SCENARIO = replace(
    ScenarioConfig(),
    goal_steps=(
        (2.0, 0.26, 0.75),    # aggressive: 0.76 m/s^2, 0.117 rad of lean
        (9.5, -0.10, 1.4),    # brake and reverse: 0.56 m/s^2
        (14.0, 0.0, 1.2),     # gentle stop: 0.18 m/s^2
    ),
    pushes=(
        (5.5, 0.05, 0.85),    # from behind at 0.26 m/s
        (7.5, 0.05, -0.85),   # from the front at 0.26 m/s
    ),
    start_pitches_deg=(1.335, -4.0107, 8.0214),    # 0.0233, -0.07, 0.14 rad
)

# Cost limits (CostConfig's Bryson rule), in the Pololu cost's proportions:
#   pitch_limit_rad 0.14: 77% of the catchable start tilt; between the 0.117 rad the goal
#     profile needs and the 0.227 rad lean cap (62% of it; Pololu 69%).
#   velocity_limit 0.04 m/s: Pololu's 0.2/1.3 of the top speed.
#   torque_limit 0.10 N m: Pololu's 0.1/0.135 of the nominal lean-cap torque (0.136), so
#     pitch and effort trade as they do there; 20% of max_torque. Holding the cruise
#     against the friction the robustness test draws takes 0.03-0.09 N m.
# Weights: velocity (0.14/0.04)^2 = 12.25, effort (0.14/0.10)^2 = 1.96. Fall penalty: the
# worst non-fall 20 s run costs 21.9 (pitch) + 9.8 (effort) + 290.7 (velocity, fastest
# corner's 0.829 m/s no-load speed + 0.26) = 322, under 1000 (PHYSICS.md 8.3).
COST = replace(
    CostConfig(),
    pitch_limit_rad=0.14,
    velocity_limit=0.04,
    torque_limit=0.10,
    fall_penalty=1000.0,
)

# --------------------------------------------------------- PUT IT TOGETHER -
LEGOBOT_CONFIG = ProjectConfig(
    name="legobot",
    robot=ROBOT,
    actuator=ACTUATOR,
    sensor=SENSOR,
    timing=TIMING,
    controller=CONTROLLER,
    cost=COST,
    scenario=SCENARIO,
    robustness=ROBUSTNESS,
)
