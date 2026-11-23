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

The motor, body and sensor values are measured (SOFTWARE_PLAN C3, C4; DATASHEET 2 and 8.7), and the
robustness spreads are shrunk to their measurement uncertainty.
"""

from dataclasses import replace

from config import (
    ActuatorConfig,
    ControllerConfig,
    CostConfig,
    SearchConfig,
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
    body_mass=0.362,               # kg, measured 2026-11-22: 440 g total minus two 39 g wheels (DATASHEET 1, 2.1)
    body_com_height=0.060,         # m, measured (balance test; the robot's 49 mm scaled by 440/362)
    body_inertia=8.8e-4,           # kg m^2 about the body's own COM, measured (compound pendulum)
    wheel_mass=0.078,              # kg, both wheels, measured: 39 g each with axle
    wheel_radius=0.0345,           # m, half the 69 mm diameter
    wheel_inertia=6.0e-5,          # kg m^2, both wheels, 0.65 m r^2 each (tire at the rim)
    max_torque=0.90,               # N m total: 2 x 0.45 N m stall, measured (lever on a scale, a x 7.6 V; DATASHEET 8.7).
                                   # LEGO's 0.25 N m per motor is a rating, not the motor's limit
    wheel_friction=0.0017,         # N m s/rad, both motors, measured: 7e-4 per motor from the slow-PWM speeds,
                                   # 9.5e-4 from coast-downs (DATASHEET 8.7)
    coulomb_friction=0.042,        # N m, both motors, measured: 0.022 / 0.027 (left / right) from the speeds,
                                   # 0.016 / 0.019 from coast-downs
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
# Measured 2026-11-23 (SOFTWARE_PLAN C3, DATASHEET 8.7) with the Build HAT in slow PWM (pwmparams 1 0.01).
# That mode behaves as torque = |d| a (sgn(d) V - k w), a = eta k / R: linear in duty at zero speed (lever
# on a scale, within 4%) with back-EMF damping proportional to duty (spin-up tau = J / (d a k + b) at every
# duty). Model "slow_pwm" (balance_sim.SlowPwmActuator, SOFTWARE_PLAN B1b) is that mode; the robot code
# sends the duty from its inverse. It reproduces the bench's steady no-load speeds within 1% at duty >= 0.2
# (the right wheel's extra friction shows below that) and the spin-up curves (DATASHEET 8.7). Slow PWM is
# pulse-density modulation: 10 ms full-voltage pulses at about 100 x duty per second, with a running sum
# (pulse_imu: one pulse per 199 ms at duty 0.05), modelled by pwm_tick_s. The HAT can't measure current, so only a = eta k / R is
# identified, not R and eta separately: R stays at 4.4 ohm (the current limit never binds) and eta carries a.
ACTUATOR = ActuatorConfig(
    model="slow_pwm",
    pwm_tick_s=0.010,              # s: the HAT's pulse-density tick (DATASHEET 8.7); a whole number of dt steps
    pwm_tick_locked=True,          # the robot paces its loop on the HAT's line grid and won't arm or run without
                                   # the lock (robot/balance.py), so each tick falls where a command arrives
                                   # and the latency below is the whole sample-to-pulse delay (review G6)
    voltage_feedforward=True,
    gear_ratio=1.0,                # lumped: constants below are output-referred
    torque_constant=0.348,         # V s/rad, output k_e, measured: 0.352 / 0.344 left / right (LEGO's sheet 0.355;
                                   # Pybricks' 0.411 is 18% high)
    resistance=4.4,                # ohm incl. driver and wiring: not separable from eta without current sensing
    supply_voltage=7.6,            # V: 2S LiPo (6.8-8.4) x plimit 1
    current_limit=2.5,             # A: no current control on the HAT; set never to bind (stall <= 2.25 A)
    n_motors=2,
    gear_efficiency=0.755,         # from a = eta k / R = 0.0597 +- 0.0019 N m/V measured: 0.0597 x 4.4 / 0.348
    rotor_inertia=5.45e-4,         # kg m^2 per motor, output side, measured: spin-up J (motor + wheel) 5.8 / 5.7e-4
                                   # left / right, less one wheel's 3e-5. Pybricks' estimate was 1.22e-3
)

# ---------------------------------------------------------------- SENSOR ---
# DATASHEET 2.3 and 3.4. The ICM-20948's gyro and accelerometer are simulated and fused by
# the robot code's complementary filter (model "imu", SOFTWARE_PLAN B2), so the pitch the
# controller sees carries the accelerometer's acceleration error (low-passed) and the gyro
# bias turns into an offset of bias x tau. In this model pitch_std and pitch_bias are the
# accelerometer tilt's. The noise is the chassis's with the motors running (measured below),
# about 10x the datasheet's: LEGO frames rattle. Velocity comes from the motors' own
# encoders as the Build HAT reports them (SOFTWARE_PLAN B3): 360 counts per turn, sampled at
# 100 Hz, about 5 ms in transit (two 2 ms data lines; DATASHEET 6), held between samples, and
# differenced 10 ms apart: 0.025 m/s of quantization noise at r = 34.5 mm, averaged over
# speeds (DATASHEET 5.4). Differencing 5 ms apart raised every baseline's cost (B3).
# Measured 2026-11-23 (SOFTWARE_PLAN C3, robot/measure/imu_still.py; DATASHEET 8.7), robot on a stand,
# wheels spinning free: with motors off the gyro read 1.5e-3 rad/s and the accelerometer tilt 1.45e-3 rad
# (datasheet level). Motor vibration dominates: gyro 0.014 / 0.015 / 0.017 rad/s and tilt 0.016 / 0.023 /
# 0.029 rad at duty 0.3 / 0.6 / 1.0. The values below are the duty-0.3 phase (6-9 rad/s at the wheel, the
# speeds balancing and the 0.27 m/s cruise use) for the tilt, and the worst phase for the gyro.
SENSOR = SensorConfig(
    model="imu",
    pitch_std=0.016,               # rad, accelerometer tilt per sample (raw, before the filter), measured
    pitch_bias=0.015,              # rad, accelerometer offset after calibration, and LEGO mount play
    pitch_rate_std=0.017,          # rad/s, gyro, measured with the motors running
    pitch_rate_bias=0.003,         # rad/s, gyro, after calibrating at boot: drift measured under 2e-4 rad/s
                                   # (Allan 1e-4 at 5 s; no shift with the motors on); 0.003 covers the
                                   # datasheet's 0.05 deg/s/degC over a 3 degC warm-up
    position_std=0.0010,           # m: unused with the encoder (kept for the noise model)
    velocity_std=0.025,            # m/s: unused with the encoder (kept for the noise model)
    encoder_counts=360,
    encoder_period_s=0.010,
    encoder_delay_s=0.005,
    imu_height=0.10,               # m above the axle, measured (11 mm forward of the axle: negligible, DATASHEET 1)
    filter_time_constant=0.5,      # s: 1 s raised the baselines' cost 9-24% more, 0.25 s was mixed (B2)
)

# ---------------------------------------------------------------- TIMING ---
# DATASHEET 6 and the locked loop (SOFTWARE_PLAN D2): from the IMU sample to the motor's pulse, the IMU
# filter (~3 ms, DLPF cfg 3, est.), read + compute until the write starts (1.3 ms mean, 3.6 ms worst,
# measured), the write to the HAT's tick (1.8 ms left, 3.8 ms right, measured by lock_sweep) and the
# current rise (L/R ~3 ms, est.): about 9 ms (left) to 11 ms (right), 15 ms at worst. 10 ms nominal.
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
        ("body_com_height", 0.08),     # measured, +-5 mm
        ("body_inertia", 0.15),        # measured; mostly the COM height's +-5 mm
        ("wheel_inertia", 0.20),
        # Motor spreads measured 2026-11-23 (DATASHEET 8.7): each covers the left-right difference and the
        # fit's uncertainty, with margin. resistance and gear_efficiency together set a = eta k / R, measured
        # to 3% on one motor; the other motor and the robot code's constants get the rest.
        ("rotor_inertia", 0.08),       # spin-up fit 2%, and a's 3% (J = tau a k)
        ("torque_constant", 0.03),     # left / right 0.352 / 0.344 (+-1.2%), fit 0.4%
        ("resistance", 0.05),          # with gear_efficiency: a measured on the left motor only
        ("supply_voltage", 0.105),     # 6.8-8.4 V (discharge log still to do)
        ("gear_efficiency", 0.02),
    ),
    randomize_range=(
        ("wheel_friction", 0.0012, 0.0022),   # N m s/rad, both motors: speeds 0.0014, coast-downs 0.0019
        ("coulomb_friction", 0.030, 0.055),   # N m, both motors: coast-downs 0.035, speeds 0.049
        # SOFTWARE_PLAN B6: the pitch offset (the accelerometer's, in the IMU model) drawn per
        # scenario in +-b, b = SENSOR.pitch_bias, the residual after balance-point calibration
        # (DATASHEET 8.4), so the tuner sees both signs and every size. It also covers a fore-aft
        # COM offset e: a pitch offset of atan(e / l), 0.011 rad per mm at l = 90 mm. Last in the
        # list, so the draws above are unchanged. Plain (non-robust) runs keep +b.
        ("pitch_bias", -0.015, 0.015),        # rad
        # The pulse ticks' phase is no longer drawn (it was uniform in a tick, B1b): the robot loop is
        # locked to the ticks (ACTUATOR.pwm_tick_locked), so each scenario's phase follows its latency.
    ),
    # The locked sample-to-pulse budget (TIMING): 9-11 ms typical, 15 ms worst; drawn in whole 2.5 ms steps,
    # so 7.5, 10, 12.5 or 15 ms. Was 5-25 ms with the tick phase drawn on top, up to 35 ms in all, against
    # 2-4 ms of tick wait on the locked robot (review G6). Final gains still get a wider latency sweep as
    # an evaluation, not as a training target.
    latency_range_s=(0.0075, 0.015),
)

# ------------------------------------------- CONTROLLER, SCENARIO, COST ---
# SOFTWARE_PLAN A3, re-derived 2026-11-23 for the bench-measured motor (C3/C4). Every number
# below is printed, with its derivation, by
#     python hardware/legobot/legobot_scenario.py          (--check re-derives config.py's)
# which repeats config.py's ScenarioConfig/CostConfig method on this robot. "Corner" means
# every corner of ROBUSTNESS above. The response starts after the top of the latency range (15 ms)
# for a start tilt, and a tick later (25 ms) for a push, which can come just after a sample whose
# command meets a tick (legobot_scenario.catch_latencies; review G4). Catch tests use the averaged
# slow-PWM actuator at full torque, which is what full-duty pulses deliver.
#
# Lean cap and top speed. config.py fixed the lean cap at 25 deg and took the top speed as the
# highest at which every randomized motor still has the torque that leaning at the cap needs.
# With the estimated motor (2.05 kg of equivalent rotor mass) that left 0.017 m/s, so the cap was
# chosen to maximize top speed x lean-cap acceleration (15 deg, 0.27 m/s). The measured motor
# (0.92 kg equivalent, 1.8x the stall torque) no longer needs that: v x a still rises at 25 deg,
# so the cap is config.py's 25 deg (0.436 rad: 2.46 m/s^2, 0.119 N m nominal), and the rule gives
# 0.474 m/s. That made every baseline fall: to accelerate, a balancing robot first drives its
# wheels well past the goal speed (+66-79% for the model-based PID), and this voltage-limited
# motor loses torque linearly with speed, so a rule with no headroom fails. Decided 2026-11-23
# (the user): cruise 0.40 m/s (Pololu's 53% of the no-load speed), and speed changes softened to
# half of Pololu's lean shares, the least softening at which the model-based PID has no falls on
# the robust training set (legobot_scenario.TOP_SPEED_CHOICE, LEAN_SOFTENING). The cost-matched
# LQR is dropped for this robot (A4_BASELINES.md).
#
# Baselines (A4): see hardware/legobot/A4_BASELINES.md. The model-based design (pid_design)
# re-designs itself from the robot; the hand-tuned gains were picked again for the measured
# robot and this scenario (13 trials, round numbers, no search).
CONTROLLER = replace(
    ControllerConfig(),
    max_lean_deg=25.0,
    hand_tuned_gains=(2.5, 1.5, 0.2, 0.5),     # kp, ki, kd, kv_p: picked by hand in 13 trials (A4_BASELINES.md)
)

# The Pololu scenario's start times, sized from this robot (slow-PWM driver, the robot code's
# constants nominal and the true ones at the corner):
#   t = 0-2 s        start tilt, goal 0.
#   t = 2-3.25 s     0 -> 0.40 m/s in 1.25 s: peak 0.70 m/s^2, a 0.125 rad lean (half of
#                    Pololu's 0.25 / 0.30 of pitch_limit_rad).
#   t = 3.25-9.5 s   cruise at 0.40 m/s, 53% of the nominal no-load speed (Pololu 53%).
#   t = 5.5, 7.5 s   pushes, forward then backward: 1.02 N for 50 ms = 0.051 N s, 98% of the
#                    largest forward push a full-torque response catches at 0.40 m/s on every
#                    corner (0.0523 N s: light, tall, high rotor inertia, weakest motor, most
#                    friction; 0.108 nominal at 10 ms). 50 ms is under the 119 ms fastest unstable
#                    time constant (123 ms nominal). Backward: 0.325 catchable, easy (braking is
#                    helped by back-EMF), as on the Pololu robot.
#   t = 9.5-11.9 s   0.40 -> -0.15 m/s in 2.4 s: 0.50 m/s^2, 0.092 rad.
#   t = 11.9-14 s    cruise backward at -0.15 m/s (Pololu's 0.5/1.3 of top speed).
#   t = 14-16 s      -0.15 -> 0 in 2.0 s: 0.16 m/s^2, 0.030 rad.
#   t = 16-20 s      standing still.
# Start tilts 0.05, -0.15, 0.30 rad: Pololu's shares (1/6, -1/2, 1) of pitch_limit_rad.
SCENARIO = replace(
    ScenarioConfig(),
    goal_steps=(
        (2.0, 0.40, 1.25),    # aggressive (softened): 0.70 m/s^2, 0.125 rad of lean
        (9.5, -0.15, 2.4),    # brake and reverse: 0.50 m/s^2
        (14.0, 0.0, 2.0),     # gentle stop: 0.16 m/s^2
    ),
    pushes=(
        (5.5, 0.05, 1.02),    # from behind at 0.40 m/s
        (7.5, 0.05, -1.02),   # from the front at 0.40 m/s
    ),
    start_pitches_deg=(2.8648, -8.5944, 17.1887),    # 0.05, -0.15, 0.30 rad
)

# Cost limits (CostConfig's Bryson rule), in the Pololu cost's proportions:
#   pitch_limit_rad 0.30: Pololu's is both 77% of its catchable start tilt and 69% of its lean
#     cap. Here those give 0.35 (of the 0.454 rad caught on every corner; 0.628 nominal at 10 ms)
#     and 0.30; the smaller keeps the goal profile's leans (shares of it) under the cap.
#   velocity_limit 0.06 m/s: Pololu's 0.2/1.3 of the top speed.
#   torque_limit 0.09 N m: Pololu's 0.1/0.135 of the nominal lean-cap torque (0.119), so pitch
#     and effort trade as they do there; 10% of max_torque.
# Weights: velocity (0.30/0.06)^2 = 25.0, effort (0.30/0.09)^2 = 11.1. Fall penalty: the worst
# non-fall 20 s run costs 21.9 (pitch) + 180.0 (effort) + 791.7 (velocity, fastest corner's
# 0.858 m/s no-load speed + 0.40) = 994, under 1000 (PHYSICS.md 8.3).
COST = replace(
    CostConfig(),
    pitch_limit_rad=0.30,
    velocity_limit=0.06,
    torque_limit=0.09,
    fall_penalty=1000.0,
)

# ----------------------------------------------------------- SEARCH SPACE -
# config.py's bounds, except kv_p. Its 0.9 was set from Pololu runs (a 2.0 bound settled at 0.82, 2.6x that
# robot's model-based kv_p). kv_p is lean per unit speed error, so a robot whose speeds are a third of
# Pololu's needs proportionally more; here the model-based design already gives 0.52 and the hand-tuned
# 0.5. 2.0 leaves the tuner room above both (review finding G1, 2026-11-23). The kp bound is searched on a
# log scale, so its Pololu-sized top costs little.
SEARCH = replace(SearchConfig(), kv_p_max=2.0)

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
    search=SEARCH,
)
