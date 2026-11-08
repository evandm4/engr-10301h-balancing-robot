"""
=============================================================================
 PROJECT CONFIGURATION
=============================================================================
The one file to edit to change the robot, the actuator model, sensor noise,
delays, cost weights, the tuner's test scenario, its search ranges, or the
robustness test the batched tuner applies.

Every script (the demos in simulation/examples/, the reference tuner
optimization/run_tuning.py, the batched tuner optimization/run_batch_tuning.py,
and both benchmarks) builds its objects from CONFIG at the bottom of this
file, so a change here reaches everywhere it's used. If you find a physical
constant, a delay, a cost weight, or a search bound hardcoded somewhere else,
it should probably move here instead. The classes these settings build
(RobotParams, MotorParams, EvalConfig, SimSpec, ...) deliberately have no
defaults for them, so this file is the only place each value is written.

Every tuning run saves a full snapshot of this configuration (see
ProjectConfig.snapshot) next to its results.

Angles are given in degrees below for readability; each section exposes a
*_rad property that does the conversion, so the rest of the code still works
in radians throughout.
"""

import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# Make simulation/ (balance_sim, batch_sim) and optimization/ (tuning,
# batch_tuning) importable no matter which script imports this file.
_ROOT = Path(__file__).resolve().parent
for _sub in ("simulation", "optimization"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from balance_sim import (
    PITCH_REFERENCES,
    CascadedPIDController,
    DCMotorActuator,
    IdealActuator,
    LQRController,
    LQRWeights,
    MotorParams,
    PIDController,
    Planar2D,
    RobotParams,
    SensorNoise,
    cost,
    critical_pitch_gain,
    lqr_gain,
    step_profile,
)


# ----------------------------------------------------------------- ROBOT ---
@dataclass(frozen=True)
class RobotConfig:
    """The physical robot the simulation models. These numbers double as the
    hardware build spec if the physical robot gets made. DATASHEET.md gives the
    source of each value (a reference build: Pololu 25D gearmotors, 70 mm
    Pololu wheels, a 2S LiPo) and the realistic range around it."""

    body_mass: float = 0.80            # kg, chassis + electronics + motors, excludes wheels (0.5-0.9 realistic)
    body_com_height: float = 0.10      # m, body center of mass above the wheel axle (set by where the battery goes)
    body_inertia: float = 0.0065       # kg m^2, body pitch inertia about its own COM. A tower tall enough
                                       # for a 0.10 m COM has about 0.006-0.007 (DATASHEET.md 4; was 0.0033)
    wheel_mass: float = 0.034          # kg, both wheels combined: 2 x (14 g Pololu 70x8 mm wheel + 3.2 g hub)
    wheel_radius: float = 0.035        # m
    wheel_inertia: float = 2.2e-5      # kg m^2, both wheels about the axle (0.65 m r^2 each: tire at the rim).
                                       # The motor rotors are separate: ActuatorConfig.rotor_inertia
    max_torque: float = 0.34           # N m, total torque limit (both motors). Applies to both actuator
                                       # models; the DC motor can limit it further. 0.34 is what the motors
                                       # give at their 2 A limit after gearbox losses (DATASHEET.md 5.1)
    # Friction between wheels and body (brushes, bearings, gear mesh), as a torque against the
    # relative wheel speed w: wheel_friction * w + coulomb_friction * tanh(w / coulomb_friction_speed).
    # Unmeasured, so zero here; RobustnessSettings.randomize_range draws both per scenario, over
    # ranges set from the motors' free-run current (DATASHEET.md 5.2). Measure them as in PHYSICS.md 9.4.
    wheel_friction: float = 0.0        # N m s/rad, viscous
    coulomb_friction: float = 0.0      # N m, Coulomb (dry), both wheels together
    coulomb_friction_speed: float = 1.0   # rad/s, smoothing of the Coulomb step (PHYSICS.md 9.4)
    gravity: float = 9.81              # m/s^2 (9.8015 at West Lafayette; the difference is 0.09%)

    def build(self, rotor_inertia: float) -> RobotParams:
        """rotor_inertia: the motor rotors reflected to the axle (ActuatorConfig.reflected_rotor_inertia);
        ProjectConfig.build_robot_params passes it."""
        return RobotParams(
            body_mass=self.body_mass,
            body_com_height=self.body_com_height,
            body_inertia=self.body_inertia,
            wheel_mass=self.wheel_mass,
            wheel_radius=self.wheel_radius,
            wheel_inertia=self.wheel_inertia,
            rotor_inertia=rotor_inertia,
            max_torque=self.max_torque,
            wheel_friction=self.wheel_friction,
            coulomb_friction=self.coulomb_friction,
            coulomb_friction_speed=self.coulomb_friction_speed,
            gravity=self.gravity,
        )


# -------------------------------------------------------------- ACTUATOR ---
@dataclass(frozen=True)
class ActuatorConfig:
    """Which motor model the dynamics use, and that motor's own parameters.

    model: "ideal" delivers the commanded torque exactly (clipped to
    RobotConfig.max_torque). "dc_motor" adds back-EMF, a supply voltage, and a
    current limit, so available torque falls as the wheels spin faster.
    """

    model: str = "dc_motor"            # "ideal" or "dc_motor"

    # Pololu 20.4:1 Metal Gearmotor 25Dx65L mm HP 6V with 48 CPR encoder (item 4803), run
    # from a 2S LiPo. Constants derived from Pololu's catalogue numbers (DATASHEET.md 3.1, 5.1).
    # Used by the DC motor model only, except gear_ratio, n_motors and rotor_inertia, which
    # also set the rotor inertia the dynamics see with either model.
    gear_ratio: float = 20.4           # motor turns per output turn
    torque_constant: float = 0.0053    # N m/A at the motor shaft (= back-EMF constant in SI)
    resistance: float = 1.25           # ohm: 1.0 winding (6 V / 6.0 A stall) + about 0.25 driver, wiring, battery
    supply_voltage: float = 7.6        # V: 2S LiPo, 8.4 full to 6.8 at cutoff under load
    current_limit: float = 2.0         # A, per motor (DRV8874: 2.1 A continuous; Pololu: <= 25% of 6 A stall)
    n_motors: int = 2
    gear_efficiency: float = 0.78      # geared stall torque / (20.4 x bare-motor stall torque)
    rotor_inertia: float = 5.0e-7      # kg m^2 per motor, at the motor shaft. Unpublished: estimated from
                                       # a 370-size armature (3-8e-7, DATASHEET.md 5.3); measure by spin-down

    @property
    def reflected_rotor_inertia(self) -> float:
        """All rotors seen at the wheel axle, n N^2 J_r (kg m^2): RobotParams.rotor_inertia."""
        return self.n_motors * self.gear_ratio**2 * self.rotor_inertia

    def build_motor_params(self) -> MotorParams:
        return MotorParams(
            gear_ratio=self.gear_ratio,
            torque_constant=self.torque_constant,
            resistance=self.resistance,
            supply_voltage=self.supply_voltage,
            current_limit=self.current_limit,
            n_motors=self.n_motors,
            gear_efficiency=self.gear_efficiency,
        )

    def build(self, robot: RobotParams):
        if self.model == "ideal":
            return IdealActuator(robot.max_torque)
        if self.model == "dc_motor":
            # The robot's torque limit still applies on top of the motor physics.
            return DCMotorActuator(self.build_motor_params(), max_torque=robot.max_torque)
        raise ValueError(f"Unknown actuator model {self.model!r}; use 'ideal' or 'dc_motor'")


# ---------------------------------------------------------------- SENSOR ---
@dataclass(frozen=True)
class SensorConfig:
    """Measurement noise the controller sees, instead of true state. Set every
    std to 0 for a perfect sensor. The random seed is set per test scenario
    elsewhere (see ScenarioConfig), not here."""

    # DATASHEET.md 2.3: the gyro and pitch noise are 10-30x the IMU datasheets' (ICM-42688-P,
    # MPU-6050), kept conservative for chassis vibration until measured on the robot; the
    # encoder values match a 48 CPR motor encoder (979.62 counts per wheel turn) at 200 Hz.
    pitch_std: float = 0.005           # rad
    pitch_bias: float = 0.01           # rad, constant pitch-estimate offset: IMU mounting and balance-point
                                       # error, typically 0.5-1 deg until calibrated out
    pitch_rate_std: float = 0.02       # rad/s, gyro
    pitch_rate_bias: float = 0.01      # rad/s, constant gyro offset (what is left after calibrating at boot)
    position_std: float = 0.0005       # m (quantization is 0.065 mm; this also stands in for gear backlash)
    velocity_std: float = 0.02         # m/s (differencing encoder counts 5 ms apart gives 0.018)

    def build(self, seed: int = 0) -> SensorNoise:
        return SensorNoise(
            pitch_std=self.pitch_std,
            pitch_bias=self.pitch_bias,
            pitch_rate_std=self.pitch_rate_std,
            pitch_rate_bias=self.pitch_rate_bias,
            position_std=self.position_std,
            velocity_std=self.velocity_std,
            seed=seed,
        )


# ---------------------------------------------------------------- TIMING ---
@dataclass(frozen=True)
class TimingConfig:
    latency_s: float = 0.010           # controller-to-actuator delay (conservative: about 1.5-6 ms
                                       # is expected from the IMU filter, bus and driver; DATASHEET.md 6)
    dt_s: float = 0.0025               # physics step used while tuning (coarser = faster)
    demo_dt_s: float = 0.001           # finer physics step used for one-off demo plots
    control_dt_s: float = 0.005        # controller period (200 Hz)
    fall_angle_deg: float = 60.0       # a run ends once pitch passes this

    @property
    def fall_angle_rad(self) -> float:
        return math.radians(self.fall_angle_deg)


# ------------------------------------------------------------ CONTROLLER ---
@dataclass(frozen=True)
class ControllerConfig:
    """The cascaded PID controller: an outer velocity loop sets a lean angle,
    an inner pitch loop sets torque. The outer loop is proportional only
    (kv_p): its integral gain always tuned to 0, so it was dropped. The
    controller classes still accept a kv_i if hardware turns out to need one."""

    max_lean_deg: float = 25.0         # cap on the outer (velocity) loop's commanded lean angle
    pitch_integral_limit: float = 1.0  # rad s, cap on the pitch loop's integral (anti-windup also applies)
    velocity_integral_limit: float = 0.5   # m, cap on the velocity loop's integral (unused while kv_i = 0),
                                           # and on the LQR's integral of velocity error
    # kp, ki, kd (pitch loop), kv_p (velocity loop): the hand-picked baseline
    # every tuning run is compared against.
    hand_tuned_gains: Tuple[float, float, float, float] = (4.0, 0.5, 0.15, 0.1)

    @property
    def max_lean_rad(self) -> float:
        return math.radians(self.max_lean_deg)


# ------------------------------------------------------------------ COST ---
@dataclass(frozen=True)
class CostConfig:
    """Weights for the cost function the tuner minimizes. Lower cost is
    better; see simulation/balance_sim/metrics.py for how each term is
    computed from a run."""

    # THE WEIGHTS ARE DERIVED FROM THESE THREE LIMITS (Bryson's rule), not set
    # directly. Each limit is the largest value of that quantity the design
    # accepts; its weight is 1 / limit^2, scaled so the pitch term has weight 1
    # (only the ratios between the terms change the optimum):
    #     velocity_weight = (pitch_limit_rad / velocity_limit)^2 = (0.3 / 0.2)^2 = 2.25
    #     effort_weight   = (pitch_limit_rad / torque_limit)^2   = (0.3 / 0.1)^2 = 9.0
    # Keep them consistent with the rest of this file. They are a statement about
    # the task, and the cost only means what they say while that holds:
    #   - pitch_limit_rad is a tolerance on pitch measured from upright. It must
    #     stay above the lean the goal profile needs (~0.12 rad for the profile
    #     below, from v_dot / 6.9 m/s^2) and below ControllerConfig.max_lean_deg
    #     (25 deg = 0.44 rad). Below the first, the task's own lean is penalized
    #     as an error; above the second, the limit says nothing.
    #   - torque_limit is a typical operating torque, deliberately well under
    #     RobotConfig.max_torque (0.34 N m), so effort is a real tradeoff. Set it
    #     near max_torque and the effort term goes quiet again (it was 0.01% of
    #     the cost at weight 0.01).
    #   - velocity_limit is the tracking error accepted on ScenarioConfig's goal.
    #     Change the goal profile and revisit it.
    #   - fall_penalty must exceed any non-fall cost. These weights scale costs up,
    #     so changing a limit can silently break that; re-check it (PHYSICS.md 8.3).
    # Changing one limit changes every score: runs under different limits are not
    # comparable (every run records them in its config snapshot). A limit of
    # math.inf switches that term off. pitch_limit_rad is in radians, not degrees
    # like the rest of this file, so the weights come out exact.
    pitch_limit_rad: float = 0.3
    velocity_limit: float = 0.2        # m/s
    torque_limit: float = 0.1          # N m
    fall_penalty: float = 500.0        # added to the cost if the robot falls
    # What pitch error is measured against. "upright": vertical, which penalizes
    # every lean including the ones the task requires, and is the only reference
    # every controller has (the LQR commands no lean, so it is scored from
    # upright either way). "commanded": the lean the cascaded PID's velocity loop
    # asked for, so only failing to hold it is penalized.
    pitch_reference: str = "upright"

    def __post_init__(self):
        if self.pitch_reference not in PITCH_REFERENCES:
            raise ValueError(f"pitch_reference must be one of {PITCH_REFERENCES}, got {self.pitch_reference!r}")
        for name in ("pitch_limit_rad", "velocity_limit", "torque_limit"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")

    @property
    def velocity_weight(self) -> float:
        """Penalty per unit of integrated velocity error^2, relative to the pitch term."""
        return self.pitch_limit_rad**2 / self.velocity_limit**2

    @property
    def effort_weight(self) -> float:
        """Penalty per unit of integrated torque^2, relative to the pitch term."""
        return self.pitch_limit_rad**2 / self.torque_limit**2


# -------------------------------------------------------------- SCENARIO ---
@dataclass(frozen=True)
class ScenarioConfig:
    """What the tuner's evaluator asks the robot to do, and the range of
    conditions it's scored under."""

    duration_s: float = 12.0
    goal_steps: Tuple[Tuple[float, float], ...] = ((2.0, 1.0), (3.5, 0), (6.0, -0.6), (9.0, 0.2))  # (start time s, goal m/s)
    # Seconds each goal change takes to blend smoothly into the new velocity
    # (see balance_sim.step_profile). 0 = instant jumps, which no robot can
    # follow, so the velocity term can never reach zero. With a smooth blend
    # the goal is physically trackable: it peaks at about 0.9 m/s^2 (the 0 -> 1
    # m/s change is cut off at 3.5 s, so the goal tops out at 0.71 m/s), about
    # 7 deg of lean, well within the 25 deg lean cap.
    goal_transition_s: float = 2.5
    start_pitches_deg: Tuple[float, ...] = (5.73, -4.58, 2.86)                 # initial disturbance, each tried with every seed
    train_seeds: Tuple[int, ...] = (1, 2, 3)                                   # sensor-noise seeds scored during tuning
    validation_seeds: Tuple[int, ...] = tuple(range(100, 110))                 # unseen seeds, checked after tuning
    # Runs per score = len(start_pitches_deg) x number of seeds: 9 per training
    # evaluation and 30 per validation with the defaults above.

    def __post_init__(self):
        overlap = set(self.train_seeds) & set(self.validation_seeds)
        if overlap:
            raise ValueError(f"train_seeds and validation_seeds overlap: {sorted(overlap)}")

    @property
    def start_pitches_rad(self) -> Tuple[float, ...]:
        return tuple(math.radians(d) for d in self.start_pitches_deg)


# ----------------------------------------------------------- SEARCH SPACE --
@dataclass(frozen=True)
class SearchConfig:
    """Bounds the tuner searches within for each gain. kp_min_factor is a
    multiple of the physics-derived minimum Kp that can hold the robot up at
    all (see balance_sim.critical_pitch_gain), not a raw value. That minimum is
    taken for the most demanding robot robust scoring can draw (every
    RobustnessSettings.randomize spread at its worst end; ProjectConfig.
    kp_floor_robot), so no candidate is below it for any scenario."""

    kp_min_factor: float = 1.1
    kp_max: float = 30.0
    ki_max: float = 5.0
    kd_max: float = 1.0
    kv_p_max: float = 0.9              # was 0.4, which every tuned result sat at; a run with 2.0 settled at 0.82


# ------------------------------------------------------------------- LQR ---
@dataclass(frozen=True)
class LQRConfig:
    """The LQR controller (simulation/balance_sim/lqr.py; LQR.md): full-state
    feedback designed on the linearized nominal robot, for the configured
    control period and latency. A design minimizes, per second,

        w_pitch pitch^2 + w_velocity (velocity error)^2 + w_pitch_rate (pitch rate)^2
        + w_integral (integral of velocity error)^2 + torque^2

    with pitch from upright. The "cost-matched" design takes its weights from
    CostConfig (pitch 1, velocity_weight, torque effort_weight, no rate or
    integral term; the same as w_pitch = 1 / effort_weight and so on): on the
    linear model it is the optimum of the cost the tuner scores. The ML-tuned design searches the weights within the bounds
    below, on a log scale, each per unit of torque weight (only ratios matter,
    so torque stays at 1). The integral is capped at
    ControllerConfig.velocity_integral_limit.

    The "direct" search skips the weights and tunes the gains themselves:
    each of the cost-matched design's gains scaled by a factor in
    [1 / direct_gain_scale, direct_gain_scale], keeping the nominal linear
    loop stable (tuning.LQRGainScaling).
    """

    # Weights per unit of torque weight. Normalized by torque since 2026-11-06: with
    # pitch fixed at 1 the search ended on w_velocity's floor and w_torque's ceiling,
    # which is the same design as pitch and velocity weights of 0 (LQR.md).
    w_pitch_bounds: Tuple[float, float] = (1e-6, 1e2)
    w_velocity_bounds: Tuple[float, float] = (1e-6, 1e2)
    w_pitch_rate_bounds: Tuple[float, float] = (1e-8, 1.0)
    w_integral_bounds: Tuple[float, float] = (1e-3, 1e3)
    direct_gain_scale: float = 100.0
    # Parameters held at "off" instead of searched; None searches them. With them searched,
    # the pitch, velocity and pitch-rate weights all went to the floor of their log range,
    # which a log scale can only approach, and the score was flat there to 1e-5 (LQR.md;
    # likewise the command-feedback scales of the direct search, 0.01-0.05).
    w_pitch_fixed: Optional[float] = 0.0            # 0 = no pitch term (pitch is still stabilized)
    w_velocity_fixed: Optional[float] = 0.0         # 0 = velocity enters only through its integral
    w_pitch_rate_fixed: Optional[float] = 0.0       # 0 = no rate term
    direct_commands_scale: Optional[float] = 0.0    # multiple of the design's gains on the commands in flight; 0 = none

    @property
    def fixed_weights(self) -> dict:
        """Weights the weight search holds fixed (tuning.lqr_weight_space's `fixed`)."""
        held = {"w_pitch": self.w_pitch_fixed, "w_velocity": self.w_velocity_fixed,
                "w_pitch_rate": self.w_pitch_rate_fixed}
        return {name: value for name, value in held.items() if value is not None}

    @property
    def bounds(self) -> dict:
        return {"w_pitch": self.w_pitch_bounds, "w_velocity": self.w_velocity_bounds,
                "w_pitch_rate": self.w_pitch_rate_bounds, "w_integral": self.w_integral_bounds}


# ------------------------------------------------------------ ROBUSTNESS ---
@dataclass(frozen=True)
class RobustnessSettings:
    """The harsher test the batched tuner (optimization/run_batch_tuning.py)
    scores candidates on by default; see optimization/batch_tuning/robust.py.
    The reference tuner (run_tuning.py) doesn't use this section.

    Start tilts come from ScenarioConfig; each is tried with every seed here.
    """

    n_train_seeds: int = 50            # noise seeds 1..N scored during tuning
    validation_seeds: Tuple[int, ...] = tuple(range(1000, 1100))   # unseen seeds (and robots)
    # (parameter, relative spread): each scenario's robot has that parameter
    # scaled by a random factor in [1 - spread, 1 + spread]. Build and datasheet
    # tolerances (DATASHEET.md 2); replace with measured spreads. Names are
    # batch_sim.BatchRobot fields: RobotParams and MotorParams, so "rotor_inertia"
    # is the reflected total (RobotParams.rotor_inertia).
    randomize: Tuple[Tuple[str, float], ...] = (
        ("body_mass", 0.15),
        ("body_com_height", 0.15),
        ("body_inertia", 0.25),
        ("wheel_inertia", 0.20),
        ("rotor_inertia", 0.50),       # estimated, not published: 2.5-6.2e-4 kg m^2 reflected
        ("torque_constant", 0.10),
        ("resistance", 0.15),          # copper +0.39%/degC: a winding 40 degC hot is 16% higher
        ("supply_voltage", 0.105),     # 2S LiPo from 6.8 V (cutoff under load) to 8.4 V (fully charged)
        ("gear_efficiency", 0.10),     # 0.70-0.86
    )
    # (parameter, low, high): each scenario's robot has that parameter drawn
    # uniformly from [low, high]. For parameters whose nominal value is 0, which
    # a relative spread can't move (PHYSICS.md 8.4). Friction is unmeasured; the
    # ranges come from the motors' 0.42 A free-run current, which means 0.091 N m
    # of drag (both motors, at the wheels) at the 51 rad/s free-run speed, split
    # anywhere from 30% to 80% Coulomb (DATASHEET.md 5.2). Replace with measured values.
    randomize_range: Tuple[Tuple[str, float, float], ...] = (
        ("wheel_friction", 0.0, 0.0015),      # N m s/rad
        ("coulomb_friction", 0.02, 0.08),     # N m
    )
    # Each scenario's latency, drawn uniformly from the whole physics steps in this
    # range (s); None keeps TimingConfig.latency_s for every scenario. Controllers are
    # still designed for latency_s: they don't know the latency they get.
    latency_range_s: Optional[Tuple[float, float]] = (0.005, 0.020)
    robot_seed: int = 0                # which set of random robots
    sensitivity_weight: float = 1.0    # chaos penalty weight; 0 turns it (and its twin runs) off
    sensitivity_nudge_deg: float = 0.0057   # start-tilt nudge for the twin runs (1e-4 rad)

    @property
    def sensitivity_nudge_rad(self) -> float:
        return math.radians(self.sensitivity_nudge_deg)


# --------------------------------------------------------- PUT IT TOGETHER -
@dataclass(frozen=True)
class ProjectConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    robustness: RobustnessSettings = field(default_factory=RobustnessSettings)

    # ---- builders: turn this configuration into the objects other code uses ----

    def build_robot_params(self) -> RobotParams:
        return self.robot.build(self.actuator.reflected_rotor_inertia)

    def build_actuator(self, robot: Optional[RobotParams] = None):
        return self.actuator.build(robot if robot is not None else self.build_robot_params())

    def build_dynamics(self, ideal_actuator: bool = False) -> Planar2D:
        """The robot model. ideal_actuator=True ignores actuator.model and
        delivers the commanded torque exactly (clipped to max_torque)."""
        robot = self.build_robot_params()
        actuator = None if ideal_actuator else self.build_actuator(robot)
        return Planar2D(robot, fall_angle=self.timing.fall_angle_rad, actuator=actuator)

    def build_sensor(self, seed: int = 0) -> SensorNoise:
        return self.sensor.build(seed)

    def build_pitch_controller(self, gains=None) -> PIDController:
        """A pitch-only PID (no velocity loop) with the configured limits.
        Defaults to the hand-tuned pitch gains (kp, ki, kd)."""
        kp, ki, kd = (self.controller.hand_tuned_gains if gains is None else gains)[:3]
        return PIDController(kp, ki, kd, integral_limit=self.controller.pitch_integral_limit,
                             output_limit=self.robot.max_torque)

    def build_goal(self):
        """The scenario's goal velocity as a function of time (m/s)."""
        return step_profile(self.scenario.goal_steps, self.scenario.goal_transition_s)

    def build_controller(self, gains=None) -> CascadedPIDController:
        """The cascaded controller the tuner scores, with the configured goal
        profile, lean cap, integral limits, and torque limit. Defaults to the
        hand-tuned gains."""
        gains = self.controller.hand_tuned_gains if gains is None else gains
        return CascadedPIDController.from_gains(
            gains,
            torque_limit=self.robot.max_torque,
            pitch_integral_limit=self.controller.pitch_integral_limit,
            goal=self.build_goal(),
            max_pitch_command=self.controller.max_lean_rad,
            integral_limit=self.controller.velocity_integral_limit,
        )

    def cost_matched_lqr_weights(self) -> LQRWeights:
        """LQR weights equal to the cost function's own (CostConfig), pitch from upright."""
        return LQRWeights(velocity=self.cost.velocity_weight, pitch=1.0, pitch_rate=0.0,
                          torque=self.cost.effort_weight)

    def build_lqr_gain(self, weights: Optional[LQRWeights] = None, design_latency: Optional[float] = None):
        """LQR gains for the nominal robot and the configured control period.
        weights default to the cost-matched ones; design_latency to the
        configured latency (0 gives the design that ignores latency)."""
        weights = self.cost_matched_lqr_weights() if weights is None else weights
        latency = self.timing.latency_s if design_latency is None else design_latency
        return lqr_gain(self.build_robot_params(), weights, self.timing.control_dt_s, latency)

    def build_lqr_controller(self, gain=None, **design) -> LQRController:
        """An LQRController with the configured goal profile, torque limit and
        integral cap. gain: these LQR gains; None designs them with
        build_lqr_gain(**design)."""
        gain = self.build_lqr_gain(**design) if gain is None else gain
        return LQRController(gain, self.robot.max_torque, goal=self.build_goal(),
                             integral_limit=self.controller.velocity_integral_limit)

    def score(self, result) -> float:
        """Score a finished run with the configured cost weights."""
        c = self.cost
        return cost(result, c.effort_weight, c.fall_penalty, c.velocity_weight, c.pitch_reference)

    def build_eval_config(self):
        """Build a tuning.EvalConfig that matches this project configuration."""
        from tuning import EvalConfig

        return EvalConfig(
            params=self.build_robot_params(),
            motor=self.actuator.build_motor_params() if self.actuator.model == "dc_motor" else None,
            sensor=self.build_sensor(),
            latency=self.timing.latency_s,
            duration=self.scenario.duration_s,
            dt=self.timing.dt_s,
            control_dt=self.timing.control_dt_s,
            goal_steps=self.scenario.goal_steps,
            goal_transition=self.scenario.goal_transition_s,
            start_pitches=self.scenario.start_pitches_rad,
            train_seeds=self.scenario.train_seeds,
            effort_weight=self.cost.effort_weight,
            fall_penalty=self.cost.fall_penalty,
            velocity_weight=self.cost.velocity_weight,
            pitch_reference=self.cost.pitch_reference,
            fall_angle=self.timing.fall_angle_rad,
            max_pitch_command=self.controller.max_lean_rad,
            pitch_integral_limit=self.controller.pitch_integral_limit,
            velocity_integral_limit=self.controller.velocity_integral_limit,
        )

    def snapshot(self) -> dict:
        """Everything this configuration defines, as plain JSON-friendly data.

        Saved alongside every tuning run so results can always be traced back
        to the exact robot, conditions, cost weights, and search ranges that
        produced them. Includes every field as written above, plus a "derived"
        section with values computed from them (radian conversions, the
        physics-derived minimum Kp, the resolved search bounds, motor limits).
        """
        data = asdict(self)
        # asdict leaves out properties; the weights are what the cost uses.
        data["cost"]["velocity_weight"] = self.cost.velocity_weight
        data["cost"]["effort_weight"] = self.cost.effort_weight
        dyn = self.build_dynamics()
        space = self.build_search_space()
        derived = {
            "fall_angle_rad": self.timing.fall_angle_rad,
            "max_lean_rad": self.controller.max_lean_rad,
            "start_pitches_rad": list(self.scenario.start_pitches_rad),
            "critical_pitch_gain": float(critical_pitch_gain(dyn.p)),
            "critical_pitch_gain_kp_floor_robot": float(critical_pitch_gain(self.kp_floor_robot())),
            "search_bounds": {
                p.name: {"low": float(p.low), "high": float(p.high), "log_scale": p.log}
                for p in space.parameters
            },
        }
        if self.actuator.model == "dc_motor":
            motor = self.actuator.build_motor_params()
            derived["motor_stall_torque_limit"] = float(motor.stall_torque_limit)   # N m, total
            derived["motor_no_load_speed_rad_s"] = float(motor.no_load_speed)       # output shaft
        derived["reflected_rotor_inertia"] = float(self.actuator.reflected_rotor_inertia)   # kg m^2
        derived["cost_matched_lqr_gain"] = [float(k) for k in self.build_lqr_gain()]
        data["derived"] = derived
        return data

    def build_robustness(self):
        """Build a batch_tuning.RobustnessConfig that matches this project configuration."""
        from batch_tuning import RobustnessConfig

        r = self.robustness
        return RobustnessConfig(
            train_seeds=tuple(range(1, r.n_train_seeds + 1)),
            validation_seeds=tuple(r.validation_seeds),
            randomize=tuple(r.randomize),
            randomize_range=tuple(r.randomize_range),
            latency_range=r.latency_range_s,
            robot_seed=r.robot_seed,
            sensitivity_weight=r.sensitivity_weight,
            sensitivity_nudge=r.sensitivity_nudge_rad,
        )

    def kp_floor_robot(self) -> RobotParams:
        """The robot whose critical pitch gain sets the search's Kp floor: of every
        corner of the RobustnessSettings.randomize box (robot parameters only; motor
        parameters don't enter critical_pitch_gain), the one needing the most Kp.
        Before 2026-11-06 the floor came from the nominal robot (0.230 N m/rad);
        the most demanding randomized robot needs 0.273, 1.19 times as much."""
        from itertools import product

        nominal = self.build_robot_params()
        robot_fields = set(RobotParams.__dataclass_fields__)
        spreads = [(name, s) for name, s in self.robustness.randomize if name in robot_fields]
        corners = (nominal.with_changes(**{name: getattr(nominal, name) * (1 + sign * s)
                                           for (name, s), sign in zip(spreads, signs)})
                   for signs in product((-1, 1), repeat=len(spreads)))
        return max(corners, key=critical_pitch_gain)

    def build_search_space(self):
        """Build a tuning.SearchSpace that matches this project configuration."""
        from tuning import cascaded_pid_space

        return cascaded_pid_space(
            self.kp_floor_robot(),
            kp_min_factor=self.search.kp_min_factor,
            kp_max=self.search.kp_max,
            ki_max=self.search.ki_max,
            kd_max=self.search.kd_max,
            kv_p_max=self.search.kv_p_max,
        )

    def build_lqr_space(self):
        """Build a tuning.SearchSpace over the LQR weights (LQRConfig bounds)."""
        from tuning import lqr_weight_space

        return lqr_weight_space(self.lqr.bounds, self.lqr.fixed_weights)

    def build_lqr_designer(self, design_latency: Optional[float] = None):
        """Build the tuning.LQRWeightDesigner that turns searched weights into
        gains: nominal robot, configured control period, design_latency
        (default: the configured latency)."""
        from tuning import LQRWeightDesigner

        latency = self.timing.latency_s if design_latency is None else design_latency
        return LQRWeightDesigner(self.build_robot_params(), self.timing.control_dt_s, latency,
                                 fixed=self.lqr.fixed_weights)

    def build_direct_lqr_space(self):
        """Build a tuning.SearchSpace over the LQR gains themselves, around the
        cost-matched design (LQRConfig.direct_*)."""
        from balance_sim import delayed_commands
        from tuning import lqr_gain_space

        n_in_flight = delayed_commands(self.timing.control_dt_s, self.timing.latency_s)
        return lqr_gain_space(n_in_flight, self.lqr.direct_gain_scale, self.lqr.direct_commands_scale)

    def build_direct_lqr_designer(self):
        """Build the tuning.LQRGainScaling for build_direct_lqr_space: scales the
        cost-matched gains, keeping the nominal linear loop at the configured
        latency stable."""
        from tuning import LQRGainScaling

        return LQRGainScaling(self.build_lqr_gain(), self.build_robot_params(),
                              self.timing.control_dt_s, self.timing.latency_s,
                              commands_scale=self.lqr.direct_commands_scale)


# Edit values above to change the project. To try an alternative all at once
# (e.g. an ideal, frictionless robot) without editing the defaults, override
# specific fields here instead, for example:
#
#   CONFIG = ProjectConfig(actuator=ActuatorConfig(model="ideal"))
#
CONFIG = ProjectConfig()
