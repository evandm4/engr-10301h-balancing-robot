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
    MotorParams,
    PIDController,
    Planar2D,
    RobotParams,
    SensorNoise,
    cost,
    critical_pitch_gain,
    step_profile,
)


# ----------------------------------------------------------------- ROBOT ---
@dataclass(frozen=True)
class RobotConfig:
    """The physical robot the simulation models. These numbers double as the
    hardware build spec if the physical robot gets made."""

    body_mass: float = 0.80            # kg, chassis + electronics, excludes wheels
    body_com_height: float = 0.10      # m, body center of mass above the wheel axle
    body_inertia: float = 0.0033       # kg m^2, body pitch inertia about its own COM
    wheel_mass: float = 0.10           # kg, both wheels combined
    wheel_radius: float = 0.035        # m
    wheel_inertia: float = 6.1e-5      # kg m^2, both wheels about the axle
    max_torque: float = 0.40           # N m, total torque limit (both motors). Applies to both
                                       # actuator models; the DC motor can limit it further.
    wheel_friction: float = 0.0        # N m s/rad, viscous friction between wheels and body
    gravity: float = 9.81              # m/s^2

    def build(self) -> RobotParams:
        return RobotParams(
            body_mass=self.body_mass,
            body_com_height=self.body_com_height,
            body_inertia=self.body_inertia,
            wheel_mass=self.wheel_mass,
            wheel_radius=self.wheel_radius,
            wheel_inertia=self.wheel_inertia,
            max_torque=self.max_torque,
            wheel_friction=self.wheel_friction,
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

    # Only used when model == "dc_motor". Rough guess for a small 7.4 V geared
    # motor; replace with datasheet or measured values.
    gear_ratio: float = 20.0           # motor turns per output turn
    torque_constant: float = 0.005     # N m/A at the motor shaft (= back-EMF constant in SI)
    resistance: float = 3.0            # ohm, winding resistance
    supply_voltage: float = 7.4        # V
    current_limit: float = 2.0         # A, per motor
    n_motors: int = 2

    def build_motor_params(self) -> MotorParams:
        return MotorParams(
            gear_ratio=self.gear_ratio,
            torque_constant=self.torque_constant,
            resistance=self.resistance,
            supply_voltage=self.supply_voltage,
            current_limit=self.current_limit,
            n_motors=self.n_motors,
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

    pitch_std: float = 0.005           # rad
    pitch_rate_std: float = 0.02       # rad/s, gyro
    pitch_rate_bias: float = 0.01      # rad/s, constant gyro offset
    position_std: float = 0.0005       # m
    velocity_std: float = 0.02         # m/s

    def build(self, seed: int = 0) -> SensorNoise:
        return SensorNoise(
            pitch_std=self.pitch_std,
            pitch_rate_std=self.pitch_rate_std,
            pitch_rate_bias=self.pitch_rate_bias,
            position_std=self.position_std,
            velocity_std=self.velocity_std,
            seed=seed,
        )


# ---------------------------------------------------------------- TIMING ---
@dataclass(frozen=True)
class TimingConfig:
    latency_s: float = 0.010           # controller-to-actuator delay
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
    velocity_integral_limit: float = 0.5   # m, cap on the velocity loop's integral (unused while kv_i = 0)
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

    effort_weight: float = 0.01        # penalty per unit of integrated torque^2
    fall_penalty: float = 100.0        # added to the cost if the robot falls
    velocity_weight: float = 1.0       # penalty per unit of integrated velocity error^2
    # What pitch error is measured against. "commanded": the lean the velocity
    # loop asked for (leaning to accelerate isn't penalized, only failing to
    # hold the commanded lean). "upright": vertical, the original cost, which
    # penalizes every lean including the ones the task requires.
    pitch_reference: str = "commanded"

    def __post_init__(self):
        if self.pitch_reference not in PITCH_REFERENCES:
            raise ValueError(f"pitch_reference must be one of {PITCH_REFERENCES}, got {self.pitch_reference!r}")


# -------------------------------------------------------------- SCENARIO ---
@dataclass(frozen=True)
class ScenarioConfig:
    """What the tuner's evaluator asks the robot to do, and the range of
    conditions it's scored under."""

    duration_s: float = 10.0
    goal_steps: Tuple[Tuple[float, float], ...] = ((2.0, 0.8), (3.5, 1.4), (6.0, -0.6), (9.0, 1.0))  # (start time s, goal m/s)
    start_pitches_deg: Tuple[float, ...] = (5.73, -4.58, 2.86)                 # initial disturbance, each tried with every seed
    train_seeds: Tuple[int, ...] = (1, 2, 3)                                   # sensor-noise seeds scored during tuning
    validation_seeds: Tuple[int, ...] = tuple(range(100, 110))                 # unseen seeds, checked after tuning
    # Runs per score = len(start_pitches_deg) x number of seeds: 9 per training
    # evaluation and 30 per validation with the defaults above.

    @property
    def start_pitches_rad(self) -> Tuple[float, ...]:
        return tuple(math.radians(d) for d in self.start_pitches_deg)


# ----------------------------------------------------------- SEARCH SPACE --
@dataclass(frozen=True)
class SearchConfig:
    """Bounds the tuner searches within for each gain. kp_min_factor is a
    multiple of the physics-derived minimum Kp that can hold the robot up at
    all (see balance_sim.critical_pitch_gain), not a raw value."""

    kp_min_factor: float = 1.1
    kp_max: float = 30.0
    ki_max: float = 5.0
    kd_max: float = 1.0
    kv_p_max: float = 0.4


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
    # scaled by a random factor in [1 - spread, 1 + spread]. Rough guesses at
    # build and datasheet tolerances; replace with measured spreads. Names are
    # RobotConfig / ActuatorConfig fields.
    randomize: Tuple[Tuple[str, float], ...] = (
        ("body_mass", 0.15),
        ("body_com_height", 0.15),
        ("body_inertia", 0.25),
        ("wheel_inertia", 0.20),
        ("torque_constant", 0.10),
        ("resistance", 0.15),
        ("supply_voltage", 0.10),      # battery from about 6.7 V (low) to 8.1 V (fully charged)
    )
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
    robustness: RobustnessSettings = field(default_factory=RobustnessSettings)

    # ---- builders: turn this configuration into the objects other code uses ----

    def build_robot_params(self) -> RobotParams:
        return self.robot.build()

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

    def build_controller(self, gains=None) -> CascadedPIDController:
        """The cascaded controller the tuner scores, with the configured goal
        profile, lean cap, integral limits, and torque limit. Defaults to the
        hand-tuned gains."""
        gains = self.controller.hand_tuned_gains if gains is None else gains
        return CascadedPIDController.from_gains(
            gains,
            torque_limit=self.robot.max_torque,
            pitch_integral_limit=self.controller.pitch_integral_limit,
            goal=step_profile(self.scenario.goal_steps),
            max_pitch_command=self.controller.max_lean_rad,
            integral_limit=self.controller.velocity_integral_limit,
        )

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
        dyn = self.build_dynamics()
        space = self.build_search_space()
        derived = {
            "fall_angle_rad": self.timing.fall_angle_rad,
            "max_lean_rad": self.controller.max_lean_rad,
            "start_pitches_rad": list(self.scenario.start_pitches_rad),
            "critical_pitch_gain": float(critical_pitch_gain(dyn.p)),
            "search_bounds": {
                p.name: {"low": float(p.low), "high": float(p.high), "log_scale": p.log}
                for p in space.parameters
            },
        }
        if self.actuator.model == "dc_motor":
            motor = self.actuator.build_motor_params()
            derived["motor_stall_torque_limit"] = float(motor.stall_torque_limit)   # N m, total
            derived["motor_no_load_speed_rad_s"] = float(motor.no_load_speed)       # output shaft
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
            robot_seed=r.robot_seed,
            sensitivity_weight=r.sensitivity_weight,
            sensitivity_nudge=r.sensitivity_nudge_rad,
        )

    def build_search_space(self):
        """Build a tuning.SearchSpace that matches this project configuration."""
        from tuning import cascaded_pid_space

        return cascaded_pid_space(
            self.build_robot_params(),
            kp_min_factor=self.search.kp_min_factor,
            kp_max=self.search.kp_max,
            ki_max=self.search.ki_max,
            kd_max=self.search.kd_max,
            kv_p_max=self.search.kv_p_max,
        )


# Edit values above to change the project. To try an alternative all at once
# (e.g. an ideal, frictionless robot) without editing the defaults, override
# specific fields here instead, for example:
#
#   CONFIG = ProjectConfig(actuator=ActuatorConfig(model="ideal"))
#
CONFIG = ProjectConfig()
