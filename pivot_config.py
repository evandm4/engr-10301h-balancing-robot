"""
=============================================================================
 PIVOT ROBOT CONFIGURATION (side project)
=============================================================================
The robot from config.py with a servo-driven pivot on top and a second body
piece carried by it. See PIVOT.md for the physics and the controllers.

Only what is new lives here: how the old body splits into a lower body and an
upper body, the servo, the pivot sensor, the pivot loop's gains, the extra
cost terms, the LQR weights, and the extra search bounds. Everything shared
with the base robot (wheels, wheel motors, IMU noise, latency, timing, the
goal-velocity scenario, the base cost weights) is read from config.py's
CONFIG, so changing it there changes it here too. Nothing in config.py or in
the base simulation is changed by this file.

Angles are in degrees below for readability; *_rad properties convert.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

from config import CONFIG, ProjectConfig  # also puts simulation/ and optimization/ on the path

from balance_sim import step_profile
from pivot_sim import (
    PIVOT_GAIN_NAMES,
    LQRWeights,
    PivotCascadedController,
    PivotLQRController,
    PivotRobot,
    PivotRobotParams,
    PivotSensor,
    ServoActuator,
    ServoParams,
    pivot_cost,
    pivot_metrics,
    simulate_pivot,
)


# ---------------------------------------------------------------- BODIES ---
@dataclass(frozen=True)
class PivotBodyConfig:
    """The old single body split in two at a pivot. Same total body mass as
    config.py's robot (0.80 kg); with the pivot straight the combined COM sits
    at about 0.104 m, close to the base robot's 0.10 m. Rough guesses: replace
    with measured values once the parts exist."""

    lower_mass: float = 0.55           # kg: chassis, battery, electronics, wheel motors, servo
    lower_com_height: float = 0.06     # m above the axle
    lower_inertia: float = 0.0012      # kg m^2 about its own COM
    pivot_height: float = 0.14         # m, pivot axis above the axle
    upper_mass: float = 0.25           # kg, everything on the pivot (include the servo horn/bracket)
    upper_com_offset: float = 0.06     # m above the pivot with the pivot straight
    upper_inertia: float = 0.0003      # kg m^2 about its own COM (add the servo's reflected rotor inertia here)


# ----------------------------------------------------------------- SERVO ---
@dataclass(frozen=True)
class ServoConfig:
    """A standard-size metal-gear hobby servo (MG996R class) at 6 V, derated a
    little from datasheet numbers (about 1.1 N m stall, 0.14 s/60 deg). It
    needs its own 6 V regulator from the 7.4 V pack. Holding the upper body
    horizontal takes upper_mass * g * upper_com_offset = 0.15 N m."""

    stall_torque: float = 1.0          # N m
    no_load_time_per_60deg_s: float = 0.15    # datasheet "speed": seconds to turn 60 degrees, no load
    full_drive_error_deg: float = 5.0  # angle error at which the servo's driver is at full voltage
    angle_limit_deg: float = 60.0      # commands are clipped to +-this
    friction: float = 0.0              # N m s/rad, extra gearbox friction
    update_period_s: float = 0.020     # 50 Hz PWM; digital servos accept faster (e.g. 0.003)

    @property
    def no_load_speed(self) -> float:
        return math.radians(60.0) / self.no_load_time_per_60deg_s

    @property
    def angle_limit_rad(self) -> float:
        return math.radians(self.angle_limit_deg)

    def build(self) -> ServoParams:
        return ServoParams(
            stall_torque=self.stall_torque,
            no_load_speed=self.no_load_speed,
            full_drive_error=math.radians(self.full_drive_error_deg),
            angle_limit=self.angle_limit_rad,
            friction=self.friction,
            update_period=self.update_period_s,
        )


# ---------------------------------------------------------------- SENSOR ---
@dataclass(frozen=True)
class PivotSensorConfig:
    """Noise on the pivot angle the controller reads (a servo with position
    feedback). The lower body's IMU and the encoders use config.py's SensorConfig."""

    pivot_angle_std_deg: float = 0.5
    pivot_rate_std: float = 0.05       # rad/s


# ------------------------------------------------------------ CONTROLLER ---
@dataclass(frozen=True)
class PivotControllerConfig:
    """The cascaded PID's wheel gains come from config.py (hand_tuned_gains).
    These are the pivot loop's (see PivotCascadedController):
    phi_cmd = -k_level * theta1 + k_phi_p * (COM pitch error) + k_phi_d * (COM pitch rate)."""

    hand_tuned_pivot_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # pivot held straight
    level_pivot_gains: Tuple[float, float, float] = (1.0, 0.0, 0.0)       # keep the upper body vertical
    pitch_source: str = "com"          # what the wheel loop balances: "com" or "lower"


# ------------------------------------------------------------------ COST ---
@dataclass(frozen=True)
class PivotCostConfig:
    """Added to config.py's cost (which is applied to the combined COM)."""

    servo_effort_weight: float = 0.01  # per unit of integrated servo torque^2
    upper_tilt_weight: float = 0.0     # per unit of integrated upper-body pitch^2; > 0 asks for a level top


# ------------------------------------------------------------------- LQR ---
@dataclass(frozen=True)
class LQRConfig:
    """Weights for the LQR benchmark (see pivot_sim.LQRWeights). Close to the
    project's cost (velocity error 1, wheel torque 0.01), with COM pitch
    weighted 3 rather than 1: at 1 the LQR leans the robot up to about 48
    degrees to chase the velocity steps. A little weight on the rates, the
    pivot and the servo command keeps the design gentle enough for the
    latency, servo frame rate and noise it doesn't model, and keeps the pivot
    well inside its range (about 25 degrees at most on the tuning scenario)."""

    velocity_integral: float = 0.5
    velocity: float = 1.0
    com_pitch: float = 3.0
    com_pitch_rate: float = 0.01
    upper_pitch: float = 0.0
    pivot: float = 0.05
    pivot_rate: float = 0.002
    torque: float = 0.01
    servo: float = 0.05
    integral_limit: float = 0.5        # m, cap on the velocity-error integral

    def build(self) -> LQRWeights:
        return LQRWeights(
            velocity_integral=self.velocity_integral, velocity=self.velocity,
            com_pitch=self.com_pitch, com_pitch_rate=self.com_pitch_rate,
            upper_pitch=self.upper_pitch, pivot=self.pivot, pivot_rate=self.pivot_rate,
            torque=self.torque, servo=self.servo,
        )


# ----------------------------------------------------------- SEARCH SPACE --
@dataclass(frozen=True)
class PivotSearchConfig:
    """Bounds for the three pivot gains; the four wheel gains use config.py's
    SearchConfig (with the minimum Kp recomputed for this robot)."""

    k_level: Tuple[float, float] = (0.0, 1.0)
    k_phi_p: Tuple[float, float] = (-3.0, 3.0)
    k_phi_d: Tuple[float, float] = (-0.5, 0.5)


# --------------------------------------------------------- PUT IT TOGETHER -
@dataclass(frozen=True)
class PivotProjectConfig:
    base: ProjectConfig = CONFIG
    body: PivotBodyConfig = field(default_factory=PivotBodyConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    sensor: PivotSensorConfig = field(default_factory=PivotSensorConfig)
    controller: PivotControllerConfig = field(default_factory=PivotControllerConfig)
    cost: PivotCostConfig = field(default_factory=PivotCostConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    search: PivotSearchConfig = field(default_factory=PivotSearchConfig)

    # ---- builders ----

    def build_params(self) -> PivotRobotParams:
        b, r = self.body, self.base.robot
        return PivotRobotParams(
            lower_mass=b.lower_mass, lower_com_height=b.lower_com_height, lower_inertia=b.lower_inertia,
            pivot_height=b.pivot_height,
            upper_mass=b.upper_mass, upper_com_offset=b.upper_com_offset, upper_inertia=b.upper_inertia,
            wheel_mass=r.wheel_mass, wheel_radius=r.wheel_radius, wheel_inertia=r.wheel_inertia,
            max_torque=r.max_torque, wheel_friction=r.wheel_friction, gravity=r.gravity,
        )

    def build_dynamics(self, ideal_actuator: bool = False) -> PivotRobot:
        """ideal_actuator=True: wheel motors deliver the commanded torque exactly
        (clipped). Otherwise config.py's actuator model (the DC motor)."""
        params = self.build_params()
        actuator = None if ideal_actuator else self.base.actuator.build(params.locked())
        return PivotRobot(params, ServoActuator(self.servo.build()),
                          fall_angle=self.base.timing.fall_angle_rad, actuator=actuator)

    def build_sensor(self, seed: int = 0) -> PivotSensor:
        return PivotSensor(self.base.build_sensor(seed),
                           pivot_angle_std=math.radians(self.sensor.pivot_angle_std_deg),
                           pivot_rate_std=self.sensor.pivot_rate_std)

    @property
    def gain_names(self) -> Tuple[str, ...]:
        return ("kp", "ki", "kd", "kv_p") + PIVOT_GAIN_NAMES

    def hand_tuned_gains(self) -> Tuple[float, ...]:
        """config.py's wheel gains with the pivot held straight."""
        return tuple(self.base.controller.hand_tuned_gains) + tuple(self.controller.hand_tuned_pivot_gains)

    def level_gains(self) -> Tuple[float, ...]:
        """config.py's wheel gains with the pivot keeping the upper body vertical."""
        return tuple(self.base.controller.hand_tuned_gains) + tuple(self.controller.level_pivot_gains)

    def build_controller(self, gains=None) -> PivotCascadedController:
        """Cascaded PID + pivot loop with the project's goal profile and limits.
        gains: [kp, ki, kd, kv_p, k_level, k_phi_p, k_phi_d]; default hand_tuned_gains()."""
        base = self.base
        return PivotCascadedController.from_gains(
            self.hand_tuned_gains() if gains is None else gains,
            self.build_params(),
            servo_limit=self.servo.angle_limit_rad,
            torque_limit=base.robot.max_torque,
            pitch_integral_limit=base.controller.pitch_integral_limit,
            pitch_source=self.controller.pitch_source,
            goal=step_profile(base.scenario.goal_steps),
            max_pitch_command=base.controller.max_lean_rad,
            integral_limit=base.controller.velocity_integral_limit,
        )

    def build_lqr(self, ideal_actuator: bool = False, latency: Optional[float] = None,
                  pivot_active: bool = True) -> PivotLQRController:
        """The LQR benchmark, designed for the project's control rate and latency
        (latency=0 for a run without delay) and the project's goal profile.
        pivot_active=False: the same design with the pivot held straight."""
        base = self.base
        return PivotLQRController.design(
            self.build_dynamics(ideal_actuator), self.lqr.build(),
            control_dt=base.timing.control_dt_s,
            latency=base.timing.latency_s if latency is None else latency,
            servo_limit=self.servo.angle_limit_rad,
            pivot_active=pivot_active,
            goal=step_profile(base.scenario.goal_steps),
            integral_limit=self.lqr.integral_limit,
        )

    def run(self, controller, start_pitch: float, seed: Optional[int] = 0, realistic: bool = True,
            dt: Optional[float] = None):
        """Simulate the tuner's scenario once. realistic=True: DC motors, sensor
        noise (this seed), latency. False: ideal motors, perfect sensing, no
        delay. dt defaults to the tuning step."""
        base = self.base
        dyn = self.build_dynamics(ideal_actuator=not realistic)
        return simulate_pivot(
            dyn, controller,
            duration=base.scenario.duration_s,
            dt=base.timing.dt_s if dt is None else dt,
            initial_state=dyn.initial_state(start_pitch),
            control_dt=base.timing.control_dt_s,
            sensor=self.build_sensor(seed) if realistic and seed is not None else None,
            latency=base.timing.latency_s if realistic else 0.0,
        )

    def score(self, result) -> float:
        c, pc = self.base.cost, self.cost
        return pivot_cost(result, c.effort_weight, c.fall_penalty, c.velocity_weight, c.pitch_reference,
                          pc.servo_effort_weight, pc.upper_tilt_weight)

    def metrics(self, result):
        return pivot_metrics(result, self.servo.stall_torque)

    def build_search_space(self):
        """config.py's wheel-gain space (Kp floor from this robot's locked
        equivalent) plus the three pivot gains."""
        from tuning import Parameter, SearchSpace, cascaded_pid_space

        s = self.base.search
        wheel = cascaded_pid_space(self.build_params().locked(), kp_min_factor=s.kp_min_factor,
                                   kp_max=s.kp_max, ki_max=s.ki_max, kd_max=s.kd_max, kv_p_max=s.kv_p_max)
        bounds = (self.search.k_level, self.search.k_phi_p, self.search.k_phi_d)
        return SearchSpace(wheel.parameters + [Parameter(n, lo, hi) for n, (lo, hi) in zip(PIVOT_GAIN_NAMES, bounds)])

    def snapshot(self) -> dict:
        """Everything this configuration defines (and the base config it builds
        on), as JSON-friendly data, for saving next to results."""
        data = asdict(self)
        data["base"] = self.base.snapshot()
        params = self.build_params()
        locked = params.locked()
        data["derived"] = {
            "servo_no_load_speed_rad_s": self.servo.no_load_speed,
            "servo_stiffness_nm_per_rad": self.servo.build().stiffness,
            "upper_body_holding_torque_horizontal_nm": params.upper_mass * params.gravity * params.upper_com_offset,
            "locked_equivalent": asdict(locked),
            "search_bounds": {p.name: {"low": float(p.low), "high": float(p.high), "log_scale": p.log}
                              for p in self.build_search_space().parameters},
        }
        return data


PIVOT_CONFIG = PivotProjectConfig()
