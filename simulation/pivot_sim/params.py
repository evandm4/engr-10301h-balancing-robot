"""Physical parameters of the pivot robot: wheels, a lower body, a servo-driven
pivot on top of it, and an upper body carried by the pivot.

    upper body  (mass m2, COM l2 above the pivot, inertia I2)
        |
      pivot     (servo, height h above the axle, angle phi relative to the lower body)
        |
    lower body  (mass m1, COM l1 above the axle, inertia I1; holds the IMU and wheel motors)
        |
      wheels    (same as the base robot)

All values are SI units. Angles are measured from vertical, forward positive.
theta1 is the lower body's pitch; phi is the servo angle (upper body relative
to lower body), so the upper body's absolute pitch is theta2 = theta1 + phi.

There are no defaults: the project's pivot robot is defined once, in
pivot_config.py, and built with PIVOT_CONFIG.build_params().
"""

import math
from dataclasses import dataclass, replace
from typing import Tuple

from balance_sim import RobotParams


@dataclass(frozen=True)
class PivotRobotParams:
    lower_mass: float        # kg, chassis, battery, electronics, wheel motors, servo body
    lower_com_height: float  # m, lower body COM above the wheel axle
    lower_inertia: float     # kg m^2, lower body pitch inertia about its own COM
    pivot_height: float      # m, pivot axis above the wheel axle (along the lower body)
    upper_mass: float        # kg, everything the pivot carries
    upper_com_offset: float  # m, upper body COM above the pivot when phi = 0
    upper_inertia: float     # kg m^2, upper body pitch inertia about its own COM
    wheel_mass: float        # kg, both wheels combined
    wheel_radius: float      # m
    wheel_inertia: float     # kg m^2, both wheels about the axle
    max_torque: float        # N m, total wheel motor torque limit (both motors)
    wheel_friction: float    # N m s/rad, viscous friction between wheels and lower body
    gravity: float           # m/s^2

    def __post_init__(self):
        positive = ("lower_mass", "lower_com_height", "lower_inertia", "pivot_height",
                    "upper_inertia", "wheel_mass", "wheel_radius", "wheel_inertia",
                    "max_torque", "gravity")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        # A massless upper body is allowed (it reduces the model to the base
        # robot, which the tests use); a negative one is not.
        for name in ("upper_mass", "upper_com_offset", "wheel_friction"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    def with_changes(self, **changes) -> "PivotRobotParams":
        return replace(self, **changes)

    # ---- coefficients of the equations of motion (see PIVOT.md) --------------

    @property
    def total_mass(self) -> float:
        return self.lower_mass + self.upper_mass

    @property
    def b1(self) -> float:
        """First moment of mass that swings with theta1: m1 l1 + m2 h."""
        return self.lower_mass * self.lower_com_height + self.upper_mass * self.pivot_height

    @property
    def b2(self) -> float:
        """First moment of mass that swings with theta2: m2 l2."""
        return self.upper_mass * self.upper_com_offset

    # ---- the combined center of mass -----------------------------------------

    def com_pitch(self, theta1: float, phi: float, theta1_dot: float = 0.0,
                  phi_dot: float = 0.0) -> Tuple[float, float]:
        """Angle from vertical of the line from the axle to the robot's combined
        COM (wheels excluded; they sit on the axle), and its rate.

        This is the angle that has to be kept over the wheels to balance. With
        the pivot straight (phi = 0) it equals theta1.
        """
        b1, b2 = self.b1, self.b2
        theta2, theta2_dot = theta1 + phi, theta1_dot + phi_dot
        s = b1 * math.sin(theta1) + b2 * math.sin(theta2)
        c = b1 * math.cos(theta1) + b2 * math.cos(theta2)
        s_dot = b1 * math.cos(theta1) * theta1_dot + b2 * math.cos(theta2) * theta2_dot
        c_dot = -b1 * math.sin(theta1) * theta1_dot - b2 * math.sin(theta2) * theta2_dot
        return math.atan2(s, c), (c * s_dot - s * c_dot) / (s * s + c * c)

    def balance_pitch(self, phi: float) -> float:
        """Lower-body pitch that puts the combined COM straight over the axle
        for a given pivot angle (the static equilibrium)."""
        return math.atan2(-self.b2 * math.sin(phi), self.b1 + self.b2 * math.cos(phi))

    def locked(self) -> RobotParams:
        """The equivalent rigid robot with the pivot frozen straight (phi = 0),
        for the base simulator (balance_sim.Planar2D) and its tools, such as
        critical_pitch_gain."""
        m1, m2, l1 = self.lower_mass, self.upper_mass, self.lower_com_height
        l2_axle = self.pivot_height + self.upper_com_offset
        mass = m1 + m2
        com = (m1 * l1 + m2 * l2_axle) / mass
        inertia = (self.lower_inertia + m1 * (l1 - com) ** 2
                   + self.upper_inertia + m2 * (l2_axle - com) ** 2)
        return RobotParams(
            body_mass=mass,
            body_com_height=com,
            body_inertia=inertia,
            wheel_mass=self.wheel_mass,
            wheel_radius=self.wheel_radius,
            wheel_inertia=self.wheel_inertia,
            max_torque=self.max_torque,
            wheel_friction=self.wheel_friction,
            gravity=self.gravity,
        )


@dataclass(frozen=True)
class ServoParams:
    """A hobby-style position servo (DC motor + gearbox + internal position loop).

    The servo is told an angle, not a torque. Its driver applies a voltage
    proportional to the angle error, saturating at full supply once the error
    reaches full_drive_error; the motor's back-EMF then makes the torque fall
    linearly with speed (stall torque at rest, zero at no_load_speed).

    No defaults: the project's servo is defined in pivot_config.py (ServoConfig).
    """

    stall_torque: float       # N m at the output shaft, at the servo's supply voltage
    no_load_speed: float      # rad/s at the output shaft
    full_drive_error: float   # rad, angle error at which the driver reaches full voltage
    angle_limit: float        # rad, commands are clipped to +-this (mechanical range)
    friction: float           # N m s/rad, viscous gearbox friction on top of back-EMF
    update_period: float      # s, how often the servo accepts a new command (20 ms for 50 Hz PWM)

    def __post_init__(self):
        for name in ("stall_torque", "no_load_speed", "full_drive_error", "angle_limit"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.friction < 0 or self.update_period < 0:
            raise ValueError("friction and update_period must be non-negative")

    @property
    def stiffness(self) -> float:
        """Small-error holding stiffness (N m/rad): stall_torque / full_drive_error."""
        return self.stall_torque / self.full_drive_error

    @property
    def damping(self) -> float:
        """Back-EMF plus gearbox damping (N m s/rad) seen by the pivot."""
        return self.stall_torque / self.no_load_speed + self.friction
