"""Planar (2D) wheeled inverted pendulum, derived with Lagrangian mechanics.

Generalized coordinates: x (axle position) and theta (body pitch). The wheels
roll without slipping, so wheel angle = x / r.

    Kinetic energy:
        T = 1/2 (m_w + I_w/r^2 + M) x'^2
            + M l cos(theta) x' theta'
            + 1/2 (M l^2 + I_b) theta'^2
    Potential energy:
        V = M g l cos(theta)

A motor torque tau acts between wheels and body, so the generalized forces are
Q_x = tau / r and Q_theta = -tau. The Euler-Lagrange equations give

    a x'' + b(theta) theta'' - M l sin(theta) theta'^2 = tau / r
    b(theta) x'' + d theta'' - M g l sin(theta)        = -tau

with a = m_w + I_w/r^2 + M, b = M l cos(theta), d = M l^2 + I_b. The 2x2 system
is solved in closed form for x'' and theta''.

Assumptions: rolling without slip and a rigid body. The motors are handled by
a swappable Actuator: by default the commanded torque is delivered as-is (clipped
to the torque limit); pass a DCMotorActuator for back-EMF and current limits.
These are the first places to look when comparing against hardware.
"""

import math

import numpy as np

from .actuators import Actuator, IdealActuator
from .dynamics import Dynamics
from .params import RobotParams


def critical_pitch_gain(p: RobotParams) -> float:
    """Smallest proportional pitch gain (N m/rad) that can hold the robot up.

    From the linearized model with no friction or torque limit: below this,
    a pitch-only PID cannot overcome gravity no matter how the other gains
    are set. Useful as a lower bound when the tuner searches for gains.
    """
    ml = p.body_mass * p.body_com_height
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
    return a * ml * p.gravity / (a + ml / p.wheel_radius)


class Planar2D(Dynamics):
    """fall_angle (rad): past this pitch the robot counts as fallen; the
    project's value is TimingConfig.fall_angle_deg in config.py.
    actuator: None means the ideal actuator, clipped to params.max_torque."""

    def __init__(
        self,
        params: RobotParams,
        fall_angle: float,
        actuator: Actuator = None,
    ):
        self.p = params
        self.fall_angle = fall_angle
        self.actuator = actuator if actuator is not None else IdealActuator(params.max_torque)

    def derivatives(self, state: np.ndarray, torque: float) -> np.ndarray:
        p = self.p
        # Plain floats are much faster than numpy scalars in this hot loop.
        x_dot, theta, theta_dot = float(state[1]), float(state[2]), float(state[3])

        relative_wheel_speed = x_dot / p.wheel_radius - theta_dot
        tau = self.actuator.torque(torque, relative_wheel_speed)
        tau -= p.wheel_friction * relative_wheel_speed

        sin_t, cos_t = math.sin(theta), math.cos(theta)
        ml = p.body_mass * p.body_com_height

        a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
        b = ml * cos_t
        d = p.body_mass * p.body_com_height**2 + p.body_inertia

        rhs_x = tau / p.wheel_radius + ml * sin_t * theta_dot**2
        rhs_t = -tau + ml * p.gravity * sin_t

        det = a * d - b * b
        x_ddot = (d * rhs_x - b * rhs_t) / det
        theta_ddot = (a * rhs_t - b * rhs_x) / det

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def initial_state(self, pitch: float = 0.0) -> np.ndarray:
        return np.array([0.0, 0.0, pitch, 0.0])

    def is_fallen(self, state: np.ndarray) -> bool:
        return abs(state[2]) > self.fall_angle

    def critical_pitch_gain(self) -> float:
        """See the module-level critical_pitch_gain."""
        return critical_pitch_gain(self.p)

    def energy(self, state: np.ndarray) -> float:
        """Total mechanical energy (kinetic + potential). Useful for tests."""
        p = self.p
        _, x_dot, theta, theta_dot = state[:4]
        ml = p.body_mass * p.body_com_height
        a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
        d = p.body_mass * p.body_com_height**2 + p.body_inertia
        kinetic = (
            0.5 * a * x_dot**2
            + ml * np.cos(theta) * x_dot * theta_dot
            + 0.5 * d * theta_dot**2
        )
        potential = ml * p.gravity * np.cos(theta)
        return kinetic + potential
