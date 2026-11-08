"""Planar (2D) wheeled inverted pendulum, derived with Lagrangian mechanics.

Generalized coordinates: x (axle position) and theta (body pitch). The wheels
roll without slipping, so wheel angle = x / r.

    Kinetic energy:
        T = 1/2 (m_w + I_w/r^2 + M) x'^2
            + M l cos(theta) x' theta'
            + 1/2 (M l^2 + I_b) theta'^2
            + 1/2 J (x'/r - theta')^2
    with J = n N^2 J_r the motor rotors reflected through the gearbox: they
    turn at N times the wheel speed relative to the body (PHYSICS.md 10.3).
    Potential energy:
        V = M g l cos(theta)

A motor torque tau acts between wheels and body, so the generalized forces are
Q_x = tau / r and Q_theta = -tau. The Euler-Lagrange equations give

    a x'' + b(theta) theta'' - M l sin(theta) theta'^2 = tau / r
    b(theta) x'' + d theta'' - M g l sin(theta)        = -tau

with a = m_w + I_w/r^2 + M + J/r^2, b = M l cos(theta) - J/r, d = M l^2 + I_b + J.
J is constant, so it adds no velocity terms. The 2x2 system is solved in closed
form for x'' and theta''.

Friction between wheels and body, tau_f = b w + tau_c tanh(w / w_eps) with w
the relative wheel speed, comes off the motor torque (viscous plus smoothed
Coulomb; both zero by default).

An external horizontal force F on the body at its center of mass (a push;
zero unless the simulator is given one) does virtual work F d(x + l sin theta),
so it adds Q_x = F and Q_theta = F l cos(theta) to the generalized forces.

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
    The rotor inertia J adds J/r^2 to a and takes J/r off b (= M l at upright),
    so the denominator a + b/r is unchanged and only the numerator grows.
    """
    ml = p.body_mass * p.body_com_height
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2 + p.rotor_inertia / p.wheel_radius**2
    b = ml - p.rotor_inertia / p.wheel_radius
    return a * ml * p.gravity / (a + b / p.wheel_radius)


def linearize(p: RobotParams):
    """The model linearized about upright: d/dt [x, x_dot, theta, theta_dot] = A s + B tau.

    PHYSICS.md 3.1, plus the wheel friction (zero by default), which takes
    b (x_dot / r - theta_dot) off the motor torque. The smoothed Coulomb term
    tau_c tanh(w / w_eps) enters through its slope at zero speed, tau_c / w_eps:
    a large damping, but only within about w_eps of zero. The actuator is taken
    as unsaturated: below its limits the DC motor model delivers the command
    exactly (PHYSICS.md 4.3). Returns A (4, 4) and B (4, 1).
    """
    ml = p.body_mass * p.body_com_height
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2 + p.rotor_inertia / p.wheel_radius**2
    b = ml - p.rotor_inertia / p.wheel_radius          # the mass matrix's off-diagonal at theta = 0
    d = p.body_mass * p.body_com_height**2 + p.body_inertia + p.rotor_inertia
    det = a * d - b * b
    # Accelerations per unit of net torque, from the 2x2 system in derivatives().
    x_per_torque = (d / p.wheel_radius + b) / det
    theta_per_torque = -(a + b / p.wheel_radius) / det
    A = np.zeros((4, 4))
    A[0, 1] = A[2, 3] = 1.0
    A[1, 2] = -b * ml * p.gravity / det
    A[3, 2] = a * ml * p.gravity / det
    damping = p.wheel_friction + p.coulomb_friction / p.coulomb_friction_speed
    friction = np.array([-damping / p.wheel_radius, damping])   # net torque per (x_dot, theta_dot)
    A[1, [1, 3]] += x_per_torque * friction
    A[3, [1, 3]] += theta_per_torque * friction
    B = np.array([[0.0], [x_per_torque], [0.0], [theta_per_torque]])
    return A, B


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

    def derivatives(self, state: np.ndarray, torque: float, force: float = 0.0) -> np.ndarray:
        """force (N): horizontal push on the body at its center of mass, forward +."""
        p = self.p
        # Plain floats are much faster than numpy scalars in this hot loop.
        x_dot, theta, theta_dot = float(state[1]), float(state[2]), float(state[3])

        relative_wheel_speed = x_dot / p.wheel_radius - theta_dot
        tau = self.actuator.torque(torque, relative_wheel_speed)
        tau -= p.wheel_friction * relative_wheel_speed
        if p.coulomb_friction:
            # np.tanh, not math.tanh: they differ in the last bit, and batch_sim uses NumPy's.
            tau -= p.coulomb_friction * float(np.tanh(relative_wheel_speed / p.coulomb_friction_speed))

        sin_t, cos_t = math.sin(theta), math.cos(theta)
        ml = p.body_mass * p.body_com_height

        a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2 + p.rotor_inertia / p.wheel_radius**2
        b = ml * cos_t - p.rotor_inertia / p.wheel_radius
        d = p.body_mass * p.body_com_height**2 + p.body_inertia + p.rotor_inertia

        rhs_x = tau / p.wheel_radius + ml * sin_t * theta_dot**2
        rhs_t = -tau + ml * p.gravity * sin_t
        if force:   # skipped without a push, so the arithmetic is unchanged (batch_sim does the same)
            rhs_x = rhs_x + force
            rhs_t = rhs_t + force * p.body_com_height * cos_t

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

    def linearize(self):
        """See the module-level linearize."""
        return linearize(self.p)

    def energy(self, state: np.ndarray) -> float:
        """Total mechanical energy (kinetic + potential). Useful for tests."""
        p = self.p
        _, x_dot, theta, theta_dot = state[:4]
        ml = p.body_mass * p.body_com_height
        a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
        d = p.body_mass * p.body_com_height**2 + p.body_inertia
        relative_wheel_speed = x_dot / p.wheel_radius - theta_dot
        kinetic = (
            0.5 * a * x_dot**2
            + ml * np.cos(theta) * x_dot * theta_dot
            + 0.5 * d * theta_dot**2
            + 0.5 * p.rotor_inertia * relative_wheel_speed**2
        )
        potential = ml * p.gravity * np.cos(theta)
        return kinetic + potential
