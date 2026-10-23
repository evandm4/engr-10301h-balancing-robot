"""Planar wheeled robot with a servo-driven pivot and a second body on top.

Three degrees of freedom, derived with Lagrangian mechanics (full derivation
in PIVOT.md). Generalized coordinates: x (axle position), theta1 (lower body
pitch) and theta2 = theta1 + phi (upper body pitch), all from vertical. Wheels
roll without slipping.

    Kinetic energy:
        T = 1/2 a x'^2
            + b1 cos(theta1) x' theta1' + b2 cos(theta2) x' theta2'
            + 1/2 d1 theta1'^2 + 1/2 d2 theta2'^2
            + c cos(theta2 - theta1) theta1' theta2'
    Potential energy:
        V = g (b1 cos(theta1) + b2 cos(theta2))

    a  = m_w + I_w/r^2 + m1 + m2       b1 = m1 l1 + m2 h      b2 = m2 l2
    d1 = m1 l1^2 + I1 + m2 h^2         d2 = m2 l2^2 + I2      c  = m2 h l2

The wheel motors apply tau_w between the wheels and the lower body; the servo
applies tau_s between the lower and upper bodies. Generalized forces:
Q_x = tau_w / r, Q_theta1 = -tau_w - tau_s, Q_theta2 = tau_s. The equations of
motion are M(q) q'' = f(q, q', tau) with

    M = [[a,            b1 cos(theta1),  b2 cos(theta2)],
         [b1 cos(theta1), d1,            c cos(phi)    ],
         [b2 cos(theta2), c cos(phi),    d2            ]]

    f = [tau_w / r + b1 sin(theta1) theta1'^2 + b2 sin(theta2) theta2'^2,
         -tau_w - tau_s + c sin(phi) theta2'^2 + g b1 sin(theta1),
          tau_s         - c sin(phi) theta1'^2 + g b2 sin(theta2)]

solved in closed form (3x3, symmetric) each call.

State layout: the base robot's four states first, so anything written for the
base robot (sensor noise, the wheel loop of a cascaded PID) works on [:4]:

    state[0]  x          axle position                      (m)
    state[1]  x_dot      forward velocity                   (m/s)
    state[2]  theta1     lower body pitch                   (rad)
    state[3]  theta1_dot lower body pitch rate              (rad/s)
    state[4]  phi        pivot angle, upper relative to lower (rad)
    state[5]  phi_dot    pivot rate                         (rad/s)

The command is a pair (wheel torque command in N m, servo angle command in rad).
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from balance_sim import Actuator, IdealActuator

from .params import PivotRobotParams
from .servo import ServoActuator


class PivotRobot:
    """fall_angle (rad): the run counts as fallen once either body tilts past
    it. actuator: the wheel motors; None means ideal, clipped to max_torque."""

    state_dim = 6

    def __init__(
        self,
        params: PivotRobotParams,
        servo: ServoActuator,
        fall_angle: float,
        actuator: Optional[Actuator] = None,
    ):
        self.p = params
        self.servo = servo
        self.fall_angle = fall_angle
        self.actuator = actuator if actuator is not None else IdealActuator(params.max_torque)
        # Constant coefficients of the equations of motion.
        p = params
        self._a = p.lower_mass + p.upper_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
        self._b1, self._b2 = p.b1, p.b2
        self._c = p.upper_mass * p.pivot_height * p.upper_com_offset
        self._d1 = p.lower_mass * p.lower_com_height**2 + p.lower_inertia + p.upper_mass * p.pivot_height**2
        self._d2 = p.upper_mass * p.upper_com_offset**2 + p.upper_inertia

    def applied_torques(self, state: Sequence[float], command: Tuple[float, float]) -> Tuple[float, float]:
        """(wheel torque, servo torque) actually applied, in N m, after the
        motor, friction and servo models. Wheel torque acts wheels-vs-lower
        body; servo torque acts upper-vs-lower body."""
        p = self.p
        x_dot, theta1_dot = float(state[1]), float(state[3])
        phi, phi_dot = float(state[4]), float(state[5])
        relative_wheel_speed = x_dot / p.wheel_radius - theta1_dot
        tau_w = self.actuator.torque(command[0], relative_wheel_speed)
        tau_w -= p.wheel_friction * relative_wheel_speed
        tau_s = self.servo.torque(command[1], phi, phi_dot)
        return tau_w, tau_s

    def derivatives(self, state: np.ndarray, command: Tuple[float, float]) -> np.ndarray:
        p = self.p
        # Plain floats: much faster than numpy scalars in this hot loop.
        x_dot, theta1, theta1_dot = float(state[1]), float(state[2]), float(state[3])
        phi, phi_dot = float(state[4]), float(state[5])
        theta2, theta2_dot = theta1 + phi, theta1_dot + phi_dot
        tau_w, tau_s = self.applied_torques(state, command)

        s1, c1 = math.sin(theta1), math.cos(theta1)
        s2, c2 = math.sin(theta2), math.cos(theta2)
        sp, cp = math.sin(phi), math.cos(phi)
        a, b1, b2, c, d1, d2 = self._a, self._b1, self._b2, self._c, self._d1, self._d2

        m01, m02, m12 = b1 * c1, b2 * c2, c * cp
        f0 = tau_w / p.wheel_radius + b1 * s1 * theta1_dot**2 + b2 * s2 * theta2_dot**2
        f1 = -tau_w - tau_s + c * sp * theta2_dot**2 + p.gravity * b1 * s1
        f2 = tau_s - c * sp * theta1_dot**2 + p.gravity * b2 * s2

        # Symmetric 3x3 solve by cofactors.
        c00 = d1 * d2 - m12 * m12
        c01 = m02 * m12 - m01 * d2
        c02 = m01 * m12 - m02 * d1
        c11 = a * d2 - m02 * m02
        c12 = m01 * m02 - a * m12
        c22 = a * d1 - m01 * m01
        det = a * c00 + m01 * c01 + m02 * c02
        x_ddot = (c00 * f0 + c01 * f1 + c02 * f2) / det
        theta1_ddot = (c01 * f0 + c11 * f1 + c12 * f2) / det
        theta2_ddot = (c02 * f0 + c12 * f1 + c22 * f2) / det

        return np.array([x_dot, x_ddot, theta1_dot, theta1_ddot, phi_dot, theta2_ddot - theta1_ddot])

    def initial_state(self, pitch: float = 0.0, pivot: float = 0.0) -> np.ndarray:
        """Lower body tilted by `pitch`, pivot at `pivot` (rad), at rest."""
        return np.array([0.0, 0.0, pitch, 0.0, pivot, 0.0])

    def is_fallen(self, state: np.ndarray) -> bool:
        return abs(state[2]) > self.fall_angle or abs(state[2] + state[4]) > self.fall_angle

    def com_pitch(self, state: Sequence[float]) -> Tuple[float, float]:
        """(angle, rate) of the combined COM from vertical; see PivotRobotParams.com_pitch."""
        return self.p.com_pitch(float(state[2]), float(state[4]), float(state[3]), float(state[5]))

    def energy(self, state: Sequence[float]) -> float:
        """Total mechanical energy (kinetic + potential), for tests."""
        x_dot, theta1, theta1_dot = state[1], state[2], state[3]
        phi, phi_dot = state[4], state[5]
        theta2, theta2_dot = theta1 + phi, theta1_dot + phi_dot
        a, b1, b2, c, d1, d2 = self._a, self._b1, self._b2, self._c, self._d1, self._d2
        kinetic = (0.5 * a * x_dot**2
                   + b1 * math.cos(theta1) * x_dot * theta1_dot
                   + b2 * math.cos(theta2) * x_dot * theta2_dot
                   + 0.5 * d1 * theta1_dot**2 + 0.5 * d2 * theta2_dot**2
                   + c * math.cos(phi) * theta1_dot * theta2_dot)
        potential = self.p.gravity * (b1 * math.cos(theta1) + b2 * math.cos(theta2))
        return kinetic + potential

    def linearize(self, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """(A, B) of the model linearized about upright and at rest, with the
        pivot straight: d(state)/dt ~= A state + B [wheel torque, servo command].

        Central differences on derivatives(), so the wheel motor and servo
        models (back-EMF, servo damping and stiffness) are included exactly as
        simulated, as long as neither is saturated at the operating point.
        """
        x0, u0 = np.zeros(self.state_dim), np.zeros(2)
        A = np.zeros((self.state_dim, self.state_dim))
        B = np.zeros((self.state_dim, 2))
        for i in range(self.state_dim):
            dx = np.zeros(self.state_dim)
            dx[i] = eps
            A[:, i] = (self.derivatives(x0 + dx, u0) - self.derivatives(x0 - dx, u0)) / (2 * eps)
        for j in range(2):
            du = np.zeros(2)
            du[j] = eps
            B[:, j] = (self.derivatives(x0, u0 + du) - self.derivatives(x0, u0 - du)) / (2 * eps)
        return A, B
