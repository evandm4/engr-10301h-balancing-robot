"""Controllers for the pivot robot. Each returns two commands per update:
(wheel torque in N m, servo angle in rad).

    PivotCascadedController  the base project's cascaded PID on the wheels, plus
                             a PD-style pivot loop. Tunable gains, same style
                             as the base robot; the ML tuner searches these.
    PivotLQRController       full-state feedback from an LQR design on the
                             linearized model, delay-aware. The model-based
                             "optimal" benchmark for the tuned PID to beat.
"""

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from balance_sim import CascadedPIDController

from .dynamics import PivotRobot
from .params import PivotRobotParams

Goal = Union[float, Callable[[float], float]]


def _clip(x: float, low: float, high: float) -> float:
    return low if x < low else high if x > high else x


def _goal_at(goal: Goal, t: float) -> float:
    return float(goal(t)) if callable(goal) else float(goal)


class PivotController(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear internal memory before a new run."""

    @abstractmethod
    def update(self, state: np.ndarray, dt: float) -> Tuple[float, float]:
        """Return (wheel torque command, servo angle command) for this state."""

    def goal_velocity(self, t: float) -> float:
        return 0.0

    def pitch_reference(self) -> float:
        """Pitch (rad) of the combined COM the controller is asking for; the
        cost measures pitch error against it (see balance_sim.pitch_error_signal)."""
        return 0.0


# ---- cascaded PID + pivot loop ----------------------------------------------

PIVOT_GAIN_NAMES = ("k_level", "k_phi_p", "k_phi_d")


class PivotCascadedController(PivotController):
    """Wheels: the base robot's CascadedPIDController, unchanged, balancing the
    combined center of mass. Pivot: a second, PD-style loop.

    Wheel loop. The outer velocity loop asks for a lean; the inner pitch PID
    turns the lean error into wheel torque. What it balances is the angle of
    the combined COM over the axle (pitch_source="com", computed from the
    lower body's IMU and the pivot angle), because with the upper body free to
    move, the lower body alone being upright no longer means balanced.
    pitch_source="lower" balances the lower body's pitch instead.

    Pivot loop:
        phi_cmd = -k_level * theta1  +  k_phi_p * e  +  k_phi_d * theta_com_dot
        e       = theta_com - lean the wheel loop is asking for
    clipped to +-servo_limit.

      k_level  0..1: how much of the lower body's tilt the upper body cancels.
               1 keeps the upper body vertical (a level payload); 0 holds the
               pivot straight.
      k_phi_p  rad/rad. Throws the upper body in response to pitch error. A
               negative value swings it backward when the robot is too far
               forward, moving the COM back toward the axle; the servo's
               reaction torque pushes the lower body the other way, so whether
               it helps depends on timing, which is what tuning is for.
      k_phi_d  rad/(rad/s), the same on the COM pitch rate.

    With all three at 0 the servo holds the pivot straight and the robot
    behaves as the base robot would with the same wheel gains.
    """

    def __init__(
        self,
        wheel: CascadedPIDController,
        params: PivotRobotParams,
        k_level: float,
        k_phi_p: float,
        k_phi_d: float,
        servo_limit: float,
        pitch_source: str = "com",
    ):
        if pitch_source not in ("com", "lower"):
            raise ValueError(f"pitch_source must be 'com' or 'lower', got {pitch_source!r}")
        self.wheel = wheel
        self.params = params
        self.k_level, self.k_phi_p, self.k_phi_d = k_level, k_phi_p, k_phi_d
        self.servo_limit = servo_limit
        self.pitch_source = pitch_source

    @classmethod
    def from_gains(
        cls,
        gains: Sequence[float],
        params: PivotRobotParams,
        servo_limit: float,
        torque_limit: Optional[float] = None,
        pitch_integral_limit: float = math.inf,
        pitch_source: str = "com",
        **wheel_kwargs,
    ) -> "PivotCascadedController":
        """Build from [kp, ki, kd, kv_p, k_level, k_phi_p, k_phi_d]: the base
        project's four wheel gains, then the three pivot gains. With only the
        four wheel gains, the pivot gains are 0 (pivot held straight).
        wheel_kwargs (goal, max_pitch_command, integral_limit) go to
        CascadedPIDController.from_gains."""
        gains = list(gains)
        if len(gains) not in (4, 7):
            raise ValueError(f"expected 4 or 7 gains, got {len(gains)}")
        wheel_gains, pivot_gains = gains[:4], (gains[4:] or [0.0, 0.0, 0.0])
        wheel = CascadedPIDController.from_gains(
            wheel_gains, torque_limit=torque_limit, pitch_integral_limit=pitch_integral_limit, **wheel_kwargs)
        return cls(wheel, params, *pivot_gains, servo_limit=servo_limit, pitch_source=pitch_source)

    def goal_velocity(self, t: float) -> float:
        return self.wheel.goal_velocity(t)

    def pitch_reference(self) -> float:
        return self.wheel.pitch_reference()

    def reset(self) -> None:
        self.wheel.reset()

    def update(self, state: np.ndarray, dt: float) -> Tuple[float, float]:
        theta1 = float(state[2])
        com, com_rate = self.params.com_pitch(theta1, float(state[4]), float(state[3]), float(state[5]))
        if self.pitch_source == "com":
            wheel_state = (float(state[0]), float(state[1]), com, com_rate)
        else:
            wheel_state = state[:4]
        torque = self.wheel.update(wheel_state, dt)
        error = com - self.wheel.pitch_reference()
        phi = -self.k_level * theta1 + self.k_phi_p * error + self.k_phi_d * com_rate
        return torque, _clip(phi, -self.servo_limit, self.servo_limit)


# ---- LQR ---------------------------------------------------------------------

@dataclass(frozen=True)
class LQRWeights:
    """Quadratic cost the LQR minimizes, per second:

        velocity_integral * (integral of velocity error)^2
      + velocity * (velocity error)^2
      + com_pitch * theta_com^2 + com_pitch_rate * theta_com_dot^2
      + upper_pitch * theta2^2 + pivot * phi^2 + pivot_rate * phi_dot^2
      + torque * wheel torque^2 + servo * servo command^2

    The first and the torque term mirror the project's cost function
    (CostConfig: pitch, velocity, effort), so "optimal" means roughly the same
    thing for the LQR and for the tuner. velocity_integral = 0 drops the
    integral state (no integral action).
    """

    velocity_integral: float
    velocity: float
    com_pitch: float
    com_pitch_rate: float
    upper_pitch: float
    pivot: float
    pivot_rate: float
    torque: float
    servo: float


def lqr_gain(dynamics: PivotRobot, weights: LQRWeights, control_dt: float, latency: float,
             pivot_active: bool = True) -> np.ndarray:
    """Discrete-time LQR gain for PivotLQRController (rows: torque, servo).

    pivot_active=False designs the best controller that leaves the servo at
    0 (its row of the gain is zero): the same optimal design without the
    extra degree of freedom, for measuring what the pivot is worth.

    The model is dynamics.linearize() with the axle position dropped (nothing
    depends on it), plus the integral of velocity error if weighted. It is
    discretized with a zero-order hold at control_dt, and the latency is
    built in by appending the last round(latency / control_dt) commands to the
    state, so the gain accounts for commands already on their way. Not built
    in: the servo's frame rate, sensor noise, and saturation.

    State order: [velocity integral (if used), velocity error, theta1,
    theta1_dot, phi, phi_dot, then the delayed commands, oldest first].
    """
    from scipy.linalg import expm, solve_discrete_are

    A6, B6 = dynamics.linearize()
    A, B = A6[1:, 1:], B6[1:, :]       # drop x: [x_dot, theta1, theta1_dot, phi, phi_dot]
    if not pivot_active:
        B = B.copy()
        B[:, 1] = 0.0                  # the servo command can't move anything: its gain comes out 0
    p = dynamics.p
    share = p.b2 / (p.b1 + p.b2)       # linearized COM pitch = theta1 + share * phi
    outputs = np.array([
        [1, 0, 0, 0, 0],               # velocity error
        [0, 1, 0, share, 0],           # COM pitch
        [0, 0, 1, 0, share],           # COM pitch rate
        [0, 1, 0, 1, 0],               # upper body pitch
        [0, 0, 0, 1, 0],               # pivot angle
        [0, 0, 0, 0, 1],               # pivot rate
    ], dtype=float)
    w = [weights.velocity, weights.com_pitch, weights.com_pitch_rate,
         weights.upper_pitch, weights.pivot, weights.pivot_rate]
    if weights.velocity_integral > 0:
        A = np.block([[np.zeros((1, 1)), np.eye(1, 5)], [np.zeros((5, 1)), A]])
        B = np.vstack([np.zeros((1, 2)), B])
        outputs = np.block([[np.ones((1, 1)), np.zeros((1, 5))], [np.zeros((6, 1)), outputs]])
        w = [weights.velocity_integral] + w
    Q = outputs.T @ np.diag(w) @ outputs
    R = np.diag([weights.torque, weights.servo])

    n = A.shape[0]
    block = expm(np.block([[A, B], [np.zeros((2, n + 2))]]) * control_dt)
    Ad, Bd = block[:n, :n], block[:n, n:]

    delay = int(round(latency / control_dt))
    if delay > 0:
        m = 2 * delay
        Aa = np.zeros((n + m, n + m))
        Aa[:n, :n] = Ad
        Aa[:n, n:n + 2] = Bd                     # oldest pending command acts now
        Aa[n:n + m - 2, n + 2:] = np.eye(m - 2)  # the rest move up the queue
        Ba = np.zeros((n + m, 2))
        Ba[n + m - 2:, :] = np.eye(2)            # the new command joins the back
        Ad, Bd = Aa, Ba
        Q = np.block([[Q, np.zeros((n, m))], [np.zeros((m, n + m))]])

    Qd, Rd = Q * control_dt, R * control_dt
    P = solve_discrete_are(Ad, Bd, Qd, Rd)
    return np.linalg.solve(Rd + Bd.T @ P @ Bd, Bd.T @ P @ Ad)


class PivotLQRController(PivotController):
    """u = -K z, with K from lqr_gain (same control_dt and latency as the
    simulation it runs in). Commands are clipped to the torque and servo
    limits; the clipped values are what it remembers as in flight.

    Its pitch_reference is upright (0): it has no separate lean command.
    """

    def __init__(
        self,
        gain: np.ndarray,
        torque_limit: float,
        servo_limit: float,
        goal: Goal = 0.0,
        integral_limit: float = math.inf,
    ):
        self.K = np.asarray(gain, dtype=float)
        self.torque_limit = torque_limit
        self.servo_limit = servo_limit
        self.goal = goal
        self.integral_limit = integral_limit
        # Width = 5 states + 2 per delayed command (+ 1 for the integral), so
        # its parity says whether the integral state is there.
        extra = self.K.shape[1] - 5
        if self.K.ndim != 2 or self.K.shape[0] != 2 or extra < 0:
            raise ValueError(f"gain has an unexpected shape {self.K.shape}")
        self.uses_integral = bool(extra % 2)
        self.n_delayed = (extra - int(self.uses_integral)) // 2
        self.reset()

    @classmethod
    def design(cls, dynamics: PivotRobot, weights: LQRWeights, control_dt: float, latency: float,
               servo_limit: float, pivot_active: bool = True, **kwargs) -> "PivotLQRController":
        """Compute the gain for this robot and timing, and build the controller."""
        gain = lqr_gain(dynamics, weights, control_dt, latency, pivot_active)
        return cls(gain, dynamics.p.max_torque, servo_limit, **kwargs)

    def goal_velocity(self, t: float) -> float:
        return _goal_at(self.goal, t)

    def reset(self) -> None:
        self._integral = 0.0
        self._time = 0.0
        self._in_flight = deque([(0.0, 0.0)] * self.n_delayed, maxlen=max(self.n_delayed, 1))

    def update(self, state: np.ndarray, dt: float) -> Tuple[float, float]:
        error = float(state[1]) - self.goal_velocity(self._time)
        z = [error, float(state[2]), float(state[3]), float(state[4]), float(state[5])]
        if self.uses_integral:
            z.insert(0, self._integral)
            self._integral = _clip(self._integral + error * dt, -self.integral_limit, self.integral_limit)
        if self.n_delayed:
            z.extend(v for command in self._in_flight for v in command)
        u = -self.K @ np.array(z)
        torque = _clip(float(u[0]), -self.torque_limit, self.torque_limit)
        servo = _clip(float(u[1]), -self.servo_limit, self.servo_limit)
        if self.n_delayed:
            self._in_flight.append((torque, servo))
        self._time += dt
        return torque, servo
