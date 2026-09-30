"""The controller interface and a PID balance controller."""

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Tuple, Union

import numpy as np


class Controller(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear internal memory (integrators etc.) before a new run."""

    @abstractmethod
    def update(self, state: np.ndarray, dt: float) -> float:
        """Return the motor torque to command, given the current state."""

    def goal_velocity(self, t: float) -> float:
        """Forward velocity (m/s) this controller is trying to hold at time t.

        The simulator records this so the cost function can score velocity
        tracking. Controllers with no velocity goal want to stand still.
        """
        return 0.0


def step_profile(steps: Sequence[Tuple[float, float]]) -> Callable[[float], float]:
    """Build a piecewise-constant goal velocity from (start_time, velocity) pairs.

    Example: step_profile([(1.0, 0.3), (5.0, -0.2)]) is 0 until t=1 s, then
    0.3 m/s until t=5 s, then -0.2 m/s. Before the first start time it is 0.
    """
    ordered = sorted(steps)

    def profile(t: float) -> float:
        velocity = 0.0
        for start, v in ordered:
            if t >= start - 1e-9:
                velocity = v
            else:
                break
        return velocity

    return profile


class PIDController(Controller):
    """PID on body pitch.

    torque = kp * e + ki * integral(e) + kd * theta_dot,   e = theta - setpoint

    Sign convention: leaning forward (theta > 0) needs positive torque, which
    drives the wheels forward underneath the body. The derivative term uses the
    measured pitch rate directly rather than differentiating the error, which
    avoids derivative kick and noise amplification from numerical differencing.

    Note: a pitch-only PID keeps the robot upright but does not control
    position or speed, so the robot may drift. For velocity control, wrap it in
    a CascadedPIDController.
    """

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        setpoint: float = 0.0,
        integral_limit: float = 1.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral_limit = integral_limit
        self._integral = 0.0

    @classmethod
    def from_gains(cls, gains, **kwargs) -> "PIDController":
        """Build from a sequence [kp, ki, kd], the form the ML tuner searches over."""
        kp, ki, kd = gains
        return cls(kp, ki, kd, **kwargs)

    def reset(self) -> None:
        self._integral = 0.0

    def update(self, state: np.ndarray, dt: float) -> float:
        theta, theta_dot = state[2], state[3]
        error = theta - self.setpoint
        self._integral += error * dt
        self._integral = float(
            np.clip(self._integral, -self.integral_limit, self.integral_limit)
        )
        return self.kp * error + self.ki * self._integral + self.kd * theta_dot


class CascadedPIDController(Controller):
    """Two nested loops: a slow velocity loop on the outside, a fast pitch PID inside.

        velocity error --(PI)--> pitch setpoint --(pitch PID)--> torque

    To speed up, a balancing robot has to lean forward first, so the outer loop
    doesn't command torque directly. It commands a lean angle, and the inner
    pitch loop (the PIDController you already have) tracks it.

    e_v = goal - measured velocity
    pitch_setpoint = kv_p * e_v + kv_i * integral(e_v), limited to +-max_pitch_command

    Tuning notes:
      * The outer loop must be much slower than the inner one. If it isn't, the
        two fight and the robot oscillates.
      * The limit on the commanded lean angle keeps a big velocity error from
        asking for a lean the robot can't recover from.
      * kv_p and kv_i have units of radians per (m/s) and radians per m.

    `goal` is either a constant (m/s) or a function of time, e.g. one built with
    step_profile(). The controller keeps its own clock, which restarts on reset().
    """

    def __init__(
        self,
        pitch_pid: PIDController,
        kv_p: float,
        kv_i: float = 0.0,
        goal: Union[float, Callable[[float], float]] = 0.0,
        max_pitch_command: float = 0.2,
        integral_limit: float = 0.5,
    ):
        self.pitch_pid = pitch_pid
        self.kv_p = kv_p
        self.kv_i = kv_i
        self.goal = goal
        self.max_pitch_command = max_pitch_command
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._time = 0.0

    @classmethod
    def from_gains(cls, gains, **kwargs) -> "CascadedPIDController":
        """Build from [kp, ki, kd, kv_p, kv_i]: inner pitch gains, then outer velocity gains."""
        kp, ki, kd, kv_p, kv_i = gains
        return cls(PIDController(kp, ki, kd), kv_p, kv_i, **kwargs)

    def goal_velocity(self, t: float) -> float:
        return float(self.goal(t)) if callable(self.goal) else float(self.goal)

    def reset(self) -> None:
        self.pitch_pid.reset()
        self._integral = 0.0
        self._time = 0.0

    def update(self, state: np.ndarray, dt: float) -> float:
        velocity_error = self.goal_velocity(self._time) - state[1]
        self._integral += velocity_error * dt
        self._integral = float(
            np.clip(self._integral, -self.integral_limit, self.integral_limit)
        )
        lean = self.kv_p * velocity_error + self.kv_i * self._integral
        self.pitch_pid.setpoint = float(
            np.clip(lean, -self.max_pitch_command, self.max_pitch_command)
        )
        self._time += dt
        return self.pitch_pid.update(state, dt)
