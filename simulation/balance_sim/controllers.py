"""The controller interface and a PID balance controller."""

import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np


def _clip(x: float, low: float, high: float) -> float:
    return low if x < low else high if x > high else x


def _limited_integral(
    integral: float,
    error: float,
    dt: float,
    other_terms: float,
    gain: float,
    integral_limit: float,
    output_limit: Optional[float],
) -> float:
    """Advance an integrator with anti-windup, for output = other_terms + gain * integral.

    Three guards, applied in order:
      1. The integral itself is capped at +-integral_limit.
      2. Conditional integration: if this step's error would push the output
         past its limit, the integrator only grows far enough to bring the
         output exactly to the limit (or holds, if it's already there). It
         still unwinds freely when the error reverses.
      3. The integral term alone may never ask for more than the output limit
         (|gain * integral| <= output_limit).
    Guards 2 and 3 only apply when output_limit is given.
    """
    new = _clip(integral + error * dt, -integral_limit, integral_limit)
    if output_limit is None:
        return new
    output = other_terms + gain * new
    if abs(output) > output_limit and error * output > 0:
        if gain > 0:
            # Integrator value that puts the output exactly on the limit. Move
            # toward it, never past the candidate step, and never backwards.
            target = (math.copysign(output_limit, output) - other_terms) / gain
            step, allowed = new - integral, target - integral
            new = integral + (min(step, max(allowed, 0.0)) if step > 0 else max(step, min(allowed, 0.0)))
        else:
            new = integral
    if gain > 0:
        cap = output_limit / gain
        new = _clip(new, -cap, cap)
    return new


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

    def pitch_reference(self) -> float:
        """Pitch (rad) the controller is currently trying to hold.

        The simulator records this after every update so the cost function
        can score pitch against what was asked for rather than against
        upright. Controllers that always aim for upright return 0.
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

    output_limit (N m): the most torque the motors can deliver. When given, the
    returned command is clipped to it and the integrator is protected against
    windup (see _limited_integral), so a large Ki can't bank up torque the
    motors will never deliver. Without it, only integral_limit applies.

    integral_limit caps the integral itself (rad s). The default is no cap;
    the project's value is ControllerConfig.pitch_integral_limit in config.py.

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
        integral_limit: float = math.inf,
        output_limit: Optional[float] = None,
    ):
        if output_limit is not None and output_limit <= 0:
            raise ValueError("output_limit must be positive")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self._integral = 0.0

    @classmethod
    def from_gains(cls, gains, **kwargs) -> "PIDController":
        """Build from a sequence [kp, ki, kd], the form the ML tuner searches over."""
        kp, ki, kd = gains
        return cls(kp, ki, kd, **kwargs)

    def reset(self) -> None:
        self._integral = 0.0

    def pitch_reference(self) -> float:
        return float(self.setpoint)

    def update(self, state: np.ndarray, dt: float) -> float:
        theta, theta_dot = float(state[2]), float(state[3])
        error = theta - self.setpoint
        proportional_derivative = self.kp * error + self.kd * theta_dot
        self._integral = _limited_integral(
            self._integral, error, dt, proportional_derivative, self.ki,
            self.integral_limit, self.output_limit,
        )
        torque = proportional_derivative + self.ki * self._integral
        if self.output_limit is not None:
            torque = _clip(torque, -self.output_limit, self.output_limit)
        return torque


class CascadedPIDController(Controller):
    """Two nested loops: a slow velocity loop on the outside, a fast pitch PID inside.

        velocity error --(PI)--> pitch setpoint --(pitch PID)--> torque

    To speed up, a balancing robot has to lean forward first, so the outer loop
    doesn't command torque directly. It commands a lean angle, and the inner
    pitch loop (the PIDController you already have) tracks it.

    e_v = goal - measured velocity
    pitch_setpoint = kv_p * e_v + kv_i * integral(e_v), limited to +-max_pitch_command

    The project tunes kv_i out (it always came out 0, so the outer loop is
    proportional only), but the controller still supports it.

    The velocity integrator has the same anti-windup as the pitch loop, with
    max_pitch_command as its output limit: it stops growing while the lean
    command is pinned at the cap. max_pitch_command and integral_limit default
    to no cap; the project's values are in config.py (ControllerConfig).

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
        max_pitch_command: float = math.inf,
        integral_limit: float = math.inf,
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
    def from_gains(cls, gains, torque_limit: Optional[float] = None,
                   pitch_integral_limit: float = math.inf, **kwargs) -> "CascadedPIDController":
        """Build from [kp, ki, kd, kv_p] (the project's gain set) or [kp, ki, kd, kv_p, kv_i].

        Inner pitch gains first, then outer velocity gains; kv_i is 0 when
        left out. torque_limit and pitch_integral_limit go to the inner pitch
        loop; other keyword arguments (goal, max_pitch_command,
        integral_limit for the velocity loop) to this controller.
        """
        if len(gains) not in (4, 5):
            raise ValueError(f"expected 4 or 5 gains, got {len(gains)}")
        kp, ki, kd, kv_p, *rest = gains
        kv_i = rest[0] if rest else 0.0
        inner = PIDController(kp, ki, kd, integral_limit=pitch_integral_limit, output_limit=torque_limit)
        return cls(inner, kv_p, kv_i, **kwargs)

    def goal_velocity(self, t: float) -> float:
        return float(self.goal(t)) if callable(self.goal) else float(self.goal)

    def pitch_reference(self) -> float:
        return self.pitch_pid.pitch_reference()

    def reset(self) -> None:
        self.pitch_pid.reset()
        self.pitch_pid.setpoint = 0.0
        self._integral = 0.0
        self._time = 0.0

    def update(self, state: np.ndarray, dt: float) -> float:
        velocity_error = self.goal_velocity(self._time) - float(state[1])
        proportional = self.kv_p * velocity_error
        self._integral = _limited_integral(
            self._integral, velocity_error, dt, proportional, self.kv_i,
            self.integral_limit, self.max_pitch_command,
        )
        lean = proportional + self.kv_i * self._integral
        self.pitch_pid.setpoint = _clip(lean, -self.max_pitch_command, self.max_pitch_command)
        self._time += dt
        return self.pitch_pid.update(state, dt)
