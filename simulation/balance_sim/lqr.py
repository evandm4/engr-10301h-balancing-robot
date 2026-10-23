"""LQR: full-state feedback designed on the linearized robot.

    LQRWeights      the quadratic cost the design minimizes
    lqr_gain        the gain: discrete time, with the latency built in
    LQRController   torque = -K z on the measured state, clipped to the torque limit

In its linear range the cascaded PID is already state feedback, with a fixed
structure: its torque is kp theta + kd theta_dot + kp kv_p (v - goal) plus two
integral terms, five gains set by four numbers (LQR.md). LQR drops the
structure and picks the gain that minimizes a quadratic cost on the linear
model instead.

The design (the same scheme as pivot_sim.lqr_gain, for the base robot):
  * Plant state [velocity error, pitch, pitch rate], from linearize() with the
    axle position dropped (nothing depends on it), plus the integral of
    velocity error when it is weighted, summed once per tick as the
    controller does.
  * Discretized exactly with a zero-order hold at control_dt, since the
    controller holds each command for one control period.
  * The latency is built in exactly. Write it as q control periods plus a
    remainder eps. The commands issued in the last q ticks (q + 1 if eps > 0)
    are still on their way to the motor, so they become part of the state, and
    the gain feeds them back too. latency = 0 gives the textbook design.
  * The cost per second is summed once per control period (Q dt, R dt).

Not built in: sensor noise, torque saturation, the motor's back-EMF limit,
large-angle nonlinearity, and the robot-to-robot spread. Those are what the
simulations test.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from .controllers import Controller, _clip
from .params import RobotParams
from .planar2d import linearize

# Layout of a gain vector, shared with batch_sim.lqr_update: these four, then
# one gain per command in flight, oldest first.
GAIN_LAYOUT = ("velocity", "pitch", "pitch_rate", "velocity_integral")
N_STATE_GAINS = len(GAIN_LAYOUT)


@dataclass(frozen=True)
class LQRWeights:
    """Weights of the cost the LQR minimizes, per second:

        velocity * (velocity - goal)^2 + pitch * theta^2 + pitch_rate * theta_dot^2
      + velocity_integral * (integral of velocity error)^2 + torque * torque^2

    Pitch is from upright. With pitch = 1, velocity = CostConfig.velocity_weight
    and torque = CostConfig.effort_weight this is the project's own cost
    (pitch_reference "upright"), which is the "cost-matched" design.
    Only the ratios matter: scaling every weight together gives the same gain.

    velocity must be positive (otherwise nothing ties the robot to its goal).
    velocity_integral = 0 leaves the integral out (no integral action).
    """

    velocity: float
    pitch: float
    pitch_rate: float
    torque: float
    velocity_integral: float = 0.0

    def __post_init__(self):
        if self.velocity <= 0 or self.pitch <= 0 or self.torque <= 0:
            raise ValueError("velocity, pitch and torque weights must be positive")
        if self.pitch_rate < 0 or self.velocity_integral < 0:
            raise ValueError("pitch_rate and velocity_integral weights must be non-negative")


def delayed_commands(control_dt: float, latency: float) -> int:
    """How many past commands are still in flight at each tick: q, or q + 1
    when the latency isn't a whole number of control periods."""
    if latency < 0 or control_dt <= 0:
        raise ValueError("latency must be non-negative and control_dt positive")
    q = math.floor(latency / control_dt + 1e-9)
    return q + int(latency - q * control_dt > 1e-12)


def discrete_model(params: RobotParams, control_dt: float, latency: float, integral: bool = False):
    """The model the LQR is designed on: z_{k+1} = Ad z_k + Bd u_k, exact for the linear robot.

    z = [velocity, pitch, pitch rate, (integral of velocity if `integral`),
    then the commands in flight, oldest first]; u_k is the command issued at
    tick k, held for one control period, reaching the plant `latency` later.
    The integral is summed once per tick, I_{k+1} = I_k + control_dt * v_k,
    exactly as LQRController does. Returns (Ad, Bd, n_plant), n_plant being
    the number of entries before the commands in flight.
    """
    from scipy.linalg import expm

    A4, B4 = linearize(params)
    A, B = A4[1:, 1:], B4[1:]          # drop x: [x_dot, theta, theta_dot]
    n = 3 + int(integral)

    def hold(seconds):
        """(e^{A t}, integral_0^t e^{A s} ds B): the plant's state and input maps for a constant input."""
        block = expm(np.block([[A, B], [np.zeros((1, 4))]]) * seconds)
        return block[:3, :3], block[:3, 3:]

    T = control_dt
    m = delayed_commands(T, latency)
    q = math.floor(latency / T + 1e-9)
    eps = latency - q * T if m > q else 0.0
    Phi, Gamma = hold(T)
    # (age, input map): a command issued `age` ticks ago drives the plant this period.
    if eps == 0.0:
        acting = [(q, Gamma)]
    else:
        # The older command still acts for the first eps, the newer for the rest (PHYSICS.md 7.4).
        late_state, late_input = hold(T - eps)
        acting = [(q + 1, late_state @ hold(eps)[1]), (q, late_input)]

    # Augmented state: plant, then the m commands in flight, oldest first
    # (a command of age j sits at index n + m - j).
    Ad = np.zeros((n + m, n + m))
    Bd = np.zeros((n + m, 1))
    Ad[:3, :3] = Phi
    if integral:
        Ad[3, 0], Ad[3, 3] = T, 1.0
    for age, G in acting:
        if age == 0:
            Bd[:3] += G
        else:
            Ad[:3, n + m - age] += G[:, 0]
    if m:
        Ad[n:n + m - 1, n + 1:] = np.eye(m - 1)   # every command in flight ages by one tick
        Bd[n + m - 1, 0] = 1.0                    # and the new one joins the back of the queue
    return Ad, Bd, n


def lqr_gain(params: RobotParams, weights: LQRWeights, control_dt: float, latency: float) -> np.ndarray:
    """The LQR gain for LQRController, designed for this robot, control rate and latency.

    Returns K with GAIN_LAYOUT's four entries (the integral's is 0 when it
    isn't weighted), then one per command in flight, oldest first.
    Raises numpy.linalg.LinAlgError (or ValueError) if the Riccati equation
    has no stabilizing solution for these weights.
    """
    from scipy.linalg import solve_discrete_are

    uses_integral = weights.velocity_integral > 0
    Ad, Bd, n = discrete_model(params, control_dt, latency, uses_integral)
    m = Ad.shape[0] - n
    q_diag = [weights.velocity, weights.pitch, weights.pitch_rate]
    if uses_integral:
        q_diag.append(weights.velocity_integral)
    T = control_dt
    Qd = np.zeros((n + m, n + m))
    Qd[:n, :n] = np.diag(q_diag) * T
    Rd = np.array([[weights.torque * T]])
    P = solve_discrete_are(Ad, Bd, Qd, Rd)
    K = np.linalg.solve(Rd + Bd.T @ P @ Bd, Bd.T @ P @ Ad)[0]
    if not np.all(np.isfinite(K)):
        raise np.linalg.LinAlgError("LQR design produced a non-finite gain")
    integral_gain = K[3] if uses_integral else 0.0
    return np.concatenate([K[:3], [integral_gain], K[n:]])


Goal = Union[float, Callable[[float], float]]


class LQRController(Controller):
    """torque = -K z, clipped to the torque limit, with

        z = [velocity - goal, pitch, pitch rate, integral of velocity error,
             the commands still in flight, oldest first]

    K comes from lqr_gain (or anywhere else, in GAIN_LAYOUT order). The
    controller remembers its last len(K) - 4 commands, after clipping: that is
    what the motor will get, and the design counts them as part of the state.
    The integral is capped at +-integral_limit (m); its gain is 0 unless the
    design weighted it.

    It has no lean command, so pitch_reference is upright (0): with the cost's
    "commanded" pitch reference, its pitch error is still measured from
    vertical. Compare it with the cascaded PID on pitch_reference "upright".

    `goal` is a constant (m/s) or a function of time, as for
    CascadedPIDController; the controller keeps its own clock, restarted on reset().
    """

    def __init__(self, gain, torque_limit: float, goal: Goal = 0.0, integral_limit: float = math.inf):
        self.K = np.asarray(gain, dtype=float)
        if self.K.ndim != 1 or len(self.K) < N_STATE_GAINS:
            raise ValueError(f"gain must be a vector of at least {N_STATE_GAINS} entries, got shape {self.K.shape}")
        if torque_limit <= 0:
            raise ValueError("torque_limit must be positive")
        self.torque_limit = torque_limit
        self.goal = goal
        self.integral_limit = integral_limit
        self.n_delayed = len(self.K) - N_STATE_GAINS
        self.reset()

    @classmethod
    def design(cls, params: RobotParams, weights: LQRWeights, control_dt: float, latency: float,
               **kwargs) -> "LQRController":
        """Compute the gain for this robot and timing, and build the controller.
        kwargs (goal, integral_limit) go to the constructor; the torque limit is the robot's."""
        return cls(lqr_gain(params, weights, control_dt, latency), params.max_torque, **kwargs)

    def goal_velocity(self, t: float) -> float:
        return float(self.goal(t)) if callable(self.goal) else float(self.goal)

    def reset(self) -> None:
        self._integral = 0.0
        self._time = 0.0
        self._in_flight = deque([0.0] * self.n_delayed, maxlen=self.n_delayed)

    def update(self, state: np.ndarray, dt: float) -> float:
        error = float(state[1]) - self.goal_velocity(self._time)
        k = self.K
        # Summed in a fixed order, the same as batch_sim.lqr_update, so the two agree to the bit.
        total = k[0] * error + k[1] * float(state[2]) + k[2] * float(state[3]) + k[3] * self._integral
        for gain, command in zip(k[N_STATE_GAINS:], self._in_flight):
            total += gain * command
        torque = _clip(-float(total), -self.torque_limit, self.torque_limit)
        self._integral = _clip(self._integral + error * dt, -self.integral_limit, self.integral_limit)
        self._in_flight.append(torque)
        self._time += dt
        return torque
