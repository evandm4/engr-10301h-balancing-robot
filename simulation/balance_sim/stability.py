"""Exact stability of the sampled, delayed linear loop, for either controller.

The analysis of PHYSICS.md 7.4, generalized: the linear model (linearize)
with its axle position dropped, each command held for a control period and
reaching the plant `latency` seconds after its tick. Both controllers are
linear in that range, of the form

    u_k     = state_gain . s_k + integral_gain * I_k + sum_j history_gain[j-1] * u_{k-j}
    I_{k+1} = I_k + control_dt * integrand . s_k

with s = (velocity, pitch, pitch rate) and a goal of 0. Stacking the plant,
the integral and the commands in flight gives z_{k+1} = M z_k, and the loop is
stable exactly when every eigenvalue of M is inside the unit circle.

Exact for the linear model; it leaves out saturation (torque, back-EMF, the
lean cap), the integral caps and sensor noise.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np

from .params import RobotParams
from .planar2d import linearize


@dataclass(frozen=True)
class LinearController:
    """A controller in the form of the module docstring. Build one with lqr() or cascaded_pid()."""

    state_gain: Tuple[float, float, float]
    integral_gain: float
    integrand: Tuple[float, float, float]
    history_gain: Tuple[float, ...] = ()     # for u_{k-1}, u_{k-2}, ...

    @classmethod
    def lqr(cls, gain: Sequence[float]) -> "LinearController":
        """An LQRController's gain (balance_sim.lqr layout). Its integral is
        used before the tick adds the current error, as I_k above."""
        k = np.asarray(gain, dtype=float)
        in_flight = k[4:]                      # oldest first, so u_{k-1} is the last
        return cls(tuple(-k[:3]), float(-k[3]), (1.0, 0.0, 0.0), tuple(-in_flight[::-1]))

    @classmethod
    def cascaded_pid(cls, gains: Sequence[float], control_dt: float) -> "LinearController":
        """CascadedPIDController gains [kp, ki, kd, kv_p] (PHYSICS.md 7.4). Its
        integrator adds the current error before use, so with I_k = the
        integral before tick k: u_k = (kp + ki T) e_k + ki I_k + kd theta_dot,
        e_k = theta + kv_p v."""
        kp, ki, kd, kv = (float(g) for g in gains[:4])
        p = kp + ki * control_dt
        return cls((p * kv, p, kd), ki, (kv, 1.0, 0.0))


@lru_cache(maxsize=64)
def _plant(params: RobotParams, control_dt: float, latency: float):
    """Phi and {age: input map}: a command issued `age` ticks ago drives the plant this period."""
    from scipy.linalg import expm

    A4, B4 = linearize(params)
    A, B = A4[1:, 1:], B4[1:]

    def hold(seconds):
        block = expm(np.block([[A, B], [np.zeros((1, 4))]]) * seconds)
        return block[:3, :3], block[:3, 3]

    T = control_dt
    q = math.floor(latency / T + 1e-9)
    eps = latency - q * T
    if eps <= 1e-12:
        return hold(T)[0], {q: hold(T)[1]}
    late_state, late_input = hold(T - eps)
    return hold(T)[0], {q: late_input, q + 1: late_state @ hold(eps)[1]}


def loop_matrix(params: RobotParams, controller: LinearController, control_dt: float, latency: float) -> np.ndarray:
    """The closed-loop transition matrix M (see the module docstring).

    State: s (3), then I if the integral gain is nonzero (otherwise the
    integral never acts and is left out), then u_{k-1}, ..., u_{k-H}.
    """
    Phi, inputs = _plant(params, control_dt, latency)
    uses_integral = controller.integral_gain != 0.0
    base = 3 + uses_integral
    H = max(max(inputs), len(controller.history_gain))
    n = base + H
    u_row = np.zeros(n)
    u_row[:3] = controller.state_gain
    if uses_integral:
        u_row[3] = controller.integral_gain
    for j, g in enumerate(controller.history_gain, start=1):
        u_row[base + j - 1] += g

    M = np.zeros((n, n))
    M[:3, :3] = Phi
    for age, G in inputs.items():
        if age == 0:
            M[:3] += np.outer(G, u_row)
        else:
            M[:3, base + age - 1] += G
    if uses_integral:
        M[3, :3] = control_dt * np.asarray(controller.integrand)
        M[3, 3] = 1.0
    if H:
        M[base] = u_row                                 # u_k becomes the newest command in flight
        M[base + 1:, base:n - 1] = np.eye(H - 1)        # the others age by a tick
    return M


def spectral_radius(params: RobotParams, controller: LinearController, control_dt: float, latency: float) -> float:
    """Largest |eigenvalue| of the loop: below 1 means stable."""
    return float(np.max(np.abs(np.linalg.eigvals(loop_matrix(params, controller, control_dt, latency)))))


def stable_latency_range(params: RobotParams, controller: LinearController, control_dt: float,
                         around: float = 0.0, max_latency: float = 0.08,
                         step: float = 0.00025) -> Optional[Tuple[float, float]]:
    """The interval of latencies, on a `step` grid, over which the loop stays
    stable, growing out from `around` (e.g. the latency a controller was
    designed for). None if it's unstable at `around` itself. An end at 0 or
    at max_latency means stable all the way to that end of the search.

    The upper end is the delay margin of PHYSICS.md 7.4. The lower end
    matters for a controller that predicts its commands in flight: if the
    real latency is shorter than designed, the prediction is wrong too.
    """
    stable = lambda lat: spectral_radius(params, controller, control_dt, lat) < 1.0
    if not stable(around):
        return None
    grid = np.round(np.arange(0.0, max_latency + step / 2, step), 10)
    start = int(round(around / step))
    low = start
    while low > 0 and stable(grid[low - 1]):
        low -= 1
    high = start
    while high + 1 < len(grid) and stable(grid[high + 1]):
        high += 1
    return float(grid[low]), float(grid[high])
