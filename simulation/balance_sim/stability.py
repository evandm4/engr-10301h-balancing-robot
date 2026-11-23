"""Exact stability of the sampled, delayed linear loop, for either controller.

The analysis of PHYSICS.md 7.4, generalized: the linear model (linearize)
with its axle position dropped, each command held for a control period and
reaching the plant `latency` seconds after its tick. Both controllers are
linear in that range, of the form

    u_k     = state_gain . s_k + sum_m integral_gain[m] * I_k[m] + sum_j history_gain[j-1] * u_{k-j}
    I_{k+1}[m] = integral_decay[m] * I_k[m] + control_dt * integrand[m] . s_k

with s = (velocity, pitch, pitch rate) and a goal of 0. The PID and the LQR have one
integrator with decay 1; a learned linear policy (linear_policy) has two leaky ones. Stacking the plant,
the integral and the commands in flight gives z_{k+1} = M z_k, and the loop is
stable exactly when every eigenvalue of M is inside the unit circle.

Exact for the linear model; it leaves out saturation (torque, back-EMF, the
lean cap), the integral caps and sensor noise.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .params import RobotParams
from .planar2d import linearize


@dataclass(frozen=True)
class LinearController:
    """A controller in the form of the module docstring. Build one with lqr(), cascaded_pid()
    or linear_policy().

    integral_gain, integrand and integral_decay describe one integrator (a number, a 3-tuple,
    a number) or several (a tuple of numbers, a tuple of 3-tuples, a tuple of numbers).
    """

    state_gain: Tuple[float, float, float]
    integral_gain: Union[float, Tuple[float, ...]]
    integrand: Union[Tuple[float, float, float], Tuple[Tuple[float, float, float], ...]]
    history_gain: Tuple[float, ...] = ()     # for u_{k-1}, u_{k-2}, ...
    integral_decay: Union[float, Tuple[float, ...]] = 1.0

    def integrators(self) -> List[Tuple[float, np.ndarray, float]]:
        """(gain, integrand (3,), decay) per integrator whose gain is nonzero (the others
        never act). Integrators sharing a decay are merged into one, gain 1 and integrand
        sum(gain * integrand): u only ever sees that sum, so the merged loop has the same
        coupled modes and none of the uncoupled ones (each would be an eigenvalue equal to
        its decay, 1 for a plain integrator, which would swamp the spectral radius)."""
        gains = np.atleast_1d(np.asarray(self.integral_gain, dtype=float))
        integrands = np.atleast_2d(np.asarray(self.integrand, dtype=float))
        decays = np.broadcast_to(np.asarray(self.integral_decay, dtype=float), gains.shape)
        if integrands.shape != (len(gains), 3):
            raise ValueError("need one 3-element integrand per integral gain")
        merged = {}
        for g, c, d in zip(gains, integrands, decays):
            if g != 0.0:
                merged.setdefault(float(d), []).append((g, c))
        if all(len(v) == 1 for v in merged.values()):    # nothing to merge: keep the gains as given
            return [(float(v[0][0]), v[0][1], d) for d, v in merged.items()]
        return [(1.0, sum(g * c for g, c in v), d) for d, v in merged.items()]

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

    @classmethod
    def linear_policy(cls, spec, params, control_dt: float) -> "LinearController":
        """A linear-only balance_sim.PolicySpec with its weights, unsaturated (no clip, no
        integrator freeze). Exact: L has no nonlinearity below saturation.

        torque = sum_i c_i raw_i with c_i = output_scale * w_i / scale_i. With the goal 0:
        goal_minus_v = -v; int_v integrates -v and int_theta integrates theta, leaky, each
        updated before use: I_used = d J_k + T e_k with J_k the integral before tick k and
        d = 1 - T / T_leak, so the state gain picks up T c_I e and the integral gain is c_I d.
        cmd_j is u_{k-j}.
        """
        from .policy import N_OBS, OBSERVATIONS, unpack

        if spec.hidden or not spec.linear_path or spec.residual:
            raise ValueError("linear_policy needs a linear-only, non-residual spec")
        linear, _ = unpack(spec, np.asarray(params, dtype=float))
        c = np.zeros(N_OBS)
        for w, i in zip(linear, spec.inputs):
            c[i] = spec.output_scale * float(w) / spec.scales[i]
        k = {name: c[i] for i, name in enumerate(OBSERVATIONS)}
        T, d = control_dt, 1.0 - control_dt * spec.leak_rate
        e_v, e_theta = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        state = np.array([k["v_meas"] - k["goal_minus_v"], k["theta_meas"], k["theta_dot_meas"]])
        state = state + T * (k["int_v"] * e_v + k["int_theta"] * e_theta)
        history = tuple(k[f"cmd_{j}"] for j in range(1, 5))
        while history and history[-1] == 0.0:
            history = history[:-1]
        return cls(tuple(state), (k["int_v"] * d, k["int_theta"] * d), (tuple(e_v), tuple(e_theta)),
                   history, (d, d))


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

    State: s (3), then the integrators (LinearController.integrators: those with a nonzero
    gain, merged where they share a decay), then u_{k-1}, ..., u_{k-H}.
    """
    Phi, inputs = _plant(params, control_dt, latency)
    integrators = controller.integrators()
    n_int = len(integrators)
    base = 3 + n_int
    H = max(max(inputs), len(controller.history_gain))
    n = base + H
    u_row = np.zeros(n)
    u_row[:3] = controller.state_gain
    for m, (g, _, _) in enumerate(integrators):
        u_row[3 + m] = g
    for j, g in enumerate(controller.history_gain, start=1):
        u_row[base + j - 1] += g

    M = np.zeros((n, n))
    M[:3, :3] = Phi
    for age, G in inputs.items():
        if age == 0:
            M[:3] += np.outer(G, u_row)
        else:
            M[:3, base + age - 1] += G
    for m, (_, integrand, decay) in enumerate(integrators):
        M[3 + m, :3] = control_dt * integrand
        M[3 + m, 3 + m] = decay
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
