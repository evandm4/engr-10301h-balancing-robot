"""Learned controllers: a linear policy or a small neural network from measurements to torque.

This is the readable, one-robot reference (like CascadedPIDController); batch_sim/policy.py
runs the same arithmetic for many robots and many networks at once, and the tests check
the two agree bit for bit in float64.

What a policy sees each controller tick (OBSERVATIONS, each multiplied by 1 / its scale):

    v_meas, theta_meas, theta_dot_meas    the measured state (same noise as the PID gets)
    goal                                  the current goal velocity
    goal_minus_v                          goal - v_meas
    int_v, int_theta                      leaky integrals of goal - v_meas and of theta_meas
    cmd_1 .. cmd_4                        its own last 4 (clipped) torque commands, newest first

It never sees the true state, the wheel position, its robot's parameters, its latency, or
future goals: the same information the cascaded PID has.

The integrators are leaky, I <- I + dt * (e - I / T_leak), and frozen on any tick where the
previous tick's command was saturated (|pre-clip output| > max_torque): conditional-integration
anti-windup, like the PID's own. Updated before use, as the PID's are. Why both, with numbers:
docs/ml-baselines-plan.md, section 3.1.

Output: torque = clip(output_scale * z, -max_torque, max_torque), where
    z = linear(x) + mlp(x)      (either part may be absent)
    linear(x) = sum_i w_i x_i   (no bias)
    mlp(x)    = W_out tanh(... tanh(W_1 x + b_1) ...) + b_out

Parameters are one flat vector, in this order: the linear path's weights (if any), then for
each MLP layer its weight matrix (row-major, out x in) followed by its bias. Every sum is
done in a fixed order (bias first, then input 0, 1, ...) so the batched code can repeat it
exactly.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .controllers import Controller

N_HISTORY = 4
OBSERVATIONS = ("v_meas", "theta_meas", "theta_dot_meas", "goal", "goal_minus_v", "int_v", "int_theta") + \
    tuple(f"cmd_{i}" for i in range(1, N_HISTORY + 1))
N_OBS = len(OBSERVATIONS)


@dataclass(frozen=True)
class PolicySpec:
    """Shape and fixed settings of a policy; the weights are separate (a flat vector).

    inputs        indices into OBSERVATIONS the policy reads, in this order
    hidden        MLP hidden-layer sizes; () for no MLP
    linear_path   include the bias-free linear term
    scales        one per OBSERVATIONS entry: inputs are raw value / scale
    output_scale  N m per unit of z
    t_leak        integrator leak time constant (s); math.inf for a plain integrator
    integral_limits  (int_v, int_theta) safety clips; far outside the normal range
    """

    inputs: Tuple[int, ...]
    hidden: Tuple[int, ...]
    linear_path: bool
    scales: Tuple[float, ...]
    output_scale: float
    t_leak: float
    integral_limits: Tuple[float, float]

    def __post_init__(self):
        if len(self.scales) != N_OBS or any(not s > 0 for s in self.scales):
            raise ValueError(f"scales must be {N_OBS} positive numbers")
        if not self.inputs or len(set(self.inputs)) != len(self.inputs) or \
                any(not 0 <= i < N_OBS for i in self.inputs):
            raise ValueError(f"inputs must be distinct indices into OBSERVATIONS, got {self.inputs}")
        if not self.linear_path and not self.hidden:
            raise ValueError("a policy needs a linear path, an MLP, or both")
        if not self.t_leak > 0 or not self.output_scale > 0 or any(not l > 0 for l in self.integral_limits):
            raise ValueError("t_leak, output_scale and integral_limits must be positive")

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def layer_shapes(self) -> List[Tuple[int, int]]:
        """(out, in) of every MLP layer, output layer last; [] without an MLP."""
        if not self.hidden:
            return []
        sizes = (self.n_inputs,) + tuple(self.hidden) + (1,)
        return [(sizes[k + 1], sizes[k]) for k in range(len(sizes) - 1)]

    @property
    def n_params(self) -> int:
        return (self.n_inputs if self.linear_path else 0) + sum(o * i + o for o, i in self.layer_shapes)

    @property
    def input_names(self) -> Tuple[str, ...]:
        return tuple(OBSERVATIONS[i] for i in self.inputs)

    @property
    def inverse_scales(self) -> Tuple[float, ...]:
        return tuple(1.0 / s for s in self.scales)

    @property
    def leak_rate(self) -> float:
        return 0.0 if math.isinf(self.t_leak) else 1.0 / self.t_leak


def unpack(spec: PolicySpec, params) -> Tuple[Optional[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    """Split parameter vectors into (linear weights, [(W, b), ...]).

    params: (n_params,) or (..., n_params). Weights come back with the same leading
    dimensions: linear (..., n_in), W (..., out, in), b (..., out).
    """
    params = np.asarray(params)
    if params.shape[-1] != spec.n_params:
        raise ValueError(f"expected {spec.n_params} parameters, got {params.shape[-1]}")
    lead, k = params.shape[:-1], 0
    linear = None
    if spec.linear_path:
        linear, k = params[..., :spec.n_inputs], spec.n_inputs
    layers = []
    for out, inp in spec.layer_shapes:
        W = params[..., k:k + out * inp].reshape(lead + (out, inp))
        k += out * inp
        b = params[..., k:k + out]
        k += out
        layers.append((W, b))
    return linear, layers


def pack(spec: PolicySpec, linear=None, layers: Sequence[Tuple[np.ndarray, np.ndarray]] = ()) -> np.ndarray:
    """The inverse of unpack, for one policy."""
    parts = []
    if spec.linear_path:
        parts.append(np.asarray(linear, dtype=float).ravel())
    for (W, b), (out, inp) in zip(layers, spec.layer_shapes):
        parts += [np.asarray(W, dtype=float).reshape(out * inp), np.asarray(b, dtype=float).reshape(out)]
    flat = np.concatenate(parts) if parts else np.empty(0)
    if flat.size != spec.n_params:
        raise ValueError(f"expected {spec.n_params} parameters, got {flat.size}")
    return flat


def pid_equivalent_weights(spec: PolicySpec, gains: Sequence[float]) -> np.ndarray:
    """Linear-path weights that reproduce the cascaded PID [kp, ki, kd, kv_p] while nothing saturates.

    The PID's torque is kp e + ki int(e) + kd theta_dot with e = theta - kv_p (goal - v), so in
    this policy's inputs (with its integrators in place of the PID's):

        kp kv_p * v + kp * theta + kd * theta_dot - kp kv_p * goal - ki kv_p * int_v + ki * int_theta

    The spec must be linear-only and read v_meas, theta_meas, theta_dot_meas, goal, int_v and
    int_theta. Every other input gets weight 0. The PID's lean cap and its own anti-windup are
    not reproduced (see the gate test).
    """
    if spec.hidden or not spec.linear_path:
        raise ValueError("pid_equivalent_weights needs a linear-only spec")
    kp, ki, kd, kv = (float(g) for g in gains[:4])
    raw = {"v_meas": kp * kv, "theta_meas": kp, "theta_dot_meas": kd, "goal": -kp * kv,
           "int_v": -ki * kv, "int_theta": ki}
    missing = set(raw) - set(spec.input_names)
    if missing:
        raise ValueError(f"the spec doesn't read {sorted(missing)}")
    # torque = output_scale * sum w_i * raw_i / scale_i, so w_i = coefficient_i * scale_i / output_scale.
    return np.array([raw.get(OBSERVATIONS[i], 0.0) * spec.scales[i] / spec.output_scale for i in spec.inputs])


def save_policy(path, spec: PolicySpec, params, max_torque: float, **extra) -> None:
    """A policy as an .npz: its weights, its spec, the torque limit, plus any extra arrays."""
    np.savez(path, params=np.asarray(params, dtype=float), inputs=np.array(spec.inputs),
             hidden=np.array(spec.hidden, dtype=int), linear_path=spec.linear_path,
             scales=np.array(spec.scales), output_scale=spec.output_scale, t_leak=spec.t_leak,
             integral_limits=np.array(spec.integral_limits), max_torque=max_torque,
             observations=np.array(OBSERVATIONS), **extra)


def load_policy(path) -> Tuple[PolicySpec, np.ndarray, float]:
    """(spec, params, max_torque) from save_policy's file."""
    with np.load(path) as f:
        if tuple(f["observations"]) != OBSERVATIONS:
            raise ValueError("saved with a different observation list")
        spec = PolicySpec(inputs=tuple(int(i) for i in f["inputs"]), hidden=tuple(int(h) for h in f["hidden"]),
                          linear_path=bool(f["linear_path"]), scales=tuple(float(s) for s in f["scales"]),
                          output_scale=float(f["output_scale"]), t_leak=float(f["t_leak"]),
                          integral_limits=tuple(float(v) for v in f["integral_limits"]))
        return spec, np.array(f["params"]), float(f["max_torque"])


class PolicyController(Controller):
    """A PolicySpec plus one parameter vector, as a Controller for balance_sim.simulate."""

    def __init__(self, spec: PolicySpec, params, max_torque: float,
                 goal: Union[float, Callable[[float], float]] = 0.0):
        self.spec = spec
        self.max_torque = float(max_torque)
        self.goal = goal
        self.linear, self.layers = unpack(spec, np.asarray(params, dtype=float))
        self.reset()

    def goal_velocity(self, t: float) -> float:
        return float(self.goal(t)) if callable(self.goal) else float(self.goal)

    def reset(self) -> None:
        self._int_v = 0.0
        self._int_theta = 0.0
        self._history = [0.0] * N_HISTORY
        self._saturated = False
        self._time = 0.0
        self.last_pre_clip = 0.0

    def _integrate(self, integral: float, error: float, dt: float, limit: float) -> float:
        if self._saturated:
            return integral
        new = integral + dt * (error - self.spec.leak_rate * integral)
        return min(max(new, -limit), limit)

    def observation(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Advance the integrators and return the raw (unscaled) OBSERVATIONS for this tick."""
        v, theta, theta_dot = float(state[1]), float(state[2]), float(state[3])
        goal = self.goal_velocity(self._time)
        error_v = goal - v
        limit_v, limit_theta = self.spec.integral_limits
        self._int_v = self._integrate(self._int_v, error_v, dt, limit_v)
        self._int_theta = self._integrate(self._int_theta, theta, dt, limit_theta)
        return np.array([v, theta, theta_dot, goal, error_v, self._int_v, self._int_theta] + self._history)

    def forward(self, raw: np.ndarray) -> float:
        """z for one raw observation vector."""
        inv = self.spec.inverse_scales
        x = [float(raw[i]) * inv[i] for i in self.spec.inputs]
        z = None
        if self.linear is not None:
            acc = 0.0
            for i, xi in enumerate(x):
                acc = acc + float(self.linear[i]) * xi
            z = acc
        if self.layers:
            h = x
            for k, (W, b) in enumerate(self.layers):
                out = []
                for j in range(W.shape[0]):
                    acc = float(b[j])
                    for i, xi in enumerate(h):
                        acc = acc + float(W[j, i]) * xi
                    out.append(acc)
                h = out if k == len(self.layers) - 1 else [float(v) for v in np.tanh(np.array(out))]
            z = h[0] if z is None else z + h[0]
        return z

    def update(self, state: np.ndarray, dt: float) -> float:
        raw = self.observation(state, dt)
        pre = self.spec.output_scale * self.forward(raw)
        self.last_pre_clip = pre
        self._saturated = abs(pre) > self.max_torque
        torque = min(max(pre, -self.max_torque), self.max_torque)
        self._history = [torque] + self._history[:-1]
        self._time += dt
        return torque
