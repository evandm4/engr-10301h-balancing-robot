"""balance_sim's PolicyController for a batch: many robots, and many different networks at once.

The batch is laid out candidate-major, as BatchEvaluator builds it: B = P x S simulations,
where rows p*S .. p*S + S - 1 all run candidate p's network (on S different scenarios). So
each network's weights are stored once, (P, ...), never once per simulation.

The arithmetic is balance_sim/policy.py's, in the same order (bias first, then inputs 0, 1,
...), written as elementwise multiply-adds rather than matrix products. That is what makes the
NumPy float64 result match the one-robot reference bit for bit; it also keeps BLAS, whose
summation order depends on the library and the CPU, out of validation. The same function runs
on PyTorch through the xp shim.

A policy's memory (integrators, command history, last tick's saturation flag) lives in a
PolicyMemory passed in and returned, like the PID's integrators in controller.py.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from balance_sim.policy import N_HISTORY, PolicySpec, unpack


@dataclass
class PolicyMemory:
    int_v: object                # (B,) leaky integral of goal - v_meas
    int_theta: object            # (B,) leaky integral of theta_meas
    history: List[object]        # N_HISTORY arrays (B,): last commands, newest first
    saturated: object            # (B,) bool: last tick's pre-clip output beyond the torque limit

    @classmethod
    def zeros(cls, B: int, dtype=np.float64) -> "PolicyMemory":
        z = lambda: np.zeros(B, dtype=dtype)
        return cls(z(), z(), [z() for _ in range(N_HISTORY)], np.zeros(B, dtype=bool))


def unpack_batch(spec: PolicySpec, params, xp=np):
    """(P, n_params) -> (linear (P, n_in) or None, [(W (P, out, in), b (P, out)), ...]).

    Works on NumPy arrays and torch tensors (slices and reshapes only, so torch gives views:
    a simulator can unpack its parameter buffer once and keep the views).
    """
    if xp is np:
        return unpack(spec, params)
    P, k = params.shape[0], 0
    linear = None
    if spec.linear_path:
        linear, k = params[:, :spec.n_inputs], spec.n_inputs
    layers = []
    for out, inp in spec.layer_shapes:
        W = params[:, k:k + out * inp].reshape(P, out, inp)
        k += out * inp
        layers.append((W, params[:, k:k + out]))
        k += out
    return linear, layers


def _tanh(x, xp):
    return np.tanh(x) if xp is np else xp.tanh(x)


def forward(spec: PolicySpec, weights, x: List[object], xp=np):
    """z for every simulation. x: the policy's inputs, each (P, S), already scaled.

    weights: unpack_batch's result for the same P candidates. Returns (P, S).
    """
    linear, layers = weights
    z = None
    if linear is not None:
        acc = 0.0
        for i, xi in enumerate(x):
            acc = acc + linear[:, i:i + 1] * xi
        z = acc
    if layers:
        h = [xi[:, :, None] for xi in x]          # each (P, S, 1)
        for k, (W, b) in enumerate(layers):
            acc = b[:, None, :]                   # (P, 1, out): bias first, as in the reference
            for i, hi in enumerate(h):
                acc = acc + W[:, None, :, i] * hi
            if k < len(layers) - 1:
                acc = _tanh(acc, xp)
                h = [acc[:, :, j:j + 1] for j in range(acc.shape[-1])]
            else:
                out = acc[:, :, 0]
        z = out if z is None else z + out
    return z


def policy_update(spec: PolicySpec, weights, n_candidates: int,
                  measured_velocity, measured_pitch, measured_pitch_rate, goal_velocity, dt,
                  memory: PolicyMemory, torque_limit, xp=np) -> Tuple[object, object, PolicyMemory]:
    """One controller tick for the whole batch.

    Returns (torque (B,), pre-clip output (B,), new memory). goal_velocity may be a scalar
    (the same goal for every robot this tick). torque_limit may be a number or (B,).
    """
    B = measured_velocity.shape[0]
    P, S = n_candidates, B // n_candidates
    leak = spec.leak_rate
    limit_v, limit_theta = spec.integral_limits
    goal = goal_velocity + 0.0 * measured_velocity        # broadcast a scalar goal to (B,)
    error_v = goal - measured_velocity

    def integrate(integral, error, limit):
        new = xp.clip(integral + dt * (error - leak * integral), -limit, limit)
        return xp.where(memory.saturated, integral, new)

    int_v = integrate(memory.int_v, error_v, limit_v)
    int_theta = integrate(memory.int_theta, measured_pitch, limit_theta)
    raw = [measured_velocity, measured_pitch, measured_pitch_rate, goal, error_v, int_v, int_theta] + \
        list(memory.history)
    inv = spec.inverse_scales
    x = [(raw[i] * inv[i]).reshape(P, S) for i in spec.inputs]
    pre = (spec.output_scale * forward(spec, weights, x, xp)).reshape(B)
    torque = xp.clip(pre, -torque_limit, torque_limit)
    saturated = xp.abs(pre) > torque_limit
    history = [torque] + list(memory.history[:-1])
    return torque, pre, PolicyMemory(int_v, int_theta, history, saturated)
