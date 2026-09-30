"""Cascaded PID controller for a batch of robots, as pure functions.

Same control law as balance_sim/controllers.py
(CascadedPIDController wrapping a PIDController with an output limit). The
integrators live in arrays passed in and returned, instead of inside objects,
so one call updates every robot in the batch.

Gains are (B,) arrays: kp, ki, kd for the pitch loop and kv_p, kv_i for the
velocity loop (kv_i is all zeros for the project's 4-gain controller). xp is
the array library: numpy (the default) or batch_sim.xp.torch_ops().
"""

import numpy as np


def five_gain_columns(gains) -> np.ndarray:
    """(B, 4) [kp, ki, kd, kv_p] or (B, 5) [..., kv_i] gains -> (B, 5), with kv_i = 0 if left out."""
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 2 or gains.shape[1] not in (4, 5):
        raise ValueError(f"gains must have shape (B, 4) or (B, 5), got {gains.shape}")
    if gains.shape[1] == 4:
        gains = np.column_stack([gains, np.zeros(len(gains))])
    return gains


def limited_integral(integral, error, dt, other_terms, gain, integral_limit, output_limit, xp=np):
    """Vectorized _limited_integral: anti-windup integrator step.

    Branch-free translation of the scalar version (every branch is computed
    and xp.where picks per element). output_limit must be given here; the
    tuner always sets one for both loops.
    """
    new = xp.clip(integral + error * dt, -integral_limit, integral_limit)
    output = other_terms + gain * new
    saturating = (xp.abs(output) > output_limit) & (error * output > 0)

    positive_gain = gain > 0
    safe_gain = xp.where(positive_gain, gain, 1.0)
    target = (xp.copysign(output_limit, output) - other_terms) / safe_gain
    step, allowed = new - integral, target - integral
    toward_limit = integral + xp.where(
        step > 0,
        xp.minimum(step, xp.maximum(allowed, 0.0)),
        xp.maximum(step, xp.minimum(allowed, 0.0)),
    )
    new = xp.where(saturating, xp.where(positive_gain, toward_limit, integral), new)

    cap = xp.divide(output_limit, safe_gain)
    return xp.where(positive_gain, xp.clip(new, -cap, cap), new)


def cascaded_update(
    measured_velocity, measured_pitch, measured_pitch_rate,
    goal_velocity, dt,
    kp, ki, kd, kv_p, kv_i,
    pitch_integral, velocity_integral,
    torque_limit, max_pitch_command,
    pitch_integral_limit, velocity_integral_limit,
    xp=np,
):
    """One controller tick for the whole batch.

    Returns (torque, pitch_setpoint, pitch_integral, velocity_integral).
    goal_velocity may be a scalar (the same goal for every robot this tick).
    """
    # Outer loop: velocity error -> commanded lean.
    velocity_error = goal_velocity - measured_velocity
    proportional = kv_p * velocity_error
    velocity_integral = limited_integral(
        velocity_integral, velocity_error, dt, proportional, kv_i,
        velocity_integral_limit, max_pitch_command, xp,
    )
    lean = proportional + kv_i * velocity_integral
    setpoint = xp.clip(lean, -max_pitch_command, max_pitch_command)

    # Inner loop: pitch error -> torque.
    error = measured_pitch - setpoint
    proportional_derivative = kp * error + kd * measured_pitch_rate
    pitch_integral = limited_integral(
        pitch_integral, error, dt, proportional_derivative, ki,
        pitch_integral_limit, torque_limit, xp,
    )
    torque = proportional_derivative + ki * pitch_integral
    torque = xp.clip(torque, -torque_limit, torque_limit)
    return torque, setpoint, pitch_integral, velocity_integral
