"""Run a whole batch of simulations at once.

This is balance_sim/simulator.py's loop, restructured so every
step advances all B simulations together:

  * The early `break` when a robot falls becomes a per-simulation `alive`
    mask plus a recorded fall time. A fallen robot's state is frozen, and it
    stops adding to its cost.
  * The latency deque becomes a fixed ring buffer of in-flight commands, with
    the schedule of which command is active worked out in advance.
  * Sensor noise comes from a precomputed table (one column per seed), so it
    matches the CPU code's per-seed default_rng exactly.
  * The cost integrals are summed as the run goes instead of storing the
    trajectory, so memory doesn't grow with run length.

The loop stops early once every simulation in the batch has fallen.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .controller import cascaded_update, five_gain_columns
from .physics import rk4_step
from .robot import BatchRobot
from .schedule import Schedule, SimSpec, build_schedule


@dataclass
class BatchResult:
    cost: np.ndarray            # (B,) same value balance_sim.cost gives the CPU run
    pitch_error: np.ndarray     # (B,) integral of pitch error^2 dt
    effort: np.ndarray          # (B,) integral of torque^2 dt
    velocity_error: np.ndarray  # (B,) integral of (velocity - goal)^2 dt
    fell: np.ndarray            # (B,) bool
    fall_time: np.ndarray       # (B,) s; NaN for robots that stayed up
    final_state: np.ndarray     # (B, 4) state at the end (at the moment of falling, for fallen robots)
    states: Optional[np.ndarray] = None  # (n_steps + 1, B, 4) if record=True; frozen after a fall


def simulate_batch(
    spec: SimSpec,
    robot: BatchRobot,
    gains: np.ndarray,
    start_pitch,
    noise: Optional[np.ndarray] = None,
    seed_index: Optional[np.ndarray] = None,
    schedule: Optional[Schedule] = None,
    record: bool = False,
    dtype=np.float64,
) -> BatchResult:
    """Simulate B robots, each with its own gains and starting tilt.

    gains       (B, 4): kp, ki, kd, kv_p (the project's gain set), or (B, 5) with kv_i as well
    start_pitch scalar or (B,) initial body tilt (rad)
    noise       (n_ticks, 4, n_seeds) from schedule.noise_table, or None for a perfect sensor
    seed_index  (B,) which noise column each simulation uses (required with noise)
    schedule    pass a prebuilt one to skip rebuilding it for every batch
    record      also return the full state history (memory: n_steps x B x 4 values)
    dtype       np.float64 to match the CPU code; np.float32 to preview GPU precision
    """
    schedule = build_schedule(spec) if schedule is None else schedule
    gains = five_gain_columns(gains).astype(dtype)
    B = gains.shape[0]
    kp, ki, kd, kv_p, kv_i = gains.T
    robot = robot.astype(dtype)
    torque_limit = robot.max_torque
    dt, control_dt, spc = spec.dt, spec.control_dt, schedule.steps_per_control
    if noise is not None:
        if seed_index is None:
            raise ValueError("seed_index is required when noise is given")
        noise = np.asarray(noise, dtype=dtype)[:, 1:, :]   # the controller never reads position
        seed_index = np.asarray(seed_index)
    goal_per_tick = schedule.goal_per_tick.astype(dtype)
    goal_per_step = schedule.goal_per_step.astype(dtype)

    zeros = lambda: np.zeros(B, dtype=dtype)
    state = (zeros(), zeros(), np.broadcast_to(np.asarray(start_pitch, dtype=dtype), (B,)).copy(), zeros())
    pitch_integral, velocity_integral, setpoint = zeros(), zeros(), zeros()
    ring = np.zeros((schedule.ring_size, B), dtype=dtype)
    torque, active = zeros(), -1
    alive = np.ones(B, dtype=bool)
    fall_time = np.full(B, np.nan, dtype=dtype)
    pitch_error, effort, velocity_error = zeros(), zeros(), zeros()
    upright_reference = spec.pitch_reference == "upright"
    history = [np.stack(state, axis=-1)] if record else None

    for k in range(schedule.n_steps):
        if k % spc == 0:
            j = k // spc
            _, x_dot, theta, theta_dot = state
            if noise is not None:
                n = noise[j][:, seed_index]
                x_dot = x_dot + spec.velocity_std * n[0]
                theta = theta + spec.pitch_std * n[1]
                theta_dot = theta_dot + (spec.pitch_rate_std * n[2] + spec.pitch_rate_bias)
            command, setpoint, pitch_integral, velocity_integral = cascaded_update(
                x_dot, theta, theta_dot, goal_per_tick[j], control_dt,
                kp, ki, kd, kv_p, kv_i, pitch_integral, velocity_integral,
                torque_limit, spec.max_pitch_command,
                spec.pitch_integral_limit, spec.velocity_integral_limit,
            )
            ring[j % schedule.ring_size] = command
        if schedule.active_command[k] != active:
            active = schedule.active_command[k]
            torque = ring[active % schedule.ring_size].copy()

        # Cost terms at this sample, only for robots still up.
        theta = state[2]
        pitch_err = theta if upright_reference else theta - setpoint
        pitch_error += np.where(alive, pitch_err**2 * dt, 0.0)
        effort += np.where(alive, torque**2 * dt, 0.0)
        velocity_error += np.where(alive, (state[1] - goal_per_step[k]) ** 2 * dt, 0.0)

        stepped = rk4_step(robot, state, torque, dt)
        state = tuple(np.where(alive, new, old) for new, old in zip(stepped, state))
        just_fell = alive & (np.abs(state[2]) > spec.fall_angle)
        fall_time[just_fell] = (k + 1) * dt
        alive &= ~just_fell
        if record:
            history.append(np.stack(state, axis=-1))
        if not alive.any():
            break

    if record:
        # Pad an early finish so the history always covers the full run.
        history.extend([history[-1]] * (schedule.n_steps + 1 - len(history)))

    fell = ~alive
    total = pitch_error + spec.effort_weight * effort + spec.velocity_weight * velocity_error
    remaining = (spec.duration - fall_time) / spec.duration
    total = np.where(fell, total + spec.fall_penalty * (1.0 + remaining), total)
    return BatchResult(
        cost=total,
        pitch_error=pitch_error,
        effort=effort,
        velocity_error=velocity_error,
        fell=fell,
        fall_time=fall_time,
        final_state=np.stack(state, axis=-1),
        states=np.stack(history) if record else None,
    )
