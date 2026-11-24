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

import math

import numpy as np

from .controller import cascaded_update, five_gain_columns, lqr_gain_columns, lqr_update
from .physics import (accelerometer_tilt, backlash_end_step, backlash_release, backlash_rk4_step, backlash_terms,
                      derivatives, pdm_update, pwm_steps_per_tick, pwm_tick_offset, rk4_step, voltage_drive)
from .policy import PolicyMemory, ResidualBase, policy_update, unpack_batch
from .robot import BatchRobot
from .schedule import Schedule, SimSpec, build_schedule, encoder_steps, per_sim_ring_size


@dataclass
class BatchResult:
    cost: np.ndarray            # (B,) same value balance_sim.cost gives the CPU run
    pitch_error: np.ndarray     # (B,) integral of pitch error^2 dt
    effort: np.ndarray          # (B,) integral of torque^2 dt
    velocity_error: np.ndarray  # (B,) integral of (velocity - goal)^2 dt
    fell: np.ndarray            # (B,) bool
    fall_time: np.ndarray       # (B,) s; NaN for robots that stayed up
    final_state: np.ndarray     # (B, 4) state at the end (at the moment of falling, for fallen robots);
                                # (B, 8) with backlash, in Planar2D's order (gap, motor_speed, shaft, contact)
    states: Optional[np.ndarray] = None  # (n_steps + 1, B, 4 or 8) if record=True; frozen after a fall
    # How the controller drove the motor, counted over control ticks while the robot was up (none
    # of these enter the cost; they are for reporting, e.g. torque roughness the cost can't see):
    torque_variation: Optional[np.ndarray] = None   # (B,) sum of |u_k - u_{k-1}| over commands (N m)
    torque_rate_sq: Optional[np.ndarray] = None     # (B,) sum of (u_k - u_{k-1})^2 (N^2 m^2)
    saturated_ticks: Optional[np.ndarray] = None    # (B,) ticks whose command sat at the torque limit
    alive_ticks: Optional[np.ndarray] = None        # (B,) controller ticks while up
    peak_pitch: Optional[np.ndarray] = None         # (B,) largest |pitch| while up (rad)


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
    delay_steps: Optional[np.ndarray] = None,
    scenarios_per_candidate: Optional[int] = None,
) -> BatchResult:
    """Simulate B robots, each with its own gains and starting tilt.

    gains       spec.controller "cascaded": (B, 4) kp, ki, kd, kv_p (the project's gain set),
                or (B, 5) with kv_i as well. "lqr": (B, 4 + m) LQR gains (balance_sim.lqr),
                the last m for the commands in flight.
    start_pitch scalar or (B,) initial body tilt (rad)
    noise       (n_ticks, 4, n_seeds) from schedule.noise_table, or None for no random noise. The
                spec's gyro bias applies either way, as in SensorNoise with every std at 0;
                for a perfect sensor, also use a spec with pitch_rate_bias = 0.
    IMU mode (spec.imu): the controller sees the complementary filter's pitch estimate and the
                gyro rate (see balance_sim.sensors.IMUSensor); gravity is the robot's.
    seed_index  (B,) which noise column each simulation uses (required with noise)
    schedule    pass a prebuilt one to skip rebuilding it for every batch
    record      also return the full state history (memory: n_steps x B x 4 values)
    dtype       np.float64 to match the CPU code; np.float32 to preview GPU precision
    delay_steps (B,) integers: each simulation's own latency, in physics steps, in place
                of spec.latency. A command issued at step k reaches the motor at step
                k + delay_steps, which is what the CPU simulator does with
                latency = delay_steps * dt (see per_sim_ring_size). The pulse ticks' phase
                stays robot.pwm_tick_phase: for ticks locked to the loop
                (ActuatorConfig.pwm_tick_locked) pass the matching per-simulation phase,
                balance_sim.locked_pwm_tick_phase(delay_steps, ...), as BatchEvaluator does.

    spec.controller "policy": gains is (P, n_params), one parameter vector per network, and
    scenarios_per_candidate S says how many simulations each runs: B = P x S, candidate-major
    (simulations p*S .. p*S + S - 1 run network p). The (B,) inputs follow that order.
    """
    schedule = build_schedule(spec) if schedule is None else schedule
    lqr = spec.controller == "lqr"
    policy = spec.controller == "policy"
    if policy:
        params = np.atleast_2d(np.asarray(gains, dtype=dtype))
        if scenarios_per_candidate is None or scenarios_per_candidate < 1:
            raise ValueError("a policy batch needs scenarios_per_candidate")
        n_candidates = params.shape[0]
        B = n_candidates * scenarios_per_candidate
        weights = unpack_batch(spec.policy, params)
        memory = PolicyMemory.zeros(B, dtype)
        base = ResidualBase.build(spec.policy, spec, B, dtype) if spec.policy.residual else None
        columns = ()
    else:
        if scenarios_per_candidate is not None:
            raise ValueError("scenarios_per_candidate is only for policy batches")
        gains = (lqr_gain_columns(gains) if lqr else five_gain_columns(gains)).astype(dtype)
        B = gains.shape[0]
        columns = tuple(gains.T)
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
    backlash = robot.has_backlash
    if backlash:
        # Planar2D.complete_state: play open and centred, motor side at the wheel's speed, shaft on the wheel.
        if np.any(np.asarray(robot.gearbox_backlash + robot.coupling_backlash) <= 0):
            raise ValueError("with backlash every simulation's play must be positive (both 0 is the rigid model)")
        x, x_dot, theta, theta_dot = state
        state = state + (zeros(), np.divide(x_dot, robot.wheel_radius) - theta_dot)
        contact, shaft = zeros(), np.divide(x, robot.wheel_radius) - theta
    full_state = (lambda: np.stack(state + (shaft, contact), axis=-1)) if backlash else (lambda: np.stack(state, axis=-1))
    pitch_integral, velocity_integral, setpoint = zeros(), zeros(), zeros()   # setpoint stays 0 for the LQR
    in_flight = [zeros() for _ in range(gains.shape[1] - 4)] if lqr else []    # LQR commands in flight, oldest first
    per_sim_delay = delay_steps is not None
    if per_sim_delay:
        delay_steps = np.broadcast_to(np.asarray(delay_steps, dtype=np.int64), (B,))
        ring_size = per_sim_ring_size(schedule, delay_steps)
        columns_b = np.arange(B)
    else:
        ring_size = schedule.ring_size
    ring = np.zeros((ring_size + 1, B), dtype=dtype)   # the extra last row stays 0: no command yet
    torque, active = zeros(), -1
    voltage = robot.voltage_mode
    pitch_bias = spec.pitch_bias if robot.pitch_bias is None else robot.pitch_bias   # per-simulation (B,) or the spec's scalar
    has_bias = bool(np.any(pitch_bias))
    ring_drive = np.zeros_like(ring) if voltage else None   # drive voltages, same slots as ring
    drive = zeros() if voltage else None
    pdm_steps = pwm_steps_per_tick(robot.pwm_tick, dt)   # slow PWM's pulse-density ticks; 0 = averaged
    pdm_total = zeros() if pdm_steps else None
    pdm_offset = pwm_tick_offset(robot.pwm_tick_phase, pdm_steps)   # steps to the first tick: int or (B,) ints
    pdm_per_sim = isinstance(pdm_offset, np.ndarray)
    applied = zeros()      # what reached the motor during the last step (drive or torque, or its PDM pulse)
    estimate = None   # the IMU complementary filter's pitch estimate; set on the first tick
    alpha = spec.filter_time_constant / (spec.filter_time_constant + control_dt)   # as ComplementaryFilter
    one_minus_alpha = 1.0 - alpha
    alive = np.ones(B, dtype=bool)
    fall_time = np.full(B, np.nan, dtype=dtype)
    pitch_error, effort, velocity_error = zeros(), zeros(), zeros()
    last_command, torque_variation, torque_rate_sq = zeros(), zeros(), zeros()
    saturated_ticks, alive_ticks, peak_pitch = zeros(), zeros(), zeros()
    upright_reference = spec.pitch_reference == "upright"
    history = [full_state()] if record else None
    encoder = spec.encoder_counts > 0
    if encoder:
        # Counts sampled every n_period steps (the state at the start of the step, before the
        # control tick), each usable n_delay steps later; a ring keeps the few still needed.
        n_period, n_delay = encoder_steps(spec)
        resolution = 2.0 * math.pi / spec.encoder_counts
        speed_per_count = resolution / spec.encoder_period
        counts = np.zeros((n_delay // n_period + 3, B), dtype=dtype)

    for k in range(schedule.n_steps):
        if encoder and k % n_period == 0:
            angle = shaft if backlash else np.divide(state[0], robot.wheel_radius) - state[2]   # Encoder.observe
            counts[(k // n_period) % len(counts)] = np.floor(angle / resolution)
        if k % spc == 0:
            j = k // spc
            x_dot, theta, theta_dot = state[1], state[2], state[3]
            if noise is not None:
                n = noise[j][:, seed_index]
                x_dot = x_dot + spec.velocity_std * n[0]
            if spec.imu:
                # Accelerations at the sample: the drive in effect just before this tick, the push now.
                force = float(schedule.force_per_step[k])
                # Released contact for the sample only (as begin_step); the live contact is not touched.
                sense_contact = backlash_release(robot, state, applied, contact, np, force) if backlash else None
                derivs = (backlash_terms(robot, state, applied, sense_contact, np, force)[0] if backlash
                          else derivatives(robot, state, applied, np, force))
                if noise is not None:
                    gyro = theta_dot + (spec.pitch_rate_std * n[2] + spec.pitch_rate_bias)
                    tilt_offset = spec.pitch_std * n[1] + pitch_bias
                else:   # the biases still apply, with zero noise
                    gyro = theta_dot + spec.pitch_rate_bias
                    tilt_offset = pitch_bias
                tilt = accelerometer_tilt(theta, theta_dot, derivs[1], derivs[3],
                                          spec.imu_height, robot.gravity, np) + tilt_offset
                if j == 0:
                    estimate = tilt
                else:
                    estimate = alpha * (estimate + control_dt * gyro) + one_minus_alpha * tilt
                theta, theta_dot = estimate, gyro
            elif noise is not None:
                theta = theta + (spec.pitch_std * n[1] + pitch_bias)
                theta_dot = theta_dot + (spec.pitch_rate_std * n[2] + spec.pitch_rate_bias)
            elif spec.pitch_rate_bias or has_bias:   # deterministic, so they need no table
                theta = theta + pitch_bias
                theta_dot = theta_dot + spec.pitch_rate_bias
            if encoder:
                m = (k - n_delay) // n_period if k >= n_delay else -1
                speed = (counts[m % len(counts)] - counts[(m - 1) % len(counts)]) * speed_per_count if m >= 1 else 0.0
                x_dot = robot.wheel_radius * (speed + theta_dot)
            if policy:
                command, _, memory = policy_update(
                    spec.policy, weights, n_candidates, x_dot, theta, theta_dot, goal_per_tick[j],
                    control_dt, memory, torque_limit, base=base,
                )
            elif lqr:
                command, velocity_integral = lqr_update(
                    x_dot, theta, theta_dot, goal_per_tick[j], control_dt,
                    columns, velocity_integral, in_flight,
                    torque_limit, spec.velocity_integral_limit,
                )
                in_flight = in_flight[1:] + [command] if in_flight else in_flight
            else:
                command, setpoint, pitch_integral, velocity_integral = cascaded_update(
                    x_dot, theta, theta_dot, goal_per_tick[j], control_dt,
                    *columns, pitch_integral, velocity_integral,
                    torque_limit, spec.max_pitch_command,
                    spec.pitch_integral_limit, spec.velocity_integral_limit,
                )
            ring[j % ring_size] = command
            if voltage:
                ring_drive[j % ring_size] = voltage_drive(robot, command, x_dot, theta_dot)
            change = command - last_command
            torque_variation += np.where(alive, np.abs(change), 0.0)
            torque_rate_sq += np.where(alive, change * change, 0.0)
            saturated_ticks += np.where(alive & (np.abs(command) >= torque_limit), 1.0, 0.0)
            alive_ticks += np.where(alive, 1.0, 0.0)
            last_command = command
        if per_sim_delay:
            since = k - delay_steps
            slot = np.where(since >= 0, (since // spc) % ring_size, ring_size)
            torque = ring[slot, columns_b]
            if voltage:
                drive = ring_drive[slot, columns_b]
        elif schedule.active_command[k] != active:
            active = schedule.active_command[k]
            torque = ring[active % ring_size].copy()
            if voltage:
                drive = ring_drive[active % ring_size].copy()

        # Cost terms at this sample, only for robots still up.
        theta = state[2]
        pitch_err = theta if upright_reference else theta - setpoint
        peak_pitch = np.maximum(peak_pitch, np.where(alive, np.abs(theta), 0.0))
        pitch_error += np.where(alive, pitch_err**2 * dt, 0.0)
        effort += np.where(alive, torque**2 * dt, 0.0)
        velocity_error += np.where(alive, (state[1] - goal_per_step[k]) ** 2 * dt, 0.0)

        if not pdm_steps:
            applied = drive if voltage else torque
        elif pdm_per_sim:
            since = k - pdm_offset
            tick = (since >= 0) & (since % pdm_steps == 0)
            if tick.any():
                total, out = pdm_update(pdm_total, drive)
                pdm_total = np.where(tick, total, pdm_total)
                applied = np.where(tick, out, applied)
        elif k >= pdm_offset and (k - pdm_offset) % pdm_steps == 0:
            pdm_total, applied = pdm_update(pdm_total, drive)
        force = float(schedule.force_per_step[k])
        if backlash:
            new_contact = backlash_release(robot, state, applied, contact, np, force)
            stepped = backlash_rk4_step(robot, state, applied, new_contact, dt, np, force)
            stepped, new_contact, new_shaft = backlash_end_step(robot, stepped, new_contact, shaft, np)
            contact, shaft = np.where(alive, new_contact, contact), np.where(alive, new_shaft, shaft)
        else:
            stepped = rk4_step(robot, state, applied, dt, np, force)
        state = tuple(np.where(alive, new, old) for new, old in zip(stepped, state))
        just_fell = alive & (np.abs(state[2]) > spec.fall_angle)
        fall_time[just_fell] = (k + 1) * dt
        alive &= ~just_fell
        if record:
            history.append(full_state())
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
        final_state=full_state(),
        states=np.stack(history) if record else None,
        torque_variation=torque_variation,
        torque_rate_sq=torque_rate_sq,
        saturated_ticks=saturated_ticks,
        alive_ticks=alive_ticks,
        peak_pitch=peak_pitch,
    )
