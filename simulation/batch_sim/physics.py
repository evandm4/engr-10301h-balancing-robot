"""Planar wheeled-inverted-pendulum physics for a batch of robots.

Same equations as balance_sim/planar2d.py and actuators.py,
written as plain functions over (B,) arrays instead of methods on objects that
hold one robot. The arithmetic is kept in the same order as the CPU code, so in
float64 the two agree to rounding error; the tests check that.

A state is the tuple (x, x_dot, theta, theta_dot) of (B,) arrays, which avoids
slicing a (B, 4) array dozens of times per step.

xp is the array library: numpy (the default) or batch_sim.xp.torch_ops().
Divisions that can mix an array with a plain number go through xp.divide
(see xp.py for why).
"""

import numpy as np

from .robot import BatchRobot


def _is_zero(value) -> bool:
    """True for a plain number equal to 0 (a per-simulation array or tensor never counts)."""
    return isinstance(value, (int, float)) and value == 0


def actuator_torque(robot: BatchRobot, command, relative_wheel_speed, xp=np):
    """Applied torque for a torque command. See Actuator.torque in the CPU code.

    In voltage mode `command` is the drive voltage from voltage_drive (VoltageModeActuator.torque);
    in slow PWM it is the duty (SlowPwmActuator.torque)."""
    if not robot.dc_motor:
        return xp.clip(command, -robot.max_torque, robot.max_torque)
    rotor_speed = robot.gear_ratio * relative_wheel_speed
    if robot.slow_pwm:
        i = xp.divide(xp.copysign(robot.supply_voltage, command) - robot.torque_constant * rotor_speed,
                      robot.resistance)
        i = xp.clip(i, -robot.current_limit, robot.current_limit)
        return robot.n_motors * robot.gear_ratio * robot.gear_efficiency * robot.torque_constant * xp.abs(command) * i
    if robot.voltage_mode:
        i = xp.divide(command - robot.torque_constant * rotor_speed, robot.resistance)
        i = xp.clip(i, -robot.current_limit, robot.current_limit)
        return robot.n_motors * robot.gear_ratio * robot.gear_efficiency * robot.torque_constant * i
    command = xp.clip(command, -robot.max_torque, robot.max_torque)
    per_motor_shaft_torque = xp.divide(command, robot.n_motors * robot.gear_ratio * robot.gear_efficiency)
    i_wanted = xp.divide(per_motor_shaft_torque, robot.torque_constant)
    v_wanted = robot.resistance * i_wanted + robot.torque_constant * rotor_speed
    v = xp.clip(v_wanted, -robot.supply_voltage, robot.supply_voltage)
    i = xp.divide(v - robot.torque_constant * rotor_speed, robot.resistance)
    i = xp.clip(i, -robot.current_limit, robot.current_limit)
    return robot.n_motors * robot.gear_ratio * robot.gear_efficiency * robot.torque_constant * i


def voltage_drive(robot: BatchRobot, command, x_dot, theta_dot, xp=np):
    """The drive voltage the robot code sends for a torque command (VoltageModeActuator.drive).

    x_dot and theta_dot are the MEASURED values (after sensor noise and bias). In slow PWM it
    returns the duty instead (SlowPwmActuator.drive)."""
    w_hat = xp.divide(x_dot, robot.wheel_radius) - theta_dot
    c = xp.clip(command, -robot.max_torque, robot.max_torque)
    if robot.slow_pwm:
        per_motor = xp.divide(c, robot.n_motors)
        a_hat = robot.gear_ratio * robot.est_gear_efficiency * robot.est_torque_constant / robot.est_resistance
        back_emf = 0.0 * w_hat     # an array, so torch keeps the dtype in where() below; v - 0 is exact
        if robot.voltage_feedforward:
            back_emf = robot.est_torque_constant * (robot.gear_ratio * w_hat)
            if robot.voltage_feedforward_gain != 1.0:
                back_emf = robot.voltage_feedforward_gain * back_emf
        v = robot.supply_voltage
        den = xp.where(c >= 0, a_hat * (v - back_emf), a_hat * (v + back_emf))
        ok = den > 0
        duty = xp.where(ok, xp.divide(per_motor, xp.where(ok, den, 1.0)), xp.copysign(1.0, per_motor))
        return xp.clip(duty, -1.0, 1.0)
    volts = robot.est_resistance * xp.divide(
        c, robot.n_motors * robot.gear_ratio * robot.est_gear_efficiency * robot.est_torque_constant)
    if robot.voltage_feedforward:
        back_emf = robot.est_torque_constant * (robot.gear_ratio * w_hat)
        gain = robot.voltage_feedforward_gain
        volts = volts + (back_emf if gain == 1.0 else gain * back_emf)
    return xp.clip(volts, -robot.supply_voltage, robot.supply_voltage)


def accelerometer_tilt(theta, theta_dot, x_ddot, theta_ddot, height, gravity, xp=np):
    """The tilt an accelerometer `height` above the axle reads (balance_sim.sensors.accelerometer_tilt),
    over (B,) arrays, with the same operation order."""
    s, c = xp.sin(theta), xp.cos(theta)
    a_x = x_ddot + height * (theta_ddot * c - theta_dot**2 * s)
    f_z = height * (-theta_ddot * s - theta_dot**2 * c) + gravity
    f_forward = a_x * c - f_z * s
    f_up = a_x * s + f_z * c
    return xp.arctan2(-f_forward, f_up)


def derivatives(robot: BatchRobot, state, torque, xp=np, force=0.0):
    """d(state)/dt for every robot in the batch. See Planar2D.derivatives.

    force: external horizontal push on the body at its center of mass (N), the
    same for the whole batch: a plain number (skipped when 0, as in Planar2D)
    or a 0-d tensor (always added; adding 0.0 changes nothing)."""
    p = robot
    _, x_dot, theta, theta_dot = state

    relative_wheel_speed = xp.divide(x_dot, p.wheel_radius) - theta_dot
    tau = actuator_torque(p, torque, relative_wheel_speed, xp)
    tau = tau - p.wheel_friction * relative_wheel_speed
    if not _is_zero(p.coulomb_friction):   # skipped for a frictionless batch, as in Planar2D
        tau = tau - p.coulomb_friction * xp.tanh(xp.divide(relative_wheel_speed, p.coulomb_friction_speed))

    sin_t, cos_t = xp.sin(theta), xp.cos(theta)
    ml = p.body_mass * p.body_com_height

    # Robot constants only (numbers, or tensors when randomized), so plain "/".
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2 + p.rotor_inertia / p.wheel_radius**2
    b = ml * cos_t - p.rotor_inertia / p.wheel_radius
    d = p.body_mass * p.body_com_height**2 + p.body_inertia + p.rotor_inertia

    rhs_x = xp.divide(tau, p.wheel_radius) + ml * sin_t * theta_dot**2
    rhs_t = -tau + ml * p.gravity * sin_t
    if not _is_zero(force):
        rhs_x = rhs_x + force
        rhs_t = rhs_t + force * p.body_com_height * cos_t

    det = a * d - b * b
    x_ddot = (d * rhs_x - b * rhs_t) / det
    theta_ddot = (a * rhs_t - b * rhs_x) / det
    return x_dot, x_ddot, theta_dot, theta_ddot


def _axpy(state, h, k):
    return tuple(s + h * ki for s, ki in zip(state, k))


def rk4_step(robot: BatchRobot, state, torque, dt: float, xp=np, force=0.0):
    """One RK4 step with the torque (and push force) held constant, as in simulator._rk4_step."""
    k1 = derivatives(robot, state, torque, xp, force)
    k2 = derivatives(robot, _axpy(state, 0.5 * dt, k1), torque, xp, force)
    k3 = derivatives(robot, _axpy(state, 0.5 * dt, k2), torque, xp, force)
    k4 = derivatives(robot, _axpy(state, dt, k3), torque, xp, force)
    h = dt / 6.0
    return tuple(s + h * (a + 2 * b + 2 * c + d) for s, a, b, c, d in zip(state, k1, k2, k3, k4))


def pdm_update(total, duty, xp=np):
    """One tick of the HAT's pulse-density modulation over (B,) arrays (balance_sim.pdm_update):
    returns (new running sum, this tick's duty: -1, 0 or +1). Both comparisons are exact, so the
    result is bit-identical to the reference for any backend."""
    total = total + xp.clip(duty, -1.0, 1.0)
    if xp is np:
        out = (total >= 0.5).astype(total.dtype) - (total <= -0.5).astype(total.dtype)
    else:
        out = (total >= 0.5).to(total.dtype) - (total <= -0.5).to(total.dtype)
    return total - out, out


def pwm_steps_per_tick(pwm_tick: float, dt: float) -> int:
    """Physics steps per PDM tick (balance_sim.pwm_steps_per_tick); 0 when off."""
    if pwm_tick <= 0:
        return 0
    n = int(round(pwm_tick / dt))
    if n < 1 or abs(n * dt - pwm_tick) > 1e-9:
        raise ValueError(f"pwm_tick {pwm_tick} s is not a whole number of physics steps of {dt} s")
    return n
