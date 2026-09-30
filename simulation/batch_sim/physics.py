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


def actuator_torque(robot: BatchRobot, command, relative_wheel_speed, xp=np):
    """Applied torque for a torque command. See Actuator.torque in the CPU code."""
    if not robot.dc_motor:
        return xp.clip(command, -robot.max_torque, robot.max_torque)
    command = xp.clip(command, -robot.max_torque, robot.max_torque)
    rotor_speed = robot.gear_ratio * relative_wheel_speed
    per_motor_shaft_torque = xp.divide(command, robot.n_motors * robot.gear_ratio)
    i_wanted = xp.divide(per_motor_shaft_torque, robot.torque_constant)
    v_wanted = robot.resistance * i_wanted + robot.torque_constant * rotor_speed
    v = xp.clip(v_wanted, -robot.supply_voltage, robot.supply_voltage)
    i = xp.divide(v - robot.torque_constant * rotor_speed, robot.resistance)
    i = xp.clip(i, -robot.current_limit, robot.current_limit)
    return robot.n_motors * robot.gear_ratio * robot.torque_constant * i


def derivatives(robot: BatchRobot, state, torque, xp=np):
    """d(state)/dt for every robot in the batch. See Planar2D.derivatives."""
    p = robot
    _, x_dot, theta, theta_dot = state

    relative_wheel_speed = xp.divide(x_dot, p.wheel_radius) - theta_dot
    tau = actuator_torque(p, torque, relative_wheel_speed, xp)
    tau = tau - p.wheel_friction * relative_wheel_speed

    sin_t, cos_t = xp.sin(theta), xp.cos(theta)
    ml = p.body_mass * p.body_com_height

    # Robot constants only (numbers, or tensors when randomized), so plain "/".
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
    b = ml * cos_t
    d = p.body_mass * p.body_com_height**2 + p.body_inertia

    rhs_x = xp.divide(tau, p.wheel_radius) + ml * sin_t * theta_dot**2
    rhs_t = -tau + ml * p.gravity * sin_t

    det = a * d - b * b
    x_ddot = (d * rhs_x - b * rhs_t) / det
    theta_ddot = (a * rhs_t - b * rhs_x) / det
    return x_dot, x_ddot, theta_dot, theta_ddot


def _axpy(state, h, k):
    return tuple(s + h * ki for s, ki in zip(state, k))


def rk4_step(robot: BatchRobot, state, torque, dt: float, xp=np):
    """One RK4 step with the torque held constant, as in simulator._rk4_step."""
    k1 = derivatives(robot, state, torque, xp)
    k2 = derivatives(robot, _axpy(state, 0.5 * dt, k1), torque, xp)
    k3 = derivatives(robot, _axpy(state, 0.5 * dt, k2), torque, xp)
    k4 = derivatives(robot, _axpy(state, dt, k3), torque, xp)
    h = dt / 6.0
    return tuple(s + h * (a + 2 * b + 2 * c + d) for s, a, b, c, d in zip(state, k1, k2, k3, k4))
