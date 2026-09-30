"""Each batched building block against its reference (balance_sim) counterpart, on many random inputs."""

from dataclasses import replace

import numpy as np
import pytest

from balance_sim import (
    CascadedPIDController,
    DCMotorActuator,
    IdealActuator,
    PIDController,
    Planar2D,
    SensorNoise,
    step_profile,
)
from balance_sim.controllers import _limited_integral
from batch_sim import (
    BatchRobot,
    actuator_torque,
    build_schedule,
    cascaded_update,
    derivatives,
    five_gain_columns,
    limited_integral,
    noise_table,
)
from batch_tuning import spec_from_eval_config
from config import CONFIG

RNG = np.random.default_rng(12345)
ROBOT = CONFIG.build_robot_params()
MOTOR = CONFIG.actuator.build_motor_params()
SPEC = spec_from_eval_config(CONFIG.build_eval_config())


def _edge_heavy(n, scale):
    """Random values, with some exact zeros and some large ones mixed in."""
    x = RNG.normal(0.0, scale, n)
    x[RNG.random(n) < 0.1] = 0.0
    x[RNG.random(n) < 0.1] *= 50
    return x


def test_limited_integral_matches_scalar_version():
    n = 20_000
    integral, error, other = _edge_heavy(n, 0.5), _edge_heavy(n, 0.3), _edge_heavy(n, 0.5)
    gain = np.abs(_edge_heavy(n, 2.0))            # includes exact zeros
    output_limit = RNG.uniform(0.05, 1.0, n)
    batched = limited_integral(integral, error, 0.005, other, gain, 1.0, output_limit)
    scalar = np.array([_limited_integral(i, e, 0.005, o, g, 1.0, lim)
                       for i, e, o, g, lim in zip(integral, error, other, gain, output_limit)])
    np.testing.assert_array_equal(batched, scalar)


@pytest.mark.parametrize("dc_motor", [False, True])
def test_derivatives_match_planar2d(dc_motor):
    params = ROBOT.with_changes(wheel_friction=0.002)
    motor = MOTOR if dc_motor else None
    actuator = DCMotorActuator(motor, params.max_torque) if dc_motor else IdealActuator(params.max_torque)
    dyn = Planar2D(params, CONFIG.timing.fall_angle_rad, actuator=actuator)
    robot = BatchRobot.from_params(params, motor)

    n = 2000
    states = np.column_stack([_edge_heavy(n, 1), _edge_heavy(n, 1), RNG.uniform(-1, 1, n), _edge_heavy(n, 3)])
    torques = _edge_heavy(n, 0.5)
    batched = np.column_stack(derivatives(robot, tuple(states.T), torques))
    scalar = np.array([dyn.derivatives(s, u) for s, u in zip(states, torques)])
    np.testing.assert_array_equal(batched, scalar)


def test_actuator_matches_dc_motor_with_per_robot_parameters():
    n = 500
    supply = RNG.uniform(5.0, 12.0, n)
    robot = BatchRobot.from_params(ROBOT, MOTOR, supply_voltage=supply)
    command, speed = _edge_heavy(n, 0.5), _edge_heavy(n, 20)
    batched = actuator_torque(robot, command, speed)
    scalar = [DCMotorActuator(replace(MOTOR, supply_voltage=v), max_torque=ROBOT.max_torque).torque(c, w)
              for v, c, w in zip(supply, command, speed)]
    np.testing.assert_array_equal(batched, scalar)


def test_ideal_actuator_robot_has_no_motor_values():
    robot = BatchRobot.from_params(ROBOT, None)
    assert not robot.dc_motor and robot.torque_constant is None
    np.testing.assert_array_equal(actuator_torque(robot, np.array([-5.0, 0.1, 5.0]), np.zeros(3)),
                                  [-ROBOT.max_torque, 0.1, ROBOT.max_torque])


def test_five_gain_columns():
    four = np.array([[4.0, 0.5, 0.15, 0.1]])
    np.testing.assert_array_equal(five_gain_columns(four), [[4.0, 0.5, 0.15, 0.1, 0.0]])
    five = np.array([[4.0, 0.5, 0.15, 0.1, 0.03]])
    np.testing.assert_array_equal(five_gain_columns(five), five)
    with pytest.raises(ValueError):
        five_gain_columns(np.zeros((2, 3)))


def test_cascaded_update_matches_controller_over_many_ticks():
    n, ticks, dt = 50, 300, 0.005
    gains = np.column_stack([RNG.uniform(1, 30, n), RNG.uniform(0, 5, n), RNG.uniform(0, 1, n),
                             RNG.uniform(0, 0.4, n), RNG.uniform(0, 0.3, n)])
    gains[:5, 1] = 0.0   # ki = 0 takes the "no positive gain" branch
    goal = step_profile([(0.3, 0.3), (1.0, -0.2)])
    # Explicit limits (not the project's) so the check doesn't depend on config.py.
    controllers = [CascadedPIDController(PIDController(*g[:3], integral_limit=1.0, output_limit=0.4), g[3], g[4],
                                         goal=goal, max_pitch_command=0.2, integral_limit=0.5) for g in gains]
    for c in controllers:
        c.reset()
    pitch_integral, velocity_integral = np.zeros(n), np.zeros(n)
    for j in range(ticks):
        measured = np.column_stack([np.zeros(n), _edge_heavy(n, 0.3), _edge_heavy(n, 0.1), _edge_heavy(n, 1)])
        torque, setpoint, pitch_integral, velocity_integral = cascaded_update(
            measured[:, 1], measured[:, 2], measured[:, 3], goal(j * dt), dt, *gains.T,
            pitch_integral, velocity_integral, 0.4, 0.2, 1.0, 0.5)
        expected = [c.update(m, dt) for c, m in zip(controllers, measured)]
        np.testing.assert_array_equal(torque, expected)
        np.testing.assert_array_equal(setpoint, [c.pitch_reference() for c in controllers])


def test_noise_table_matches_sensor_noise_stream():
    seeds, ticks = (1, 2, 107), 50
    table = noise_table(seeds, ticks)
    assert table.shape == (ticks, 4, len(seeds))
    for s_i, seed in enumerate(seeds):
        sensor = SensorNoise(position_std=1.0, velocity_std=1.0, pitch_std=1.0, pitch_rate_std=1.0, seed=seed)
        for j in range(ticks):
            np.testing.assert_array_equal(sensor.measure(np.zeros(4)), table[j, :, s_i])


def test_spec_from_config_carries_every_setting():
    ec = CONFIG.build_eval_config()
    assert (SPEC.dt, SPEC.control_dt, SPEC.latency, SPEC.fall_angle) == (ec.dt, ec.control_dt, ec.latency,
                                                                          ec.fall_angle)
    assert SPEC.pitch_std == ec.sensor.pitch_std and SPEC.effort_weight == ec.effort_weight


def test_schedule_latency_and_goal():
    spec = replace(SPEC, goal=step_profile([(0.5, 0.3)]), duration=1.0, dt=0.0025, control_dt=0.005, latency=0.010)
    s = build_schedule(spec)
    assert (s.n_steps, s.steps_per_control, s.n_ticks) == (400, 2, 200)
    # 10 ms latency = 4 physics steps: command j (issued at step 2j) takes effect at step 2j + 4.
    assert list(s.active_command[:8]) == [-1, -1, -1, -1, 0, 0, 1, 1]
    assert s.ring_size == 3
    assert s.goal_per_tick[99] == 0.0 and s.goal_per_tick[100] == 0.3
    assert s.goal_per_step[199] == 0.0 and s.goal_per_step[200] == 0.3


def test_zero_latency_applies_command_immediately():
    s = build_schedule(replace(SPEC, goal=lambda t: 0.0, duration=0.1, dt=0.001, control_dt=0.001, latency=0.0))
    assert list(s.active_command[:5]) == [0, 1, 2, 3, 4]
    assert s.ring_size == 1


def test_negative_latency_rejected():
    with pytest.raises(ValueError):
        replace(SPEC, latency=-0.001)
