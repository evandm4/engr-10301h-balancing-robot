import numpy as np
import pytest

from balance_sim import (
    DCMotorActuator,
    IdealActuator,
    MotorParams,
    PIDController,
    Planar2D,
    RobotParams,
    SensorNoise,
    compute_metrics,
    simulate,
)

GAINS = (4.0, 0.5, 0.15)


def run(dyn, **kwargs):
    return simulate(
        dyn,
        PIDController(*GAINS),
        duration=5.0,
        initial_state=dyn.initial_state(0.1),
        control_dt=0.005,
        **kwargs,
    )


@pytest.fixture
def ideal():
    return Planar2D(RobotParams())


@pytest.fixture
def motor_dyn():
    return Planar2D(RobotParams(), actuator=DCMotorActuator(MotorParams()))


# ---- sensor noise ----------------------------------------------------------

def test_zero_noise_sensor_matches_no_sensor(ideal):
    a = run(ideal)
    b = run(ideal, sensor=SensorNoise())
    assert np.allclose(a.states, b.states)


def test_noise_is_reproducible_and_seed_dependent(ideal):
    a = run(ideal, sensor=SensorNoise.typical(seed=1))
    b = run(ideal, sensor=SensorNoise.typical(seed=1))
    c = run(ideal, sensor=SensorNoise.typical(seed=2))
    assert np.allclose(a.states, b.states)
    assert not np.allclose(a.states, c.states)


def test_noise_statistics_match_settings():
    sensor = SensorNoise(pitch_std=0.01, pitch_rate_bias=0.05, seed=3)
    truth = np.zeros(4)
    samples = np.array([sensor.measure(truth) for _ in range(5000)])
    assert abs(samples[:, 2].std() - 0.01) < 0.001
    assert abs(samples[:, 3].mean() - 0.05) < 0.001


# ---- latency ---------------------------------------------------------------

def test_zero_latency_is_the_default_behavior(ideal):
    a = run(ideal)
    b = run(ideal, latency=0.0)
    assert np.allclose(a.states, b.states)


def test_small_latency_is_tolerated_and_large_latency_is_not(ideal):
    assert not run(ideal, latency=0.005).fell
    assert run(ideal, latency=0.05).fell


def test_latency_delays_the_torque_command(ideal):
    result = run(ideal, latency=0.02)
    assert np.all(result.torques[:20] == 0.0)   # nothing arrives for the first 20 ms
    assert result.torques[21] != 0.0


def test_negative_latency_rejected(ideal):
    with pytest.raises(ValueError):
        run(ideal, latency=-0.01)


# ---- motor model -----------------------------------------------------------

def test_motor_delivers_command_at_standstill():
    act = DCMotorActuator(MotorParams())
    assert act.torque(0.1, 0.0) == pytest.approx(0.1)
    assert act.torque(-0.1, 0.0) == pytest.approx(-0.1)


def test_motor_current_limit_sets_stall_torque():
    m = MotorParams()
    act = DCMotorActuator(m)
    assert act.torque(10.0, 0.0) == pytest.approx(m.stall_torque_limit)


def test_back_emf_reduces_torque_as_speed_rises():
    m = MotorParams()
    act = DCMotorActuator(m)
    slow = act.torque(10.0, 0.0)
    half = act.torque(10.0, 0.5 * m.no_load_speed)
    at_limit = act.torque(10.0, m.no_load_speed)
    assert slow > half > at_limit
    assert at_limit == pytest.approx(0.0, abs=1e-9)


def test_motor_matches_ideal_actuator_when_not_saturated():
    p = RobotParams()
    ideal = Planar2D(p)
    motor = Planar2D(p, actuator=DCMotorActuator(MotorParams()))
    state = ideal.initial_state(0.1)
    assert np.allclose(ideal.derivatives(state, 0.2), motor.derivatives(state, 0.2))


def test_ideal_actuator_clips():
    act = IdealActuator(0.4)
    assert act.torque(5.0, 0.0) == 0.4
    assert act.torque(-5.0, 123.0) == -0.4


def test_invalid_motor_params_rejected():
    with pytest.raises(ValueError):
        MotorParams(resistance=0.0)


# ---- everything together ---------------------------------------------------

def test_robot_still_balances_under_combined_moderate_conditions(motor_dyn):
    result = run(motor_dyn, sensor=SensorNoise.typical(), latency=0.01)
    assert not result.fell
    assert compute_metrics(result).max_abs_pitch < 0.2
