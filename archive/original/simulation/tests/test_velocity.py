import numpy as np
import pytest

from balance_sim import (
    CascadedPIDController,
    DCMotorActuator,
    MotorParams,
    PIDController,
    Planar2D,
    RobotParams,
    SensorNoise,
    compute_metrics,
    cost,
    simulate,
)

INNER = (4.0, 0.5, 0.15)


@pytest.fixture
def dyn():
    return Planar2D(RobotParams())


def make(kv_p=0.1, kv_i=0.03, **kwargs):
    return CascadedPIDController(PIDController(*INNER), kv_p, kv_i, **kwargs)


def run(dyn, controller, duration=9.0, pitch=0.1, **kwargs):
    return simulate(dyn, controller, duration=duration,
                    initial_state=dyn.initial_state(pitch), control_dt=0.005, **kwargs)


def at(result, t):
    return result.states[int(round(t / 0.001)), 1]


# ---- goal profiles ---------------------------------------------------------

def test_step_profile_values():
    from balance_sim import step_profile
    prof = step_profile([(1.0, 0.3), (5.0, -0.2)])
    assert prof(0.0) == 0.0
    assert prof(0.999) == 0.0
    assert prof(1.0) == 0.3
    assert prof(4.9) == 0.3
    assert prof(5.0) == -0.2
    assert prof(100.0) == -0.2


def test_controllers_without_a_velocity_loop_have_goal_zero(dyn):
    assert PIDController(4.0).goal_velocity(3.0) == 0.0
    result = run(dyn, PIDController(*INNER), duration=1.0)
    assert np.all(result.velocity_goal == 0.0)
    assert len(result.velocity_goal) == len(result.t)


def test_result_records_goal_profile(dyn):
    from balance_sim import step_profile
    result = run(dyn, make(goal=step_profile([(1.0, 0.3)])), duration=2.0)
    assert result.velocity_goal[500] == 0.0
    assert result.velocity_goal[1500] == pytest.approx(0.3)


# ---- cascaded controller ---------------------------------------------------

def test_cascade_tracks_step_velocity_commands(dyn):
    from balance_sim import step_profile
    ctrl = make(kv_p=0.15, kv_i=0.0, goal=step_profile([(1.0, 0.3), (5.0, -0.2)]))
    result = run(dyn, ctrl)
    assert not result.fell
    assert at(result, 4.9) == pytest.approx(0.3, abs=0.03)
    assert at(result, 8.9) == pytest.approx(-0.2, abs=0.03)


def test_moving_forward_requires_leaning_forward(dyn):
    ctrl = make(kv_p=0.15, kv_i=0.0, goal=0.3)
    result = run(dyn, ctrl, duration=1.5, pitch=0.0)
    assert result.states[:, 2].max() > 0.02      # leaned forward to accelerate
    assert result.states[-1, 1] > 0.1            # and is now moving forward


def test_velocity_loop_reduces_drift_from_gyro_bias(dyn):
    bias = SensorNoise(pitch_rate_bias=0.05)
    pitch_only = run(dyn, PIDController(*INNER), duration=10.0, pitch=0.05, sensor=bias)
    cascaded = run(dyn, make(), duration=10.0, pitch=0.05, sensor=bias)
    assert abs(cascaded.states[-1, 0]) < 0.25 * abs(pitch_only.states[-1, 0])
    assert abs(cascaded.states[-1, 1]) < 0.25 * abs(pitch_only.states[-1, 1])


def test_pitch_command_is_limited(dyn):
    ctrl = make(kv_p=10.0, goal=5.0, max_pitch_command=0.2)
    ctrl.reset()
    ctrl.update(dyn.initial_state(0.0), 0.005)
    assert ctrl.pitch_pid.setpoint == pytest.approx(0.2)


def test_reset_restarts_the_clock_and_integrator(dyn):
    ctrl = make(goal=0.3)
    ctrl.update(dyn.initial_state(0.0), 0.005)
    ctrl.reset()
    assert ctrl._time == 0.0 and ctrl._integral == 0.0


def test_from_gains_builds_both_loops():
    ctrl = CascadedPIDController.from_gains([4.0, 0.5, 0.15, 0.1, 0.03])
    assert (ctrl.pitch_pid.kp, ctrl.pitch_pid.ki, ctrl.pitch_pid.kd) == (4.0, 0.5, 0.15)
    assert (ctrl.kv_p, ctrl.kv_i) == (0.1, 0.03)


def test_cascade_still_balances_under_harder_conditions():
    from balance_sim import step_profile
    real = Planar2D(RobotParams(), actuator=DCMotorActuator(MotorParams()))
    ctrl = make(goal=step_profile([(1.0, 0.3), (5.0, -0.2)]))
    result = run(real, ctrl, sensor=SensorNoise.typical(), latency=0.01)
    assert not result.fell


# ---- metrics and cost ------------------------------------------------------

def test_velocity_error_metric_reflects_tracking(dyn):
    from balance_sim import step_profile
    prof = step_profile([(1.0, 0.3)])
    metrics = compute_metrics(run(dyn, make(kv_p=0.15, kv_i=0.0, goal=prof), duration=6.0))
    # A robot that ignored the goal would sit at 0 m/s and miss by 0.3 m/s for
    # 5 of 6 seconds, an RMS error of about 0.27 m/s.
    assert metrics.velocity_rms_error < 0.15


def test_velocity_term_increases_cost_when_drifting(dyn):
    bias = SensorNoise(pitch_rate_bias=0.05)
    result = run(dyn, PIDController(*INNER), duration=10.0, pitch=0.05, sensor=bias)
    assert cost(result, velocity_weight=1.0) > cost(result, velocity_weight=0.0)


def test_cost_prefers_the_controller_that_holds_position(dyn):
    bias = SensorNoise(pitch_rate_bias=0.05)
    pitch_only = run(dyn, PIDController(*INNER), duration=10.0, pitch=0.05, sensor=bias)
    cascaded = run(dyn, make(), duration=10.0, pitch=0.05, sensor=bias)
    assert cost(cascaded) < cost(pitch_only)
