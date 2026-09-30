from dataclasses import replace

import numpy as np
import pytest

from balance_sim import (
    CascadedPIDController,
    PIDController,
    SensorNoise,
    compute_metrics,
    simulate,
)
from config import CONFIG

INNER = (4.0, 0.5, 0.15)
LIMITS = CONFIG.controller


@pytest.fixture
def dyn():
    return CONFIG.build_dynamics(ideal_actuator=True)


def inner():
    return PIDController(*INNER, integral_limit=LIMITS.pitch_integral_limit)


def make(kv_p=0.1, kv_i=0.03, **kwargs):
    """A cascaded controller with the project's lean cap and integral limits (overridable).

    These tests keep a nonzero kv_i by default: the project tunes it out, but
    the controller still supports it and these tests check that it works.
    """
    options = dict(max_pitch_command=LIMITS.max_lean_rad, integral_limit=LIMITS.velocity_integral_limit)
    options.update(kwargs)
    return CascadedPIDController(inner(), kv_p, kv_i, **options)


def score(result, **cost_changes):
    """The project's cost, with some weights changed."""
    return replace(CONFIG, cost=replace(CONFIG.cost, **cost_changes)).score(result)


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
    pitch_only = run(dyn, inner(), duration=10.0, pitch=0.05, sensor=bias)
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
    assert ctrl.pitch_pid.output_limit is None
    limited = CascadedPIDController.from_gains([4.0, 0.5, 0.15, 0.1, 0.03], torque_limit=0.4,
                                               pitch_integral_limit=0.7)
    assert limited.pitch_pid.output_limit == 0.4
    assert limited.pitch_pid.integral_limit == 0.7


def test_from_gains_with_four_gains_has_no_velocity_integral():
    ctrl = CascadedPIDController.from_gains([4.0, 0.5, 0.15, 0.1])
    assert (ctrl.kv_p, ctrl.kv_i) == (0.1, 0.0)
    with pytest.raises(ValueError):
        CascadedPIDController.from_gains([4.0, 0.5, 0.15])


def test_project_controller_uses_the_configured_limits():
    ctrl = CONFIG.build_controller()
    assert ctrl.kv_i == 0.0
    assert ctrl.max_pitch_command == pytest.approx(LIMITS.max_lean_rad)
    assert ctrl.integral_limit == LIMITS.velocity_integral_limit
    assert ctrl.pitch_pid.integral_limit == LIMITS.pitch_integral_limit
    assert ctrl.pitch_pid.output_limit == CONFIG.robot.max_torque


def test_reset_clears_the_pitch_setpoint(dyn):
    ctrl = make(goal=0.3)
    ctrl.update(dyn.initial_state(0.0), 0.005)
    assert ctrl.pitch_reference() != 0.0
    ctrl.reset()
    assert ctrl.pitch_reference() == 0.0


# ---- anti-windup -----------------------------------------------------------

def pitch_state(theta):
    return np.array([0.0, 0.0, theta, 0.0])


def test_pid_output_is_clipped_to_the_limit():
    pid = PIDController(kp=100.0, output_limit=0.4)
    assert pid.update(pitch_state(0.5), 0.005) == pytest.approx(0.4)
    assert pid.update(pitch_state(-0.5), 0.005) == pytest.approx(-0.4)


def test_pid_integral_never_asks_for_more_than_the_limit():
    pid = PIDController(kp=0.01, ki=5.0, output_limit=0.4)
    for _ in range(2000):
        pid.update(pitch_state(0.02), 0.005)
    assert abs(pid.ki * pid._integral) <= 0.4 + 1e-12


def test_pid_recovers_from_saturation_without_windup():
    # Hold a large error long enough to saturate, then reverse it. Without a
    # limit the integrator banks up torque that keeps pushing the wrong way
    # after the error flips; with it, the output reverses straight away.
    limited = PIDController(kp=1.0, ki=5.0, output_limit=0.4)
    unlimited = PIDController(kp=1.0, ki=5.0)
    for _ in range(400):
        limited.update(pitch_state(0.5), 0.005)
        unlimited.update(pitch_state(0.5), 0.005)
    assert limited.update(pitch_state(-0.05), 0.005) < 0.0
    assert unlimited.update(pitch_state(-0.05), 0.005) > 0.0


def test_pid_integrator_unwinds_when_error_reverses():
    pid = PIDController(kp=0.01, ki=5.0, output_limit=0.4)
    for _ in range(2000):
        pid.update(pitch_state(0.02), 0.005)
    before = pid._integral
    pid.update(pitch_state(-0.02), 0.005)
    assert pid._integral < before


def test_velocity_integrator_limited_by_lean_cap(dyn):
    ctrl = make(kv_p=0.0, kv_i=0.3, goal=1.0, max_pitch_command=0.05)
    for _ in range(1000):
        ctrl.update(dyn.initial_state(0.0), 0.005)
    assert abs(ctrl.kv_i * ctrl._integral) <= 0.05 + 1e-12
    assert ctrl.pitch_pid.setpoint == pytest.approx(0.05)


def test_cascade_still_balances_under_harder_conditions():
    from balance_sim import step_profile
    real = CONFIG.build_dynamics()   # the configured DC motor
    ctrl = make(goal=step_profile([(1.0, 0.3), (5.0, -0.2)]))
    result = run(real, ctrl, sensor=CONFIG.build_sensor(), latency=CONFIG.timing.latency_s)
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
    result = run(dyn, inner(), duration=10.0, pitch=0.05, sensor=bias)
    assert score(result, velocity_weight=1.0) > score(result, velocity_weight=0.0)


def test_cost_prefers_the_controller_that_holds_position(dyn):
    bias = SensorNoise(pitch_rate_bias=0.05)
    pitch_only = run(dyn, inner(), duration=10.0, pitch=0.05, sensor=bias)
    cascaded = run(dyn, make(), duration=10.0, pitch=0.05, sensor=bias)
    assert CONFIG.score(cascaded) < CONFIG.score(pitch_only)


# ---- pitch reference -------------------------------------------------------

def test_simulator_records_the_commanded_pitch(dyn):
    from balance_sim import step_profile
    result = run(dyn, make(kv_p=0.15, goal=step_profile([(1.0, 0.3)])), duration=2.0, pitch=0.0)
    assert len(result.pitch_reference) == len(result.t)
    assert np.all(result.pitch_reference[:900] == 0.0)   # goal is 0 and robot starts still
    assert result.pitch_reference[1001] > 0.0            # leans forward once the goal steps up


def test_pitch_only_controller_references_upright(dyn):
    result = run(dyn, inner(), duration=1.0)
    assert np.all(result.pitch_reference == 0.0)


def test_commanded_reference_does_not_penalize_required_lean(dyn):
    from balance_sim import pitch_error_signal, step_profile
    result = run(dyn, make(kv_p=0.15, goal=step_profile([(1.0, 0.3)])), duration=4.0, pitch=0.0)
    commanded = score(result, pitch_reference="commanded")
    upright = score(result, pitch_reference="upright")
    assert commanded < upright
    assert np.allclose(pitch_error_signal(result, "upright"), result.states[:, 2])


def test_commanded_and_upright_agree_without_a_velocity_loop(dyn):
    result = run(dyn, inner(), duration=2.0)
    assert score(result, pitch_reference="commanded") == score(result, pitch_reference="upright")


def test_unknown_pitch_reference_rejected(dyn):
    from balance_sim import cost
    result = run(dyn, inner(), duration=0.5)
    c = CONFIG.cost
    with pytest.raises(ValueError):
        cost(result, c.effort_weight, c.fall_penalty, c.velocity_weight, "sideways")
