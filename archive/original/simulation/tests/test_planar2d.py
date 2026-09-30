import numpy as np
import pytest

from balance_sim import (
    Planar2D,
    PIDController,
    RobotParams,
    compute_metrics,
    cost,
    simulate,
)


@pytest.fixture
def dyn():
    return Planar2D(RobotParams())


def test_upright_equilibrium_holds(dyn):
    d = dyn.derivatives(dyn.initial_state(0.0), 0.0)
    assert np.allclose(d, 0.0)


def test_uncontrolled_robot_falls(dyn):
    result = simulate(dyn, PIDController(0.0), duration=3.0, initial_state=dyn.initial_state(0.05))
    assert result.fell
    assert 0.1 < result.fall_time < 1.0


def test_positive_torque_pushes_wheels_forward_and_body_back(dyn):
    d = dyn.derivatives(dyn.initial_state(0.0), 0.1)
    assert d[1] > 0   # x_ddot
    assert d[3] < 0   # theta_ddot


def test_leaning_forward_accelerates_pitch_forward_without_torque(dyn):
    d = dyn.derivatives(dyn.initial_state(0.1), 0.0)
    assert d[3] > 0


def test_energy_conserved_without_torque_or_friction(dyn):
    state = dyn.initial_state(0.3)
    e0 = dyn.energy(state)
    from balance_sim.simulator import _rk4_step
    for _ in range(300):
        state = _rk4_step(dyn, state, 0.0, 0.001)
    assert abs(dyn.energy(state) - e0) / abs(e0) < 1e-6


def test_friction_dissipates_energy():
    dyn = Planar2D(RobotParams(wheel_friction=0.05))
    state = dyn.initial_state(0.3)
    e0 = dyn.energy(state)
    from balance_sim.simulator import _rk4_step
    for _ in range(300):
        state = _rk4_step(dyn, state, 0.0, 0.001)
    assert dyn.energy(state) < e0


def test_torque_is_clipped_to_motor_limit(dyn):
    limit = dyn.p.max_torque
    over = dyn.derivatives(dyn.initial_state(0.1), 100 * limit)
    at_limit = dyn.derivatives(dyn.initial_state(0.1), limit)
    assert np.allclose(over, at_limit)


def test_pid_balances_small_perturbation(dyn):
    ctrl = PIDController(kp=4.0, ki=0.5, kd=0.15)
    result = simulate(dyn, ctrl, duration=5.0, initial_state=dyn.initial_state(0.1), control_dt=0.005)
    m = compute_metrics(result)
    assert not m.fell
    assert m.settling_time is not None and m.settling_time < 1.0
    assert abs(result.states[-1, 2]) < 0.01


def test_gain_threshold_matches_simulation(dyn):
    kp_crit = dyn.critical_pitch_gain()
    start = dyn.initial_state(0.1)
    below = simulate(dyn, PIDController(0.5 * kp_crit, kd=0.05), duration=3.0, initial_state=start)
    above = simulate(dyn, PIDController(2.5 * kp_crit, kd=0.05), duration=3.0, initial_state=start)
    assert below.fell
    assert not above.fell


def test_cost_ranks_falling_worse_than_balancing(dyn):
    good = simulate(dyn, PIDController(4.0, 0.5, 0.15), duration=5.0,
                    initial_state=dyn.initial_state(0.1), control_dt=0.005)
    bad = simulate(dyn, PIDController(0.1, 0.0, 0.05), duration=5.0,
                   initial_state=dyn.initial_state(0.1), control_dt=0.005)
    assert cost(bad) > cost(good)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        RobotParams(wheel_radius=0.0)
