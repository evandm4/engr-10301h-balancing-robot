import numpy as np
import pytest

from balance_sim import (
    Planar2D,
    PIDController,
    compute_metrics,
    simulate,
)
from config import CONFIG

ROBOT = CONFIG.build_robot_params()


@pytest.fixture
def dyn():
    return CONFIG.build_dynamics(ideal_actuator=True)


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
    dyn = Planar2D(ROBOT.with_changes(wheel_friction=0.05), CONFIG.timing.fall_angle_rad)
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
    assert CONFIG.score(bad) > CONFIG.score(good)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        ROBOT.with_changes(wheel_radius=0.0)


# ---- motor rotor inertia (RobotParams.rotor_inertia; PHYSICS.md 10.3) ----------------

def test_config_robot_has_reflected_rotor_inertia():
    a = CONFIG.actuator
    assert ROBOT.rotor_inertia == pytest.approx(a.n_motors * a.gear_ratio**2 * a.rotor_inertia)
    assert ROBOT.rotor_inertia > 0


def test_rotor_inertia_enters_the_mass_matrix_as_derived():
    """derivatives() solves M q'' = f with PHYSICS.md 10.3's mass matrix: the rotors spin at
    x'/r - theta' relative to the body, so J sits on a, d and (with -J/r) the coupling."""
    p = ROBOT
    dyn = Planar2D(p, fall_angle=10.0)
    state = np.array([0.0, 0.7, 0.3, -1.1])
    tau = 0.12
    _, x_dot, theta, theta_dot = state
    J, r = p.rotor_inertia, p.wheel_radius
    ml = p.body_mass * p.body_com_height
    M = np.array([
        [p.body_mass + p.wheel_mass + p.wheel_inertia / r**2 + J / r**2, ml * np.cos(theta) - J / r],
        [ml * np.cos(theta) - J / r, p.body_mass * p.body_com_height**2 + p.body_inertia + J],
    ])
    f = np.array([tau / r + ml * np.sin(theta) * theta_dot**2, -tau + ml * p.gravity * np.sin(theta)])
    x_ddot, theta_ddot = np.linalg.solve(M, f)
    d = dyn.derivatives(state, tau)
    np.testing.assert_allclose(d[[1, 3]], [x_ddot, theta_ddot], rtol=1e-12)


def test_rotor_inertia_is_not_the_same_as_wheel_inertia():
    """Folding J into the wheel inertia gets a right but misses the coupling and the body term."""
    folded = ROBOT.with_changes(wheel_inertia=ROBOT.wheel_inertia + ROBOT.rotor_inertia, rotor_inertia=0.0)
    state = np.array([0.0, 0.0, 0.1, 0.0])
    real = Planar2D(ROBOT, fall_angle=10.0).derivatives(state, 0.1)
    approx = Planar2D(folded, fall_angle=10.0).derivatives(state, 0.1)
    assert not np.allclose(real, approx, rtol=1e-3)


def test_energy_is_conserved_with_rotor_inertia():
    p = ROBOT.with_changes(rotor_inertia=5 * ROBOT.rotor_inertia)
    dyn = Planar2D(p, fall_angle=10.0)
    state = np.array([0.0, 0.3, 0.2, 0.5])
    e0 = dyn.energy(state)
    h = 1e-4
    for _ in range(2000):   # RK4, no torque, no friction
        k1 = dyn.derivatives(state, 0.0)
        k2 = dyn.derivatives(state + 0.5 * h * k1, 0.0)
        k3 = dyn.derivatives(state + 0.5 * h * k2, 0.0)
        k4 = dyn.derivatives(state + h * k3, 0.0)
        state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert dyn.energy(state) == pytest.approx(e0, abs=1e-9)


def test_rotor_inertia_must_be_non_negative():
    with pytest.raises(ValueError):
        ROBOT.with_changes(rotor_inertia=-1e-6)
