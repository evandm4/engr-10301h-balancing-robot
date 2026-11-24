"""The backlash model's events, one at a time (Planar2D; PHYSICS.md 10.3, SOFTWARE_PLAN B5).

Whole runs against the batched engines are in test_equivalence.py and test_torch.py.
"""

import math

import numpy as np
import pytest

from balance_sim import IdealActuator, Planar2D, RobotParams, simulate
from balance_sim.planar2d import CONTACT, GAP, MOTOR_SPEED, SHAFT
from batch_sim import BatchRobot, SimSpec, simulate_batch
from config import CONFIG

GEARBOX, COUPLING = math.radians(2.8), math.radians(1.5)
RIGID = CONFIG.build_robot_params().with_changes(gearbox_backlash=0.0, coupling_backlash=0.0)
PLAY = RIGID.with_changes(gearbox_backlash=GEARBOX, coupling_backlash=COUPLING)
FALL = math.radians(60)


def _dyn(params=PLAY):
    return Planar2D(params, FALL, IdealActuator(params.max_torque))


def _momentum(p, state):
    """Generalized momentum of (x, theta) with the motor side's own speed (the open mass matrix plus J)."""
    _, x_dot, theta, theta_dot, _, motor_speed = state[:6]
    ml = p.body_mass * p.body_com_height
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
    b = ml * math.cos(theta)
    d = p.body_mass * p.body_com_height**2 + p.body_inertia
    j, r = p.rotor_inertia, p.wheel_radius
    return np.array([a * x_dot + b * theta_dot + j / r * motor_speed, b * x_dot + d * theta_dot - j * motor_speed])


def _kinetic(p, state):
    _, x_dot, theta, theta_dot, _, motor_speed = state[:6]
    ml = p.body_mass * p.body_com_height
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
    d = p.body_mass * p.body_com_height**2 + p.body_inertia
    return (0.5 * a * x_dot**2 + ml * math.cos(theta) * x_dot * theta_dot + 0.5 * d * theta_dot**2
            + 0.5 * p.rotor_inertia * motor_speed**2)


def test_zero_play_is_the_rigid_model():
    dyn = _dyn(RIGID)
    assert not dyn.has_backlash and len(dyn.initial_state(0.1)) == 4
    assert not BatchRobot.from_params(RIGID).has_backlash
    assert BatchRobot.from_params(PLAY).has_backlash


def test_closed_play_has_the_rigid_derivatives_bit_for_bit():
    rigid, play = _dyn(RIGID), _dyn(PLAY)
    s = np.array([0.1, 0.3, 0.05, -0.4])
    wheel_speed = s[1] / PLAY.wheel_radius - s[3]
    for contact in (1.0, -1.0):
        full = np.array([*s, contact * PLAY.backlash, wheel_speed, 0.0, contact])
        got = play.derivatives(full, 0.2, 0.7)
        np.testing.assert_array_equal(got[:4], rigid.derivatives(s, 0.2, 0.7))
        assert got[GAP] == got[MOTOR_SPEED] == 0.0


def test_open_play_leaves_the_wheel_unactuated():
    """Open: the motor torque only spins the motor side (J w' = tau); the body and wheels don't feel it."""
    dyn = _dyn(PLAY.with_changes(wheel_friction=0.0, coulomb_friction=0.0))
    s = np.array([0.0, 0.2, 0.05, 0.1, 0.0, 3.0, 0.0, 0.0])
    a, b = dyn.derivatives(s, 0.3), dyn.derivatives(s, -0.3)
    np.testing.assert_array_equal(a[:4], b[:4])
    assert a[MOTOR_SPEED] == pytest.approx(0.3 / PLAY.rotor_inertia)
    assert a[GAP] == pytest.approx(3.0 - (0.2 / PLAY.wheel_radius - 0.1))


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_engagement_is_inelastic_and_conserves_momentum(side):
    dyn = _dyn()
    r = PLAY.wheel_radius
    s = np.array([0.0, 0.1, 0.05, 0.2, side * 1.01 * PLAY.backlash, 0.1 / r - 0.2 + side * 8.0, -0.05, 0.0])
    after = dyn.end_step(s)
    assert after[CONTACT] == side and after[GAP] == side * PLAY.backlash
    assert after[MOTOR_SPEED] == after[1] / r - after[3]                  # moving together
    np.testing.assert_allclose(_momentum(PLAY, after), _momentum(PLAY, s), rtol=1e-12)
    assert _kinetic(PLAY, after) < _kinetic(PLAY, s)                       # restitution 0 loses energy
    np.testing.assert_array_equal(after[[0, 2]], s[[0, 2]])                # positions don't jump


def test_open_play_inside_the_dead_zone_stays_open():
    s = np.array([0.0, 0.1, 0.05, 0.2, 0.9 * PLAY.backlash, 5.0, 0.0, 0.0])
    after = _dyn().end_step(s)
    assert after[CONTACT] == 0.0
    np.testing.assert_array_equal(after[:6], s[:6])


def test_play_opens_when_the_transmitted_torque_reverses():
    """Closed on the + side (motor pushing forward): a backward drive pulls, so the play opens; a forward one
    keeps it closed."""
    dyn = _dyn()
    s = np.array([0.0, 0.0, 0.0, 0.0, PLAY.backlash, 0.0, -0.0, 1.0])
    assert dyn.begin_step(s, 0.3)[CONTACT] == 1.0
    assert dyn.begin_step(s, -0.3)[CONTACT] == 0.0
    s[CONTACT], s[GAP] = -1.0, -PLAY.backlash
    assert dyn.begin_step(s, -0.3)[CONTACT] == -1.0
    assert dyn.begin_step(s, 0.3)[CONTACT] == 0.0


def test_the_encoder_shaft_is_pushed_by_the_plays_and_otherwise_stays_put():
    dyn = _dyn()
    r = PLAY.wheel_radius
    # Closed forward: the shaft sits at the far end of the coupling play, ahead of the wheel.
    closed = dyn.end_step(np.array([0.02, 0.0, 0.0, 0.0, PLAY.backlash, 0.0, 0.0, 1.0]))
    assert closed[SHAFT] == pytest.approx(0.02 / r + COUPLING)
    # Open, motor side centred on the wheel: a shaft anywhere within both plays stays where it is.
    for shaft in (-0.5 * COUPLING, 0.0, 0.9 * COUPLING):
        s = dyn.end_step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, shaft, 0.0]))
        assert s[SHAFT] == shaft
    # Motor side far back in the gap: it drags the shaft back with it (gearbox play closed on the shaft).
    s = dyn.end_step(np.array([0.0, 0.0, 0.0, 0.0, -0.9 * PLAY.backlash, 0.0, 0.0, 0.0]))
    assert s[SHAFT] == pytest.approx(-0.9 * PLAY.backlash + GEARBOX)


def test_a_free_fall_crosses_the_play_and_engages():
    """The robot tips from rest with the motor off: the wheels roll back under the body's lean, the motor side
    (held only by its friction) lags, the play closes, and the run goes on with contact events."""
    dyn = _dyn(PLAY.with_changes(coulomb_friction=0.03, wheel_friction=0.002))

    class Off:
        def reset(self): pass
        def update(self, measured, dt): return 0.0
        def pitch_reference(self): return 0.0
        def goal_velocity(self, t): return 0.0

    res = simulate(dyn, Off(), duration=1.0, dt=0.0025, initial_state=dyn.initial_state(0.1))
    assert res.states.shape[1] == 8
    assert np.any(res.states[:, CONTACT] != 0)
    assert np.all(np.abs(res.states[:, GAP]) <= PLAY.backlash + 1e-12)


def test_play_needs_rotor_inertia_and_no_negative_values():
    with pytest.raises(ValueError, match="rotor_inertia"):
        PLAY.with_changes(rotor_inertia=0.0)
    with pytest.raises(ValueError, match="backlash"):
        RIGID.with_changes(coupling_backlash=-0.01)


def test_batched_play_must_be_positive_for_every_simulation():
    from batch_tuning import BatchEvaluator
    ev = BatchEvaluator(CONFIG.build_eval_config(), CONFIG.build_search_space())
    robot = BatchRobot.from_params(PLAY, gearbox_backlash=np.array([GEARBOX, 0.0]),
                                   coupling_backlash=np.array([COUPLING, 0.0]))
    with pytest.raises(ValueError, match="play must be positive"):
        simulate_batch(ev.spec, robot, np.tile(CONFIG.controller.hand_tuned_gains, (2, 1)), 0.05,
                       schedule=ev.schedule)
