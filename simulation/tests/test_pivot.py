"""Tests for pivot_sim, the side-project robot with a servo-driven pivot and a
second body (see PIVOT.md). Built from pivot_config.PIVOT_CONFIG."""

import json
import math

import numpy as np
import pytest

from balance_sim import DCMotorActuator, IdealActuator, Planar2D, critical_pitch_gain, simulate
from pivot_config import PIVOT_CONFIG as PC
from pivot_sim import (
    PivotCascadedController,
    PivotLQRController,
    PivotRobot,
    PivotSensor,
    ServoActuator,
    lqr_gain,
    simulate_pivot,
)

RNG = np.random.default_rng(12345)
PARAMS = PC.build_params()
SERVO = PC.servo.build()


def random_states(n, scale=1.0):
    s = RNG.uniform(-1, 1, (n, 6)) * scale
    s[:, 2] *= 0.9   # keep the bodies within about 50 degrees of upright
    s[:, 4] *= 0.9
    return s


# ---- equations of motion ------------------------------------------------------

@pytest.mark.parametrize("dc_motor", [False, True])
def test_energy_changes_only_by_actuator_power(dc_motor):
    """dE/dt along the dynamics equals the power the wheel motors and servo put
    in. Checks the mass matrix and every Coriolis and gravity term at once."""
    dyn = PC.build_dynamics(ideal_actuator=not dc_motor)
    for s in random_states(300, 2.0):
        u = (RNG.uniform(-0.5, 0.5), RNG.uniform(-1, 1))
        f = dyn.derivatives(s, u)
        h = 1e-6
        dE = (dyn.energy(s + h * f) - dyn.energy(s - h * f)) / (2 * h)
        tau_w, tau_s = dyn.applied_torques(s, u)
        power = tau_w * (s[1] / PARAMS.wheel_radius - s[3]) + tau_s * s[5]
        assert dE == pytest.approx(power, abs=1e-6)


def test_massless_upper_body_reduces_to_base_robot():
    """With no upper mass and the servo exerting no torque, the wheels and lower
    body move exactly as balance_sim's Planar2D says they should."""
    params = PARAMS.with_changes(upper_mass=0.0, wheel_friction=0.002)
    dyn = PivotRobot(params, ServoActuator(SERVO), fall_angle=1.0)
    base = Planar2D(params.locked().with_changes(
        body_mass=params.lower_mass, body_com_height=params.lower_com_height,
        body_inertia=params.lower_inertia), fall_angle=1.0)
    for s in random_states(200, 2.0):
        s[4] = np.clip(s[4], -SERVO.angle_limit, SERVO.angle_limit)   # an in-range command
        s[5] = 0.0
        torque = RNG.uniform(-0.5, 0.5)
        pivot = dyn.derivatives(s, (torque, s[4]))   # command = angle: servo torque 0
        planar = base.derivatives(s[:4], torque)
        np.testing.assert_allclose(pivot[:4], planar, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("phi_deg", [-40.0, 0.0, 25.0])
def test_static_balance_with_pivot_bent(phi_deg):
    """Lower body at balance_pitch(phi), servo holding the upper body's weight:
    nothing accelerates, and the combined COM is straight over the axle."""
    dyn = PC.build_dynamics(ideal_actuator=True)
    phi = math.radians(phi_deg)
    theta1 = PARAMS.balance_pitch(phi)
    holding = -PARAMS.gravity * PARAMS.b2 * math.sin(theta1 + phi)   # servo torque that cancels gravity
    command = phi + holding / SERVO.stiffness
    state = np.array([0.0, 0.0, theta1, 0.0, phi, 0.0])
    np.testing.assert_allclose(dyn.derivatives(state, (0.0, command)), 0.0, atol=1e-12)
    assert PARAMS.com_pitch(theta1, phi)[0] == pytest.approx(0.0, abs=1e-12)


def test_com_pitch_and_rate():
    assert PARAMS.com_pitch(0.3, 0.0, 1.2, 0.0) == pytest.approx((0.3, 1.2))
    theta1, phi, theta1_dot, phi_dot, h = 0.2, -0.5, 0.7, 1.9, 1e-7
    angle, rate = PARAMS.com_pitch(theta1, phi, theta1_dot, phi_dot)
    ahead = PARAMS.com_pitch(theta1 + h * theta1_dot, phi + h * phi_dot)[0]
    behind = PARAMS.com_pitch(theta1 - h * theta1_dot, phi - h * phi_dot)[0]
    assert rate == pytest.approx((ahead - behind) / (2 * h), rel=1e-6)
    # Straight: the combined COM is the locked robot's COM.
    assert PARAMS.b1 + PARAMS.b2 == pytest.approx(PARAMS.total_mass * PARAMS.locked().body_com_height)


def test_locked_pivot_matches_rigid_robot():
    """Pivot held straight by the servo: the combined COM follows the rigid
    equivalent on balance_sim's own simulator, with the same controller. Not
    exactly: the servo is a stiff spring, not a weld, so the pivot flexes a
    little (about a degree) when the wheels accelerate hard."""
    gains = PC.base.controller.hand_tuned_gains
    start = math.radians(5.73)
    pivot = PC.run(PC.build_controller(gains), start, realistic=False)
    rigid_dyn = Planar2D(PARAMS.locked(), PC.base.timing.fall_angle_rad)
    rigid = simulate(rigid_dyn, PC.base.build_controller(gains), duration=PC.base.scenario.duration_s,
                     dt=PC.base.timing.dt_s, initial_state=rigid_dyn.initial_state(start),
                     control_dt=PC.base.timing.control_dt_s)
    assert not pivot.fell and not rigid.fell
    assert np.max(np.abs(pivot.com_pitch - rigid.states[:, 2])) < math.radians(0.25)
    assert np.max(np.abs(pivot.states[:, 1] - rigid.states[:, 1])) < 0.03
    assert abs(pivot.states[-1, 1] - rigid.states[-1, 1]) < 0.001
    assert np.max(np.abs(pivot.states[:, 4])) < math.radians(2.0)


def test_linearization_matches_derivatives_near_upright():
    dyn = PC.build_dynamics()
    A, B = dyn.linearize()
    x = np.array([0.0, 0.01, 0.002, -0.003, 0.004, 0.001])
    u = np.array([0.001, 0.002])
    np.testing.assert_allclose(dyn.derivatives(x, u), A @ x + B @ u, rtol=1e-4, atol=1e-6)
    assert np.max(np.linalg.eigvals(A).real) > 1.0   # upright is unstable


# ---- servo -----------------------------------------------------------------------

def test_servo_torque_speed_line_and_limits():
    servo = ServoActuator(SERVO)
    p = SERVO
    assert servo.torque(1.0, 0.0, 0.0) == pytest.approx(p.stall_torque)             # big error, at rest
    assert servo.torque(1.0, 0.0, p.no_load_speed) == pytest.approx(0.0, abs=1e-12)  # flat out
    assert servo.torque(1.0, 0.0, 0.5 * p.no_load_speed) == pytest.approx(0.5 * p.stall_torque)
    small = 0.1 * p.full_drive_error
    assert servo.torque(small, 0.0, 0.0) == pytest.approx(p.stiffness * small)
    assert servo.torque(1.0, 0.0, -p.no_load_speed) == pytest.approx(p.stall_torque)  # current-limited
    assert servo.clip_command(10.0) == p.angle_limit
    assert servo.clip_command(-10.0) == -p.angle_limit


def test_servo_takes_commands_once_per_frame():
    dyn = PC.build_dynamics(ideal_actuator=True)

    class Sweep(PivotCascadedController):
        def update(self, state, dt):
            self._k = getattr(self, "_k", 0) + 1
            return 0.0, 0.001 * self._k

    controller = Sweep(PC.base.build_controller(), PARAMS, 0, 0, 0, servo_limit=1.0)
    r = simulate_pivot(dyn, controller, duration=0.2, dt=PC.base.timing.dt_s,
                       control_dt=PC.base.timing.control_dt_s, latency=0.0)
    changes = np.flatnonzero(np.diff(r.servo_commands)) + 1
    frame = round(SERVO.update_period / PC.base.timing.dt_s)
    assert len(changes) > 3
    assert all(k % frame == 0 for k in changes)


def test_latency_delays_both_commands():
    from dataclasses import replace
    controller = PC.build_controller(PC.base.controller.hand_tuned_gains + (0.0, 1.0, 0.0))
    dt, latency = 0.001, 0.010
    dyn = PivotRobot(PARAMS, ServoActuator(replace(SERVO, update_period=0.0)), fall_angle=1.0)
    r = simulate_pivot(dyn, controller, duration=0.05, dt=dt, control_dt=0.005,
                       initial_state=dyn.initial_state(0.1), latency=latency)
    first_torque = np.flatnonzero(r.torques)[0]
    first_servo = np.flatnonzero(r.servo_commands)[0]
    assert first_torque == first_servo == round(latency / dt)


# ---- sensing -----------------------------------------------------------------------

def test_sensor_matches_base_noise_on_shared_channels():
    sensor = PC.build_sensor(seed=7)
    base = PC.base.build_sensor(seed=7)
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    first = [sensor.measure(state) for _ in range(5)]
    for m in first:
        np.testing.assert_array_equal(m[:4], base.measure(state[:4]))
    sensor.reset()
    np.testing.assert_array_equal([sensor.measure(state) for _ in range(5)], first)
    assert np.std([sensor.measure(state)[4] for _ in range(4000)]) == pytest.approx(
        math.radians(PC.sensor.pivot_angle_std_deg), rel=0.1)


# ---- controllers -----------------------------------------------------------------

def test_from_gains_checks_length_and_defaults_pivot_to_held():
    with pytest.raises(ValueError):
        PC.build_controller((1.0, 2.0, 3.0))
    controller = PC.build_controller(PC.base.controller.hand_tuned_gains)
    assert (controller.k_level, controller.k_phi_p, controller.k_phi_d) == (0.0, 0.0, 0.0)


def test_level_mode_keeps_upper_body_more_upright_than_held():
    start = math.radians(5.73)
    held = PC.metrics(PC.run(PC.build_controller(PC.hand_tuned_gains()), start, seed=1))
    level = PC.metrics(PC.run(PC.build_controller(PC.level_gains()), start, seed=1))
    assert not held.base.fell and not level.base.fell
    assert level.rms_upper_pitch < 0.5 * held.rms_upper_pitch


def test_lower_pitch_source_still_balances_with_pivot_held():
    controller = PivotCascadedController.from_gains(
        PC.hand_tuned_gains(), PARAMS, servo_limit=1.0, torque_limit=PARAMS.max_torque,
        pitch_integral_limit=1.0, pitch_source="lower", goal=0.0)
    r = PC.run(controller, math.radians(5.73), seed=1)
    assert not r.fell
    with pytest.raises(ValueError):
        PivotCascadedController(controller.wheel, PARAMS, 0, 0, 0, 1.0, pitch_source="upper")


def test_lqr_is_stable_and_beats_hand_tuned():
    dyn = PC.build_dynamics()
    timing = PC.base.timing
    K = lqr_gain(dyn, PC.lqr.build(), timing.control_dt_s, timing.latency_s)
    delay = round(timing.latency_s / timing.control_dt_s)
    assert K.shape == (2, 6 + 2 * delay)   # integral + 5 states + delayed commands
    controller = PC.build_lqr()
    assert controller.uses_integral and controller.n_delayed == delay
    start = math.radians(5.73)
    lqr = PC.run(controller, start, seed=1)
    held = PC.run(PC.build_controller(), start, seed=1)
    assert not lqr.fell
    assert PC.score(lqr) < PC.score(held)


def test_lqr_without_integral_or_latency():
    from dataclasses import replace
    weights = replace(PC.lqr.build(), velocity_integral=0.0)
    K = lqr_gain(PC.build_dynamics(ideal_actuator=True), weights, 0.005, 0.0)
    assert K.shape == (2, 5)
    controller = PivotLQRController(K, PARAMS.max_torque, 1.0)
    assert not controller.uses_integral and controller.n_delayed == 0
    r = PC.run(controller, math.radians(5.73), realistic=False)
    assert not r.fell
    assert abs(r.com_pitch[-1]) < math.radians(1.0)


def test_lqr_rejects_bad_gain_shape():
    with pytest.raises(ValueError):
        PivotLQRController(np.zeros((2, 4)), 1.0, 1.0)


# ---- cost, config, tuner -------------------------------------------------------------

def test_cost_adds_pivot_terms_to_base_cost():
    from dataclasses import replace
    r = PC.run(PC.build_controller(PC.level_gains()), math.radians(5.73), seed=1)
    plain = PC.base.score(r.as_planar("com"))
    no_extras = replace(PC, cost=replace(PC.cost, servo_effort_weight=0.0, upper_tilt_weight=0.0))
    assert no_extras.score(r) == pytest.approx(plain)
    tilted = replace(PC, cost=replace(PC.cost, servo_effort_weight=0.0, upper_tilt_weight=2.0))
    dt = np.diff(r.t)
    assert tilted.score(r) == pytest.approx(plain + 2.0 * np.sum(r.upper_pitch[:-1] ** 2 * dt))


def test_fall_ends_run_and_is_penalized():
    r = PC.run(PC.build_controller((0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), math.radians(5.73), seed=1)
    assert r.fell and r.fall_time < PC.base.scenario.duration_s
    assert PC.score(r) > PC.base.cost.fall_penalty


def test_config_builds_and_snapshots():
    dyn = PC.build_dynamics()
    assert isinstance(dyn.actuator, DCMotorActuator if PC.base.actuator.model == "dc_motor" else IdealActuator)
    assert isinstance(PC.build_dynamics(ideal_actuator=True).actuator, IdealActuator)
    assert isinstance(PC.build_sensor(3), PivotSensor) and PC.build_sensor(3).seed == 3
    assert PARAMS.total_mass == pytest.approx(PC.base.robot.body_mass)   # same robot, split in two
    space = PC.build_search_space()
    assert space.names == list(PC.gain_names)
    assert space.parameters[0].low == pytest.approx(
        PC.base.search.kp_min_factor * critical_pitch_gain(PARAMS.locked()))
    json.dumps(PC.snapshot())


def test_evaluator_is_deterministic_and_held_gains_match():
    from pivot_tuning import PivotEvaluator
    from dataclasses import replace

    quick = replace(PC, base=replace(PC.base, scenario=replace(
        PC.base.scenario, duration_s=2.0, start_pitches_deg=(5.73,), train_seeds=(1, 2))))
    ev = PivotEvaluator(quick)
    assert ev.space.dim == 7 and len(ev.scenarios()) == 2
    unit = np.full(7, 0.5)
    assert ev(unit) == ev(unit)
    held4 = ev.evaluate_gains(PC.base.controller.hand_tuned_gains)
    held7 = ev.evaluate_gains(PC.hand_tuned_gains())
    assert held4.costs == held7.costs



def test_lqr_with_pivot_held_leaves_servo_alone_and_does_worse():
    from dataclasses import replace
    held = PC.build_lqr(pivot_active=False)
    np.testing.assert_allclose(held.K[1], 0.0, atol=1e-9)
    upright = replace(PC, base=replace(PC.base, cost=replace(PC.base.cost, pitch_reference="upright")))
    start = math.radians(5.73)
    held_run = PC.run(held, start, seed=1)
    active_run = PC.run(PC.build_lqr(), start, seed=1)
    assert np.max(np.abs(held_run.states[:, 4])) < math.radians(2.0)
    assert upright.score(active_run) < upright.score(held_run)
