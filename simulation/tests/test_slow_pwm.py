"""SlowPwmActuator (SOFTWARE_PLAN B1b): the Build HAT's slow-PWM model and the robot code's inverse of it."""

import math

import pytest

from balance_sim import MotorParams, SlowPwmActuator

# The LEGO motor as measured (legobot_config): k 0.348, a = eta k / R = 0.0597 N m/V.
MOTOR = MotorParams(gear_ratio=1.0, torque_constant=0.348, resistance=4.4, supply_voltage=7.6,
                    current_limit=10.0, n_motors=2, gear_efficiency=0.755)
A = MOTOR.gear_efficiency * MOTOR.torque_constant / MOTOR.resistance


def test_zero_speed_torque_is_linear_in_duty_without_dead_band():
    act = SlowPwmActuator(MOTOR)
    for d in (0.05, 0.1, 0.3, 0.7, 1.0):
        assert act.torque(d, 0.0) == pytest.approx(2 * A * d * MOTOR.supply_voltage, rel=1e-12)
        assert act.torque(-d, 0.0) == pytest.approx(-2 * A * d * MOTOR.supply_voltage, rel=1e-12)


def test_zero_duty_coasts_at_any_speed():
    act = SlowPwmActuator(MOTOR)
    assert act.torque(0.0, 10.0) == 0.0
    assert act.torque(0.0, -10.0) == 0.0


def test_damping_scales_with_duty():
    """d(torque)/d(speed) = -n |d| a k: the spin-up time constant J / (|d| a k + b) of the bench data."""
    act = SlowPwmActuator(MOTOR)
    for d in (0.2, 0.5, 1.0):
        slope = act.torque(d, 1.0) - act.torque(d, 0.0)
        assert slope == pytest.approx(-2 * d * A * MOTOR.torque_constant, rel=1e-9)


def test_no_load_speed_saturates_like_the_bench():
    """With Coulomb friction tau_c per motor and no load, |d| a (V - k w) = tau_c gives
    w = V/k - tau_c / (|d| a k): the A - B/d shape measured in slow PWM (DATASHEET 8.7)."""
    tau_c = 0.022
    for d in (0.1, 0.5, 1.0):
        w = MOTOR.supply_voltage / MOTOR.torque_constant - tau_c / (d * A * MOTOR.torque_constant)
        assert SlowPwmActuator(MOTOR).torque(d, w) == pytest.approx(2 * tau_c, rel=1e-9)


@pytest.mark.parametrize("w", [-15.0, -3.0, 0.0, 2.5, 12.0])
@pytest.mark.parametrize("tau", [-0.6, -0.1, 0.0, 0.05, 0.4])
def test_drive_inverts_the_model_with_exact_constants(tau, w):
    act = SlowPwmActuator(MOTOR, max_torque=0.9)
    d = act.drive(tau, w)
    if abs(d) < 1.0:
        assert act.torque(d, w) == pytest.approx(tau, abs=1e-12)
    else:   # saturated: the most the motor can give in that direction at this speed
        assert math.copysign(1.0, d) == math.copysign(1.0, tau)
        assert abs(act.torque(d, w)) <= abs(tau) + 1e-12


def test_drive_coasts_at_or_above_no_load_speed_in_the_commanded_direction():
    """No duty gives torque toward the command there (a duty of its sign brakes), so drive() coasts.
    It used to send full duty of the command's sign: full braking for any command >= 0, zero included."""
    act = SlowPwmActuator(MOTOR, max_torque=0.9)
    w_nl = MOTOR.supply_voltage / MOTOR.torque_constant
    for tau in (0.0, -0.0, 1e-6, 0.1, 0.9):
        assert act.drive(tau, w_nl * 1.1) == 0.0
        assert act.drive(-tau, -w_nl * 1.1) == 0.0
    assert act.drive(0.1, w_nl) == 0.0                 # exactly at the no-load speed: den = 0
    assert act.drive(-0.1, w_nl * 1.1) < 0.0           # the other direction still drives: it has the headroom


@pytest.mark.parametrize("w", [-30.0, -21.9, 21.9, 30.0])
@pytest.mark.parametrize("tau", [-0.9, -0.1, 0.0, 0.1, 0.9])
def test_drive_never_gives_torque_against_the_command(tau, w):
    """Delivered torque has the command's sign or is zero, at any speed (exact constants)."""
    act = SlowPwmActuator(MOTOR, max_torque=0.9)
    assert act.torque(act.drive(tau, w), w) * tau >= -1e-12


def test_constant_mismatch_gives_an_error_proportional_to_the_command():
    """Unlike the linear voltage mode, wrong constants add no drag at zero command."""
    est = MotorParams(gear_ratio=1.0, torque_constant=0.348 * 1.1, resistance=4.4 * 0.9, supply_voltage=7.6,
                      current_limit=10.0, n_motors=2, gear_efficiency=0.755)
    act = SlowPwmActuator(MOTOR, est)
    assert act.torque(act.drive(0.0, 5.0), 5.0) == 0.0
    r1 = act.torque(act.drive(0.1, 0.0), 0.0) / 0.1
    r2 = act.torque(act.drive(0.2, 0.0), 0.0) / 0.2
    assert r1 == pytest.approx(r2, rel=1e-12)



# ---- pulse-density modulation (pwm_tick) ----
from balance_sim import pdm_update, pwm_steps_per_tick  # noqa: E402


@pytest.mark.parametrize("duty", [0.05, 0.1, 0.37, -0.2, 1.0, -1.0, 0.0])
def test_pdm_delivers_the_duty_on_average_within_one_pulse(duty):
    total, delivered = 0.0, 0.0
    for n in range(1, 401):
        total, out = pdm_update(total, duty)
        assert out in (-1.0, 0.0, 1.0)
        delivered += out
        assert abs(delivered - n * duty) <= 0.5 + 1e-9
    assert delivered / 400 == pytest.approx(duty, abs=1 / 400)


def test_pdm_at_exactly_half_is_a_coast_not_an_oscillation():
    """H11: total = +-0.5 followed by duty 0 stays 0 (the thresholds are strict), in every engine."""
    import numpy as np
    from batch_sim.physics import pdm_update as batch_pdm
    for total in (0.5, -0.5):
        assert pdm_update(total, 0.0) == (total, 0.0)
        for _ in range(5):
            total, out = pdm_update(total, 0.0)
            assert out == 0.0
        new, out = batch_pdm(np.array([0.5, -0.5, 0.75, -0.75]), np.zeros(4))
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, -1.0])
        np.testing.assert_array_equal(new, [0.5, -0.5, -0.25, 0.25])


def test_pdm_pulse_rate_is_100_times_duty_at_10_ms_ticks():
    """The measured pattern (DATASHEET 8.7): duty 0.05 gives one 10 ms pulse every 200 ms."""
    total, outs = 0.0, []
    for _ in range(100):
        total, out = pdm_update(total, 0.05)
        outs.append(out)
    starts = [i for i, o in enumerate(outs) if o]
    assert all(b - a == 20 for a, b in zip(starts, starts[1:]))


def test_pwm_steps_per_tick():
    assert pwm_steps_per_tick(0.0, 0.0025) == 0
    assert pwm_steps_per_tick(0.01, 0.0025) == 4
    with pytest.raises(ValueError):
        pwm_steps_per_tick(0.01, 0.003)



def test_pwm_tick_offset():
    from balance_sim import pwm_tick_offset
    assert pwm_tick_offset(0.25, 4) == 1
    assert pwm_tick_offset(0.0, 4) == 0
    assert pwm_tick_offset(0.999, 4) == 3
    assert pwm_tick_offset(1.0, 4) == 3       # clipped into the tick
    assert pwm_tick_offset(0.25, 20) == 5
    assert pwm_tick_offset(0.5, 0) == 0


class _Constant:
    """A controller that asks for a fixed torque from the first sample on (above duty 0.5, so the first tick pulses)."""

    def __init__(self, torque):
        self.torque = torque

    def reset(self):
        pass

    def update(self, measured, dt):
        return self.torque

    def pitch_reference(self):
        return 0.0

    def goal_velocity(self, t):
        return 0.0


@pytest.mark.parametrize("delay_steps", [3, 4, 5, 6, 7])
def test_locked_ticks_fall_where_each_command_takes_effect(delay_steps):
    """Review G6: with the loop locked to the HAT (pwm_tick_locked), the first pulse starts on the step the
    first command arrives, at every latency: the latency is the whole delay. A fixed phase instead waits
    for the next tick, up to a tick later."""
    from balance_sim import Planar2D, RobotParams, locked_pwm_tick_phase, simulate

    dt, control_dt, tick = 0.0025, 0.005, 0.010
    robot = RobotParams(body_mass=0.36, body_com_height=0.06, body_inertia=8.8e-4, wheel_mass=0.078,
                        wheel_radius=0.0345, wheel_inertia=6e-5, rotor_inertia=1.09e-3, max_torque=0.9, wheel_friction=0.0,
                        coulomb_friction=0.0, coulomb_friction_speed=1.0, gravity=9.81)

    def first_motion(phase):
        act = SlowPwmActuator(MOTOR, max_torque=0.9, pwm_tick=tick, pwm_tick_phase=phase)
        run = simulate(Planar2D(robot, fall_angle=1.5, actuator=act), _Constant(0.8), duration=0.06, dt=dt,
                       control_dt=control_dt, latency=delay_steps * dt)
        moving = [k for k, s in enumerate(run.states) if s[1] != 0.0]
        return moving[0] - 1          # the step whose torque first moved the wheel

    assert first_motion(locked_pwm_tick_phase(delay_steps, 4, 2)) == delay_steps
    late = first_motion(0.25 if delay_steps % 4 != 1 else 0.0)   # a phase that misses the arrival step
    assert delay_steps < late < delay_steps + 4


class _ReactToMotion:
    """Zero until a sample sees the robot move, then a fixed torque (duty above 0.5); never acts if torque is 0."""

    def __init__(self, torque):
        self.torque, self.on = torque, False

    def reset(self):
        self.on = False

    def update(self, measured, dt):
        self.on = self.on or any(abs(v) > 0 for v in measured[:4])
        return self.torque if self.on else 0.0

    def pitch_reference(self):
        return 0.0

    def goal_velocity(self, t):
        return 0.0


def _legobot_scenario():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "hardware" / "legobot" / "legobot_scenario.py"
    spec = importlib.util.spec_from_file_location("legobot_scenario", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("delay_steps", [3, 4, 5])
def test_free_tick_phase_reaction_is_within_the_catch_bound(delay_steps):
    """Bench B1: the HAT's ticks run at a free phase, so a command waits for the next tick. In the reference
    engine, over every phase, the first pulse that acts on a start tilt comes at most latency + tick - one
    step after the tilt, and on a push at most latency + control period + tick - one step after it starts
    (the push starting on a sample, which still sees the robot at rest). legobot_scenario.catch_latencies
    gives latency + tick and latency + control period + tick: one step more, the hardware's continuous wait,
    and both are reached to within that step at the worst phase."""
    from dataclasses import replace

    from balance_sim import Planar2D, RobotParams, simulate
    from legobot_config import LEGOBOT_CONFIG

    dt, control_dt, tick = 0.0025, 0.005, 0.010
    robot = RobotParams(body_mass=0.36, body_com_height=0.06, body_inertia=8.8e-4, wheel_mass=0.078,
                        wheel_radius=0.0345, wheel_inertia=6e-5, rotor_inertia=1.09e-3, max_torque=0.9,
                        wheel_friction=0.0, coulomb_friction=0.0, coulomb_friction_speed=1.0, gravity=9.81)
    push_step = 8                                      # a sample step (every 2nd), 20 ms in

    def first_effect(phase, tilt):
        """Steps from the disturbance to the step whose pulse first changed the motion."""
        runs = []
        for torque in (0.0, 0.9):
            act = SlowPwmActuator(MOTOR, max_torque=0.9, pwm_tick=tick, pwm_tick_phase=phase)
            plant = Planar2D(robot, fall_angle=1.5, actuator=act)
            start = plant.initial_state()
            start[2] = 0.05 if tilt else 0.0
            push = None if tilt else (lambda t: 2.0 if push_step * dt - 1e-9 <= t < (push_step + 4) * dt else 0.0)
            runs.append(simulate(plant, _ReactToMotion(torque), duration=0.1, dt=dt, control_dt=control_dt,
                                 latency=delay_steps * dt, initial_state=start, disturbance=push).states)
        differ = [k for k in range(min(len(runs[0]), len(runs[1]))) if (runs[0][k] != runs[1][k]).any()]
        return differ[0] - 1 - (0 if tilt else push_step)

    phases = [(j + 0.5) / 4 for j in range(4)]
    worst_tilt = max(first_effect(p, True) for p in phases)
    worst_push = max(first_effect(p, False) for p in phases)
    assert worst_tilt == delay_steps + 4 - 1
    assert worst_push == delay_steps + 2 + 4 - 1

    scenario = _legobot_scenario()
    cfg = replace(LEGOBOT_CONFIG, actuator=replace(LEGOBOT_CONFIG.actuator, pwm_tick_locked=False),
                  timing=replace(LEGOBOT_CONFIG.timing, dt_s=dt, control_dt_s=control_dt))
    bound_tilt, bound_push = scenario.catch_latencies(cfg, delay_steps * dt)
    assert bound_tilt == pytest.approx((worst_tilt + 1) * dt)
    assert bound_push == pytest.approx((worst_push + 1) * dt)
    locked = replace(cfg, actuator=replace(cfg.actuator, pwm_tick_locked=True))
    assert scenario.catch_latencies(locked, delay_steps * dt) == pytest.approx((delay_steps * dt, delay_steps * dt + tick))


def test_locked_tick_must_be_whole_control_periods():
    from balance_sim import locked_pwm_tick_phase

    assert locked_pwm_tick_phase(6, 4, 2) == pytest.approx(2.5 / 4)
    with pytest.raises(ValueError):
        locked_pwm_tick_phase(6, 3, 2)
