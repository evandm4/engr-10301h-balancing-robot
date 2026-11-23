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


def test_drive_saturates_above_no_load_speed():
    act = SlowPwmActuator(MOTOR)
    w_nl = MOTOR.supply_voltage / MOTOR.torque_constant
    assert act.drive(0.1, w_nl * 1.1) == 1.0
    assert act.drive(-0.1, -w_nl * 1.1) == -1.0


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
