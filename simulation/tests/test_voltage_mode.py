"""The voltage-mode driver with back-EMF feedforward (balance_sim.VoltageModeActuator; SOFTWARE_PLAN B1)."""

from dataclasses import replace

import numpy as np
import pytest

from config import CONFIG
from balance_sim import DCMotorActuator, Planar2D, VoltageModeActuator, simulate

MOTOR = CONFIG.actuator.build_motor_params()
ROBOT = CONFIG.build_robot_params().with_changes(gearbox_backlash=0.0, coupling_backlash=0.0)   # rigid: compares two drivers, not the play
LIMIT = ROBOT.max_torque


def big_current_limit(m):
    return replace(m, current_limit=1e3)


@pytest.mark.parametrize("command", [-0.3, -0.05, 0.0, 0.02, 0.2])
@pytest.mark.parametrize("speed", [-10.0, 0.0, 3.0, 12.0])
def test_exact_feedforward_matches_the_current_mode_motor(command, speed):
    """Exact constants and the true wheel speed: the voltage mode delivers what current mode does."""
    m = big_current_limit(MOTOR)
    volt = VoltageModeActuator(m, m, max_torque=LIMIT, feedforward=True)
    current = DCMotorActuator(m, max_torque=LIMIT)
    assert volt.torque(volt.drive(command, speed), speed) == pytest.approx(current.torque(command, speed), abs=1e-12)


def test_feedforward_gain_scales_the_back_emf_term():
    """Exact constants: torque = current-mode torque + (gain - 1) n N eta k (k N w) / R."""
    m = big_current_limit(MOTOR)
    gain = 1.2
    volt = VoltageModeActuator(m, m, max_torque=LIMIT, feedforward=True, feedforward_gain=gain)
    current = DCMotorActuator(m, max_torque=LIMIT)
    k, N = m.torque_constant, m.gear_ratio
    for command in (-0.2, 0.0, 0.15):
        for w in (-8.0, 0.0, 5.0):
            extra = (gain - 1) * m.n_motors * N * m.gear_efficiency * k * (k * N * w) / m.resistance
            assert volt.torque(volt.drive(command, w), w) == pytest.approx(current.torque(command, w) + extra,
                                                                           abs=1e-12)


def test_plain_voltage_mode_is_viscous_drag_below_the_limits():
    """Feedforward off: tau = tau_c - c_m w, c_m = n N^2 eta k^2 / R (PHYSICS.md 9.3)."""
    m = big_current_limit(MOTOR)
    plain = VoltageModeActuator(m, m, max_torque=LIMIT, feedforward=False)
    c_m = m.n_motors * m.gear_ratio**2 * m.gear_efficiency * m.torque_constant**2 / m.resistance
    w = 0.1 * m.no_load_speed
    command = 0.25 * LIMIT
    assert plain.torque(plain.drive(command, w), w) == pytest.approx(command - c_m * w, rel=1e-12)


def test_drive_is_clipped_to_the_supply_and_the_command_to_max_torque():
    act = VoltageModeActuator(MOTOR, MOTOR, max_torque=LIMIT)
    assert act.drive(1e6, 0.0) == pytest.approx(min(MOTOR.supply_voltage, MOTOR.resistance * LIMIT / (
        MOTOR.n_motors * MOTOR.gear_ratio * MOTOR.gear_efficiency * MOTOR.torque_constant)))
    assert act.drive(1e6, 1e6) == MOTOR.supply_voltage
    assert act.drive(-1e6, -1e6) == -MOTOR.supply_voltage


def test_a_low_resistance_estimate_under_delivers():
    """The robot code's constants are its own: an R^ below the true R gives less torque than asked."""
    true = big_current_limit(MOTOR)
    act = VoltageModeActuator(true, replace(true, resistance=0.9 * true.resistance), max_torque=LIMIT)
    w = 0.2 * true.no_load_speed
    assert act.torque(act.drive(0.1, w), w) == pytest.approx(0.9 * 0.1, rel=1e-9)


def test_simulator_holds_the_drive_voltage_and_reports_the_torque_command():
    """The simulator passes drive() to the dynamics but records (and scores) the torque command;
    with exact constants it tracks the current-mode run closely (w^ is sampled per control tick)."""
    m = big_current_limit(MOTOR)
    runs = {}
    for name, act in (("current", DCMotorActuator(m, max_torque=LIMIT)),
                      ("voltage", VoltageModeActuator(m, m, max_torque=LIMIT))):
        dyn = Planar2D(ROBOT, CONFIG.timing.fall_angle_rad, act)
        runs[name] = simulate(dyn, CONFIG.build_controller(), duration=4.0, dt=CONFIG.timing.dt_s,
                              initial_state=dyn.initial_state(0.05), control_dt=CONFIG.timing.control_dt_s)
    cur, volt = runs["current"], runs["voltage"]
    assert not cur.fell and not volt.fell
    assert np.max(np.abs(volt.torques)) <= LIMIT + 1e-12        # torque commands, not volts
    assert np.max(np.abs(volt.states[:, 2] - cur.states[:, 2])) < 0.01
    assert CONFIG.score(volt) == pytest.approx(CONFIG.score(cur), rel=0.05)


def test_config_builds_the_voltage_model():
    cfg = replace(CONFIG, actuator=replace(CONFIG.actuator, model="voltage"))
    act = cfg.build_dynamics().actuator
    assert isinstance(act, VoltageModeActuator) and act.feedforward
    ec = cfg.build_eval_config()
    assert ec.voltage_estimate == cfg.actuator.build_motor_params() and ec.voltage_feedforward
    if CONFIG.actuator.model not in ("voltage", "slow_pwm"):    # slow PWM has the driver estimate too
        assert CONFIG.build_eval_config().voltage_estimate is None
