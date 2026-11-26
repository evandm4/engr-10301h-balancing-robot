"""Tests for sim_vs_real: the window stepper matches simulate(), and re-sync round-trips a sim-made "real" run."""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # hardware/legobot
from robot import control, sim_vs_real as svr  # noqa: E402

SEED = 7_300_101   # ad-hoc seeds >= 7.3M


@pytest.fixture(scope="module")
def cfg():
    return control.load_robot_config()


@pytest.fixture(scope="module")
def bundle(cfg):
    return control.load_policy_bundle("policy:l_e2_s1", cfg, allow_config_mismatch=True)


def _goals(n, cdt):
    t = np.arange(n) * cdt
    return np.where(t < 0.4, 0.0, np.where(t < 1.0, 0.12, -0.05))


def _controller(cfg, bundle, goal):
    return cfg.build_policy_controller(bundle["spec"], bundle["params"], goal=goal)


@pytest.mark.parametrize("noise", [False, True])
def test_run_window_matches_simulate(cfg, bundle, noise):
    """Fresh chain + reset controller + step0 = 0: bit-identical to simulate() (PDM, backlash, latency on)."""
    _, _, sim = svr._bs()
    dyn = cfg.build_dynamics()
    assert dyn.has_backlash and getattr(dyn.actuator, "pwm_tick", 0) > 0
    dt, cdt, lat = cfg.timing.dt_s, cfg.timing.control_dt_s, cfg.timing.latency_s
    goal = svr.goal_from_log(_goals(200, cdt), 0, cdt)
    x0 = np.array([0.0, 0.0, 0.05, -0.1])
    ref = sim.simulate(dyn, _controller(cfg, bundle, goal), duration=0.6, dt=dt, initial_state=x0, control_dt=cdt,
                       latency=lat, sensor=cfg.build_sensor(SEED) if noise else None)
    ctrl = _controller(cfg, bundle, goal)
    ctrl.reset()
    out = svr.run_window(dyn, ctrl, x0, len(ref.t) - 1, dt, cdt, svr.ActuatorChain(dyn, dt, lat),
                         sensor=cfg.build_sensor(SEED) if noise else None)
    assert not ref.fell and not out.fell
    assert np.array_equal(out.states, ref.states)
    assert np.array_equal(out.torques, ref.torques)


def test_latency_steps():
    assert svr.latency_steps(0.010, 0.0025) == 4
    assert svr.latency_steps(0.0095, 0.0025) == 4
    assert svr.latency_steps(0.0095, 0.0005) == 19
    assert svr.latency_steps(0.0, 0.0025) == 0


def _roundtrip(cfg, bundle, dyn, T, backlash_init, sensor="pipeline"):
    dt, cdt, lat = cfg.timing.dt_s, cfg.timing.control_dt_s, cfg.timing.latency_s
    n = 301   # 1.5 s
    log = svr.synthesize_log(cfg, bundle, dyn, _goals(n, cdt), np.array([0.3, 0.0, 0.04, 0.0]), dt, lat,
                             sensor=svr.build_sensor(cfg, sensor))
    ghost = svr.resync(log, cfg, bundle, T, dynamics=dyn, backlash_init=backlash_init, sensor=sensor)
    assert len(ghost.windows) == int(np.ceil((n - 1) / round(T / cdt)))
    return log, ghost


@pytest.mark.parametrize("sensor", ["pipeline", "ideal"])
def test_resync_roundtrip_without_backlash(cfg, bundle, sensor):
    """Zero model error (no play, so no hidden state): re-sync from the fake log reproduces it exactly,
    including the controller's integrators / history, the in-flight commands and PDM sum, and (pipeline)
    the complementary filter's estimate and the encoder's latest counts."""
    robot = replace(cfg.build_robot_params(), gearbox_backlash=0.0, coupling_backlash=0.0)
    from balance_sim import Planar2D
    dyn = Planar2D(robot, fall_angle=cfg.timing.fall_angle_rad, actuator=cfg.build_actuator(robot))
    assert not dyn.has_backlash
    log, ghost = _roundtrip(cfg, bundle, dyn, 0.25, "closed", sensor)
    assert np.max(np.abs(ghost.state - log.state)) < 1e-12
    assert np.nanmax(np.abs(ghost.command[:-1] - log.command[:-1])) < 1e-12
    assert ghost.first_command_err < 1e-12
    assert abs(log.int_v).max() > 0 and abs(log.history).max() > 0   # the state that had to be carried over


def test_resync_roundtrip_with_backlash_given_hidden_state(cfg, bundle):
    """With play, the hidden states (gap, motor speed, shaft, contact) are the only thing missing: given
    them, re-sync is exact; with the default (closed on the duty's side) it stays close. A 0.5 s window
    from a sensor that restarts from scratch instead (filter from the tilt, no encoder counts) is not exact,
    which is what init_sensor is for."""
    dyn = cfg.build_dynamics()
    log, ghost = _roundtrip(cfg, bundle, dyn, 0.25, "log")
    assert np.max(np.abs(ghost.state - log.state)) < 1e-12
    _, guess = _roundtrip(cfg, bundle, dyn, 0.25, "closed")
    err = svr.horizon_errors(log, guess)
    assert np.all(np.isfinite(err["theta"])) and err["theta"].max() < np.radians(3.0)


def test_init_sensor_matters(cfg, bundle, monkeypatch):
    robot = replace(cfg.build_robot_params(), gearbox_backlash=0.0, coupling_backlash=0.0)
    from balance_sim import Planar2D
    dyn = Planar2D(robot, fall_angle=cfg.timing.fall_angle_rad, actuator=cfg.build_actuator(robot))
    monkeypatch.setattr(svr, "init_sensor", lambda sensor, log, k0: sensor.reset())
    log, ghost = _roundtrip(cfg, bundle, dyn, 0.25, "closed")
    assert np.max(np.abs(ghost.state - log.state)) > 1e-6


def test_init_sensor_matches_real_log_speed(cfg, bundle):
    """A real log (no full_state): the encoder's first speed is the whole count nearest the logged v."""
    dyn = cfg.build_dynamics()
    dt, cdt, lat = cfg.timing.dt_s, cfg.timing.control_dt_s, cfg.timing.latency_s
    log = svr.synthesize_log(cfg, bundle, dyn, _goals(150, cdt), np.array([0.0, 0.0, -0.03, 0.0]), dt, lat,
                             sensor=svr.build_sensor(cfg))
    log = replace(log, full_state=None)
    for k0, v in ((37, 0.137), (120, -0.21)):
        log.state[k0, 1] = v
        sens = svr.build_sensor(cfg)
        svr.init_sensor(sens, log, k0)
        enc = sens._encoder
        got = enc.wheel_radius * (enc.speed() + log.state[k0, 3])
        assert abs(got - v) <= 0.5 * enc.wheel_radius * enc.speed_per_count + 1e-12


def test_init_controller_reproduces_logged_command(cfg, bundle):
    """init_controller at tick k0, then the logged state: the logged command at k0 (the replay check)."""
    dyn = cfg.build_dynamics()
    dt, cdt, lat = cfg.timing.dt_s, cfg.timing.control_dt_s, cfg.timing.latency_s
    log = svr.synthesize_log(cfg, bundle, dyn, _goals(150, cdt), np.array([0.0, 0.0, -0.03, 0.0]), dt, lat)
    for k0 in (0, 1, 37, 120):
        ctrl = cfg.build_policy_controller(bundle["spec"], bundle["params"], goal=svr.goal_from_log(log.goal, k0, cdt))
        svr.init_controller(ctrl, log, k0)
        assert ctrl.update(log.state[k0], cdt) == log.command[k0]


def test_wobble_stats_finds_frequency():
    dt = 0.005
    t = np.arange(4000) * dt
    y = 0.01 * np.sin(2 * np.pi * 2.5 * t) + 0.002 * t
    w = svr.wobble_stats(y, dt)
    assert abs(w["peak_hz"] - 2.5) < 0.3
    assert abs(w["rms"] - 0.01 / np.sqrt(2)) < 0.002
