"""Tests for the balancing loop's hardware-free parts (control.py) and the whole loop against the simulator
(balance.py --sim through sim_hw.py). No Pi needed.

    python -m pytest hardware/legobot/robot/test_control.py -q          (from the repo root)
"""
import json
import math
import os
import sys
from collections import namedtuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # hardware/legobot
from robot import balance, control, loop_config as lc  # noqa: E402

M = namedtuple("M", "t speed pos apos")


@pytest.fixture(scope="module")
def cfg():
    return control.load_robot_config()


def test_wheel_speed_matches_encoder_differencing():
    assert control.wheel_speed((None, M(0, 0, 5, 5)), 0.01) == 0.0
    # 3 counts in one 10 ms period, whatever the arrival times
    assert control.wheel_speed((M(0.000, 0, 10, 10), M(0.0001, 0, 13, 13)), 0.01) == pytest.approx(math.radians(300))


def test_feedforward_is_the_simulators_drive(cfg):
    """The robot's duty is the config actuator's own drive() (a duty for slow_pwm, volts / V for voltage)."""
    ff = control.Feedforward(cfg)
    act = cfg.build_actuator()
    scale = cfg.actuator.supply_voltage if cfg.actuator.model == "voltage" else 1.0
    for tau, w in [(0.0, 0.0), (0.1, 3.0), (-0.2, -5.0), (0.3, -4.0), (0.5, 15.0)]:
        dl, dr = ff.duties(tau, w, w)
        assert dl == dr
        assert dl * scale == pytest.approx(act.drive(tau, w))
    # each wheel its own back-EMF
    dl, dr = ff.duties(0.1, 2.0, 4.0)
    assert dl * scale == pytest.approx(act.drive(0.1, 2.0)) and dr * scale == pytest.approx(act.drive(0.1, 4.0))
    assert dr > dl


def test_feedforward_uses_measured_battery(cfg):
    ff = control.Feedforward(cfg)
    ff.set_supply(8.2)
    assert abs(ff.duties(0.5, 30.0, 30.0)[0]) == pytest.approx(1.0)       # clipped at the real supply
    d76 = control.Feedforward(cfg).duties(0.05, 0.0, 0.0)[0]
    assert ff.duties(0.05, 0.0, 0.0)[0] == pytest.approx(d76 * 7.6 / 8.2)


def test_sense_conventions(cfg):
    controller, _ = control.build_controller(cfg, "hand")
    brain = control.Brain(cfg, controller, control.PitchEstimator(cfg, gyro_bias=0.01, pitch_zero=0.02),
                          control.Feedforward(cfg), 1.0, 0.01)
    g = 9.80665
    tilt = 0.1
    pair = (M(0, 0, 0, 0), M(0.01, 0, 2, 2))       # 2 counts in 10 ms
    state, ex = brain.sense(0.51, -g * math.sin(tilt), g * math.cos(tilt), pair, pair)
    assert ex["tilt"] == pytest.approx(tilt - 0.02)
    assert state[2] == pytest.approx(tilt - 0.02)   # first filter update starts at the tilt
    assert state[3] == pytest.approx(0.50)
    r = cfg.robot.wheel_radius
    assert state[1] == pytest.approx(r * (math.radians(200) + 0.50))


def test_safety_trips(cfg):
    dt = cfg.timing.control_dt_s
    ok = dict(motor_ages=(0.005, 0.005), vin_reading=(0.0, 7.6), fault=None, reader_alive=True, running=True)
    s = control.Safety(dt)
    assert s.check(0.0, 0.0, (0.0, 0.0), (0.0, 0.0), **ok) is None
    assert "tilt" in s.check(0.0, 0.8, (0.0, 0.0), (0.0, 0.0), **ok)
    assert s.check(0.0, 0.8, (0.0, 0.0), (0.0, 0.0), **{**ok, "running": False}) is None   # not armed
    assert "stale" in s.check(0.0, 0.0, (0, 0), (0, 0), **{**ok, "motor_ages": (0.005, 0.2)})
    assert "fault" in s.check(0.0, 0.0, (0, 0), (0, 0), **{**ok, "fault": (0.0, "Motor power fault")})

    s = control.Safety(dt)                        # stall: STALL_S at high duty without motion
    n = int(round(lc.STALL_S / dt))
    results = [s.check(k * dt, 0.0, (0.9, 0.0), (0.0, 20.0), **ok) for k in range(n)]
    assert results[-2] is None and "left wheel stalled" in results[-1]

    s = control.Safety(dt)                        # vin: VIN_LOW_COUNT distinct low readings
    low = [s.check(k * 0.5, 0.0, (0, 0), (0, 0), **{**ok, "vin_reading": (k * 0.5, 6.3)}) for k in range(3)]
    assert low[:2] == [None, None] and "battery" in low[2]
    s = control.Safety(dt)                        # the same reading seen many ticks counts once
    assert all(s.check(k * dt, 0.0, (0, 0), (0, 0), **{**ok, "vin_reading": (0.0, 6.3)}) is None for k in range(50))

    s = control.Safety(dt)                        # wrong way: duty forward, wheel fast backward
    out = [s.check(k * dt, 0.0, (0.8, 0.0), (-10.0, 0.0), **ok) for k in range(60)]
    assert any(o and "against its duty" in o for o in out)


def test_gyro_calibration_rejects_motion():
    g = 9.80665
    still = [(0.01 + 0.002 * ((k % 5) - 2), 0.0, g) for k in range(200)]
    bias, sd, ratio = control.gyro_bias_from(still)
    assert bias == pytest.approx(0.01) and ratio == pytest.approx(1.0)
    moving = [(0.3 * math.sin(k / 10), 0.0, g) for k in range(200)]
    with pytest.raises(ValueError):
        control.gyro_bias_from(moving)


def test_gains_file_formats(tmp_path, cfg):
    p = tmp_path / "pid.json"
    p.write_text(json.dumps({"gains": {"kp": 5.0, "ki": 1.0, "kd": 0.4, "kv_p": 0.3}, "feedforward_gain": 1.05}))
    c, d = control.build_controller(cfg, str(p))
    assert d["kind"] == "pid" and d["gains"] == [5.0, 1.0, 0.4, 0.3] and d["feedforward_gain"] == 1.05
    assert c.goal == 0.0 and c.pitch_pid.output_limit == cfg.robot.max_torque
    assert c.max_pitch_command == pytest.approx(cfg.controller.max_lean_rad)
    q = tmp_path / "lqr.json"
    q.write_text(json.dumps({"K": [-4.0, -4.7, -0.5, 0.0, 0.1, 0.1]}))
    c, d = control.build_controller(cfg, str(q), goal="scenario")
    assert d["kind"] == "lqr" and callable(c.goal)
    w = tmp_path / "weights.json"
    w.write_text(json.dumps({"gains": {"s_velocity": 1.0, "s_pitch": 1.0}}))
    with pytest.raises(ValueError):
        control.build_controller(cfg, str(w))


@pytest.mark.parametrize("controller", ["hand", "model", "lqr"])
def test_closed_loop_with_simulator(tmp_path, controller):
    from balance_sim import load_trajectory
    reason = balance.main(["--sim", "--sim-noise", "--goal", "scenario", "--controller", controller,
                           "--seconds", "20", "--outdir", str(tmp_path), "--label", controller])
    assert reason == "done"
    (run_csv,) = tmp_path.glob("*_run.csv")
    result, meta = load_trajectory(run_csv)
    assert meta["source"] == "sim_hw" and not result.fell
    assert result.t[-1] == pytest.approx(20.0, abs=0.01)
    cruise = (result.t > 5.0) & (result.t < 9.0)                 # goal 0.27 m/s
    assert abs(result.states[cruise, 1].mean() - 0.27) < 0.07
    assert abs(result.states[:, 2]).max() < 0.3


@pytest.mark.parametrize("inject, expect", [("loop", "loop stalled"), ("stale", "stale"), ("vin", "battery"),
                                            ("fault", "fault")])
def test_injected_trips(tmp_path, inject, expect):
    reason = balance.main(["--sim", "--seconds", "3", "--inject", inject, "--outdir", str(tmp_path)])
    assert expect in reason


def test_too_weak_to_balance_falls_and_trips(tmp_path):
    """With the duty capped far below what balancing needs, the robot falls and the tilt trip ends the run."""
    reason = balance.main(["--sim", "--duty-cap", "0.02", "--sim-pitch", "0.045", "--seconds", "10",
                           "--outdir", str(tmp_path)])
    assert reason.startswith("tilt")


def test_every_robot_module_compiles():
    """The Pi-only modules (pyserial, smbus2, fcntl) are not imported by the tests above; compile them."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    for path in glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(here, "measure", "*.py")):
        with open(path, encoding="utf-8") as f:
            compile(f.read(), path, "exec")
