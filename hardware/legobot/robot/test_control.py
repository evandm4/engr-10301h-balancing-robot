"""Tests for the balancing loop's hardware-free parts (control.py) and the whole loop against the simulator
(balance.py --sim through sim_hw.py). No Pi needed.

    python -m pytest hardware/legobot/robot/test_control.py -q          (from the repo root)
"""
import json
import math
import os
import sys
from collections import namedtuple

import numpy as np
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
    assert abs(ff.duties(0.5, 20.0, 20.0)[0]) == pytest.approx(1.0)       # clipped at the real supply
    # 22 rad/s is under the no-load speed at 8.2 V but over it at 7.6 V: full duty here, coast at 7.6 V
    assert ff.duties(0.5, 22.0, 22.0)[0] == pytest.approx(1.0)
    assert control.Feedforward(cfg).duties(0.5, 22.0, 22.0)[0] == 0.0
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


def test_runaway_trip_catches_free_spinning_wheels(cfg):
    """Lifted while armed and held upright: no tilt, both wheels near the no-load speed. Review G11."""
    dt = cfg.timing.control_dt_s
    ok = dict(motor_ages=(0.005, 0.005), vin_reading=(0.0, 7.6), fault=None, reader_alive=True, running=True)
    n = int(round(lc.RUNAWAY_S / dt))
    s = control.Safety(dt)
    out = [s.check(k * dt, 0.0, (0.6, 0.6), (19.0, 18.5), **ok) for k in range(n)]
    assert all(o is None for o in out[:-1]) and "running away" in out[-1]
    s = control.Safety(dt)                        # bench A5: one count per tick is 3.5 rad/s, so at ~19.4 rad/s some
    rng = np.random.default_rng(0)                # ticks read under 17; they mustn't hold the trip off (2026-11-23)
    counts = 3.49 * rng.choice([4, 5, 6, 7], size=(n, 2), p=[0.044, 0.30, 0.50, 0.156])
    assert (counts < lc.RUNAWAY_SPEED).mean() > 0.03 and counts.mean() > 19.0
    out = [s.check(k * dt, 0.0, (-0.6, -0.6), tuple(-counts[k]), **ok) for k in range(n)]
    assert all(o is None for o in out[:-1]) and "running away" in out[-1]
    s = control.Safety(dt)                        # bursts near no-load while cruising at 11.6 rad/s don't trip
    burst = lambda k: (19.0, 19.0) if (k * dt) % 1.0 < 0.5 else (11.6, 11.6)
    assert all(s.check(k * dt, 0.0, (0.6, 0.6), burst(k), **{**ok, "vin_reading": (k * dt, 7.6)}) is None for k in range(4 * n))
    s = control.Safety(dt)                        # turning on the spot (wheels opposite) is not a runaway
    assert all(s.check(k * dt, 0.0, (0.6, -0.6), (19.0, -19.0), **ok) is None for k in range(n + 50))


def test_runaway_threshold_follows_the_battery_voltage(cfg):
    """H6: 0.85 of the no-load speed at the measured vin (0.85 * 22.2 rad/s at 8.4 V but 0.85 * 17.8 at 6.8 V)."""
    assert control.runaway_speed_for_vin(6.8) == pytest.approx(15.13, abs=0.01)
    assert control.runaway_speed_for_vin(7.6) == pytest.approx(17.01, abs=0.01)
    assert control.runaway_speed_for_vin(8.4) == pytest.approx(18.89, abs=0.01)
    assert control.runaway_speed_for_vin(None) == lc.RUNAWAY_SPEED
    assert control.runaway_speed_for_vin(0.0) >= 15.0       # a glitched reading never drives the threshold to zero
    dt = cfg.timing.control_dt_s
    n = int(round(lc.RUNAWAY_S / dt))
    ok = dict(motor_ages=(0.005, 0.005), fault=None, reader_alive=True, running=True)
    for vin, speed, trips in [(6.8, 16.0, True), (7.6, 16.0, False), (8.4, 18.0, False), (8.4, 19.5, True)]:
        s = control.Safety(dt)
        out = [s.check(k * dt, 0.0, (0.6, 0.6), (speed, speed), vin_reading=(0.0, vin), **ok) for k in range(n + 5)]
        assert any(o and "running away" in o for o in out) == trips, (vin, speed)
    s = control.Safety(dt)      # a transient 0 V reading at a normal speed does not false-trip the runaway check
    out = [s.check(k * dt, 0.0, (0.3, 0.3), (5.0, 5.0), vin_reading=(0.0, 0.0), **ok) for k in range(n + 5)]
    assert not any(o and "running away" in o for o in out)
    # L9: startup vin seeds Safety.vin before the first in-loop vin reply arrives (vin_reading=None)
    s84 = control.Safety(dt, vin=8.4)
    assert all(s84.check(k * dt, 0.0, (0.6, 0.6), (18.0, 18.0), vin_reading=None, **ok) is None for k in range(n + 5))
    s68 = control.Safety(dt, vin=6.8)
    assert any(o and "running away" in o
               for o in [s68.check(k * dt, 0.0, (0.6, 0.6), (16.0, 16.0), vin_reading=None, **ok) for k in range(n + 5)])


def test_watchdog_stop_joins_its_thread():
    """H6: stop() waits for the thread to end, so the run does not close the HAT under a coasting watchdog."""
    hat = type("H", (), {"emergency_coast": lambda self: None})()
    wd = balance.Watchdog(hat)
    wd.start()
    wd.kick()
    wd.stop()
    assert not wd.is_alive()
    wd.stop()                                     # idempotent once the thread has ended


def test_calibrate_gyro_bridges_one_i2c_error_not_two():
    """H6: one transient OSError during the gyro calibration is skipped (as run() bridges it); two in a row raise."""
    class Clock:
        t = 0.0
        def now(self): return self.t
        def wait_until(self, d): self.t = max(self.t, d)

    class Imu:
        def __init__(self, fail):
            self.fail, self.n = set(fail), 0
        def read(self):
            self.n += 1
            if self.n in self.fail:
                raise OSError("NACK")
            return namedtuple("R", "pitch_rate acc_fwd acc_up")(0.001, 0.0, 9.81)

    bias, sd, g = balance.calibrate_gyro(Imu({5}), Clock(), 0.005, say=lambda m: None)
    assert bias == pytest.approx(0.001)
    with pytest.raises(OSError):
        balance.calibrate_gyro(Imu({5, 6}), Clock(), 0.005, say=lambda m: None)


def test_sim_world_applies_slow_pwm_pulses(cfg):
    """H7: under slow PWM the simulated motor sees pulses in {-1, 0, +1} (pulse-density), as the real HAT gives."""
    from robot.sim_hw import SimWorld
    if getattr(cfg.build_actuator(), "pwm_tick", 0.0) <= 0:
        pytest.skip("this robot's actuator is not slow PWM")
    w = SimWorld(cfg, pitch=0.0)
    w.release()
    seen = set()
    for k in range(1, 400):
        w.drive = 0.3
        w.advance(k * 0.0025)
        seen.add(round(w.applied, 6))
    assert seen <= {-1.0, 0.0, 1.0} and {0.0, 1.0} <= seen
    from dataclasses import replace
    with pytest.raises(ValueError, match="HAT_LINE_PERIOD"):
        SimWorld(replace(cfg, actuator=replace(cfg.actuator, pwm_tick_s=0.020)))


def test_a_crash_in_the_loop_still_writes_the_logs(tmp_path, monkeypatch):
    """Review G2: an exception escaping the loop used to skip write_logs, losing the run's only record."""
    act = control.Brain.act
    calls = []

    def flaky(self, *args):
        calls.append(1)
        if len(calls) > 100:
            raise RuntimeError("boom")
        return act(self, *args)

    monkeypatch.setattr(control.Brain, "act", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        balance.main(["--sim", "--seconds", "3", "--outdir", str(tmp_path)])
    runs = list(tmp_path.glob("*_run.csv"))
    assert runs and sum(1 for _ in open(runs[0])) > 100
    assert any("crashed: RuntimeError: boom" in f.read_text() for f in tmp_path.glob("*"))


def test_stall_trip_sees_a_held_wheel_through_encoder_counts(cfg):
    """Bench 2026-11-23: a hand-held wheel at duty +0.3 read single counts (-1.75 rad/s) most ticks and 0 on
    others. That is a stall; a free wheel at 15 rad/s, or one spinning up from rest within 0.3 s, is not."""
    dt = cfg.timing.control_dt_s
    ok = dict(motor_ages=(0.005, 0.005), vin_reading=(0.0, 7.6), fault=None, reader_alive=True, running=True)
    n = int(round(2.0 / dt))
    held = [-1.745 if k % 3 else 0.0 for k in range(n)]
    s = control.Safety(dt, stall_duty=0.24)
    out = [s.check(k * dt, 0.0, (0.3, 0.3), (held[k], 15.0), **ok) for k in range(n)]
    first = next(k for k, o in enumerate(out) if o)
    assert "left wheel stalled" in out[first] and abs(first * dt - lc.STALL_S) < 0.02
    s = control.Safety(dt, stall_duty=0.24)
    spin_up = [min(15.0, 50.0 * k * dt) for k in range(n)]
    assert all(s.check(k * dt, 0.0, (0.3, 0.3), (spin_up[k], 15.0), **ok) is None for k in range(n))
    s = control.Safety(dt, stall_duty=0.24)              # duty dropping below the threshold restarts the window
    out = [s.check(k * dt, 0.0, (0.3 if (k // 150) % 2 == 0 else 0.1, 0.0), (0.0, 0.0), **ok) for k in range(n)]
    assert all(o is None for o in out)


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
    c, d = control.build_controller(cfg, str(p), goal="scenario")
    assert callable(c.goal)
    q = tmp_path / "lqr.json"                                   # the LQR was dropped for the LEGO robot
    q.write_text(json.dumps({"K": [-4.0, -4.7, -0.5, 0.0, 0.1, 0.1]}))
    with pytest.raises(ValueError):
        control.build_controller(cfg, str(q))
    for spec in ("lqr", "nonexistent.json"):                    # no "lqr" spec any more: read as a path
        with pytest.raises(OSError):
            control.build_controller(cfg, spec)
    w = tmp_path / "weights.json"
    w.write_text(json.dumps({"gains": {"s_velocity": 1.0, "s_pitch": 1.0}}))
    with pytest.raises(ValueError):
        control.build_controller(cfg, str(w))


def _reference_falls(spec):
    """Whether the reference simulator's own run of this controller falls on the current scenario (first
    start tilt, sensor seed 0). Then the gains or the scenario are the cause, not the robot code."""
    import numpy as np
    from balance_sim import simulate
    cfg = control.load_robot_config()
    controller, _ = control.build_controller(cfg, spec, goal="scenario")
    r = simulate(cfg.build_dynamics(), controller, duration=20.0, dt=cfg.timing.dt_s,
                 initial_state=np.array([0.0, 0.0, cfg.scenario.start_pitches_rad[0], 0.0]),
                 control_dt=cfg.timing.control_dt_s, sensor=cfg.build_sensor(0), latency=cfg.timing.latency_s,
                 disturbance=cfg.build_disturbance())
    return r.fell


@pytest.mark.parametrize("controller", ["hand", "model"])
def test_closed_loop_balances_with_simulator(tmp_path, controller):
    """The robot code (estimator, controller, slow-PWM feedforward, line-locked loop) balances the simulated
    robot standing still from a 0.02 rad tilt, with noise (the sim adds its 0.015 rad offset). Independent of the scenario's tuning."""
    from balance_sim import load_trajectory
    reason = balance.main(["--sim", "--sim-noise", "--sim-pitch", "0.02", "--controller", controller,
                           "--seconds", "10", "--outdir", str(tmp_path), "--label", controller])
    assert reason == "done"
    (run_csv,) = tmp_path.glob("*_run.csv")
    result, meta = load_trajectory(run_csv)
    assert meta["source"] == "sim_hw" and not result.fell
    assert abs(result.states[:, 2]).max() < 0.15
    assert abs(result.states[result.t > 5.0, 1].mean()) < 0.15     # no runaway (a P-only velocity loop creeps)


@pytest.mark.parametrize("controller", ["hand", "model"])
def test_closed_loop_tracks_the_scenario(tmp_path, controller):
    """Velocity signs and units: during the scenario's forward cruise the robot moves forward at a good share of
    the goal. Skipped where the reference simulator's own run falls (the gains or scenario, not this code)."""
    from balance_sim import load_trajectory
    if _reference_falls(controller):
        pytest.skip(f"{controller} falls in the reference simulator too on this scenario (gains, not robot code)")
    reason = balance.main(["--sim", "--sim-noise", "--goal", "scenario", "--controller", controller,
                           "--seconds", "20", "--outdir", str(tmp_path), "--label", controller])
    assert reason == "done"
    (run_csv,) = tmp_path.glob("*_run.csv")
    result, meta = load_trajectory(run_csv)
    cruise = (result.t > 5.0) & (result.t < 9.0)
    goal = result.velocity_goal[cruise].mean()
    assert goal > 0.1
    assert 0.5 * goal < result.states[cruise, 1].mean() < 1.3 * goal


@pytest.mark.parametrize("inject, expect", [("loop", "loop stalled"), ("stale", "stale"), ("vin", "battery"),
                                            ("fault", "fault"), ("imu", "IMU read failed 2 times")])
def test_injected_trips(tmp_path, inject, expect):
    reason = balance.main(["--sim", "--seconds", "3", "--inject", inject, "--outdir", str(tmp_path)])
    assert expect in reason


def test_too_weak_to_balance_falls_and_trips(tmp_path):
    """With the duty capped far below what balancing needs, the robot falls and the tilt trip ends the run."""
    reason = balance.main(["--sim", "--duty-cap", "0.02", "--sim-pitch", "0.045", "--seconds", "10",
                           "--outdir", str(tmp_path)])
    assert reason.startswith("tilt")


def _arrivals(seed, n=900, period=0.0099912, delay=0.0018):
    """HAT line arrivals as the Pi sees them: jitter (exp, mean 0.65 ms), 2% dropped lines, 1% late by
    5-20 ms (reader thread), and 5% stamped with the next line's arrival (two lines in one read)."""
    import random
    rng = random.Random(seed)
    arr = []
    for k in range(n):
        if rng.random() < 0.02:
            continue
        j = rng.expovariate(1 / 0.00065) + (rng.uniform(0.005, 0.02) if rng.random() < 0.01 else 0.0)
        arr.append(100.0 + k * period + delay + j)
    for i in range(1, len(arr)):
        if rng.random() < 0.05:
            arr[i - 1] = arr[i]
    return sorted(arr), period, delay


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_line_clock_finds_the_hat_grid(seed):
    arr, period, delay = _arrivals(seed)
    clock = control.LineClock()
    for n in range(60, len(arr) + 1, 20):       # updated every 20 lines, as the loop does (every 100 ms)
        clock.update(arr[:n])
    assert abs(clock.period / period - 1) < 60e-6
    envelope_delay = (clock.t_ref - 100.0) % period
    assert delay < envelope_delay < delay + 0.0003       # the lower envelope, not the mean (0.65 ms later)
    assert clock.locked(arr[-1] + 0.01) and not clock.locked(arr[-1] + 1.0)


def test_pacer_ticks_twice_per_line_at_the_offset():
    arr, period, _ = _arrivals(4)
    clock = control.LineClock()
    clock.update(arr)
    pacer = control.Pacer(0.005, clock, offset=0.003)
    now = prev = arr[-1]
    ticks = []
    for _ in range(8):
        prev, locked = pacer.next(prev, now)
        ticks.append(prev)
        assert locked
    assert all(abs((b - a) - clock.period / 2) < 1e-9 for a, b in zip(ticks, ticks[1:]))
    assert all(abs(clock.phase(t) % (clock.period / 2) - 0.003) < 1e-6 for t in ticks)
    assert control.Pacer(0.005).next(1.0, 1.0) == (1.005, False)              # no line clock: free-running
    assert control.Pacer(0.005, clock).next(prev, prev + 1.0) == (prev + 0.005, False)   # lock stale


def test_closed_loop_ticks_are_locked(tmp_path):
    import csv
    balance.main(["--sim", "--sim-noise", "--seconds", "5", "--line-phase", "0.004", "--outdir", str(tmp_path)])
    (raw,) = tmp_path.glob("*_raw.csv")
    rows = [r for r in csv.DictReader(open(raw)) if r["phase"] == "1"]
    assert all(r["locked"] == "1" for r in rows)
    phases = [float(r["cmd_phase"]) % 0.0049956 for r in rows]      # the sim's control takes no time
    target = (0.004 - lc.COMPUTE_LEAD_S) % 0.0049956
    assert max(abs(p - target) for p in phases) < 0.0003


def test_every_robot_module_compiles():
    """The Pi-only modules (pyserial, smbus2, fcntl) are not imported by the tests above; compile them."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    for path in glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(here, "measure", "*.py")):
        with open(path, encoding="utf-8") as f:
            compile(f.read(), path, "exec")


# ------------------------------------------------- the Pi layout: only what deploy.sh ships (2026-11-25) ---
# The laptop's tests run with optimization/ (tuning, batch_tuning) and scipy importable (pytest.ini), so they
# can't see a robot-side import of either. On the Pi there is neither: deploy.sh ships config.py,
# legobot_config.py, simulation/balance_sim, robot/ and the policy bundles, and ~/legobot-venv has numpy but no
# scipy (and no room for it). The policy config check once failed there (scoring_differences -> snapshot() ->
# build_search_space -> tuning, and build_lqr_gain -> scipy). This rebuilds that layout and runs the robot code in
# a subprocess that can import nothing else.
PI_BLOCKED = ("scipy", "tuning", "batch_tuning", "batch_sim", "torch", "cma", "matplotlib", "export_policy")

_PI_CHILD = r'''
import importlib, importlib.abc, json, os, pkgutil, sys, tempfile, types
repo, blocked = os.path.normcase(os.path.abspath(sys.argv[1])), set(sys.argv[2].split(","))
attempts = []

class PiBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in blocked:
            attempts.append(name)
            raise ModuleNotFoundError(f"{name} is not on the Pi", name=name)
        return None

sys.meta_path.insert(0, PiBlocker())
# Nothing from the repository: '' (the working directory, ~/legobot's stand-in) and the standard library and venv.
sys.path[:] = [p for p in sys.path if not os.path.normcase(os.path.abspath(p or ".")).startswith(repo)]
for probe in ("scipy", "tuning"):
    try:
        importlib.import_module(probe)
        raise SystemExit(f"blocker failed: {probe} imported")
    except ModuleNotFoundError:
        pass
probes = len(attempts)
# The Pi has pyserial, smbus2 and fcntl; this machine may not. Stand-ins, so the hardware modules import.
for hw in ("serial", "smbus2", "fcntl"):
    try:
        importlib.import_module(hw)
    except ImportError:
        stub = types.ModuleType(hw)
        stub.__getattr__ = lambda attr: object
        sys.modules[hw] = stub

import robot
from robot import control, balance
modules = sorted(m.name for m in pkgutil.walk_packages(robot.__path__, "robot."))
for m in modules:
    importlib.import_module(m)

cfg = control.load_robot_config()
bundles = {}
for name in sorted(os.listdir("policies")):
    b = control.load_policy_bundle("policy:" + name, cfg, allow_config_mismatch=True)
    strict = None
    if not b["config_differences"]:
        strict = control.load_policy_bundle("policy:" + name, cfg)["config_differences"]
    controller, desc = control.build_controller(cfg, "policy:" + name, allow_config_mismatch=True)
    st = control.self_test_policy(controller, b["golden"], cfg=cfg)
    bundles[name] = {"differences": b["config_differences"], "strict": strict, "kind": desc["policy_kind"],
                     "self_test_max_err": st["max_torque_err"]}
with tempfile.TemporaryDirectory() as out:
    reason = balance.main(["--sim", "--sim-noise", "--seconds", "1", "--controller", "policy:l_pid", "--outdir", out])
print(json.dumps({"modules": modules, "bundles": bundles, "sim_reason": reason, "attempts": attempts[probes:],
                  "loaded_blocked": sorted(m for m in sys.modules if m.split(".")[0] in blocked),
                  "config_file": os.path.abspath(sys.modules["config"].__file__),
                  "balance_sim_file": os.path.abspath(sys.modules["balance_sim"].__file__)}))
'''


def deploy_files(root):
    """deploy.sh's FILES list (the repo-root files and folders it ships besides robot/ and the policies)."""
    import re
    from pathlib import Path
    text = (Path(root) / "hardware" / "legobot" / "deploy.sh").read_text()
    match = re.search(r"^FILES=\(([^)]*)\)", text, re.M)
    assert match, "deploy.sh has no FILES=(...) line"
    return match.group(1).split()


def build_pi_layout(root, dst):
    """~/legobot as deploy.sh leaves it: FILES at the top, hardware/legobot/robot as robot/, the bundles as
    policies/, without __pycache__ and test_*.py (deploy.sh's excludes)."""
    import shutil
    from pathlib import Path
    root, dst = Path(root), Path(dst)
    skip = shutil.ignore_patterns("__pycache__", "test_*.py", "*.pyc")
    for rel in deploy_files(root):
        src, out = root / rel, dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, out, ignore=skip)
        else:
            shutil.copy2(src, out)
    shutil.copytree(root / "hardware" / "legobot" / "robot", dst / "robot", ignore=skip)
    if (root / "results" / "legobot" / "policies").is_dir():
        shutil.copytree(root / "results" / "legobot" / "policies", dst / "policies", ignore=skip)
    return dst


def run_on_pi_layout(root, layout):
    """Run _PI_CHILD in `layout` (a build_pi_layout folder): (returncode, parsed JSON or None, stdout+stderr)."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env["ROBOT"] = "legobot"
    proc = subprocess.run([sys.executable, "-c", _PI_CHILD, str(root), ",".join(PI_BLOCKED)], cwd=str(layout),
                          env=env, capture_output=True, text=True, timeout=600)
    lines = proc.stdout.strip().splitlines()
    try:
        data = json.loads(lines[-1]) if lines else None
    except ValueError:
        data = None
    return proc.returncode, data, proc.stdout + proc.stderr


def test_robot_code_runs_on_the_deployed_layout_without_scipy_or_tuning(tmp_path, cfg):
    """P2b on the Pi: every robot module imports, and each committed bundle loads, passes the config check
    (the same differences the laptop finds, with optimization/ and scipy), builds and passes its golden
    self-test, with only deploy.sh's files importable and scipy, tuning, batch_tuning (...) blocked. A
    policy then runs 1 s closed loop in --sim. Nothing may even try to import a blocked module."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    assert "config.py" in deploy_files(root) and "simulation/balance_sim" in deploy_files(root)
    layout = build_pi_layout(root, tmp_path / "legobot")
    code, data, output = run_on_pi_layout(root, layout)
    assert code == 0 and data is not None, output[-4000:]
    assert data["attempts"] == [] and data["loaded_blocked"] == [], data
    assert Path(data["config_file"]).parent == layout.resolve()                     # the deployed copy, not the repo's
    assert Path(data["balance_sim_file"]).parent == (layout / "simulation" / "balance_sim").resolve()
    assert {"robot.balance", "robot.control", "robot.calibrate", "robot.replay_policy", "robot.score_run",
            "robot.sim_hw", "robot.measure.fit"} <= set(data["modules"])
    assert data["sim_reason"] == "done"
    committed = sorted(p.name for p in (root / "results" / "legobot" / "policies").iterdir() if p.is_dir())
    assert sorted(data["bundles"]) == committed and {"l_pid", "a_warm_zero", "b1_zero", "quick_l"} <= set(committed)
    for name, b in data["bundles"].items():
        snapshot = json.loads((root / "results" / "legobot" / "policies" / name / "snapshot.json").read_text())
        laptop = cfg.scoring_differences(snapshot["config"])     # here, with tuning and scipy importable
        assert b["differences"] == laptop, name
        assert b["strict"] == ([] if not laptop else None), name   # no difference -> loads without the override
        assert b["self_test_max_err"] <= lc.GOLDEN_TOL, name
    # Today every committed bundle matches legobot_config; when E2 changes the config, re-export them.
    assert all(b["differences"] == [] for b in data["bundles"].values()), data["bundles"]


def test_scoring_differences_needs_no_derived_section(cfg):
    """scoring_differences compares snapshot(derived=False): "derived" is NOT_SCORING, so leaving it out
    changes nothing it reports, for either robot, and saved snapshots keep their derived section."""
    import config
    assert "derived" in cfg.NOT_SCORING
    for c in (cfg, config.load_config("pololu")):
        full, lean = c.snapshot(), c.snapshot(derived=False)
        assert "derived" in full and "derived" not in lean
        assert {k: v for k, v in full.items() if k != "derived"} == lean
        assert c.scoring_differences(full) == [] and c.scoring_differences(lean) == []
    changed = cfg.snapshot()
    changed["timing"]["control_dt_s"] *= 2
    changed["derived"]["cost_matched_lqr_gain"] = [0.0]           # NOT_SCORING: never reported
    assert cfg.scoring_differences(changed) == ["timing.control_dt_s"]


def test_port_lag_reads_the_offset_between_the_two_ports_line_grids():
    """B1 (2026-11-23): each port streams on its own grid, and the lag between them changes per run."""
    rng = np.random.default_rng(1)
    period = 0.0099984
    left = 3.0 + period * np.arange(300) + rng.exponential(0.0006, 300)
    right = 3.0 + 0.0042 + period * np.arange(300) + rng.exponential(0.0006, 300)
    lag = control.port_lag(list(left), list(right), 0.010)
    assert abs(lag - 0.0042) < 0.0005
    assert control.port_lag(list(left[:10]), list(right), 0.010) is None


def test_losing_the_line_lock_falls_back_to_the_pi_clock(tmp_path):
    """Bench B1: the lock can't fix the wait to a pulse, so losing it is logged, not a trip. The run goes
    on, ticking on the Pi's clock, and the summary says how often it was lost."""
    reason = balance.main(["--sim", "--sim-noise", "--seconds", "3", "--inject", "unlock", "--outdir", str(tmp_path)])
    assert reason == "done"
    summary = next(tmp_path.glob("*_summary.txt")).read_text()
    assert "line lock lost while armed: 1 times" in summary
    raw = next(tmp_path.glob("*_raw.csv")).read_text().splitlines()
    locked = [int(row.split(",")[raw[0].split(",").index("locked")]) for row in raw[1:]]
    assert 1 in locked and locked[-1] == 0          # locked before the injection, Pi clock after


# ----------------------------------------------------------------- policy-on-robot tests (P1-P6, P8) ---
def _sha256_file(path):
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_bundle(src, dst):
    import shutil
    from pathlib import Path
    dst = Path(dst)
    shutil.copytree(src, dst)
    return dst


def _refresh_manifest_hashes(bundle_dir):
    """Recompute manifest.json sha256 and files_sha256 after a deliberate test mutation."""
    from pathlib import Path
    bundle_dir = Path(bundle_dir)
    mpath = bundle_dir / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["sha256"] = _sha256_file(bundle_dir / "final.npz")
    if isinstance(manifest.get("files_sha256"), dict):
        for fname in list(manifest["files_sha256"]):
            fpath = bundle_dir / fname
            if fpath.is_file():
                manifest["files_sha256"][fname] = _sha256_file(fpath)
    mpath.write_text(json.dumps(manifest))
    return manifest


def _rewrite_npz(src, dst, **overrides):
    """Copy every array in `src` .npz to `dst`, applying key overrides."""
    with np.load(src) as data:
        arrays = {k: data[k] for k in data.files}
    arrays.update(overrides)
    np.savez(dst, **arrays)
    return dst


@pytest.fixture(scope="module")
def synthetic_bundles(tmp_path_factory, cfg):
    """Export the three synthetic bundles (l_pid, a_warm_zero, b1_zero) once per module via P1's exporter."""
    control.import_config()
    import export_policy
    root = tmp_path_factory.mktemp("policies")
    out = {"root": root}
    for name in ("l_pid", "a_warm_zero", "b1_zero"):
        out[name] = export_policy.export_bundle(synthetic=name, name=name, out_root=root, cfg=cfg,
                                                envelope_seeds=1, seed=0)
    return out


# ---- 1. L-PID through robot code ~= PID through robot code -----------------------------------------
def test_l_pid_through_robot_code_is_close_to_hand_pid(tmp_path, cfg, synthetic_bundles):
    """P6: PID-equivalent linear policy (leaky, frozen integrators) vs hand-tuned PID through sim_hw."""
    from balance_sim import load_trajectory

    d_pid, d_lin = tmp_path / "pid", tmp_path / "l_pid"
    common = ["--sim", "--sim-noise", "--sim-pitch", "0.02", "--seconds", "10"]
    r_pid_reason = balance.main(common + ["--controller", "hand", "--outdir", str(d_pid), "--label", "pid"])
    r_lin_reason = balance.main(common + ["--controller", str(synthetic_bundles["l_pid"]),
                                          "--outdir", str(d_lin), "--label", "l_pid"])
    assert r_pid_reason == "done" and r_lin_reason == "done"

    r_pid, _ = load_trajectory(next(d_pid.glob("*_run.csv")))
    r_lin, _ = load_trajectory(next(d_lin.glob("*_run.csv")))
    assert not r_pid.fell and not r_lin.fell
    assert abs(r_lin.states[:, 2]).max() < 0.15
    assert abs(r_lin.states[r_lin.t > 5.0, 1].mean()) < 0.15

    # Close, not identical (leaky/frozen integrators vs PID anti-windup can drift slightly in phase under noise)
    n = min(len(r_pid.t), len(r_lin.t))
    pitch_rms_diff = float(np.sqrt(np.mean((r_lin.states[:n, 2] - r_pid.states[:n, 2]) ** 2)))
    vel_rms_diff = float(np.sqrt(np.mean((r_lin.states[:n, 1] - r_pid.states[:n, 1]) ** 2)))
    vel_rms_pid = float(np.sqrt(np.mean(r_pid.states[:n, 1] ** 2)))
    vel_rms_lin = float(np.sqrt(np.mean(r_lin.states[:n, 1] ** 2)))
    assert pitch_rms_diff < 0.05
    assert vel_rms_diff < 0.25
    assert abs(vel_rms_lin - vel_rms_pid) < 0.10
    cost_pid, cost_lin = cfg.score(r_pid), cfg.score(r_lin)
    assert cost_pid > 0.0 and abs(cost_lin / cost_pid - 1.0) < 0.35


# ---- 2. B1 with zero output layer == reference PID bit-for-bit (catches Problem 1) -----------------
@pytest.mark.parametrize("goal, seconds", [("still", "5"), ("scenario", "15")])
def test_b1_zero_equals_hand_pid_bit_for_bit(tmp_path, cfg, synthetic_bundles, goal, seconds):
    """P6 + Problem 1: residual B1 with zero output layer equals the reference PID bit-for-bit on both
    --goal still and --goal scenario (goal must reach PolicyController.base, not just PolicyController)."""
    from balance_sim import load_trajectory

    c, _ = control.build_controller(cfg, str(synthetic_bundles["b1_zero"]), goal=goal)
    assert c.base is not None
    if goal == "still":
        assert c.goal == 0.0 and c.base.goal == 0.0
    else:
        assert callable(c.goal) and callable(c.base.goal)

    d_pid, d_b1 = tmp_path / f"pid_{goal}", tmp_path / f"b1_{goal}"
    common = ["--sim", "--sim-noise", "--goal", goal, "--seconds", seconds]
    reason_pid = balance.main(common + ["--controller", "hand", "--outdir", str(d_pid), "--label", "pid"])
    reason_b1 = balance.main(common + ["--controller", str(synthetic_bundles["b1_zero"]),
                                       "--outdir", str(d_b1), "--label", "b1_zero"])
    assert reason_pid == "done" and reason_b1 == "done"

    r_pid, _ = load_trajectory(next(d_pid.glob("*_run.csv")))
    r_b1, _ = load_trajectory(next(d_b1.glob("*_run.csv")))
    np.testing.assert_array_equal(r_b1.t, r_pid.t)
    np.testing.assert_array_equal(r_b1.states, r_pid.states)
    np.testing.assert_array_equal(r_b1.torques, r_pid.torques)
    np.testing.assert_array_equal(r_b1.velocity_goal, r_pid.velocity_goal)
    np.testing.assert_array_equal(r_b1.pitch_reference, r_pid.pitch_reference)


# ---- 3. A-warm with zero output layer == L bit-for-bit ---------------------------------------------
@pytest.mark.parametrize("goal, seconds", [("still", "5"), ("scenario", "15")])
def test_a_warm_zero_equals_l_pid_bit_for_bit(tmp_path, synthetic_bundles, goal, seconds):
    """P6: A-warm with zero output layer produces bit-identical trajectories and torques to L."""
    from balance_sim import load_trajectory

    d_lin, d_warm = tmp_path / f"l_{goal}", tmp_path / f"warm_{goal}"
    common = ["--sim", "--sim-noise", "--goal", goal, "--seconds", seconds]
    reason_lin = balance.main(common + ["--controller", str(synthetic_bundles["l_pid"]),
                                        "--outdir", str(d_lin), "--label", "l_pid"])
    reason_warm = balance.main(common + ["--controller", str(synthetic_bundles["a_warm_zero"]),
                                         "--outdir", str(d_warm), "--label", "a_warm_zero"])
    assert reason_lin == "done" and reason_warm == "done"

    r_lin, _ = load_trajectory(next(d_lin.glob("*_run.csv")))
    r_warm, _ = load_trajectory(next(d_warm.glob("*_run.csv")))
    np.testing.assert_array_equal(r_warm.t, r_lin.t)
    np.testing.assert_array_equal(r_warm.states, r_lin.states)
    np.testing.assert_array_equal(r_warm.torques, r_lin.torques)
    np.testing.assert_array_equal(r_warm.velocity_goal, r_lin.velocity_goal)


# ---- 4. Closed-loop balance & scenario tracking with policy:<name> lookup --------------------------
def test_closed_loop_balances_and_tracks_policy_by_name(tmp_path, cfg, synthetic_bundles, monkeypatch):
    """P2c + P6: policy:<name> resolves via lc.POLICIES_DIR (and repo results/legobot/policies/),
    balances standing still, and tracks the forward cruise in --goal scenario."""
    from pathlib import Path
    from balance_sim import load_trajectory

    # 1) Lookup via lc.POLICIES_DIR
    monkeypatch.setattr(lc, "POLICIES_DIR", str(synthetic_bundles["root"]))
    d_still, d_scen = tmp_path / "still", tmp_path / "scenario"
    reason_still = balance.main(["--sim", "--sim-noise", "--sim-pitch", "0.02", "--controller", "policy:l_pid",
                                 "--seconds", "10", "--outdir", str(d_still), "--label", "l_pid_still"])
    assert reason_still == "done"
    r_still, meta_still = load_trajectory(next(d_still.glob("*_run.csv")))
    assert meta_still["source"] == "sim_hw" and not r_still.fell
    assert abs(r_still.states[:, 2]).max() < 0.15
    assert abs(r_still.states[r_still.t > 5.0, 1].mean()) < 0.15

    reason_scen = balance.main(["--sim", "--sim-noise", "--goal", "scenario", "--controller", "policy:l_pid",
                                "--seconds", "20", "--outdir", str(d_scen), "--label", "l_pid_scen"])
    assert reason_scen == "done"
    r_scen, _ = load_trajectory(next(d_scen.glob("*_run.csv")))
    assert not r_scen.fell
    cruise = (r_scen.t > 5.0) & (r_scen.t < 9.0)
    goal_v = r_scen.velocity_goal[cruise].mean()
    assert goal_v > 0.1
    assert 0.5 * goal_v < r_scen.states[cruise, 1].mean() < 1.3 * goal_v

    # 2) Fallback lookup via <repo_root>/results/legobot/policies/<name> when not in lc.POLICIES_DIR
    repo_l_pid = Path(__file__).resolve().parents[3] / "results" / "legobot" / "policies" / "l_pid"
    if repo_l_pid.is_dir():
        monkeypatch.setattr(lc, "POLICIES_DIR", str(tmp_path / "empty_policies"))
        b = control.load_policy_bundle("policy:l_pid", cfg, allow_config_mismatch=True)
        assert Path(b["bundle_dir"]).resolve() == repo_l_pid.resolve()


def test_quick_l_exported_bundle_loads_runs_and_replays(tmp_path, cfg, monkeypatch):
    """P1 + P2a-P2d + P4b + P2-3: the committed quick_l bundle (a real trained, exported policy) passes the strict
    config check and the golden self-test, runs closed loop in still and --goal scenario without falling, stays
    inside its own exported envelope, and replays <= 1e-12.

    Robot-path test only, no performance gates. quick_l is the generation-6 checkpoint of a 30-generation "quick"
    CMA-ES run (dev20 objective ~191). It saturates ~77% of ticks and cruises at only ~0.2-0.4x the scenario goal,
    in its own training simulator too. So "theta < 0.20" and "tracks within 0.5-1.3x goal" never held for it,
    even at the commit that added it (d5fc533). The absolute balance/tracking gates live in
    test_closed_loop_balances_and_tracks_policy_by_name (l_pid). Replace quick_l with the E2 full-profile L
    bundle once it's trained, and restore the tracking assertions then."""
    import json
    from pathlib import Path
    from balance_sim import load_trajectory
    from robot import replay_policy

    repo_quick_l = Path(__file__).resolve().parents[3] / "results" / "legobot" / "policies" / "quick_l"
    if not repo_quick_l.is_dir():
        pytest.skip(f"no exported quick_l bundle at {repo_quick_l}")
    monkeypatch.setattr(lc, "POLICIES_DIR", str(tmp_path / "empty_policies"))

    # Strict load (no --allow-config-mismatch): the bundle's snapshot must match the current legobot_config.
    bundle = control.load_policy_bundle("policy:quick_l", cfg)
    assert Path(bundle["bundle_dir"]).resolve() == repo_quick_l.resolve()
    assert not bundle["unchecked"] and bundle["config_differences"] == []
    controller, desc = control.build_controller(cfg, "policy:quick_l", goal="still")
    assert desc["kind"] == "policy" and desc["bundle_name"] == "quick_l"
    st = control.self_test_policy(controller, bundle["golden"], cfg=cfg)
    assert st["n_ticks"] >= 300 and st["max_torque_err"] <= lc.GOLDEN_TOL

    envelope = json.loads((repo_quick_l / "envelope.json").read_text())
    theta_max = envelope["observations"]["theta_meas"]["max_abs"]

    d_still, d_scen = tmp_path / "ql_still", tmp_path / "ql_scen"
    reason_still = balance.main(["--sim", "--sim-noise", "--sim-pitch", "0.02", "--controller", "policy:quick_l",
                                 "--seconds", "10", "--outdir", str(d_still), "--label", "quick_l_still"])
    assert reason_still == "done"
    r_still, _ = load_trajectory(next(d_still.glob("*_run.csv")))
    assert not r_still.fell and abs(r_still.states[:, 2]).max() <= theta_max

    reason_scen = balance.main(["--sim", "--sim-noise", "--goal", "scenario", "--controller", "policy:quick_l",
                                "--seconds", "20", "--outdir", str(d_scen), "--label", "quick_l_scen"])
    assert reason_scen == "done"
    (run_scen,) = d_scen.glob("*_run.csv")
    r_scen, _ = load_trajectory(run_scen)
    assert not r_scen.fell and abs(r_scen.states[:, 2]).max() <= theta_max
    cruise = (r_scen.t > 5.0) & (r_scen.t < 9.0)
    assert r_scen.velocity_goal[cruise].mean() > 0.1    # the scenario did command a forward cruise

    rep = replay_policy.replay_log(run_scen)
    assert rep["n_ticks"] > 100 and rep["max_torque_diff"] <= 1e-12


# ---- 5. Bundle validation, refusals & startup self-test (P2a-P2d, P3b) -----------------------------
def test_bundle_round_trip_and_self_test(tmp_path, cfg):
    """P1 + P2a + P2d + P3b: export_bundle -> load_policy_bundle -> build_controller -> self_test_policy."""
    control.import_config()
    import export_policy

    bdir = export_policy.export_bundle(synthetic="l_pid", name="roundtrip_l", out_root=tmp_path, cfg=cfg,
                                       envelope_seeds=1)
    bundle = control.load_policy_bundle(bdir, cfg)
    assert bundle["name"] == "roundtrip_l" and bundle["kind"] == "linear"
    assert not bundle["unchecked"] and bundle["config_differences"] == []
    assert bundle["sha256"] == _sha256_file(bdir / "final.npz")

    controller, desc = control.build_controller(cfg, str(bdir), goal="still")
    assert desc["kind"] == "policy" and desc["policy_kind"] == "linear"
    assert desc["bundle_name"] == "roundtrip_l" and desc["sha256"] == bundle["sha256"]
    assert desc["n_params"] == bundle["spec"].n_params and not desc["unchecked"]
    # desc must stay plain JSON-serializable metadata
    json.dumps(desc)

    st = control.self_test_policy(controller, bundle["golden"], cfg=cfg)
    assert st["n_ticks"] == len(bundle["golden"]["states"]) and st["n_ticks"] >= 300
    assert st["max_torque_err"] <= lc.GOLDEN_TOL
    assert st["compute_lead_s"] >= lc.COMPUTE_LEAD_S
    assert st["compute_lead_s"] == pytest.approx(lc.COMPUTE_LEAD_S + max(0.0, st["p99_s"] - st["pid_p99_s"]))


def test_bare_npz_requires_allow_unchecked(tmp_path, cfg, synthetic_bundles):
    """P2a: bare .npz refused without --allow-unchecked, accepted with --allow-unchecked."""
    import shutil
    bare = tmp_path / "bare_l_pid.npz"
    shutil.copyfile(synthetic_bundles["l_pid"] / "final.npz", bare)

    with pytest.raises(ValueError):
        control.load_policy_bundle(bare, cfg)
    with pytest.raises(SystemExit):
        balance.main(["--sim", "--controller", str(bare), "--seconds", "1", "--outdir", str(tmp_path / "refused")])

    bundle = control.load_policy_bundle(bare, cfg, allow_unchecked=True)
    assert bundle["unchecked"] is True and bundle["kind"] == "linear"
    reason = balance.main(["--sim", "--controller", str(bare), "--allow-unchecked",
                           "--seconds", "1", "--outdir", str(tmp_path / "accepted")])
    assert reason == "done"


def test_corrupt_wrong_obs_sha_and_unknown_kind_refused(tmp_path, cfg, synthetic_bundles):
    """P2a + H5 + M5: corrupt .npz, wrong observations list, sha256 mismatch (final.npz and snapshot.json),
    path traversal, NaN weights (C1), and unknown kind ('gru') refused with ValueError."""
    # Corrupt .npz -> wrapped as ValueError (M5)
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"definitely not a zip/npz archive")
    with pytest.raises(ValueError):
        control.load_policy_bundle(corrupt, cfg, allow_unchecked=True)

    # Path traversal in policy:<name> refused
    with pytest.raises(ValueError):
        control.load_policy_bundle("policy:../l_pid", cfg)

    # Wrong observations list
    src_npz = synthetic_bundles["l_pid"] / "final.npz"
    with np.load(src_npz) as f:
        bad_obs = np.array([f"bad_{k}" for k in range(len(f["observations"]))])
    wrong_obs_npz = _rewrite_npz(src_npz, tmp_path / "wrong_obs.npz", observations=bad_obs)
    with pytest.raises(ValueError, match="(?i)observation"):
        control.load_policy_bundle(wrong_obs_npz, cfg, allow_unchecked=True)

    # sha256 mismatch in manifest.json (final.npz)
    bad_sha = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "bad_sha")
    manifest = json.loads((bad_sha / "manifest.json").read_text())
    manifest["sha256"] = "0" * 64
    (bad_sha / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="(?i)sha256|hash|checksum"):
        control.load_policy_bundle(bad_sha, cfg)

    # H5: tampering with snapshot.json without updating files_sha256 is caught even with allow_config_mismatch
    tampered_snap = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "tampered_snap")
    snap_t = json.loads((tampered_snap / "snapshot.json").read_text())
    snap_t["config"]["timing"]["control_dt_s"] = 0.010
    (tampered_snap / "snapshot.json").write_text(json.dumps(snap_t))
    with pytest.raises(ValueError, match="(?i)sha256"):
        control.load_policy_bundle(tampered_snap, cfg, allow_config_mismatch=True)

    # P2-1: deleting files_sha256 from manifest.json must NOT bypass file integrity checks
    missing_fsha = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "missing_fsha")
    m_nof = json.loads((missing_fsha / "manifest.json").read_text())
    m_nof.pop("files_sha256", None)
    (missing_fsha / "manifest.json").write_text(json.dumps(m_nof))
    with pytest.raises(ValueError, match="(?i)files_sha256|sha256"):
        control.load_policy_bundle(missing_fsha, cfg, allow_config_mismatch=True)

    # C1: NaN weights in .npz refused by load_policy_bundle
    with np.load(src_npz) as f:
        nan_params = f["params"].copy()
    nan_params[0] = np.nan
    nan_npz = _rewrite_npz(src_npz, tmp_path / "nan_params.npz", params=nan_params)
    with pytest.raises(ValueError, match="(?i)finite|nan"):
        control.load_policy_bundle(nan_npz, cfg, allow_unchecked=True)

    # Unknown kind ("gru") in manifest.json
    gru_bundle = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "gru_bundle")
    m_gru = json.loads((gru_bundle / "manifest.json").read_text())
    m_gru["kind"] = "gru"
    (gru_bundle / "manifest.json").write_text(json.dumps(m_gru))
    with pytest.raises(ValueError, match="(?i)gru|kind"):
        control.load_policy_bundle(gru_bundle, cfg)

    # Unknown kind ("gru") in bare .npz
    gru_npz = _rewrite_npz(src_npz, tmp_path / "gru_kind.npz", kind=np.array("gru"))
    with pytest.raises(ValueError):
        control.load_policy_bundle(gru_npz, cfg, allow_unchecked=True)

    gru_key_npz = _rewrite_npz(src_npz, tmp_path / "gru_key.npz", gru=np.zeros(4))
    with pytest.raises(ValueError):
        control.load_policy_bundle(gru_key_npz, cfg, allow_unchecked=True)


def test_config_mismatch_max_torque_and_ff_gain_refusals(tmp_path, cfg, synthetic_bundles):
    """P2b + P2c: config mismatch refused unless --allow-config-mismatch (and logged);
    max_torque mismatch ALWAYS refused; --ff-gain refused for policies."""
    # 1) control_dt_s mismatch refused
    bad_dt = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "bad_dt")
    snap = json.loads((bad_dt / "snapshot.json").read_text())
    snap["config"]["timing"]["control_dt_s"] = 0.010
    (bad_dt / "snapshot.json").write_text(json.dumps(snap))
    _refresh_manifest_hashes(bad_dt)
    with pytest.raises(ValueError, match="control_dt_s"):
        control.load_policy_bundle(bad_dt, cfg)
    with pytest.raises(SystemExit):
        balance.main(["--sim", "--controller", str(bad_dt), "--seconds", "1", "--outdir", str(tmp_path / "o_dt")])

    # 2) voltage_feedforward_gain mismatch refused by default, accepted and logged with --allow-config-mismatch
    bad_ff = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "bad_ff")
    snap_ff = json.loads((bad_ff / "snapshot.json").read_text())
    snap_ff["config"]["actuator"]["voltage_feedforward_gain"] = 0.85
    (bad_ff / "snapshot.json").write_text(json.dumps(snap_ff))
    _refresh_manifest_hashes(bad_ff)
    with pytest.raises(ValueError, match="voltage_feedforward_gain"):
        control.load_policy_bundle(bad_ff, cfg)

    loaded_ff = control.load_policy_bundle(bad_ff, cfg, allow_config_mismatch=True)
    assert "actuator.voltage_feedforward_gain" in loaded_ff["config_differences"]
    assert loaded_ff["feedforward_gain"] == pytest.approx(0.85)

    out_ff = tmp_path / "o_ff"
    reason = balance.main(["--sim", "--controller", str(bad_ff), "--allow-config-mismatch",
                           "--seconds", "1", "--outdir", str(out_ff)])
    assert reason == "done"
    summary_text = next(out_ff.glob("*_summary.txt")).read_text()
    assert "voltage_feedforward_gain" in summary_text

    # 3) max_torque mismatch refused EVEN WITH allow_config_mismatch=True (bare .npz and bundle)
    bad_tau_npz = _rewrite_npz(synthetic_bundles["l_pid"] / "final.npz", tmp_path / "bad_tau.npz",
                               max_torque=cfg.robot.max_torque * 0.5)
    with pytest.raises(ValueError, match="(?i)torque"):
        control.load_policy_bundle(bad_tau_npz, cfg, allow_unchecked=True, allow_config_mismatch=True)

    bad_tau_bundle = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "bad_tau_bundle")
    _rewrite_npz(bad_tau_bundle / "final.npz", bad_tau_bundle / "final.npz", max_torque=cfg.robot.max_torque * 0.5)
    _refresh_manifest_hashes(bad_tau_bundle)
    with pytest.raises(ValueError, match="(?i)torque"):
        control.load_policy_bundle(bad_tau_bundle, cfg, allow_config_mismatch=True)

    # 4) --ff-gain refused for policies
    with pytest.raises(ValueError):
        control.build_controller(cfg, str(synthetic_bundles["l_pid"]), ff_gain_override=1.0)
    with pytest.raises(SystemExit):
        balance.main(["--sim", "--controller", str(synthetic_bundles["l_pid"]), "--ff-gain", "1.0",
                      "--seconds", "1", "--outdir", str(tmp_path / "o_ff_refused")])


def test_self_test_catches_perturbed_weights_and_restores_state(tmp_path, cfg, synthetic_bundles):
    """P2d + C1 + H3 + M1: self_test_policy fails on perturbed linear weights and NaN weights,
    restores controller.goal and controller.base.goal, resets integrators, and blocks arming in balance.main."""
    # 1) Goal & integrator restoration on B1 for both goal='still' and goal='scenario'
    b1_bundle = control.load_policy_bundle(synthetic_bundles["b1_zero"], cfg)
    for goal in ("still", "scenario"):
        c, _ = control.build_controller(cfg, str(synthetic_bundles["b1_zero"]), goal=goal)
        orig_goal, orig_base_goal = c.goal, c.base.goal
        st = control.self_test_policy(c, b1_bundle["golden"], cfg=cfg)
        assert st["max_torque_err"] <= lc.GOLDEN_TOL
        assert c.goal is orig_goal and c.base.goal is orig_base_goal
        assert c._int_v == 0.0 and c._int_theta == 0.0 and c._time == 0.0
        assert c.base.pitch_pid._integral == 0.0 and c.base._integral == 0.0 and c.base._time == 0.0

    # 2) Perturbed in-memory weight fails self_test_policy and STILL restores goal & resets integrators
    l_bundle = control.load_policy_bundle(synthetic_bundles["l_pid"], cfg)
    c_l, _ = control.build_controller(cfg, str(synthetic_bundles["l_pid"]), goal="scenario")
    orig_l_goal = c_l.goal
    c_l.linear[0] += 1e-2
    with pytest.raises(ValueError):
        control.self_test_policy(c_l, l_bundle["golden"], cfg=cfg)
    assert c_l.goal is orig_l_goal
    assert c_l._int_v == 0.0 and c_l._int_theta == 0.0 and c_l._time == 0.0

    # 3) C1: NaN weight in controller MUST fail self_test_policy (never silently pass NaN > tol)
    c_nan, _ = control.build_controller(cfg, str(synthetic_bundles["l_pid"]), goal="still")
    c_nan.linear[0] = float("nan")
    with pytest.raises(ValueError):
        control.self_test_policy(c_nan, l_bundle["golden"], cfg=cfg)

    # 4) Perturbed final.npz with updated hashes passes integrity check but fails startup self-test in balance.main
    perturbed = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "perturbed_bundle")
    with np.load(perturbed / "final.npz") as f:
        p_mod = f["params"].copy()
    p_mod[0] += 1e-2
    _rewrite_npz(perturbed / "final.npz", perturbed / "final.npz", params=p_mod)
    _refresh_manifest_hashes(perturbed)
    with pytest.raises(SystemExit):
        balance.main(["--sim", "--controller", str(perturbed), "--seconds", "1", "--outdir", str(tmp_path / "o_pert")])


# ---- 6. Logging, animate.py, & replay_policy (P4a-P4c) ---------------------------------------------
@pytest.mark.parametrize("bundle_key, expected_n_obs", [("l_pid", 11), ("b1_zero", 13)])
def test_policy_logging_summary_and_animate_compatibility(tmp_path, cfg, synthetic_bundles, bundle_key, expected_n_obs):
    """P4a + P4c: policy _raw.csv has 11/13 obs_* + pre_clip/saturated/integrators_frozen/residual_torque,
    _summary.txt has provenance + saturation + envelope lines, and _run.csv works with load_trajectory
    and simulation/animate.py RunAnimation."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from balance_sim import load_trajectory
    from balance_sim.policy import OBSERVATIONS, RESIDUAL_INPUTS
    import animate

    bdir = synthetic_bundles[bundle_key]
    bundle = control.load_policy_bundle(bdir, cfg)
    out = tmp_path / bundle_key
    reason = balance.main(["--sim", "--sim-noise", "--goal", "scenario", "--controller", str(bdir),
                           "--seconds", "3", "--outdir", str(out), "--label", bundle_key])
    assert reason == "done"

    # Check _raw.csv schema and values
    (raw_csv,) = out.glob("*_raw.csv")
    with open(raw_csv, newline="") as f:
        reader = csv.DictReader(f)
        obs_names = OBSERVATIONS + (RESIDUAL_INPUTS if bundle_key == "b1_zero" else ())
        assert len(obs_names) == expected_n_obs
        expected_cols = list(balance.RAW_COLUMNS) + [f"obs_{n}" for n in obs_names] + [
            "pre_clip", "saturated", "integrators_frozen", "residual_torque"
        ]
        assert reader.fieldnames == expected_cols
        rows = list(reader)
    pre_arm = [r for r in rows if r["phase"] == "0"]
    armed = [r for r in rows if r["phase"] == "1" and r["pre_clip"] != ""]
    assert pre_arm and armed
    assert all(r["pre_clip"] == "" and r["obs_v_meas"] == "" for r in pre_arm)
    for r in armed[:5]:
        for n in obs_names:
            assert math.isfinite(float(r[f"obs_{n}"]))
        assert math.isfinite(float(r["pre_clip"]))
        assert r["saturated"] in ("0", "1")
        assert r["integrators_frozen"] in ("0", "1")
        assert math.isfinite(float(r["residual_torque"]))

    # Check _summary.txt lines (P2e, P4c)
    summary = next(out.glob("*_summary.txt")).read_text()
    assert bundle["name"] in summary and bundle["kind"] in summary and bundle["sha256"] in summary
    assert "saturated:" in summary and "integrators frozen:" in summary
    assert "envelope" in summary.lower()

    # Check _run.csv with load_trajectory and animate.RunAnimation
    (run_csv,) = out.glob("*_run.csv")
    result, meta = load_trajectory(run_csv)
    assert meta["controller"]["bundle_name"] == bundle["name"]
    assert meta["controller"]["sha256"] == bundle["sha256"]
    runs = animate.runs_from_csv([run_csv])
    viewer = animate.RunAnimation(runs, robot=cfg.build_robot_params(), pushes=())
    try:
        artists = viewer.draw(0)
        assert len(artists) > 0
        viewer.draw(viewer.n_frames - 1)
    finally:
        plt.close(viewer.fig)


def test_replay_policy_on_normal_tripped_and_perturbed_logs(tmp_path, synthetic_bundles):
    """P4b: replay_policy.replay_log matches tick-for-tick (<= 1e-12) on normal and tripped runs,
    and raises ValueError if a logged torque in _run.csv is perturbed by 1e-6."""
    from robot import replay_policy

    # 1) Normal "done" scenario run (B1 residual exercises base PID + policy state)
    out_ok = tmp_path / "ok"
    reason = balance.main(["--sim", "--sim-noise", "--goal", "scenario",
                           "--controller", str(synthetic_bundles["b1_zero"]),
                           "--seconds", "4", "--outdir", str(out_ok), "--label", "b1_ok"])
    assert reason == "done"
    (run_ok,) = out_ok.glob("*_run.csv")
    rep_ok = replay_policy.replay_log(run_ok)
    assert rep_ok["n_ticks"] > 100 and rep_ok["max_torque_diff"] <= 1e-12

    # 2) Tripped run: tilt trip (duty capped low)
    out_tilt = tmp_path / "tilt"
    reason_tilt = balance.main(["--sim", "--duty-cap", "0.02", "--sim-pitch", "0.045",
                                "--controller", str(synthetic_bundles["l_pid"]),
                                "--seconds", "10", "--outdir", str(out_tilt), "--label", "l_tilt"])
    assert reason_tilt.startswith("tilt")
    (run_tilt,) = out_tilt.glob("*_run.csv")
    rep_tilt = replay_policy.replay_log(run_tilt)
    assert rep_tilt["n_ticks"] > 0 and rep_tilt["max_torque_diff"] <= 1e-12

    # 3) Tripped run: --inject ood + --envelope-trip
    out_ood = tmp_path / "ood"
    reason_ood = balance.main(["--sim", "--controller", str(synthetic_bundles["l_pid"]),
                               "--envelope-trip", "2.0", "--inject", "ood", "--inject-at", "0.5",
                               "--seconds", "3", "--outdir", str(out_ood), "--label", "l_ood"])
    assert "envelope" in reason_ood
    (run_ood,) = out_ood.glob("*_run.csv")
    rep_ood = replay_policy.replay_log(run_ood)
    assert rep_ood["n_ticks"] > 0 and rep_ood["max_torque_diff"] <= 1e-12

    # 4) Perturb one logged torque in _run.csv by 1e-6 -> replay_log must raise ValueError
    lines = run_ok.read_text().splitlines()
    header = lines[1].split(",")
    t_idx = header.index("torque")
    row_idx = min(20, len(lines) - 1)
    fields = lines[row_idx].split(",")
    fields[t_idx] = repr(float(fields[t_idx]) + 1e-6)
    lines[row_idx] = ",".join(fields)
    run_ok.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        replay_policy.replay_log(run_ok)


# ---- 7. Envelope monitor & trip (P5a) --------------------------------------------------------------
def test_envelope_monitor_and_ood_trip(tmp_path, cfg, synthetic_bundles):
    """P5a: --envelope-trip trips ~0.1 s after --inject ood; without --envelope-trip it logs excursions
    without tripping on envelope; --envelope-trip / --inject ood refused for non-policy controllers."""
    from balance_sim import load_trajectory

    # Unit check on EnvelopeMonitor: brief spike resets counter, sustained excursion trips at trip_s
    bundle = control.load_policy_bundle(synthetic_bundles["l_pid"], cfg)
    dt = cfg.timing.control_dt_s
    n_trip = max(1, int(round(lc.ENVELOPE_TRIP_S / dt)))
    mon = control.EnvelopeMonitor(bundle["envelope"], dt, trip_k=2.0)
    in_bounds = np.zeros(len(bundle["envelope"]["observation_names"]))
    ood_vec = in_bounds.copy()
    ood_vec[0] = 10.0 * max(1.0, bundle["envelope"]["observations"]["v_meas"]["max_abs"])
    for _ in range(n_trip - 1):
        assert mon.observe(ood_vec) is None
    assert mon.observe(in_bounds) is None          # one in-bounds tick resets the streak
    out = [mon.observe(ood_vec) for _ in range(n_trip)]
    assert all(x is None for x in out[:-1]) and out[-1] is not None
    assert "envelope" in out[-1].lower() and "v_meas" in out[-1]
    assert any("envelope" in line.lower() for line in mon.summary_lines())

    # Closed loop WITH --envelope-trip 2.0 and --inject ood
    out_trip = tmp_path / "trip"
    reason_trip = balance.main(["--sim", "--controller", str(synthetic_bundles["l_pid"]),
                                "--envelope-trip", "2.0", "--inject", "ood", "--inject-at", "1.0",
                                "--seconds", "3", "--outdir", str(out_trip), "--label", "env_trip"])
    assert "envelope" in reason_trip.lower()
    r_trip, _ = load_trajectory(next(out_trip.glob("*_run.csv")))
    assert 1.05 <= float(r_trip.t[-1]) <= 1.30

    # H4: --inject ood WITHOUT --envelope-trip is refused (must require --envelope-trip AND sim/bench/no-drive)
    with pytest.raises(SystemExit):
        balance.main(["--sim", "--controller", str(synthetic_bundles["l_pid"]),
                      "--inject", "ood", "--inject-at", "1.0",
                      "--seconds", "2", "--outdir", str(tmp_path / "ood_no_trip")])

    # P4c: WITHOUT --envelope-trip, a tightened envelope logs OOD excursions without tripping on envelope
    tight = _copy_bundle(synthetic_bundles["l_pid"], tmp_path / "tight_env")
    env_data = json.loads((tight / "envelope.json").read_text())
    env_data["observations"]["theta_meas"] = {"p00_1": -1e-4, "p99_9": 1e-4, "max_abs": 1e-4}
    (tight / "envelope.json").write_text(json.dumps(env_data))
    _refresh_manifest_hashes(tight)
    out_log = tmp_path / "log_only"
    reason_log = balance.main(["--sim", "--sim-noise", "--controller", str(tight),
                               "--seconds", "2", "--outdir", str(out_log), "--label", "env_log"])
    assert reason_log == "done"
    summary_log = next(out_log.glob("*_summary.txt")).read_text()
    assert "envelope" in summary_log.lower() and "theta_meas" in summary_log

    # Refused for PID (no envelope)
    with pytest.raises(SystemExit):
        balance.main(["--sim", "--controller", "hand", "--envelope-trip", "2.0",
                      "--seconds", "1", "--outdir", str(tmp_path / "pid_env")])
    with pytest.raises(SystemExit):
        balance.main(["--sim", "--controller", "hand", "--inject", "ood", "--envelope-trip", "2.0",
                      "--seconds", "1", "--outdir", str(tmp_path / "pid_ood")])


def test_non_finite_torque_or_duty_in_live_loop_trips_and_coasts(tmp_path, cfg, monkeypatch):
    """H1 + P2-2: NaN torque or NaN duty must trip and coast immediately (never clamp to +duty_cap),
    and Feedforward.set_supply rejects non-finite or <= 0 voltage."""
    import csv

    ff = control.Feedforward(cfg)
    good_supply = ff.supply
    for bad_vin in (float("nan"), float("inf"), 0.0, -7.6):
        assert ff.set_supply(bad_vin) is False
        assert ff.supply == good_supply

    # 1) NaN torque from controller
    orig_act = control.Brain.act
    calls = []

    def nan_torque_after_10(self, *args):
        calls.append(1)
        if len(calls) == 10:
            return float("nan"), float("nan"), float("nan")
        return orig_act(self, *args)

    monkeypatch.setattr(control.Brain, "act", nan_torque_after_10)
    d1 = tmp_path / "nan_torque"
    reason = balance.main(["--sim", "--seconds", "3", "--outdir", str(d1), "--label", "nan_trip"])
    assert "non-finite" in reason.lower()
    rows = list(csv.DictReader(open(next(d1.glob("*_raw.csv")))))
    assert float(rows[-1]["duty_left"]) == 0.0 and float(rows[-1]["duty_right"]) == 0.0
    monkeypatch.undo()

    # 2) P2-2: finite torque but NaN duty from feedforward must NOT clamp to +duty_cap in Brain.act
    orig_duties = control.Feedforward.duties
    dcalls = []

    def nan_duty_after_10(self, torque, wl, wr):
        dcalls.append(1)
        if len(dcalls) == 10:
            return float("nan"), 0.0
        return orig_duties(self, torque, wl, wr)

    monkeypatch.setattr(control.Feedforward, "duties", nan_duty_after_10)
    d2 = tmp_path / "nan_duty"
    reason2 = balance.main(["--sim", "--seconds", "3", "--outdir", str(d2), "--label", "nan_duty"])
    assert "non-finite" in reason2.lower()
    rows2 = list(csv.DictReader(open(next(d2.glob("*_raw.csv")))))
    assert float(rows2[-1]["duty_left"]) == 0.0 and float(rows2[-1]["duty_right"]) == 0.0


# ---- 8. Comparison hooks: score_trajectory & generate_run_sheet (P8) -------------------------------
def test_score_run_and_counterbalanced_run_sheet(tmp_path, cfg, synthetic_bundles):
    """P8: score_trajectory matches cfg.score(result) including duration scaling on fallen runs;
    generate_run_sheet counterbalances controller order across blocks."""
    from dataclasses import replace
    from balance_sim import load_trajectory
    from robot import score_run

    # 1) Normal completed run
    out_ok = tmp_path / "score_ok"
    reason_ok = balance.main(["--sim", "--sim-noise", "--controller", str(synthetic_bundles["l_pid"]),
                              "--seconds", "5", "--outdir", str(out_ok), "--label", "l_score"])
    assert reason_ok == "done"
    (run_ok,) = out_ok.glob("*_run.csv")
    res_ok, _ = load_trajectory(run_ok)
    s_ok = score_run.score_trajectory(run_ok, cfg=cfg)
    assert not s_ok["fell"] and s_ok["complete"] is True
    assert s_ok["cost"] == pytest.approx(cfg.score(res_ok))
    assert s_ok["duration"] == pytest.approx(5.0)

    # 2) Fallen run: duration must equal --seconds (10.0) for fall penalty scaling, and override works
    out_fall = tmp_path / "score_fall"
    reason_fall = balance.main(["--sim", "--duty-cap", "0.02", "--sim-pitch", "0.045",
                                "--seconds", "10", "--outdir", str(out_fall), "--label", "fall_score"])
    assert reason_fall.startswith("tilt")
    (run_fall,) = out_fall.glob("*_run.csv")
    res_fall, _ = load_trajectory(run_fall)
    assert res_fall.fell and res_fall.duration == pytest.approx(10.0)
    s_fall = score_run.score_trajectory(run_fall, cfg=cfg)
    assert s_fall["fell"] and s_fall["complete"] is True and s_fall["cost"] == pytest.approx(cfg.score(res_fall))
    s_fall_20 = score_run.score_trajectory(run_fall, cfg=cfg, duration=20.0)
    assert s_fall_20["duration"] == pytest.approx(20.0)
    assert s_fall_20["cost"] == pytest.approx(cfg.score(replace(res_fall, duration=20.0)))
    assert s_fall_20["cost"] > s_fall["cost"]       # same fall time out of a longer planned run -> larger penalty

    # 3) Early non-fall abort (e.g. stale trip) is marked complete=False (M6)
    out_abort = tmp_path / "score_abort"
    reason_abort = balance.main(["--sim", "--inject", "stale", "--seconds", "5",
                                 "--outdir", str(out_abort), "--label", "abort_score"])
    assert "stale" in reason_abort
    s_abort = score_run.score_trajectory(next(out_abort.glob("*_run.csv")), cfg=cfg)
    assert not s_abort["fell"] and s_abort["complete"] is False

    # 4) Counterbalanced run sheet (ABBA across blocks balances average run index across controllers)
    ctrls = ["hand", "policy:l_pid", "policy:b1_zero"]
    sheet = score_run.generate_run_sheet(ctrls, blocks=4, goal="scenario", seconds=20.0, alternate=True)
    assert len(sheet) == 12
    assert [e["run_index"] for e in sheet] == list(range(1, 13))
    assert [e["controller"] for e in sheet[:3]] == ctrls
    assert [e["controller"] for e in sheet[3:6]] == list(reversed(ctrls))
    assert [e["controller"] for e in sheet[6:9]] == ctrls
    assert [e["controller"] for e in sheet[9:12]] == list(reversed(ctrls))
    mean_indices = {c: np.mean([e["run_index"] for e in sheet if e["controller"] == c]) for c in ctrls}
    assert len({round(v, 9) for v in mean_indices.values()}) == 1
    assert all("robot.balance" in e["command"] and "--label" in e["command"] for e in sheet)
