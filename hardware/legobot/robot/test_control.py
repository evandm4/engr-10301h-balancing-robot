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
