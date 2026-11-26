"""balance.py --added-delay-ms (E3 Block D): a k-tick FIFO on the final per-wheel duty, after feedforward.

    python -m pytest hardware/legobot/robot/test_added_delay.py -q          (from the repo root)

The FIFO sits where balance_sim.simulate's `latency` does (simulator.py:126-129): the controller keeps what it
issued, the HAT gets it k ticks later, zeros before the first command is due, and trips coast at once.
"""
import csv
import functools
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # hardware/legobot
from robot import balance, control, loop_config as lc  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return control.load_robot_config()


def _raw(outdir):
    (path,) = outdir.glob("*_raw.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _acted(rows):
    return [r for r in rows if r["phase"] == "1" and r["cmd_at"] != ""]


def _f(r, k):
    return float(r[k])


# ---------------------------------------------------------------------------------- the FIFO itself ---
def test_added_delay_ticks_accepts_multiples_and_refuses_the_rest():
    dt = 0.005
    assert [control.added_delay_ticks(ms, dt) for ms in (0, 0.0, 5, 10.0, 15, 20, 25, "25")] == [0, 0, 1, 2, 3, 4, 5, 5]
    assert control.added_delay_ticks(lc.MAX_ADDED_DELAY_S * 1e3, dt) == round(lc.MAX_ADDED_DELAY_S / dt)
    for bad, match in ((2.5, "whole multiple"), (7, "whole multiple"), (10.01, "whole multiple"),
                       (-5, "non-negative"), (float("nan"), "non-negative"), (float("inf"), "non-negative"),
                       (lc.MAX_ADDED_DELAY_S * 1e3 + 5, "limit"), ("ten", "number")):
        with pytest.raises(ValueError, match=match):
            control.added_delay_ticks(bad, dt)


def test_balance_refuses_a_non_multiple_before_opening_anything(tmp_path):
    for bad in ("7", "-5", "2.5"):
        with pytest.raises(SystemExit, match="refused: --added-delay-ms"):
            balance.main(["--sim", "--seconds", "1", "--added-delay-ms", bad, "--outdir", str(tmp_path)])
    assert not list(tmp_path.glob("*"))           # refused before the run: no logs


def test_fifo_k0_is_identity_and_k2_delays_by_exactly_two_ticks():
    seq = [(0.1 * math.sin(0.37 * n) + 1e-17 * n, -0.2 * math.cos(0.11 * n)) for n in range(50)]
    d0 = control.CommandDelay(0)
    out0 = [d0.push(x) for x in seq]
    assert all(a is b for a, b in zip(out0, seq))  # k = 0: the very same tuples (bit-identical)
    d2 = control.CommandDelay(2)
    out2 = [d2.push(x) for x in seq]
    assert out2[:2] == [(0.0, 0.0), (0.0, 0.0)] and out2[2:] == seq[:-2]
    d2.reset()                                     # re-armed: empty (zeros) again
    assert [d2.push(x) for x in seq[:3]] == [(0.0, 0.0), (0.0, 0.0), seq[0]]
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            control.CommandDelay(bad)


def test_brain_act_is_untouched_by_the_delay(cfg):
    """act() returns the issued torque and duties whatever k is; the loop applies the FIFO after it."""
    outs = []
    for k in (0, 3):
        controller, _ = control.build_controller(cfg, "model")
        brain = control.Brain(cfg, controller, control.PitchEstimator(cfg, 0.0, 0.0), control.Feedforward(cfg),
                              1.0, 0.01, added_delay_ticks=k)
        controller.reset()
        outs.append([brain.act([0.0, 0.01 * n, 0.02 * math.sin(n / 7), 0.1], 0.5, 0.4) for n in range(40)])
        assert brain.delay.ticks == k
    assert outs[0] == outs[1]


# ------------------------------------------------------------------------- the loop (balance --sim) ---
def test_k0_is_bit_identical_to_the_loop_without_a_fifo(tmp_path, monkeypatch):
    """--added-delay-ms 0 against the pre-FIFO path (brain.delay = None: duties = issued, as before)."""
    args = ["--sim", "--sim-noise", "--goal", "scenario", "--controller", "model", "--seconds", "4"]
    assert balance.main(args + ["--added-delay-ms", "0", "--outdir", str(tmp_path / "k0")]) == "done"
    init = control.Brain.__init__

    def no_fifo(self, *a, **kw):
        init(self, *a, **kw)
        self.delay = None

    monkeypatch.setattr(control.Brain, "__init__", no_fifo)
    assert balance.main(args + ["--outdir", str(tmp_path / "none")]) == "done"
    a, b = _raw(tmp_path / "k0"), _raw(tmp_path / "none")
    assert len(a) == len(b) > 700 and a == b      # every column of every row, as text
    assert all(r["duty_left"] == r["issued_left"] and r["duty_right"] == r["issued_right"] for r in a)


def test_k2_writes_each_issued_duty_two_ticks_later_and_logs_it(tmp_path):
    from balance_sim import load_trajectory
    out = tmp_path / "k2"
    assert balance.main(["--sim", "--sim-noise", "--goal", "scenario", "--controller", "model", "--seconds", "4",
                         "--added-delay-ms", "10", "--outdir", str(out)]) == "done"
    rows = _acted(_raw(out))
    issued = [(_f(r, "issued_left"), _f(r, "issued_right")) for r in rows]
    written = [(_f(r, "duty_left"), _f(r, "duty_right")) for r in rows]
    assert written[:2] == [(0.0, 0.0)] * 2 and written[2:] == issued[:-2]
    assert any(abs(x[0]) > 0.05 for x in issued)
    (run_csv,) = out.glob("*_run.csv")
    _, meta = load_trajectory(run_csv)
    assert meta["added_delay_ms"] == 10.0 and meta["added_delay_ticks"] == 2
    assert meta["imu_dlpf"]["gyro_dlpfcfg"] == 3
    summary = next(out.glob("*_summary.txt")).read_text()
    assert "added delay: 10.0 ms (2 ticks)" in summary
    assert "added delay realised: 2 ticks x mean tick period" in summary
    assert "IMU DLPF: gyro DLPFCFG 3 (51.2 Hz 3 dB)" in summary
    assert "write-to-tick wait, left (ESTIMATED" in summary and "write-to-tick wait, right (ESTIMATED" in summary
    assert "motor data age at the tick, left" in summary


def test_policy_keeps_its_issued_history_and_replays_exactly_with_a_delay(tmp_path, monkeypatch):
    """L's cmd_1 on tick n is the torque it ISSUED on tick n - 1 (not the delayed one), as in simulate(), so the
    laptop replay (logged observations through the policy) still matches to 0."""
    from robot import replay_policy
    monkeypatch.setattr(lc, "POLICIES_DIR", str(tmp_path / "empty_policies"))   # the repo's results/legobot/policies
    out = tmp_path / "pol"
    assert balance.main(["--sim", "--sim-noise", "--goal", "scenario", "--controller", "policy:l_pid",
                         "--seconds", "4", "--added-delay-ms", "15", "--outdir", str(out)]) == "done"
    rows = _acted(_raw(out))
    torque, cmd1 = [_f(r, "torque") for r in rows], [_f(r, "obs_cmd_1") for r in rows]
    assert cmd1[0] == 0.0 and cmd1[1:] == torque[:-1]
    written = [_f(r, "duty_left") for r in rows]
    assert written[:3] == [0.0] * 3 and written[3:] == [_f(r, "issued_left") for r in rows][:-3]
    rep = replay_policy.replay_log(next(out.glob("*_run.csv")))
    assert rep["n_ticks"] == len(rows) and rep["max_torque_diff"] == 0.0


def test_trips_coast_at_once_and_never_send_what_is_queued(tmp_path, monkeypatch):
    """A trip coasts on its own tick; the k issued duties still in the FIFO are dropped (no set_pwm after it)."""
    from robot import sim_hw
    calls = []
    set_pwm, coast = sim_hw.SimHat.set_pwm, sim_hw.SimHat.coast
    monkeypatch.setattr(sim_hw.SimHat, "set_pwm",
                        lambda self, l, r, drain=False: (calls.append(("pwm", self.w.now, l, r)), set_pwm(self, l, r))[1])
    monkeypatch.setattr(sim_hw.SimHat, "coast", lambda self: (calls.append(("coast", self.w.now)), coast(self))[1])
    monkeypatch.setattr(sim_hw.SimHat, "emergency_coast", sim_hw.SimHat.coast)
    reason = balance.main(["--sim", "--seconds", "3", "--inject", "fault", "--added-delay-ms", "25",
                           "--outdir", str(tmp_path)])
    assert "fault" in reason
    rows = [r for r in _raw(tmp_path) if r["phase"] == "1"]
    acted = _acted(rows)
    pwms = [i for i, c in enumerate(calls) if c[0] == "pwm"]
    assert len(pwms) == len(acted)                                            # one write per acted tick, no flush
    first_coast = next(i for i, c in enumerate(calls) if c[0] == "coast" and i > pwms[0])
    assert first_coast > pwms[-1]                                             # nothing written after the trip's coast
    t0 = calls[pwms[0]][1] - float(acted[0]["t"])                             # the log's time origin (virtual clock)
    assert rows[-1]["cmd_at"] == "" and abs(calls[first_coast][1] - t0 - float(rows[-1]["t"])) < 1e-9   # same tick


def test_hardware_path_delay_equals_the_sim_worlds_latency(tmp_path, monkeypatch):
    """balance --sim --added-delay-ms 10 (FIFO in the robot code) against the same run with SimWorld's command
    latency raised by 10 ms instead: the same issued commands and the same trajectory, bit for bit. Pi-clock ticks
    (--no-line-lock), so k ticks are exactly k x 5 ms; locked ticks are half the HAT line period (4.9956 ms)."""
    from robot import sim_hw
    args = ["--sim", "--sim-noise", "--no-line-lock", "--goal", "scenario", "--controller", "model", "--seconds", "6"]
    assert balance.main(args + ["--added-delay-ms", "10", "--outdir", str(tmp_path / "fifo")]) == "done"
    world = sim_hw.SimWorld
    monkeypatch.setattr(sim_hw, "SimWorld", functools.partial(world, latency=0.005 + 0.010))
    assert balance.main(args + ["--outdir", str(tmp_path / "lat")]) == "done"
    a, b = _raw(tmp_path / "fifo"), _raw(tmp_path / "lat")
    assert len(a) == len(b) > 1000
    for ra, rb in zip(a, b):
        for k in ("t", "theta", "gyro", "x", "v", "w_left", "w_right", "torque", "issued_left", "issued_right"):
            assert ra[k] == rb[k], (ra["tick"], k)
