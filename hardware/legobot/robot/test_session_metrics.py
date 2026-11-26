"""Tests for session_metrics (E3_SESSION1_PREREG.md sections 2 and 4): synthetic signals, then last session's logs."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # hardware/legobot
from robot import session_metrics as sm  # noqa: E402

SEED = 7_330_001   # ad-hoc seeds >= 7.33M
E2_LOGS = Path(r"C:\balancebot_training\legobot\pi_logs\e2_deploy")
FS = 200.0


def _t(seconds):
    return np.arange(int(round(seconds * FS)) + 1) / FS


def test_mean_theta_and_velocity():
    t = _t(10.0)
    rng = np.random.default_rng(SEED)
    theta = 0.012 + 0.01 * np.sin(2 * np.pi * 1.3 * t) + rng.normal(0, 1e-4, t.size)
    x = -0.01 * t + 0.002 * np.sin(2 * np.pi * 0.7 * t)   # the wiggle is ~0 at both ends
    assert sm.mean_theta(theta) == pytest.approx(0.012, abs=2e-4)
    assert sm.mean_velocity(t, x) == pytest.approx((x[-1] - x[0]) / 10.0, rel=1e-12)
    assert sm.mean_velocity(t, x) == pytest.approx(-0.01, abs=3e-4)


def test_ols_recovers_slope_and_residual_sd():
    rng = np.random.default_rng(SEED + 1)
    th = rng.uniform(-0.02, 0.02, 200)
    v = -0.85 * th - 0.0007 + rng.normal(0, 0.0015, th.size)
    f = sm.ols(th, v)
    assert f["slope"] == pytest.approx(-0.85, abs=0.05)
    assert f["intercept"] == pytest.approx(-0.0007, abs=3e-4)
    assert f["resid_sd"] == pytest.approx(0.0015, rel=0.15)
    assert f["r"] < -0.9
    exact = sm.ols([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert exact["slope"] == pytest.approx(2.0) and exact["r"] == pytest.approx(1.0) and exact["resid_sd"] < 1e-12


def test_theta_pp_in_consecutive_2s_windows():
    t = _t(20.0)
    theta = 0.05 * np.sin(2 * np.pi * 1.5 * t + 0.3)
    pp = sm.theta_pp_windows(t, theta)
    assert len(pp) == 10
    assert np.allclose(pp, 0.1, atol=1e-3)


@pytest.mark.parametrize("freq,inband", [(3.75, True), (1.5, False)])
def test_band_fraction_and_peak(freq, inband):
    rng = np.random.default_rng(SEED + 2)
    t = _t(20.0)
    theta = 0.01 + 0.03 * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.002, t.size)
    w = sm.band_windows(t, theta)
    assert len(w) == 5 and [x["t0"] for x in w] == [0.0, 4.0, 8.0, 12.0, 16.0]
    for x in w:
        assert x["peak_hz"] == pytest.approx(freq, abs=1e-9)
        assert (x["fraction"] > 0.9) if inband else (x["fraction"] < 0.05)


def test_band_total_excludes_the_0p25_hz_bin():
    """A 0.25 Hz component sits on bin 1 of a 4 s window and must not count in the total (> 0.25 Hz).

    Equal-amplitude 3.75 / 1.5 / 0.25 Hz sines: excluding bin 1 leaves only its Hann leakage into 0.5 Hz (1/4 of
    its power), so the band fraction is 1 / 2.25 = 0.44; counting bin 1 would give 1 / 3.25 = 0.31.
    """
    t = _t(8.0)
    base = 0.01 * np.sin(2 * np.pi * 3.75 * t) + 0.01 * np.sin(2 * np.pi * 1.5 * t)
    a = sm.band_windows(t, base)
    b = sm.band_windows(t, base + 0.01 * np.cos(2 * np.pi * 0.25 * t))
    assert [x["fraction"] for x in a] == pytest.approx([0.5, 0.5], abs=0.01)
    assert [x["fraction"] for x in b] == pytest.approx([1 / 2.25, 1 / 2.25], abs=0.02)


def _scenario(dt=1 / FS):
    """Goal 0 (2 s), ramp to +0.4, hold, ramp to -0.15, hold, ramp to 0, hold; v = goal + a per-plateau offset."""
    t = np.arange(0, 20.0 + dt / 2, dt)
    g = np.interp(t, [0, 2, 3.25, 9.5, 11.9, 14.0, 16.0, 20.0], [0, 0, 0.4, 0.4, -0.15, -0.15, 0, 0])
    offset = np.select([g == 0.4, g == -0.15, (g == 0) & (t > 10)], [-0.03, 0.017, 0.015], 0.0)
    return t, g, g + offset


def test_plateau_errors_on_scenario_profile():
    t, g, v = _scenario()
    p = sm.plateau_errors(t, v, g)
    assert [round(x["goal"], 2) for x in p] == [0.0, 0.4, -0.15, 0.0]
    assert [x["error"] for x in p] == pytest.approx([0.0, -0.03, 0.017, 0.015], abs=1e-12)
    assert p[1]["t0"] == pytest.approx(3.25, abs=0.006) and p[1]["t1"] == pytest.approx(9.5, abs=0.006)


def test_short_constant_goal_is_not_a_plateau():
    t = _t(3.0)
    g = np.where(t < 1.0, 0.0, np.where(t < 1.4, 0.2, 0.1))    # the 0.2 hold lasts 0.4 s
    assert [x["goal"] for x in sm.plateau_errors(t, np.zeros_like(t), g)] == [0.0, 0.1]


def test_wheel_ratio_ignores_missing_samples():
    t = _t(5.0)
    left, right = 100.0 * t, 80.0 * t + 3.0
    left[:3] = np.nan
    assert sm.wheel_ratio(left, right) == pytest.approx(0.8)


def test_holm():
    adj, rej = sm.holm([0.04, 0.01])
    assert adj == pytest.approx([0.04, 0.02]) and rej == [True, True]
    adj, rej = sm.holm([0.03, 0.04])
    assert adj == pytest.approx([0.06, 0.06]) and rej == [False, False]


def _b_runs(n, rng):
    runs = []
    for code, mu in (("L", 2.8), ("P", 3.4), ("M", 3.8)):
        for c in mu + rng.normal(0, 0.05, n):
            runs.append({"controller": code, "cost": float(c), "theta_pp_deg": [mu * 3, mu * 3 + 1]})
    return runs


def test_block_b_exact_mann_whitney_holm():
    runs = _b_runs(6, np.random.default_rng(SEED + 3))
    h = {x["H"]: x for x in sm.analyze_block_b(runs)}
    assert h["H-B1"]["value"]["p"] == pytest.approx(1 / 924)       # complete separation, n = 6 vs 6
    assert h["H-B1"]["value"]["p_holm"] == pytest.approx(2 / 924)
    assert h["H-B1"]["verdict"] == "pass" and h["H-B2"]["verdict"] == "pass"
    assert h["H-B3"]["value"] == pytest.approx(3.4 / 3.8, abs=0.03) and "in [0.85" in h["H-B3"]["verdict"]
    assert "True" in h["H-B4"]["verdict"]
    interim = {x["H"]: x for x in sm.analyze_block_b(_b_runs(5, np.random.default_rng(SEED + 4)))}
    assert interim["H-B1"]["verdict"].startswith("interim")


def test_block_b_fails_when_l_is_not_lower():
    runs = _b_runs(6, np.random.default_rng(SEED + 5))
    for r in runs:
        if r["controller"] == "L":
            r["cost"] += 1.0                                              # L now above P
    h = {x["H"]: x for x in sm.analyze_block_b(runs)}
    assert h["H-B1"]["verdict"] == "fail"


def _a(run, th, v, pz=-0.0027):
    return {"run": run, "theta_mean": th, "v_mean": v, "pitch_zero_file": pz}


def test_block_a_verdicts():
    rng = np.random.default_rng(SEED + 6)
    th = {"A1": 0.012, "A2": 0.001, "A3+": -0.015, "A3-": 0.015, "A5": 0.0025}
    runs = [_a(k, x, -0.85 * x + rng.normal(0, 0.0005)) for k, x in th.items()]
    h = {x["H"]: x for x in sm.analyze_block_a(runs, temps={"A1": (50.0, 52.0), "A5": (55.0, 56.0)})}
    assert h["H-A1"]["verdict"] == "pass" and h["H-A1"]["trim_pitch_zero_new"] == pytest.approx(-0.0027 + 0.012)
    assert h["H-A2 (A2)"]["verdict"] == "pass"
    assert h["H-A3"]["verdict"] == "pass" and h["H-A3"]["value"]["slope"] == pytest.approx(-0.85, abs=0.1)
    assert h["H-A4"]["verdict"] == "not supported"                   # |0.0015| < 0.005
    assert h["H-A5"]["verdict"] == "not supported"                   # 0.0008 * 6 = 0.0048 > 0.00075
    runs[-1] = _a("A5", 0.013, -0.011)
    h = {x["H"]: x for x in sm.analyze_block_a(runs)}
    assert h["H-A4"]["verdict"] == "supported" and h["H-A4"]["forward_sign"]
    assert h["H-A5"]["verdict"].startswith("not run")
    runs[2] = _a("A3+", -0.015, -0.001)                               # A3+ backward: wrong sign
    assert {x["H"]: x for x in sm.analyze_block_a(runs)}["H-A3"]["verdict"] == "fail"
    assert {x["H"]: x for x in sm.analyze_block_a([_a("A1", 0.02, 0.0)])}["H-A1"]["verdict"] == "fail"


def test_block_c_verdicts():
    band = [{"t0": float(k), "t1": float(k + 4), "fraction": 0.05, "peak_hz": 1.75} for k in range(0, 20, 4)]
    good = {"stamp": "s1", "controller": "L", "plateaus": [{"error": -0.03}, {"error": 0.02}],
            "wheel_ratio": 0.8, "band": band, "cost": 6.5, "fell": False}
    h = {x["H"]: x for x in sm.analyze_block_c([good], svr_dir="does-not-exist")}
    assert h["H-C1"]["verdict"] == "pass" and h["H-C2"]["verdict"].startswith("pass") and h["H-C3"]["verdict"] == "pass"
    assert h["H-C4"]["verdict"].startswith("not run") and h["H-C5"]["verdict"].startswith("not run")
    bad = dict(good, stamp="s2", plateaus=[{"error": 0.06}], wheel_ratio=1.05,
               band=[dict(w, peak_hz=3.75) if w["t0"] == 4.0 else w for w in band])
    m = dict(good, stamp="s3", controller="M", fell=True, cost=40.0)
    h = {x["H"]: x for x in sm.analyze_block_c([good, bad, m], svr_dir="does-not-exist")}
    assert h["H-C1"]["verdict"] == "fail" and h["H-C2"]["verdict"].startswith("fail")
    assert h["H-C3"]["verdict"] == "fail" and h["H-C5"]["verdict"] == "fail"


def test_labels_and_controllers():
    assert sm.run_id("e3_A3p_pz0.0243") == "A3+" and sm.run_id("e3_A3m_pz-0.0057") == "A3-"
    assert sm.run_id("e3_B1_M") == "B1" and sm.run_id("e2_scenario20") == ""
    pol = {"controller": {"kind": "policy", "bundle_name": "l_e2_s1", "spec": "policy:l_e2_s1"}}
    assert sm.controller_code(pol, "e3_A3p_pz0.0243") == "L"
    assert sm.controller_code({"controller": {"spec": "model", "kind": "pid"}}, "e3_B1_M") == "M"
    assert sm.controller_code({}, "e3_B2_P") == "P"
    with pytest.raises(ValueError):
        sm.controller_code(pol, "e3_B1_M")


def _write_fake_run(folder, stamp, label, theta, x, v, goal, pos_l, pos_r):
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(theta)) / FS
    meta = {"label": f"{label} {stamp}", "controller": {"kind": "policy", "bundle_name": "l_e2_s1"},
            "calibration": {"pitch_zero": -0.0027}, "reason": "done"}
    base = folder / f"{stamp}_{label}"
    with open(f"{base}_run.csv", "w", newline="") as f:
        f.write("# " + json.dumps(meta) + "\n")
        f.write("t,x,x_dot,theta,theta_dot,torque,pitch_reference,velocity_goal\n")
        for row in zip(t, x, v, theta, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), goal):
            f.write(",".join(repr(float(c)) for c in row) + "\n")
    with open(f"{base}_raw.csv", "w", newline="") as f:
        f.write("tick,t,phase,pos_left,pos_right,goal,obs_cmd_1\n")
        for k in range(20):                                    # 0.1 s unarmed, wheels parked
            f.write(f"{k},{k / FS},0,500,400,0.0,\n")
        for k, (pl, pr, g) in enumerate(zip(pos_l, pos_r, goal)):
            f.write(f"{k + 20},{(k + 20) / FS},1,{pl},{pr},{g},0.1\n")


def test_find_and_measure_a_logged_run(tmp_path):
    rng = np.random.default_rng(SEED + 7)
    n = int(10 * FS) + 1
    t = np.arange(n) / FS
    theta = 0.01 + rng.normal(0, 0.003, n)
    x = -0.0085 * t
    _write_fake_run(tmp_path / "pulled", "20261127_101010", "e3_A1_L", theta, x, np.full(n, -0.0085),
                    np.zeros(n), 500 + np.round(100 * t), 400 + np.round(70 * t))
    for key in ("20261127_101010", "101010", "e3_A1_L"):
        [p] = sm.find_runs([key], tmp_path)
    with pytest.raises(FileNotFoundError):
        sm.find_runs(["e3_A9_L"], tmp_path)
    r = sm.run_metrics(p, with_cost=False)
    assert (r["controller"], r["run"], r["block"]) == ("L", "A1", "A")
    assert r["theta_mean"] == pytest.approx(float(np.mean(theta)), rel=1e-12)
    assert r["v_mean"] == pytest.approx(-0.0085, rel=1e-9)
    assert r["wheel_ratio"] == pytest.approx(0.7, abs=0.001)
    cropped = sm.run_metrics(p, with_cost=False, sheet_row={"t_max": "5"})
    assert cropped["duration"] == pytest.approx(5.0) and len(cropped["theta_pp_deg"]) == 2


# ------------------------------------------------------------------- regression on last session's logs ---
needs_logs = pytest.mark.skipif(not E2_LOGS.is_dir(), reason="2026-11-25 floor logs not on this machine")


def _metrics(stamp, **row):
    [p] = sm.find_runs([stamp], E2_LOGS)
    return sm.run_metrics(p, with_cost=False, sheet_row=row or None)


@needs_logs
def test_reproduces_prereg_numbers():
    """The prereg's [LOG] numbers: L fit over 5 runs (000949 cut at 35 s, as prereg_stats.py), run 9 scenario."""
    L = [_metrics(s) for s in ("000104", "000439", "000828", "010508")] + [_metrics("000949", t_max="35")]
    f = sm.ols([r["theta_mean"] for r in L], [r["v_mean"] for r in L])
    assert f["slope"] == pytest.approx(-0.849, abs=0.003) and f["r"] == pytest.approx(-0.992, abs=0.001)
    assert f["intercept"] == pytest.approx(-0.0007, abs=1e-4) and f["resid_sd"] == pytest.approx(0.0015, abs=1e-4)
    s9 = _metrics("011006")
    assert [p["error"] for p in s9["plateaus"][1:]] == pytest.approx([-0.031, 0.017, 0.015], abs=6e-4)
    assert s9["wheel_ratio"] == pytest.approx(0.81, abs=0.005)
    assert s9["wheel_ratio_cruise"] == pytest.approx(0.73, abs=0.005)
    fr = [w["fraction"] for w in s9["band"]]
    assert min(fr) == pytest.approx(0.026, abs=6e-4) and max(fr) == pytest.approx(0.103, abs=6e-4)


@needs_logs
def test_reproduces_prereg_costs():
    cfg = sm.score_run.control.load_robot_config()
    cost = {s: sm.run_metrics(sm.find_runs([s], E2_LOGS)[0], cfg=cfg)["cost"]
            for s in ("000104", "010508", "235826", "235553")}
    assert [cost[s] for s in ("000104", "010508", "235826", "235553")] == pytest.approx(
        [2.70, 2.88, 3.40, 3.84], abs=0.006)
