"""Tests for fit.py on synthetic data with the real measurement's limits: 100 Hz lines, 1 degree
encoder quantization and +-1 ms timestamp jitter. numpy and pytest only.

    python -m pytest hardware/legobot/robot/measure -q          (from the repo root)
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # self-contained: import fit.py directly
import fit  # noqa: E402

DEG = math.pi / 180.0


def stream(rng, t0, t1, fs=100.0, jitter=1e-3):
    """True sample times with a random phase, and the recorded (jittered) timestamps."""
    true = np.arange(t0, t1, 1.0 / fs) + rng.uniform(0, 1.0 / fs)
    return true, true + rng.uniform(-jitter, jitter, true.size)


def encoder(theta):
    """1 count = 1 degree, as radians."""
    return np.round(theta / DEG) * DEG


# ---- spin-up ----
def spin_position(t, t_cmd, t_d, tau, w, theta0=0.0):
    s = np.maximum(t - t_cmd - t_d, 0.0)
    return theta0 + w * (s - tau * (1 - np.exp(-s / tau)))


@pytest.mark.parametrize("w", [-18.0, 6.0, 18.0])
def test_spinup_recovery(w):
    """15 random draws (timestamp jitter, sampling phase, encoder offset).

    Over 200 draws the scatter (sd) was: fast 18 rad/s: tau 1.5 %, t_d 0.8 ms, w_inf 0.1 %; slow 6 rad/s
    (a duty near the bottom of the range, only ~20 counts of rise): tau 2.8 %, t_d 1.9 ms. The bounds are
    about 3 sd: fast tau 5 % and t_d 2.5 ms, w_inf 1 %; slow tau 10 %, t_d 5 ms, w_inf 2 %. The mean error
    must also be small, which catches a bias."""
    fast = abs(w) > 10
    tol_tau, tol_td, tol_w = (0.05, 0.0025, 0.01) if fast else (0.10, 0.005, 0.02)
    rng = np.random.default_rng(1)
    d_tau, d_td = [], []
    for _ in range(15):
        t_cmd = rng.uniform(100, 200)
        true, rec = stream(rng, t_cmd - 0.3, t_cmd + 1.0)
        y = encoder(spin_position(true, t_cmd, 0.012, 0.060, w, theta0=rng.uniform(-1, 1)))
        f = fit.fit_spinup(rec, y, t_cmd)
        assert abs(f.tau / 0.060 - 1) < tol_tau
        assert abs(f.t_d - 0.012) < tol_td
        assert abs(f.omega / w - 1) < tol_w
        assert f.rms < 0.02          # about the 1 degree quantization (0.0175 rad) / sqrt(12) plus jitter
        d_tau.append(f.tau / 0.060 - 1)
        d_td.append(f.t_d - 0.012)
    assert abs(np.mean(d_tau)) < 0.015
    assert abs(np.mean(d_td)) < 0.001


# ---- coast-down ----
def coast_position(t, alpha, beta, w0):
    """Exact solution of w_dot = -alpha - beta w until w = 0, then at rest."""
    k = alpha / beta
    t_stop = math.log((w0 + k) / k) / beta
    s = np.minimum(t, t_stop)
    return (w0 + k) / beta * (1 - np.exp(-beta * s)) - k * s


@pytest.mark.parametrize("sign", [+1, -1])
def test_coastdown_recovery(sign):
    """alpha within 10 %, beta within 15 %: speed comes from 50 ms windows over 1 degree counts, so the
    friction terms carry a few percent of noise and the two terms are partly collinear."""
    rng = np.random.default_rng(2)
    alphas, betas = [], []
    for _ in range(10):
        t_coast = rng.uniform(100, 200)
        true, rec = stream(rng, t_coast, t_coast + 2.5)
        y = encoder(sign * coast_position(true - t_coast, 15.0, 0.8, 25.0))
        f = fit.fit_coastdown(rec, y, t_coast)
        alphas.append(f.alpha)
        betas.append(f.beta)
    assert abs(np.mean(alphas) - 15.0) < 1.5
    assert abs(np.mean(betas) - 0.8) < 0.12


# ---- load drop ----
def test_load_drop_recovery_with_nonlinear_duty_map():
    """c = 0.010 N m s/rad with a nonlinear no-load speed (duty^0.7): c within 3 %, SE a sensible size."""
    rng = np.random.default_rng(3)
    c_true, r = 0.010, 0.0345
    rows = []
    for m_g in (50, 100, 200, 300):
        tau = m_g / 1000 * 9.80665 * r
        for d in (0.3, 0.5, 0.7, 1.0):
            w_nl = 30.0 * d ** 0.7
            rows.append((d, tau, w_nl - tau / c_true + rng.normal(0, 0.3), w_nl + rng.normal(0, 0.3)))
    res = fit.fit_load_drop(rows)
    assert abs(res["c"] - c_true) < 0.03 * c_true
    assert 0 < res["se"] < 0.1 * c_true
    assert len(res["per_mass"]) == 4


# ---- balance point ----
def simulate_fall(rng, c, e, theta0, theta_start, rate_bias_error, dt_sim=0.0005, fs=200.0):
    """Release at rest from theta_start; theta_ddot = c (theta - theta0) + e theta_dot for up to 1.5 s or
    |theta| > 0.35. Returns sample times, true theta, and the gyro reading (noise 0.01 rad/s, bias error)."""
    th, w, t = theta_start, 0.0, 0.0
    ts, ths, ws = [], [], []
    every = int(round(1 / fs / dt_sim))
    i = 0
    while t < 1.5 and abs(th) < 0.35:
        if i % every == 0:
            ts.append(t)
            ths.append(th)
            ws.append(w)
        acc = c * (th - theta0) + e * w
        w += acc * dt_sim
        th += w * dt_sim
        t += dt_sim
        i += 1
    ts, ws = np.array(ts), np.array(ws)
    return ts, np.array(ths), ws + rate_bias_error + rng.normal(0, 0.01, ws.size)


def test_balance_point_recovery():
    """10 falls from both sides: pooled theta0 within 0.005 rad (the plan's accuracy target) and c within 10 %.
    Noise: gyro 0.01 rad/s white plus 0.003 rad/s of residual bias, accelerometer anchor 0.002 rad per trial."""
    rng = np.random.default_rng(4)
    c, e, theta0 = 45.0, -0.4, 0.025
    bias_err = 0.003
    thetas, rates, accs = [], [], []
    for k in range(10):
        start = theta0 + (0.04 if k % 2 else -0.04) * rng.uniform(0.5, 1.0)
        t, th_true, gyro = simulate_fall(rng, c, e, theta0, start, bias_err)
        th_anchor = start + rng.normal(0, 0.002)
        # what the script does: remove the (imperfect) bias estimate, integrate from the anchor, derive theta_ddot
        th, rate, acc = fit.balance_segment(t, gyro - 0.0, th_anchor)
        thetas.append(th)
        rates.append(rate)
        accs.append(acc)
    th_all, r_all, a_all = (np.concatenate(v) for v in (thetas, rates, accs))
    res = fit.fit_balance(th_all, r_all, a_all)
    assert abs(res["theta0"] - theta0) < 0.005
    assert abs(res["c"] - c) < 0.1 * c
    assert 0 < res["se_theta0"] < 0.01


def test_balance_trim_drops_outliers():
    rng = np.random.default_rng(5)
    th = rng.uniform(-0.2, 0.2, 400)
    td = rng.normal(0, 0.3, 400)
    acc = 40 * (th - 0.02) - 0.3 * td + rng.normal(0, 0.2, 400)
    acc[:5] += 5.0                      # glitches, 25 sd of the noise
    res = fit.fit_balance(th, td, acc)
    assert res["n_dropped"] >= 5
    assert abs(res["theta0"] - 0.02) < 0.003


# ---- small pieces ----
def test_linfit_se_and_steady_speed():
    rng = np.random.default_rng(6)
    t = np.arange(0, 1, 0.01)
    y = 3.0 * t + 1.0 + rng.normal(0, 0.05, t.size)
    coef, se, sd = fit.linfit(np.column_stack([np.ones_like(t), t]), y)
    assert abs(coef[1] - 3.0) < 4 * se[1]
    assert abs(sd - 0.05) < 0.01
    w, sew = fit.steady_speed(t, y, 0.5, 1.0)
    assert abs(w - 3.0) < 4 * sew


def test_allan_white_noise():
    """White rate noise of sd s gives adev = s / sqrt(m) (m samples per tau)."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, 200 * 120)
    taus, ad = fit.allan_dev(x, 200.0, [0.005, 0.05, 0.5, 5.0])
    for tau, a in zip(taus, ad):
        m = round(tau * 200)
        assert abs(a - 0.01 / math.sqrt(m)) < 0.1 * 0.01 / math.sqrt(m)
    assert math.isnan(fit.allan_dev(x[:100], 200.0, [5.0])[1][0])
