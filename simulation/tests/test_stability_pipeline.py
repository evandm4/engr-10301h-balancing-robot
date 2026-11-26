"""balance_sim.stability_pipeline: the LEGO loop linearized WITH its sensor pipeline.

  * every stage off, the model is stability.py's loop exactly (spectrum and margins, incl. the recorded
    l_e2_s1 / MB PID cruise margins 12.25 / 25.25 ms of REVIEW_l_e2_s1 Finding 2);
  * the configured pipeline and the back-EMF feedforward gain are read from the config;
  * the complementary filter reduces to ideal sensing as tau -> 0 with the accelerometer coupling off;
  * the full model matches a small-amplitude, noise-free, backlash-free nonlinear simulate() run, at standstill
    and at cruise (feedforward on, w0 != 0);
  * the pipeline margins of reports/delay_margin_check (worst PDM phase) are pinned, and L's stays far above the
    ideal 12.25 ms (the L-vs-MB ordering is deliberately not pinned: it is a timing-dependent tie);
  * the hardware-timing knobs (IMU-only DLPF delay, right encoder age, right-wheel lag) are neutral at their
    defaults, and the DLPF-split margins are pinned.
Run with the LEGO config:  ROBOT=legobot python -m pytest simulation/tests/test_stability_pipeline.py
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from balance_sim import LinearController, load_policy, loop_matrix, simulate, spectral_radius
from balance_sim.stability_pipeline import Model, Pipeline, _sub_maps, feedforward_gain, margin, rho_tick
from config import CONFIG

ROOT = Path(__file__).resolve().parents[2]
if not (CONFIG.sensor.model == "imu" and CONFIG.sensor.encoder_counts and CONFIG.actuator.model == "slow_pwm"):
    pytest.skip("stability_pipeline models the LEGO robot's pipeline (ROBOT=legobot)", allow_module_level=True)
POLICY = ROOT / "results" / "legobot" / "policies" / "l_e2_s1" / "final.npz"
if not POLICY.exists() or CONFIG.learning.reference_pid_file is None:
    pytest.skip("needs l_e2_s1 and the LEGO reference PID", allow_module_level=True)

T = CONFIG.timing.control_dt_s
BASE = CONFIG.build_robot_params()
CORNER = replace(BASE, body_com_height=BASE.body_com_height * 1.08, rotor_inertia=BASE.rotor_inertia * 1.08)
ROBOTS = {("standstill", "nominal"): BASE, ("standstill", "corner"): CORNER,
          # cruise at 0.40 m/s: the tanh Coulomb term is saturated (slope ~4e-12), so 0 in the linear model
          ("cruise", "nominal"): replace(BASE, coulomb_friction=0.0),
          ("cruise", "corner"): replace(CORNER, coulomb_friction=0.0)}
_SPEC, _PARAMS, _ = load_policy(POLICY)
CTRL = {"L": LinearController.linear_policy(_SPEC, _PARAMS, T),
        "MB": LinearController.cascaded_pid(tuple(float(g) for g in CONFIG.reference_pid_gains()), T)}


def _cruise_pipe(key, phase_sub):
    """c_ff from the full robot (its Coulomb friction sets the cruise drive torque), not the linearization's."""
    full = ROBOTS[("standstill", key[1])]
    return replace(Pipeline.from_config(CONFIG, c_ff=feedforward_gain(CONFIG, full, 0.40)), pdm_offset_sub=phase_sub)


@pytest.mark.parametrize("cname", list(CTRL))
@pytest.mark.parametrize("key", list(ROBOTS))
@pytest.mark.parametrize("lat", [0.0, 0.005, 0.010, 0.01225, 0.0125, 0.02525, 0.0375])
def test_identity_pipeline_is_stability_py(cname, key, lat):
    ctrl, p = CTRL[cname], ROBOTS[key]
    r10, rt, f, ev = Model(p, ctrl, lat, Pipeline.ideal()).critical()
    assert rt == pytest.approx(spectral_radius(p, ctrl, T, lat), rel=1e-9, abs=1e-12)
    M = loop_matrix(p, ctrl, T, lat)
    ref = np.linalg.eigvals(M @ M)
    ref, got = ref[np.abs(ref) > 1e-5], ev[np.abs(ev) > 1e-5]   # defective zero blocks split to ~sqrt(eps)
    assert len(got) == len(ref)
    for z in ref:
        assert np.min(np.abs(got - z)) < 1e-7


def test_ideal_margins_are_the_reviews():
    assert margin(ROBOTS[("cruise", "nominal")], CTRL["L"], Pipeline.ideal()) == pytest.approx(0.01225)
    assert margin(ROBOTS[("cruise", "corner")], CTRL["L"], Pipeline.ideal()) == pytest.approx(0.0100)
    assert margin(ROBOTS[("cruise", "nominal")], CTRL["MB"], Pipeline.ideal()) == pytest.approx(0.02525)
    assert margin(ROBOTS[("cruise", "corner")], CTRL["MB"], Pipeline.ideal()) == pytest.approx(0.02175)


def test_from_config_and_feedforward():
    pipe = Pipeline.from_config(CONFIG)
    assert (pipe.cf, pipe.accel, pipe.encoder, pipe.pdm) == (True, True, True, True)
    assert pipe.tau_cf == CONFIG.sensor.filter_time_constant and pipe.imu_height == CONFIG.sensor.imu_height
    assert pipe.enc_delay_sub == 20 and pipe.pdm_offset_sub == 10       # 5 ms; phase 0.25 -> 2.5 ms
    a = CONFIG.actuator
    w0 = 0.40 / BASE.wheel_radius
    import math
    tau0 = BASE.wheel_friction * w0 + BASE.coulomb_friction * math.tanh(w0 / BASE.coulomb_friction_speed)
    k, v = a.torque_constant, a.supply_voltage
    assert feedforward_gain(CONFIG, BASE, 0.40) == pytest.approx(tau0 * k / (v - k * w0), rel=1e-5)
    assert feedforward_gain(CONFIG, BASE, 0.0) == 0.0


def test_filter_limit_is_ideal():
    p = ROBOTS[("cruise", "nominal")]
    for lat in (0.005, 0.015):
        ideal = rho_tick(p, CTRL["MB"], lat, Pipeline.ideal())
        cf0 = rho_tick(p, CTRL["MB"], lat, replace(Pipeline.ideal(), cf=True, accel=False, tau_cf=1e-12))
        assert cf0 == pytest.approx(ideal, rel=1e-6)


@pytest.mark.parametrize("cname", list(CTRL))
@pytest.mark.parametrize("latency", [0.0075, 0.015])
def test_matches_small_signal_simulation(cname, latency, monkeypatch):
    """Standstill, 1e-4 rad: noise, biases, backlash and encoder quantization off; the PDM replaced by its
    signal transfer (a 10 ms hold of the duty). Trajectories agree to 1e-3 of the peak."""
    import balance_sim.simulator as simmod
    monkeypatch.setattr(simmod, "pdm_update", lambda total, duty: (0.0, max(-1.0, min(1.0, duty))))
    cfg = replace(CONFIG, robot=replace(CONFIG.robot, gearbox_backlash=0.0, coupling_backlash=0.0),
                  sensor=replace(CONFIG.sensor, pitch_std=0.0, pitch_bias=0.0, pitch_rate_std=0.0,
                                 pitch_rate_bias=0.0, encoder_counts=10 ** 9))
    goal = lambda t: 0.0
    ctrl = (cfg.build_policy_controller(_SPEC, _PARAMS, goal=goal) if cname == "L"
            else cfg.build_controller(tuple(float(g) for g in CONFIG.reference_pid_gains()), goal=goal))
    th0, duration = 1e-4, 1.5
    res = simulate(cfg.build_dynamics(), ctrl, duration=duration, dt=0.0025, control_dt=T, sensor=cfg.build_sensor(0),
                   latency=latency, initial_state=np.array([0.0, 0.0, th0, 0.0]))
    th_nl = res.states[:, 2]
    pipe = Pipeline.from_config(cfg)
    m = Model(BASE, CTRL[cname], latency, pipe, sim_rounding=True)
    z0 = np.zeros(m.n)
    z0[m.ix["th"]] = th0
    _, _, A, _ = _sub_maps(BASE, 0.0)
    s = np.array([0.0, th0, 0.0])
    z0[m.ix["est"]] = th0 - (A[0] @ s + pipe.imu_height * (A[2] @ s)) / pipe.gravity
    th_lin = m.trajectory(z0, int(round(duration / 0.010)))[::10, m.ix["th"]][: len(th_nl)]
    assert np.max(np.abs(th_lin - th_nl)) < 1e-3 * np.max(np.abs(th_nl))


@pytest.mark.parametrize("cname, key, phase_sub, margin_ms", [
    ("L", ("cruise", "nominal"), 36, 29.0),     # worst PDM phase (9 ms); best 35.0 ms at 5 ms
    ("L", ("cruise", "corner"), 8, 22.0),       # worst 2 ms; best 26.0
    ("MB", ("cruise", "nominal"), 24, 26.0),    # worst 6 ms; best 31.0
    ("MB", ("cruise", "corner"), 0, 20.0),
])
def test_pipeline_cruise_margins_pinned(cname, key, phase_sub, margin_ms):
    """Point checks of reports/delay_margin_check/margins.json: stable just below, unstable just above."""
    p = ROBOTS[key]
    pipe = _cruise_pipe(key, phase_sub)
    assert rho_tick(p, CTRL[cname], margin_ms / 1000, pipe) < 1.0
    assert rho_tick(p, CTRL[cname], margin_ms / 1000 + 0.00025, pipe) > 1.0


def test_pipeline_lifts_l_above_the_ideal_margin():
    """The robust part of the reviews' conclusion: with pipeline sensing (sim lumping or hardware-like timing)
    L's worst-phase cruise margin is well above the ideal-sensing 12.25 ms. The L-vs-MB ordering is NOT pinned:
    it flips with 1.5-3 ms of sensor timing (review REVIEW.md 1a), so it is a tie, not "L >= PID"."""
    full = ROBOTS[("standstill", "nominal")]
    p = ROBOTS[("cruise", "nominal")]
    c_ff = feedforward_gain(CONFIG, full, 0.40)
    for pipe in (Pipeline.from_config(CONFIG, c_ff=c_ff), Pipeline.hardware_like(CONFIG, c_ff=c_ff)):
        for o in range(0, 40, 4):
            assert rho_tick(p, CTRL["L"], 0.0125, replace(pipe, pdm_offset_sub=o)) < 1.0
        assert margin(p, CTRL["L"], pipe, pdm_phases=range(0, 40, 4)) + pipe.imu_delay > 0.01225 + 0.005


@pytest.mark.parametrize("cname", list(CTRL))
@pytest.mark.parametrize("latency", [0.0075, 0.015])
def test_matches_small_signal_simulation_cruise(cname, latency, monkeypatch):
    """Cruise at 0.40 m/s (feedforward c_ff != 0, wheel speed w0 != 0, Coulomb saturated): the response to a
    small push, taken as the difference of a pushed and an unpushed nonlinear run, against the linear model
    started from the push's plant deviation. Noise, biases, backlash and encoder quantization off; the PDM
    replaced by its signal transfer. The push (7.9975-8.0 s) ends on a 10 ms period boundary, before any
    sample sees it, so every non-plant deviation is zero there."""
    import balance_sim.simulator as simmod
    from balance_sim import push_profile
    monkeypatch.setattr(simmod, "pdm_update", lambda total, duty: (0.0, max(-1.0, min(1.0, duty))))
    cfg = replace(CONFIG, robot=replace(CONFIG.robot, gearbox_backlash=0.0, coupling_backlash=0.0),
                  sensor=replace(CONFIG.sensor, pitch_std=0.0, pitch_bias=0.0, pitch_rate_std=0.0,
                                 pitch_rate_bias=0.0, encoder_counts=10 ** 12))
    goal = lambda t: 0.40 * min(max((t - 0.5) / 1.5, 0.0), 1.0)
    t_star, after, dt = 8.0, 1.5, 0.0025
    runs = []
    for push in (None, push_profile(((t_star - dt, 0.002, 0.05),))):
        ctrl = (cfg.build_policy_controller(_SPEC, _PARAMS, goal=goal) if cname == "L"
                else cfg.build_controller(tuple(float(g) for g in CONFIG.reference_pid_gains()), goal=goal))
        sensor = cfg.build_sensor(0)
        sensor = replace(sensor, **{f: 0.0 for f in ("position_std", "velocity_std") if hasattr(sensor, f)})
        runs.append(simulate(cfg.build_dynamics(), ctrl, duration=t_star + after, dt=dt, control_dt=T,
                             sensor=sensor, latency=latency, disturbance=push).states)
    k = int(round(t_star / dt))
    d = runs[1] - runs[0]
    v0 = runs[0][k, 1]       # L cruises ~5 % slow (leaky integrators, still creeping): linearize where it is
    assert np.max(np.abs(d[:k - 1])) == 0.0 and abs(v0 - 0.40) < 0.03 and abs(v0 - runs[0][k - 400, 1]) < 3e-3
    full = ROBOTS[("standstill", "nominal")]
    pipe = Pipeline.from_config(cfg, c_ff=feedforward_gain(CONFIG, full, v0))
    assert pipe.c_ff > 0.0
    m = Model(ROBOTS[("cruise", "nominal")], CTRL[cname], latency, pipe, sim_rounding=True)
    z0 = np.zeros(m.n)
    z0[m.ix["v"]], z0[m.ix["th"]], z0[m.ix["thd"]] = d[k, 1], d[k, 2], d[k, 3]
    z0[m.ix["acc"]] = d[k, 0] / BASE.wheel_radius - d[k, 2]       # wheel angle moved since the 7.99 s sample
    th_lin = m.trajectory(z0, int(round(after / 0.010)))[::10, m.ix["th"]]
    th_nl = d[k:, 2]
    n = min(len(th_lin), len(th_nl))
    assert np.max(np.abs(th_lin[:n] - th_nl[:n])) < 2e-3 * np.max(np.abs(th_nl[:n]))


def test_new_knobs_neutral_values_are_the_sim_pipeline():
    """A right encoder with the left's age, zero IMU delay and zero right lag change nothing."""
    p = ROBOTS[("cruise", "corner")]
    pipe = _cruise_pipe(("cruise", "corner"), 10)
    for c in CTRL:
        for lat in (0.010, 0.025):
            ref = rho_tick(p, CTRL[c], lat, pipe)
            assert rho_tick(p, CTRL[c], lat, replace(pipe, right_enc_age_sub=pipe.enc_delay_sub)) == \
                pytest.approx(ref, rel=1e-10)
            hw0 = Pipeline.hardware_like(CONFIG, c_ff=pipe.c_ff, pdm_phase=0.25, imu_delay=0.0, right_lag=0.0)
            assert rho_tick(p, CTRL[c], lat, hw0) == pytest.approx(ref, rel=1e-12)


def test_knob_validation():
    p, ctrl = ROBOTS[("cruise", "nominal")], CTRL["MB"]
    with pytest.raises(ValueError):
        Model(p, ctrl, 0.01, replace(Pipeline.from_config(CONFIG), imu_delay_sub=20))
    with pytest.raises(ValueError):
        Model(p, ctrl, 0.01, replace(Pipeline.ideal(), right_enc_age_sub=10))
    with pytest.raises(ValueError):
        Pipeline.hardware_like(CONFIG, imu_delay=0.0031)


# Linear cruise margins (no PDM; TOTAL = latency + IMU-only delay, last stable latency) for the IMU-only DLPF
# split of REVIEW.md 1a. The reviewer's nonlinear first-unstable values (nl_bisect.py) in the comments: the
# linear model is 0.75-1.75 ms later, as it already is with no split (36.5 vs 37.0 L, 31.75 vs 33.0 MB).
@pytest.mark.parametrize("cname, robot, dlpf_sub, total_ms", [
    ("L", "nominal", 0, 37.0),      # NL 36.5
    ("L", "nominal", 12, 31.0),     # NL 30.25: L loses ~2 ms per ms of DLPF
    ("L", "corner", 12, 25.25),     # NL 24.0
    ("MB", "nominal", 0, 33.0),     # NL 31.75
    ("MB", "nominal", 12, 31.5),    # NL 30.25: MB loses ~0.5 ms per ms
    ("MB", "corner", 12, 26.5),     # NL 25.25
])
def test_imu_only_delay_margins_pinned(cname, robot, dlpf_sub, total_ms):
    p = ROBOTS[("cruise", robot)]
    pipe = replace(_cruise_pipe(("cruise", robot), 10), pdm=False, imu_delay_sub=dlpf_sub)
    lat = total_ms / 1000 - pipe.imu_delay
    assert rho_tick(p, CTRL[cname], lat, pipe) < 1.0
    assert rho_tick(p, CTRL[cname], lat + 0.00025, pipe) > 1.0


def test_margin_table_ideal_column_is_stability_py():
    from balance_sim.stability_pipeline import margin_table
    rows = margin_table(CONFIG, CTRL, BASE, speeds=(("cruise", 0.40),), pdm_phases=[10])
    ideal = {(r["controller"], r["robot"]): r["worst"] for r in rows if r["model"] == "ideal"}
    assert ideal[("L", "nominal")] == pytest.approx(0.01225) and ideal[("MB", "nominal")] == pytest.approx(0.02525)
    assert ideal[("L", "corner")] == pytest.approx(0.0100) and ideal[("MB", "corner")] == pytest.approx(0.02175)
    hw = {(r["controller"], r["robot"]): r["worst"] for r in rows if r["model"] == "hw"}
    assert all(v > 0.0125 for v in hw.values())
