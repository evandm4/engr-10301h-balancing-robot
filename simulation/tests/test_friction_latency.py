"""Coulomb friction and per-simulation latency, in both engines.

Coulomb friction: tau_c tanh(w / w_eps) against the relative wheel speed w
(Planar2D.derivatives, batch_sim.physics.derivatives). Per-simulation latency:
simulate_batch(delay_steps=...) and TorchSimulator(max_delay_steps=...), which
must reproduce the reference simulator run with latency = delay_steps * dt.
"""

from dataclasses import replace

import numpy as np
import pytest

from balance_sim import Planar2D, linearize
from batch_sim import BatchRobot, build_schedule, latency_steps, simulate_batch
from batch_tuning import BatchEvaluator, spec_from_eval_config
from config import CONFIG
from tuning import Evaluator

P = CONFIG.build_robot_params()
HAND_TUNED = np.array(CONFIG.controller.hand_tuned_gains)
SPACE = CONFIG.build_search_space()
SHORT = replace(CONFIG.build_eval_config(), duration=2.0, goal_steps=((0.3, 0.3), (1.2, -0.2)),
                start_pitches=(0.1, -0.08), train_seeds=(1, 2))
DT = SHORT.dt


# ---- Coulomb friction -----------------------------------------------------------------

def test_coulomb_friction_validation():
    with pytest.raises(ValueError, match="coulomb_friction"):
        P.with_changes(coulomb_friction=-0.01)
    with pytest.raises(ValueError, match="coulomb_friction_speed"):
        P.with_changes(coulomb_friction_speed=0.0)


def test_coulomb_torque_opposes_wheel_speed_and_saturates():
    """With no command, the body-joint torque is -tau_c tanh(w / w_eps): read it off the pitch
    acceleration difference against the frictionless robot."""
    tau_c, w_eps = 0.02, 1.0
    rough = Planar2D(P.with_changes(coulomb_friction=tau_c, coulomb_friction_speed=w_eps), 1.0)
    smooth = Planar2D(P, 1.0)
    A, B = linearize(P)
    for w in (-50.0, -1.0, 0.3, 50.0):
        state = np.array([0.0, w * P.wheel_radius, 0.0, 0.0])   # relative wheel speed = w
        extra = rough.derivatives(state, 0.0) - smooth.derivatives(state, 0.0)
        friction_torque = extra[3] / B[3, 0]
        assert friction_torque == pytest.approx(-tau_c * np.tanh(w / w_eps), rel=1e-9)


def test_linearize_includes_coulomb_slope():
    params = P.with_changes(coulomb_friction=0.02, coulomb_friction_speed=2.0, wheel_friction=0.001,
                            max_torque=1e9)
    dyn = Planar2D(params, fall_angle=1.0)
    A, _ = linearize(params)
    h = 1e-7
    numeric = np.column_stack([(dyn.derivatives(h * e, 0.0) - dyn.derivatives(-h * e, 0.0)) / (2 * h)
                               for e in np.eye(4)])
    np.testing.assert_allclose(A, numeric, atol=1e-5)
    # Same as viscous friction of b + tau_c / w_eps.
    viscous, _ = linearize(P.with_changes(wheel_friction=0.001 + 0.02 / 2.0))
    np.testing.assert_allclose(A, viscous, rtol=1e-12)


def test_energy_balance_with_coulomb_friction():
    """Free motion with friction: the energy lost is the friction's work, to integrator accuracy."""
    params = P.with_changes(coulomb_friction=0.01, coulomb_friction_speed=0.5)
    dyn = Planar2D(params, fall_angle=2.0)
    state, h, lost = np.array([0.0, 0.5, 0.05, 0.0]), 1e-4, 0.0
    e0 = dyn.energy(state)
    from balance_sim.simulator import _rk4_step
    for _ in range(2000):
        w0 = state[1] / P.wheel_radius - state[3]
        state = _rk4_step(dyn, state, 0.0, h)
        w1 = state[1] / P.wheel_radius - state[3]
        f = lambda w: 0.01 * np.tanh(w / 0.5) * w
        lost += 0.5 * h * (f(w0) + f(w1))
    assert dyn.energy(state) == pytest.approx(e0 - lost, abs=2e-6)
    assert lost > 1e-3


def test_batch_matches_reference_with_coulomb_friction():
    config = replace(SHORT, params=SHORT.params.with_changes(coulomb_friction=0.02, wheel_friction=0.001))
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = HAND_TUNED * np.random.default_rng(1).uniform(0.7, 1.3, (6, 4))
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        np.testing.assert_allclose(result.costs[i], ref.evaluate_gains(g).costs, rtol=1e-9)


# ---- per-simulation latency -------------------------------------------------------------

def test_latency_steps_rounds_up():
    assert latency_steps(0.010, DT) == 4
    assert latency_steps(0.0037, DT) == 2
    assert latency_steps(0.0, DT) == 0


def _batch(config, gains, delay_steps=None):
    spec = spec_from_eval_config(config)
    robot = BatchRobot.from_params(config.params, config.motor)
    ev = BatchEvaluator(config, SPACE)
    scen = ev.scenarios()
    n = len(gains)
    pitch = np.array([p for p, _ in scen] * n)
    seed_col = np.array([config.train_seeds.index(s) for _, s in scen] * n)
    g = np.repeat(gains, len(scen), axis=0)
    delay = None if delay_steps is None else np.repeat(delay_steps, len(scen))
    return simulate_batch(spec, robot, g, pitch, noise=ev.noise_for(config.train_seeds),
                          seed_index=seed_col, delay_steps=delay), len(scen)


def test_uniform_delay_steps_reproduce_scalar_latency_exactly():
    gains = HAND_TUNED * np.random.default_rng(2).uniform(0.7, 1.3, (5, 4))
    scalar, _ = _batch(SHORT, gains)
    per_sim, _ = _batch(SHORT, gains, delay_steps=np.full(5, latency_steps(SHORT.latency, DT)))
    np.testing.assert_array_equal(per_sim.cost, scalar.cost)
    np.testing.assert_array_equal(per_sim.final_state, scalar.final_state)


@pytest.mark.parametrize("controller", ["cascaded", "lqr"])
def test_mixed_delays_match_reference_runs_at_each_latency(controller):
    """Each simulation in one batch with its own delay = the reference simulator at that latency."""
    delays = np.array([0, 1, 2, 3, 4, 6, 8])
    if controller == "cascaded":
        gains = np.tile(HAND_TUNED * 0.8, (len(delays), 1))
        mixed, n_scen = _batch(SHORT, gains, delays)
        for i, d in enumerate(delays):
            ref = Evaluator(replace(SHORT, latency=d * DT), SPACE)
            np.testing.assert_allclose(mixed.cost[i * n_scen:(i + 1) * n_scen],
                                       ref.evaluate_gains(gains[i]).costs, rtol=1e-12)
    else:
        gain = CONFIG.build_lqr_gain()
        spec = spec_from_eval_config(SHORT, controller="lqr")
        robot = BatchRobot.from_params(SHORT.params, SHORT.motor)
        mixed = simulate_batch(spec, robot, np.tile(gain, (len(delays), 1)), 0.1, delay_steps=delays)
        for i, d in enumerate(delays):
            single = simulate_batch(replace(spec, latency=d * DT), robot, gain[None], 0.1)
            assert mixed.cost[i] == single.cost[0]
            np.testing.assert_array_equal(mixed.final_state[i], single.final_state[0])


def test_ring_is_large_enough_for_long_delays():
    """A delay much longer than the configured latency (which sets the schedule's ring) still works."""
    gains = np.tile(HAND_TUNED * 0.5, (2, 1))
    mixed, n_scen = _batch(replace(SHORT, sensor=replace(SHORT.sensor)), gains, np.array([4, 13]))
    ref = Evaluator(replace(SHORT, latency=13 * DT), SPACE)
    np.testing.assert_allclose(mixed.cost[n_scen:], ref.evaluate_gains(gains[1]).costs, rtol=1e-12)
    assert build_schedule(spec_from_eval_config(SHORT)).ring_size < 13 // 2 + 2


def _torch_devices():
    torch = pytest.importorskip("torch")
    return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("controller", ["cascaded", "lqr"])
def test_torch_per_sim_delay_matches_numpy(device, controller):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from batch_sim.torch_sim import TorchSimulator

    config = replace(SHORT, params=SHORT.params.with_changes(coulomb_friction=0.015, wheel_friction=0.0005))
    spec = spec_from_eval_config(config, controller=controller)
    B = 12
    rng = np.random.default_rng(5)
    delays = rng.integers(0, 9, B)
    if controller == "lqr":
        gains = np.tile(CONFIG.build_lqr_gain(), (B, 1)) * rng.uniform(0.8, 1.2, (B, 1))
    else:
        gains = HAND_TUNED * rng.uniform(0.8, 1.2, (B, 4))
    pitch = rng.uniform(-0.1, 0.1, B)
    seeds = (1, 2)
    noise = BatchEvaluator(config, SPACE).noise_for(seeds)
    seed_index = rng.integers(0, 2, B)
    robot = BatchRobot.from_params(config.params, config.motor, body_mass=rng.uniform(0.7, 0.9, B))
    ref = simulate_batch(spec, robot, gains, pitch, noise=noise, seed_index=seed_index, delay_steps=delays)
    sim = TorchSimulator(spec, robot, B, noise=noise, device=device, dtype=torch.float64,
                         n_gains=gains.shape[1], max_delay_steps=8)
    got = sim.run(gains, pitch, seed_index, robot=robot, delay_steps=delays)
    np.testing.assert_allclose(got.cost, ref.cost, rtol=1e-9)
    np.testing.assert_array_equal(got.fell, ref.fell)
    # A second run (replaying captured graphs on the GPU) with other delays.
    delays2 = delays[::-1].copy()
    ref2 = simulate_batch(spec, robot, gains, pitch, noise=noise, seed_index=seed_index, delay_steps=delays2)
    got2 = sim.run(gains, pitch, seed_index, robot=robot, delay_steps=delays2)
    np.testing.assert_allclose(got2.cost, ref2.cost, rtol=1e-9)
    with pytest.raises(ValueError, match="delay_steps"):
        sim.run(gains, pitch, seed_index, robot=robot)
    with pytest.raises(ValueError, match="delay_steps"):
        sim.run(gains, pitch, seed_index, robot=robot, delay_steps=np.full(B, 9))
