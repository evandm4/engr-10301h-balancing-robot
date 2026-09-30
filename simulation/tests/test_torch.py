"""The PyTorch backend against the NumPy float64 version (itself checked against the CPU code).

What "agrees" means here: single operations match exactly or to the last bit
(the GPU's sin/cos round differently from NumPy's in the last place). Whole
runs match closely for well-behaved gains. For random gains, some runs are
chaotic: a difference of 1e-15 grows until the two runs have nothing in
common. That happens between two NumPy runs too, so the test for those is
that every run the GPU disagrees on is one NumPy itself can't reproduce when
nudged by 1e-15.
"""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from batch_sim import BatchRobot, cascaded_update, derivatives, limited_integral, simulate_batch, torch_ops
from batch_sim.torch_sim import TorchSimulator
from batch_tuning import BatchEvaluator, decode_units, differential_evolution_search
from config import CONFIG

HAS_CUDA = torch.cuda.is_available()
DEVICES = ["cuda", "cpu"] if HAS_CUDA else ["cpu"]
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")

HAND_TUNED = np.array(CONFIG.controller.hand_tuned_gains)   # kp, ki, kd, kv_p
SPACE = CONFIG.build_search_space()
BASE = CONFIG.build_eval_config()
SHORT = replace(BASE, duration=2.0, goal_steps=((0.3, 0.3), (1.2, -0.2)), start_pitches=(0.1, -0.08),
                train_seeds=(1, 2))
RNG = np.random.default_rng(7)


def _t(a, device, dtype=torch.float64):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


def _flat(evaluator, gains, seeds=None):
    """Candidates x scenarios as flat arrays, in BatchEvaluator's row order."""
    seeds = tuple(evaluator.config.train_seeds if seeds is None else seeds)
    scen = evaluator.scenarios(seeds)
    n = len(scen)
    return (np.repeat(gains, n, axis=0), np.tile([p for p, _ in scen], len(gains)),
            np.tile([seeds.index(s) for _, s in scen], len(gains)), evaluator.noise_for(seeds))


def _near_hand_tuned(n, spread=0.2, seed=4):
    return HAND_TUNED * np.random.default_rng(seed).uniform(1 - spread, 1 + spread, (n, len(HAND_TUNED)))


@pytest.fixture(scope="module")
def short_eval():
    return BatchEvaluator(SHORT, SPACE)


# ---- single operations -------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_limited_integral_matches_numpy_exactly(device):
    n = 20_000
    args = [RNG.normal(0, 0.5, n), RNG.normal(0, 0.3, n), 0.005, RNG.normal(0, 0.5, n),
            np.abs(RNG.normal(0, 2, n)) * (RNG.random(n) > 0.1), 1.0, RNG.uniform(0.05, 1.0, n)]
    expected = limited_integral(*args)
    torch_args = [a if np.isscalar(a) else _t(a, device) for a in args]
    got = limited_integral(*torch_args, xp=torch_ops())
    np.testing.assert_array_equal(got.cpu().numpy(), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_limited_integral_with_scalar_limit_keeps_float64(device):
    # A scalar output limit goes through copysign; it must not be rounded to float32.
    x = _t([0.0], device), _t([1.0], device), 0.005, _t([0.39], device), _t([2.0], device)
    got = limited_integral(*x, 1.0, 0.4, xp=torch_ops())
    expected = limited_integral(*(np.array([v]) if not np.isscalar(v) else v for v in (0.0, 1.0, 0.005, 0.39, 2.0)),
                                1.0, 0.4)
    assert got.dtype == torch.float64
    np.testing.assert_array_equal(got.cpu().numpy(), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_cascaded_update_matches_numpy_exactly(device):
    n = 5000
    cols = [RNG.normal(0, 0.3, n), RNG.normal(0, 0.1, n), RNG.normal(0, 1, n)]
    gains = [RNG.uniform(1, 30, n), RNG.uniform(0, 5, n), RNG.uniform(0, 1, n), RNG.uniform(0, 0.4, n),
             RNG.uniform(0, 0.3, n)]
    integrals = [RNG.normal(0, 0.3, n), RNG.normal(0, 0.3, n)]
    expected = cascaded_update(*cols, 0.3, 0.005, *gains, *integrals, 0.4, 0.2, 1.0, 0.5)
    got = cascaded_update(*(_t(c, device) for c in cols), 0.3, 0.005, *(_t(g, device) for g in gains),
                          *(_t(i, device) for i in integrals), 0.4, 0.2, 1.0, 0.5, xp=torch_ops())
    for g, e in zip(got, expected):
        np.testing.assert_array_equal(g.cpu().numpy(), e)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dc_motor", [False, True])
def test_derivatives_match_numpy_to_the_last_bit(device, dc_motor, short_eval):
    robot = BatchRobot.from_params(SHORT.params, SHORT.motor if dc_motor else None)
    n = 5000
    state = [RNG.normal(0, 1, n), RNG.normal(0, 1, n), RNG.uniform(-1, 1, n), RNG.normal(0, 3, n)]
    torque = RNG.normal(0, 0.5, n)
    expected = derivatives(robot, tuple(state), torque)
    got = derivatives(robot, tuple(_t(s, device) for s in state), _t(torque, device), xp=torch_ops())
    for g, e in zip(got, expected):
        g = g.cpu().numpy()
        # sin/cos can differ from NumPy's in the last bit, and x_ddot / theta_ddot
        # are differences of large terms, so compare against the scale of the terms.
        assert (g == e).mean() > 0.5   # most are bit-identical (about 90% on CUDA)
        np.testing.assert_allclose(g, e, rtol=0, atol=1e-13 * np.abs(e).max())


# ---- whole runs ---------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_graph_replay_matches_running_step_by_step(dtype, short_eval):
    gains = decode_units(short_eval.space, RNG.random((64, SPACE.dim)))
    G, pitch, seed_index, noise = _flat(short_eval, gains)
    make = lambda graphs: TorchSimulator(short_eval.spec, short_eval.robot, len(G), noise, short_eval.schedule,
                                         dtype=dtype, use_graphs=graphs)
    graphed, eager = make(True).run(G, pitch, seed_index), make(False).run(G, pitch, seed_index)
    np.testing.assert_array_equal(graphed.cost, eager.cost)
    np.testing.assert_array_equal(graphed.final_state, eager.final_state)


def test_repeated_runs_are_identical(short_eval):
    G, pitch, seed_index, noise = _flat(short_eval, _near_hand_tuned(8))
    sim = TorchSimulator(short_eval.spec, short_eval.robot, len(G), noise, short_eval.schedule,
                         device=DEVICES[0], dtype=torch.float32)
    a, b = sim.run(G, pitch, seed_index), sim.run(G, pitch, seed_index)
    np.testing.assert_array_equal(a.cost, b.cost)


@pytest.mark.parametrize("device", DEVICES)
def test_float64_matches_numpy_for_well_behaved_gains(device, short_eval):
    G, pitch, seed_index, noise = _flat(short_eval, _near_hand_tuned(32))
    ref = simulate_batch(short_eval.spec, short_eval.robot, G, pitch, noise, seed_index, short_eval.schedule,
                         record=True)
    got = TorchSimulator(short_eval.spec, short_eval.robot, len(G), noise, short_eval.schedule, device=device,
                         dtype=torch.float64, record=True).run(G, pitch, seed_index)
    assert not ref.fell.any() and not got.fell.any()
    np.testing.assert_allclose(got.cost, ref.cost, rtol=1e-9)
    np.testing.assert_allclose(got.states, ref.states, rtol=0, atol=1e-9)


@needs_cuda
def test_full_default_scenario_float64_matches_numpy():
    from config import CONFIG
    config, space = CONFIG.build_eval_config(), CONFIG.build_search_space()
    gains = _near_hand_tuned(16, seed=5)
    ref = BatchEvaluator(config, space).evaluate_population(gains)
    got = BatchEvaluator(config, space, backend="torch").evaluate_population(gains)
    np.testing.assert_array_equal(got.n_fell, ref.n_fell)
    np.testing.assert_allclose(got.costs, ref.costs, rtol=1e-9)


@needs_cuda
def test_every_disagreement_on_random_gains_is_chaos(short_eval):
    gains = decode_units(short_eval.space, np.random.default_rng(0).random((300, SPACE.dim)))
    G, pitch, seed_index, noise = _flat(short_eval, gains)
    run = lambda p: simulate_batch(short_eval.spec, short_eval.robot, G, p, noise, seed_index, short_eval.schedule)
    ref, nudged = run(pitch), run(pitch * (1 + 1e-15))
    gpu = TorchSimulator(short_eval.spec, short_eval.robot, len(G), noise, short_eval.schedule,
                         dtype=torch.float64).run(G, pitch, seed_index)

    differs = lambda r: np.abs(r.cost - ref.cost) > 1e-6 * np.abs(ref.cost)
    gpu_differs, sensitive = differs(gpu), differs(nudged)
    # Most runs agree, and those that don't are ones NumPy can't reproduce either.
    assert gpu_differs.mean() < 0.5
    assert (gpu_differs & ~sensitive).sum() <= 0.05 * max(1, gpu_differs.sum())


@pytest.mark.parametrize("device", DEVICES)
def test_float32_close_to_float64_for_well_behaved_gains(device, short_eval):
    G, pitch, seed_index, noise = _flat(short_eval, _near_hand_tuned(32))
    ref = simulate_batch(short_eval.spec, short_eval.robot, G, pitch, noise, seed_index, short_eval.schedule)
    got = TorchSimulator(short_eval.spec, short_eval.robot, len(G), noise, short_eval.schedule, device=device,
                         dtype=torch.float32).run(G, pitch, seed_index)
    assert not got.fell.any()
    np.testing.assert_allclose(got.cost, ref.cost, rtol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_everyone_falls_early_stop_and_fall_times(device, short_eval):
    """Gains too weak to balance: every robot falls, the loop stops early, fall times match NumPy."""
    weak = np.tile([0.1, 0.0, 0.05, 0.0, 0.0], (4, 1))
    G, pitch, seed_index, noise = _flat(short_eval, weak)
    ref = simulate_batch(short_eval.spec, short_eval.robot, G, pitch, noise, seed_index, short_eval.schedule,
                         record=True)
    got = TorchSimulator(short_eval.spec, short_eval.robot, len(G), noise, short_eval.schedule, device=device,
                         dtype=torch.float64, record=True).run(G, pitch, seed_index)
    assert got.fell.all()
    np.testing.assert_array_equal(got.fall_time, ref.fall_time)
    np.testing.assert_allclose(got.cost, ref.cost, rtol=1e-9)
    np.testing.assert_allclose(got.states, ref.states, atol=1e-9)   # history padded after the stop


@pytest.mark.parametrize("device", DEVICES)
def test_no_noise_zero_latency_and_per_robot_parameters(device):
    config = replace(BASE, duration=1.0, goal_steps=((0.3, 0.2),), latency=0.0)
    ev = BatchEvaluator(config, SPACE)
    spec = replace(ev.spec, pitch_std=0.0)
    n = 16
    robot = BatchRobot.from_params(config.params, config.motor,
                                   body_mass=np.random.default_rng(1).uniform(0.6, 1.0, n))
    gains = _near_hand_tuned(n)
    ref = simulate_batch(spec, robot, gains, 0.05, schedule=ev.schedule)
    got = TorchSimulator(spec, robot, n, None, ev.schedule, device=device, dtype=torch.float64).run(gains, 0.05)
    np.testing.assert_allclose(got.cost, ref.cost, rtol=1e-9)


def test_wrong_gains_shape_rejected(short_eval):
    sim = TorchSimulator(short_eval.spec, short_eval.robot, 4, None, short_eval.schedule, device="cpu")
    with pytest.raises(ValueError, match="shape"):
        sim.run(np.zeros((3, 5)), 0.1)


# ---- evaluator and optimizers on the torch backend -------------------------------

@needs_cuda
def test_torch_evaluator_matches_numpy_evaluator(short_eval):
    gains = _near_hand_tuned(40, seed=9)
    ref = short_eval.evaluate_population(gains)
    ev = BatchEvaluator(SHORT, short_eval.space, backend="torch")
    np.testing.assert_allclose(ev.evaluate_population(gains).costs, ref.costs, rtol=1e-9)
    # Validation seeds get their own noise table and simulator.
    val = ev.evaluate_gains(HAND_TUNED, seeds=[100, 101])
    assert val.mean_cost == pytest.approx(short_eval.evaluate_gains(HAND_TUNED, seeds=[100, 101]).mean_cost,
                                          rel=1e-9)


@needs_cuda
def test_torch_chunking_padding_and_cache(short_eval):
    gains = _near_hand_tuned(30, seed=10)
    whole = BatchEvaluator(SHORT, short_eval.space, backend="torch").evaluate_population(gains)
    ev = BatchEvaluator(SHORT, short_eval.space, backend="torch", max_batch=40)   # 10 candidates per chunk
    chunked = ev.evaluate_population(gains)
    np.testing.assert_allclose(chunked.costs, whole.costs, rtol=1e-12)
    for n in (1, 3, 7, 11):   # several batch sizes; the cache stays bounded
        ev.evaluate_population(gains[:n], seeds=[n])
    assert len(ev._torch_sims) <= ev.TORCH_CACHE_SIZE


@needs_cuda
def test_differential_evolution_on_gpu(short_eval):
    ev = BatchEvaluator(SHORT, short_eval.space, backend="torch", dtype=np.float32)
    result = differential_evolution_search(ev, max_iter=3, pop_multiplier=4, seed=1)
    assert result.n_evals == 4 * SPACE.dim * 4
    assert short_eval.evaluate_gains(result.gains).mean_cost == pytest.approx(result.best_cost, rel=1e-3)
