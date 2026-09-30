from dataclasses import replace

import numpy as np
import pytest

from batch_tuning import BatchEvaluator, cma_es_search, decode_units, differential_evolution_search, random_search
from config import CONFIG
from tuning import Evaluator
from tuning import differential_evolution_search as cpu_differential_evolution_search
from tuning import random_search as cpu_random_search

FAST = replace(CONFIG.build_eval_config(), duration=2.0, goal_steps=((0.3, 0.2),), start_pitches=(0.1, -0.05),
               train_seeds=(1, 2))
HAND_TUNED = list(CONFIG.controller.hand_tuned_gains)


@pytest.fixture(scope="module")
def space():
    return CONFIG.build_search_space()


@pytest.fixture(scope="module")
def cpu(space):
    return Evaluator(FAST, space)


@pytest.fixture
def batched(space):
    return BatchEvaluator(FAST, space)


def test_decode_units_matches_search_space(space):
    units = np.random.default_rng(0).uniform(-0.2, 1.2, (100, space.dim))   # includes out-of-range
    np.testing.assert_allclose(decode_units(space, units), [space.decode(u) for u in units], rtol=1e-14)


def test_scenarios_match_cpu_order(cpu, batched):
    assert batched.scenarios() == cpu.scenarios()
    assert batched.scenarios([7, 8]) == cpu.scenarios([7, 8])


def test_evaluate_gains_matches_cpu_including_validation_seeds(cpu, batched):
    for seeds in (None, [100, 101, 102]):
        a, b = cpu.evaluate_gains(HAND_TUNED, seeds), batched.evaluate_gains(HAND_TUNED, seeds)
        assert b.mean_cost == pytest.approx(a.mean_cost, rel=1e-12)
        assert b.n_fell == a.n_fell and b.n_runs == a.n_runs


def test_call_follows_scipy_conventions(batched, space):
    units = np.random.default_rng(1).random((6, space.dim))
    many = batched(units.T)                       # vectorized: (dim, S) -> (S,)
    assert many.shape == (6,)
    assert isinstance(batched(units[0]), float)   # single point -> float
    assert batched(units[0]) == pytest.approx(many[0], rel=1e-15)


def test_chunking_does_not_change_results(space):
    gains = decode_units(space, np.random.default_rng(2).random((25, space.dim)))
    whole = BatchEvaluator(FAST, space).evaluate_population(gains)
    chunked = BatchEvaluator(FAST, space, max_batch=7).evaluate_population(gains)
    np.testing.assert_array_equal(whole.costs, chunked.costs)


def test_random_search_matches_cpu(cpu, batched):
    a = cpu_random_search(cpu, n_evals=30, seed=5)
    b = random_search(batched, n_evals=30, seed=5, batch_size=8)
    np.testing.assert_allclose(b.gains, a.gains, rtol=1e-14)
    assert b.best_cost == pytest.approx(a.best_cost, rel=1e-12)
    np.testing.assert_allclose([c for _, c in b.history], [c for _, c in a.history], rtol=1e-12)


def test_differential_evolution_matches_cpu_deferred(cpu, batched):
    # workers=map makes the CPU version use deferred updating, like the batched one.
    a = cpu_differential_evolution_search(cpu, max_iter=3, pop_multiplier=3, seed=1, workers=map)
    b = differential_evolution_search(batched, max_iter=3, pop_multiplier=3, seed=1)
    assert b.n_evals == a.n_evals
    assert [n for n, _ in b.history] == [n for n, _ in a.history]
    np.testing.assert_allclose(b.gains, a.gains, rtol=1e-12)
    assert b.best_cost == pytest.approx(a.best_cost, rel=1e-12)


def test_cma_es_is_reproducible_and_improves(batched):
    a = cma_es_search(batched, max_iter=6, popsize=12, seed=3)
    b = cma_es_search(batched, max_iter=6, popsize=12, seed=3)
    np.testing.assert_array_equal(a.gains, b.gains)
    assert a.n_evals == 72 and len(a.history) == 6
    costs = [c for _, c in a.history]
    assert all(x >= y for x, y in zip(costs, costs[1:]))
    assert batched.evaluate_gains(a.gains).mean_cost == pytest.approx(a.best_cost, rel=1e-12)
