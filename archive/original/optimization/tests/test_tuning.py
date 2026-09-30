import json

import numpy as np
import pytest

from balance_sim import Planar2D, RobotParams
from tuning import (
    EvalConfig,
    Evaluator,
    Parameter,
    SearchSpace,
    cascaded_pid_space,
    differential_evolution_search,
    random_search,
    save_result,
)

# A short, cheap scenario so the tests run in seconds.
FAST = EvalConfig(duration=2.0, goal_steps=((0.3, 0.2),), start_pitches=(0.1,), train_seeds=(1,))
BASELINE = [4.0, 0.5, 0.15, 0.1, 0.03]


@pytest.fixture
def evaluator():
    return Evaluator(FAST, cascaded_pid_space(FAST.params))


# ---- search space ----------------------------------------------------------

def test_encode_decode_round_trip():
    space = cascaded_pid_space(RobotParams())
    values = np.array([2.0, 1.0, 0.3, 0.1, 0.05])
    assert np.allclose(space.decode(space.encode(values)), values)


def test_decode_respects_bounds_and_clips():
    space = cascaded_pid_space(RobotParams())
    low, high = space.decode(np.zeros(5)), space.decode(np.ones(5))
    assert np.allclose(space.decode(-3 * np.ones(5)), low)
    assert np.allclose(space.decode(7 * np.ones(5)), high)
    assert np.all(low <= high)


def test_kp_lower_bound_is_above_the_critical_gain():
    params = RobotParams()
    space = cascaded_pid_space(params)
    assert space.decode(np.zeros(5))[0] > Planar2D(params).critical_pitch_gain()


def test_log_parameter_midpoint_is_geometric_mean():
    p = Parameter("x", 1.0, 100.0, log=True)
    assert p.decode(0.5) == pytest.approx(10.0)


def test_space_names_and_dict():
    space = SearchSpace([Parameter("a", 0, 1), Parameter("b", 0, 2)])
    assert space.names == ["a", "b"]
    assert space.as_dict([0.5, 1.0]) == {"a": 0.5, "b": 1.0}


# ---- evaluator -------------------------------------------------------------

def test_same_gains_always_get_the_same_score(evaluator):
    a = evaluator.evaluate_gains(BASELINE).mean_cost
    b = evaluator.evaluate_gains(BASELINE).mean_cost
    assert a == b


def test_objective_accepts_unit_vectors(evaluator):
    unit = evaluator.space.encode(BASELINE)
    assert evaluator(unit) == pytest.approx(evaluator.evaluate_gains(BASELINE).mean_cost, rel=1e-6)


def test_falling_is_heavily_penalized(evaluator):
    too_weak = [0.1, 0.0, 0.05, 0.0, 0.0]   # Kp below the critical gain
    result = evaluator.evaluate_gains(too_weak)
    assert result.n_fell == 1
    assert result.mean_cost > 100.0
    assert result.mean_cost > 100 * evaluator.evaluate_gains(BASELINE).mean_cost


def test_different_noise_seeds_give_different_scores(evaluator):
    a = evaluator.evaluate_gains(BASELINE, seeds=[1]).mean_cost
    b = evaluator.evaluate_gains(BASELINE, seeds=[2]).mean_cost
    assert a != b


def test_scores_one_cost_per_scenario():
    cfg = EvalConfig(duration=1.0, goal_steps=((0.3, 0.2),), train_seeds=(1, 2, 3))
    result = Evaluator(cfg, cascaded_pid_space(cfg.params)).evaluate_gains(BASELINE)
    assert len(result.costs) == 3
    assert result.mean_cost == pytest.approx(np.mean(result.costs))


# ---- optimizers ------------------------------------------------------------

def test_random_search_is_reproducible_and_history_never_worsens(evaluator):
    a = random_search(evaluator, n_evals=15, seed=3)
    b = random_search(evaluator, n_evals=15, seed=3)
    assert np.allclose(a.gains, b.gains)
    assert a.n_evals == 15 and len(a.history) == 15
    costs = [c for _, c in a.history]
    assert all(x >= y for x, y in zip(costs, costs[1:]))
    assert a.best_cost == costs[-1]


def test_differential_evolution_runs_and_reports_consistent_cost(evaluator):
    result = differential_evolution_search(evaluator, max_iter=2, pop_multiplier=3, seed=1)
    assert result.method == "differential_evolution"
    assert result.n_evals >= 3 * 5 * 2
    # Reported best cost matches re-scoring the reported gains.
    assert evaluator.evaluate_gains(result.gains).mean_cost == pytest.approx(result.best_cost, rel=1e-6)
    costs = [c for _, c in result.history]
    assert all(x >= y - 1e-12 for x, y in zip(costs, costs[1:]))


def test_search_beats_a_poor_starting_guess(evaluator):
    poor = evaluator.evaluate_gains([0.4, 0.0, 0.0, 0.0, 0.0]).mean_cost
    result = random_search(evaluator, n_evals=25, seed=0)
    assert result.best_cost < poor


# ---- reporting -------------------------------------------------------------

def test_save_result_writes_summary_and_history(tmp_path, evaluator):
    result = random_search(evaluator, n_evals=5, seed=0)
    path = save_result(result, tmp_path / "run", extra={"note": "test"})
    summary = json.loads(path.read_text())
    assert summary["method"] == "random_search"
    assert set(summary["gains"]) == {"kp", "ki", "kd", "kv_p", "kv_i"}
    assert summary["note"] == "test"
    assert (tmp_path / "run" / "random_search_history.csv").exists()


# ---- bayesian optimization ---------------------------------------------

def test_bayesian_optimization_runs_or_skips_cleanly(evaluator):
    pytest.importorskip("skopt")
    from tuning import bayesian_optimization_search
    result = bayesian_optimization_search(evaluator, n_calls=6, n_initial_points=4, seed=1)
    assert result.method == "bayesian_optimization"
    assert result.n_evals == 6
    assert len(result.history) == 6
    assert evaluator.evaluate_gains(result.gains).mean_cost == pytest.approx(result.best_cost, rel=1e-6)


def test_bayesian_optimization_missing_dependency_raises_helpfully(monkeypatch, evaluator):
    from tuning import optimizers
    monkeypatch.setattr(optimizers, "gp_minimize", None)
    with pytest.raises(ImportError, match="scikit-optimize"):
        optimizers.bayesian_optimization_search(evaluator, n_calls=5)
