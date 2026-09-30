import json
from dataclasses import replace

import numpy as np
import pytest

from balance_sim import critical_pitch_gain
from config import CONFIG
from tuning import (
    Evaluator,
    Parameter,
    SearchSpace,
    differential_evolution_search,
    gains_cells,
    gains_header,
    random_search,
    save_result,
)

# The project's scenario, cut down so the tests run in seconds.
BASE = CONFIG.build_eval_config()
FAST = replace(BASE, duration=2.0, goal_steps=((0.3, 0.2),), start_pitches=(0.1,), train_seeds=(1,))
BASELINE = list(CONFIG.controller.hand_tuned_gains)   # kp, ki, kd, kv_p
SPACE = CONFIG.build_search_space()


@pytest.fixture
def evaluator():
    return Evaluator(FAST, SPACE)


# ---- search space ----------------------------------------------------------

def test_search_space_has_four_gains():
    assert SPACE.names == ["kp", "ki", "kd", "kv_p"]


def test_encode_decode_round_trip():
    values = np.array([2.0, 1.0, 0.3, 0.1])
    assert np.allclose(SPACE.decode(SPACE.encode(values)), values)


def test_decode_respects_bounds_and_clips():
    low, high = SPACE.decode(np.zeros(SPACE.dim)), SPACE.decode(np.ones(SPACE.dim))
    assert np.allclose(SPACE.decode(-3 * np.ones(SPACE.dim)), low)
    assert np.allclose(SPACE.decode(7 * np.ones(SPACE.dim)), high)
    assert np.all(low <= high)


def test_kp_lower_bound_is_above_the_critical_gain():
    assert SPACE.decode(np.zeros(SPACE.dim))[0] > critical_pitch_gain(CONFIG.build_robot_params())


def test_log_parameter_midpoint_is_geometric_mean():
    p = Parameter("x", 1.0, 100.0, log=True)
    assert p.decode(0.5) == pytest.approx(10.0)


def test_space_names_and_dict():
    space = SearchSpace([Parameter("a", 0, 1), Parameter("b", 0, 2)])
    assert space.names == ["a", "b"]
    assert space.as_dict([0.5, 1.0]) == {"a": 0.5, "b": 1.0}


def test_at_bounds_flags_parameters_on_the_edge():
    space = SearchSpace([Parameter("a", 0, 1), Parameter("b", 0, 2), Parameter("c", 1, 100, log=True)])
    assert space.at_bounds([0.0, 1.0, 100.0]) == ["a", "c"]
    assert space.at_bounds([0.5, 1.999, 10.0]) == ["b"]


# ---- evaluator -------------------------------------------------------------

def test_same_gains_always_get_the_same_score(evaluator):
    a = evaluator.evaluate_gains(BASELINE).mean_cost
    b = evaluator.evaluate_gains(BASELINE).mean_cost
    assert a == b


def test_objective_accepts_unit_vectors(evaluator):
    unit = evaluator.space.encode(BASELINE)
    assert evaluator(unit) == pytest.approx(evaluator.evaluate_gains(BASELINE).mean_cost, rel=1e-6)


def test_falling_is_heavily_penalized(evaluator):
    too_weak = [0.1, 0.0, 0.05, 0.0]   # Kp below the critical gain
    result = evaluator.evaluate_gains(too_weak)
    assert result.n_fell == 1
    assert result.mean_cost > 100.0
    assert result.mean_cost > 100 * evaluator.evaluate_gains(BASELINE).mean_cost


def test_different_noise_seeds_give_different_scores(evaluator):
    a = evaluator.evaluate_gains(BASELINE, seeds=[1]).mean_cost
    b = evaluator.evaluate_gains(BASELINE, seeds=[2]).mean_cost
    assert a != b


def test_scores_one_cost_per_scenario():
    cfg = replace(BASE, duration=1.0, goal_steps=((0.3, 0.2),), start_pitches=(0.1, -0.08, 0.05),
                  train_seeds=(1, 2, 3))
    result = Evaluator(cfg, SPACE).evaluate_gains(BASELINE)
    assert len(result.costs) == result.n_runs == 9   # 3 tilts x 3 seeds
    assert result.mean_cost == pytest.approx(np.mean(result.costs))


def test_scenarios_try_every_seed_with_every_start_tilt():
    cfg = replace(BASE, start_pitches=(0.1, -0.05), train_seeds=(1, 2, 3))
    ev = Evaluator(cfg, SPACE)
    assert sorted(ev.scenarios()) == sorted((p, s) for p in (0.1, -0.05) for s in (1, 2, 3))
    assert len(ev.scenarios(seeds=[7])) == 2    # validation seeds get the same tilts


def test_evaluator_controller_knows_the_configured_limits(evaluator):
    ctrl = evaluator.controller(BASELINE)
    assert ctrl.pitch_pid.output_limit == evaluator.config.params.max_torque
    assert ctrl.pitch_pid.integral_limit == evaluator.config.pitch_integral_limit
    assert ctrl.integral_limit == evaluator.config.velocity_integral_limit
    assert ctrl.kv_i == 0.0


def test_evaluator_dc_motor_respects_robot_torque_limit():
    cfg = replace(BASE, params=BASE.params.with_changes(max_torque=0.25))
    act = Evaluator(cfg, SPACE).dynamics().actuator
    assert act.torque(10.0, 0.0) == pytest.approx(0.25)


def test_pitch_reference_option_changes_the_score():
    short = replace(BASE, duration=2.0, goal_steps=((0.3, 0.2),), start_pitches=(0.05,), train_seeds=(1,))
    a = Evaluator(replace(short, pitch_reference="commanded"), SPACE).evaluate_gains(BASELINE).mean_cost
    b = Evaluator(replace(short, pitch_reference="upright"), SPACE).evaluate_gains(BASELINE).mean_cost
    assert a < b   # leaning to accelerate is only penalized by the "upright" reference


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
    assert result.n_evals >= 3 * SPACE.dim * 2
    # Reported best cost matches re-scoring the reported gains.
    assert evaluator.evaluate_gains(result.gains).mean_cost == pytest.approx(result.best_cost, rel=1e-6)
    costs = [c for _, c in result.history]
    assert all(x >= y - 1e-12 for x, y in zip(costs, costs[1:]))


def test_differential_evolution_history_counts_real_evaluations(evaluator):
    result = differential_evolution_search(evaluator, max_iter=3, pop_multiplier=2, seed=1)
    evals = [n for n, _ in result.history]
    assert len(evals) == 3
    assert all(a < b for a, b in zip(evals, evals[1:]))
    assert evals[-1] == result.n_evals


def test_best_cost_at_budget(evaluator):
    result = random_search(evaluator, n_evals=10, seed=0)
    assert result.best_cost_at(10) == result.history[-1]
    assert result.best_cost_at(4) == result.history[3]
    assert result.best_cost_at(0) is None


def test_search_beats_a_poor_starting_guess(evaluator):
    poor = evaluator.evaluate_gains([0.4, 0.0, 0.0, 0.0]).mean_cost
    result = random_search(evaluator, n_evals=25, seed=0)
    assert result.best_cost < poor


# ---- reporting -------------------------------------------------------------

def test_save_result_writes_summary_and_history(tmp_path, evaluator):
    result = random_search(evaluator, n_evals=5, seed=0)
    path = save_result(result, tmp_path / "run", extra={"note": "test"})
    summary = json.loads(path.read_text())
    assert summary["method"] == "random_search"
    assert set(summary["gains"]) == {"kp", "ki", "kd", "kv_p"}
    assert summary["note"] == "test"
    assert (tmp_path / "run" / "random_search_history.csv").exists()


def test_gains_table_columns_line_up():
    header, row = gains_header(SPACE.names), gains_cells(BASELINE)
    assert len(header) == len(row)
    assert header.split() == SPACE.names


def test_summary_figure_draws_four_panels(evaluator):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from tuning import tuning_summary_figure

    result = random_search(evaluator, n_evals=5, seed=0)
    run = evaluator.run(BASELINE, 0.1, 1)
    fig = tuning_summary_figure({"random search": result.history}, 1.0, {"hand-tuned": run},
                                log_evaluations=True)
    assert len(fig.axes) == 4
    assert fig.axes[0].get_xscale() == "log"


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
