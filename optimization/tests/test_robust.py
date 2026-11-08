"""Robust scoring: many seeds, randomized robots, and the chaos penalty."""

from dataclasses import replace

import numpy as np
import pytest

from config import CONFIG
from batch_tuning import (
    BatchEvaluator,
    RobustnessConfig,
    cma_es_search,
    decode_units,
    robot_at,
    sample_delay_steps,
    sample_robots,
)
from tuning import Evaluator

SHORT = replace(CONFIG.build_eval_config(), duration=2.0, goal_steps=((0.3, 0.3), (1.2, -0.2)),
                start_pitches=(0.1, -0.08), train_seeds=(1, 2))
HAND_TUNED = np.array(CONFIG.controller.hand_tuned_gains)
# The project's robustness settings from config.py, with fewer seeds so the tests are quick.
ROBUST = replace(CONFIG.build_robustness(), train_seeds=(1, 2, 3, 4), validation_seeds=(50, 51, 52))


def test_project_config_turns_everything_on():
    assert CONFIG.build_robustness().randomize and CONFIG.build_robustness().sensitivity_weight > 0
    assert len(CONFIG.build_robustness().train_seeds) == CONFIG.robustness.n_train_seeds


@pytest.fixture(scope="module")
def space():
    return CONFIG.build_search_space()


def _near_hand_tuned(n, seed=4):
    return HAND_TUNED * np.random.default_rng(seed).uniform(0.8, 1.2, (n, len(HAND_TUNED)))


def test_config_validation():
    with pytest.raises(ValueError, match="randomize"):
        RobustnessConfig((1,), randomize=(("n_motors", 0.1),))
    with pytest.raises(ValueError, match="spread"):
        RobustnessConfig((1,), randomize=(("body_mass", 1.5),))
    with pytest.raises(ValueError):
        RobustnessConfig((1,), sensitivity_nudge=0.0)


def test_range_and_latency_validation():
    with pytest.raises(ValueError, match="low <= high"):
        RobustnessConfig((1,), randomize_range=(("wheel_friction", 0.002, 0.001),))
    with pytest.raises(ValueError, match="both"):
        RobustnessConfig((1,), randomize=(("body_mass", 0.1),), randomize_range=(("body_mass", 0.7, 0.9),))
    with pytest.raises(ValueError, match="latency_range"):
        RobustnessConfig((1,), latency_range=(0.02, 0.01))
    with pytest.raises(ValueError, match="whole number"):
        sample_delay_steps(RobustnessConfig((1,), latency_range=(0.006, 0.007)), (1,), 1, 0.0025)


def test_ranged_parameters_are_drawn_in_range_without_moving_the_scaled_ones():
    base = BatchEvaluator(SHORT, CONFIG.build_search_space()).robot
    scaled_only = replace(ROBUST, randomize_range=())
    both = replace(ROBUST, randomize_range=(("wheel_friction", 0.0, 0.001), ("coulomb_friction", 0.005, 0.02)))
    a = sample_robots(base, scaled_only, (1, 2, 3), 3)
    b = sample_robots(base, both, (1, 2, 3), 3)
    np.testing.assert_array_equal(a.body_mass, b.body_mass)
    assert b.wheel_friction.min() >= 0 and b.wheel_friction.max() <= 0.001
    assert b.coulomb_friction.min() >= 0.005 and b.coulomb_friction.max() <= 0.02
    assert len(set(np.round(b.coulomb_friction, 12))) == 9


def test_latency_draws_are_whole_steps_in_range_and_repeatable():
    config = replace(ROBUST, latency_range=(0.005, 0.020))
    d = sample_delay_steps(config, tuple(range(1, 41)), 3, 0.0025)
    assert d.min() >= 2 and d.max() <= 8 and set(d) == set(range(2, 9))
    np.testing.assert_array_equal(d, sample_delay_steps(config, tuple(range(1, 41)), 3, 0.0025))
    # The same scenario gets the same latency whatever else is in the seed list.
    np.testing.assert_array_equal(d[3:6], sample_delay_steps(config, (2,), 3, 0.0025))
    assert sample_delay_steps(replace(ROBUST, latency_range=None), (1,), 3, 0.0025) is None


def test_twins_share_their_scenario_latency(space):
    ev = BatchEvaluator(SHORT, space, robustness=ROBUST)
    s = ev._scenario_set(ROBUST.train_seeds)
    np.testing.assert_array_equal(s.delay_steps[:s.n_base], s.delay_steps[s.n_base:])


def test_project_config_randomizes_friction_and_latency():
    r = CONFIG.build_robustness()
    assert {name for name, _, _ in r.randomize_range} == {"wheel_friction", "coulomb_friction"}
    assert r.latency_range is not None and r.latency_range[0] <= CONFIG.timing.latency_s <= r.latency_range[1]


def test_simulations_per_evaluation(space):
    assert BatchEvaluator(SHORT, space).simulations_per_evaluation() == 2 * 2          # tilts x seeds
    assert BatchEvaluator(SHORT, space, robustness=ROBUST).simulations_per_evaluation() == 2 * 2 * 4   # + twins
    no_twins = replace(ROBUST, sensitivity_weight=0.0)
    assert BatchEvaluator(SHORT, space, robustness=no_twins).simulations_per_evaluation() == 2 * 4


def test_ideal_actuator_robots_randomize_only_body_parameters(space):
    ev = BatchEvaluator(replace(SHORT, motor=None), space, robustness=ROBUST)
    robots = ev.robots()
    assert robots.torque_constant is None and robots.body_mass.shape == (len(ev.scenarios()),)
    # The body parameters are drawn the same way as with the DC motor.
    np.testing.assert_array_equal(robots.body_mass, BatchEvaluator(SHORT, space, robustness=ROBUST).robots().body_mass)


def test_nothing_enabled_scores_like_the_reference_evaluator(space):
    """Robust mode with no randomization and no penalty, on the reference seeds, is the plain evaluator."""
    off = RobustnessConfig(train_seeds=SHORT.train_seeds, randomize=(), sensitivity_weight=0.0)
    gains = _near_hand_tuned(6)
    plain = BatchEvaluator(SHORT, space).evaluate_population(gains)
    robust = BatchEvaluator(SHORT, space, robustness=off).evaluate_population(gains)
    np.testing.assert_array_equal(robust.costs, plain.costs)
    np.testing.assert_array_equal(robust.objective, plain.mean_cost)
    np.testing.assert_array_equal(robust.sensitivity, 0.0)


def test_robots_are_fixed_per_scenario_and_within_spread(space):
    ev = BatchEvaluator(SHORT, space, robustness=ROBUST)
    robots = ev.robots()
    assert robots.body_mass.shape == (len(ev.scenarios()),)
    for name, spread in ROBUST.randomize:
        ratio = getattr(robots, name) / getattr(ev.robot, name)
        assert np.all(np.abs(ratio - 1) <= spread)
        assert len(np.unique(ratio)) == len(ratio)   # every scenario gets its own robot
    # Same seed -> same robot, whatever else is in the seed list; new seeds -> new robots.
    again = sample_robots(ev.robot, ROBUST, (3, 99), len(SHORT.start_pitches))
    np.testing.assert_array_equal(again.body_mass[:2], robots.body_mass[4:6])   # seed 3's two tilts
    assert not np.isin(ev.robots(ROBUST.validation_seeds).body_mass, robots.body_mass).any()
    # Unrandomized fields stay plain numbers.
    assert robots.wheel_radius == ev.robot.wheel_radius


def test_randomized_robot_scores_match_reference_runs_with_those_parameters(space):
    ev = BatchEvaluator(SHORT, space, robustness=replace(ROBUST, train_seeds=(1, 2), sensitivity_weight=0.0))
    result = ev.evaluate_population([HAND_TUNED])
    robots, delays = ev.robots(), ev.delay_steps()
    for i, (pitch, seed) in enumerate(ev.scenarios()):
        params, motor = robot_at(robots, i)
        ref = Evaluator(replace(SHORT, params=params, motor=motor, latency=delays[i] * SHORT.dt), space)
        assert result.costs[0, i] == pytest.approx(ref.score(ref.run(HAND_TUNED, pitch, seed)), rel=1e-12)


def test_sensitivity_is_the_gap_to_the_nudged_twin(space):
    config = RobustnessConfig(train_seeds=(1, 2), sensitivity_weight=1.0, sensitivity_nudge=1e-3)
    ev = BatchEvaluator(SHORT, space, robustness=config)
    result = ev.evaluate_population([HAND_TUNED])
    cpu = Evaluator(SHORT, space)
    gaps = [abs(cpu.score(cpu.run(HAND_TUNED, p, s)) - cpu.score(cpu.run(HAND_TUNED, p + 1e-3, s)))
            for p, s in ev.scenarios()]
    assert result.sensitivity[0] == pytest.approx(np.mean(gaps), rel=1e-9)
    assert result.objective[0] == pytest.approx(result.mean_cost[0] + result.sensitivity[0], rel=1e-12)


def test_penalty_singles_out_chaotic_gains(space):
    """Across random gains, the nudge changes the cost of some far more than well-behaved ones."""
    ev = BatchEvaluator(SHORT, space, robustness=RobustnessConfig(train_seeds=(1, 2), sensitivity_weight=1.0))
    well_behaved = ev.evaluate_population(_near_hand_tuned(8))
    random = ev.evaluate_population(decode_units(space, np.random.default_rng(0).random((200, space.dim))))
    relative = lambda r: r.sensitivity / r.mean_cost
    up = well_behaved.n_fell == 0          # a perturbation that falls is not well-behaved
    assert up.sum() >= 6
    assert relative(well_behaved)[up].max() < 1e-2
    # A few percent of random gains are chaotic (8 of 200 on the 2026-11-08 robot, whose rotor
    # inertia makes it slower and most random gains calmer; it was over 10% before).
    assert (relative(random) > 10 * relative(well_behaved)[up].max()).sum() >= 5


def test_chunking_keeps_robots_and_twins_lined_up(space):
    gains = _near_hand_tuned(7, seed=2)
    whole = BatchEvaluator(SHORT, space, robustness=ROBUST).evaluate_population(gains)
    chunked = BatchEvaluator(SHORT, space, robustness=ROBUST, max_batch=20).evaluate_population(gains)
    np.testing.assert_array_equal(chunked.costs, whole.costs)
    np.testing.assert_array_equal(chunked.objective, whole.objective)


def test_optimizer_best_cost_matches_rescoring(space):
    ev = BatchEvaluator(SHORT, space, robustness=ROBUST)
    result = cma_es_search(ev, max_iter=3, popsize=8, seed=1)
    assert ev.evaluate_gains(result.gains).mean_cost == pytest.approx(result.best_cost, rel=1e-12)


def test_validation_seeds_come_from_the_robustness_config(space):
    ev = BatchEvaluator(SHORT, space, robustness=ROBUST)
    assert ev.train_seeds == ROBUST.train_seeds and ev.validation_seeds == ROBUST.validation_seeds
    assert BatchEvaluator(SHORT, space).validation_seeds is None
    assert len(ev.evaluate_gains(HAND_TUNED, seeds=ev.validation_seeds).costs) == 3 * len(SHORT.start_pitches)


def test_torch_backend_matches_numpy_in_robust_mode(space):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    gains = _near_hand_tuned(20, seed=6)
    ref = BatchEvaluator(SHORT, space, robustness=ROBUST).evaluate_population(gains)
    gpu = BatchEvaluator(SHORT, space, robustness=ROBUST, backend="torch", max_batch=64)   # several chunks
    got = gpu.evaluate_population(gains)
    np.testing.assert_allclose(got.costs, ref.costs, rtol=1e-9)
    np.testing.assert_allclose(got.sensitivity, ref.sensitivity, rtol=1e-6, atol=1e-12)
