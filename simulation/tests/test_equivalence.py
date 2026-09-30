"""Whole runs: the batched simulator against the reference simulator, run by run.

In float64 the two do the same arithmetic in the same order, so trajectories,
fall times and final states match exactly. Costs agree to rounding error; the
only difference is that the reference code sums each integral with np.sum
while the batched code adds it up step by step.
"""

from dataclasses import replace

import numpy as np
import pytest

from balance_sim import SensorNoise
from batch_sim import BatchRobot, simulate_batch
from batch_tuning import BatchEvaluator, decode_units
from config import CONFIG
from tuning import Evaluator

HAND_TUNED = list(CONFIG.controller.hand_tuned_gains)   # kp, ki, kd, kv_p
COST_RTOL = 1e-12
SPACE = CONFIG.build_search_space()

# The project's scenario, shortened so the tests run quickly, plus variations of it.
BASE = CONFIG.build_eval_config()
SHORT = replace(BASE, duration=2.0, goal_steps=((0.3, 0.3), (1.2, -0.2)), start_pitches=(0.1, -0.08),
                train_seeds=(1, 2))
CONFIGS = {
    "default": SHORT,
    "ideal_actuator": replace(SHORT, motor=None),
    "no_noise_no_latency": replace(SHORT, sensor=SensorNoise(), latency=0.0),
    "latency_between_steps": replace(SHORT, latency=0.0037),
    "control_every_step": replace(SHORT, control_dt=0.0025),
    "upright_reference": replace(SHORT, pitch_reference="upright"),
    "friction_weak_motor": replace(SHORT, params=SHORT.params.with_changes(wheel_friction=0.003, max_torque=0.3),
                                   motor=replace(SHORT.motor, supply_voltage=6.0)),
}


def _candidates(n=24, seed=0):
    """Random gains plus corner cases: every gain at its lower / upper bound, the hand-tuned set,
    and one too weak to balance (Kp below the critical gain), so the falling path is always covered."""
    units = np.random.default_rng(seed).random((n, SPACE.dim))
    units[0], units[1] = 0.0, 1.0
    gains = decode_units(SPACE, units)
    gains[2] = HAND_TUNED
    gains[3] = [0.1, 0.0, 0.05, 0.0]
    return gains


@pytest.mark.parametrize("name", list(CONFIGS))
def test_costs_and_falls_match_reference(name):
    config = CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = _candidates()

    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        expected = ref.evaluate_gains(g)
        np.testing.assert_allclose(result.costs[i], expected.costs, rtol=COST_RTOL)
        assert result.n_fell[i] == expected.n_fell
    # The candidates should cover both outcomes, or the test isn't checking much.
    assert 0 < result.n_fell.sum() < result.costs.size


def test_five_gain_sets_with_a_velocity_integral_still_match():
    """The project dropped kv_i, but both simulators still support it."""
    ref, batched = Evaluator(SHORT, SPACE), BatchEvaluator(SHORT, SPACE)
    rng = np.random.default_rng(8)
    gains = np.column_stack([_candidates(n=10, seed=8), rng.uniform(0.0, 0.3, 10)])
    gains[0, 4] = 0.03
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        np.testing.assert_allclose(result.costs[i], ref.evaluate_gains(g).costs, rtol=COST_RTOL)
    # A zero kv_i is the same as leaving it out.
    four = batched.evaluate_population(gains[:, :4])
    zero = batched.evaluate_population(np.column_stack([gains[:, :4], np.zeros(10)]))
    np.testing.assert_array_equal(four.costs, zero.costs)


@pytest.mark.parametrize("name", ["default", "latency_between_steps"])
def test_trajectories_match_reference_exactly(name):
    config = CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = _candidates(n=12, seed=1)
    pitch, seed = ref.scenarios()[1]

    run = simulate_batch(batched.spec, batched.robot, gains, pitch, noise=batched.noise_for((seed,)),
                         seed_index=np.zeros(len(gains), dtype=int), schedule=batched.schedule, record=True)
    for i, g in enumerate(gains):
        expected = ref.run(g, pitch, seed)
        n = len(expected.t)
        np.testing.assert_array_equal(run.states[:n, i], expected.states)
        np.testing.assert_array_equal(run.final_state[i], expected.states[-1])
        assert run.fell[i] == expected.fell
        if expected.fell:
            assert run.fall_time[i] == expected.fall_time
            assert np.all(run.states[n:, i] == expected.states[-1])   # frozen after the fall
        else:
            assert np.isnan(run.fall_time[i])


def test_per_robot_parameters_match_separate_reference_runs():
    """A batch where every robot has its own mass and COM height (domain randomization)."""
    config = SHORT
    rng = np.random.default_rng(3)
    n = 10
    masses, heights = rng.uniform(0.6, 1.0, n), rng.uniform(0.07, 0.13, n)
    batched = BatchEvaluator(config, SPACE)
    robot = BatchRobot.from_params(config.params, config.motor, body_mass=masses, body_com_height=heights)
    pitch, seed = batched.scenarios()[0]
    run = simulate_batch(batched.spec, robot, np.tile(HAND_TUNED, (n, 1)), pitch,
                         noise=batched.noise_for((seed,)), seed_index=np.zeros(n, dtype=int),
                         schedule=batched.schedule)
    for i in range(n):
        params = config.params.with_changes(body_mass=masses[i], body_com_height=heights[i])
        ref = Evaluator(replace(config, params=params), SPACE)
        expected = ref.run(HAND_TUNED, pitch, seed)
        np.testing.assert_allclose(run.cost[i], ref.score(expected), rtol=COST_RTOL)
        np.testing.assert_array_equal(run.final_state[i], expected.states[-1])


def test_full_default_scenario_matches_reference():
    """The real tuning setup from config.py (6 s runs, 9 scenarios), on a few candidates."""
    ref, batched = Evaluator(BASE, SPACE), BatchEvaluator(BASE, SPACE)
    gains = _candidates(n=4, seed=2)
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        np.testing.assert_allclose(result.costs[i], ref.evaluate_gains(g).costs, rtol=COST_RTOL)


def test_float32_stays_close_to_float64():
    """Preview of GPU precision: float32 costs for stable gains within 0.1% of float64.

    Gains near the edge of falling can legitimately diverge in float32 (a tiny
    difference decides fall / no fall), so this checks the hand-tuned gains and
    small perturbations of them.
    """
    gains = np.array(HAND_TUNED) * np.random.default_rng(4).uniform(0.8, 1.2, (16, SPACE.dim))
    r64 = BatchEvaluator(SHORT, SPACE, dtype=np.float64).evaluate_population(gains)
    r32 = BatchEvaluator(SHORT, SPACE, dtype=np.float32).evaluate_population(gains)
    assert r64.n_fell.sum() == 0 and r32.n_fell.sum() == 0
    np.testing.assert_allclose(r32.costs, r64.costs, rtol=1e-3)
