"""The LEGO robot's whole simulated pipeline at once, whatever ROBOT is set to.

The other equivalence tests run against CONFIG, which is the Pololu robot under a plain `pytest`, so the
LEGO robot's features (IMU and complementary filter, quantized encoder, slow PWM with 10 ms pulses locked to
the loop, pushes, gearbox and coupling backlash) were only checked together under ROBOT=legobot, and only one or
two at a time. Here every one is on, with robust scoring drawing the robot (its free play included), latency and
pitch offset per scenario (the pulse phase following the latency), and each scenario is checked against its own reference run (balance_sim), then the
torch engine against NumPy.
"""

from dataclasses import replace

import numpy as np
import pytest

from config import load_config
from batch_tuning import BatchEvaluator, pitch_bias_at, robot_at
from tuning import Evaluator, tick_phase

LEGO = load_config("legobot")
SPACE = LEGO.build_search_space()
# Short runs with a push each way inside them (the scenario's pushes come later than 2 s).
SHORT = replace(LEGO.build_eval_config(), duration=2.0, goal_steps=((0.3, 0.2), (1.2, -0.1)),
                start_pitches=(0.05, -0.15), train_seeds=(1, 2), pushes=((0.8, 0.05, 1.02), (1.5, 0.05, -1.02)))
ROBUST = replace(LEGO.build_robustness(), train_seeds=(1, 2, 3), validation_seeds=(50, 51, 52),
                 sensitivity_weight=0.0)
GAINS = np.array([LEGO.controller.hand_tuned_gains, LEGO.controller.hand_tuned_gains]) * [[1.0], [1.15]]


def test_every_lego_feature_is_on():
    assert SHORT.slow_pwm and SHORT.pwm_tick > 0 and SHORT.voltage_estimate is not None and SHORT.pwm_tick_locked
    assert SHORT.sensor.has_encoder and type(SHORT.sensor).__name__ == "IMUSensor"
    ranged = {name for name, _, _ in ROBUST.randomize_range}
    assert {"pitch_bias", "coulomb_friction", "gearbox_backlash", "coupling_backlash"} <= ranged
    assert "pwm_tick_phase" not in ranged
    assert SHORT.params.gearbox_backlash > 0 and SHORT.params.coupling_backlash > 0
    assert ROBUST.randomize and ROBUST.latency_range is not None


def test_robust_scores_match_reference_runs_scenario_by_scenario():
    """Each scenario's robot, pitch offset and latency, given to the reference simulator, gives the batched
    engine's cost to 1e-12; the batched engine's per-scenario pulse phase is the one the reference derives
    from that latency (locked ticks)."""
    ev = BatchEvaluator(SHORT, SPACE, robustness=ROBUST)
    result = ev.evaluate_population(GAINS)
    robots, delays = ev.robots(), ev.delay_steps()
    phases = np.broadcast_to(np.asarray(robots.pwm_tick_phase, dtype=float), (len(ev.scenarios()),))
    assert len(set(np.round(phases, 6))) > 1          # the phase follows each scenario's own latency
    assert len(set(delays)) > 1
    assert len(set(robots.gearbox_backlash)) == len(set(robots.coupling_backlash)) == len(ev.scenarios())
    for i, (pitch, seed) in enumerate(ev.scenarios()):
        params, motor = robot_at(robots, i)
        assert params.gearbox_backlash == robots.gearbox_backlash[i] and params.backlash > 0
        sensor = replace(SHORT.sensor, pitch_bias=pitch_bias_at(robots, i))
        config = replace(SHORT, params=params, motor=motor, sensor=sensor, latency=delays[i] * SHORT.dt)
        assert tick_phase(config) == phases[i]
        ref = Evaluator(config, SPACE)
        for k, g in enumerate(GAINS):
            assert result.costs[k, i] == pytest.approx(ref.score(ref.run(g, pitch, seed)), rel=1e-12)


def test_torch_float64_matches_numpy_in_robust_mode():
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = BatchEvaluator(SHORT, SPACE, robustness=ROBUST).evaluate_population(GAINS)
    got = BatchEvaluator(SHORT, SPACE, robustness=ROBUST, backend="torch", device=device,
                         max_batch=4).evaluate_population(GAINS)     # several chunks
    np.testing.assert_allclose(got.costs, ref.costs, rtol=1e-9)
    np.testing.assert_array_equal(got.n_fell, ref.n_fell)


def test_locked_phase_cannot_also_be_drawn():
    bad = replace(ROBUST, randomize_range=ROBUST.randomize_range + (("pwm_tick_phase", 0.0, 1.0),))
    with pytest.raises(ValueError, match="pwm_tick_locked"):
        BatchEvaluator(SHORT, SPACE, robustness=bad).evaluate_population(GAINS)
