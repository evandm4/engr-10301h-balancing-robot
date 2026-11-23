"""The batch engines' reporting counters (not in the cost): torque roughness, saturated ticks and
peak pitch, checked against the one-robot reference's own commands, and the GPU against NumPy."""

from dataclasses import replace

import numpy as np
import pytest

from batch_sim import BatchRobot, build_schedule, noise_table, simulate_batch
from batch_tuning import BatchEvaluator, spec_from_eval_config
from config import CONFIG
from tuning import Evaluator

CONFIG_SHORT = replace(CONFIG.build_eval_config(), duration=3.0, train_seeds=(1, 2))
GAINS = np.array(CONFIG.reference_pid_gains())


class Recorder:
    """Wraps a controller and keeps every command it issues."""

    def __init__(self, inner):
        self.inner, self.commands = inner, []

    def reset(self):
        self.inner.reset()
        self.commands = []

    def goal_velocity(self, t):
        return self.inner.goal_velocity(t)

    def pitch_reference(self):
        return self.inner.pitch_reference()

    def update(self, state, dt):
        u = self.inner.update(state, dt)
        self.commands.append(u)
        return u


def test_counters_match_the_reference_commands():
    ev = Evaluator(CONFIG_SHORT, CONFIG.build_search_space())
    scen = ev.scenarios()
    sim = spec_from_eval_config(CONFIG_SHORT)
    schedule = build_schedule(sim)
    seeds = tuple(CONFIG_SHORT.train_seeds)
    result = simulate_batch(sim, BatchRobot.from_params(CONFIG_SHORT.params, CONFIG_SHORT.motor),
                            np.tile(GAINS, (len(scen), 1)), [p for p, _ in scen],
                            noise=noise_table(seeds, schedule.n_ticks),
                            seed_index=[seeds.index(s) for _, s in scen], schedule=schedule)
    limit = CONFIG_SHORT.params.max_torque
    for i, (pitch, seed) in enumerate(scen):
        rec = Recorder(ev.controller(GAINS))
        run = ev.run_controller(rec, pitch, seed)
        assert not run.fell
        u = np.array(rec.commands)
        change = np.diff(np.concatenate([[0.0], u]))
        np.testing.assert_allclose(result.torque_variation[i], np.sum(np.abs(change)), rtol=1e-12)
        np.testing.assert_allclose(result.torque_rate_sq[i], np.sum(change**2), rtol=1e-12)
        assert result.saturated_ticks[i] == np.sum(np.abs(u) >= limit)
        assert result.alive_ticks[i] == len(u)
        assert result.peak_pitch[i] == np.max(np.abs(run.states[:-1, 2]))
    assert result.saturated_ticks.max() > 0          # the 0.30 rad start saturates


def test_evaluator_carries_the_counters_and_torch_agrees():
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robust = replace(CONFIG.build_robustness(), train_seeds=(1, 2))
    a = BatchEvaluator(CONFIG_SHORT, CONFIG.build_search_space(), robustness=robust).evaluate_population([GAINS])
    b = BatchEvaluator(CONFIG_SHORT, CONFIG.build_search_space(), robustness=robust, backend="torch",
                       device=device).evaluate_population([GAINS])
    for field in BatchEvaluator.REPORT_FIELDS:
        assert getattr(a, field).shape == a.costs.shape
        np.testing.assert_allclose(getattr(b, field), getattr(a, field), rtol=1e-9, err_msg=field)
