"""Scoring pivot-robot gains: the reference tuner's Evaluator, for the pivot robot.

Same scenarios as the base tuner (every start tilt with every noise seed, from
config.py), same averaging, same EvalResult. It has the two things the
reference tuner's search methods use (`space` and calling it with a
unit-cube vector), so tuning.differential_evolution_search, random_search and
bayesian_optimization_search work on it unchanged.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from tuning import EvalResult, SearchSpace


class PivotEvaluator:
    """Callable objective: unit-cube vector in, mean cost out.

    config: a pivot_config.PivotProjectConfig. space: defaults to
    config.build_search_space() (7 gains: wheel loop then pivot loop).
    """

    def __init__(self, config, space: Optional[SearchSpace] = None):
        self.config = config
        self.space = space if space is not None else config.build_search_space()

    def scenarios(self, seeds: Optional[Sequence[int]] = None) -> List[Tuple[float, int]]:
        """Every (start_pitch, seed) pair: each seed against each starting tilt."""
        scenario = self.config.base.scenario
        seeds = scenario.train_seeds if seeds is None else seeds
        return [(pitch, seed) for seed in seeds for pitch in scenario.start_pitches_rad]

    def run(self, gains: Sequence[float], start_pitch: float, seed: int):
        """Simulate one scenario with the cascaded PID + pivot loop."""
        return self.config.run(self.config.build_controller(gains), start_pitch, seed)

    def evaluate_controller(self, make_controller: Callable[[], object],
                            seeds: Optional[Sequence[int]] = None) -> EvalResult:
        """Score any pivot controller (e.g. the LQR) on the same scenarios.
        make_controller() is called once per scenario."""
        costs, n_fell = [], 0
        for start_pitch, seed in self.scenarios(seeds):
            result = self.config.run(make_controller(), start_pitch, seed)
            costs.append(self.config.score(result))
            n_fell += int(result.fell)
        return EvalResult(mean_cost=float(np.mean(costs)), costs=tuple(costs), n_fell=n_fell)

    def evaluate_gains(self, gains: Sequence[float], seeds: Optional[Sequence[int]] = None) -> EvalResult:
        gains = [float(g) for g in gains]
        return self.evaluate_controller(lambda: self.config.build_controller(gains), seeds)

    def __call__(self, unit: Sequence[float]) -> float:
        return self.evaluate_gains(self.space.decode(unit)).mean_cost
