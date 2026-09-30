"""Search methods. Each takes an Evaluator and returns a TuningResult.

All searches are black-box: they only see the score for a set of gains, never
the physics inside. All three are seeded, so a run can be repeated exactly.

    random_search                 baseline: sample gains at random, keep the best
    differential_evolution_search evolves a population of gain sets toward lower cost
    bayesian_optimization_search  fits a surrogate model of the cost and uses it to
                                   pick promising gains to try next

To try a new method, write a function with the signature
`(evaluator: Evaluator, **kwargs) -> TuningResult` that scores gains by calling
`evaluator(unit_vector)`, and add it to __init__.py. All of the plumbing around
it (search space encoding, multi-scenario scoring, validation, reporting)
already works with any function of that shape.
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from .evaluation import Evaluator

# Optional: only needed for bayesian_optimization_search. If it's missing,
# every other function in this module still works; only that one raises a
# clear error, at call time, telling you how to install it.
try:
    from skopt import gp_minimize
    _SKOPT_IMPORT_ERROR = None
except ImportError as _e:
    gp_minimize = None
    _SKOPT_IMPORT_ERROR = _e


@dataclass
class TuningResult:
    method: str
    gains: np.ndarray                  # best gains found
    gain_names: List[str]
    best_cost: float                   # mean training cost of the best gains
    n_evals: int                       # number of times the objective was evaluated
    elapsed_s: float
    history: List[Tuple[int, float]] = field(default_factory=list)  # (evals so far, best cost so far)

    def gains_dict(self):
        return {n: float(g) for n, g in zip(self.gain_names, self.gains)}

    def best_cost_at(self, n_evals: int):
        """Best cost this search had found after at most n_evals evaluations.

        Returns (evals, cost) for the last history entry within the budget, or
        None if the history has no entry that early. Differential evolution
        only reports once per generation, so its evals can fall a little short
        of n_evals.
        """
        within = [(n, c) for n, c in self.history if n <= n_evals]
        return within[-1] if within else None


def random_search(evaluator: Evaluator, n_evals: int = 500, seed: int = 0) -> TuningResult:
    rng = np.random.default_rng(seed)
    dim = evaluator.space.dim
    start = time.time()
    best_unit, best_cost, history = None, np.inf, []
    for i in range(1, n_evals + 1):
        unit = rng.random(dim)
        c = evaluator(unit)
        if c < best_cost:
            best_unit, best_cost = unit, c
        history.append((i, best_cost))
    return TuningResult(
        method="random_search",
        gains=evaluator.space.decode(best_unit),
        gain_names=evaluator.space.names,
        best_cost=float(best_cost),
        n_evals=n_evals,
        elapsed_s=time.time() - start,
        history=history,
    )


def differential_evolution_search(
    evaluator: Evaluator,
    max_iter: int = 20,
    pop_multiplier: int = 8,
    seed: int = 0,
    workers: int = 1,
) -> TuningResult:
    """Differential evolution (scipy). Population size is pop_multiplier x number of gains.

    workers=-1 uses every CPU core. The objective is evaluated
    pop_size x (max_iter + 1) times.
    """
    history: List[Tuple[int, float]] = []

    def on_generation(intermediate_result):
        # nfev is scipy's own count of objective calls so far (scipy >= 1.12).
        history.append((int(intermediate_result.nfev), float(intermediate_result.fun)))

    start = time.time()
    result = differential_evolution(
        evaluator,
        bounds=evaluator.space.unit_bounds,
        maxiter=max_iter,
        popsize=pop_multiplier,
        tol=0.0,
        atol=0.0,
        polish=False,          # the objective is not smooth, so skip gradient polishing
        seed=seed,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
        callback=on_generation,
    )
    return TuningResult(
        method="differential_evolution",
        gains=evaluator.space.decode(result.x),
        gain_names=evaluator.space.names,
        best_cost=float(result.fun),
        n_evals=int(result.nfev),
        elapsed_s=time.time() - start,
        history=history,
    )


def bayesian_optimization_search(
    evaluator: Evaluator,
    n_calls: int = 100,
    n_initial_points: int = 15,
    seed: int = 0,
) -> TuningResult:
    """Bayesian optimization via scikit-optimize's gp_minimize.

    A Gaussian process models cost as a function of the gains, fit on every
    point tried so far. Each new candidate balances trying where the model
    predicts a low cost against trying where the model is most uncertain. This
    tends to need far fewer evaluations than random search or differential
    evolution to get close to the best gains, which matters because each
    evaluation here is itself several simulation runs.

    The first n_initial_points are sampled randomly (there's no model yet to
    guide the search), then every remaining call is chosen by the model. Needs
    scikit-optimize: pip install scikit-optimize.

    Not parallel: unlike differential_evolution_search, gp_minimize picks one
    point at a time based on all previous results, so there's no `workers`
    argument here.
    """
    if gp_minimize is None:
        raise ImportError(
            "bayesian_optimization_search needs scikit-optimize, which isn't installed.\n"
            "Install it with:\n\n    pip install scikit-optimize\n"
        ) from _SKOPT_IMPORT_ERROR

    history: List[Tuple[int, float]] = []

    def objective(unit) -> float:
        c = evaluator(unit)
        best_so_far = c if not history else min(history[-1][1], c)
        history.append((len(history) + 1, best_so_far))
        return c

    start = time.time()
    result = gp_minimize(
        objective,
        dimensions=evaluator.space.unit_bounds,
        n_calls=n_calls,
        n_initial_points=min(n_initial_points, n_calls),
        random_state=seed,
    )
    return TuningResult(
        method="bayesian_optimization",
        gains=evaluator.space.decode(result.x),
        gain_names=evaluator.space.names,
        best_cost=float(result.fun),
        n_evals=n_calls,
        elapsed_s=time.time() - start,
        history=history,
    )
