"""Search methods that hand the evaluator a whole population at once.

    random_search                 same samples as the CPU version, scored in big batches
    differential_evolution_search scipy DE with vectorized=True, updating="deferred"
    cma_es_search                 CMA-ES (the `cma` package), one batch per generation

All return the CPU code's TuningResult, so its reporting (save_result,
best_cost_at, ...) works unchanged. Bayesian optimization isn't here: it picks
one point at a time, so batching doesn't help it.
"""

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from tuning import TuningResult

from .evaluation import BatchEvaluator


def _result(method, evaluator, best_unit, best_cost, n_evals, start, history) -> TuningResult:
    return TuningResult(
        method=method,
        gains=evaluator.space.decode(best_unit),
        gain_names=evaluator.space.names,
        best_cost=float(best_cost),
        n_evals=int(n_evals),
        elapsed_s=time.time() - start,
        history=history,
    )


def random_search(evaluator: BatchEvaluator, n_evals: int = 500, seed: int = 0,
                  batch_size: int = 10_000) -> TuningResult:
    """Draws exactly the candidates CPU tuning.random_search draws for the same seed."""
    rng = np.random.default_rng(seed)
    start = time.time()
    units = rng.random((n_evals, evaluator.space.dim))
    costs = np.concatenate([evaluator.evaluate_units(units[i:i + batch_size])
                            for i in range(0, n_evals, batch_size)])
    best_so_far = np.minimum.accumulate(costs)
    best = int(np.argmin(costs))   # first occurrence, like the CPU loop's strict "<"
    history = [(i + 1, float(c)) for i, c in enumerate(best_so_far)]
    return _result("random_search", evaluator, units[best], costs[best], n_evals, start, history)


def differential_evolution_search(evaluator: BatchEvaluator, max_iter: int = 20, pop_multiplier: int = 8,
                                  seed: int = 0) -> TuningResult:
    """scipy DE, scoring each generation's whole population in one batch.

    Population size is pop_multiplier x number of gains, as in the CPU
    version. "deferred" updating is what the CPU version uses with
    workers != 1, so results match a CPU run with several workers.
    """
    history: List[Tuple[int, float]] = []
    n_evals = 0

    # With vectorized=True, scipy's nfev counts calls (one per generation), not
    # candidates, so count the candidates here.
    def objective(x):
        nonlocal n_evals
        n_evals += 1 if x.ndim == 1 else x.shape[1]
        return evaluator(x)

    def on_generation(intermediate_result):
        history.append((n_evals, float(intermediate_result.fun)))

    start = time.time()
    result = differential_evolution(
        objective,
        bounds=evaluator.space.unit_bounds,
        maxiter=max_iter,
        popsize=pop_multiplier,
        tol=0.0,
        atol=0.0,
        polish=False,
        seed=seed,
        vectorized=True,
        updating="deferred",
        callback=on_generation,
    )
    return _result("differential_evolution", evaluator, result.x, result.fun, n_evals, start, history)


def cma_es_search(evaluator: BatchEvaluator, max_iter: int = 50, popsize: Optional[int] = None,
                  sigma0: float = 0.25, x0: Optional[Sequence[float]] = None, seed: int = 0) -> TuningResult:
    """CMA-ES on the unit cube: adapts a Gaussian search distribution each generation.

    popsize defaults to cma's own choice (4 + 3 ln(dim), 8 for five gains);
    with a GPU it pays to raise it into the hundreds or thousands. x0 is a
    unit-cube start point (default: the middle of the search space).
    """
    import cma

    dim = evaluator.space.dim
    options = {"bounds": [0.0, 1.0], "maxiter": max_iter, "seed": seed + 1,   # cma treats seed 0 as "random"
               "verbose": -9, "tolx": 0, "tolfun": 0, "tolfunhist": 0, "tolstagnation": max_iter + 1}
    if popsize is not None:
        options["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(np.full(dim, 0.5) if x0 is None else list(x0), sigma0, options)

    start = time.time()
    best_unit, best_cost, n_evals, history = None, np.inf, 0, []
    while not es.stop():
        candidates = np.array(es.ask())
        costs = evaluator.evaluate_units(candidates)
        es.tell(list(candidates), costs.tolist())
        n_evals += len(candidates)
        i = int(np.argmin(costs))
        if costs[i] < best_cost:
            best_unit, best_cost = np.clip(candidates[i], 0.0, 1.0), float(costs[i])
        history.append((n_evals, best_cost))
    return _result("cma_es", evaluator, best_unit, best_cost, n_evals, start, history)
