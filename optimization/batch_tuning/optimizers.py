"""Search methods that hand the evaluator a whole population at once.

    random_search                 same samples as the CPU version, scored in big batches
    differential_evolution_search scipy DE with vectorized=True, updating="deferred"
    cma_es_search                 CMA-ES (the `cma` package), one batch per generation
    cma_es_policy_search          CMA-ES over a learned policy's weights (unbounded; own result type)
    PolicySearch                  the same, one generation at a time, saveable and resumable

All return the CPU code's TuningResult, so its reporting (save_result,
best_cost_at, ...) works unchanged. Bayesian optimization isn't here: it picks
one point at a time, so batching doesn't help it.
"""

import time
from dataclasses import dataclass
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


@dataclass
class PolicySearchResult:
    """What cma_es_policy_search found. Parameters are flat policy vectors (balance_sim.policy)."""

    best_params: np.ndarray      # best candidate ever scored (the start point counts)
    best_cost: float             # its training objective
    mean_params: np.ndarray      # CMA's distribution mean at the end (often generalizes better)
    n_evals: int                 # candidates scored, the start point included
    robot_seconds: float         # simulated time charged: candidates x simulations each x run length
    elapsed_s: float
    history: List[Tuple[int, float]]   # (n_evals, best objective so far), once per generation
    method: str


class PolicySearch:
    """CMA-ES over a policy's weights, one generation at a time, with state that can be saved
    and restored (for checkpoints and resumed runs). cma_es_policy_search runs it to the end.

    Unbounded: no unit cube (see BatchEvaluator.evaluate_policies).
    x0: start point (n_params,). It is scored first (start()) and kept as the best candidate
    unless something beats it, so a warm start can't end worse than it began.
    stds: optional per-parameter initial standard deviations, as multiples of sigma0
    (cma's CMA_stds), for example small on a warm-started linear path and larger on the MLP.
    Full covariance up to full_cma_max_params parameters, diagonal (sep-CMA) above.

    `cma` draws its samples from NumPy's global generator, so state() saves that generator's
    state with the ES, and restore() puts both back: a resumed run continues exactly.
    """

    def __init__(self, evaluator: BatchEvaluator, x0, sigma0: float, popsize: int, max_iter: int = 10**9,
                 seed: int = 0, stds: Optional[Sequence[float]] = None, full_cma_max_params: int = 600):
        import cma

        x0 = np.asarray(x0, dtype=float)
        self.evaluator = evaluator
        self.popsize = popsize
        self.diagonal = x0.size > full_cma_max_params
        options = {"maxiter": max_iter, "seed": seed + 1, "popsize": popsize, "verbose": -9,
                   "tolx": 0, "tolfun": 0, "tolfunhist": 0, "tolstagnation": max_iter + 1,
                   "CMA_diagonal": self.diagonal}
        if stds is not None:
            options["CMA_stds"] = list(stds)
        self.es = cma.CMAEvolutionStrategy(list(x0), sigma0, options)
        self.x0 = x0.copy()
        self.best_params, self.best_cost = x0.copy(), np.inf
        self.n_evals = 0
        self.generation = 0
        self.history: List[Tuple[int, float]] = []
        self.elapsed_s = 0.0

    @property
    def method(self) -> str:
        return "sep_cma_es" if self.diagonal else "cma_es"

    @property
    def per_eval(self) -> float:
        """Robot-seconds charged per candidate: its simulations x the run length."""
        return self.evaluator.simulations_per_evaluation() * self.evaluator.config.duration

    @property
    def robot_seconds(self) -> float:
        return self.n_evals * self.per_eval

    @property
    def mean_params(self) -> np.ndarray:
        return np.array(self.es.result.xfavorite)

    @property
    def sigma(self) -> float:
        return float(self.es.sigma)

    def start(self) -> float:
        """Score x0 (once). Returns its cost."""
        if self.n_evals == 0:
            t = time.time()
            self.best_cost = float(self.evaluator.evaluate_policies(self.x0[None, :]).objective[0])
            self.n_evals, self.history = 1, [(1, self.best_cost)]
            self.elapsed_s += time.time() - t
        return self.best_cost

    def stopped(self) -> bool:
        return bool(self.es.stop())

    def step(self) -> np.ndarray:
        """One generation: ask, score, tell. Returns the generation's costs."""
        self.start()
        t = time.time()
        candidates = np.array(self.es.ask())
        costs = self.evaluator.evaluate_policies(candidates).objective
        self.es.tell(list(candidates), costs.tolist())
        self.n_evals += len(candidates)
        self.generation += 1
        i = int(np.argmin(costs))
        if costs[i] < self.best_cost:
            self.best_params, self.best_cost = candidates[i].copy(), float(costs[i])
        self.history.append((self.n_evals, self.best_cost))
        self.elapsed_s += time.time() - t
        return costs

    def state(self) -> dict:
        """Everything needed to continue: pickle-able (the ES object pickles)."""
        return {"es": self.es, "numpy_rng": np.random.get_state(), "x0": self.x0, "popsize": self.popsize,
                "diagonal": self.diagonal, "best_params": self.best_params, "best_cost": self.best_cost,
                "n_evals": self.n_evals, "generation": self.generation, "history": list(self.history),
                "elapsed_s": self.elapsed_s}

    @classmethod
    def restore(cls, evaluator: BatchEvaluator, state: dict) -> "PolicySearch":
        self = cls.__new__(cls)
        self.evaluator = evaluator
        for key in ("es", "x0", "popsize", "diagonal", "best_params", "best_cost", "n_evals", "generation",
                    "history", "elapsed_s"):
            setattr(self, key, state[key])
        np.random.set_state(state["numpy_rng"])
        return self

    def result(self) -> PolicySearchResult:
        return PolicySearchResult(self.best_params, self.best_cost, self.mean_params, self.n_evals,
                                  self.robot_seconds, self.elapsed_s, list(self.history), self.method)


def cma_es_policy_search(evaluator: BatchEvaluator, x0, sigma0: float, popsize: int, max_iter: int,
                         seed: int = 0, stds: Optional[Sequence[float]] = None,
                         full_cma_max_params: int = 600,
                         max_robot_seconds: Optional[float] = None) -> PolicySearchResult:
    """PolicySearch run to the end: max_iter generations, or until the next generation would
    take it past max_robot_seconds of simulation."""
    search = PolicySearch(evaluator, x0, sigma0, popsize, max_iter, seed, stds, full_cma_max_params)
    search.start()
    while not search.stopped():
        if max_robot_seconds is not None and (search.n_evals + popsize) * search.per_eval > max_robot_seconds:
            break
        search.step()
    return search.result()


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
