"""Scoring many sets of gains in one batch.

BatchEvaluator is the batched counterpart of CPU tuning.Evaluator: it takes
the same EvalConfig and SearchSpace, runs the same scenarios (every start tilt
with every noise seed), and gives the same scores. The difference is that a
whole population of candidates x scenarios goes through simulate_batch
together.

Pass a RobustnessConfig (robust.py) to score more demandingly: many more
seeds, a randomized robot per scenario, and a penalty for chaotic gains.
Without one, scores match the CPU Evaluator.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from balance_sim import step_profile
from batch_sim import BatchRobot, SimSpec, build_schedule, noise_table, simulate_batch
from tuning import EvalConfig, EvalResult, SearchSpace

from .robust import RobustnessConfig, sample_robots


def spec_from_eval_config(config: EvalConfig) -> SimSpec:
    s = config.sensor
    return SimSpec(
        goal=step_profile(config.goal_steps),
        duration=config.duration,
        dt=config.dt,
        control_dt=config.control_dt,
        latency=config.latency,
        fall_angle=config.fall_angle,
        max_pitch_command=config.max_pitch_command,
        pitch_integral_limit=config.pitch_integral_limit,
        velocity_integral_limit=config.velocity_integral_limit,
        pitch_std=s.pitch_std,
        pitch_rate_std=s.pitch_rate_std,
        pitch_rate_bias=s.pitch_rate_bias,
        position_std=s.position_std,
        velocity_std=s.velocity_std,
        effort_weight=config.effort_weight,
        fall_penalty=config.fall_penalty,
        velocity_weight=config.velocity_weight,
        pitch_reference=config.pitch_reference,
    )


def decode_units(space: SearchSpace, units) -> np.ndarray:
    """Vectorized SearchSpace.decode: (P, dim) unit-cube points -> (P, dim) gains."""
    units = np.clip(np.asarray(units, dtype=float), 0.0, 1.0)
    out = np.empty_like(units)
    for i, p in enumerate(space.parameters):
        u = units[:, i]
        if p.log:
            out[:, i] = np.exp(np.log(p.low) + u * (np.log(p.high) - np.log(p.low)))
        else:
            out[:, i] = p.low + u * (p.high - p.low)
    return out


@dataclass
class BatchEvalResult:
    mean_cost: np.ndarray    # (P,) mean over scenarios
    costs: np.ndarray        # (P, n_scenarios), columns in the order of scenarios()
    n_fell: np.ndarray       # (P,)
    sensitivity: np.ndarray  # (P,) mean |cost - nudged-twin cost|; zeros without twin runs
    objective: np.ndarray    # (P,) what the optimizers minimize: mean_cost + weight * sensitivity


@dataclass(frozen=True)
class _ScenarioSet:
    pitch: np.ndarray        # (n_total,) start tilt per simulation column
    seed_col: np.ndarray     # (n_total,) column of the noise table
    robot: BatchRobot        # nominal, or (n_total,) arrays for per-scenario robots
    n_base: int              # the first n_base columns are the scenarios; any after are the nudged twins


class BatchEvaluator:
    """Objective for batched optimizers: many unit-cube vectors in, scores out.

    backend "numpy" runs batch_sim.simulate_batch on the CPU; "torch" runs
    batch_sim.torch_sim.TorchSimulator on `device` (a GPU by default).
    dtype is np.float64 or np.float32 for either backend. On the GPU, float32
    is the fast one; float64 is for checking against NumPy.

    robustness: None scores exactly like the CPU Evaluator. A RobustnessConfig
    switches to its seeds, randomized robots and chaos penalty (see robust.py).

    max_batch caps how many simulations run at once (candidates x scenarios);
    bigger populations are split into chunks of that size. Defaults: 12k for
    NumPy, where throughput falls off once the temporary arrays outgrow the
    CPU cache, and 200k for torch (see benchmark.py).
    """

    TORCH_BATCH_ROUNDING = 1024    # batch sizes are padded up to a multiple of this ...
    TORCH_CACHE_SIZE = 3           # ... so a few captured simulators cover most calls

    def __init__(self, config: EvalConfig, space: SearchSpace, dtype=np.float64,
                 max_batch: Optional[int] = None, backend: str = "numpy", device="cuda",
                 robustness: Optional[RobustnessConfig] = None):
        if backend not in ("numpy", "torch"):
            raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")
        self.config = config
        self.space = space
        self.dtype = dtype
        self.backend = backend
        self.device = device
        self.robustness = robustness
        self.max_batch = max_batch if max_batch is not None else (12_000 if backend == "numpy" else 200_000)
        self.spec = spec_from_eval_config(config)
        self.schedule = build_schedule(self.spec)
        self.robot = BatchRobot.from_params(config.params, config.motor)
        self._noise: Dict[Tuple[int, ...], np.ndarray] = {}
        self._scenario_sets: Dict[Tuple[int, ...], _ScenarioSet] = {}
        self._torch_sims: "OrderedDict[Tuple[int, Tuple[int, ...]], object]" = OrderedDict()
        self.n_sims = 0   # simulations run so far, for throughput reports

    @property
    def train_seeds(self) -> Tuple[int, ...]:
        return tuple(self.robustness.train_seeds if self.robustness else self.config.train_seeds)

    @property
    def validation_seeds(self) -> Optional[Tuple[int, ...]]:
        """The robustness config's validation seeds (None in plain mode, where they come from CPU config)."""
        return tuple(self.robustness.validation_seeds) if self.robustness else None

    def scenarios(self, seeds: Optional[Sequence[int]] = None) -> List[Tuple[float, int]]:
        """(start_pitch, seed) pairs, in the same order as Evaluator.scenarios (twin runs not listed)."""
        seeds = self.train_seeds if seeds is None else seeds
        return [(pitch, seed) for seed in seeds for pitch in self.config.start_pitches]

    def robots(self, seeds: Optional[Sequence[int]] = None) -> BatchRobot:
        """The robot for each scenario of scenarios(seeds): nominal, or per-scenario arrays."""
        return self._scenario_set(tuple(self.train_seeds if seeds is None else seeds)).robot.take(
            np.arange(len(self.scenarios(seeds))))

    def simulations_per_evaluation(self, seeds: Optional[Sequence[int]] = None) -> int:
        """Simulations it takes to score one candidate: scenarios, plus their twins in robust mode."""
        return len(self._scenario_set(tuple(self.train_seeds if seeds is None else seeds)).pitch)

    def noise_for(self, seeds: Sequence[int]) -> np.ndarray:
        """The sensor-noise table for these seeds, shape (ticks, 4, len(seeds)); built once, then cached.

        Column i holds exactly the draws SensorNoise(seed=seeds[i]) would make.
        Pass it to simulate_batch with seed_index choosing each simulation's column.
        """
        seeds = tuple(seeds)
        if seeds not in self._noise:
            self._noise[seeds] = noise_table(seeds, self.schedule.n_ticks)
        return self._noise[seeds]

    def _scenario_set(self, seeds: Tuple[int, ...]) -> _ScenarioSet:
        if seeds not in self._scenario_sets:
            scenarios = self.scenarios(seeds)
            pitch = np.array([p for p, _ in scenarios])
            seed_col = np.array([seeds.index(s) for _, s in scenarios])
            robot, n_base = self.robot, len(scenarios)
            r = self.robustness
            if r is not None:
                robot = sample_robots(self.robot, r, seeds, len(self.config.start_pitches))
                if r.twin_runs:
                    # Twins: same noise and robot, start tilt nudged.
                    pitch = np.concatenate([pitch, pitch + r.sensitivity_nudge])
                    seed_col = np.concatenate([seed_col, seed_col])
                    robot = robot.take(np.tile(np.arange(n_base), 2))
            self._scenario_sets[seeds] = _ScenarioSet(pitch, seed_col, robot, n_base)
        return self._scenario_sets[seeds]

    def _torch_simulator(self, batch_size: int, seeds: Tuple[int, ...], robot: BatchRobot):
        """A captured TorchSimulator for this (padded) batch size and seed set, reused across calls.

        robot only sets which fields are per-simulation; their values are
        copied in on every run.
        """
        key = (batch_size, seeds)
        if key in self._torch_sims:
            self._torch_sims.move_to_end(key)
            return self._torch_sims[key]
        import torch
        from batch_sim.torch_sim import TorchSimulator

        while len(self._torch_sims) >= self.TORCH_CACHE_SIZE:
            self._torch_sims.popitem(last=False)
            torch.cuda.empty_cache()
        sim = TorchSimulator(self.spec, robot, batch_size, noise=self.noise_for(seeds),
                             schedule=self.schedule, device=self.device,
                             dtype=torch.float32 if self.dtype == np.float32 else torch.float64)
        self._torch_sims[key] = sim
        return sim

    def _simulate_chunk(self, gains, start_pitch, seed_index, robot: BatchRobot, seeds: Tuple[int, ...]):
        """Run one chunk on the chosen backend; returns (cost, fell) arrays."""
        if self.backend == "numpy":
            result = simulate_batch(self.spec, robot, gains, start_pitch, noise=self.noise_for(seeds),
                                    seed_index=seed_index, schedule=self.schedule, dtype=self.dtype)
            return result.cost, result.fell
        n = len(gains)
        padded = -(-n // self.TORCH_BATCH_ROUNDING) * self.TORCH_BATCH_ROUNDING
        filler = np.concatenate([np.arange(n), np.zeros(padded - n, dtype=int)])   # extra rows copy row 0, ignored
        robot = robot.take(filler)
        sim = self._torch_simulator(padded, seeds, robot)
        result = sim.run(gains[filler], start_pitch[filler], seed_index[filler], robot=robot)
        return result.cost[:n], result.fell[:n]

    def evaluate_population(self, gains, seeds: Optional[Sequence[int]] = None) -> BatchEvalResult:
        """Score P sets of gains, shape (P, 5), on every scenario (and twin, in robust mode)."""
        gains = np.atleast_2d(np.asarray(gains, dtype=float))
        seeds = tuple(self.train_seeds if seeds is None else seeds)
        s = self._scenario_set(seeds)
        n_total, P = len(s.pitch), len(gains)

        # Row i of the flattened batch is candidate i // n_total on simulation column i % n_total.
        costs = np.empty(P * n_total)
        fell = np.empty(P * n_total, dtype=bool)
        per_chunk = max(1, self.max_batch // n_total) * n_total
        for start in range(0, P * n_total, per_chunk):
            rows = np.arange(start, min(start + per_chunk, P * n_total))
            col = rows % n_total
            costs[rows], fell[rows] = self._simulate_chunk(
                gains[rows // n_total], s.pitch[col], s.seed_col[col], s.robot.take(col), seeds)
        self.n_sims += P * n_total

        costs, fell = costs.reshape(P, n_total), fell.reshape(P, n_total)
        base = costs[:, :s.n_base]
        mean = base.mean(axis=1)
        if n_total > s.n_base:
            sensitivity = np.abs(base - costs[:, s.n_base:]).mean(axis=1)
            objective = mean + self.robustness.sensitivity_weight * sensitivity
        else:
            sensitivity, objective = np.zeros(P), mean
        return BatchEvalResult(mean, base, fell[:, :s.n_base].sum(axis=1), sensitivity, objective)

    def evaluate_gains(self, gains: Sequence[float], seeds: Optional[Sequence[int]] = None) -> EvalResult:
        """One set of gains, returned as the CPU Evaluator's EvalResult.

        Its mean_cost is the objective (the mean cost, plus the chaos penalty
        in robust mode), so it matches what the optimizers report as best_cost.
        Use evaluate_population for the separate parts.
        """
        r = self.evaluate_population([gains], seeds)
        return EvalResult(mean_cost=float(r.objective[0]), costs=tuple(float(c) for c in r.costs[0]),
                          n_fell=int(r.n_fell[0]))

    def evaluate_units(self, units) -> np.ndarray:
        """(P, dim) unit-cube points -> (P,) objective values on the training seeds."""
        return self.evaluate_population(decode_units(self.space, np.atleast_2d(units))).objective

    def __call__(self, x):
        """scipy convention: x of shape (dim,) -> float, or (dim, S) with vectorized=True -> (S,)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(self.evaluate_units(x[None, :])[0])
        return self.evaluate_units(x.T)
