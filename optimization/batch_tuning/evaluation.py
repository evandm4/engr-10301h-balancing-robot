"""Scoring many sets of gains in one batch.

BatchEvaluator is the batched counterpart of CPU tuning.Evaluator: it takes
the same EvalConfig and SearchSpace, runs the same scenarios (every start tilt
with every noise seed), and gives the same scores. The difference is that a
whole population of candidates x scenarios goes through simulate_batch
together.

Pass a RobustnessConfig (robust.py) to score more demandingly: many more
seeds, a randomized robot per scenario, and a penalty for chaotic gains.
Without one, scores match the CPU Evaluator.

By default the candidates are cascaded-PID gains. controller="lqr" runs
balance_sim's LQRController instead, and to_gains turns each point of the
search space into the gains the simulator runs: for example LQR weights into
LQR gains (tuning.LQRWeightDesigner). controller="policy" runs a learned policy
(balance_sim.policy): pass its PolicySpec as `policy`, and score flat parameter vectors with
evaluate_policies. Each network's weights go to the simulator once, not once per simulation,
and they never pass through the unit cube (decode_units clips to [0, 1], which is right for
bounded PID gains and wrong for signed, unbounded weights).
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from balance_sim import push_profile, step_profile
from batch_sim import BatchRobot, SimSpec, build_schedule, noise_table, simulate_batch
from tuning import EvalConfig, EvalResult, SearchSpace

from .robust import RobustnessConfig, sample_delay_steps, sample_robots


def spec_from_eval_config(config: EvalConfig, controller: str = "cascaded", policy=None) -> SimSpec:
    s = config.sensor
    return SimSpec(
        controller=controller,
        policy=policy,
        goal=step_profile(config.goal_steps, config.goal_transition),
        disturbance=push_profile(config.pushes) if config.pushes else None,
        duration=config.duration,
        dt=config.dt,
        control_dt=config.control_dt,
        latency=config.latency,
        fall_angle=config.fall_angle,
        max_pitch_command=config.max_pitch_command,
        pitch_integral_limit=config.pitch_integral_limit,
        velocity_integral_limit=config.velocity_integral_limit,
        pitch_std=s.pitch_std,
        pitch_bias=s.pitch_bias,
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
    # The cost's parts per scenario, (P, n_scenarios), unweighted as in BatchResult:
    # integrals of pitch error^2, torque^2 and velocity error^2. Optional so
    # results built elsewhere needn't carry them.
    pitch_error: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None
    velocity_error: Optional[np.ndarray] = None
    # Per scenario, (P, n_scenarios): whether it fell, and when (s; NaN if it stayed up).
    fell: Optional[np.ndarray] = None
    fall_time: Optional[np.ndarray] = None
    # Not in the cost, for reporting (batch_sim BatchResult): torque roughness, saturation, peak pitch.
    torque_variation: Optional[np.ndarray] = None
    torque_rate_sq: Optional[np.ndarray] = None
    saturated_ticks: Optional[np.ndarray] = None
    alive_ticks: Optional[np.ndarray] = None
    peak_pitch: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _ScenarioSet:
    pitch: np.ndarray        # (n_total,) start tilt per simulation column
    seed_col: np.ndarray     # (n_total,) column of the noise table
    robot: BatchRobot        # nominal, or (n_total,) arrays for per-scenario robots
    n_base: int              # the first n_base columns are the scenarios; any after are the nudged twins
    delay_steps: Optional[np.ndarray] = None   # (n_total,) latency per column in physics steps; None: spec.latency


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

    controller: "cascaded" (the default) or "lqr", the batch_sim controller
    the gains are for. to_gains: maps decoded search-space points (P, dim) to
    those gains (P, n); None means the points are the gains already, as for
    the cascaded PID's search space. evaluate_population always takes gains.
    """

    TORCH_BATCH_ROUNDING = 1024    # batch sizes are padded up to a multiple of this ...
    TORCH_CACHE_SIZE = 3           # ... so a few captured simulators cover most calls

    def __init__(self, config: EvalConfig, space: SearchSpace, dtype=np.float64,
                 max_batch: Optional[int] = None, backend: str = "numpy", device="cuda",
                 robustness: Optional[RobustnessConfig] = None, controller: str = "cascaded",
                 to_gains: Optional[Callable[[np.ndarray], np.ndarray]] = None, policy=None):
        if backend not in ("numpy", "torch"):
            raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")
        self.config = config
        self.space = space
        self.dtype = dtype
        self.backend = backend
        self.device = device
        self.robustness = robustness
        self.max_batch = max_batch if max_batch is not None else (12_000 if backend == "numpy" else 200_000)
        self.to_gains = to_gains
        self.policy = policy
        self.spec = spec_from_eval_config(config, controller, policy)
        self.schedule = build_schedule(self.spec)
        self.robot = BatchRobot.from_params(config.params, config.motor)
        self._noise: Dict[Tuple[int, ...], np.ndarray] = {}
        self._scenario_sets: Dict[Tuple[int, ...], _ScenarioSet] = {}
        self._torch_sims: "OrderedDict[Tuple[int, Tuple[int, ...], int], object]" = OrderedDict()
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
            robot, n_base, delay = self.robot, len(scenarios), None
            r = self.robustness
            if r is not None:
                n_pitches = len(self.config.start_pitches)
                robot = sample_robots(self.robot, r, seeds, n_pitches)
                delay = sample_delay_steps(r, seeds, n_pitches, self.spec.dt)
                if r.twin_runs:
                    # Twins: same noise, robot and latency, start tilt nudged.
                    pitch = np.concatenate([pitch, pitch + r.sensitivity_nudge])
                    seed_col = np.concatenate([seed_col, seed_col])
                    robot = robot.take(np.tile(np.arange(n_base), 2))
                    delay = None if delay is None else np.tile(delay, 2)
            self._scenario_sets[seeds] = _ScenarioSet(pitch, seed_col, robot, n_base, delay)
        return self._scenario_sets[seeds]

    def delay_steps(self, seeds: Optional[Sequence[int]] = None) -> Optional[np.ndarray]:
        """Each scenario's latency in physics steps (order of scenarios(seeds)); None if all use the configured one."""
        s = self._scenario_set(tuple(self.train_seeds if seeds is None else seeds))
        return None if s.delay_steps is None else s.delay_steps[:s.n_base]

    def _torch_simulator(self, batch_size: int, seeds: Tuple[int, ...], robot: BatchRobot, n_gains: int,
                         max_delay_steps: Optional[int] = None, scenarios_per_candidate: Optional[int] = None):
        """A captured TorchSimulator for this (padded) batch size, seed set and gain width, reused across calls.

        robot only sets which fields are per-simulation; their values are
        copied in on every run.
        """
        key = (batch_size, seeds, n_gains, max_delay_steps, scenarios_per_candidate)
        if key in self._torch_sims:
            self._torch_sims.move_to_end(key)
            return self._torch_sims[key]
        import torch
        from batch_sim.torch_sim import TorchSimulator

        while len(self._torch_sims) >= self.TORCH_CACHE_SIZE:
            self._torch_sims.popitem(last=False)
            torch.cuda.empty_cache()
        sim = TorchSimulator(self.spec, robot, batch_size, noise=self.noise_for(seeds),
                             schedule=self.schedule, device=self.device, n_gains=n_gains,
                             max_delay_steps=max_delay_steps, scenarios_per_candidate=scenarios_per_candidate,
                             dtype=torch.float32 if self.dtype == np.float32 else torch.float64)
        self._torch_sims[key] = sim
        return sim

    CHUNK_FIELDS = ("cost", "fell", "pitch_error", "effort", "velocity_error", "fall_time",
                    "torque_variation", "torque_rate_sq", "saturated_ticks", "alive_ticks", "peak_pitch")
    REPORT_FIELDS = CHUNK_FIELDS[2:]   # per-scenario arrays BatchEvalResult carries besides the cost

    def _simulate_chunk(self, gains, start_pitch, seed_index, robot: BatchRobot, seeds: Tuple[int, ...],
                        delay_steps=None, max_delay_steps=None):
        """Run one chunk on the chosen backend; returns the CHUNK_FIELDS arrays of its BatchResult."""
        if self.backend == "numpy":
            result = simulate_batch(self.spec, robot, gains, start_pitch, noise=self.noise_for(seeds),
                                    seed_index=seed_index, schedule=self.schedule, dtype=self.dtype,
                                    delay_steps=delay_steps)
            return tuple(getattr(result, f) for f in self.CHUNK_FIELDS)
        n = len(gains)
        padded = -(-n // self.TORCH_BATCH_ROUNDING) * self.TORCH_BATCH_ROUNDING
        filler = np.concatenate([np.arange(n), np.zeros(padded - n, dtype=int)])   # extra rows copy row 0, ignored
        robot = robot.take(filler)
        sim = self._torch_simulator(padded, seeds, robot, gains.shape[1], max_delay_steps)
        result = sim.run(gains[filler], start_pitch[filler], seed_index[filler], robot=robot,
                         delay_steps=None if delay_steps is None else delay_steps[filler])
        return tuple(getattr(result, f)[:n] for f in self.CHUNK_FIELDS)

    def evaluate_population(self, gains, seeds: Optional[Sequence[int]] = None) -> BatchEvalResult:
        """Score P sets of gains on every scenario (and twin, in robust mode).

        Gains as the simulator takes them: (P, 4) or (P, 5) cascaded-PID gains,
        or (P, 4 + m) LQR gains for controller="lqr". For controller="policy" this is
        evaluate_policies.
        """
        if self.spec.controller == "policy":
            return self.evaluate_policies(gains, seeds)
        gains = np.atleast_2d(np.asarray(gains, dtype=float))
        seeds = tuple(self.train_seeds if seeds is None else seeds)
        s = self._scenario_set(seeds)
        n_total, P = len(s.pitch), len(gains)

        # Row i of the flattened batch is candidate i // n_total on simulation column i % n_total.
        out = {f: np.empty(P * n_total, dtype=bool if f == "fell" else float) for f in self.CHUNK_FIELDS}
        per_chunk = max(1, self.max_batch // n_total) * n_total
        max_delay = None if s.delay_steps is None else int(s.delay_steps.max())
        for start in range(0, P * n_total, per_chunk):
            rows = np.arange(start, min(start + per_chunk, P * n_total))
            col = rows % n_total
            chunk = self._simulate_chunk(
                gains[rows // n_total], s.pitch[col], s.seed_col[col], s.robot.take(col), seeds,
                None if s.delay_steps is None else s.delay_steps[col], max_delay)
            for f, values in zip(self.CHUNK_FIELDS, chunk):
                out[f][rows] = values
        self.n_sims += P * n_total
        return self._aggregate(out, s, P)

    def _aggregate(self, out, s: _ScenarioSet, P: int) -> BatchEvalResult:
        n_total = len(s.pitch)
        out = {f: v.reshape(P, n_total) for f, v in out.items()}
        costs = out["cost"]
        base = costs[:, :s.n_base]
        mean = base.mean(axis=1)
        if n_total > s.n_base:
            sensitivity = np.abs(base - costs[:, s.n_base:]).mean(axis=1)
            objective = mean + self.robustness.sensitivity_weight * sensitivity
        else:
            sensitivity, objective = np.zeros(P), mean
        parts = {f: out[f][:, :s.n_base] for f in ("fell",) + self.REPORT_FIELDS}
        return BatchEvalResult(mean, base, out["fell"][:, :s.n_base].sum(axis=1), sensitivity, objective, **parts)

    def candidates_per_chunk(self, seeds: Optional[Sequence[int]] = None) -> int:
        """How many policies one chunk of max_batch simulations holds (each needs all its scenarios)."""
        return max(1, self.max_batch // self.simulations_per_evaluation(seeds))

    def evaluate_policies(self, params, seeds: Optional[Sequence[int]] = None) -> BatchEvalResult:
        """Score P parameter vectors (P, n_params) of the policy this evaluator was built for."""
        if self.spec.controller != "policy":
            raise ValueError("evaluate_policies needs an evaluator built with controller='policy'")
        params = np.atleast_2d(np.asarray(params, dtype=float))
        if params.shape[1] != self.policy.n_params:
            raise ValueError(f"expected {self.policy.n_params} parameters per policy, got {params.shape[1]}")
        seeds = tuple(self.train_seeds if seeds is None else seeds)
        s = self._scenario_set(seeds)
        n_total, P = len(s.pitch), len(params)
        out = {f: np.empty(P * n_total, dtype=bool if f == "fell" else float) for f in self.CHUNK_FIELDS}
        per_chunk = self.candidates_per_chunk(seeds)
        max_delay = None if s.delay_steps is None else int(s.delay_steps.max())
        for start in range(0, P, per_chunk):
            cands = np.arange(start, min(start + per_chunk, P))
            col = np.tile(np.arange(n_total), len(cands))
            chunk = self._simulate_policy_chunk(
                params[cands], s.pitch[col], s.seed_col[col], s.robot.take(col), seeds, n_total,
                None if s.delay_steps is None else s.delay_steps[col], max_delay, per_chunk)
            rows = slice(start * n_total, (start + len(cands)) * n_total)
            for f, values in zip(self.CHUNK_FIELDS, chunk):
                out[f][rows] = values
        self.n_sims += P * n_total
        return self._aggregate(out, s, P)

    def _simulate_policy_chunk(self, params, start_pitch, seed_index, robot: BatchRobot, seeds, n_total: int,
                               delay_steps=None, max_delay_steps=None, chunk_candidates: int = 1):
        """One chunk of whole candidates (each with all n_total of its simulations, in order)."""
        if self.backend == "numpy":
            result = simulate_batch(self.spec, robot, params, start_pitch, noise=self.noise_for(seeds),
                                    seed_index=seed_index, schedule=self.schedule, dtype=self.dtype,
                                    delay_steps=delay_steps, scenarios_per_candidate=n_total)
            return tuple(getattr(result, f) for f in self.CHUNK_FIELDS)
        # Torch: pad with whole dummy candidates (copies of candidate 0, discarded) to a fixed
        # count, so one captured graph serves every chunk of that size. A population smaller
        # than half a chunk is padded to a multiple of 8 candidates instead.
        n = len(params)
        padded = chunk_candidates if n > chunk_candidates // 2 else -(-n // 8) * 8
        fill = np.concatenate([np.arange(n), np.zeros(padded - n, dtype=int)])
        rows = (fill[:, None] * n_total + np.arange(n_total)[None, :]).ravel()
        robot = robot.take(rows)
        sim = self._torch_simulator(padded * n_total, seeds, robot, params.shape[1], max_delay_steps,
                                    scenarios_per_candidate=n_total)
        result = sim.run(params[fill], start_pitch[rows], seed_index[rows], robot=robot,
                         delay_steps=None if delay_steps is None else delay_steps[rows])
        return tuple(getattr(result, f)[:n * n_total] for f in self.CHUNK_FIELDS)

    def evaluate_gains(self, gains: Sequence[float], seeds: Optional[Sequence[int]] = None) -> EvalResult:
        """One set of gains, returned as the CPU Evaluator's EvalResult.

        Its mean_cost is the objective (the mean cost, plus the chaos penalty
        in robust mode), so it matches what the optimizers report as best_cost.
        Use evaluate_population for the separate parts.
        """
        r = self.evaluate_population([gains], seeds)
        return EvalResult(mean_cost=float(r.objective[0]), costs=tuple(float(c) for c in r.costs[0]),
                          n_fell=int(r.n_fell[0]))

    def gains_for(self, points) -> np.ndarray:
        """Decoded search-space points (P, dim) -> the gains the simulator runs (P, n)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return points if self.to_gains is None else np.atleast_2d(self.to_gains(points))

    def evaluate_units(self, units) -> np.ndarray:
        """(P, dim) unit-cube points -> (P,) objective values on the training seeds."""
        return self.evaluate_population(self.gains_for(decode_units(self.space, np.atleast_2d(units)))).objective

    def __call__(self, x):
        """scipy convention: x of shape (dim,) -> float, or (dim, S) with vectorized=True -> (S,)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(self.evaluate_units(x[None, :])[0])
        return self.evaluate_units(x.T)
