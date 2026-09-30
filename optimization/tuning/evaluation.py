"""Scoring a set of gains: run the simulation under fixed conditions and average the cost.

Each candidate is scored on several scenarios and the costs are averaged. The
scenarios are every combination of starting tilt and sensor-noise seed, so
each seed is tried against each tilt. That stops the optimizer from finding
gains that only work for one lucky noise sequence or one starting condition.
Scenarios are fixed by seed, so the same gains always get the same score,
which the optimizer needs.

After tuning, score the winner again on seeds it never saw (validation) to check
it hasn't overfit.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from balance_sim import (
    CascadedPIDController,
    DCMotorActuator,
    MotorParams,
    Planar2D,
    RobotParams,
    SensorNoise,
    SimResult,
    cost,
    simulate,
    step_profile,
)

from .search_space import SearchSpace


@dataclass(frozen=True)
class EvalConfig:
    """Everything that defines the test the tuner scores against.

    No defaults: build it with CONFIG.build_eval_config() (config.py), and
    use dataclasses.replace for variations (the tests do this).
    """

    params: RobotParams
    motor: Optional[MotorParams]               # None = ideal actuator
    sensor: SensorNoise                        # seed is overridden per scenario
    latency: float                             # s
    duration: float                            # s per scenario
    dt: float                                  # physics step
    control_dt: float                          # controller period
    goal_steps: Tuple[Tuple[float, float], ...]   # (start time s, goal m/s)
    start_pitches: Tuple[float, ...]           # rad, each tried with every seed
    train_seeds: Tuple[int, ...]
    effort_weight: float
    fall_penalty: float
    velocity_weight: float
    pitch_reference: str                       # see balance_sim.pitch_error_signal
    fall_angle: float                          # rad; run ends past this pitch
    max_pitch_command: float                   # rad; cap on the outer loop's commanded lean
    pitch_integral_limit: float                # rad s; cap on the pitch loop's integral
    velocity_integral_limit: float             # m; cap on the velocity loop's integral


@dataclass
class EvalResult:
    mean_cost: float
    costs: Tuple[float, ...]     # per scenario, in the order of Evaluator.scenarios()
    n_fell: int

    @property
    def n_runs(self) -> int:
        return len(self.costs)


class Evaluator:
    """Callable objective: unit-cube vector in, mean cost out."""

    def __init__(self, config: EvalConfig, space: SearchSpace):
        self.config = config
        self.space = space

    def dynamics(self) -> Planar2D:
        c = self.config
        actuator = (DCMotorActuator(c.motor, max_torque=c.params.max_torque)
                    if c.motor is not None else None)
        return Planar2D(c.params, fall_angle=c.fall_angle, actuator=actuator)

    def controller(self, gains: Sequence[float]) -> CascadedPIDController:
        c = self.config
        return CascadedPIDController.from_gains(
            gains,
            torque_limit=c.params.max_torque,
            pitch_integral_limit=c.pitch_integral_limit,
            goal=step_profile(c.goal_steps),
            max_pitch_command=c.max_pitch_command,
            integral_limit=c.velocity_integral_limit,
        )

    def scenarios(self, seeds: Optional[Sequence[int]] = None) -> List[Tuple[float, int]]:
        """Every (start_pitch, seed) pair to score: each seed against each starting tilt."""
        c = self.config
        seeds = c.train_seeds if seeds is None else seeds
        return [(pitch, seed) for seed in seeds for pitch in c.start_pitches]

    def run(self, gains: Sequence[float], start_pitch: float, seed: int) -> SimResult:
        """Simulate one scenario. Also useful for plotting a specific run."""
        c = self.config
        dyn = self.dynamics()
        sensor = SensorNoise(
            pitch_std=c.sensor.pitch_std,
            pitch_rate_std=c.sensor.pitch_rate_std,
            pitch_rate_bias=c.sensor.pitch_rate_bias,
            position_std=c.sensor.position_std,
            velocity_std=c.sensor.velocity_std,
            seed=seed,
        )
        return simulate(
            dyn,
            self.controller(gains),
            duration=c.duration,
            dt=c.dt,
            initial_state=dyn.initial_state(start_pitch),
            control_dt=c.control_dt,
            sensor=sensor,
            latency=c.latency,
        )

    def score(self, result: SimResult) -> float:
        c = self.config
        return cost(result, c.effort_weight, c.fall_penalty, c.velocity_weight, c.pitch_reference)

    def evaluate_gains(self, gains: Sequence[float], seeds: Optional[Sequence[int]] = None) -> EvalResult:
        costs, n_fell = [], 0
        for start_pitch, seed in self.scenarios(seeds):
            result = self.run(gains, start_pitch, seed)
            costs.append(self.score(result))
            n_fell += int(result.fell)
        return EvalResult(mean_cost=float(np.mean(costs)), costs=tuple(costs), n_fell=n_fell)

    def __call__(self, unit: Sequence[float]) -> float:
        return self.evaluate_gains(self.space.decode(unit)).mean_cost
