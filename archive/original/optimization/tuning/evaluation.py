"""Scoring a set of gains: run the simulation under fixed conditions and average the cost.

Each candidate is scored on several scenarios (different starting tilts and
sensor-noise seeds) and the costs are averaged. That stops the optimizer from
finding gains that only work for one lucky noise sequence. Scenarios are fixed
by seed, so the same gains always get the same score, which the optimizer needs.

After tuning, score the winner again on seeds it never saw (validation) to check
it hasn't overfit.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

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
    """Everything that defines the test the tuner scores against."""

    params: RobotParams = field(default_factory=RobotParams)
    motor: Optional[MotorParams] = field(default_factory=MotorParams)   # None = ideal actuator
    sensor: SensorNoise = field(default_factory=SensorNoise.typical)    # seed is overridden per scenario
    latency: float = 0.010                     # s
    duration: float = 6.0                      # s per scenario
    dt: float = 0.0025                         # physics step; coarser than the 1 ms default for speed
    control_dt: float = 0.005                  # controller period (200 Hz)
    goal_steps: Tuple[Tuple[float, float], ...] = ((0.5, 0.3), (3.5, -0.2))  # (start time s, m/s)
    start_pitches: Tuple[float, ...] = (0.1, -0.08, 0.05)                    # rad, cycled per scenario
    train_seeds: Tuple[int, ...] = (1, 2, 3)
    effort_weight: float = 0.01
    fall_penalty: float = 100.0
    velocity_weight: float = 1.0
    fall_angle: float = np.pi / 3               # rad; run ends past this pitch
    max_pitch_command: float = 0.2              # rad; cap on the outer loop's commanded lean


@dataclass
class EvalResult:
    mean_cost: float
    costs: Tuple[float, ...]     # per scenario
    n_fell: int


class Evaluator:
    """Callable objective: unit-cube vector in, mean cost out."""

    def __init__(self, config: EvalConfig, space: SearchSpace):
        self.config = config
        self.space = space

    def dynamics(self) -> Planar2D:
        c = self.config
        actuator = DCMotorActuator(c.motor) if c.motor is not None else None
        return Planar2D(c.params, fall_angle=c.fall_angle, actuator=actuator)

    def controller(self, gains: Sequence[float]) -> CascadedPIDController:
        c = self.config
        return CascadedPIDController.from_gains(
            gains, goal=step_profile(c.goal_steps), max_pitch_command=c.max_pitch_command
        )

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

    def evaluate_gains(self, gains: Sequence[float], seeds: Optional[Sequence[int]] = None) -> EvalResult:
        c = self.config
        seeds = c.train_seeds if seeds is None else seeds
        costs, n_fell = [], 0
        for i, seed in enumerate(seeds):
            result = self.run(gains, c.start_pitches[i % len(c.start_pitches)], seed)
            costs.append(cost(result, c.effort_weight, c.fall_penalty, c.velocity_weight))
            n_fell += int(result.fell)
        return EvalResult(mean_cost=float(np.mean(costs)), costs=tuple(costs), n_fell=n_fell)

    def __call__(self, unit: Sequence[float]) -> float:
        return self.evaluate_gains(self.space.decode(unit)).mean_cost
