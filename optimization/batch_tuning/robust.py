"""Robust scoring: many noise seeds, randomized robots, and a penalty for chaotic gains.

The CPU tuner scores gains on 3 tilts x 3 noise seeds, all with the nominal
robot. That's cheap, but tuned gains can end up fitted to that exact robot
and those few noise sequences. With batched simulation it's affordable to ask
more of every candidate:

  1. Many seeds. Every start tilt is tried with every one of (in config.py)
     50 training seeds, so 150 scenarios per candidate instead of 9.

  2. Randomized robots (domain randomization). Every scenario gets its own
     robot, with each listed parameter scaled by a random factor in
     [1 - spread, 1 + spread] (`randomize`), or drawn outright from a range
     [low, high] (`randomize_range`: for parameters whose nominal value is 0,
     such as friction, which a scale factor can't move). Each scenario can
     also get its own latency, drawn from `latency_range` in whole physics
     steps. A scenario's robot is drawn from a generator seeded by
     (robot_seed, noise seed, tilt index), and its latency from one seeded by
     (robot_seed, noise seed, tilt index, 1), so:
       * every candidate faces exactly the same robots, which keeps scores
         repeatable (the optimizers rely on that);
       * the robot doesn't depend on batch order or chunking;
       * unseen validation seeds automatically bring unseen robots.

  3. A chaos (sensitivity) penalty. Some gains are chaotic: a change far
     below anything physical sends the run somewhere completely different,
     so their score is partly luck. Each scenario is run again with its start
     tilt nudged by `sensitivity_nudge` rad (same noise, same robot), and the
     score becomes

         objective = mean cost + sensitivity_weight * mean |cost - nudged cost|

     Well-behaved gains barely change under the nudge, so they're barely
     penalized. The twin runs double the simulations.
"""

from dataclasses import dataclass, fields, replace
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from batch_sim import BatchRobot

# BatchRobot fields that can be randomized: everything numeric except the
# gravity constant and the (integer) motor count.
RANDOMIZABLE = tuple(f.name for f in fields(BatchRobot) if f.name not in ("dc_motor", "n_motors", "gravity"))
ROBOT_PARAM_FIELDS = ("body_mass", "body_com_height", "body_inertia", "wheel_mass", "wheel_radius",
                      "wheel_inertia", "rotor_inertia", "max_torque", "wheel_friction", "coulomb_friction",
                      "coulomb_friction_speed", "gravity")
MOTOR_PARAM_FIELDS = ("gear_ratio", "torque_constant", "resistance", "supply_voltage", "current_limit", "n_motors",
                      "gear_efficiency")


@dataclass(frozen=True)
class RobustnessConfig:
    """The project's values live in config.py (RobustnessSettings); build this
    with CONFIG.build_robustness(). The defaults here are deliberately neutral
    (no randomization, no penalty) so there's only one place the real
    spreads and weights are written down.
    """

    train_seeds: Tuple[int, ...]                  # scored during tuning
    validation_seeds: Tuple[int, ...] = ()        # unseen seeds (and robots) for checking
    randomize: Tuple[Tuple[str, float], ...] = ()  # (parameter, relative spread)
    randomize_range: Tuple[Tuple[str, float, float], ...] = ()   # (parameter, low, high): drawn uniformly
    latency_range: Optional[Tuple[float, float]] = None          # s, (low, high); None: the configured latency
    robot_seed: int = 0
    sensitivity_weight: float = 0.0               # 0 turns the chaos penalty (and its twin runs) off
    sensitivity_nudge: float = 1e-4               # rad: far below anything physical, far above rounding
    # The other evaluation tiers (config.py RobustnessSettings): smoke must lie inside dev;
    # every other pair of sets must be disjoint, and dev, validation and fall-rate must avoid
    # the retired seeds.
    smoke_seeds: Tuple[int, ...] = ()
    dev_seeds: Tuple[int, ...] = ()
    fall_rate_seeds: Tuple[int, ...] = ()
    retired_seeds: Tuple[int, ...] = ()

    def __post_init__(self):
        for name, spread in self.randomize:
            if name not in RANDOMIZABLE:
                raise ValueError(f"can't randomize {name!r}; choose from {RANDOMIZABLE}")
            if not 0 <= spread < 1:
                raise ValueError(f"spread for {name} must be in [0, 1), got {spread}")
        for name, low, high in self.randomize_range:
            if name not in RANDOMIZABLE:
                raise ValueError(f"can't randomize {name!r}; choose from {RANDOMIZABLE}")
            if not low <= high:
                raise ValueError(f"range for {name} must have low <= high, got ({low}, {high})")
        both = {name for name, _ in self.randomize} & {name for name, _, _ in self.randomize_range}
        if both:
            raise ValueError(f"{sorted(both)} can't be in both randomize and randomize_range")
        if self.latency_range is not None and not 0 <= self.latency_range[0] <= self.latency_range[1]:
            raise ValueError(f"latency_range must have 0 <= low <= high, got {self.latency_range}")
        overlap = set(self.train_seeds) & set(self.validation_seeds)
        if overlap:
            # Validation seeds also pick the validation robots, so an overlap would score
            # "unseen" robots that training already saw (e.g. --seeds 1000 or more).
            raise ValueError(f"train_seeds and validation_seeds overlap: {sorted(overlap)[:5]}...")
        tiers = {"train": self.train_seeds, "validation": self.validation_seeds, "dev": self.dev_seeds,
                 "fall_rate": self.fall_rate_seeds}
        names = list(tiers)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = set(tiers[a]) & set(tiers[b])
                if both:
                    raise ValueError(f"{a} and {b} seeds overlap: {sorted(both)[:5]}...")
        if not set(self.smoke_seeds) <= set(self.dev_seeds):
            raise ValueError("smoke_seeds must be a subset of dev_seeds")
        for name in ("validation", "dev", "fall_rate"):
            reused = set(tiers[name]) & set(self.retired_seeds)
            if reused:
                raise ValueError(f"{name} seeds include retired seeds: {sorted(reused)[:5]}...")
        if self.sensitivity_weight < 0 or self.sensitivity_nudge <= 0:
            raise ValueError("sensitivity_weight must be >= 0 and sensitivity_nudge > 0")

    @property
    def twin_runs(self) -> bool:
        return self.sensitivity_weight > 0


def sample_robots(base: BatchRobot, config: RobustnessConfig, seeds: Sequence[int], n_pitches: int) -> BatchRobot:
    """One robot per scenario, in scenario order (seed-major, then tilt), as per-simulation arrays.

    Motor parameters are left alone when the robot uses the ideal actuator
    (they're unused), but still drawn, so the other parameters come out the
    same either way.
    """
    if not config.randomize and not config.randomize_range:
        return base
    names = [name for name, _ in config.randomize]
    spreads = np.array([spread for _, spread in config.randomize])
    ranged = [name for name, _, _ in config.randomize_range]
    lows = np.array([low for _, low, _ in config.randomize_range])
    highs = np.array([high for _, _, high in config.randomize_range])
    factors, values = [], []
    for seed in seeds:
        for i in range(n_pitches):
            # Scale factors first, then ranged values, from the same generator: adding
            # randomize_range leaves the scaled parameters exactly as they were.
            rng = np.random.default_rng([config.robot_seed, seed, i])
            factors.append(1.0 + spreads * rng.uniform(-1.0, 1.0, len(names)))
            values.append(rng.uniform(lows, highs) if ranged else np.empty(0))
    factors, values = np.array(factors), np.array(values)
    changes = {name: getattr(base, name) * factors[:, j] for j, name in enumerate(names)
               if getattr(base, name) is not None}
    changes.update({name: values[:, j] for j, name in enumerate(ranged) if getattr(base, name) is not None})
    return replace(base, **changes)


def sample_delay_steps(config: RobustnessConfig, seeds: Sequence[int], n_pitches: int,
                       dt: float) -> Optional[np.ndarray]:
    """Each scenario's latency in whole physics steps (scenario order as in sample_robots),
    or None when config.latency_range is None.

    Drawn uniformly from the whole numbers of steps in the range: those are exactly
    the latencies the simulators represent (a command takes effect at the first
    physics step at or after it is due).
    """
    if config.latency_range is None:
        return None
    low = math.ceil(config.latency_range[0] / dt - 1e-9)
    high = math.floor(config.latency_range[1] / dt + 1e-9)
    if high < low:
        raise ValueError(f"latency_range {config.latency_range} contains no whole number of {dt} s steps")
    return np.array([np.random.default_rng([config.robot_seed, seed, i, 1]).integers(low, high + 1)
                     for seed in seeds for i in range(n_pitches)], dtype=np.int64)


def robot_at(robot: BatchRobot, index: int):
    """(RobotParams, MotorParams or None) for one simulation of a BatchRobot, for the CPU code."""
    from balance_sim import MotorParams, RobotParams

    value = lambda name: float(np.asarray(getattr(robot, name))[index] if np.ndim(getattr(robot, name))
                               else getattr(robot, name))
    params = RobotParams(**{name: value(name) for name in ROBOT_PARAM_FIELDS})
    motor = None
    if robot.dc_motor:
        motor = MotorParams(**{name: (int(value(name)) if name == "n_motors" else value(name))
                               for name in MOTOR_PARAM_FIELDS})
    return params, motor
