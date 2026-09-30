"""Robust scoring: many noise seeds, randomized robots, and a penalty for chaotic gains.

The CPU tuner scores gains on 3 tilts x 3 noise seeds, all with the nominal
robot. That's cheap, but tuned gains can end up fitted to that exact robot
and those few noise sequences. With batched simulation it's affordable to ask
more of every candidate:

  1. Many seeds. Every start tilt is tried with every one of (in config.py)
     50 training seeds, so 150 scenarios per candidate instead of 9.

  2. Randomized robots (domain randomization). Every scenario gets its own
     robot, with each listed parameter scaled by a random factor in
     [1 - spread, 1 + spread]. A scenario's robot is drawn from a generator
     seeded by (robot_seed, noise seed, tilt index), so:
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
from typing import Sequence, Tuple

import numpy as np

from batch_sim import BatchRobot

# BatchRobot fields that can be randomized: everything numeric except the
# gravity constant and the (integer) motor count.
RANDOMIZABLE = tuple(f.name for f in fields(BatchRobot) if f.name not in ("dc_motor", "n_motors", "gravity"))
ROBOT_PARAM_FIELDS = ("body_mass", "body_com_height", "body_inertia", "wheel_mass", "wheel_radius",
                      "wheel_inertia", "max_torque", "wheel_friction", "gravity")
MOTOR_PARAM_FIELDS = ("gear_ratio", "torque_constant", "resistance", "supply_voltage", "current_limit", "n_motors")


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
    robot_seed: int = 0
    sensitivity_weight: float = 0.0               # 0 turns the chaos penalty (and its twin runs) off
    sensitivity_nudge: float = 1e-4               # rad: far below anything physical, far above rounding

    def __post_init__(self):
        for name, spread in self.randomize:
            if name not in RANDOMIZABLE:
                raise ValueError(f"can't randomize {name!r}; choose from {RANDOMIZABLE}")
            if not 0 <= spread < 1:
                raise ValueError(f"spread for {name} must be in [0, 1), got {spread}")
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
    if not config.randomize:
        return base
    names = [name for name, _ in config.randomize]
    spreads = np.array([spread for _, spread in config.randomize])
    factors = np.array([
        1.0 + spreads * np.random.default_rng([config.robot_seed, seed, i]).uniform(-1.0, 1.0, len(names))
        for seed in seeds for i in range(n_pitches)
    ])
    return replace(base, **{name: getattr(base, name) * factors[:, j] for j, name in enumerate(names)
                            if getattr(base, name) is not None})


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
