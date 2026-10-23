"""What the controller sees: the base robot's IMU and encoders, plus the pivot angle."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from balance_sim import SensorNoise


@dataclass
class PivotSensor:
    """base: noise on the first four states (the lower body's IMU and the wheel
    encoders), exactly as for the base robot. With the same seed, those four
    channels get the same noise as a base-robot run.

    The pivot angle and rate get their own Gaussian noise from a separate
    stream. This assumes a servo with position feedback (an analog feedback
    tap or a smart servo); a plain hobby servo reports nothing, and the
    controller would have to use its own last command instead.

    All defaults are zero (a perfect pivot sensor). The project's levels are
    in pivot_config.py (PivotSensorConfig).
    """

    base: SensorNoise
    pivot_angle_std: float = 0.0   # rad
    pivot_rate_std: float = 0.0    # rad/s
    _rng: Optional[np.random.Generator] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.reset()

    @property
    def seed(self) -> int:
        return self.base.seed

    def reset(self) -> None:
        self.base.reset()
        self._rng = np.random.default_rng([self.base.seed, 1])

    def measure(self, state: np.ndarray) -> np.ndarray:
        measured = np.empty(len(state))
        measured[:4] = self.base.measure(state[:4])
        noise = self._rng.standard_normal(2)
        measured[4] = state[4] + self.pivot_angle_std * noise[0]
        measured[5] = state[5] + self.pivot_rate_std * noise[1]
        return measured
