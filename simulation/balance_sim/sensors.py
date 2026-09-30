"""Sensor model: what the controller actually sees, as opposed to true state."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SensorNoise:
    """Gaussian measurement noise (plus optional gyro bias) on the first four
    state entries. Standard deviations are per sample; a new sample is drawn
    each time the controller runs.

    Runs are reproducible: `reset()` restarts the random stream from `seed`, so
    the same seed gives the same noise every time. Change the seed to test
    against a different noise realization.

    All defaults are zero (a perfect sensor). The project's noise levels are
    set once, in config.py (SensorConfig), and built with CONFIG.build_sensor(seed).
    """

    pitch_std: float = 0.0          # rad, pitch angle estimate
    pitch_rate_std: float = 0.0     # rad/s, gyro
    pitch_rate_bias: float = 0.0    # rad/s, constant gyro offset
    position_std: float = 0.0       # m, encoder-derived position
    velocity_std: float = 0.0       # m/s, encoder-derived velocity
    seed: int = 0
    _rng: Optional[np.random.Generator] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def measure(self, state: np.ndarray) -> np.ndarray:
        measured = np.array(state, dtype=float)
        noise = self._rng.standard_normal(4)
        measured[0] += self.position_std * noise[0]
        measured[1] += self.velocity_std * noise[1]
        measured[2] += self.pitch_std * noise[2]
        measured[3] += self.pitch_rate_std * noise[3] + self.pitch_rate_bias
        return measured
