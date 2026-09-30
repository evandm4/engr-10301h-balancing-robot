"""The space of controller gains the tuner searches.

Optimizers work on the unit cube [0, 1]^n, which keeps every parameter on the
same scale. SearchSpace converts between that and real gain values. Gains that
span orders of magnitude (like Kp) are searched on a log scale so the optimizer
spends as much effort between 0.3 and 3 as between 3 and 30.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from balance_sim import Planar2D, RobotParams


@dataclass(frozen=True)
class Parameter:
    name: str
    low: float
    high: float
    log: bool = False

    def decode(self, u: float) -> float:
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            return math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        return self.low + u * (self.high - self.low)

    def encode(self, value: float) -> float:
        value = min(max(float(value), self.low), self.high)
        if self.log:
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)


class SearchSpace:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters: List[Parameter] = list(parameters)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def dim(self) -> int:
        return len(self.parameters)

    @property
    def unit_bounds(self):
        return [(0.0, 1.0)] * self.dim

    def decode(self, unit: Sequence[float]) -> np.ndarray:
        return np.array([p.decode(u) for p, u in zip(self.parameters, unit)])

    def encode(self, values: Sequence[float]) -> np.ndarray:
        return np.array([p.encode(v) for p, v in zip(self.parameters, values)])

    def as_dict(self, values: Sequence[float]) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(self.names, values)}


def cascaded_pid_space(
    params: RobotParams,
    kp_min_factor: float = 1.1,
    kp_max: float = 30.0,
    ki_max: float = 5.0,
    kd_max: float = 1.0,
    kv_p_max: float = 0.4,
    kv_i_max: float = 0.3,
) -> SearchSpace:
    """Search space for CascadedPIDController.from_gains([kp, ki, kd, kv_p, kv_i]).

    The lower bound on Kp is kp_min_factor times the analytic minimum that can
    hold this robot up at all (Planar2D.critical_pitch_gain), so no search
    effort is spent on gains that are guaranteed to fall.
    """
    kp_min = kp_min_factor * Planar2D(params).critical_pitch_gain()
    return SearchSpace([
        Parameter("kp", kp_min, kp_max, log=True),
        Parameter("ki", 0.0, ki_max),
        Parameter("kd", 0.0, kd_max),
        Parameter("kv_p", 0.0, kv_p_max),
        Parameter("kv_i", 0.0, kv_i_max),
    ])
