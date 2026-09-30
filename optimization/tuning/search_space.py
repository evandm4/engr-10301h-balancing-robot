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

from balance_sim import RobotParams, critical_pitch_gain


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

    def at_bounds(self, values: Sequence[float], margin: float = 0.01) -> List[str]:
        """Names of the parameters within `margin` (in unit-cube terms) of a search bound.

        A result sitting on a bound suggests the best value may lie outside the range.
        """
        return [n for n, u in zip(self.names, self.encode(values)) if u < margin or u > 1 - margin]


def cascaded_pid_space(
    params: RobotParams,
    kp_min_factor: float,
    kp_max: float,
    ki_max: float,
    kd_max: float,
    kv_p_max: float,
) -> SearchSpace:
    """Search space for CascadedPIDController.from_gains([kp, ki, kd, kv_p]).

    Bounds come from config.py (SearchConfig); build it with
    CONFIG.build_search_space(). The lower bound on Kp is kp_min_factor times
    the analytic minimum that can hold this robot up at all
    (critical_pitch_gain), so no search effort is spent on gains that are
    guaranteed to fall. kv_i isn't searched: it always came out 0.
    """
    kp_min = kp_min_factor * critical_pitch_gain(params)
    return SearchSpace([
        Parameter("kp", kp_min, kp_max, log=True),
        Parameter("ki", 0.0, ki_max),
        Parameter("kd", 0.0, kd_max),
        Parameter("kv_p", 0.0, kv_p_max),
    ])
