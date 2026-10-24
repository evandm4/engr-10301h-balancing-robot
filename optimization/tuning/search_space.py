"""The space of controller gains the tuner searches.

Optimizers work on the unit cube [0, 1]^n, which keeps every parameter on the
same scale. SearchSpace converts between that and real gain values. Gains that
span orders of magnitude (like Kp) are searched on a log scale so the optimizer
spends as much effort between 0.3 and 3 as between 3 and 30.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


# ---- LQR weights ---------------------------------------------------------------

LQR_WEIGHT_NAMES = ("w_velocity", "w_pitch_rate", "w_torque", "w_integral")


def _searched_weights(fixed: Optional[Dict[str, float]]) -> List[str]:
    fixed = fixed or {}
    unknown = set(fixed) - set(LQR_WEIGHT_NAMES)
    if unknown:
        raise ValueError(f"unknown LQR weights {sorted(unknown)}; expected names from {LQR_WEIGHT_NAMES}")
    return [name for name in LQR_WEIGHT_NAMES if name not in fixed]


def lqr_weight_space(bounds: Dict[str, Tuple[float, float]], fixed: Optional[Dict[str, float]] = None) -> SearchSpace:
    """Search space for the LQR's cost weights, relative to a pitch weight of 1.

    Four numbers, like the cascaded PID's four gains: the weights on velocity
    error, pitch rate, torque and the integral of velocity error
    (balance_sim.LQRWeights). Only ratios matter to an LQR, so fixing the
    pitch weight loses nothing. All on a log scale: each spans orders of
    magnitude, and the low end of the rate and integral ranges is "off" in
    practice. `fixed` holds weights at a value instead of searching them
    ({"w_pitch_rate": 0.0} is exactly off, which a log scale cannot reach); the
    space then has only the others. Bounds come from config.py (LQRConfig);
    build it with CONFIG.build_lqr_space().
    """
    return SearchSpace([Parameter(name, *bounds[name], log=True) for name in _searched_weights(fixed)])


def lqr_weights(values: Sequence[float], fixed: Optional[Dict[str, float]] = None):
    """A point of lqr_weight_space (built with the same `fixed`) -> balance_sim.LQRWeights."""
    from balance_sim import LQRWeights

    searched = _searched_weights(fixed)
    if len(values) != len(searched):
        raise ValueError(f"expected {len(searched)} weights {searched}, got {len(values)}")
    w = {**(fixed or {}), **{name: float(v) for name, v in zip(searched, values)}}
    return LQRWeights(velocity=w["w_velocity"], pitch=1.0, pitch_rate=w["w_pitch_rate"], torque=w["w_torque"],
                      velocity_integral=w["w_integral"])


class LQRWeightDesigner:
    """Turns points of lqr_weight_space into LQR gains: the to_gains map for
    batch_tuning.BatchEvaluator(controller="lqr").

    Every design is for the same robot, control period and latency (normally
    the nominal robot and the configured timing): the gains are what would be
    flashed to the robot, and the randomized robots then test them. A design
    whose Riccati equation has no solution gets all-zero gains, which can't
    balance, so the fall penalty scores it.
    """

    def __init__(self, params: RobotParams, control_dt: float, latency: float,
                 fixed: Optional[Dict[str, float]] = None):
        from balance_sim import delayed_commands

        self.params, self.control_dt, self.latency = params, control_dt, latency
        self.fixed = dict(fixed or {})      # weights not searched (see lqr_weight_space)
        self.n_gains = 4 + delayed_commands(control_dt, latency)

    def gain(self, values: Sequence[float]) -> np.ndarray:
        from balance_sim import lqr_gain

        try:
            return lqr_gain(self.params, lqr_weights(values, self.fixed), self.control_dt, self.latency)
        except (np.linalg.LinAlgError, ValueError):
            return np.zeros(self.n_gains)

    def __call__(self, points) -> np.ndarray:
        """(P, 4) weights -> (P, 4 + commands in flight) gains."""
        return np.array([self.gain(w) for w in np.atleast_2d(points)])


# ---- LQR gains, searched directly -----------------------------------------------

def lqr_gain_names(n_in_flight: int, commands: bool = True) -> List[str]:
    """Parameter names for lqr_gain_space: the scale on each searched LQR gain."""
    names = ["s_velocity", "s_pitch", "s_pitch_rate"]
    return names + [f"s_command_{i + 1}" for i in range(n_in_flight)] if commands else names


def lqr_gain_space(n_in_flight: int, scale: float, commands_scale: Optional[float] = None) -> SearchSpace:
    """Search space for the LQR's gains themselves, around a reference design.

    The reference's gains on velocity, pitch, pitch rate and the commands in
    flight (oldest first) are each multiplied by a factor in [1 / scale, scale],
    on a log scale, so the middle of the space is the reference itself. With
    `commands_scale` set, the gains on the commands in flight are not searched
    (LQRGainScaling holds them at that multiple of the design's; 0.0 is off,
    which a log scale cannot reach). Bounds come from config.py (LQRConfig);
    build it with CONFIG.build_direct_lqr_space().
    """
    names = lqr_gain_names(n_in_flight, commands=commands_scale is None)
    return SearchSpace([Parameter(name, 1.0 / scale, scale, log=True) for name in names])


class LQRGainScaling:
    """Turns points of lqr_gain_space into LQR gains: the to_gains map for
    batch_tuning.BatchEvaluator(controller="lqr").

    Gains = the reference's gains, scaled entry by entry. The integral gain
    stays at the reference's (0 for a design without integral action):
    searching it too lets the search find gains with a slow runaway that the
    integral cap and the scenario's finite length hide (LQR.md).

    Every LQR design stabilizes the nominal linear loop, and this keeps that
    guarantee: gains that don't (balance_sim.spectral_radius >= 1 at the
    design latency) are pulled back along the straight line toward the
    reference, which does, to the last stable point (by bisection). So every
    candidate the simulator sees is stable, and candidates past the edge still
    score differently from each other, which the optimizers need.

    `commands_scale` (see lqr_gain_space) fixes the gains on the commands in
    flight at that multiple of the reference design's instead of searching
    them. It is applied to the reference itself, so the pull-back above heads
    for the design with those gains already set, and cannot blend them back in.
    """

    BISECTION_STEPS = 20

    def __init__(self, reference_gain, params: RobotParams, control_dt: float, latency: float,
                 commands_scale: Optional[float] = None):
        self.reference = np.array(reference_gain, dtype=float)
        n_in_flight = len(self.reference) - 4
        if commands_scale is not None:
            self.reference[4:] *= commands_scale
        self.params, self.control_dt, self.latency = params, control_dt, latency
        # The gains the search scales: all but the integral's, and the commands' if they are fixed.
        self.scaled = np.r_[0:3] if commands_scale is not None else np.r_[0:3, 4:4 + n_in_flight]
        if not self.stable(self.reference):
            raise ValueError("the reference gains must stabilize the nominal linear loop")

    def stable(self, gain) -> bool:
        from balance_sim import LinearController, spectral_radius

        return spectral_radius(self.params, LinearController.lqr(gain), self.control_dt, self.latency) < 1.0

    def gain(self, values: Sequence[float]) -> np.ndarray:
        gain = self.reference.copy()
        gain[self.scaled] *= np.asarray(values, dtype=float)
        if self.stable(gain):
            return gain
        step = gain - self.reference
        low, high = 0.0, 1.0          # fraction of the way from the reference: stable at low, not at high
        for _ in range(self.BISECTION_STEPS):
            mid = 0.5 * (low + high)
            if self.stable(self.reference + mid * step):
                low = mid
            else:
                high = mid
        return self.reference + low * step

    def __call__(self, points) -> np.ndarray:
        """(P, n) points -> (P, n) gains."""
        return np.array([self.gain(v) for v in np.atleast_2d(points)])
