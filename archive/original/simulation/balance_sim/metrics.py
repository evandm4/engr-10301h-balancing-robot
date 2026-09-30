"""Scoring a finished run. The ML tuner minimizes `cost`.

The weights and penalty here are a first pass and are meant to be tuned once
the optimizer is running and we see what behavior they actually reward.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .simulator import SimResult


@dataclass
class Metrics:
    fell: bool
    fall_time: Optional[float]
    settling_time: Optional[float]   # None if pitch never stays inside the band
    max_abs_pitch: float             # rad
    rms_pitch: float                 # rad
    effort: float                    # integral of torque^2 dt
    drift: float                     # net forward travel, m
    velocity_rms_error: float        # RMS gap between actual and goal velocity, m/s


def compute_metrics(result: SimResult, settle_band: float = 0.02) -> Metrics:
    t, s, u = result.t, result.states, result.torques
    pitch = s[:, 2]
    dt = np.diff(t)

    outside = np.where(np.abs(pitch) > settle_band)[0]
    if len(outside) == 0:
        settling = 0.0
    elif outside[-1] + 1 < len(t):
        settling = float(t[outside[-1] + 1])
    else:
        settling = None
    if result.fell:
        settling = None

    return Metrics(
        fell=result.fell,
        fall_time=result.fall_time,
        settling_time=settling,
        max_abs_pitch=float(np.max(np.abs(pitch))),
        rms_pitch=float(np.sqrt(np.mean(pitch**2))),
        effort=float(np.sum(u[:-1] ** 2 * dt)),
        drift=float(s[-1, 0] - s[0, 0]),
        velocity_rms_error=float(np.sqrt(np.mean((s[:, 1] - result.velocity_goal) ** 2))),
    )


def cost(
    result: SimResult,
    effort_weight: float = 0.01,
    fall_penalty: float = 100.0,
    velocity_weight: float = 1.0,
) -> float:
    """Lower is better: pitch error + effort + velocity error, plus a fall penalty.

    The velocity term is the integral of (velocity - goal velocity)^2. With a
    goal of zero it punishes drifting; with a moving goal it punishes failing to
    follow it. Robots without a velocity loop are scored against a goal of zero.

    A fall is always worse than any non-fall, and falling sooner is worse than
    falling later, which gives the optimizer a gradient to follow away from
    hopeless gains.
    """
    t, s, u = result.t, result.states, result.torques
    dt = np.diff(t)
    pitch_error = float(np.sum(s[:-1, 2] ** 2 * dt))
    effort = float(np.sum(u[:-1] ** 2 * dt))
    velocity_error = float(np.sum((s[:-1, 1] - result.velocity_goal[:-1]) ** 2 * dt))
    total = pitch_error + effort_weight * effort + velocity_weight * velocity_error
    if result.fell:
        remaining = (result.duration - result.fall_time) / result.duration
        total += fall_penalty * (1.0 + remaining)
    return total
