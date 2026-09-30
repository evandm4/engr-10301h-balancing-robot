"""Scoring a finished run. The ML tuner minimizes `cost`.

The project's weights and fall penalty are set in config.py (CostConfig) and
applied with CONFIG.score(result).
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
    rms_pitch: float                 # rad, from upright
    rms_pitch_tracking_error: float  # rad, from the commanded lean
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
        rms_pitch_tracking_error=float(np.sqrt(np.mean(pitch_error_signal(result) ** 2))),
        effort=float(np.sum(u[:-1] ** 2 * dt)),
        drift=float(s[-1, 0] - s[0, 0]),
        velocity_rms_error=float(np.sqrt(np.mean((s[:, 1] - result.velocity_goal) ** 2))),
    )


PITCH_REFERENCES = ("commanded", "upright")


def pitch_error_signal(result: SimResult, pitch_reference: str = "commanded") -> np.ndarray:
    """Pitch error at every sample, measured against the chosen reference.

    "commanded": error from the lean the controller was asking for. To follow
        a velocity goal the robot has to lean, so this doesn't penalize the
        lean the task requires, only failing to hold it.
    "upright": error from vertical (the original cost). Penalizes every lean,
        including the ones needed to accelerate, so it fights velocity tracking.
    """
    if pitch_reference not in PITCH_REFERENCES:
        raise ValueError(f"pitch_reference must be one of {PITCH_REFERENCES}, got {pitch_reference!r}")
    pitch = result.states[:, 2]
    if pitch_reference == "upright" or result.pitch_reference is None:
        return pitch
    return pitch - result.pitch_reference


def cost(
    result: SimResult,
    effort_weight: float,
    fall_penalty: float,
    velocity_weight: float,
    pitch_reference: str,
) -> float:
    """Lower is better: pitch error + effort + velocity error, plus a fall penalty.

    The pitch term is the integral of pitch error^2, measured against
    `pitch_reference` (see pitch_error_signal).

    The velocity term is the integral of (velocity - goal velocity)^2. With a
    goal of zero it punishes drifting; with a moving goal it punishes failing to
    follow it. Robots without a velocity loop are scored against a goal of zero.

    A fall is always worse than any non-fall, and falling sooner is worse than
    falling later, which gives the optimizer a gradient to follow away from
    hopeless gains.
    """
    t, s, u = result.t, result.states, result.torques
    dt = np.diff(t)
    pitch_error = float(np.sum(pitch_error_signal(result, pitch_reference)[:-1] ** 2 * dt))
    effort = float(np.sum(u[:-1] ** 2 * dt))
    velocity_error = float(np.sum((s[:-1, 1] - result.velocity_goal[:-1]) ** 2 * dt))
    total = pitch_error + effort_weight * effort + velocity_weight * velocity_error
    if result.fell:
        remaining = (result.duration - result.fall_time) / result.duration
        total += fall_penalty * (1.0 + remaining)
    return total
