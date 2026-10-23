"""Scoring a pivot-robot run. Built on balance_sim's cost and metrics, applied
to the combined center of mass, plus terms for the servo and the upper body.

The project's weights are in pivot_config.py (PivotCostConfig, and the base
CostConfig in config.py), applied with PIVOT_CONFIG.score(result).
"""

from dataclasses import dataclass

import numpy as np

from balance_sim import Metrics, compute_metrics, cost

from .simulator import PivotSimResult


@dataclass
class PivotMetrics:
    base: Metrics                 # balance_sim.Metrics of the combined COM (pitch = COM pitch)
    max_abs_pivot: float          # rad
    rms_upper_pitch: float        # rad, upper body from vertical
    max_abs_upper_pitch: float    # rad
    servo_effort: float           # integral of servo torque^2 dt
    servo_saturated: float        # fraction of the run the servo was at its torque limit


def pivot_metrics(result: PivotSimResult, stall_torque: float, settle_band: float = 0.02) -> PivotMetrics:
    upper = result.upper_pitch
    dt = np.diff(result.t)
    return PivotMetrics(
        base=compute_metrics(result.as_planar("com"), settle_band),
        max_abs_pivot=float(np.max(np.abs(result.states[:, 4]))),
        rms_upper_pitch=float(np.sqrt(np.mean(upper**2))),
        max_abs_upper_pitch=float(np.max(np.abs(upper))),
        servo_effort=float(np.sum(result.servo_torques[:-1] ** 2 * dt)),
        servo_saturated=float(np.mean(np.abs(result.servo_torques) >= 0.999 * stall_torque)),
    )


def pivot_cost(
    result: PivotSimResult,
    effort_weight: float,
    fall_penalty: float,
    velocity_weight: float,
    pitch_reference: str,
    servo_effort_weight: float,
    upper_tilt_weight: float,
) -> float:
    """Lower is better. The base project's cost (balance_sim.cost) with the
    combined COM's pitch in place of the body pitch, plus

        servo_effort_weight * integral of servo torque^2
        upper_tilt_weight   * integral of upper body pitch^2

    The second term is how to ask for a level upper body (a camera, a tray):
    it's 0 by default, so the pivot is free to do whatever balances best.
    """
    t = result.t
    dt = np.diff(t)
    total = cost(result.as_planar("com"), effort_weight, fall_penalty, velocity_weight, pitch_reference)
    total += servo_effort_weight * float(np.sum(result.servo_torques[:-1] ** 2 * dt))
    total += upper_tilt_weight * float(np.sum(result.upper_pitch[:-1] ** 2 * dt))
    return total
