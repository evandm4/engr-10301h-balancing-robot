"""pivot_sim: the balancing robot with a servo-driven pivot and a second body on top.

A side project next to balance_sim: it reuses balance_sim's motor, sensor,
controller and cost code but changes none of it. See PIVOT.md at the project
root for the physics and the controllers, and pivot_config.py for the numbers.
"""

from .params import PivotRobotParams, ServoParams
from .servo import ServoActuator
from .dynamics import PivotRobot
from .sensors import PivotSensor
from .controllers import (
    PIVOT_GAIN_NAMES,
    LQRWeights,
    PivotCascadedController,
    PivotController,
    PivotLQRController,
    lqr_gain,
)
from .simulator import PivotSimResult, simulate_pivot
from .metrics import PivotMetrics, pivot_cost, pivot_metrics

__all__ = [
    "PivotRobotParams",
    "ServoParams",
    "ServoActuator",
    "PivotRobot",
    "PivotSensor",
    "PivotController",
    "PivotCascadedController",
    "PivotLQRController",
    "LQRWeights",
    "lqr_gain",
    "PIVOT_GAIN_NAMES",
    "PivotSimResult",
    "simulate_pivot",
    "PivotMetrics",
    "pivot_metrics",
    "pivot_cost",
]
