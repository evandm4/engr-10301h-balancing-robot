"""balance_sim: simulation core for an ML-tuned two-wheeled balancing robot."""

from .params import RobotParams
from .actuators import Actuator, DCMotorActuator, IdealActuator, MotorParams
from .sensors import SensorNoise
from .dynamics import Dynamics
from .planar2d import Planar2D, critical_pitch_gain
from .controllers import CascadedPIDController, Controller, PIDController, step_profile
from .simulator import SimResult, simulate
from .metrics import PITCH_REFERENCES, Metrics, compute_metrics, cost, pitch_error_signal
from .trajectory import load_trajectory, save_trajectory

__all__ = [
    "RobotParams",
    "Actuator",
    "IdealActuator",
    "DCMotorActuator",
    "MotorParams",
    "SensorNoise",
    "Dynamics",
    "Planar2D",
    "critical_pitch_gain",
    "Controller",
    "PIDController",
    "CascadedPIDController",
    "step_profile",
    "SimResult",
    "simulate",
    "Metrics",
    "compute_metrics",
    "cost",
    "pitch_error_signal",
    "PITCH_REFERENCES",
    "save_trajectory",
    "load_trajectory",
]
