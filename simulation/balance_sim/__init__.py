"""balance_sim: simulation core for an ML-tuned two-wheeled balancing robot."""

from .params import RobotParams
from .actuators import (Actuator, DCMotorActuator, IdealActuator, MotorParams, SlowPwmActuator, VoltageModeActuator,
                        locked_pwm_tick_phase, locked_pwm_tick_phase_for, pdm_update, pwm_steps_per_tick, pwm_tick_offset)
from .sensors import ComplementaryFilter, Encoder, IMUSensor, SensorNoise, accelerometer_tilt
from .dynamics import Dynamics
from .planar2d import Planar2D, critical_pitch_gain, linearize
from .controllers import CascadedPIDController, Controller, PIDController, push_profile, step_profile
from .lqr import LQRController, LQRWeights, delayed_commands, discrete_model, lqr_gain
from .stability import LinearController, loop_matrix, spectral_radius, stable_latency_range
from .pid_design import model_based_pid_gains
from .policy import (OBSERVATIONS, RESIDUAL_INPUTS, FastPolicyController, PolicyController, PolicySpec, load_policy,
                     pack, pid_equivalent_weights, compile_forward, save_policy, unpack)
from .simulator import SimResult, simulate, steps_per_control
from .metrics import PITCH_REFERENCES, Metrics, compute_metrics, cost, pitch_error_signal
from .trajectory import load_trajectory, save_trajectory

__all__ = [
    "RobotParams",
    "Actuator",
    "IdealActuator",
    "DCMotorActuator",
    "SlowPwmActuator",
    "VoltageModeActuator",
    "MotorParams",
    "SensorNoise",
    "IMUSensor",
    "ComplementaryFilter",
    "Encoder",
    "accelerometer_tilt",
    "locked_pwm_tick_phase",
    "locked_pwm_tick_phase_for",
    "Dynamics",
    "Planar2D",
    "critical_pitch_gain",
    "linearize",
    "Controller",
    "PIDController",
    "CascadedPIDController",
    "step_profile",
    "push_profile",
    "LQRController",
    "LQRWeights",
    "lqr_gain",
    "delayed_commands",
    "discrete_model",
    "LinearController",
    "loop_matrix",
    "spectral_radius",
    "stable_latency_range",
    "model_based_pid_gains",
    "OBSERVATIONS",
    "RESIDUAL_INPUTS",
    "PolicySpec",
    "PolicyController",
    "FastPolicyController",
    "compile_forward",
    "pack",
    "unpack",
    "pid_equivalent_weights",
    "save_policy",
    "load_policy",
    "SimResult",
    "simulate",
    "steps_per_control",
    "Metrics",
    "compute_metrics",
    "cost",
    "pitch_error_signal",
    "PITCH_REFERENCES",
    "save_trajectory",
    "load_trajectory",
]
