"""batch_sim: the balance simulation, stepping many robots at once.

A batched rewrite of balance_sim (next to it in simulation/), for NumPy or a
GPU. balance_sim stays the readable reference; the tests in simulation/tests
check this package against it.
"""

from .robot import BatchRobot
from .physics import actuator_torque, derivatives, rk4_step
from .controller import cascaded_update, five_gain_columns, limited_integral
from .schedule import Schedule, SimSpec, build_schedule, noise_table
from .simulate import BatchResult, simulate_batch
from .xp import torch_ops

# TorchSimulator (torch_sim.py) is imported on its own so NumPy-only use never needs PyTorch:
#   from batch_sim.torch_sim import TorchSimulator

__all__ = [
    "BatchRobot",
    "actuator_torque",
    "derivatives",
    "rk4_step",
    "cascaded_update",
    "five_gain_columns",
    "limited_integral",
    "SimSpec",
    "Schedule",
    "build_schedule",
    "noise_table",
    "BatchResult",
    "simulate_batch",
    "torch_ops",
]
