"""The dynamics interface.

Anything that can answer "given this state and motor torque, how is the state
changing?" can plug into the simulator: the hand-derived planar model, a 3D
model, a PyBullet wrapper, or later a fit to real hardware data.

State layout convention (shared by every implementation so controllers and
metrics stay model-agnostic):

    state[0]  x          forward position of the wheel axle   (m)
    state[1]  x_dot      forward velocity                     (m/s)
    state[2]  theta      body pitch from vertical, forward +  (rad)
    state[3]  theta_dot  pitch rate                           (rad/s)

Models with extra states (yaw, motor current, ...) append them after index 3.
"""

from abc import ABC, abstractmethod

import numpy as np


class Dynamics(ABC):
    @abstractmethod
    def derivatives(self, state: np.ndarray, torque: float) -> np.ndarray:
        """Return d(state)/dt for the given state and commanded motor torque."""

    @abstractmethod
    def initial_state(self, pitch: float = 0.0) -> np.ndarray:
        """Return a starting state with the body tilted by `pitch` radians."""

    @abstractmethod
    def is_fallen(self, state: np.ndarray) -> bool:
        """Return True once the robot has tipped past recovery."""
