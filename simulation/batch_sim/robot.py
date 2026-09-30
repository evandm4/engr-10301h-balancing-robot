"""Robot and motor parameters for a whole batch of simulations.

Every field is either one number shared by the whole batch or a (B,) array
with one value per simulation. Arrays are how to randomize the robot (masses,
inertias, motor constants) across a batch; the physics broadcasts either way.
"""

from dataclasses import dataclass, fields, replace

import numpy as np


@dataclass(frozen=True)
class BatchRobot:
    body_mass: object
    body_com_height: object
    body_inertia: object
    wheel_mass: object
    wheel_radius: object
    wheel_inertia: object
    max_torque: object
    wheel_friction: object
    gravity: object
    # DC motor model; ideal actuator (and these unused) when dc_motor is False.
    # Filled from MotorParams by from_params.
    dc_motor: bool = False
    gear_ratio: object = None
    torque_constant: object = None
    resistance: object = None
    supply_voltage: object = None
    current_limit: object = None
    n_motors: object = None

    @classmethod
    def from_params(cls, params, motor=None, **overrides) -> "BatchRobot":
        """Build from balance_sim's RobotParams and (optional) MotorParams.

        motor=None means the ideal actuator, as in the CPU code. Keyword
        overrides replace any field, e.g. body_mass=np.array([...]) to give
        every simulation its own mass.
        """
        values = {name: getattr(params, name) for name in (
            "body_mass", "body_com_height", "body_inertia", "wheel_mass", "wheel_radius",
            "wheel_inertia", "max_torque", "wheel_friction", "gravity")}
        if motor is not None:
            values["dc_motor"] = True
            values.update({name: getattr(motor, name) for name in (
                "gear_ratio", "torque_constant", "resistance", "supply_voltage",
                "current_limit", "n_motors")})
        values.update(overrides)
        return cls(**values)

    def astype(self, dtype) -> "BatchRobot":
        """Cast array fields to dtype so float32 runs stay float32 throughout.

        Plain Python numbers are left alone: NumPy treats them as "weak" and
        keeps the array's precision.
        """
        changes = {f.name: np.asarray(v, dtype=dtype)
                   for f in fields(self) if isinstance((v := getattr(self, f.name)), np.ndarray)}
        return replace(self, **changes)

    def to_torch(self, device, dtype) -> "BatchRobot":
        """Move array fields to a torch device. Plain numbers stay plain numbers."""
        import torch

        changes = {f.name: torch.as_tensor(v, dtype=dtype, device=device)
                   for f in fields(self) if isinstance((v := getattr(self, f.name)), np.ndarray)}
        return replace(self, **changes)

    def take(self, index) -> "BatchRobot":
        """Select simulations by index from any per-simulation (array) fields."""
        changes = {f.name: v[index]
                   for f in fields(self) if isinstance((v := getattr(self, f.name)), np.ndarray) and v.ndim}
        return replace(self, **changes)

    def critical_pitch_gain(self):
        """Same formula as Planar2D.critical_pitch_gain, per simulation."""
        ml = self.body_mass * self.body_com_height
        a = self.body_mass + self.wheel_mass + self.wheel_inertia / self.wheel_radius**2
        return a * ml * self.gravity / (a + ml / self.wheel_radius)


def per_sim(value, size: int, dtype=np.float64) -> np.ndarray:
    """Broadcast a scalar or (B,) value to a (B,) array."""
    return np.broadcast_to(np.asarray(value, dtype=dtype), (size,)).copy()
