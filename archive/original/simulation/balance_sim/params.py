"""Physical parameters describing one robot configuration.

The ML loop varies these to explore different robot designs, and the same
numbers become the build spec if the physical robot gets made.

All values are SI units. "Body" means everything above the wheels
(chassis, battery, electronics, motors). Wheel values are for both wheels
combined, since the planar model treats them as one rolling body.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RobotParams:
    body_mass: float = 0.80        # kg, chassis + electronics, excluding wheels
    body_com_height: float = 0.10  # m, body center of mass above the wheel axle
    body_inertia: float = 0.0033   # kg m^2, body pitch inertia about its own COM
    wheel_mass: float = 0.10       # kg, both wheels combined
    wheel_radius: float = 0.035    # m
    wheel_inertia: float = 6.1e-5  # kg m^2, both wheels about the axle
    max_torque: float = 0.40       # N m, total motor torque limit (both motors)
    wheel_friction: float = 0.0    # N m s/rad, viscous friction between wheels and body
    gravity: float = 9.81          # m/s^2

    def __post_init__(self):
        positive = (
            "body_mass",
            "body_com_height",
            "body_inertia",
            "wheel_mass",
            "wheel_radius",
            "wheel_inertia",
            "max_torque",
            "gravity",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.wheel_friction < 0:
            raise ValueError("wheel_friction must be non-negative")

    def with_changes(self, **changes) -> "RobotParams":
        """Return a copy with some fields changed (handy for design sweeps)."""
        return replace(self, **changes)
