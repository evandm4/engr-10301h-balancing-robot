"""Physical parameters describing one robot configuration.

The ML loop varies these to explore different robot designs, and the same
numbers become the build spec if the physical robot gets made.

All values are SI units. "Body" means everything above the wheels
(chassis, battery, electronics, motors). Wheel values are for both wheels
combined, since the planar model treats them as one rolling body.

There are no defaults: the project's robot is defined once, in config.py
(RobotConfig), and built with CONFIG.build_robot_params().
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RobotParams:
    body_mass: float        # kg, chassis + electronics, excluding wheels
    body_com_height: float  # m, body center of mass above the wheel axle
    body_inertia: float     # kg m^2, body pitch inertia about its own COM
    wheel_mass: float       # kg, both wheels combined
    wheel_radius: float     # m
    wheel_inertia: float    # kg m^2, both wheels about the axle
    max_torque: float       # N m, total motor torque limit (both motors)
    wheel_friction: float   # N m s/rad, viscous friction between wheels and body
    gravity: float          # m/s^2

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
