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
    rotor_inertia: float    # kg m^2, all motor rotors reflected through the gearbox, n N^2 J_r.
                            # They spin relative to the body, so this is not part of
                            # wheel_inertia (PHYSICS.md 10.3); 0 leaves them out
    max_torque: float       # N m, total motor torque limit (both motors)
    wheel_friction: float   # N m s/rad, viscous friction between wheels and body
    coulomb_friction: float  # N m, Coulomb (dry) friction torque between wheels and body
    coulomb_friction_speed: float  # rad/s, relative wheel speed over which the Coulomb torque
                                   # rises smoothly: tau_c tanh(speed / this) (keeps RK4 accurate)
    gravity: float          # m/s^2
    # Free play between motor and wheel (Planar2D; PHYSICS.md 10.3), half-widths in rad of motor
    # angle relative to the body. The motor's encoder sits between the two: gearbox_backlash is
    # between the motor side (rotor and gear train, rotor_inertia) and the output shaft the encoder
    # reads, coupling_backlash between that shaft and the wheel, which the encoder can't see. Both 0
    # (the default) is a rigid drive train.
    gearbox_backlash: float = 0.0
    coupling_backlash: float = 0.0

    @property
    def backlash(self) -> float:
        """Total free play between the motor side and the wheel, half-width (rad)."""
        return self.gearbox_backlash + self.coupling_backlash

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
        if self.rotor_inertia < 0:
            raise ValueError("rotor_inertia must be non-negative")
        if self.wheel_friction < 0:
            raise ValueError("wheel_friction must be non-negative")
        if self.coulomb_friction < 0:
            raise ValueError("coulomb_friction must be non-negative")
        if self.coulomb_friction_speed <= 0:
            raise ValueError("coulomb_friction_speed must be positive")
        if self.gearbox_backlash < 0 or self.coupling_backlash < 0:
            raise ValueError("gearbox_backlash and coupling_backlash must be non-negative")
        if self.backlash > 0 and self.rotor_inertia <= 0:
            raise ValueError("backlash needs a positive rotor_inertia: the motor side moves on its own in the play")

    def with_changes(self, **changes) -> "RobotParams":
        """Return a copy with some fields changed (handy for design sweeps)."""
        return replace(self, **changes)
