"""Actuator models: turn the controller's torque command into applied torque.

The dynamics model asks its actuator "the controller wants this much torque and
the wheels are turning this fast relative to the body; how much do you actually
deliver?" That keeps the motor physics separate from the body physics.

Both classes report TOTAL torque across both wheels (the same convention as the
rest of the model).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


def _clip(x: float, low: float, high: float) -> float:
    # Plain-Python clip: much faster than np.clip on single numbers, and this
    # runs millions of times during tuning.
    return low if x < low else high if x > high else x


class Actuator(ABC):
    @abstractmethod
    def torque(self, command: float, relative_wheel_speed: float) -> float:
        """Return applied torque (N m) for a torque command (N m).

        relative_wheel_speed is wheel angular velocity minus body pitch rate
        (rad/s), i.e. how fast the motors are actually spinning.
        """


class IdealActuator(Actuator):
    """Delivers the commanded torque, clipped to a fixed limit."""

    def __init__(self, max_torque: float):
        self.max_torque = max_torque

    def torque(self, command: float, relative_wheel_speed: float) -> float:
        return float(_clip(command, -self.max_torque, self.max_torque))


@dataclass(frozen=True)
class MotorParams:
    """Brushed DC motor + gearbox, per motor. No defaults: the project's motor
    is defined once, in config.py (ActuatorConfig), and built with
    CONFIG.actuator.build_motor_params()."""

    gear_ratio: float        # motor turns per output turn
    torque_constant: float   # N m/A at the motor shaft (= back-EMF constant in SI)
    resistance: float        # ohm, winding resistance
    supply_voltage: float    # V
    current_limit: float     # A, per motor (driver or thermal limit)
    n_motors: int

    def __post_init__(self):
        for name in ("gear_ratio", "torque_constant", "resistance",
                     "supply_voltage", "current_limit", "n_motors"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def stall_torque_limit(self) -> float:
        """Total output torque at zero speed, after the current limit (N m)."""
        i = min(self.current_limit, self.supply_voltage / self.resistance)
        return self.n_motors * self.gear_ratio * self.torque_constant * i

    @property
    def no_load_speed(self) -> float:
        """Output shaft speed (rad/s) where back-EMF uses up the supply voltage."""
        return self.supply_voltage / self.torque_constant / self.gear_ratio


class DCMotorActuator(Actuator):
    """Brushed DC motor with back-EMF, voltage supply, and current limit.

    The controller's torque command is turned into the current that would
    produce it. The driver applies the voltage needed to push that current
    against the back-EMF (V = R i + k_e w), but the voltage can't exceed the
    supply. The resulting current is then clipped to the limit. The result is
    that available torque falls as the wheels spin faster, so the robot has
    less authority exactly when it is moving quickly.

    max_torque, if given, caps the torque command before it reaches the motor
    (RobotParams.max_torque: a firmware or gearbox limit). The motor's own
    physics can limit it further, so the effective limit is whichever is lower.
    Leaving it as None means only the motor physics limits torque.

    Not modeled yet: winding inductance, gearbox efficiency and backlash,
    rotor inertia reflected to the wheel (fold it into wheel_inertia for now),
    and driver PWM effects.
    """

    def __init__(self, motor: MotorParams, max_torque: Optional[float] = None):
        if max_torque is not None and max_torque <= 0:
            raise ValueError("max_torque must be positive")
        self.m = motor
        self.max_torque = max_torque

    def torque(self, command: float, relative_wheel_speed: float) -> float:
        m = self.m
        if self.max_torque is not None:
            command = _clip(command, -self.max_torque, self.max_torque)
        rotor_speed = m.gear_ratio * relative_wheel_speed
        per_motor_shaft_torque = command / (m.n_motors * m.gear_ratio)
        i_wanted = per_motor_shaft_torque / m.torque_constant

        v_wanted = m.resistance * i_wanted + m.torque_constant * rotor_speed
        v = float(_clip(v_wanted, -m.supply_voltage, m.supply_voltage))

        i = (v - m.torque_constant * rotor_speed) / m.resistance
        i = float(_clip(i, -m.current_limit, m.current_limit))
        return m.n_motors * m.gear_ratio * m.torque_constant * i
