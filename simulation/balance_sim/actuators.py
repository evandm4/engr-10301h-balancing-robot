"""Actuator models: turn the controller's torque command into applied torque.

The dynamics model asks its actuator "the controller wants this much torque and
the wheels are turning this fast relative to the body; how much do you actually
deliver?" That keeps the motor physics separate from the body physics.

Both classes report TOTAL torque across both wheels (the same convention as the
rest of the model).
"""

import math
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

    def drive(self, command: float, measured_wheel_speed: float) -> float:
        """What the robot code sends the driver for a torque command, computed
        once per control tick from what it measures (measured_wheel_speed: the
        relative wheel speed it sees, rad/s). The simulator delays and holds this
        value and passes it to torque() as `command`. Drivers that take the
        torque command itself (all but VoltageModeActuator) pass it through."""
        return command


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
    gear_efficiency: float   # output torque / (gear_ratio x motor torque), in (0, 1]; back-EMF is unaffected

    def __post_init__(self):
        for name in ("gear_ratio", "torque_constant", "resistance",
                     "supply_voltage", "current_limit", "n_motors", "gear_efficiency"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.gear_efficiency > 1:
            raise ValueError(f"gear_efficiency must be at most 1, got {self.gear_efficiency}")

    @property
    def stall_torque_limit(self) -> float:
        """Total output torque at zero speed, after the current limit (N m)."""
        i = min(self.current_limit, self.supply_voltage / self.resistance)
        return self.n_motors * self.gear_ratio * self.gear_efficiency * self.torque_constant * i

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

    Gearbox losses: the output gets gear_efficiency of the ideal torque
    (n N k_t i), while the back-EMF still uses the full k_t. The firmware is
    taken to know the measured output torque per amp (n N eta k_t), so below
    the limits the command is still delivered exactly; the loss shows up as
    less torque at the current limit. (Real gear losses are partly Coulomb,
    which RobotParams.coulomb_friction covers, and differ when back-driven.)

    Not modeled yet: winding inductance, backlash and driver PWM effects.
    Rotor inertia is part of the dynamics (RobotParams.rotor_inertia).
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
        per_motor_shaft_torque = command / (m.n_motors * m.gear_ratio * m.gear_efficiency)
        i_wanted = per_motor_shaft_torque / m.torque_constant

        v_wanted = m.resistance * i_wanted + m.torque_constant * rotor_speed
        v = float(_clip(v_wanted, -m.supply_voltage, m.supply_voltage))

        i = (v - m.torque_constant * rotor_speed) / m.resistance
        i = float(_clip(i, -m.current_limit, m.current_limit))
        return m.n_motors * m.gear_ratio * m.gear_efficiency * m.torque_constant * i


class VoltageModeActuator(Actuator):
    """A voltage-mode driver (PWM duty, no current control) with the robot code's
    back-EMF feedforward (SOFTWARE_PLAN B1; PHYSICS.md 9.3).

    Once per control tick the robot code turns the torque command into a voltage
    with ITS constants (`estimate`: the nominal, measured-on-the-bench motor):

        V = R^ tau_c / (n N eta^ k^) + k^ N w^,   clipped to +-supply_voltage,

    where tau_c is the command clipped to max_torque and w^ is the relative wheel
    speed it measures (drive()). The simulator delays and holds V like any
    command. The motor then responds with the TRUE constants (`motor`, which
    robust scoring randomizes): i = (V - k N w) / R, clipped to the current limit,
    and torque n N eta k i (torque()). With exact constants and w^ = w this is
    DCMotorActuator; the differences are the constants' mismatch and w^ being a
    sampled, noisy, delayed measurement.

    feedforward=False is the plain voltage mode of PHYSICS.md 9.3 (duty
    proportional to the torque command): extra drag n N^2 eta k^2 / R, and a
    voltage that only reaches R^ max_torque / (n N eta^ k^).

    supply_voltage is the true supply (the robot code reads the battery voltage,
    e.g. the Build HAT's vin), so duty = V / supply_voltage is exact. The Build
    HAT's `bias` dead band is not modeled: the robot sets bias 0 (DATASHEET 7.2).
    """

    def __init__(self, motor: MotorParams, estimate: Optional[MotorParams] = None,
                 max_torque: Optional[float] = None, feedforward: bool = True, feedforward_gain: float = 1.0):
        if max_torque is not None and max_torque <= 0:
            raise ValueError("max_torque must be positive")
        self.m = motor
        self.estimate = motor if estimate is None else estimate
        self.max_torque = max_torque
        self.feedforward = feedforward
        self.feedforward_gain = feedforward_gain   # share of the back-EMF the robot code cancels (1 = all)

    def drive(self, command: float, measured_wheel_speed: float) -> float:
        e = self.estimate
        if self.max_torque is not None:
            command = _clip(command, -self.max_torque, self.max_torque)
        volts = e.resistance * (command / (e.n_motors * e.gear_ratio * e.gear_efficiency * e.torque_constant))
        if self.feedforward:
            back_emf = e.torque_constant * (e.gear_ratio * measured_wheel_speed)
            volts = volts + (back_emf if self.feedforward_gain == 1.0 else self.feedforward_gain * back_emf)
        return float(_clip(volts, -self.m.supply_voltage, self.m.supply_voltage))

    def torque(self, command: float, relative_wheel_speed: float) -> float:
        """command: the drive voltage from drive() (V), not a torque."""
        m = self.m
        rotor_speed = m.gear_ratio * relative_wheel_speed
        i = (command - m.torque_constant * rotor_speed) / m.resistance
        i = float(_clip(i, -m.current_limit, m.current_limit))
        return m.n_motors * m.gear_ratio * m.gear_efficiency * m.torque_constant * i


class SlowPwmActuator(VoltageModeActuator):
    """The Raspberry Pi Build HAT in slow PWM (pwmparams 1 0.01), with the robot code's
    inverse of it (SOFTWARE_PLAN B1b; hardware/legobot/DATASHEET.md 8.7).

    Measured on the LEGO motors: in slow PWM the driver lets the motor coast in the
    PWM off-time and the current falls to zero within the period, so the drive acts
    only during the on-time. Per motor, with duty d in [-1, 1]:

        i_on = (copysign(V, d) - k N w) / R,  clipped to the current limit,
        torque = N eta k |d| i_on       (total: n_motors times that).

    So the zero-speed torque is linear in duty (no dead band; lever on a scale, within
    4%) and the back-EMF damping scales with |d|: spin-up time constants follow
    J / (|d| a k + b) at every duty, a = N eta k / R. At d = 0 the motor coasts.

    drive() is the robot code: it inverts the same model with ITS constants
    (`estimate`) and the speed it measures, w^, and the true supply V (the HAT's vin):

        d = (tau_c / n) / (a^ (V - k^ N w^))   for tau_c >= 0,
        d = (tau_c / n) / (a^ (V + k^ N w^))   for tau_c < 0,     clipped to +-1,

    with tau_c the command clipped to max_torque and a^ = N eta^ k^ / R^. Where the
    denominator is not positive (wheel faster than the no-load speed in the commanded
    direction) the duty saturates at copysign(1, tau_c). feedforward=False drops the
    k^ N w^ terms; feedforward_gain scales them, as in VoltageModeActuator. With exact
    constants and w^ = w the command is delivered exactly below saturation. Unlike the
    linear voltage mode, a mismatch in the constants gives a torque error proportional
    to the command, not a drag at zero command.

    Not modeled: the PWM ripple itself (the period is short against the robot's
    dynamics), regeneration through the diodes above the no-load speed, and the
    `minpwm` 0.01 cut-off.
    """

    def drive(self, command: float, measured_wheel_speed: float) -> float:
        e = self.estimate
        if self.max_torque is not None:
            command = _clip(command, -self.max_torque, self.max_torque)
        per_motor = command / e.n_motors
        a_hat = e.gear_ratio * e.gear_efficiency * e.torque_constant / e.resistance
        back_emf = 0.0
        if self.feedforward:
            back_emf = e.torque_constant * (e.gear_ratio * measured_wheel_speed)
            if self.feedforward_gain != 1.0:
                back_emf = self.feedforward_gain * back_emf
        v = self.m.supply_voltage
        den = a_hat * (v - back_emf) if command >= 0 else a_hat * (v + back_emf)
        duty = per_motor / den if den > 0 else math.copysign(1.0, per_motor)
        return float(_clip(duty, -1.0, 1.0))

    def torque(self, command: float, relative_wheel_speed: float) -> float:
        """command: the duty from drive(), in [-1, 1], not a torque."""
        m = self.m
        rotor_speed = m.gear_ratio * relative_wheel_speed
        i = (math.copysign(m.supply_voltage, command) - m.torque_constant * rotor_speed) / m.resistance
        i = float(_clip(i, -m.current_limit, m.current_limit))
        return m.n_motors * m.gear_ratio * m.gear_efficiency * m.torque_constant * abs(command) * i
