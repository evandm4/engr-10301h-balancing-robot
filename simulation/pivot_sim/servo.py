"""Servo model: turns an angle command into the torque the servo applies at the pivot."""

from .params import ServoParams


def _clip(x: float, low: float, high: float) -> float:
    return low if x < low else high if x > high else x


class ServoActuator:
    """Position servo. torque(command, angle, rate) returns the torque the servo
    applies to the upper body (and, reversed, to the lower body).

        drive  = clip((command - angle) / full_drive_error, -1, 1)   driver duty, -1..1
        torque = stall_torque * (drive - rate / no_load_speed) - friction * rate

    then clipped to +-stall_torque (the driver's current limit: without it,
    reversing at full speed would briefly ask for twice the stall current).

    Not modeled: deadband, gear backlash, reflected rotor inertia (fold it into
    the upper body's inertia), supply sag, and the servo's internal loop rate
    (it is treated as continuous; its command rate is ServoParams.update_period,
    applied by the simulator).
    """

    def __init__(self, params: ServoParams):
        self.p = params

    def clip_command(self, command: float) -> float:
        return _clip(float(command), -self.p.angle_limit, self.p.angle_limit)

    def torque(self, command: float, angle: float, rate: float) -> float:
        p = self.p
        drive = _clip((self.clip_command(command) - angle) / p.full_drive_error, -1.0, 1.0)
        tau = p.stall_torque * (drive - rate / p.no_load_speed) - p.friction * rate
        return _clip(tau, -p.stall_torque, p.stall_torque)
