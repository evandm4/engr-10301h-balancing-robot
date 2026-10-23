"""The simulator loop for the pivot robot: like balance_sim.simulate, with two
commands (wheel torque, servo angle) instead of one."""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from balance_sim import SimResult

from .controllers import PivotController
from .dynamics import PivotRobot
from .sensors import PivotSensor


@dataclass
class PivotSimResult:
    t: np.ndarray               # (N,) time stamps
    states: np.ndarray          # (N, 6) state history, see dynamics.py
    torques: np.ndarray         # (N,) wheel torque command reaching the motors (after latency)
    servo_commands: np.ndarray  # (N,) angle command the servo is holding (after latency and its frame rate)
    servo_torques: np.ndarray   # (N,) torque the servo actually applies (N m)
    com_pitch: np.ndarray       # (N,) combined-COM pitch from vertical (rad)
    com_pitch_rate: np.ndarray  # (N,) its rate (rad/s)
    velocity_goal: np.ndarray   # (N,) goal forward velocity (m/s)
    pitch_reference: np.ndarray  # (N,) pitch the controller was asking for (rad)
    fell: bool
    fall_time: Optional[float]
    duration: float

    @property
    def upper_pitch(self) -> np.ndarray:
        """Upper body pitch from vertical (rad): theta1 + phi."""
        return self.states[:, 2] + self.states[:, 4]

    def as_planar(self, pitch: str = "com") -> SimResult:
        """This run as a base-robot SimResult, so balance_sim's metrics and cost
        apply unchanged. pitch: "com" puts the combined-COM pitch in the pitch
        column (what balancing is about); "lower" keeps the lower body's."""
        if pitch == "com":
            states = np.column_stack([self.states[:, :2], self.com_pitch, self.com_pitch_rate])
        elif pitch == "lower":
            states = self.states[:, :4]
        else:
            raise ValueError(f"pitch must be 'com' or 'lower', got {pitch!r}")
        return SimResult(
            t=self.t, states=states, torques=self.torques, velocity_goal=self.velocity_goal,
            fell=self.fell, fall_time=self.fall_time, duration=self.duration,
            pitch_reference=self.pitch_reference,
        )


def _rk4_step(dynamics: PivotRobot, state: np.ndarray, command, dt: float) -> np.ndarray:
    k1 = dynamics.derivatives(state, command)
    k2 = dynamics.derivatives(state + 0.5 * dt * k1, command)
    k3 = dynamics.derivatives(state + 0.5 * dt * k2, command)
    k4 = dynamics.derivatives(state + dt * k3, command)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_pivot(
    dynamics: PivotRobot,
    controller: PivotController,
    duration: float,
    dt: float,
    initial_state: Optional[np.ndarray] = None,
    control_dt: Optional[float] = None,
    sensor: Optional[PivotSensor] = None,
    latency: float = 0.0,
) -> PivotSimResult:
    """Run one simulation. Same timing model as balance_sim.simulate:

    control_dt  controller period; commands are held between updates.
    sensor      if given, the controller sees noisy measurements.
    latency     delay from the controller to both actuators.

    On top of that, the servo only picks up a new angle command once every
    ServoParams.update_period (its PWM frame; 0 = every physics step), and
    holds the last one in between. Before its first command arrives the servo
    holds the starting pivot angle.
    """
    if latency < 0:
        raise ValueError("latency must be non-negative")
    if control_dt is None:
        control_dt = dt
    steps_per_control = max(1, round(control_dt / dt))
    steps_per_servo_frame = max(1, round(dynamics.servo.p.update_period / dt))
    n_steps = int(round(duration / dt))

    state = dynamics.initial_state() if initial_state is None else np.array(initial_state, dtype=float)
    controller.reset()
    if sensor is not None:
        sensor.reset()

    ts, states = [0.0], [state.copy()]
    torques, servo_commands, servo_torques, references = [], [], [], []
    pending = deque()            # (time the command takes effect, (torque, servo angle))
    torque = 0.0                 # wheel torque command reaching the motors
    servo_target = float(state[4])   # latest servo command to have arrived
    servo_command = servo_target     # command the servo is acting on (latched per frame)
    fell, fall_time = False, None

    for k in range(n_steps):
        t = k * dt
        if k % steps_per_control == 0:
            measured = state if sensor is None else sensor.measure(state)
            pending.append((t + latency, controller.update(measured, control_dt)))
        while pending and pending[0][0] <= t + 1e-12:
            torque, servo_target = pending.popleft()[1]
        if k % steps_per_servo_frame == 0:
            servo_command = dynamics.servo.clip_command(servo_target)
        command = (torque, servo_command)
        torques.append(torque)
        servo_commands.append(servo_command)
        servo_torques.append(dynamics.applied_torques(state, command)[1])
        references.append(controller.pitch_reference())
        state = _rk4_step(dynamics, state, command, dt)
        ts.append((k + 1) * dt)
        states.append(state.copy())
        if dynamics.is_fallen(state):
            fell, fall_time = True, ts[-1]
            break

    # Keep every series the same length as t.
    command = (torque, servo_command)
    torques.append(torque)
    servo_commands.append(servo_command)
    servo_torques.append(dynamics.applied_torques(state, command)[1])
    references.append(controller.pitch_reference())

    states = np.array(states)
    com = np.array([dynamics.com_pitch(s) for s in states])
    return PivotSimResult(
        t=np.array(ts),
        states=states,
        torques=np.array(torques),
        servo_commands=np.array(servo_commands),
        servo_torques=np.array(servo_torques),
        com_pitch=com[:, 0],
        com_pitch_rate=com[:, 1],
        velocity_goal=np.array([controller.goal_velocity(t) for t in ts]),
        pitch_reference=np.array(references),
        fell=fell,
        fall_time=fall_time,
        duration=duration,
    )
