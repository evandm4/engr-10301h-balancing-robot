"""The simulator loop: steps a Dynamics model forward under a Controller."""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .controllers import Controller
from .dynamics import Dynamics
from .sensors import SensorNoise


@dataclass
class SimResult:
    t: np.ndarray          # (N,) time stamps
    states: np.ndarray     # (N, state_dim) state history
    torques: np.ndarray    # (N,) torque command reaching the actuator (after latency)
    velocity_goal: np.ndarray  # (N,) goal forward velocity the controller was chasing (m/s)
    fell: bool             # True if the robot tipped over before the run ended
    fall_time: Optional[float]
    duration: float        # requested run length in seconds


def _rk4_step(dynamics: Dynamics, state: np.ndarray, torque: float, dt: float) -> np.ndarray:
    k1 = dynamics.derivatives(state, torque)
    k2 = dynamics.derivatives(state + 0.5 * dt * k1, torque)
    k3 = dynamics.derivatives(state + 0.5 * dt * k2, torque)
    k4 = dynamics.derivatives(state + dt * k3, torque)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    dynamics: Dynamics,
    controller: Controller,
    duration: float = 5.0,
    dt: float = 0.001,
    initial_state: Optional[np.ndarray] = None,
    control_dt: Optional[float] = None,
    sensor: Optional[SensorNoise] = None,
    latency: float = 0.0,
) -> SimResult:
    """Run one simulation.

    dt          physics step (RK4 integration).
    control_dt  how often the controller runs. Defaults to dt. Set it larger
                (e.g. 0.005 for a 200 Hz loop) to mimic a real microcontroller;
                the torque is held constant between controller updates.
    sensor      if given, the controller sees noisy measurements instead of the
                true state. The noise stream restarts from the sensor's seed at
                the start of every run.
    latency     seconds between the controller computing a torque and that
                torque reaching the actuator (sensing, computation, and driver
                delay lumped together). On top of the hold set by control_dt.
    The run stops early if the dynamics reports the robot has fallen.
    """
    if latency < 0:
        raise ValueError("latency must be non-negative")
    if control_dt is None:
        control_dt = dt
    steps_per_control = max(1, round(control_dt / dt))
    n_steps = int(round(duration / dt))

    state = dynamics.initial_state() if initial_state is None else np.array(initial_state, dtype=float)
    controller.reset()
    if sensor is not None:
        sensor.reset()

    ts, states, torques = [0.0], [state.copy()], []
    pending = deque()   # (time the command takes effect, torque)
    torque = 0.0        # torque currently reaching the actuator
    fell, fall_time = False, None

    for k in range(n_steps):
        t = k * dt
        if k % steps_per_control == 0:
            measured = state if sensor is None else sensor.measure(state)
            pending.append((t + latency, controller.update(measured, control_dt)))
        while pending and pending[0][0] <= t + 1e-12:
            torque = pending.popleft()[1]
        torques.append(torque)
        state = _rk4_step(dynamics, state, torque, dt)
        ts.append((k + 1) * dt)
        states.append(state.copy())
        if dynamics.is_fallen(state):
            fell, fall_time = True, ts[-1]
            break

    torques.append(torque)  # keep torques the same length as t
    return SimResult(
        t=np.array(ts),
        states=np.array(states),
        torques=np.array(torques),
        velocity_goal=np.array([controller.goal_velocity(t) for t in ts]),
        fell=fell,
        fall_time=fall_time,
        duration=duration,
    )
