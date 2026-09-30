"""Everything about a run that is the same for every simulation in a batch.

The CPU simulator decides things on the fly: whether this step is a control
tick, when a delayed command pops out of the latency deque, what the goal
velocity is. Those depend only on timing, not on the robot, so here they are
worked out once, before the batch runs, using the same float arithmetic as the
CPU loop so the two agree exactly.
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class SimSpec:
    """The shared settings of a batch: timing, controller limits, sensor noise, cost.

    Normally built from the project config with
    batch_tuning.spec_from_eval_config(CONFIG.build_eval_config()). Only
    neutral settings (no latency, a perfect sensor) have defaults.
    """

    goal: Callable[[float], float]          # goal velocity (m/s) vs time, e.g. balance_sim.step_profile
    duration: float
    dt: float
    control_dt: float
    fall_angle: float
    max_pitch_command: float
    pitch_integral_limit: float
    velocity_integral_limit: float
    # Cost weights, as in balance_sim.cost.
    effort_weight: float
    fall_penalty: float
    velocity_weight: float
    pitch_reference: str                    # "commanded" or "upright"
    latency: float = 0.0
    # Sensor noise (standard deviations per control tick), as in SensorNoise.
    pitch_std: float = 0.0
    pitch_rate_std: float = 0.0
    pitch_rate_bias: float = 0.0
    position_std: float = 0.0
    velocity_std: float = 0.0

    def __post_init__(self):
        if self.latency < 0:
            raise ValueError("latency must be non-negative")
        if self.pitch_reference not in ("commanded", "upright"):
            raise ValueError(f"pitch_reference must be 'commanded' or 'upright', got {self.pitch_reference!r}")


@dataclass(frozen=True)
class Schedule:
    n_steps: int
    steps_per_control: int
    n_ticks: int                   # controller updates in a full-length run
    active_command: np.ndarray     # (n_steps,) index of the command reaching the motor at each step; -1 = none yet
    ring_size: int                 # commands that must be kept in flight to model the latency
    goal_per_tick: np.ndarray      # (n_ticks,) goal velocity the controller sees at each update
    goal_per_step: np.ndarray      # (n_steps,) goal velocity recorded at each physics step (for the cost)


def build_schedule(spec: SimSpec) -> Schedule:
    dt, control_dt, latency = spec.dt, spec.control_dt, spec.latency
    steps_per_control = max(1, round(control_dt / dt))
    n_steps = int(round(spec.duration / dt))

    # Replay the CPU simulator's latency deque with the same float comparisons.
    active = np.empty(n_steps, dtype=np.int64)
    pending, current, issued, ring_size = deque(), -1, 0, 1
    for k in range(n_steps):
        t = k * dt
        if k % steps_per_control == 0:
            pending.append((t + latency, issued))
            issued += 1
        while pending and pending[0][0] <= t + 1e-12:
            current = pending.popleft()[1]
        active[k] = current
        # Commands issued so far but not yet reached the motor, plus the one in use.
        ring_size = max(ring_size, issued - max(current, 0))
    n_ticks = issued

    # The controller keeps its own clock by adding control_dt every tick.
    goal_per_tick, clock = np.empty(n_ticks), 0.0
    for j in range(n_ticks):
        goal_per_tick[j] = spec.goal(clock)
        clock += control_dt
    goal_per_step = np.array([spec.goal(k * dt) for k in range(n_steps)])

    return Schedule(n_steps, steps_per_control, n_ticks, active, ring_size, goal_per_tick, goal_per_step)


def noise_table(seeds: Sequence[int], n_ticks: int) -> np.ndarray:
    """Standard-normal sensor noise per (control tick, channel, seed): shape (n_ticks, 4, n_seeds).

    Drawn exactly as SensorNoise does: a fresh default_rng(seed) per run and
    four numbers per controller update, so a batched run sees the same noise
    as the CPU run with that seed.
    """
    table = np.stack([np.random.default_rng(s).standard_normal((n_ticks, 4)) for s in seeds], axis=-1)
    return np.ascontiguousarray(table)
