"""Sensor model: what the controller actually sees, as opposed to true state."""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class Encoder:
    """The motors' angle sensors as the Build HAT reports them (SOFTWARE_PLAN B3).

    Every `period` the HAT samples the motor angle (wheel relative to body,
    x / r - theta) in whole counts of 2 pi / counts_per_turn; each sample reaches
    the robot code `delay` later and is held until the next one arrives. The
    robot code's wheel speed is the difference of the two latest counts over one
    period, so it changes once per period and carries quantization noise of
    (2 pi r / counts) / (period sqrt 6) in ground speed (DATASHEET 5.4).

    observe(t, state) is called at the start of every physics step; speed() and
    angle() return what the robot code has at that moment. With backlash the angle
    is the motor's output shaft's (Planar2D's state[6]), not the wheel's.
    """

    def __init__(self, counts_per_turn: int, period: float, delay: float, wheel_radius: float):
        if counts_per_turn <= 0 or not (period > 0) or delay < 0 or not (wheel_radius > 0):
            raise ValueError("counts_per_turn, period and wheel_radius must be positive, delay non-negative")
        self.counts_per_turn, self.period, self.delay, self.wheel_radius = counts_per_turn, period, delay, wheel_radius
        self.resolution = 2.0 * math.pi / counts_per_turn       # rad per count
        self.speed_per_count = self.resolution / period         # rad/s per count of difference
        self.reset()

    def reset(self) -> None:
        self._in_flight = deque()   # (time available, count)
        self._n_samples = 0
        self._latest = self._previous = None

    def observe(self, t: float, state) -> None:
        if t >= self._n_samples * self.period - 1e-12:
            # With backlash (Planar2D) the encoder reads the output shaft, state[6], which the gearbox
            # play separates from the motor side and the coupling play from the wheel (PHYSICS.md 10.3).
            relative_angle = (float(state[6]) if len(state) > 6
                              else float(state[0]) / self.wheel_radius - float(state[2]))
            self._in_flight.append((self._n_samples * self.period + self.delay,
                                    float(np.floor(relative_angle / self.resolution))))
            self._n_samples += 1
        while self._in_flight and self._in_flight[0][0] <= t + 1e-12:
            self._previous, self._latest = self._latest, self._in_flight.popleft()[1]

    def speed(self) -> float:
        """Relative wheel speed (rad/s) from the two latest counts; 0 until there are two."""
        if self._previous is None:
            return 0.0
        return (self._latest - self._previous) * self.speed_per_count

    def angle(self) -> float:
        """Relative wheel angle (rad) of the latest count; 0 until the first arrives."""
        return 0.0 if self._latest is None else self._latest * self.resolution


@dataclass
class SensorNoise:
    """Gaussian measurement noise (plus optional pitch and gyro biases) on the first four
    state entries. Standard deviations are per sample; a new sample is drawn
    each time the controller runs.

    Runs are reproducible: `reset()` restarts the random stream from `seed`, so
    the same seed gives the same noise every time. Change the seed to test
    against a different noise realization.

    All defaults are zero (a perfect sensor). The project's noise levels are
    set once, in config.py (SensorConfig), and built with CONFIG.build_sensor(seed).
    """

    pitch_std: float = 0.0          # rad, pitch angle estimate
    pitch_bias: float = 0.0         # rad, constant pitch-estimate offset (IMU mounting, balance point)
    pitch_rate_std: float = 0.0     # rad/s, gyro
    pitch_rate_bias: float = 0.0    # rad/s, constant gyro offset
    position_std: float = 0.0       # m, encoder-derived position
    velocity_std: float = 0.0       # m/s, encoder-derived velocity
    seed: int = 0
    _rng: Optional[np.random.Generator] = field(default=None, repr=False, compare=False)
    # Encoder model (Encoder; SOFTWARE_PLAN B3), on when encoder_counts > 0: velocity is then
    # wheel_radius x (the encoder's wheel speed + the measured pitch rate) and position
    # wheel_radius x (its angle + the measured pitch), in place of velocity_std and position_std.
    # The simulator calls observe() at every physics step.
    encoder_counts: int = 0             # counts per wheel turn relative to the body; 0 = off
    encoder_period: float = 0.010       # s between the driver's angle samples
    encoder_delay: float = 0.0          # s from a sample to the robot code having it
    wheel_radius: float = 0.0           # m (needed with the encoder)

    def __post_init__(self):
        self._encoder = (Encoder(self.encoder_counts, self.encoder_period, self.encoder_delay, self.wheel_radius)
                         if self.encoder_counts > 0 else None)
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if getattr(self, "_encoder", None) is not None:
            self._encoder.reset()

    @property
    def has_encoder(self) -> bool:
        return self._encoder is not None

    def observe(self, t: float, state) -> None:
        """Called by the simulator at the start of every physics step (time t)."""
        if self._encoder is not None:
            self._encoder.observe(t, state)

    def _apply_encoder(self, measured: np.ndarray) -> None:
        """Replace position and velocity with the encoder's, given measured pitch and pitch rate."""
        e = self._encoder
        measured[1] = self.wheel_radius * (e.speed() + measured[3])
        measured[0] = self.wheel_radius * (e.angle() + measured[2])

    def measure(self, state: np.ndarray) -> np.ndarray:
        measured = np.array(state, dtype=float)
        noise = self._rng.standard_normal(4)
        measured[0] += self.position_std * noise[0]
        measured[1] += self.velocity_std * noise[1]
        measured[2] += self.pitch_std * noise[2] + self.pitch_bias
        measured[3] += self.pitch_rate_std * noise[3] + self.pitch_rate_bias
        if self._encoder is not None:
            self._apply_encoder(measured)
        return measured


class ComplementaryFilter:
    """The robot code's pitch estimator (SOFTWARE_PLAN B2): gyro at high frequency,
    accelerometer tilt at low frequency, crossing over at 1 / (2 pi time_constant).

        estimate = alpha (estimate + dt gyro) + (1 - alpha) tilt,   alpha = tau / (tau + dt)

    The first update starts from the accelerometer tilt (the robot is held still at
    startup). The same class runs on the robot, so the simulation tests the estimator
    it will fly with.
    """

    def __init__(self, time_constant: float, dt: float):
        if not (time_constant > 0 and dt > 0):
            raise ValueError("time_constant and dt must be positive")
        self.time_constant, self.dt = time_constant, dt
        self.alpha = time_constant / (time_constant + dt)
        self.one_minus_alpha = 1.0 - self.alpha
        self.reset()

    def reset(self) -> None:
        self.estimate = None

    def update(self, gyro: float, tilt: float) -> float:
        if self.estimate is None:
            self.estimate = tilt
        else:
            self.estimate = self.alpha * (self.estimate + self.dt * gyro) + self.one_minus_alpha * tilt
        return self.estimate


def accelerometer_tilt(theta: float, theta_dot: float, x_ddot: float, theta_ddot: float,
                       height: float, gravity: float) -> float:
    """The tilt an accelerometer at `height` above the axle (on the body's axis) reads,
    atan2 of its forward and up specific-force components: theta when still, off by
    about -(x_ddot + height theta_ddot) / g while accelerating (DATASHEET 3.4)."""
    s, c = math.sin(theta), math.cos(theta)
    a_x = x_ddot + height * (theta_ddot * c - theta_dot**2 * s)
    f_z = height * (-theta_ddot * s - theta_dot**2 * c) + gravity
    f_forward = a_x * c - f_z * s
    f_up = a_x * s + f_z * c
    return float(np.arctan2(-f_forward, f_up))


@dataclass
class IMUSensor(SensorNoise):
    """A gyro and an accelerometer read every control tick, and the robot code's
    complementary filter, in place of SensorNoise's noisy true pitch (SOFTWARE_PLAN B2).

    The controller sees [position, velocity, pitch estimate, gyro rate]:
      - gyro = theta_dot + pitch_rate_std n + pitch_rate_bias;
      - accelerometer tilt = accelerometer_tilt(...) + pitch_std n + pitch_bias, so in this
        model pitch_std and pitch_bias are the accelerometer tilt's noise per sample and
        its offset (mounting and balance-point error), not the estimate's;
      - pitch estimate = ComplementaryFilter(filter_time_constant, control_dt) of the two,
        so the gyro bias becomes a pitch offset of about pitch_rate_bias x tau, and
        accelerations bias the estimate through the accelerometer (low-passed).
    Position and velocity are SensorNoise's. The same four random numbers per tick, in
    the same order, so batched noise tables apply unchanged.

    The simulator calls measure(state, accelerations) with (x_ddot, theta_ddot) at the
    sample time. The gyro and accelerometer low-pass filters' delay is part of the
    latency (DATASHEET 6), not modeled again here.
    """

    imu_height: float = 0.0             # m, accelerometer above the axle along the body axis
    filter_time_constant: float = 0.5   # s
    control_dt: float = 0.005           # s, the filter's update period (= the simulator's control_dt)
    gravity: float = 9.81
    needs_accelerations = True          # class attribute: the simulator passes (x_ddot, theta_ddot)

    def __post_init__(self):
        self._filter = ComplementaryFilter(self.filter_time_constant, self.control_dt)
        super().__post_init__()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_filter"):
            self._filter.reset()

    def measure(self, state: np.ndarray, accelerations=(0.0, 0.0)) -> np.ndarray:
        x, x_dot, theta, theta_dot = (float(v) for v in state[:4])
        x_ddot, theta_ddot = (float(a) for a in accelerations)
        noise = self._rng.standard_normal(4)
        gyro = theta_dot + (self.pitch_rate_std * noise[3] + self.pitch_rate_bias)
        tilt = (accelerometer_tilt(theta, theta_dot, x_ddot, theta_ddot, self.imu_height, self.gravity)
                + (self.pitch_std * noise[2] + self.pitch_bias))
        estimate = self._filter.update(gyro, tilt)
        measured = np.array(state, dtype=float)
        measured[0] += self.position_std * noise[0]
        measured[1] += self.velocity_std * noise[1]
        measured[2] = estimate
        measured[3] = gyro
        if self._encoder is not None:
            self._apply_encoder(measured)
        return measured
