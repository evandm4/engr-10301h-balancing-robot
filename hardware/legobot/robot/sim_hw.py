"""Stand-ins for the Build HAT, the IMU and the clock, backed by the simulator (laptop testing of D1-D5).

    python -m robot.balance --sim                      # from hardware/legobot

balance.py runs its real loop against these: the IMU reads the specific force and pitch rate of
balance_sim's Planar2D model of legobot_config's robot, the HAT reports encoder counts sampled every
10 ms and delivered 5 ms later, and a duty command reaches the motors `latency` later as
duty x supply volts into the voltage-mode motor model. The clock is virtual, so a run takes milliseconds.
This checks signs, units, the estimator, feedforward and the loop's logic end to end; it is not a
substitute for the simulator's own scoring.
"""
import math
from collections import deque, namedtuple

import numpy as np

from . import control

# Same fields as buildhat_serial.MotorReading and imu.ImuReading (not imported: they need pyserial, smbus2, fcntl)
MotorReading = namedtuple("MotorReading", "t speed pos apos")
ImuReading = namedtuple("ImuReading", "t pitch_rate acc_fwd acc_up acc gyro read_s")


class SimWorld:
    def __init__(self, cfg, pitch=0.03, noise=False, latency=0.005, encoder_delay=0.005, seed=0):
        control.import_config()
        from balance_sim.simulator import _rk4_step
        self._rk4 = _rk4_step
        self.cfg = cfg
        self.dyn = cfg.build_dynamics()
        self.state = self.dyn.initial_state(pitch)
        self.h = cfg.timing.dt_s
        self.t = 0.0
        self.volts = 0.0
        self.supply = cfg.actuator.supply_voltage
        self.latency, self.encoder_delay = latency, encoder_delay
        self.period = 0.010
        self.held = True                      # the robot is held still until the loop arms
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.s = cfg.sensor
        self._pending = deque()               # (time due, volts)
        self._next_sample = 0.0
        self._encoder = deque()               # (arrival time, pos deg)
        self._pair = (None, None)
        self.hat, self.imu, self.clock = SimHat(self), SimImu(self), SimClock(self)

    def release(self):
        self.held = False

    def _relative_deg(self):
        return math.degrees(self.state[0] / self.cfg.robot.wheel_radius - self.state[2])

    def advance(self, target):
        while self.t + 0.5 * self.h <= target:
            while self._pending and self._pending[0][0] <= self.t + 1e-12:
                self.volts = self._pending.popleft()[1]
            if self.t >= self._next_sample - 1e-12:
                self._encoder.append((self.t + self.encoder_delay, math.floor(self._relative_deg())))
                self._next_sample += self.period
            while self._encoder and self._encoder[0][0] <= self.t + 1e-12:
                t_arr, pos = self._encoder.popleft()
                self._pair = (self._pair[1], MotorReading(t_arr, 0, pos, pos))
            if not self.held and abs(self.state[2]) < 1.4:
                self.state = self._rk4(self.dyn, self.state, self.volts, self.h)
            self.t += self.h

    def specific_force(self):
        """(forward, up) specific force at the IMU, as balance_sim.accelerometer_tilt computes it."""
        x, x_dot, th, th_dot = (float(v) for v in self.state[:4])
        if self.held:
            x_dd = th_dd = th_dot = 0.0
        else:
            d = self.dyn.derivatives(self.state, self.volts)
            x_dd, th_dd = float(d[1]), float(d[3])
        h, g = self.s.imu_height, self.cfg.robot.gravity
        s, c = math.sin(th), math.cos(th)
        a_x = x_dd + h * (th_dd * c - th_dot ** 2 * s)
        f_z = h * (-th_dd * s - th_dot ** 2 * c) + g
        return a_x * c - f_z * s, a_x * s + f_z * c


class SimClock:
    def __init__(self, world):
        self.w = world

    def now(self):
        return self.w.t

    def wait_until(self, deadline):
        self.w.advance(deadline)
        return max(0.0, self.w.t - deadline)


class SimImu:
    def __init__(self, world):
        self.w = world

    def read(self):
        w = self.w
        f_fwd, f_up = w.specific_force()
        rate = 0.0 if w.held else float(w.state[3])
        if w.noise:
            n = w.rng.standard_normal(2)
            rate += w.s.pitch_rate_std * n[0] + w.s.pitch_rate_bias
            tilt = math.atan2(-f_fwd, f_up) + w.s.pitch_std * n[1] + w.s.pitch_bias
            mag = math.hypot(f_fwd, f_up)
            f_fwd, f_up = -mag * math.sin(tilt), mag * math.cos(tilt)
        return ImuReading(w.t, rate, f_fwd, f_up, (0.0, f_up, f_fwd), (rate, 0.0, 0.0), 0.0005)

    def close(self):
        pass


class SimHat:
    def __init__(self, world):
        self.w = world
        self.fault = None
        self.vin_reading = None
        self.write_log = []

    def configure(self, **kw):
        pass

    def pair(self, port):
        return self.w._pair

    def latest(self, port):
        return self.w._pair[1]

    def set_pwm(self, left, right, drain=False):
        w = self.w
        w._pending.append((w.t + w.latency, 0.5 * (left + right) * w.supply))

    def coast(self):
        self.w._pending.clear()
        self.w.volts = 0.0

    emergency_coast = coast

    def request_vin(self):
        self.vin_reading = (self.w.t, self.w.supply)

    def vin(self):
        return self.w.supply

    def reader_alive(self):
        return True

    def close(self):
        self.coast()
