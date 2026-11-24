"""Stand-ins for the Build HAT, the IMU and the clock, backed by the simulator (laptop testing of D1-D5).

    python -m robot.balance --sim                      # from hardware/legobot

balance.py runs its real loop against these: the IMU reads the specific force and pitch rate of
balance_sim's Planar2D model of legobot_config's robot, the HAT reports encoder counts sampled every
10 ms and delivered 5 ms later, and a duty command reaches the motor model `latency` later (for "slow_pwm",
as -1/0/+1 pulses by pulse-density modulation on the HAT's tick grid; duty x supply volts for "voltage"). The clock is virtual, so a run takes milliseconds.
This checks signs, units, the estimator, feedforward and the loop's logic end to end; it is not a
substitute for the simulator's own scoring.
"""
import math
from collections import deque, namedtuple

import numpy as np

from . import control

# Same fields as buildhat_serial.MotorReading and imu.ImuReading (not imported: they need pyserial, smbus2, fcntl)
MotorReading = namedtuple("MotorReading", "t speed pos apos")
HAT_LINE_PERIOD = 0.0099912     # s by the Pi's clock: the HAT's lines arrive at 100.09 Hz (C3, tick_lock.py)
ImuReading = namedtuple("ImuReading", "t pitch_rate acc_fwd acc_up acc gyro read_s")


class SimWorld:
    def __init__(self, cfg, pitch=0.03, noise=False, latency=0.005, encoder_delay=0.005, seed=0):
        control.import_config()
        from balance_sim.simulator import _rk4_step
        from balance_sim.actuators import pdm_update
        self._rk4 = _rk4_step
        self._pdm_update = pdm_update
        self.cfg = cfg
        self.dyn = cfg.build_dynamics()
        self.state = self.dyn.initial_state(pitch)
        self.h = cfg.timing.dt_s
        self.t = 0.0
        self.drive = 0.0                      # what the dynamics' actuator takes: a duty, or volts
        self.applied = 0.0                    # what reaches the motor: the drive, or its pulse-density pulse
        self.per_volt = cfg.actuator.model == "voltage"
        self.supply = cfg.actuator.supply_voltage
        self.latency, self.encoder_delay = latency, encoder_delay
        self.period = HAT_LINE_PERIOD          # the HAT's own clock: data lines (and slow-PWM ticks) on it
        # Slow PWM is pulse-density modulation (balance_sim.pdm_update): every tick the HAT sends a -1, 0 or +1
        # pulse from a running sum of the duty. The ticks keep the sim's phase against the first sample and
        # run on the same HAT clock as the lines (not the physics grid).
        actuator = getattr(self.dyn, "actuator", None)
        self.pwm_tick = getattr(actuator, "pwm_tick", 0.0)
        self.pdm_total = 0.0
        self._next_pwm_tick = getattr(actuator, "pwm_tick_phase", 0.0) * self.period
        self.now = 0.0                        # the loop's clock (continuous); physics steps on the h grid
        self._line_times = deque(maxlen=512)
        self.held = True                      # the robot is held still until the loop arms
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.s = cfg.sensor
        self._pending = deque()               # (time due, drive)
        self._next_sample = 0.0
        self._encoder = deque()               # (arrival time, pos deg)
        self._pair = (None, None)
        self.hat, self.imu, self.clock = SimHat(self), SimImu(self), SimClock(self)

    def release(self):
        self.held = False

    def _relative_deg(self):
        if len(self.state) > 6:     # free play modelled: the encoder reads the output shaft (Planar2D state[6])
            return math.degrees(self.state[6])
        return math.degrees(self.state[0] / self.cfg.robot.wheel_radius - self.state[2])

    def advance(self, target):
        """Physics up to `target` on the loop's clock; encoder lines that have arrived by then are visible.
        A line is sampled on the HAT's grid and arrives encoder_delay later, plus jitter with --sim-noise
        (exponential, mean 0.65 ms, as the Pi's reader thread shows)."""
        self.now = max(self.now, target)
        while self.t + 0.5 * self.h <= target:
            while self._pending and self._pending[0][0] <= self.t + 1e-12:
                self.drive = self._pending.popleft()[1]
            if self.pwm_tick > 0:
                while self.t >= self._next_pwm_tick - 1e-12:
                    if not self.held:
                        self.pdm_total, self.applied = self._pdm_update(self.pdm_total, self.drive)
                    self._next_pwm_tick += self.period
            else:
                self.applied = self.drive
            while self.t >= self._next_sample - 1e-12:
                jitter = self.rng.exponential(0.00065) if self.noise else 0.0
                self._encoder.append((self._next_sample + self.encoder_delay + jitter,
                                      math.floor(self._relative_deg())))
                self._next_sample += self.period
            if not self.held and abs(self.state[2]) < 1.4:
                if getattr(self.dyn, "has_backlash", False):
                    self.state = self.dyn.step(self.state, self.applied, self.h)   # with the free play's events
                else:
                    self.state = self._rk4(self.dyn, self.state, self.applied, self.h)
            self.t += self.h
        while self._encoder and self._encoder[0][0] <= self.now + 1e-12:
            t_arr, pos = self._encoder.popleft()
            self._pair = (self._pair[1], MotorReading(t_arr, 0, pos, pos))
            self._line_times.append(t_arr)

    def specific_force(self):
        """(forward, up) specific force at the IMU, as balance_sim.accelerometer_tilt computes it."""
        x, x_dot, th, th_dot = (float(v) for v in self.state[:4])
        if self.held:
            x_dd = th_dd = th_dot = 0.0
        else:
            # With free play, the contact a reversed drive would release (begin_step, as the simulator's IMU
            # sample does); the live state is not touched.
            sensed = self.dyn.begin_step(self.state, self.applied) if getattr(self.dyn, "has_backlash", False) else self.state
            d = self.dyn.derivatives(sensed, self.applied)
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
        return self.w.now

    def wait_until(self, deadline):
        late = max(0.0, self.w.now - deadline)
        self.w.advance(deadline)
        return late


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
        return ImuReading(w.now, rate, f_fwd, f_up, (0.0, f_up, f_fwd), (rate, 0.0, 0.0), 0.0005)

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

    def line_times(self, port):
        return list(self.w._line_times)

    def latest(self, port):
        return self.w._pair[1]

    def set_pwm(self, left, right, drain=False):
        w = self.w
        d = 0.5 * (left + right)
        w._pending.append((w.now + w.latency, d * w.supply if w.per_volt else d))

    def coast(self):
        self.w._pending.clear()
        self.w.drive = 0.0
        self.w.applied = 0.0     # coasting stops the pulses at once, and the running sum with them
        self.w.pdm_total = 0.0

    emergency_coast = coast

    def request_vin(self):
        self.vin_reading = (self.w.now, self.w.supply)

    def vin(self):
        return self.w.supply

    def reader_alive(self):
        return True

    def close(self):
        self.coast()
