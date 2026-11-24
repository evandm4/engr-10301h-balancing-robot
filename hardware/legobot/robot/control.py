"""The balancing loop's hardware-free core (SOFTWARE_PLAN D1, D3, D4): estimator, controller, feedforward, safety.

Everything the simulator also models comes from the simulator's own code and constants, so the robot
runs what was tuned:
  - the pitch estimate is balance_sim.ComplementaryFilter with legobot_config's time constant and the
    200 Hz control period (B2);
  - wheel speed is the difference of the two newest encoder counts over the HAT's 10 ms sample period,
    as balance_sim.Encoder computes it (B3), and velocity = r (wheel speed + gyro rate) as SensorNoise does;
  - the controller is balance_sim's CascadedPIDController, built by legobot_config's
    ProjectConfig with its torque cap, lean cap and integral limits;
  - the duty is the config's actuator model's own drive() (B1, B1b) with its nominal constants and the
    measured battery voltage, per wheel with that wheel's speed: SlowPwmActuator's inverse of the HAT's slow
    PWM ("slow_pwm"), or VoltageModeActuator's voltage divided by vin ("voltage").

Nothing here touches the hardware, so robot.sim_hw can run it in closed loop with the simulator.

Robot code on the Pi imports config.py and legobot_config.py from ~/legobot and balance_sim from
~/legobot/simulation (hardware/legobot/deploy.sh copies them); in the repo they are found from here.
"""
import json
from collections import deque
import math
import sys
from dataclasses import replace
from pathlib import Path

from . import loop_config as lc


DRIVER_MODELS = ("slow_pwm", "voltage")   # actuator models whose drive() the robot can send to the HAT


def import_config():
    """The `config` module (config.py), from the working directory on the Pi or the repo root here."""
    try:
        import config
    except ImportError:
        root = Path(__file__).resolve().parents[3]
        if not (root / "config.py").is_file():
            raise
        sys.path.insert(0, str(root))
        import config
    return config


def load_robot_config():
    """legobot_config's ProjectConfig. Refuses an actuator model the Build HAT can't run (ideal, dc_motor)."""
    cfg = import_config().load_config("legobot")
    if cfg.actuator.model not in DRIVER_MODELS:
        raise ValueError(f"legobot actuator model is {cfg.actuator.model!r}; the robot code runs {DRIVER_MODELS}")
    return cfg


# ------------------------------------------------------------------------------------ controller ---
def load_gains_file(path):
    """A cascaded-PID tuning result JSON: {"gains": {"kp", "ki", "kd", "kv_p"[, "kv_i"]}} (or a list in that
    order). An optional "feedforward_gain" is the robot-code feedforward gain tuned with these gains
    (A4_BASELINES). The LEGO robot runs the cascaded PID only (the LQR was dropped, 2026-11-23)."""
    with open(path) as f:
        data = json.load(f)
    out = {"source": str(path), "feedforward_gain": data.get("feedforward_gain")}
    if "K" in data:
        raise ValueError(f"{path} holds LQR gains; the LEGO robot runs the cascaded PID only")
    g = data["gains"]
    if isinstance(g, dict):
        missing = {"kp", "ki", "kd", "kv_p"} - set(g)
        if missing:
            raise ValueError(f"{path}: gains lack {sorted(missing)} (not a cascaded-PID result?)")
        g = [g["kp"], g["ki"], g["kd"], g["kv_p"]] + ([g["kv_i"]] if "kv_i" in g else [])
    out["kind"], out["gains"] = "pid", [float(x) for x in g]
    return out


def build_controller(cfg, spec="hand", goal="still"):
    """The cascaded PID (CascadedPIDController, built by ProjectConfig.build_controller). spec: "hand"
    (legobot hand-tuned gains), "model" (model-based baseline), or a gains JSON path. goal: "still" (hold
    0 m/s) or "scenario" (the tuning scenario's goal profile, from the moment the loop arms). Returns
    (controller, description dict)."""
    ff_gain = None
    kind = "pid"
    if spec == "hand":
        gains = list(cfg.controller.hand_tuned_gains)
    elif spec == "model":
        gains = list(cfg.model_based_gains())
    else:
        loaded = load_gains_file(spec)
        gains, ff_gain = loaded["gains"], loaded["feedforward_gain"]
    controller = cfg.build_controller(gains)
    if goal == "still":
        controller.goal = 0.0
    elif goal != "scenario":
        raise ValueError(f"goal must be 'still' or 'scenario', not {goal!r}")
    return controller, {"spec": spec, "kind": kind, "gains": gains, "goal": goal, "feedforward_gain": ff_gain}


# ------------------------------------------------------------------------------------- sensing ---
def wheel_speed(pair, period):
    """Relative wheel speed (rad/s) from a (previous, latest) MotorReading pair: the count difference over
    one HAT sample period, as balance_sim.Encoder.speed. 0 until two readings exist.

    Consecutive data lines are consecutive HAT samples (the HAT samples every `period`); arrival times
    jitter by up to 20 ms when the reader thread is late, so they are not used to scale the difference."""
    prev, latest = pair
    if prev is None or latest is None:
        return 0.0
    return math.radians(latest.pos - prev.pos) / period


def wheel_angle(pair):
    latest = pair[1]
    return 0.0 if latest is None else math.radians(latest.pos)


def accel_tilt(acc_fwd, acc_up):
    """Accelerometer tilt, theta > 0 leaning forward (imu.pitch_from_accel)."""
    return math.atan2(-acc_fwd, acc_up)


class PitchEstimator:
    """Gyro bias and balance-point zero removed, then the simulator's complementary filter (B2)."""

    def __init__(self, cfg, gyro_bias=0.0, pitch_zero=0.0):
        sim = import_config()  # noqa: F841  (makes balance_sim importable)
        from balance_sim import ComplementaryFilter
        self.filter = ComplementaryFilter(cfg.sensor.filter_time_constant, cfg.timing.control_dt_s)
        self.gyro_bias = gyro_bias
        self.pitch_zero = pitch_zero

    def update(self, pitch_rate, acc_fwd, acc_up):
        """Returns (pitch estimate, bias-corrected gyro rate, zeroed accelerometer tilt)."""
        gyro = pitch_rate - self.gyro_bias
        tilt = accel_tilt(acc_fwd, acc_up) - self.pitch_zero
        return self.filter.update(gyro, tilt), gyro, tilt


# ---------------------------------------------------------------------------------- actuation ---
class Feedforward:
    """Torque command -> duty per wheel, by the config actuator model's drive() with the config's nominal
    motor constants, the robot's torque cap, and the config's feedforward on/off and gain. Per wheel, with
    a^ = eta^ k^ / R^ and w that wheel's measured speed:

      "slow_pwm" (B1b, pwmparams 1 0.01): d = (tau_c / n) / (a^ (V -+ gain k^ w)), - for tau >= 0, + below
      "voltage"  (B1):  V_drive = R^ tau_c / (n eta^ k^) + gain k^ w, clipped to +-V;  d = V_drive / V

    Each wheel gets half the torque (the n) and its own back-EMF. V is the battery voltage the HAT reports
    (set_supply) times port_plimit, so the duty uses the real voltage as the sim's true supply does."""

    def __init__(self, cfg, gain=None, plimit=1.0):
        sim = import_config()  # noqa: F841
        import balance_sim
        self.model = cfg.actuator.model
        if self.model not in DRIVER_MODELS:
            raise ValueError(f"no robot-side driver for actuator model {self.model!r}")
        self._cls = balance_sim.SlowPwmActuator if self.model == "slow_pwm" else balance_sim.VoltageModeActuator
        self._per_volt = self.model == "voltage"     # its drive() returns volts, not a duty
        self.motor = cfg.actuator.build_motor_params()
        self.max_torque = cfg.build_robot_params().max_torque
        self.enabled = cfg.actuator.voltage_feedforward
        self.gain = cfg.actuator.voltage_feedforward_gain if gain is None else float(gain)
        self.plimit = plimit
        self.set_supply(cfg.actuator.supply_voltage)

    def set_supply(self, vin):
        self.supply = vin * self.plimit
        m = replace(self.motor, supply_voltage=self.supply)
        self.actuator = self._cls(m, m, max_torque=self.max_torque, feedforward=self.enabled,
                                  feedforward_gain=self.gain)

    def duties(self, torque, w_left, w_right):
        d = self.actuator.drive
        if self._per_volt:
            return d(torque, w_left) / self.supply, d(torque, w_right) / self.supply
        return d(torque, w_left), d(torque, w_right)


# ------------------------------------------------------------------------------------- the brain ---
class Brain:
    """One control tick: sense() turns raw readings into the controller's state, act() into duties."""

    def __init__(self, cfg, controller, estimator, feedforward, duty_cap, hat_period):
        self.cfg = cfg
        self.dt = cfg.timing.control_dt_s
        self.r = cfg.robot.wheel_radius
        self.controller = controller
        self.estimator = estimator
        self.feedforward = feedforward
        self.duty_cap = duty_cap
        self.hat_period = hat_period

    def sense(self, pitch_rate, acc_fwd, acc_up, pair_left, pair_right):
        """Returns (state [x, v, theta, theta_dot], extras dict) in the simulator's conventions."""
        theta, gyro, tilt = self.estimator.update(pitch_rate, acc_fwd, acc_up)
        wl, wr = wheel_speed(pair_left, self.hat_period), wheel_speed(pair_right, self.hat_period)
        angle = 0.5 * (wheel_angle(pair_left) + wheel_angle(pair_right))
        v = self.r * (0.5 * (wl + wr) + gyro)
        x = self.r * (angle + theta)
        return [x, v, theta, gyro], {"tilt": tilt, "w_left": wl, "w_right": wr}

    def act(self, state, w_left, w_right):
        """Returns (torque command, duty_left, duty_right), duties clamped to duty_cap."""
        torque = self.controller.update(state, self.dt)
        dl, dr = self.feedforward.duties(torque, w_left, w_right)
        cap = self.duty_cap
        return torque, max(-cap, min(cap, dl)), max(-cap, min(cap, dr))


# ------------------------------------------------------------------------- HAT line clock (D2) ---
class LineClock:
    """The Build HAT's 100 Hz data-line grid, estimated from line arrival times (perf_counter s).

    The HAT samples and sends a port's line on its own clock (about 880 ppm off the Pi's), and in slow PWM
    its pulse ticks are locked to the same grid (C3, tick_lock.py). Each arrival is the grid time plus a
    transit delay plus non-negative jitter (sd about 0.65 ms; up to 20 ms when the reader thread is late),
    so the grid is the LOWER envelope of the arrivals:

        period  = least-squares slope of arrival time against line index (index by rounding to the
                  previous period estimate, so dropped lines don't shift it);
        t_ref   = the `envelope` quantile of arrival - period x index, placed at the newest line's index.

    locked() is true once `min_lines` lines are in and the newest line at the last update is under `stale` s
    old: longer than the loop's update interval (100 ms), since stale motor data trips the run at 50 ms anyway
    (Safety). Pure Python
    and cheap enough (about 0.2 ms for 256 lines on the Pi 3B) to update every few ticks."""

    def __init__(self, nominal_period=0.010, window=256, min_lines=50, envelope=0.10, stale=0.25):
        self.period = nominal_period
        self.window, self.min_lines, self.envelope, self.stale = window, min_lines, envelope, stale
        self.smoothing = 0.1                  # each window's slope is good to about +-55 ppm (jitter sd 0.65 ms)
        self.t_ref = None
        self.newest = None
        self.n = 0

    def update(self, arrivals):
        a = arrivals[-self.window:]
        if len(a) < self.min_lines:
            return False
        p = self.period
        ref = a[0] if self.t_ref is None else self.t_ref
        # index each line by rounding against the median residual, so one late line can't shift the indices
        k0 = [round((t - ref) / p) for t in a]
        r0 = sorted(t - ref - p * k for k, t in zip(k0, a))
        ref += r0[len(r0) // 2]
        ks = [round((t - ref) / p) for t in a]
        n = len(a)
        km = sum(ks) / n
        tm = sum(a) / n
        skk = sum((k - km) ** 2 for k in ks)
        if skk <= 0:
            return False
        period = sum((k - km) * (t - tm) for k, t in zip(ks, a)) / skk
        if not (0.98 * p < period < 1.02 * p):
            return False                      # nonsense (e.g. a stream restart): keep the old estimate
        if self.t_ref is not None:            # after the first lock, smooth the period across updates
            period = p + self.smoothing * (period - p)
        resid = sorted(t - period * k for k, t in zip(ks, a))
        base = resid[min(n - 1, int(self.envelope * n))]
        self.period, self.t_ref, self.newest, self.n = period, base + period * ks[-1], a[-1], n
        return True

    def locked(self, now):
        return self.t_ref is not None and now - self.newest < self.stale

    def phase(self, t):
        """Time since the latest grid point at or before t (0 <= phase < period)."""
        return (t - self.t_ref) % self.period


def port_lag(ref_times, other_times, nominal_period):
    """How far the other port's line grid lags the reference port's, in [0, period) s; None if too few lines.

    Each port streams on its own grid, started by configure(), so the lag changes from run to run (1.5-8 ms
    in the 2026-11-23 logs). Logged as a diagnostic only: the pulse ticks don't follow their own port's grid,
    and neither port's tick holds a fixed phase between runs (B1 follow-up and cable-swap sweeps, 2026-11-23:
    port A 8.8 ms, port B 7.2 / 4.3 ms, then both 3.2 ms after re-plugging the motors, one power-up)."""
    a, b = LineClock(nominal_period), LineClock(nominal_period)
    if not (a.update(ref_times) and b.update(other_times)):
        return None
    return (b.t_ref - a.t_ref) % a.period


class Pacer:
    """The loop's tick times. Locked: two ticks per HAT line period, at `offset` s after each half-period
    point of the LineClock grid, so every command lands at the same phase of the HAT's pulse ticks.
    Unlocked (no lock yet, or lines stale): ticks every dt, as before."""

    def __init__(self, dt, line_clock=None, offset=0.0):
        self.dt, self.lc, self.offset = dt, line_clock, offset

    def next(self, prev, now):
        lc = self.lc
        if lc is None or not lc.locked(now):
            return prev + self.dt, False
        half = 0.5 * lc.period
        base = lc.t_ref + self.offset
        m = math.ceil((prev + 0.5 * half - base) / half)   # the first grid tick more than half a tick after prev
        return base + m * half, True


# --------------------------------------------------------------------------------------- safety ---
def runaway_speed_for_vin(vin):
    """Wheel speed (rad/s) at which the runaway trip fires: RUNAWAY_FRACTION of the no-load speed at this
    supply. In slow PWM at duty 1 the no-load speed is (a V - tau_c) / (a k + b) with a = 0.0597 N m/V,
    k = 0.348 V s/rad, b = 0.00085 N m s/rad and the Coulomb term 0.021 N m per motor (DATASHEET 8.7): 17.8 rad/s
    at 6.8 V, 20.0 at 7.6, 22.2 at 8.4. A fixed threshold is too high on a sagging battery and too low on a
    full one (review H6). Falls back to lc.RUNAWAY_SPEED when vin is unknown."""
    if vin is None:
        return lc.RUNAWAY_SPEED
    vin = max(float(vin), lc.VIN_CUTOFF)    # a glitched low reading (0 V) must not push the threshold to zero or below
    return lc.RUNAWAY_FRACTION * (0.0597 * vin - 0.021) / (0.0597 * 0.348 + 0.00085)


class Safety:
    """The per-tick trips of D3 that need no clock of their own. check() returns a reason string or None.
    The loop itself handles the loop-stall trip and the watchdog thread (balance.py)."""

    def __init__(self, dt, tilt_limit=lc.TILT_LIMIT, vin_cutoff=lc.VIN_CUTOFF, stall_duty=lc.STALL_DUTY):
        self.dt = dt
        self.stall_duty = stall_duty
        self.tilt_limit = tilt_limit
        self.vin_cutoff = vin_cutoff
        n = max(1, int(round(lc.STALL_S / dt)))
        self._stall = [deque(maxlen=n), deque(maxlen=n)]   # speed in the duty's direction, while |duty| is high
        self.wrong_way_ticks = [0, 0]
        m = max(1, int(round(lc.RUNAWAY_S / dt)))
        self._runaway = [deque(maxlen=m), deque(maxlen=m)]  # each wheel's signed speed over the last RUNAWAY_S
        self.vin_low = 0
        self.vin = None                 # the latest vin reading, for the runaway threshold
        self._last_vin_t = None
        self.first_vin_t = None

    def check(self, t, theta, duties, speeds, motor_ages, vin_reading, fault, reader_alive, running):
        if fault is not None:
            return f"motor power fault: {fault[1]!r}"
        if not reader_alive:
            return "serial reader thread died"
        for side, age in zip(("left", "right"), motor_ages):
            if age is None or age > lc.MOTOR_STALE_S:
                return f"{side} motor data stale ({'none' if age is None else f'{age * 1e3:.0f} ms'})"
        if vin_reading is not None and 5.0 <= vin_reading[1] <= 10.0:   # ignore an implausible reading here
            self.vin = vin_reading[1]
        if vin_reading is not None and vin_reading[0] != self._last_vin_t:
            self._last_vin_t = vin_reading[0]
            self.vin_low = self.vin_low + 1 if vin_reading[1] < self.vin_cutoff else 0
            if self.vin_low >= lc.VIN_LOW_COUNT:
                return f"battery low: {vin_reading[1]:.2f} V < {self.vin_cutoff} V"
        if self.first_vin_t is None:
            self.first_vin_t = t
        last_vin = self.first_vin_t if vin_reading is None else max(vin_reading[0], self.first_vin_t)
        if t - last_vin > lc.VIN_STALE_S:
            return "no vin reply from the HAT"
        if not running:
            return None
        if abs(theta) > self.tilt_limit:
            return f"tilt {theta:+.2f} rad past {self.tilt_limit} rad"
        for i, side in enumerate(("left", "right")):
            d, w = duties[i], speeds[i]
            # Stall: |duty| high on every tick of the last STALL_S, and the wheel's MEAN speed in the duty's
            # direction under STALL_SPEED. A mean, because a held wheel still creeps or is back-driven by single
            # encoder counts (1.75 rad/s each), which reset any per-tick test (bench, 2026-11-23).
            q = self._stall[i]
            if abs(d) >= self.stall_duty:
                q.append(w if d > 0 else -w)
                if len(q) == q.maxlen:
                    mean = sum(q) / len(q)
                    if mean < lc.STALL_SPEED:
                        return (f"{side} wheel stalled (duty {d:+.2f}, mean {mean:+.2f} rad/s in its direction "
                                f"over {lc.STALL_S} s)")
            else:
                q.clear()
            # Encoder against drive: a strong duty and the wheel turning fast the other way means a sign
            # error, which makes the feedforward positive feedback (D4's sign check, at run time).
            self.wrong_way_ticks[i] = (self.wrong_way_ticks[i] + 1
                                       if abs(d) >= self.stall_duty and d * w < 0 and abs(w) > 5.0 else 0)
            if self.wrong_way_ticks[i] * self.dt >= 0.2:
                return f"{side} wheel turns against its duty (duty {d:+.2f}, {w:+.2f} rad/s): sign error?"
        # Runaway: both wheels' MEAN speed over the last RUNAWAY_S near the no-load speed, the same way. On the
        # floor that is past 0.59 m/s for 1.5 s, beyond anything the scenario asks; lifted while armed (held
        # upright, so no tilt trip) the free wheels spin up to the no-load speed and the stall trip can't see it
        # (review G11). A mean, because one encoder count per tick is 3.5 rad/s: at 19.4 rad/s, 4.4% of ticks read
        # under 17 and reset a per-tick counter, which delayed the trip to ~4 s (bench A5, 2026-11-23). The speed is
        # 0.85 of the no-load speed at the measured vin (runaway_speed_for_vin), not a fixed 17 rad/s.
        for q, w in zip(self._runaway, speeds):
            q.append(w)
        ql, qr = self._runaway
        if len(ql) == ql.maxlen:
            ml, mr = sum(ql) / len(ql), sum(qr) / len(qr)
            if min(abs(ml), abs(mr)) >= runaway_speed_for_vin(self.vin) and ml * mr > 0:
                return f"wheels running away (mean {ml:+.1f}, {mr:+.1f} rad/s over {lc.RUNAWAY_S} s): lifted?"
        return None


# ------------------------------------------------------------------------------------ calibration ---
def load_calibration(path):
    """{"pitch_zero": rad, "port_sign": {...}, ...} written by robot.calibrate; None if absent."""
    p = Path(path)
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def gyro_bias_from(samples):
    """samples: (pitch_rate, acc_fwd, acc_up) with the robot still. Returns (bias, sd, |a|/g mean) or
    raises ValueError if the robot moved (D4)."""
    n = len(samples)
    if n < 50:
        raise ValueError(f"only {n} IMU samples for the gyro bias")
    rates = [s[0] for s in samples]
    mean = sum(rates) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rates) / (n - 1))
    g = sum(math.hypot(s[1], s[2]) for s in samples) / n / 9.80665
    if sd > lc.GYRO_CAL_MAX_SD:
        raise ValueError(f"robot moved during gyro calibration (pitch rate sd {sd:.4f} rad/s)")
    if abs(mean) > lc.GYRO_CAL_MAX_BIAS:
        raise ValueError(f"gyro bias {mean:.4f} rad/s is implausibly large (robot turning?)")
    if abs(g - 1.0) > lc.ACCEL_G_TOLERANCE:
        raise ValueError(f"accelerometer reads {g:.3f} g in the pitch plane (robot moving or not upright-ish)")
    return mean, sd, g
