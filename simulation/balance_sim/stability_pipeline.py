"""Linear stability of the LEGO loop WITH its sensor pipeline (the model the policies are trained in).

`stability.py` linearizes the loop with ideal sensing: the controller sees the true (v, theta, theta_dot),
torque = command, a pure latency and a 5 ms hold. The LEGO robot (legobot_config: SensorConfig model "imu" with
encoder, ActuatorConfig "slow_pwm") is simulated and trained with a complementary filter, a delayed and
differenced encoder, the Build HAT's 10 ms pulse-density modulation and back-EMF feedforward. Encoder lag in
particular changes the delay margins substantially (STABILITY_PIPELINE.md): with ideal sensing L's cruise margin
is 12.25 ms, with the sim's pipeline 29-35 ms depending on the PDM tick phase, and ~21-31 ms with hardware-like
timing (below).

This module is that loop, linearized:

  * complementary filter (sensors.py IMUSensor): est <- a (est + T gyro) + (1-a) tilt, a = tau/(tau+T),
    tilt = theta - (x_dd + h theta_dd)/g (accelerometer_tilt linearized), with the accelerations from the state
    and the drive applied in the physics step before the sample (simulator.py);
  * encoder (sensors.py Encoder): the relative wheel angle sampled every 10 ms from t = 0, available 5 ms later,
    speed = difference of the two latest / 10 ms, v_meas = r (encoder speed + gyro); quantization dropped;
  * Build HAT PDM (simulator.py, actuators.pdm_update) linearized as its signal transfer: a sample-and-hold of
    the drive every 10 ms at a phase offset (quantization noise dropped);
  * back-EMF feedforward from the measured wheel speed (SlowPwmActuator.drive) linearized about the operating
    point: d tau = d tau_cmd + c_ff d w_hat(held) - c_ff d w(t)  (c_ff = 0 at standstill);
  * latency: a command takes effect `latency` after its tick (rounded up to the 0.25 ms sub-step; with
    sim_rounding=True to the 2.5 ms physics step, as simulate() does).

Hardware timing the sim does not model (reports/delay_margin_check/review/REVIEW.md, section 1), all off by
default so the defaults are the sim's loop:

  * imu_delay_sub: an IMU-only pure delay (the ICM's ~3 ms DLPF) on the gyro and the accelerometer, so on the
    complementary filter and the gyro term of v_meas but NOT on the encoder path. The sim lumps the DLPF into
    `latency`, which also delays the encoder; with imu_delay_sub > 0, `latency` is the actuation part only and
    the comparable "total" (sample -> HAT, the sim's TimingConfig quantity) is latency + imu delay.
  * right_enc_age_sub: a separate right encoder, sampled on its own 10 ms grid; its sample age at the tick where
    the left one's new sample becomes available (the left's age is enc_delay_sub). v_meas and the feedforward
    use the mean of the two wheel speeds, as control.py does.
  * right_lag_sub: extra actuation lag of the right wheel's half of the drive (command-string order on the HAT).

The loop is periodic with period 10 ms (lcm of the 5 ms control period, the 10 ms encoder and the 10 ms PDM
tick); it is discretized exactly on a 0.25 ms grid and the monodromy matrix over one period gives stability
(rho10 < 1). rho_tick = sqrt(rho10) is comparable with stability.spectral_radius's per-5 ms rho, and with every
stage switched off (Pipeline.ideal()) the model reproduces stability.py to round-off (tests).

The PDM phase is free on the hardware (ActuatorConfig.pwm_tick_locked = False), so quote margins over the
phase: margin(..., pdm_phases=range(0, 40, 4)) is the worst case over 1 ms steps; margin_table() does this for
the three sensing models. Not modelled: backlash (a ~4 Hz cruise limit cycle for L in the nonlinear sim), PDM
quantization, saturation, noise.
"""
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy.linalg import expm

from .planar2d import linearize

SUB = 0.00025                  # s, sub-step
PERIOD_SUB = 40                # 10 ms
TICK_SUB = 20                  # 5 ms control period


@dataclass(frozen=True)
class Pipeline:
    cf: bool = True                 # pitch from the complementary filter (else true pitch)
    accel: bool = True              # accelerometer sees x_dd + h theta_dd (only with cf)
    encoder: bool = True            # velocity from the delayed, differenced encoder (else true v)
    pdm: bool = True                # 10 ms sample-and-hold of the drive (PDM signal transfer)
    pdm_offset_sub: int = 10        # sub-steps from an even control tick to a PDM tick (sim: floor(phase*4)*10)
    c_ff: float = 0.0               # N m s/rad, back-EMF feedforward linear gain (cruise)
    tau_cf: float = 0.5
    imu_height: float = 0.10
    gravity: float = 9.81
    enc_delay_sub: int = 20         # 5 ms
    control_dt: float = 0.005
    imu_delay_sub: int = 0          # IMU-only delay (DLPF) on gyro + accelerometer, 0..19 sub-steps
    right_enc_age_sub: Optional[int] = None   # separate right encoder: its age (1..39 sub-steps); None = one encoder
    right_lag_sub: int = 0          # extra actuation lag of the right wheel's half of the drive

    @staticmethod
    def ideal():
        """Every stage off: the loop stability.py analyses."""
        return Pipeline(cf=False, accel=False, encoder=False, pdm=False, c_ff=0.0)

    @staticmethod
    def from_config(cfg, c_ff=0.0, pdm_phase=None):
        """The pipeline a RobotConfig simulates. Only the LEGO structure (10 ms encoder and PDM, 5 ms control)
        is supported; anything else raises."""
        s, a, t = cfg.sensor, cfg.actuator, cfg.timing
        if abs(t.control_dt_s - TICK_SUB * SUB) > 1e-12:
            raise NotImplementedError("stability_pipeline assumes a 5 ms control period")
        cf = s.model == "imu"
        enc = bool(getattr(s, "encoder_counts", 0))
        if enc and (abs(s.encoder_period_s - PERIOD_SUB * SUB) > 1e-12 or s.encoder_delay_s % SUB > 1e-12):
            raise NotImplementedError("stability_pipeline assumes a 10 ms encoder with a 0.25 ms-grid delay")
        pdm = a.model == "slow_pwm" and a.pwm_tick_s > 0
        if pdm and abs(a.pwm_tick_s - PERIOD_SUB * SUB) > 1e-12:
            raise NotImplementedError("stability_pipeline assumes a 10 ms PDM tick")
        phase = a.pwm_tick_phase if pdm_phase is None else pdm_phase
        off = int(math.floor(phase * (PERIOD_SUB * SUB) / t.dt_s + 1e-9)) * int(round(t.dt_s / SUB))
        return Pipeline(cf=cf, accel=cf, encoder=enc, pdm=pdm, pdm_offset_sub=off, c_ff=c_ff,
                        tau_cf=s.filter_time_constant if cf else 0.5, imu_height=s.imu_height if cf else 0.10,
                        enc_delay_sub=int(round(s.encoder_delay_s / SUB)) if enc else 20,
                        control_dt=t.control_dt_s)

    @staticmethod
    def hardware_like(cfg, c_ff=0.0, pdm_phase=None, imu_delay=0.003, right_lag=0.0025, right_enc_age=None):
        """from_config plus the hardware timing of REVIEW.md (1a-c): the ~3 ms DLPF on the IMU only, the right
        wheel acting 2.5 ms late, and optionally the right encoder's own sample age (s; its per-run phase is
        arbitrary on the robot, so None keeps it equal to the left's). With imu_delay > 0, `latency` is the
        actuation part only; the total comparable with the sim's latency is latency + imu_delay."""
        return replace(Pipeline.from_config(cfg, c_ff=c_ff, pdm_phase=pdm_phase),
                       imu_delay_sub=_to_sub(imu_delay), right_lag_sub=_to_sub(right_lag),
                       right_enc_age_sub=None if right_enc_age is None else _to_sub(right_enc_age))

    @property
    def imu_delay(self):
        return self.imu_delay_sub * SUB


def _to_sub(seconds):
    n = seconds / SUB
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"{seconds} s is not on the {SUB * 1e3} ms grid")
    return int(round(n))


def feedforward_gain(cfg, params, speed):
    """c_ff at a cruise speed (m/s): d torque / d measured wheel speed of the configured actuator (drive from the
    robot code's estimate, torque from the true motor), by central differences about the steady drive torque.
    0 for actuators without back-EMF feedforward. Pass the full robot params (with its Coulomb friction, which
    sets the cruise torque), not the cruise linearization's copy with coulomb_friction = 0."""
    act = cfg.build_actuator()
    if not hasattr(act, "drive") or speed == 0.0:
        return 0.0
    w0 = speed / params.wheel_radius
    tau0 = params.wheel_friction * w0 + params.coulomb_friction * math.tanh(w0 / params.coulomb_friction_speed)
    h = 1e-4

    def T(w_hat):
        return act.torque(act.drive(tau0, w_hat), w0)

    return (T(w0 + h) - T(w0 - h)) / (2 * h)


@lru_cache(maxsize=256)
def _sub_maps(params, c_ff):
    """Plant [v, th, thd, acc] (acc = wheel-angle accumulator for the encoder) with the extra back-EMF damping
    c_ff on the wheel speed, discretized exactly over SUB with the input held; plus the continuous A, B rows
    for the accelerometer."""
    p = replace(params, wheel_friction=params.wheel_friction + c_ff)
    A4, B4 = linearize(p)
    A, B = A4[1:, 1:], B4[1:, 0]
    r = params.wheel_radius
    Aa = np.zeros((4, 4))
    Aa[:3, :3] = A
    Aa[3, 0] = 1.0 / r
    Aa[3, 2] = -1.0
    Ba = np.concatenate([B, [0.0]])
    blk = np.zeros((5, 5))
    blk[:4, :4] = Aa
    blk[:4, 4] = Ba
    E = expm(blk * SUB)
    return E[:4, :4], E[:4, 4], A, B


class Model:
    """The periodic linear loop. `controller` is a stability.LinearController."""

    def __init__(self, params, controller, latency, pipe: Pipeline, sim_rounding=False):
        self.params, self.ctrl, self.pipe = params, controller, pipe
        if not 0 <= pipe.imu_delay_sub < TICK_SUB:
            raise ValueError("imu_delay_sub must be in [0, 20) sub-steps (under one control period)")
        if pipe.right_enc_age_sub is not None:
            if not pipe.encoder:
                raise ValueError("right_enc_age_sub needs the encoder")
            if not 0 < pipe.right_enc_age_sub < PERIOD_SUB:
                raise ValueError("right_enc_age_sub must be in (0, 40) sub-steps")
        if pipe.right_lag_sub < 0:
            raise ValueError("right_lag_sub must be >= 0")
        lat_r = latency + pipe.right_lag_sub * SUB
        if sim_rounding:      # simulate() applies a command at the first 2.5 ms step at/after it is due
            latency = math.ceil(latency / 0.0025 - 1e-9) * 0.0025
            lat_r = math.ceil(lat_r / 0.0025 - 1e-9) * 0.0025
        self.latency = latency
        self.Ls = int(math.ceil(latency / SUB - 1e-9))
        self.split = pipe.right_lag_sub > 0               # right wheel's drive on its own delay tap
        self.LsR = int(math.ceil(lat_r / SUB - 1e-9)) if self.split else self.Ls
        self.renc = pipe.right_enc_age_sub is not None
        self.imu_d = pipe.imu_delay_sub
        self.ints = controller.integrators()
        self.H = len(controller.history_gain)
        self.Q = max(self.Ls, self.LsR) // TICK_SUB + 1
        names = ["v", "th", "thd", "acc"]
        if self.renc:
            names += ["accR"]
        if pipe.encoder:
            names += ["pend", "spd"]
        if self.renc:
            names += ["pendR", "spdR"]
        if pipe.cf:
            names += ["est"]
        if self.imu_d:
            names += ["sv", "sth", "sthd", "sap"]         # IMU snapshot (plant + applied drive) imu_d before a tick
        names += [f"I{m}" for m in range(len(self.ints))]
        names += [f"u{j}" for j in range(1, self.H + 1)]
        names += [f"q{j}" for j in range(self.Q)]
        names += ["D"]
        if self.split:
            names += ["DR"]
        if pipe.pdm:
            names += ["P"]
            if self.split:
                names += ["PR"]
        self.names = names
        self.ix = {n: i for i, n in enumerate(names)}
        self.n = len(names)

    def _I(self):
        return np.eye(self.n)

    def _e(self, *names, w=1.0):
        row = np.zeros(self.n)
        for nm in names:
            row[self.ix[nm]] += w
        return row

    def _applied_row(self):
        """The drive reaching the motors now (mean of the two wheels' when they are split)."""
        held = ("P", "PR") if self.pipe.pdm else ("D", "DR")
        return self._e(*held, w=0.5) if self.split else self._e(held[0])

    def _obs_rows(self):
        """Measured (v_meas, th_meas, gyro) as rows on z at a control tick; th_meas is the CF output after this
        tick's update (est_new), which also replaces the est row."""
        ix, p, pipe = self.ix, self.params, self.pipe
        n = self.n
        imu_src = ["sv", "sth", "sthd"] if self.imu_d else ["v", "th", "thd"]     # what the IMU sees
        gyro = self._e(imu_src[2])
        spd = self._e("spd", "spdR", w=0.5) if self.renc else (self._e("spd") if pipe.encoder else None)
        if pipe.encoder:
            v_meas = p.wheel_radius * (gyro + spd)
        else:
            v_meas = self._e("v")
        if pipe.cf:
            _, _, A, B = _sub_maps(p, pipe.c_ff)
            tilt = self._e(imu_src[1])
            if pipe.accel:
                applied = self._e("sap") if self.imu_d else self._applied_row()
                xdd = np.zeros(n); xdd[[ix[s] for s in imu_src]] = A[0]; xdd += B[0] * applied
                thdd = np.zeros(n); thdd[[ix[s] for s in imu_src]] = A[2]; thdd += B[2] * applied
                tilt = tilt - (xdd + pipe.imu_height * thdd) / pipe.gravity
            a = pipe.tau_cf / (pipe.tau_cf + pipe.control_dt)
            est_e = self._e("est")
            est_new = a * (est_e + pipe.control_dt * gyro) + (1 - a) * tilt
            th_meas = est_new
        else:
            th_meas = self._e(imu_src[1])
            est_new = None
        if pipe.encoder:
            w_hat = spd
        else:
            w_hat = v_meas / p.wheel_radius - gyro
        return v_meas, th_meas, gyro, est_new, w_hat

    def _tick_map(self):
        ix, c = self.ix, self.ctrl
        v_meas, th_meas, gyro, est_new, w_hat = self._obs_rows()
        S = np.vstack([v_meas, th_meas, gyro])
        u = np.asarray(c.state_gain, float) @ S
        for m, (g, _, _) in enumerate(self.ints):
            u = u.copy(); u[ix[f"I{m}"]] += g
        for j, g in enumerate(c.history_gain, start=1):
            u = u.copy(); u[ix[f"u{j}"]] += g
        M = self._I()
        if est_new is not None:
            M[ix["est"]] = est_new
        for m, (_, integrand, decay) in enumerate(self.ints):
            row = self.pipe.control_dt * (np.asarray(integrand, float) @ S)
            row[ix[f"I{m}"]] += decay
            M[ix[f"I{m}"]] = row
        if self.H:
            for j in range(self.H, 1, -1):
                M[ix[f"u{j}"]] = np.eye(self.n)[ix[f"u{j-1}"]]
            M[ix["u1"]] = u
        q = u + self.pipe.c_ff * w_hat
        for j in range(self.Q - 1, 0, -1):
            M[ix[f"q{j}"]] = np.eye(self.n)[ix[f"q{j-1}"]]
        M[ix["q0"]] = q
        return M

    def _latch(self, dst, src, scale=1.0):
        L = self._I(); L[self.ix[dst]] = 0.0; L[self.ix[dst], self.ix[src]] = scale
        return L

    def sub_maps(self):
        """The 40 per-sub-step maps (events, then the propagation), in order."""
        ix, pipe = self.ix, self.pipe
        Ad, Bd, _, _ = _sub_maps(self.params, pipe.c_ff)
        plant = [ix["v"], ix["th"], ix["thd"], ix["acc"]]
        Prop = self._I()
        Prop[np.ix_(plant, plant)] = Ad
        Prop[plant] += np.outer(Bd, self._applied_row())
        if self.renc:                                   # right accumulator: the same wheel angle, own reset
            row = Prop[ix["acc"]].copy()
            row[ix["accR"]], row[ix["acc"]] = row[ix["acc"]], 0.0
            Prop[ix["accR"]] = row
        tick = self._tick_map()
        enc_avail = pipe.enc_delay_sub % PERIOD_SUB
        r_sample = (pipe.enc_delay_sub - pipe.right_enc_age_sub) % PERIOD_SUB if self.renc else None
        r_avail = (r_sample + pipe.right_enc_age_sub) % PERIOD_SUB if self.renc else None
        maps = []
        for s in range(PERIOD_SUB):
            Mtot = self._I()
            if self.imu_d and (s + self.imu_d) % TICK_SUB == 0:   # IMU sample the next tick will use
                L = self._I()
                for dst, src in (("sv", "v"), ("sth", "th"), ("sthd", "thd")):
                    L[ix[dst]] = 0.0; L[ix[dst], ix[src]] = 1.0
                L[ix["sap"]] = self._applied_row()
                Mtot = L @ Mtot
            if s % PERIOD_SUB == 0:           # encoder sample: latch the increment, reset the accumulator
                L = self._I()
                if pipe.encoder:
                    L[ix["pend"]] = 0.0; L[ix["pend"], ix["acc"]] = 1.0
                L[ix["acc"]] = 0.0
                Mtot = L @ Mtot
            if self.renc and s == r_sample:
                L = self._latch("pendR", "accR"); L[ix["accR"]] = 0.0
                Mtot = L @ Mtot
            if pipe.encoder and s == enc_avail:
                Mtot = self._latch("spd", "pend", 1.0 / (PERIOD_SUB * SUB)) @ Mtot
            if self.renc and s == r_avail:
                Mtot = self._latch("spdR", "pendR", 1.0 / (PERIOD_SUB * SUB)) @ Mtot
            if s % TICK_SUB == 0:
                Mtot = tick @ Mtot
            if (s - self.Ls) % TICK_SUB == 0:
                Mtot = self._latch("D", f"q{self.Ls // TICK_SUB}") @ Mtot
            if self.split and (s - self.LsR) % TICK_SUB == 0:
                Mtot = self._latch("DR", f"q{self.LsR // TICK_SUB}") @ Mtot
            if pipe.pdm and (s - pipe.pdm_offset_sub) % PERIOD_SUB == 0:
                Mtot = self._latch("P", "D") @ Mtot
                if self.split:
                    Mtot = self._latch("PR", "DR") @ Mtot
            maps.append(Prop @ Mtot)
        return maps

    def monodromy(self):
        Mtot = self._I()
        for m in self.sub_maps():
            Mtot = m @ Mtot
        return Mtot

    def trajectory(self, z0, n_periods):
        """z at every sub-step boundary, shape (n_periods*40 + 1, n)."""
        maps = self.sub_maps()
        z = np.array(z0, float)
        out = [z.copy()]
        for _ in range(n_periods):
            for m in maps:
                z = m @ z
                out.append(z.copy())
        return np.array(out)

    def critical(self):
        """(rho10, rho_tick, frequency in Hz of the dominant eigenvalue, all eigenvalues)."""
        ev = np.linalg.eigvals(self.monodromy())
        i = int(np.argmax(np.abs(ev)))
        rho10 = float(abs(ev[i]))
        return rho10, math.sqrt(rho10), float(abs(np.angle(ev[i])) / (2 * math.pi * PERIOD_SUB * SUB)), ev


def rho_tick(params, ctrl, latency, pipe, sim_rounding=False):
    return Model(params, ctrl, latency, pipe, sim_rounding).critical()[1]


def margin(params, ctrl, pipe, max_latency=0.08, step=0.00025, sim_rounding=False, pdm_phases=None):
    """Largest latency L such that the loop is stable on the whole grid [0, L] (stability.stable_latency_range
    with around = 0). With pdm_phases (PDM offsets in sub-steps), stability must hold for every one of them.
    `latency` is the actuation delay; with pipe.imu_delay_sub > 0 add pipe.imu_delay for the sim-comparable
    total. None if unstable at 0."""
    phases = [pipe.pdm_offset_sub] if (pdm_phases is None or not pipe.pdm) else list(pdm_phases)
    grid = np.round(np.arange(0.0, max_latency + step / 2, step), 10)

    def stable(lat):
        return all(rho_tick(params, ctrl, lat, replace(pipe, pdm_offset_sub=o), sim_rounding) < 1.0
                   for o in phases)

    if not stable(0.0):
        return None
    hi = 0
    while hi + 1 < len(grid) and stable(grid[hi + 1]):
        hi += 1
    return float(grid[hi])


# ----------------------------------------------------------------------------------------------- margin table

def cruise_params(params):
    """The cruise linearization's robot: the tanh Coulomb term is saturated at 0.40 m/s (slope ~4e-12)."""
    return replace(params, coulomb_friction=0.0)


def corner_params(params):
    """The +8 % COM height / +8 % rotor inertia corner used throughout the delay-margin reviews."""
    return replace(params, body_com_height=params.body_com_height * 1.08,
                   rotor_inertia=params.rotor_inertia * 1.08)


def sensing_models(cfg, full_params, speed, imu_delay=0.003, right_lag=0.0025, right_enc_age=None):
    """{name: Pipeline} for the three sensing models of margin_table at a speed (c_ff from the full robot)."""
    c_ff = feedforward_gain(cfg, full_params, speed)
    return {"ideal": Pipeline.ideal(),
            "sim": Pipeline.from_config(cfg, c_ff=c_ff),
            "hw": Pipeline.hardware_like(cfg, c_ff=c_ff, imu_delay=imu_delay, right_lag=right_lag,
                                         right_enc_age=right_enc_age)}


def margin_table(cfg, controllers, base_params, speeds=(("standstill", 0.0), ("cruise", 0.40)),
                 pdm_phases=range(0, 40, 4), max_latency=0.08, **hw):
    """Rows {controller, speed, robot, model, worst, best, worst_phase} of TOTAL margins in seconds (latency +
    IMU-only delay; the sim's sample -> HAT quantity), worst/best over the PDM phases (one value for ideal
    sensing, which has no PDM). None = unstable at every latency. `hw` goes to sensing_models."""
    rows = []
    for sname, speed in speeds:
        for rname, full in (("nominal", base_params), ("corner", corner_params(base_params))):
            lin = cruise_params(full) if speed else full
            for mname, pipe in sensing_models(cfg, full, speed, **hw).items():
                for cname, ctrl in controllers.items():
                    phases = list(pdm_phases) if pipe.pdm else [pipe.pdm_offset_sub]
                    per = {o: margin(lin, ctrl, replace(pipe, pdm_offset_sub=o), max_latency=max_latency)
                           for o in phases}
                    tot = {o: (None if m is None else m + pipe.imu_delay) for o, m in per.items()}
                    vals = [v for v in tot.values() if v is not None]
                    worst_o = min(tot, key=lambda o: -1.0 if tot[o] is None else tot[o])
                    rows.append(dict(controller=cname, speed=sname, robot=rname, model=mname,
                                     worst=None if len(vals) < len(tot) else min(vals),
                                     best=max(vals) if vals else None, worst_phase_sub=worst_o,
                                     max_latency=max_latency + pipe.imu_delay))
    return rows


def format_table(rows):
    """Markdown table of margin_table rows, in ms."""
    def cell(r):
        def f(x):
            if x is None:
                return "unst."
            return f"{x * 1e3:.2f}".rstrip("0").rstrip(".") + ("+" if x >= r["max_latency"] - 1e-9 else "")
        if r["worst"] == r["best"]:
            return f(r["worst"])
        return f"{f(r['worst'])}-{f(r['best'])}"
    models = list(dict.fromkeys(r["model"] for r in rows))
    heads = {"ideal": "ideal (stability.py)", "sim": "sim pipeline", "hw": "hardware-like"}
    out = ["| controller | condition | " + " | ".join(heads.get(m, m) for m in models) + " |",
           "|---|---|" + "---|" * len(models)]
    keys = list(dict.fromkeys((r["controller"], r["speed"], r["robot"]) for r in rows))
    for k in sorted(keys, key=lambda k: (k[0] != "L", k[1] != "standstill", k[2] != "nominal")):
        cells = [cell(next(r for r in rows if (r["controller"], r["speed"], r["robot"]) == k and r["model"] == m))
                 for m in models]
        out.append(f"| {k[0]} | {k[1]} {k[2]} | " + " | ".join(cells) + " |")
    return "\n".join(out)
