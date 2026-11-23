"""Helpers shared by the measurement scripts: safety checks, one-motor drive, encoder series, fit JSON.

Not imported by fit.py (which must run on a laptop without pyserial). Run the scripts from ~/legobot:
    robot/run_safe.sh robot.measure.motor_steps --wheels-off --side left

Conventions: duties and positions are forward-positive (hw_config.PORT_SIGN is applied here); times are
time.perf_counter seconds, the same clock as the HAT driver's data_log and write_log.
"""
import json
import math
import os
import time

import numpy as np

from .. import hw_config as cfg

SIDE_PORT = {"left": cfg.LEFT_PORT, "right": cfg.RIGHT_PORT}
VIN_MIN = 6.6            # refuse to start below this (V)


# ---- names, files ----
def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def base_path(outdir, stamp_, name):
    """logs/<stamp>_<name>; the caller adds .csv, _fit.json and so on."""
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"{stamp_}_{name}")


def sides_of(choice):
    return ["left", "right"] if choice == "both" else [choice]


def key(name, side=None):
    """Constant name in the fit JSON: 'k_eff@left' for per-side constants, plain name otherwise."""
    return f"{name}@{side}" if side else name


def const(value, se, unit, n, se_kind):
    """One fitted constant as the fit JSON stores it. se_kind: 'ls' (least-squares from residuals),
    'repeats' (sd / sqrt(n) over repeated trials), or another short label explained by the script
    ('none' = a single measurement with no uncertainty estimate)."""
    def f(x):
        return None if x is None or not math.isfinite(float(x)) else float(x)
    return {"value": f(value), "se": f(se), "unit": unit, "n": int(n), "se_kind": se_kind}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_fit_json(base, script, constants, meta, diagnostics=None):
    """Write <base>_fit.json: {script, meta, constants, diagnostics}. Returns the path."""
    path = base + "_fit.json"
    doc = {"script": script, "meta": meta, "constants": constants, "diagnostics": diagnostics or {}}
    with open(path, "w") as f:
        json.dump(_clean(doc), f, indent=2)
    return path


def show(name, c):
    """One printable line for a constant dict made by const()."""
    if c["value"] is None:
        return f"  {name:<22} n/a"
    se = "no estimate" if c["se"] is None else f"{c['se']:.3g}"
    return f"  {name:<22} {c['value']:.4g} +- {se} {c['unit']}   (n={c['n']}, se: {c['se_kind']})"


def ask(prompt):
    """input() that treats end-of-input as a blank answer."""
    try:
        return input(prompt)
    except EOFError:
        return ""


def ask_float(prompt):
    """Prompt until a number or a blank line (returns None)."""
    while True:
        s = ask(prompt).strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            print("  not a number")


# ---- safety ----
def require_wheels_off(args):
    if not getattr(args, "wheels_off", False):
        raise SystemExit("Refusing to run: this drives the motors at up to the duties you give. Lift the robot so "
                         "no wheel touches anything (or remove the tires) and add --wheels-off.")


def check_duties(duties):
    if any(abs(d) > cfg.MEASURE_DUTY_CAP for d in duties):
        raise SystemExit(f"duty above {cfg.MEASURE_DUTY_CAP} refused (hw_config.MEASURE_DUTY_CAP)")


def read_vin(hat, minimum=VIN_MIN):
    """Battery voltage from the HAT (retries once); exits if unreadable or below `minimum`."""
    v = hat.vin() or hat.vin()
    if v is None:
        raise SystemExit("could not read vin from the HAT")
    if v < minimum:
        raise SystemExit(f"vin {v:.2f} V is below {minimum} V: charge the battery")
    return v


def open_hat(pwmparams=cfg.PWMPARAMS, vin_min=VIN_MIN):
    """BuildHat with the data log on, coast-on-exit installed, configured for the bench. Returns (hat, vin)."""
    from ..buildhat_serial import BuildHat, install_safety
    hat = BuildHat(log_lines=True)
    install_safety(hat)
    hat.configure(plimit=cfg.BENCH_PLIMIT, pwmparams=tuple(pwmparams))
    time.sleep(0.3)                     # let both streams start
    return hat, read_vin(hat, vin_min)


def stop(hat):
    """Zero duty on both wheels, then coast."""
    hat.set_pwm(0.0, 0.0)
    hat.coast()


def check_fault(hat):
    """Stop and exit at once if the HAT reported a fault."""
    if hat.fault:
        try:
            stop(hat)
        except Exception:
            pass
        raise SystemExit(f"HAT fault, motors coasted: {hat.fault[1]}")


def set_side(hat, side, duty):
    """Forward duty on one wheel, the other at 0 (side "both": the same duty on both)."""
    if side == "both":
        return hat.set_pwm(duty, duty)
    if side == "left":
        return hat.set_pwm(duty, 0.0)
    return hat.set_pwm(0.0, duty)


def sleep_until(t):
    """Sleep to the perf_counter time t, spinning the last 0.5 ms."""
    while True:
        dt = t - time.perf_counter()
        if dt <= 0:
            return
        time.sleep(dt - 0.0005 if dt > 0.001 else 0)


def hold_duty(hat, side, duty, seconds, period=0.01, stop_if=None):
    """Command `duty` on one wheel and re-send it every `period` s for `seconds`, checking for a fault.

    Returns (t_cmd, t_end, stopped_early): t_cmd is the start of the first write (write_log[-1][0]).
    stop_if(), if given, is called each period; a true result ends the hold early."""
    set_side(hat, side, duty)
    t_cmd = hat.write_log[-1][0]
    nxt = t_cmd
    early = False
    while time.perf_counter() - t_cmd < seconds:
        nxt += period
        sleep_until(nxt)
        check_fault(hat)
        if stop_if is not None and stop_if():
            early = True
            break
        set_side(hat, side, duty)
    return t_cmd, time.perf_counter(), early


# ---- encoder data ----
def rows_for(hat, side, since=0):
    """data_log rows (t, port, speed_raw, pos_raw, apos_raw) of one wheel, starting at list index `since`."""
    port = SIDE_PORT[side]
    return [r for r in hat.data_log[since:] if r[1] == port]


def series(rows, port):
    """(t, pos_rad) forward-positive from raw rows of one port."""
    a = np.array([(r[0], r[3]) for r in rows if r[1] == port], float)
    if a.size == 0:
        return np.array([]), np.array([])
    return a[:, 0], np.radians(cfg.PORT_SIGN[port] * a[:, 1])


def wait_rest(hat, side, quiet=0.3, timeout=3.0):
    """Wait until the wheel's speed reads 0 and its position moves at most 1 degree for `quiet` s.
    Returns True if it came to rest, False on timeout (the caller carries on)."""
    port = SIDE_PORT[side]
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        now = time.perf_counter()
        recent = []
        for r in reversed(hat.data_log):
            if r[0] < now - quiet:
                break
            if r[1] == port:
                recent.append(r)
        if len(recent) >= 0.8 * quiet * 100:
            pos = [r[3] for r in recent]
            if all(r[2] == 0 for r in recent) and max(pos) - min(pos) <= 1:
                return True
        time.sleep(0.02)
    return False


def write_raw_csv(path, rows):
    """Raw HAT stream rows (t, port, speed_raw, pos_raw, apos_raw), RAW HAT signs (multiply by
    hw_config.PORT_SIGN[port] for forward-positive)."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "port", "speed_raw", "pos_raw", "apos_raw"])
        w.writerows(rows)


# ---- IMU ----
IMU_COLS = ["t", "pitch_rate", "acc_fwd", "acc_up", "theta_acc", "gx", "gy", "gz", "ax", "ay", "az"]


def imu_row(r):
    """One ImuReading as a flat row matching IMU_COLS (tilt from the accelerometer included)."""
    from ..imu import pitch_from_accel
    return (r.t, r.pitch_rate, r.acc_fwd, r.acc_up, pitch_from_accel(r.acc_fwd, r.acc_up), *r.gyro, *r.acc)


def sample_imu(imu, seconds, rate=200.0, tick=None):
    """Read the IMU on a fixed-period perf_counter loop (like bench_log). Returns a list of IMU_COLS rows.

    tick(n), if given, is called after each read with the sample index (used to re-send motor duties)."""
    period = 1.0 / rate
    out = []
    t_start = time.perf_counter()
    nxt = t_start
    n = 0
    while time.perf_counter() - t_start < seconds:
        out.append(imu_row(imu.read()))
        if tick is not None:
            tick(n)
        n += 1
        nxt += period
        delay = nxt - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -period:
            nxt = time.perf_counter()
    return out
