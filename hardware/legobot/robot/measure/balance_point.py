"""Pitch zero (balance point) from free falls on the wheels (DATASHEET 8.4). Interactive, motors coasting.

    cd ~/legobot && robot/run_safe.sh robot.measure.balance_point --trials 10

SETUP: the robot stands on its wheels on the floor (motors unpowered, the script coasts them), and you
catch it every time. First the gyro bias: hold the robot still for 2 s (checked: sd < 0.02 rad/s).
Then per trial: hold the robot near upright and still. When the pitch rate stays under 0.05 rad/s for
0.5 s (and the tilt is within 0.3 rad of vertical) the script prints "RELEASE" and beeps, and takes the mean
accelerometer tilt over that 0.5 s as theta_start. Let go gently. From the first sample of motion (rate
above 0.05 rad/s) the script integrates the gyro (minus the bias) from theta_start for up to 1.5 s or until
|theta| > 0.35 rad; catch the robot. Odd trials ask you to start a little behind where it seems to balance
and even trials a little ahead, so falls go both ways.

KNOWN PROBLEM (first run, 2026-11-23, DATASHEET 8.7): for falls from rest the regression below is ill-conditioned,
because in an exponential fall theta_dot stays proportional to theta - theta0, so c and e trade off and theta0
comes out with a large SE. Use the release angles and fall directions instead (the bracket in DATASHEET 8.7)
until this is replaced by a bracket estimate.

Model (fit.fit_balance): theta_ddot = c theta + e theta_dot + h, theta0 = -h / c, where theta_ddot is the
slope of the gyro rate over 5 samples (20 ms at 200 Hz; the unfiltered gyro has a ~3 ms DLPF lag that this
ignores). The angle is integrated from the accelerometer anchor only from the first motion, so the bias error
accumulates over at most 1.5 s (0.005 rad/s of residual bias is 0.007 rad at the end, mostly late in the
fall). All falls are pooled in a window |theta - median(theta_start)| < 0.25 rad and fitted once; each fall
is also fitted alone (poorly conditioned: a single short fall has a narrow theta range), and the trial-to-
trial sd of theta0 is reported next to the pooled least-squares SE.

theta is in the IMU's frame, so theta0 is directly the pitch zero the controller should use. It already
includes the accelerometer offset and the mount angle; the residual after calibration is what `pitch_bias`
stands for. The wheels roll during the fall, which the simple model ignores (it lumps their effect into
c and e).
Files: logs/<stamp>_balance_point.csv (trial, t, theta, rate, theta_ddot, theta_acc, in_fit), _fit.json.
"""
import argparse
import csv
import math
import time
from collections import deque

import numpy as np

from .. import hw_config as cfg
from . import common as cm
from . import fit

STILL_RATE = 0.05          # rad/s
STILL_SECONDS = 0.5
HOLD_TILT = 0.30           # rad: how far from vertical a "near upright" hold may be
MAX_SECONDS = 1.5
MAX_THETA = 0.35
POOL_WINDOW = 0.25


def one_trial(imu, bias, rate_hz=200.0):
    """Wait for a still hold, anchor, wait for motion, record the fall. Returns dict or None on timeout."""
    from ..imu import pitch_from_accel
    period = 1.0 / rate_hz
    buf = deque(maxlen=int(STILL_SECONDS * rate_hz))
    state, theta_start, prev, seg = "wait", None, None, []
    t0 = time.perf_counter()
    nxt = t0
    told = False
    while True:
        r = imu.read()
        rate = r.pitch_rate - bias
        th_acc = pitch_from_accel(r.acc_fwd, r.acc_up)
        if state == "wait":
            buf.append((rate, th_acc))
            if len(buf) == buf.maxlen:
                rates = np.array([b[0] for b in buf])
                ths = np.array([b[1] for b in buf])
                if np.abs(rates).max() < STILL_RATE:
                    if abs(ths.mean()) < HOLD_TILT:
                        theta_start = float(ths.mean())
                        print(f"\a  RELEASE  (held at {theta_start:+.3f} rad)")
                        state, t_arm = "armed", r.t
                    elif not told:
                        print(f"  tilt {ths.mean():+.2f} rad is too far from upright; hold it closer")
                        told = True
            if r.t - t0 > 60:
                return None
        elif state == "armed":
            if abs(rate) > STILL_RATE:
                seg = [prev, (r.t, rate, th_acc)]
                state = "fall"
                th_run = theta_start
            elif r.t - t_arm > 10:
                print("  not released within 10 s: trial skipped")
                return None
        else:
            seg.append((r.t, rate, th_acc))
            th_run += 0.5 * (seg[-1][1] + seg[-2][1]) * (seg[-1][0] - seg[-2][0])
            if r.t - seg[1][0] >= MAX_SECONDS or abs(th_run) > MAX_THETA:
                break
        prev = (r.t, rate, th_acc)
        nxt += period
        delay = nxt - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -period:
            nxt = time.perf_counter()
    a = np.array(seg)
    theta, rate_v, acc = fit.balance_segment(a[:, 0], a[:, 1], theta_start)
    return {"t": a[:, 0], "theta": theta, "rate": rate_v, "ddot": acc, "theta_acc": a[:, 2], "theta_start": theta_start}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--no-hat", action="store_true", help="do not open the Build HAT (motors then keep whatever state they were left in)")
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()

    from ..imu import Imu
    imu = Imu()
    hat = vin = None
    if not a.no_hat:
        hat, vin = cm.open_hat(a.pwmparams)
        cm.stop(hat)
        print(f"vin {vin:.2f} V; motors coasting")
    base = cm.base_path(a.outdir, cm.stamp(), "balance_point")

    # gyro bias
    while True:
        cm.ask("\nHold the robot still (propped upright is fine) and press Enter for the 2 s bias measurement. ")
        arr = np.array(cm.sample_imu(imu, 2.0))
        if arr[:, 1].std(ddof=1) < 0.02:
            bias = float(arr[:, 1].mean())
            break
        print("  not still enough (sd %.3f rad/s), repeat" % arr[:, 1].std(ddof=1))
    print(f"gyro bias {bias:+.4f} rad/s")

    trials = []
    try:
        for k in range(a.trials):
            side_txt = "a little BEHIND where it seems to balance (it should fall back)" if k % 2 == 0 else \
                       "a little AHEAD of where it seems to balance (it should fall forward)"
            cm.ask(f"\nTrial {k + 1}/{a.trials}: hold it near upright, {side_txt}. Press Enter, then hold still. ")
            tr = one_trial(imu, bias)
            if hat is not None:
                cm.check_fault(hat)
            if tr is None:
                continue
            trials.append(tr)
            print(f"  fall recorded: {len(tr['t'])} samples over {tr['t'][-1] - tr['t'][0]:.2f} s, "
                  f"theta {tr['theta'][0]:+.3f} -> {tr['theta'][-1]:+.3f} rad")
    finally:
        if hat is not None:
            try:
                cm.stop(hat)
            except Exception:
                pass

    consts, diag = {"gyro_bias_start": cm.const(bias, None, "rad/s", 400, "none")}, {}
    if len(trials) < 2:
        print("too few falls to fit")
        cm.write_fit_json(base, "balance_point", consts, {"args": vars(a)})
        return
    med = float(np.median([tr["theta_start"] for tr in trials]))
    per, pool = [], ([], [], [])
    rows_out = []
    for i, tr in enumerate(trials):
        use = np.isfinite(tr["ddot"]) & (np.abs(tr["theta"] - med) < POOL_WINDOW)
        for j, key_ in enumerate(("theta", "rate", "ddot")):
            pool[j].append(tr[key_][use])
        if use.sum() >= 30:
            try:
                f = fit.fit_balance(tr["theta"][use], tr["rate"][use], tr["ddot"][use])
                if f["c"] > 0:
                    per.append(f)
            except (ValueError, np.linalg.LinAlgError):
                pass
        rows_out += [(i + 1, *v, bool(u)) for v, u in zip(zip(tr["t"], tr["theta"], tr["rate"], tr["ddot"], tr["theta_acc"]), use)]
    with open(base + ".csv", "w", newline="") as f_:
        w = csv.writer(f_)
        w.writerow(["trial", "t", "theta", "rate", "theta_ddot", "theta_acc", "in_fit"])
        w.writerows(rows_out)
    res = fit.fit_balance(*(np.concatenate(p) for p in pool))
    n = res["n"]
    consts["theta0"] = cm.const(res["theta0"], res["se_theta0"], "rad", n, "ls")
    consts["c_fall"] = cm.const(res["c"], res["se_c"], "1/s^2", n, "ls")
    consts["e_fall"] = cm.const(res["e"], None, "1/s", n, "none")
    print("\n=== balance point ===")
    print(f"  {len(trials)} falls, median theta_start {med:+.3f} rad, {n} samples in the pooled fit ({res['n_dropped']} trimmed)")
    print(cm.show("theta0 (pooled)", consts["theta0"]))
    print(cm.show("c = theta_ddot / theta", consts["c_fall"]))
    print(f"  e = {res['e']:+.3f} 1/s, residual sd {res['resid_sd']:.2f} rad/s^2")
    if len(per) >= 2:
        t0s = np.array([p["theta0"] for p in per])
        sd = float(t0s.std(ddof=1))
        consts["theta0_trial_mean"] = cm.const(t0s.mean(), sd / math.sqrt(len(t0s)), "rad", len(t0s), "repeats")
        consts["theta0_trial_sd"] = cm.const(sd, sd / math.sqrt(2 * (len(t0s) - 1)), "rad", len(t0s), "normal")
        print(cm.show("theta0 (mean of falls)", consts["theta0_trial_mean"]))
        print(cm.show("theta0 trial-to-trial sd", consts["theta0_trial_sd"]))
        diag["per_trial_theta0"] = [p["theta0"] for p in per]
        diag["per_trial_se"] = [p["se_theta0"] for p in per]
    print("  (theta0 is in the IMU frame: use it directly as the controller's pitch zero. The part not explained by")
    print("   the accelerometer offset from accel_tumble is the robot's own stance; the residual after calibration is pitch_bias)")
    meta = {"pwmparams": a.pwmparams, "vin": vin, "side": "both", "wheel": "on the floor, motors coasting",
            "args": vars(a), "gyro_bias": bias, "median_theta_start": med}
    print("wrote", base + ".csv", cm.write_fit_json(base, "balance_point", consts, meta, diag))
    if hat is not None:
        hat.close()
    imu.close()


if __name__ == "__main__":
    main()
