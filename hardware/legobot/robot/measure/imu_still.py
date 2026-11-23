"""Gyro and accelerometer noise and bias with the robot still, motors off and then on (DATASHEET 8.4).

    cd ~/legobot && robot/run_safe.sh robot.measure.imu_still --wheels-off --seconds 30
    cd ~/legobot && robot/run_safe.sh robot.measure.imu_still --seconds 30 --motor-duties     # motors off only

Robot on a stand with the wheels off the ground and the body not touched. Phase 1: motors off for --seconds
(default 30), the IMU read at 200 Hz. Then, unless --motor-duties is given with no values, both wheels at
each duty in --motor-duties (default 0.3 0.6 1.0) for 10 s each; that needs --wheels-off.

Per phase: mean and sd of the pitch rate (the mean is the gyro bias), sd of all three chip-frame gyro axes,
and mean and sd of the accelerometer tilt (pitch_from_accel). For the motors-off phase the Allan deviation of
the pitch rate at tau = 0.005, 0.05, 0.5 and 5 s. Motor phases drop their first 1.5 s (spin-up) from the
statistics but keep them in the CSV. The motors-on increase in sd is reported both as the difference and in
quadrature, sqrt(sd_on^2 - sd_off^2), the added vibration noise.

Standard errors: the SE of a mean comes from the spread of six block means (se_kind 'repeats'); the SE of an
sd assumes independent samples, sd / sqrt(2 (n - 1)) (se_kind 'normal'); Allan values carry the usual
adev / sqrt(2 (N/m - 1)) (se_kind 'allan'). Sample times are not exactly uniform: fs for the Allan
deviation is the mean rate.
Files: logs/<stamp>_imu_still.csv (phase, t, pitch_rate, acc_fwd, acc_up, theta_acc, gx, gy, gz, ax, ay, az),
_fit.json.
"""
import argparse
import csv
import math
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm
from . import fit

TAUS = (0.005, 0.05, 0.5, 5.0)
SETTLE = 1.5        # s dropped from the start of each motor phase
BLOCKS = 6


def mean_se(x):
    """Mean and the SE from the spread of BLOCKS equal block means."""
    b = np.array_split(np.asarray(x, float), BLOCKS)
    means = np.array([v.mean() for v in b])
    return float(np.mean(x)), float(means.std(ddof=1) / math.sqrt(BLOCKS))


def sd_se(x):
    return float(np.std(x, ddof=1)), float(np.std(x, ddof=1) / math.sqrt(2 * (len(x) - 1)))


def stats(arr):
    """arr columns follow common.IMU_COLS. Returns dict of (value, se) pairs."""
    c = {n: arr[:, i] for i, n in enumerate(cm.IMU_COLS)}
    out = {"rate_mean": mean_se(c["pitch_rate"]), "rate_sd": sd_se(c["pitch_rate"]),
           "theta_acc_mean": mean_se(c["theta_acc"]), "theta_acc_sd": sd_se(c["theta_acc"])}
    for ax in ("gx", "gy", "gz"):
        out[ax + "_sd"] = sd_se(c[ax])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required for the motor phases")
    ap.add_argument("--seconds", type=float, default=30.0, help="motors-off phase")
    ap.add_argument("--motor-duties", type=float, nargs="*", default=[0.3, 0.6, 1.0])
    ap.add_argument("--motor-seconds", type=float, default=10.0)
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    if a.motor_duties:
        cm.require_wheels_off(a)
        cm.check_duties(a.motor_duties)

    from ..imu import Imu
    imu = Imu()
    hat = vin = None
    if a.motor_duties:
        hat, vin = cm.open_hat(a.pwmparams)
        hat.coast()
    base = cm.base_path(a.outdir, cm.stamp(), "imu_still")
    phases = {}            # name -> rows
    try:
        print(f"phase off: {a.seconds:g} s, motors off, do not touch the robot")
        phases["off"] = cm.sample_imu(imu, a.seconds)
        for d in a.motor_duties:
            hat.coast()
            time.sleep(2.0)

            def tick(n, d=d):
                if n % 2 == 0:
                    hat.set_pwm(d, d)
                    cm.check_fault(hat)
            print(f"phase duty {d:g}: both wheels, {a.motor_seconds:g} s")
            phases[f"duty_{d:g}"] = cm.sample_imu(imu, a.motor_seconds, tick=tick)
            cm.stop(hat)
    finally:
        if hat is not None:
            try:
                cm.stop(hat)
            except Exception:
                pass
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase"] + cm.IMU_COLS)
            for name, rows in phases.items():
                w.writerows((name,) + tuple(r) for r in rows)

    consts, diag = {}, {}
    off = np.array(phases["off"])
    s_off = stats(off)
    fs = (len(off) - 1) / (off[-1, 0] - off[0, 0])
    print(f"\nmotors off: {len(off)} samples, {fs:.1f} Hz")
    for name, unit in (("rate_mean", "rad/s"), ("rate_sd", "rad/s"), ("theta_acc_mean", "rad"), ("theta_acc_sd", "rad"),
                       ("gx_sd", "rad/s"), ("gy_sd", "rad/s"), ("gz_sd", "rad/s")):
        v, se = s_off[name]
        label = {"rate_mean": "gyro_bias", "rate_sd": "gyro_pitch_sd"}.get(name, name)
        kind = "repeats" if name.endswith("mean") else "normal"
        consts[label] = cm.const(v, se, unit, len(off), kind)
        print(cm.show(label, consts[label]))
    taus, ad = fit.allan_dev(off[:, 1], fs, TAUS)
    for tau, v in zip(taus, ad):
        m = round(tau * fs)
        n_avg = len(off) / m
        se = v / math.sqrt(2 * (n_avg - 1)) if n_avg > 2 else float("nan")
        name = f"adev_{tau:g}s"
        consts[name] = cm.const(v, se, "rad/s", len(off), "allan")
        print(cm.show(name, consts[name]))
    sd_off = s_off["rate_sd"][0]
    for name, rows in phases.items():
        if name == "off":
            continue
        arr = np.array(rows)
        arr = arr[arr[:, 0] >= arr[0, 0] + SETTLE]
        s_on = stats(arr)
        sd_on = s_on["rate_sd"][0]
        extra = math.sqrt(max(sd_on**2 - sd_off**2, 0.0))
        consts[f"gyro_sd_{name}"] = cm.const(sd_on, s_on["rate_sd"][1], "rad/s", len(arr), "normal")
        consts[f"gyro_bias_{name}"] = cm.const(s_on["rate_mean"][0], s_on["rate_mean"][1], "rad/s", len(arr), "repeats")
        consts[f"gyro_sd_added_{name}"] = cm.const(extra, None, "rad/s", len(arr), "none")
        print(f"{name}: pitch-rate sd {sd_on:.4f} rad/s (off {sd_off:.4f}; +{sd_on - sd_off:.4f}, "
              f"added in quadrature {extra:.4f}); bias {s_on['rate_mean'][0]:+.4f} rad/s "
              f"(shift {s_on['rate_mean'][0] - s_off['rate_mean'][0]:+.4f}); tilt sd {s_on['theta_acc_sd'][0]:.4f} rad")
    meta = {"pwmparams": a.pwmparams, "vin": vin, "side": "both", "wheel": "off the ground", "args": vars(a),
            "imu_config": imu.config_readback, "sample_rate_hz": fs}
    print("wrote", base + ".csv", cm.write_fit_json(base, "imu_still", consts, meta, diag))
    if hat is not None:
        hat.close()
    imu.close()


if __name__ == "__main__":
    main()
