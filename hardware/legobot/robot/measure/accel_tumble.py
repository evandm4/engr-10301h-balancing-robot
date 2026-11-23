"""Accelerometer zero-g offset and scale on the forward axis (and optionally the up axis). Interactive, no motors.

    cd ~/legobot && robot/run_safe.sh robot.measure.accel_tumble
    cd ~/legobot && robot/run_safe.sh robot.measure.accel_tumble --up-axis

Near upright the tilt is theta = atan2(-acc_fwd, acc_up) ~ -acc_fwd / g, so what matters for the pitch
estimate is the zero-g offset of the FORWARD axis. Measure it by turning that axis to point straight up and
then straight down: the reading is +g + off in one and -g + off in the other, so
    offset = (reading_up + reading_down) / 2,     scale error = ((reading_up - reading_down) / 2 - g) / g,
and the pitch offset this implies is -offset / g rad. With --up-axis the same is done for the up axis
(upright, then inverted). A small misalignment (angle phi from vertical) changes each reading by about
g phi^2 / 2 and cancels in the sum to first order: 3 degrees is 1.3 mrad of tilt offset, and the script
rejects a pose more than 5 degrees off.

For each pose the script prints what to do, waits for Enter, records 3 s at 200 Hz, and checks the robot was
still: each gyro axis sd < 0.02 rad/s over the window and its mean < 0.05 rad/s (a loose bound so that an
uncorrected gyro bias does not fail the check). If not, it asks you to repeat. It averages the 3 s of accel.

WHY THIS REPLACES DATASHEET 8.4's "rotate 180 degrees about the vertical". Rotating about the vertical axis
leaves the pitch unchanged, so the two readings are equal for any offset, and their average is the surface's
tilt plus the robot's stance on its wheels, not the sensor's offset (the two are indistinguishable). Turning
the sensor axis against gravity is the only way to separate the two.

SE: sd / sqrt(n) within each 3 s window (se_kind 'window'; the 50 Hz DLPF leaves samples nearly independent at
200 Hz); the pose error (above) is not included. Files: logs/<stamp>_accel_tumble.csv (one row per pose),
_fit.json.
"""
import argparse
import csv
import math

import numpy as np

from .. import hw_config as cfg
from . import common as cm

POSES = {
    "fwd_up": ("fwd", +1, "Lay the robot on its BACK so its FRONT (the direction of travel) points straight UP at the ceiling."),
    "fwd_down": ("fwd", -1, "Lay the robot on its FACE (belly down) so its FRONT points straight DOWN at the floor, "
                            "e.g. standing on its nose against a wall, or hanging over a table edge."),
    "up_up": ("up", +1, "Stand the robot UPRIGHT, held vertical (head up, as in normal balance)."),
    "up_down": ("up", -1, "Turn the robot UPSIDE DOWN, resting on its head, held vertical."),
}
COL = {n: i for i, n in enumerate(cm.IMU_COLS)}


def measure_pose(imu, label, text, seconds):
    """Prompt, sample, check still and vertical; repeat until good. Returns dict of means and se."""
    from ..imu import G
    axis, sign, _ = POSES[label]
    main = "acc_fwd" if axis == "fwd" else "acc_up"
    while True:
        cm.ask(f"\n[{label}] {text}\nHold it still and press Enter. ")
        arr = np.array(cm.sample_imu(imu, seconds))
        gyro = arr[:, [COL["gx"], COL["gy"], COL["gz"]]]
        sd, mean = gyro.std(axis=0, ddof=1), np.abs(gyro.mean(axis=0))
        if sd.max() >= 0.02 or mean.max() >= 0.05:
            print(f"  not still (gyro sd {sd.max():.3f}, |mean| {mean.max():.3f} rad/s): repeat")
            continue
        acc = arr[:, [COL["ax"], COL["ay"], COL["az"]]].mean(axis=0)
        r = arr[:, COL[main]]
        tilt = math.degrees(math.acos(min(1.0, abs(r.mean()) / np.linalg.norm(acc))))
        if r.mean() * sign < 0:
            print(f"  the {axis} axis reads {r.mean():+.2f} m/s^2, the wrong sign for this pose: repeat")
            continue
        if tilt > 5:
            print(f"  axis is {tilt:.1f} deg off vertical (limit 5): repeat")
            continue
        print(f"  {main} = {r.mean():+.4f} m/s^2 (sd {r.std(ddof=1):.4f}), |a| = {np.linalg.norm(acc) / G:.4f} g, "
              f"{tilt:.1f} deg off vertical")
        return {"label": label, "axis": axis, "mean": float(r.mean()), "se": float(r.std(ddof=1) / math.sqrt(len(r))),
                "n": len(r), "tilt_deg": tilt, "norm_g": float(np.linalg.norm(acc) / G),
                "acc_mean": acc.tolist(), "sd": float(r.std(ddof=1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--up-axis", action="store_true", help="also measure the up axis (upright and inverted)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    from ..imu import G, Imu
    imu = Imu()
    base = cm.base_path(a.outdir, cm.stamp(), "accel_tumble")
    labels = ["fwd_up", "fwd_down"] + (["up_up", "up_down"] if a.up_axis else [])
    res = {}
    for lab in labels:
        res[lab] = measure_pose(imu, lab, POSES[lab][2], a.seconds)

    consts = {}
    print("\n=== accelerometer offsets ===")
    for axis, (lp, lm) in (("fwd", ("fwd_up", "fwd_down")), ("up", ("up_up", "up_down"))):
        if lp not in res:
            continue
        p, m = res[lp], res[lm]
        off = (p["mean"] + m["mean"]) / 2
        se_off = 0.5 * math.hypot(p["se"], m["se"])
        scale = ((p["mean"] - m["mean"]) / 2 - G) / G
        consts[f"accel_offset_{axis}"] = cm.const(off, se_off, "m/s^2", p["n"] + m["n"], "window")
        consts[f"accel_scale_err_{axis}"] = cm.const(scale, se_off / G, "fraction", p["n"] + m["n"], "window")
        print(cm.show(f"offset ({axis} axis)", consts[f"accel_offset_{axis}"]))
        print(cm.show(f"scale error ({axis})", consts[f"accel_scale_err_{axis}"]))
        if axis == "fwd":
            consts["tilt_offset_implied"] = cm.const(-off / G, se_off / G, "rad", p["n"] + m["n"], "window")
            print(cm.show("implied tilt offset", consts["tilt_offset_implied"]))
            print("  (theta_acc = -acc_fwd / g near upright: subtract this from the tilt estimate; the balance "
                  "point from balance_point.py also contains the robot's own stance)")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pose", "axis", "mean_m_s2", "se", "sd", "n", "tilt_deg", "norm_g", "ax", "ay", "az"])
        for r in res.values():
            w.writerow([r["label"], r["axis"], r["mean"], r["se"], r["sd"], r["n"], r["tilt_deg"], r["norm_g"], *r["acc_mean"]])
    meta = {"side": "n/a", "wheel": "as built", "args": vars(a), "imu_config": imu.config_readback,
            "accel_fs_sel": cfg.ACCEL_FS_SEL, "pwmparams": None, "vin": None}
    print("wrote", base + ".csv", cm.write_fit_json(base, "accel_tumble", consts, meta))
    imu.close()


if __name__ == "__main__":
    main()
