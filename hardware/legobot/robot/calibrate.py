"""Startup calibration for the balancing loop (SOFTWARE_PLAN D4). Writes ~/legobot/calibration.json.

    cd ~/legobot && ~/legobot-venv/bin/python -m robot.calibrate balance --save      # IMU only
    robot/run_safe.sh robot.calibrate signs --wheels-off --save                      # spins each wheel

balance: the pitch zero. Hold the robot on its wheels at the angle where it balances (it falls neither
forward nor back when you loosen your grip), still, for 3 s; repeated for --trials. The mean accelerometer
tilt there becomes pitch_zero, which the loop subtracts, so theta = 0 is the balance point. It absorbs the
accelerometer's own offset and the IMU's mounting angle as well; the simulator's pitch_bias (+-0.015 rad)
is what this leaves. Re-run after any change to the build.

signs: the motor and encoder signs (one motor is mirrored). Each wheel in turn gets duty +0.2 for 0.5 s;
its position, forward-positive after hw_config.PORT_SIGN, must increase, and you confirm it turned
FORWARD (the top of the tire moving toward the robot's front). Saves PORT_SIGN, which balance.py checks.

The gyro's bias is measured at every start (balance.py), and the IMU axes by robot.imu_axes.
"""
import argparse
import json
import math
import statistics as st
import time

from . import hw_config as hc
from . import loop_config as lc


def merge_save(path, updates):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data.update(updates)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {sorted(updates)} to {path}")


def hold_still(imu, seconds):
    rows = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        r = imu.read()
        rows.append((math.atan2(-r.acc_fwd, r.acc_up), r.pitch_rate))
        time.sleep(0.005)
    return rows


def balance(a):
    from .imu import Imu
    imu = Imu()
    means = []
    try:
        for k in range(a.trials):
            input(f"[{k + 1}/{a.trials}] hold the robot at its balance point, still, then press Enter...")
            rows = hold_still(imu, a.seconds)
            tilts, rates = [r[0] for r in rows], [r[1] for r in rows]
            sd_rate = st.pstdev(rates)
            print(f"  tilt {st.fmean(tilts):+.4f} rad (sd {st.pstdev(tilts):.4f}), pitch rate sd {sd_rate:.3f} rad/s")
            if sd_rate > lc.GYRO_CAL_MAX_SD:
                print("  moved too much: trial discarded")
                continue
            means.append(st.fmean(tilts))
    finally:
        imu.close()
    if not means:
        raise SystemExit("no usable trial")
    zero = st.fmean(means)
    spread = st.stdev(means) if len(means) > 1 else float("nan")
    print(f"pitch_zero {zero:+.4f} rad ({math.degrees(zero):+.2f} deg); spread across trials {spread:.4f} rad")
    if len(means) > 1 and spread > 0.015:
        print("spread is above the simulator's 0.015 rad pitch_bias range: try again more carefully")
    if a.save:
        merge_save(a.file, {"pitch_zero": zero, "pitch_zero_trials": means,
                            "pitch_zero_date": time.strftime("%Y-%m-%d %H:%M")})


def signs(a):
    if not a.wheels_off:
        raise SystemExit("spins each wheel: lift the robot so both wheels are off the ground, and pass --wheels-off")
    from .buildhat_serial import BuildHat, install_safety
    hat = BuildHat()
    install_safety(hat)
    hat.configure(plimit=hc.BENCH_PLIMIT)
    time.sleep(0.3)
    ok = True
    try:
        for side, port in (("left", hc.LEFT_PORT), ("right", hc.RIGHT_PORT)):
            p0 = hat.latest(port).pos
            hat.set_pwm(0.2 if side == "left" else 0.0, 0.2 if side == "right" else 0.0)
            time.sleep(0.5)
            hat.coast()
            time.sleep(0.4)
            moved = hat.latest(port).pos - p0
            print(f"{side}: encoder moved {moved:+d} deg (forward-positive after PORT_SIGN)")
            if moved <= 20:
                print(f"  {side} encoder does not read forward for forward duty: PORT_SIGN or wiring is wrong")
                ok = False
            answer = input(f"  did the {side.upper()} wheel turn FORWARD (top of the tire toward the front)? [y/n] ")
            if not answer.strip().lower().startswith("y"):
                print(f"  {side} drives backward: flip PORT_SIGN for port {port} in hw_config.py")
                ok = False
    finally:
        hat.close()
    if not ok:
        raise SystemExit("sign check failed: nothing saved")
    print("signs OK")
    if a.save:
        merge_save(a.file, {"port_sign": {str(k): v for k, v in hc.PORT_SIGN.items()},
                            "signs_date": time.strftime("%Y-%m-%d %H:%M")})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("balance")
    b.add_argument("--trials", type=int, default=3)
    b.add_argument("--seconds", type=float, default=3.0)
    s = sub.add_parser("signs")
    s.add_argument("--wheels-off", action="store_true")
    for p in (b, s):
        p.add_argument("--save", action="store_true")
        p.add_argument("--file", default=lc.CALIBRATION_FILE)
    a = ap.parse_args(argv)
    {"balance": balance, "signs": signs}[a.cmd](a)


if __name__ == "__main__":
    main()
