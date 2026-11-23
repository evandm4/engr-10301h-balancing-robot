"""Find the IMU's pitch axis and signs by tilting the robot about its axle (C2; DATASHEET 3.4).

    cd ~/legobot && ~/legobot-venv/bin/python -m robot.imu_axes
Start with the robot upright and still (first 1.5 s); then, any time within 90 s, lean it FORWARD
about 30-45 degrees about the axle, hold, and bring it back upright. Prints the mapping to put in
hw_config.py (IMU_PITCH_RATE, IMU_ACCEL_FORWARD, IMU_ACCEL_UP) and checks it against the
current one. Motors are not touched.
"""
import math
import time

from .imu import Imu


def main(timeout=90.0):
    imu = Imu()
    rows = []
    print("hold upright and still...", flush=True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.5:
        rows.append(imu.read_raw())
        time.sleep(0.005)
    g0 = [sum(r[1][i] for r in rows) / len(rows) for i in range(3)]
    up0 = max(range(3), key=lambda i: abs(g0[i]))
    print(f"now lean it FORWARD about 30-45 deg, hold, and return upright (waiting {timeout:.0f} s)", flush=True)
    tilted_at = None
    while time.perf_counter() - t0 < timeout:
        r = imu.read_raw()
        rows.append(r)
        c = r[1][up0] / g0[up0]               # cos(tilt), roughly
        if tilted_at is None and c < 0.94:    # past about 20 degrees
            tilted_at = r[0]
        if tilted_at is not None and (c > 0.985 and r[0] - tilted_at > 0.5 or r[0] - tilted_at > 5):
            break
        time.sleep(0.005)
    imu.close()
    if tilted_at is None:
        raise SystemExit("no tilt seen")

    still = [r for r in rows if r[0] - rows[0][0] < 1.5]
    acc0 = [sum(r[1][i] for r in still) / len(still) for i in range(3)]
    bias = [sum(r[2][i] for r in still) / len(still) for i in range(3)]
    # Up axis: carries gravity when upright.
    up = max(range(3), key=lambda i: abs(acc0[i]))
    up_sign = 1 if acc0[up] > 0 else -1
    # Most-forward-leaning sample: where the up axis reading is smallest.
    lean = min(rows, key=lambda r: up_sign * r[1][up])
    others = [i for i in range(3) if i != up]
    fwd = max(others, key=lambda i: abs(lean[1][i] - acc0[i]))
    # Leaning forward, the specific force along "forward" is -g sin(theta) < 0.
    fwd_sign = -1 if lean[1][fwd] - acc0[fwd] > 0 else 1
    # Pitch rate axis: largest integrated rate up to the lean; positive while leaning forward.
    i_lean = rows.index(lean)
    ang = [0.0, 0.0, 0.0]
    for a, b in zip(rows[:i_lean], rows[1:i_lean + 1]):
        for i in range(3):
            ang[i] += (b[2][i] - bias[i]) * (b[0] - a[0])
    # The pitch axis is the one left over (the largest integral can be yaw from handling).
    pr = 3 - up - fwd
    pr_sign = 1 if ang[pr] > 0 else -1
    tilt = math.atan2(-fwd_sign * lean[1][fwd], up_sign * lean[1][up])
    print(f"lean from accelerometer {tilt:.3f} rad, from gyro {pr_sign * ang[pr]:.3f} rad "
          f"(should agree within ~10%)")

    print(f"still: accel {['%.2f' % x for x in acc0]} m/s^2, gyro bias {['%.4f' % x for x in bias]} rad/s")
    print(f"lean:  accel {['%.2f' % x for x in lean[1]]}, integrated gyro {['%.3f' % x for x in ang]} rad")
    print(f"IMU_PITCH_RATE = ({pr}, {pr_sign:+d})\nIMU_ACCEL_FORWARD = ({fwd}, {fwd_sign:+d})\n"
          f"IMU_ACCEL_UP = ({up}, {up_sign:+d})")
    from . import hw_config as cfg
    same = (cfg.IMU_PITCH_RATE, cfg.IMU_ACCEL_FORWARD, cfg.IMU_ACCEL_UP) == \
        ((pr, pr_sign), (fwd, fwd_sign), (up, up_sign))
    print("matches hw_config" if same else "DIFFERS from hw_config: update it")


if __name__ == "__main__":
    main()
