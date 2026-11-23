"""Log both motors (100 Hz) and the IMU (200 Hz) to CSV and report the timing (C1, C2 checks).

    cd ~/legobot && robot/run_safe.sh robot.bench_log --seconds 5 --pwm 0.15
    ~/legobot-venv/bin/python -m robot.bench_log --no-hat --seconds 10     # IMU only

A 200 Hz loop on time.perf_counter reads the IMU every tick and sends one command for both motors
every second tick. Any nonzero duty spins the wheels: wheels off the ground. Files go to
logs/<stamp>_{imu,motors,writes}.csv; the summary is printed and saved as _summary.txt.
"""
import argparse
import csv
import os
import statistics as st
import time

from . import hw_config as cfg


def stats_ms(xs):
    xs = sorted(x * 1e3 for x in xs if x is not None)
    if not xs:
        return "n/a"
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    return (f"mean {st.fmean(xs):.3f}  sd {sd:.3f}  min {xs[0]:.3f}  p50 {p(.5):.3f}  "
            f"p99 {p(.99):.3f}  max {xs[-1]:.3f} ms  (n={len(xs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--pwm", type=float, default=0.0, help="forward duty for both wheels")
    ap.add_argument("--left", type=float, help="overrides --pwm for the left wheel")
    ap.add_argument("--right", type=float, help="overrides --pwm for the right wheel")
    ap.add_argument("--plimit", type=float, default=cfg.BENCH_PLIMIT)
    ap.add_argument("--rate", type=float, default=200.0, help="loop rate, Hz")
    ap.add_argument("--no-hat", action="store_true")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--drain", choices=("none", "lsr", "tcdrain"), default="lsr",
                    help="lsr: busy-wait until the UART is empty to time the last byte (default)")
    ap.add_argument("--cmd", help="override BuildHat.PWM_CMD, e.g. 'po {l};set {a};po {r};set {b}'")
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    left = a.pwm if a.left is None else a.left
    right = a.pwm if a.right is None else a.right
    if max(abs(left), abs(right)) > cfg.BENCH_DUTY_CAP:
        raise SystemExit(f"bench duty above {cfg.BENCH_DUTY_CAP} refused (hw_config.BENCH_DUTY_CAP)")

    hat = imu = None
    if not a.no_hat:
        from .buildhat_serial import BuildHat, install_safety
        if a.cmd:
            BuildHat.PWM_CMD = a.cmd
        hat = BuildHat(log_lines=True)
        install_safety(hat)
        hat.configure(plimit=a.plimit)
        time.sleep(0.3)                         # let both streams start
    if not a.no_imu:
        from .imu import Imu, pitch_from_accel
        imu = Imu()

    drain = {"none": False, "lsr": "lsr", "tcdrain": True}[a.drain]
    period = 1.0 / a.rate
    imu_rows, ticks = [], []
    t_start = time.perf_counter()
    nxt = t_start
    n = overruns = 0
    try:
        while True:
            now = time.perf_counter()
            if now - t_start >= a.seconds:
                break
            ticks.append(now)
            if imu is not None:
                r = imu.read()
                imu_rows.append((r.t, r.read_s, r.pitch_rate, r.acc_fwd, r.acc_up,
                                 pitch_from_accel(r.acc_fwd, r.acc_up), *r.gyro, *r.acc))
            if hat is not None and n % 2 == 0:
                hat.set_pwm(left, right, drain=drain)
            n += 1
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                overruns += 1
                if delay < -period:
                    nxt = time.perf_counter()   # don't try to catch up a long stall
    finally:
        if hat is not None:
            hat.set_pwm(0.0, 0.0)
            hat.close()
        if imu is not None:
            imu.close()

    # ---- files ----
    os.makedirs(a.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(a.outdir, stamp)
    out = [f"bench_log {stamp}: {a.seconds} s, loop {a.rate} Hz, duty L {left} R {right}, "
           f"plimit {a.plimit}, drain {a.drain}"]
    out.append(f"loop period:     {stats_ms(b - a_ for a_, b in zip(ticks, ticks[1:]))}; overruns {overruns}")
    if imu_rows:
        with open(base + "_imu.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "read_s", "pitch_rate", "acc_fwd", "acc_up", "theta_acc",
                        "gx", "gy", "gz", "ax", "ay", "az"])
            w.writerows(imu_rows)
        ts = [r[0] for r in imu_rows]
        out.append(f"IMU rate:        {len(ts) / (ts[-1] - ts[0]):.1f} Hz")
        out.append(f"IMU read time:   {stats_ms(r[1] for r in imu_rows)}")
        out.append(f"IMU interval:    {stats_ms(b - a_ for a_, b in zip(ts, ts[1:]))}")
        out.append(f"IMU config:      {imu.config_readback}")
    if hat is not None:
        with open(base + "_motors.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "port", "speed_raw", "pos_raw", "apos_raw", "speed_fwd", "pos_fwd"])
            for t, p, sp, pos, apos in hat.data_log:
                s = cfg.PORT_SIGN[p]
                w.writerow([t, p, sp, pos, apos, s * sp, s * pos])
        with open(base + "_writes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t0", "t_written", "t_drained", "bytes", "age_left", "age_right"])
            w.writerows(hat.write_log)
        out.append(f"HAT: {hat.version}; PWM_CMD {hat.PWM_CMD!r}")
        for p in (cfg.LEFT_PORT, cfg.RIGHT_PORT):
            ts = [r[0] for r in hat.data_log if r[1] == p]
            if len(ts) > 1:
                out.append(f"port {p} lines:   {len(ts) / (ts[-1] - ts[0]):.1f} Hz; "
                           f"interval {stats_ms(b - a_ for a_, b in zip(ts, ts[1:]))}")
                rows = [r for r in hat.data_log if r[1] == p]
                half = rows[len(rows) // 2:]
                out.append(f"port {p} speed:   raw mean {st.fmean(r[2] for r in half):.1f} % (second half); "
                           f"pos {rows[0][3]} -> {rows[-1][3]} deg raw")
            else:
                out.append(f"port {p}: no data lines")
        wl = hat.write_log[:-1]                 # last entry is the stop command
        out.append(f"cmd bytes:       {wl[0][3] if wl else ''}")
        out.append(f"write() return:  {stats_ms(r[1] - r[0] for r in wl)}")
        out.append(f"write to done:   {stats_ms(r[2] - r[0] for r in wl)}")
        out.append(f"age left @cmd:   {stats_ms(r[4] for r in wl)}")
        out.append(f"age right @cmd:  {stats_ms(r[5] for r in wl)}")
        msgs = [m for t, m in hat.messages]
        out.append(f"HAT messages:    {msgs[-10:]}")
        if hat.fault:
            out.append(f"FAULT: {hat.fault}")
    text = "\n".join(out)
    print(text)
    with open(base + "_summary.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
