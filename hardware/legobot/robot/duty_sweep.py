"""Steady speed of one wheel against duty, port_plimit and pwmparams (C1 checks; feeds C3, B1).

    cd ~/legobot && robot/run_safe.sh robot.duty_sweep --side left --plimits 0.3 1 --duties 0.1 0.2
WHEELS OFF THE GROUND. Each step holds the duty for --hold s and reports the mean raw speed (%)
over its second half, then coasts for 0.5 s. Prints a table and saves logs/<stamp>_sweep.csv.
"""
import argparse
import csv
import os
import statistics as st
import time

from . import hw_config as cfg
from .buildhat_serial import BuildHat, install_safety


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--plimits", type=float, nargs="+", default=[1.0])
    ap.add_argument("--duties", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--hold", type=float, default=1.5)
    a = ap.parse_args()
    if any(abs(d) > cfg.BENCH_DUTY_CAP for d in a.duties):
        raise SystemExit(f"duty above {cfg.BENCH_DUTY_CAP} refused (hw_config.BENCH_DUTY_CAP)")
    if any(abs(d) * p > 0.5 for d in a.duties for p in a.plimits):
        raise SystemExit("duty x plimit above 0.5 refused on the bench")
    port = cfg.LEFT_PORT if a.side == "left" else cfg.RIGHT_PORT

    hat = BuildHat(log_lines=True)
    install_safety(hat)
    hat.configure(plimit=a.plimits[0], pwmparams=tuple(a.pwmparams))
    time.sleep(0.3)
    rows = []
    for pl in a.plimits:
        hat.command(f"port {port} ; port_plimit {pl}", 0.05)
        for d in a.duties:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < a.hold:
                if a.side == "left":
                    hat.set_pwm(d, 0.0)
                else:
                    hat.set_pwm(0.0, d)
                time.sleep(0.01)
            half = [r for r in hat.data_log if r[1] == port and r[0] > t0 + a.hold / 2]
            sp = st.fmean(r[2] for r in half) if half else float("nan")
            # Encoder check, independent of the motor's own SPEED estimate: position slope, deg/s.
            sgn = cfg.PORT_SIGN[port]
            slope = (sgn * (half[-1][3] - half[0][3]) / (half[-1][0] - half[0][0])
                     if len(half) > 1 else float("nan"))
            rows.append((a.side, pl, d, a.pwmparams[0], a.pwmparams[1], sp, slope))
            print(f"plimit {pl:4}  duty {d:6}  -> speed {sgn * sp:6.1f} % fwd, encoder {slope:7.1f} deg/s fwd"
                  f"   {[m for t, m in hat.messages][-2:]}")
            hat.set_pwm(0.0, 0.0)
            hat.coast()
            time.sleep(0.5)
    hat.close()
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{time.strftime('%Y%m%d_%H%M%S')}_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "plimit", "duty", "pwmthresh", "minpwm", "speed_raw_pct", "pos_slope_fwd_deg_s"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
