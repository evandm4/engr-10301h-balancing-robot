"""Does duty 0 coast or brake? Hand-spun wheel, both modes, alternating (B2b; triage G3.1). Interactive.

    cd ~/legobot && robot/run_safe.sh robot.measure.coast_vs_brake --wheels-off --side left

The wheel is spun by hand (no motor drive), once under `port N ; coast` and once under `pwm ; set 0`
(set_pwm(0, 0), the "duty 0" the robot sends), alternating, --trials of each. The script waits until the
wheel is at rest, then for a flick faster than --trigger rad/s, then records --record s. Flick the wheel
the same way every time (a firm push on the tire, about 15-25 rad/s) and let go. Which mode a trial is in is
not shown, so the flick can't be biased. The wheel and robot must be free to turn: wheels off the ground.

For each trial it reports the speed after the release and the time the speed takes to fall from --hi to --lo
rad/s. If `set 0` brakes (shorted winding), that time is much shorter than under coast, and the speed decays
nearly exponentially instead of with the friction's constant torque.
Files: logs/<stamp>_coast_vs_brake.csv (t, trial, mode, pos_deg_fwd) and _summary.txt.
"""
import argparse
import csv
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm


def speed(t, pos_deg, n=5):
    """Speed (rad/s) from position, a least-squares slope over a sliding window of n samples."""
    p = np.radians(pos_deg)
    w = np.full(len(t), np.nan)
    for i in range(n - 1, len(t)):
        tt, pp = t[i - n + 1:i + 1], p[i - n + 1:i + 1]
        w[i] = np.polyfit(tt - tt.mean(), pp, 1)[0]
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required: the wheel must turn freely")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--trials", type=int, default=4, help="per mode")
    ap.add_argument("--trigger", type=float, default=8.0, help="rad/s that counts as a flick")
    ap.add_argument("--record", type=float, default=3.0)
    ap.add_argument("--wait", type=float, default=60.0, help="s to wait for each flick")
    ap.add_argument("--hi", type=float, default=12.0)
    ap.add_argument("--lo", type=float, default=3.0)
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.require_wheels_off(a)
    port = cm.SIDE_PORT[a.side]
    sign = cfg.PORT_SIGN[port]

    hat, vin = cm.open_hat(a.pwmparams)
    base = cm.base_path(a.outdir, cm.stamp(), "coast_vs_brake")
    plan = [m for _ in range(a.trials) for m in ("coast", "pwm0")]
    print(f"vin {vin:.2f} V, {a.side} wheel, {len(plan)} flicks, alternating modes (hidden). Flick when it is at rest.")
    out, lines = [], []
    try:
        for k, mode in enumerate(plan):
            if mode == "coast":
                hat.coast()
            else:
                hat.set_pwm(0.0, 0.0)
            cm.wait_rest(hat, a.side, quiet=0.5, timeout=10.0)
            print(f"  trial {k + 1}/{len(plan)}: flick the wheel now")
            i0 = len(hat.data_log)
            t_hit = None
            end = time.perf_counter() + a.wait
            while time.perf_counter() < end and t_hit is None:
                cm.check_fault(hat)
                rows = cm.rows_for(hat, a.side, i0)
                if len(rows) >= 6:
                    t = np.array([r[0] for r in rows[-6:]])
                    p = sign * np.array([r[3] for r in rows[-6:]], float)
                    w = np.polyfit(t - t.mean(), np.radians(p), 1)[0]
                    if abs(w) >= a.trigger:
                        t_hit = time.perf_counter()
                time.sleep(0.01)
            if t_hit is None:
                print("    no flick seen; skipping")
                continue
            time.sleep(a.record)
            rows = cm.rows_for(hat, a.side, i0)
            t = np.array([r[0] for r in rows])
            pos = sign * np.array([r[3] for r in rows], float)
            pos = np.degrees(np.unwrap(np.radians(pos)))
            out += [(k + 1, mode, ti, pi) for ti, pi in zip(t, pos)]
            w = speed(t, pos)
            wa = np.abs(w)
            i_pk = int(np.nanargmax(wa))
            peak = float(wa[i_pk])
            t_hi = t_lo = None
            for i in range(i_pk, len(t)):
                if t_hi is None and wa[i] <= a.hi:
                    t_hi = t[i]
                if t_hi is not None and wa[i] <= a.lo:
                    t_lo = t[i]
                    break
            dt = (t_lo - t_hi) if (t_hi is not None and t_lo is not None) else float("nan")
            msg = f"    {mode:5s} peak {peak:5.1f} rad/s, {a.hi:g} -> {a.lo:g} rad/s in {dt:.3f} s"
            print(msg)
            lines.append(f"trial {k + 1} {mode} peak {peak:.2f} t_{a.hi:g}_to_{a.lo:g} {dt:.4f}")
        for mode in ("coast", "pwm0"):
            v = [float(l.split()[-1]) for l in lines if f" {mode} " in l]
            v = [x for x in v if np.isfinite(x)]
            if v:
                s = f"{mode}: {len(v)} trials, {a.hi:g}->{a.lo:g} rad/s in {np.mean(v):.3f} s (sd {np.std(v):.3f})"
                print(s)
                lines.append(s)
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "mode", "t", "pos_deg_fwd"])
            w.writerows(out)
        with open(base + "_summary.txt", "w") as f:
            f.write(f"vin {vin:.2f} side {a.side} pwmparams {a.pwmparams}\n" + "\n".join(lines) + "\n")
        hat.close()
    print("wrote", base + ".csv")


if __name__ == "__main__":
    main()
