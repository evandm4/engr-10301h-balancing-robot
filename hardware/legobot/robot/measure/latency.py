"""Command-to-motion latency: duty step to the first motion in the position stream (DATASHEET 8.6). Hands-free.

    cd ~/legobot && robot/run_safe.sh robot.measure.latency --wheels-off --side left --trials 30

WHEELS OFF THE GROUND. Each trial waits for rest, waits a random 0-10 ms (so the command lands at a uniform
phase of the HAT's 100 Hz stream), steps one wheel from 0 to --duty (default 0.5), re-sends it every 10 ms
for 0.7 s, and fits the spin-up with fit.fit_spinup. t_d is the delay between the START of the command write
(write_log t0) and the motion onset in the position stream. It is NOT subtracted from anything here. It
includes: the serial write itself and the HAT's parsing, the electrical L/R rise (about 3 ms estimated), the
motor's mechanical start, the HAT's 100 Hz sampling of the encoder, and the data line's transit back to the
Pi (about 2 ms, DATASHEET 6), plus the timing noise of the reader thread. Note also that the first count
needs 0.5-1 degree of rotation, which is part of why t_d looks longer than the pure electrical delay.

Reports mean, sd, p5 and p95 of t_d over the valid trials (|w_inf| > 5 rad/s and the fit off the edge of its
grid), and the regression of t_d on the age of the newest data line at the time of the command
(write_log age), which shows whether the stream's phase matters. se_kind 'repeats' (sd / sqrt(n)).
Files: logs/<stamp>_latency.csv (one row per trial), _fit.json.
"""
import argparse
import csv
import math
import random
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm
from . import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required: wheels clear of the ground")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--hold", type=float, default=0.7)
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.require_wheels_off(a)
    cm.check_duties([a.duty])
    side, port = a.side, cm.SIDE_PORT[a.side]
    age_col = 4 if side == "left" else 5

    hat, vin = cm.open_hat(a.pwmparams)
    base = cm.base_path(a.outdir, cm.stamp(), "latency")
    rows = []
    try:
        for k in range(a.trials):
            cm.check_fault(hat)
            if not cm.wait_rest(hat, side, timeout=4.0):
                print("  (not at rest; carrying on)")
            extra = random.uniform(0.0, 0.010)
            cm.sleep_until(time.perf_counter() + extra)
            i0, w0 = len(hat.data_log), len(hat.write_log)
            t_cmd, _, _ = cm.hold_duty(hat, side, a.duty, a.hold)
            age = hat.write_log[w0][age_col]          # age of the newest line when the step was written
            hat.coast()
            t, pos = cm.series(cm.rows_for(hat, side, i0), port)
            try:
                f = fit.fit_spinup(t, pos, t_cmd)
            except ValueError:
                continue
            valid = abs(f.omega) > 5 and 0.0 < f.t_d < 0.04
            rows.append((k, extra, t_cmd, age, f.t_d, f.tau, f.omega, f.rms, valid))
            print(f"  trial {k + 1:2d}: extra {extra * 1e3:4.1f} ms  t_d {f.t_d * 1e3:5.1f} ms  tau {f.tau * 1e3:5.1f} ms  "
                  f"w {f.omega:5.1f}{'' if valid else '  (invalid)'}")
            time.sleep(0.3)
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        with open(base + ".csv", "w", newline="") as f_:
            w = csv.writer(f_)
            w.writerow(["trial", "extra_wait_s", "t_cmd", "age_newest_line_s", "t_d_s", "tau_s", "w_inf", "rms", "valid"])
            w.writerows(rows)

    consts, diag = {}, {}
    td = np.array([r[4] for r in rows if r[8]])
    print("\n=== latency ===")
    if len(td) >= 3:
        sd = float(td.std(ddof=1))
        consts[cm.key("t_d", side)] = cm.const(td.mean(), sd / math.sqrt(len(td)), "s", len(td), "repeats")
        diag.update({"sd_s": sd, "p5_s": float(np.percentile(td, 5)), "p95_s": float(np.percentile(td, 95)),
                     "min_s": float(td.min()), "max_s": float(td.max())})
        print(f"  t_d: mean {td.mean() * 1e3:.2f} ms, sd {sd * 1e3:.2f}, p5 {diag['p5_s'] * 1e3:.2f}, "
              f"p95 {diag['p95_s'] * 1e3:.2f} ms  (n={len(td)} of {len(rows)} trials)")
        print(cm.show("t_d (mean)", consts[cm.key("t_d", side)]))
        ages = np.array([r[3] for r in rows if r[8] and r[3] is not None])
        if len(ages) >= 5:
            tdv = np.array([r[4] for r in rows if r[8] and r[3] is not None])
            coef, se, _ = fit.linfit(np.column_stack([np.ones_like(ages), ages]), tdv)
            diag["t_d_vs_age_slope"] = [float(coef[1]), float(se[1])]
            print(f"  t_d vs age of the newest line at the command: slope {coef[1]:+.2f} +- {se[1]:.2f} (0 means no dependence on the stream phase)")
        print("  note: t_d includes data-line transit (about 2 ms, DATASHEET 6) and the electrical L/R rise (about 3 ms est.); nothing is subtracted")
    else:
        print("  fewer than 3 valid trials")
    meta = {"pwmparams": a.pwmparams, "vin": vin, "side": side, "wheel": "off the ground", "args": vars(a),
            "firmware": hat.version}
    print("wrote", base + ".csv", cm.write_fit_json(base, "latency", consts, meta, diag))
    hat.close()


if __name__ == "__main__":
    main()
