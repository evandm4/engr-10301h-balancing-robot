"""Battery discharge log: does the motor rail follow the battery? (DATASHEET 8.2 first row, 8.5). Long run.

    cd ~/legobot && robot/run_safe.sh robot.measure.discharge --wheels-off
    cd ~/legobot && robot/run_safe.sh robot.measure.discharge --wheels-off --multimeter    # type rail readings

WHEELS OFF THE GROUND, battery fully charged, left running for the whole discharge (an hour or more).
Every --period s (default 30) it reads vin and `vcgencmd get_throttled`, drives both wheels at --duty
(default 0.5) for --step-seconds (default 2) and measures each wheel's steady speed over the last second,
then loads the battery for the rest of the period by alternating the --cycle duties (default 0.3 0.6, 1 s
each, both wheels). One row per period is appended to the CSV and flushed, so a crash keeps the data.

It stops (and fits what it has) at vin below --cutoff (default 6.8 V), on a Pi under-voltage or throttle
event (low four bits of get_throttled set, or any bit changed since the first reading; the sticky "has
occurred" bits from earlier in the boot are tolerated), on a HAT fault, or on Ctrl-C.

Fit, per wheel: speed = intercept + slope vin by least squares. If the motor rail follows the battery,
speed is proportional to the drive voltage d vin less a dead-band term, so the slope is clearly positive and
the elasticity (slope mean(vin) / mean(speed), the fractional speed change per fractional vin change) is near
1 or above (above 1 when the intercept is negative, the dead band). If the rail is regulated, the slope is
near 0 and the elasticity near 0. The script prints both and which the data favour: "follows" if the slope is
more than 3 SE from 0 and the elasticity above 0.5, "regulated" if the slope is within 3 SE of 0 or the
elasticity below 0.25, else "ambiguous". Weak if vin only spans a few tenths of a volt. Warming of the
motors during the run also changes speed and is confounded with the falling vin; the sd of speed at the
start (first rows) shows the scale of other drift.
--multimeter prompts, after each step, for the rail voltage you read at the motor terminals during it (blank
to skip); the rail-vs-vin slope is fitted too (1 = follows exactly).
Files: logs/<stamp>_discharge.csv (one row per period), _fit.json.
"""
import argparse
import csv
import math
import subprocess
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm
from . import fit

COLS = ["t_wall", "elapsed_s", "vin", "throttled", "w_left", "w_left_se", "w_right", "w_right_se", "rail_v", "vin_after"]


def get_throttled():
    """Integer from `vcgencmd get_throttled`, or None if it cannot be read."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return int(out.split("=")[1].strip(), 16)
    except Exception:
        return None


def read_vin_any(hat):
    return hat.vin() or hat.vin()


def fit_side(name, vin, w, consts, side):
    """Speed against vin for one wheel; prints the verdict."""
    ok = np.isfinite(w) & np.isfinite(vin)
    vin, w = vin[ok], w[ok]
    if len(vin) < 5 or np.ptp(vin) < 0.05:
        print(f"  {name}: too few rows or vin span below 0.05 V: no fit")
        return
    coef, cov, sd = fit.linfit_cov(np.column_stack([np.ones_like(vin), vin]), w)
    i0, sl = coef
    se_sl = math.sqrt(cov[1, 1])
    eps = sl * vin.mean() / w.mean()
    se_eps = se_sl * vin.mean() / abs(w.mean())
    z = sl / se_sl
    if abs(z) < 3 or eps < 0.25:
        verdict = "regulated (speed does not follow vin)"
    elif eps > 0.5:
        verdict = "FOLLOWS the battery"
    else:
        verdict = "ambiguous"
    consts[cm.key("omega_vs_vin_slope", side)] = cm.const(sl, se_sl, "rad/s/V", len(vin), "ls")
    consts[cm.key("omega_vs_vin_intercept", side)] = cm.const(i0, math.sqrt(cov[0, 0]), "rad/s", len(vin), "ls")
    consts[cm.key("elasticity", side)] = cm.const(eps, se_eps, "-", len(vin), "ls")
    print(f"  {name}: vin {vin.min():.2f}-{vin.max():.2f} V ({len(vin)} rows), speed {w.min():.1f}-{w.max():.1f} rad/s")
    for k in ("omega_vs_vin_slope", "omega_vs_vin_intercept", "elasticity"):
        print(cm.show(k, consts[cm.key(k, side)]))
    print(f"    z = {z:.1f}; data favour: {verdict}")
    if np.ptp(vin) < 0.4:
        print("    (vin span under 0.4 V: a weak test)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required: wheels clear of the ground")
    ap.add_argument("--period", type=float, default=30.0)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--step-seconds", type=float, default=2.0)
    ap.add_argument("--cycle", type=float, nargs="*", default=[0.3, 0.6], help="duties alternated 1 s each for the rest of the period")
    ap.add_argument("--cutoff", type=float, default=6.8)
    ap.add_argument("--multimeter", action="store_true")
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.require_wheels_off(a)
    cm.check_duties([a.duty] + a.cycle)

    hat, vin0 = cm.open_hat(a.pwmparams)
    base = cm.base_path(a.outdir, cm.stamp(), "discharge")
    print(f"start vin {vin0:.2f} V; one row every {a.period:g} s to {base}.csv; Ctrl-C to stop and fit")
    rows, reason, pending_exit = [], "?", None
    thr0 = get_throttled()
    t_start = time.perf_counter()
    nxt = t_start
    fcsv = open(base + ".csv", "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(COLS)
    try:
        while True:
            vin = read_vin_any(hat)
            thr = get_throttled()
            if vin is None:
                reason = "could not read vin"
                break
            if vin < a.cutoff:
                reason = f"vin {vin:.2f} V below the cutoff {a.cutoff} V"
                break
            if thr is not None and thr0 is not None and ((thr & 0xF) or thr != thr0):
                reason = f"Pi throttled/under-voltage flags changed: {thr:#x}"
                break
            cm.check_fault(hat)
            del hat.data_log[:]
            p_start = time.perf_counter()
            if a.multimeter:
                print(f"  [{p_start - t_start:6.0f} s] vin {vin:.2f} V: READ THE MOTOR RAIL NOW (both wheels at {a.duty:g})")
            t_cmd, _, _ = cm.hold_duty(hat, "both", a.duty, a.step_seconds)
            rows_raw = list(hat.data_log)
            cm.stop(hat)
            w, se = {}, {}
            for side in ("left", "right"):
                t, pos = cm.series(rows_raw, cm.SIDE_PORT[side])
                w[side], se[side] = fit.steady_speed(t, pos, t_cmd + a.step_seconds - 1.0, t_cmd + a.step_seconds)
            rail = cm.ask_float("  rail voltage seen (V, blank to skip): ") if a.multimeter else None
            # load the battery for the rest of the period
            k = 0
            while a.cycle and time.perf_counter() - p_start < a.period - 1.0:
                cm.hold_duty(hat, "both", a.cycle[k % len(a.cycle)], 1.0)
                k += 1
            cm.stop(hat)
            time.sleep(0.3)
            vin_after = read_vin_any(hat)
            row = [time.time(), time.perf_counter() - t_start, vin, thr, w["left"], se["left"], w["right"], se["right"],
                   rail, vin_after]
            rows.append(row)
            wr.writerow(row)
            fcsv.flush()
            print(f"  [{row[1]:6.0f} s] vin {vin:.2f} V  left {w['left']:6.2f}  right {w['right']:6.2f} rad/s  throttled {thr}")
            nxt += a.period
            if nxt > time.perf_counter():
                cm.sleep_until(nxt)
    except (SystemExit, KeyboardInterrupt) as e:
        reason = f"stopped: {e.code if isinstance(e, SystemExit) else 'Ctrl-C'}"
        pending_exit = e
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        fcsv.close()

    print(f"\nrun ended: {reason}; {len(rows)} rows")
    consts, diag = {}, {"end_reason": reason}
    print("=== speed vs vin ===")
    if rows:
        arr = np.array([[float(x) if x is not None else float("nan") for x in r] for r in rows])
        fit_side("left", arr[:, 2], arr[:, 4], consts, "left")
        fit_side("right", arr[:, 2], arr[:, 6], consts, "right")
        rail_ok = np.isfinite(arr[:, 8])
        if rail_ok.sum() >= 5 and np.ptp(arr[rail_ok, 2]) > 0.05:
            coef, se, _ = fit.linfit(np.column_stack([np.ones(rail_ok.sum()), arr[rail_ok, 2]]), arr[rail_ok, 8])
            consts["rail_vs_vin_slope"] = cm.const(coef[1], se[1], "V/V", int(rail_ok.sum()), "ls")
            print(cm.show("rail_vs_vin_slope", consts["rail_vs_vin_slope"]))
    meta = {"pwmparams": a.pwmparams, "vin": {"start": vin0, "last": rows[-1][2] if rows else None}, "side": "both",
            "wheel": "off the ground", "args": vars(a), "firmware": hat.version}
    print("wrote", base + ".csv", cm.write_fit_json(base, "discharge", consts, meta, diag))
    try:
        hat.close()
    except Exception:
        pass
    if pending_exit is not None:
        raise pending_exit


if __name__ == "__main__":
    main()
