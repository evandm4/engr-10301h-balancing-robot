"""Duty steps and coast-downs, one wheel at a time (C3; DATASHEET 8.2: k_e, friction, spin-up, duty map).

    cd ~/legobot && robot/run_safe.sh robot.measure.motor_steps --wheels-off --side both --wheel off

WHEELS OFF THE GROUND (or tires removed: say so with --wheel off|on, it only labels the output). Hands-free.
For each side, direction (+, -), duty and repeat: wait for rest, step to the duty, re-send it every 10 ms for
--hold s, coast, record --coast s more. Each step is fitted three ways:
  * steady speed over the last 40 % of the hold (linear fit of position),
  * spin-up: delayed first-order rise fitted to position -> w_inf, tau, t_d,
  * coast-down: w_dot = -alpha sgn(w) - beta w -> alpha = tau_c / J, beta = b / J.
Per side the summary gives the duty map (speed vs duty, both directions), a linear fit of |speed| against
duty x vin (slope = 1 / k_eff with k_eff = k + b / a in V s/rad, and the dead-band voltage from the
intercept), the median spin-up tau and t_d over duties >= 0.3, pooled alpha and beta, and how far the map
is from linear.

Notes on reading the numbers. g(duty) may not be duty x vin (slow/fast PWM, DATASHEET 3.3); the duty-map
linearity figure says how bad that is. tau = J / (a k + b) holds only if g is linear. The coast-down beta
assumes the HAT's `coast` is open-circuit; if it is a brake, beta includes electrical braking.
Files: logs/<stamp>_motor_steps.csv (raw stream, raw HAT signs), _steps.csv (one row per step),
_fit.json (constants with SE; se_kind 'ls' for the vin fit, 'repeats' for sd/sqrt(n) over steps).
"""
import argparse
import csv
import math
import statistics as st
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm
from . import fit

STEP_COLS = ["side", "dir", "duty", "rep", "t_cmd", "t_coast", "vin", "w_ss", "w_ss_se", "w_inf", "tau", "t_d",
             "spinup_rms", "alpha", "beta"]
MOVES = 2.0          # rad/s: below this a step counts as "did not move" and its dynamics are not fitted


def analyse_step(rows, port, t_cmd, t_coast, hold):
    """Fit one step; returns a dict with nan where a fit is not possible."""
    nan = float("nan")
    out = dict(w_ss=nan, w_ss_se=nan, w_inf=nan, tau=nan, t_d=nan, spinup_rms=nan, alpha=nan, beta=nan)
    t, pos = cm.series(rows, port)
    if t.size < 20:
        return out
    out["w_ss"], out["w_ss_se"] = fit.steady_speed(t, pos, t_cmd + 0.6 * hold, t_cmd + hold)
    if not abs(out["w_ss"]) > MOVES:
        return out
    try:
        s = fit.fit_spinup(t, pos, t_cmd)
        out.update(w_inf=s.omega, tau=s.tau, t_d=s.t_d, spinup_rms=s.rms)
    except ValueError:
        pass
    try:
        c = fit.fit_coastdown(t, pos, t_coast)
        out.update(alpha=c.alpha, beta=c.beta)
    except ValueError:
        pass
    return out


def run_side(hat, side, a, vin, steps):
    port = cm.SIDE_PORT[side]
    for direction in (+1, -1):
        for duty in a.duties:
            for rep in range(a.repeats):
                cm.check_fault(hat)
                if not cm.wait_rest(hat, side):
                    print(f"  (not at rest after 3 s before {side} {direction * duty:+.2f}; carrying on)")
                i0 = len(hat.data_log)
                t_cmd, _, _ = cm.hold_duty(hat, side, direction * duty, a.hold)
                t_coast = time.perf_counter()
                hat.coast()
                while time.perf_counter() - t_coast < a.coast:
                    cm.check_fault(hat)
                    time.sleep(0.05)
                r = analyse_step(cm.rows_for(hat, side, i0), port, t_cmd, t_coast, a.hold)
                steps.append(dict(side=side, dir=direction, duty=duty, rep=rep, t_cmd=t_cmd, t_coast=t_coast,
                                  vin=vin, **r))
                print(f"  {side:5} {direction * duty:+.2f} rep {rep}: w_ss {r['w_ss']:7.2f} rad/s  "
                      f"tau {r['tau'] * 1e3:5.1f} ms  t_d {r['t_d'] * 1e3:4.1f} ms  "
                      f"alpha {r['alpha']:6.1f}  beta {r['beta']:5.2f}")


def mean_sd(xs):
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return float("nan"), float("nan"), 0
    return st.fmean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan")), len(xs)


def summarise(side, steps, vin, hold):
    """Print the per-side summary and return (constants, diagnostics)."""
    S = [s for s in steps if s["side"] == side]
    duties = sorted({s["duty"] for s in S})
    consts, diag = {}, {}

    # duty map
    table = []
    for d in duties:
        row = {"duty": d}
        for name, dirn in (("w_pos", +1), ("w_neg", -1)):
            m, sd, n = mean_sd([s["w_ss"] for s in S if s["duty"] == d and s["dir"] == dirn])
            row[name], row[name + "_sd"] = m, sd
        table.append(row)

    # linear fit of |w_ss| against duty x vin over the duties where the motor always moves
    d_min = None
    for d in reversed(duties):
        if all(abs(s["w_ss"]) > MOVES for s in S if s["duty"] == d):
            d_min = d
        else:
            break
    print(f"\n=== {side}: vin {vin:.2f} V, {len(S)} steps ===")
    fitted = {}
    if d_min is not None:
        P = [s for s in S if s["duty"] >= d_min]
        x = np.array([s["duty"] * s["vin"] for s in P])
        y = np.array([abs(s["w_ss"]) for s in P])
        if len(set(x)) >= 3:
            coef, cov, sd = fit.linfit_cov(np.column_stack([np.ones_like(x), x]), y)
            i0, sl = coef
            k_eff = 1 / sl
            v_dead = -i0 / sl
            g = np.array([-1 / sl, i0 / sl**2])
            consts[cm.key("k_eff", side)] = cm.const(k_eff, math.sqrt(cov[1, 1]) / sl**2, "V s/rad", len(x), "ls")
            consts[cm.key("v_dead", side)] = cm.const(v_dead, math.sqrt(g @ cov @ g), "V", len(x), "ls")
            consts[cm.key("omega_per_volt", side)] = cm.const(sl, math.sqrt(cov[1, 1]), "rad/s/V", len(x), "ls")
            dev = (y - (i0 + sl * x)) / y.max() * 100
            diag[cm.key("linearity_max_dev_pct", side)] = float(np.abs(dev).max())
            diag[cm.key("fit_duties_from", side)] = d_min
            fitted = {id(s): dv for s, dv in zip(P, dev)}
    for r in table:
        devs = [fitted[id(s)] for s in S if s["duty"] == r["duty"] and id(s) in fitted]
        r["dev_pct"] = st.fmean(devs) if devs else float("nan")
    print("  duty    w(+) rad/s    w(-) rad/s    deviation from linear fit (% of full scale)")
    for r in table:
        dev = "" if math.isnan(r["dev_pct"]) else f"{r['dev_pct']:+.1f}"
        print(f"  {r['duty']:4.2f}  {r['w_pos']:8.2f}      {r['w_neg']:8.2f}      {dev}")
    diag[cm.key("duty_map", side)] = table
    if fitted:
        print(cm.show("k_eff = k + b/a", consts[cm.key("k_eff", side)]))
        print(cm.show("dead-band voltage", consts[cm.key("v_dead", side)]))
        print(f"  duty map: largest deviation from linear {diag[cm.key('linearity_max_dev_pct', side)]:.1f} % of full scale "
              f"(fit over duties >= {d_min})")
    else:
        print("  not enough duties where the motor always moves: no vin fit")

    # dynamics
    good = [s for s in S if s["duty"] >= 0.3 and math.isfinite(s["tau"])]
    for name, unit, scale in (("tau", "s", 1.0), ("t_d", "s", 1.0)):
        xs = [s[name] for s in good]
        if len(xs) >= 2:
            sd = st.stdev(xs)
            consts[cm.key(name, side)] = cm.const(st.median(xs), sd / math.sqrt(len(xs)), unit, len(xs), "repeats")
            print(cm.show(f"{name} (median)", consts[cm.key(name, side)]))
    edge = [s for s in good if s["t_d"] in (0.0, 0.04) or s["tau"] <= 0.0051 or s["tau"] >= 0.299]
    if edge:
        print(f"  warning: {len(edge)} spin-up fits sit on the edge of the search grid")
    print("  (tau = J / (a k + b) only if the duty map is linear; t_d includes ~2 ms of data-line transit)")
    for name, unit in (("alpha", "rad/s^2"), ("beta", "1/s")):
        m, sd, n = mean_sd([s[name] for s in S])
        if n >= 2:
            consts[cm.key(name, side)] = cm.const(m, sd / math.sqrt(n), unit, n, "repeats")
            print(cm.show(f"{name} (pooled mean)", consts[cm.key(name, side)]))
    print("  (alpha = tau_c/J, beta = b/J. beta assumes `coast` is open-circuit; a brake adds electrical damping)")
    return consts, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required: wheels clear of the ground")
    ap.add_argument("--side", choices=("left", "right", "both"), default="both")
    ap.add_argument("--wheel", choices=("on", "off"), default="off", help="label: tire attached or removed")
    ap.add_argument("--duties", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--hold", type=float, default=1.5, help="s at each duty")
    ap.add_argument("--coast", type=float, default=1.5, help="s recorded after coasting")
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.require_wheels_off(a)
    cm.check_duties(a.duties)

    hat, vin0 = cm.open_hat(a.pwmparams)
    st0 = cm.stamp()
    base = cm.base_path(a.outdir, st0, "motor_steps")
    steps, vins = [], {}
    consts, diag = {}, {}
    try:
        for side in cm.sides_of(a.side):
            vin = cm.read_vin(hat)
            vins[side] = [vin]
            print(f"{side}: vin {vin:.2f} V, pwmparams {a.pwmparams}, wheel {a.wheel}")
            run_side(hat, side, a, vin, steps)
            cm.stop(hat)
            vins[side].append(hat.vin())
        for side in cm.sides_of(a.side):
            c, d = summarise(side, steps, vins[side][0], a.hold)
            consts.update(c)
            diag.update(d)
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        cm.write_raw_csv(base + ".csv", list(hat.data_log))
        with open(base + "_steps.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(STEP_COLS)
            w.writerows([[s[c] for c in STEP_COLS] for s in steps])
    meta = {"pwmparams": a.pwmparams, "vin": vins, "side": a.side, "wheel": a.wheel, "wheels_off_ground": True,
            "args": vars(a), "firmware": hat.version, "stamp": st0}
    print("\nwrote", base + ".csv", base + "_steps.csv", cm.write_fit_json(base, "motor_steps", consts, meta, diag))
    hat.close()


if __name__ == "__main__":
    main()
