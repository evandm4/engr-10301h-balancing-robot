"""String-and-mass lift: the motor's torque slope c = a k + b and the lift-off duty (DATASHEET 8.2). Interactive.

    cd ~/legobot && robot/run_safe.sh robot.measure.load --side left --lift-dir +1 --direction-test
    cd ~/legobot && robot/run_safe.sh robot.measure.load --side left --lift-dir +1 --hold-search

SETUP (this script is the exception to --wheels-off: the wheel is loaded, so it prints the setup and waits
for Enter). Clamp the robot so ONE wheel overhangs the table edge with room below it. Wind a string on the
tire (or on a pulley of radius --radius-mm, default 34.5 = the tire radius) with a hook for masses at the
free end. The mass should rest on the floor or a support with the string just taut at the start of every
lift, and be allowed to rise at least --max-turns turns of the wheel (default 1.5) before it would hit the
wheel. Keep a hand near the mass: when the script coasts, the wheel back-drives and the mass sinks.

LIFT DIRECTION. --lift-dir +1 or -1 is the sign of the forward duty that winds the string up. Check it
first with --direction-test (a 0.4 s pulse at 0.15 duty, no mass): if the string wound up, the sign is
right; if it unwound, run again with the other sign.

PROCEDURE. Enter masses in grams, 0 g FIRST (the no-load speed with the string attached at each duty), then
any others, blank to finish. For each mass the script lifts at each duty in --duties for --hold s (stopping
early at --max-turns turns) and measures the lifting speed over the last half. For masses above 0 g it then
switches straight to the next duty in --lower-duties (lift direction, too small to hold the mass) and lets
the mass sink back to its start against the motor: the braking quadrant (duty and speed of opposite sign),
which B1b needs (SOFTWARE_PLAN; DATASHEET 8.7). Duty 0 there shows whether `set 0` brakes or coasts. A
lowering run ends when the mass is back within 0.15 turn of its start, after 4 s, or after 1 s without
sinking (the duty held it: the script coasts, and the mass drops onto the towel). Then the next lift starts
by itself; you are asked to re-tension the string only between masses. Put a towel under the mass. With --hold-search it also ramps the duty from 0 in 0.02 steps
(0.3 s each) and records the first duty at which the wheel turns more than 2 degrees in the lifting
direction: the lift-off duty. (For 0 g that is the unloaded breakaway.)

MODEL. Per motor, tau_L = m g r; at the same duty the loaded and no-load runs have the same drive voltage,
so tau_L = c (w_nl - w) with c = a k + b, whatever g(duty) is (fit.fit_load_drop). Since
k_eff = k + b/a (motor_steps), a = c / k_eff. At lift-off (static): a g(d) = tau_L + tau_c, so the lift-off
duty is linear in tau_L if g is linear in the duty; slope = 1 / (a vin) and intercept = tau_c / (a vin).
Note: a lift stopped early at --max-turns leaves only a short window for the speed, which still holds some of the
spin-up (tau ~ 60 ms); use a larger --max-turns or lower duties if the run is short. The no-load run is stopped the
same way, so the transient largely cancels in c.
Files: logs/<stamp>_load.csv (one row per lift), _fit.json.
"""
import argparse
import csv
import math
import time

import numpy as np

from .. import hw_config as cm_cfg
from . import common as cm
from . import fit

G = 9.80665


def lift_run(hat, side, port, duty, a):
    """One lift at `duty`. Returns (w_lift rad/s positive when lifting, turns moved, stopped_early)."""
    cm.wait_rest(hat, side)
    i0 = len(hat.data_log)
    p0 = hat.latest(port).pos
    over = lambda: abs(hat.latest(port).pos - p0) > a.max_turns * 360
    t_cmd, t_end, early = cm.hold_duty(hat, side, a.lift_dir * duty, a.hold, stop_if=over)
    hat.coast()
    t, pos = cm.series(cm.rows_for(hat, side, i0), port)
    w, _ = fit.steady_speed(t, pos, t_cmd + (t_end - t_cmd) / 2, t_end)
    turns = abs(hat.latest(port).pos - p0) / 360
    return a.lift_dir * w, turns, early


def lift_lower(hat, side, port, duty, d_low, a):
    """Lift at `duty` (as lift_run), then without coasting hold the lift-direction duty d_low while the mass
    sinks back. Returns (w_lift, turns, early, w_low or None, low_note). Speeds positive when lifting."""
    cm.wait_rest(hat, side)
    i0 = len(hat.data_log)
    p0 = hat.latest(port).pos
    over = lambda: abs(hat.latest(port).pos - p0) > a.max_turns * 360
    t_cmd, t_end, early = cm.hold_duty(hat, side, a.lift_dir * duty, a.hold, stop_if=over)
    top = hat.latest(port).pos
    turns = abs(top - p0) / 360
    w_low, note = None, ""
    if d_low is not None and a.lift_dir * (top - p0) > 0.4 * 360:
        t_low = time.perf_counter()

        def done():
            pos = hat.latest(port).pos
            if a.lift_dir * (pos - p0) < 0.15 * 360:
                return True
            return time.perf_counter() - t_low > 1.0 and a.lift_dir * (pos - top) > -10
        t2, t3, _ = cm.hold_duty(hat, side, a.lift_dir * d_low, 4.0, stop_if=done)
        hat.coast()
        t, pos = cm.series(cm.rows_for(hat, side, i0), port)
        if a.lift_dir * (hat.latest(port).pos - top) > -10 and t3 - t2 > 0.9:
            note = "held (did not sink)"
        if t3 - t2 > 0.4:
            w, _ = fit.steady_speed(t, pos, t2 + 0.15 + 0.25 * (t3 - t2 - 0.15), t3 - 0.05)
            w_low = a.lift_dir * w
    else:
        hat.coast()
        if d_low is not None:
            note = "not lifted far enough to lower"
    t, pos = cm.series(cm.rows_for(hat, side, i0), port)
    w, _ = fit.steady_speed(t, pos, t_cmd + (t_end - t_cmd) / 2, t_end)
    return a.lift_dir * w, turns, early, w_low, note


def find_liftoff(hat, side, port, a):
    """Ramp the duty up in 0.02 steps; return the first duty that turns the wheel > 2 deg in 0.3 s."""
    cm.wait_rest(hat, side)
    d = 0.0
    p_start = hat.latest(port).pos
    while d < cm_cfg.MEASURE_DUTY_CAP - 1e-9:
        d = round(d + 0.02, 3)
        p0 = hat.latest(port).pos
        cm.hold_duty(hat, side, a.lift_dir * d, 0.3)
        moved = a.lift_dir * (hat.latest(port).pos - p0)
        if moved > 2:
            hat.coast()
            return d
        if abs(hat.latest(port).pos - p_start) > a.max_turns * 360:
            break
    hat.coast()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--radius-mm", type=float, default=34.5)
    ap.add_argument("--duties", type=float, nargs="+", default=[0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--lower-duties", type=float, nargs="*", default=[0.0, 0.05, 0.1, 0.15, 0.2],
                    help="lift-direction duties used, in turn, to lower the mass after each lift (none: no lowering)")
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--max-turns", type=float, default=1.5)
    ap.add_argument("--lift-dir", type=int, choices=(1, -1), default=1)
    ap.add_argument("--hold-search", action="store_true", help="also find the lift-off duty for each mass")
    ap.add_argument("--direction-test", action="store_true", help="0.4 s pulse at 0.15 duty to check --lift-dir, then exit")
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cm_cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.check_duties(a.duties)
    side, port = a.side, cm.SIDE_PORT[a.side]

    print(__doc__.split("PROCEDURE")[0].split("SETUP")[1])
    cm.ask("Set up the rig as above, keep a hand near the mass, then press Enter. ")
    hat, vin = cm.open_hat(a.pwmparams)
    print(f"vin {vin:.2f} V; {side} wheel on port {port}; lift direction {a.lift_dir:+d}")

    if a.direction_test:
        cm.ask("No mass on the string. Press Enter for the 0.4 s test pulse. ")
        p0 = hat.latest(port).pos
        cm.hold_duty(hat, side, a.lift_dir * 0.15, 0.4)
        cm.stop(hat)
        time.sleep(0.2)
        print(f"wheel turned {a.lift_dir * (hat.latest(port).pos - p0):+.0f} degrees in the lift direction. "
              "If the string wound UP, --lift-dir is right; if it unwound, use the other sign.")
        hat.close()
        return

    r = a.radius_mm / 1000.0
    table = []                       # (mass_g, duty, tau_L, w_loaded, turns, early)
    lowers = []                      # (mass_g, lift duty before, lower duty, tau_L, w_low, note)
    w_nl = {}                        # duty -> no-load speed with the string on
    liftoff = {}                     # mass_g -> duty or None
    base = cm.base_path(a.outdir, cm.stamp(), "load")
    try:
        while True:
            m_g = cm.ask_float("\nMass in grams (0 first; blank to finish): ")
            if m_g is None:
                break
            if m_g > 0 and not w_nl:
                print("  run 0 g first: the no-load speeds come from it")
                continue
            cm.read_vin(hat)
            cm.ask(f"Hang {m_g:g} g, resting on the towel with the string taut. Press Enter to start the lifts. ")
            tau_l = m_g / 1000.0 * G * r
            for i, d in enumerate(a.duties):
                if m_g == 0:
                    w, turns, early = lift_run(hat, side, port, d, a)
                    w_nl.setdefault(d, []).append(w)
                    w_low = None
                else:
                    d_low = a.lower_duties[i % len(a.lower_duties)] if a.lower_duties else None
                    w, turns, early, w_low, note = lift_lower(hat, side, port, d, d_low, a)
                    if d_low is not None:
                        lowers.append((m_g, d, d_low, tau_l, w_low, note))
                cm.check_fault(hat)
                table.append((m_g, d, tau_l, w, turns, early))
                low = "" if w_low is None else f";  lowered at duty {d_low:.2f}: {w_low:6.2f} rad/s"
                if m_g > 0 and note:
                    low += f"  ({note})"
                print(f"  duty {d:.2f}: lift speed {w:6.2f} rad/s over {turns:.2f} turns"
                      f"{'  (stopped at max turns)' if early else ''}{low}")
                if m_g == 0:
                    cm.ask("  Re-wind/re-tension the string if needed, then press Enter. ")
                else:
                    time.sleep(0.5)
            if a.hold_search:
                d_lift = find_liftoff(hat, side, port, a)
                liftoff[m_g] = d_lift
                print(f"  lift-off duty: {d_lift}")
                cm.ask("  Lower the mass back / re-tension the string, then press Enter. ")
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mass_g", "duty", "tau_L_Nm", "w_lift_rad_s", "turns", "stopped_early"])
            w.writerows(table)
        with open(base + "_lower.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mass_g", "lift_duty_before", "lower_duty", "tau_L_Nm", "w_lower_rad_s", "note"])
            w.writerows(lowers)
        cm.write_raw_csv(base + "_raw.csv", list(hat.data_log))

    # ---- fit ----
    consts, diag = {}, {"liftoff_duty_by_mass_g": liftoff}
    nl = {d: float(np.mean(v)) for d, v in w_nl.items()}
    rows = [(d, tl, w, nl[d]) for m, d, tl, w, _, _ in table if m > 0 and d in nl]
    print("\n=== load fit ===")
    res = None
    if len(rows) >= 2:
        try:
            res = fit.fit_load_drop(rows)
        except ValueError as e:
            print("  no fit:", e)
    if res is not None:
        consts[cm.key("c_load", side)] = cm.const(res["c"], res["se"], "N m s/rad", res["n"], "ls")
        diag[cm.key("c_load_resid_by_tau_L", side)] = res["per_mass"]
        print(cm.show("c = a k + b", consts[cm.key("c_load", side)]))
        print("  residual by load (N m): " + ", ".join(f"{k:.4f}: {v:+.4f}" for k, v in res["per_mass"].items()))
        print("  (a = c / k_eff, with k_eff from motor_steps; residuals trending with load mean the model is not linear)")
    elif len(rows) < 2:
        print("  need a 0 g run and at least one loaded run to fit c")
    pts = [(m / 1000.0 * G * r, d) for m, d in liftoff.items() if d is not None]
    for m, d in liftoff.items():
        consts[cm.key(f"liftoff_duty_{m:g}g", side)] = cm.const(d, None, "duty", 1, "none")
    if len(pts) >= 3:
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        coef, se, _ = fit.linfit(np.column_stack([np.ones_like(x), x]), y)
        consts[cm.key("liftoff_intercept", side)] = cm.const(coef[0], se[0], "duty", len(x), "ls")
        consts[cm.key("liftoff_slope", side)] = cm.const(coef[1], se[1], "duty per N m", len(x), "ls")
        print(cm.show("lift-off intercept", consts[cm.key("liftoff_intercept", side)]))
        print(cm.show("lift-off slope", consts[cm.key("liftoff_slope", side)]))
        print(f"  if g(duty) = duty x vin: a = {1 / (coef[1] * vin):.3g} N m/V, tau_c = {coef[0] / coef[1]:.3g} N m")
    for m, d in liftoff.items():
        print(f"  lift-off duty at {m:g} g: {d}")
    if lowers:
        print("  lowering (braking quadrant), speed negative = sinking:")
        for m, dl, dd, tl, wl, note in lowers:
            print(f"    {m:5g} g  duty {dd:.2f}: {'n/a' if wl is None else f'{wl:6.2f} rad/s'}  {note}")
        diag["lowering"] = [dict(mass_g=m, lower_duty=dd, tau_L=tl, w=wl, note=note) for m, dl, dd, tl, wl, note in lowers]
    meta = {"pwmparams": a.pwmparams, "vin": vin, "side": side, "wheel": "on, loaded", "radius_mm": a.radius_mm,
            "lift_dir": a.lift_dir, "args": vars(a), "firmware": hat.version}
    print("wrote", base + ".csv", cm.write_fit_json(base, "load", consts, meta, diag))
    hat.close()


if __name__ == "__main__":
    main()
