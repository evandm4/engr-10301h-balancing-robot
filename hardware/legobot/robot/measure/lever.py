"""Stall torque against duty with a LEGO lever on a kitchen scale: the absolute torque scale (C3; DATASHEET 8.7).

    cd ~/legobot && robot/run_safe.sh robot.measure.lever --side left --arm-mm 96
    cd ~/legobot && robot/run_safe.sh robot.measure.lever --side left --arm-mm 96 --pwmparams 1 0.01

SETUP. Wheel off one motor; a Technic beam on that motor's axle through its end hole, sticking out sideways;
a pin or short axle through a hole near the far end as a foot, pointing down; the foot resting on a kitchen
scale raised on books so the beam is about level. --arm-mm is 8 mm x (holes from the axle hole to the foot
hole). Hold the robot so the body can't move. Tare the scale with the beam resting on it and the motor off.

PROCEDURE. A 0.5 s test pulse at 0.08 duty checks that the motor presses the beam DOWN (answer y/n; on n
the sign flips and it pulses again). Then for each duty in --duties: press Enter, the duty is held for --hold s
(the line "READ NOW" appears after 1.5 s), the motor coasts, and you type the scale reading in grams.
The encoder moves 4-7 degrees as the free play and the beam flex take up; more than 15 marks a slip.

MODEL. At zero speed the motor's output torque is tau = a g(d) with a = eta k / R (fit.py's notation) and
g(d) the drive voltage. tau = m g_0 L. A linear fit of tau against d x vin gives a (slope, N m/V) and the
dead-band voltage (-intercept / slope). Static friction blurs the zero-speed torque by up to tau_c; the scale
reading is the net torque the beam receives, which is what moves the robot.
Files: logs/<stamp>_lever.csv (one row per reading), _fit.json.
"""
import argparse
import csv
import math
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm
from . import fit

G0 = 9.80665
SLIP_DEG = 15     # degrees; 4-7 deg is normal: gearbox free play and beam flex taking up (first run)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--arm-mm", type=float, required=True, help="axle to foot, 8 mm per hole")
    ap.add_argument("--duties", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7])
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--dir", type=int, choices=(1, -1), default=1, help="sign of the duty that presses down")
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.check_duties(a.duties)
    if max(a.duties) > 0.8:
        raise SystemExit("keep stall duties at or below 0.8 (winding heat)")
    side, port = a.side, cm.SIDE_PORT[a.side]
    arm = a.arm_mm / 1000.0

    print(__doc__.split("PROCEDURE")[0].split("SETUP.")[1])
    cm.ask("Rig set up, scale tared, hand on the robot? Press Enter. ")
    hat, vin = cm.open_hat(a.pwmparams)
    print(f"vin {vin:.2f} V; {side} motor; pwmparams {a.pwmparams}; arm {a.arm_mm:g} mm")

    sign = a.dir
    while True:
        cm.ask(f"Test pulse at duty {sign * 0.08:+.2f}: press Enter and watch the beam. ")
        cm.hold_duty(hat, side, sign * 0.08, 0.5)
        cm.stop(hat)
        ans = cm.ask("Did the beam press DOWN on the scale? [y/n] ").strip().lower()
        if ans.startswith("y"):
            break
        sign = -sign
        print("  flipping the sign; reset the beam onto the scale")

    base = cm.base_path(a.outdir, cm.stamp(), "lever")
    rows = []                         # (duty, grams, torque, vin, moved_deg, slipped)
    try:
        for d in a.duties:
            cm.ask(f"\nDuty {d:.2f}: press Enter, read the scale when READ NOW appears. ")
            p0 = hat.latest(port).pos
            v = cm.read_vin(hat)
            shown = []

            def note():
                if not shown and time.perf_counter() - t0 > 1.5:
                    print("  READ NOW")
                    shown.append(1)
                return False
            t0 = time.perf_counter()
            cm.hold_duty(hat, side, sign * d, a.hold, stop_if=note)
            moved = abs(hat.latest(port).pos - p0)
            cm.stop(hat)
            cm.check_fault(hat)
            g = cm.ask_float("  scale reading, grams (blank to skip): ")
            if g is None:
                continue
            slipped = moved > SLIP_DEG
            tau = g / 1000.0 * G0 * arm
            rows.append((d, g, tau, v, moved, slipped))
            print(f"  {g:g} g -> {tau:.4f} N m{'   (SLIPPED: beam moved %d deg; repeat this duty)' % moved if slipped else ''}")
            time.sleep(1.0)            # let the winding cool a little between holds
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["duty", "grams", "torque_Nm", "vin", "moved_deg", "slipped"])
            w.writerows(rows)

    consts, diag = {}, {"readings": rows}
    good = [r for r in rows if not r[5] and r[1] > 0]
    print("\n=== lever fit ===")
    for d, g, tau, v, moved, slipped in rows:
        print(f"  duty {d:.2f}: {tau:.4f} N m{'  slipped' if slipped else ''}")
    if len(good) >= 3:
        x = np.array([r[0] * r[3] for r in good])
        y = np.array([r[2] for r in good])
        coef, cov, sd = fit.linfit_cov(np.column_stack([np.ones_like(x), x]), y)
        i0, sl = coef
        gvec = np.array([-1 / sl, i0 / sl**2])
        consts[cm.key("a_stall", side)] = cm.const(sl, math.sqrt(cov[1, 1]), "N m/V", len(x), "ls")
        consts[cm.key("v_dead_stall", side)] = cm.const(-i0 / sl, math.sqrt(gvec @ cov @ gvec), "V", len(x), "ls")
        consts[cm.key("tau_stall_full_duty", side)] = cm.const(sl * vin + i0, None, "N m", len(x), "none")
        dev = (y - (i0 + sl * x)) / y.max() * 100
        diag["linearity_max_dev_pct"] = float(np.abs(dev).max())
        print(cm.show("a = torque per volt", consts[cm.key("a_stall", side)]))
        print(cm.show("dead-band voltage", consts[cm.key("v_dead_stall", side)]))
        print(f"  extrapolated stall torque at duty 1: {sl * vin + i0:.3f} N m;  "
              f"largest deviation from linear {diag['linearity_max_dev_pct']:.1f} % of the largest reading")
    else:
        print("  fewer than 3 good readings: no fit")
    meta = {"pwmparams": a.pwmparams, "vin": vin, "side": side, "arm_mm": a.arm_mm, "press_sign": sign,
            "args": vars(a), "firmware": hat.version}
    print("wrote", base + ".csv", cm.write_fit_json(base, "lever", consts, meta, diag))
    hat.close()


if __name__ == "__main__":
    main()
