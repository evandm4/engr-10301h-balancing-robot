"""Winding resistance from the stall current (DATASHEET 8.2, "R"). Interactive; needs a multimeter.

    cd ~/legobot && robot/run_safe.sh robot.measure.stall --side left --duties 0.2 0.25 0.3

SETUP (this is a loaded run, so no --wheels-off): put the multimeter in series with the battery's positive
lead (current range, 10 A), with the robot on a stand. You hold the chosen wheel stalled by hand (grip the
tire hard) while the script applies --duty / each of --duties (default 0.25, never above 0.35) for 2 s.
First it asks for the idle current: motors off, type the multimeter reading in amps and press Enter. For
each stalled run you press Enter, read the meter during the 2 s drive, and type the current afterwards.

R = d vin / (I - I_idle) for each run, and V = R I (through the origin) over all runs gives R with a
least-squares standard error (se_kind 'ls'; with one run there is no estimate).

ASSUMPTION: the voltage across the winding is g(d) = d vin, ignoring battery sag under the stall current and
the PWM map. That is valid only if motor_steps found the duty map linear for the same --pwmparams; the
script prints a warning to that effect every time. Stall currents are 1-2 A, so the sag across the battery's
internal resistance can add a few percent. The encoder must not move: more than 3 degrees prints "wheel
slipped" and that run is excluded from the fit.
Files: logs/<stamp>_stall.csv (one row per run), _fit.json.
"""
import argparse
import csv

import numpy as np

from .. import hw_config as cfg
from . import common as cm
from . import fit

MAX_DUTY = 0.35


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--duty", type=float, default=None, help="one duty (default 0.25)")
    ap.add_argument("--duties", type=float, nargs="+", default=None, help="several duties, each run --repeats times")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    duties = a.duties if a.duties else [a.duty if a.duty is not None else 0.25]
    if any(not 0 < d <= MAX_DUTY for d in duties):
        raise SystemExit(f"stall duty must be in (0, {MAX_DUTY}]")
    side, port = a.side, cm.SIDE_PORT[a.side]

    print(__doc__.split("R = d vin")[0].split("SETUP")[1])
    print("WARNING: R = d x vin / I assumes the duty map is linear (g = d x vin); trust it only if motor_steps "
          f"showed that for pwmparams {a.pwmparams}.")
    hat, vin = cm.open_hat(a.pwmparams)
    base = cm.base_path(a.outdir, cm.stamp(), "stall")
    i_idle = cm.ask_float("Idle current with motors off (A): ")
    if i_idle is None:
        raise SystemExit("idle current needed")
    runs = []        # (duty, vin, I, I_net, R, moved_deg, used)
    try:
        for d in duties:
            for rep in range(a.repeats):
                cm.ask(f"\nGrip the {side} wheel so it cannot turn. Duty {d:.2f}, run {rep + 1}/{a.repeats}. Press Enter, then read the meter. ")
                v = cm.read_vin(hat)
                p0 = hat.latest(port).pos
                cm.hold_duty(hat, side, d, 2.0)
                moved = abs(hat.latest(port).pos - p0)
                cm.stop(hat)
                cm.check_fault(hat)
                i_tot = cm.ask_float("  current seen during the 2 s (A, blank to discard this run): ")
                if i_tot is None:
                    continue
                if moved > 3:
                    print(f"  wheel slipped ({moved:.0f} deg): run excluded")
                i_net = i_tot - i_idle
                r_ohm = d * v / i_net if i_net > 0 else float("nan")
                runs.append((d, v, i_tot, i_net, r_ohm, moved, moved <= 3 and i_net > 0))
                print(f"  vin {v:.2f} V, V = {d * v:.3f} V, I - I_idle = {i_net:.3f} A  ->  R = {r_ohm:.2f} ohm")
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["duty", "vin", "I_total_A", "I_net_A", "R_ohm", "moved_deg", "used", "I_idle_A"])
            w.writerows([r + (i_idle,) for r in runs])

    used = [r for r in runs if r[6]]
    consts = {}
    print("\n=== stall fit ===")
    if len(used) >= 2:
        V = np.array([r[0] * r[1] for r in used])
        I = np.array([r[3] for r in used])
        coef, se, _ = fit.linfit(I, V)
        consts[cm.key("R", side)] = cm.const(coef[0], se[0], "ohm", len(used), "ls")
        print(cm.show("R (V = R I)", consts[cm.key("R", side)]))
        print("  datasheet candidates: 3.73 or 5.14 ohm")
    elif used:
        consts[cm.key("R", side)] = cm.const(used[0][4], None, "ohm", 1, "none")
        print(cm.show("R (one run)", consts[cm.key("R", side)]))
    else:
        print("  no usable runs")
    meta = {"pwmparams": a.pwmparams, "vin": vin, "side": side, "wheel": "stalled by hand", "i_idle_A": i_idle,
            "assumes": "g(duty) = duty x vin", "args": vars(a), "firmware": hat.version}
    print("wrote", base + ".csv", cm.write_fit_json(base, "stall", consts, meta))
    hat.close()


if __name__ == "__main__":
    main()
