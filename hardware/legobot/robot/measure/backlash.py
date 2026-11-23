"""Gearbox free play (backlash), measured at the encoder and by eye (DATASHEET 8.2). Interactive, motors off.

    cd ~/legobot && robot/run_safe.sh robot.measure.backlash --side left --seconds 10

The motor is put in pwm mode at duty 0 (set_pwm(0, 0), NOT coast), so the output is as stiff as the winding
makes it. Rock the wheel gently by hand back and forth between the two soft stops of the free play for
--seconds, not hard enough to back-drive the gearbox. The script logs POS and APOS and reports the
peak-to-peak position, and from the time spent at each position the separation of the two dwell clusters
(the two ends of the free play) in degrees. It then asks for the free play you saw at the tire tread in mm
(blank to skip) and converts it to degrees of wheel rotation, 360 mm / (pi 69), for a 69 mm tire.

WHY THE VISUAL READING MATTERS. If the encoder sits before the gearbox's last stage, the HAT reports about
0 degrees of free play while the wheel visibly moves, because the play is downstream of the encoder. The
encoder numbers then only measure the play inside the encoder's own loop. The reading at the tread is the
check; the one that matters for control is the play between the motor's position and the wheel, so if the
two disagree, trust the visual one.

Only one measurement per side: the JSON values carry no standard error (se_kind 'none').
Files: logs/<stamp>_backlash.csv (t, side, pos_deg, apos_deg, forward-positive), _fit.json.
"""
import argparse
import csv
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm

WHEEL_DIAMETER_MM = 69.0       # tire diameter used to convert tread travel to wheel rotation


def analyse(t, pos):
    """Peak-to-peak and the two dwell clusters. Returns dict (cluster_sep is nan if one end is barely visited)."""
    p2p = float(pos.max() - pos.min())
    dt = np.diff(t, append=t[-1])
    mid = 0.5 * (pos.max() + pos.min())
    lo, hi = pos < mid, pos >= mid
    out = {"ptp_deg": p2p, "cluster_sep_deg": float("nan"), "dwell_frac_lo": float(dt[lo].sum() / dt.sum())}
    if dt[lo].sum() > 0.15 * dt.sum() and dt[hi].sum() > 0.15 * dt.sum():
        c_lo = float((pos[lo] * dt[lo]).sum() / dt[lo].sum())
        c_hi = float((pos[hi] * dt[hi]).sum() / dt[hi].sum())
        out["cluster_sep_deg"] = c_hi - c_lo
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", choices=("left", "right", "both"), default="both")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--pwmparams", type=float, nargs=2, default=list(cfg.PWMPARAMS))
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()

    hat, vin = cm.open_hat(a.pwmparams)
    base = cm.base_path(a.outdir, cm.stamp(), "backlash")
    consts, diag, out_rows = {}, {}, []
    try:
        for side in cm.sides_of(a.side):
            port = cm.SIDE_PORT[side]
            cm.ask(f"\n{side}: rock the wheel gently between the soft stops for {a.seconds:g} s after you press Enter. ")
            i0 = len(hat.data_log)
            cm.set_side(hat, side, 0.0)
            end = time.perf_counter() + a.seconds
            while time.perf_counter() < end:
                cm.check_fault(hat)
                time.sleep(0.05)
            rows = cm.rows_for(hat, side, i0)
            cm.stop(hat)
            t = np.array([r[0] for r in rows])
            s = cfg.PORT_SIGN[port]
            pos = s * np.array([r[3] for r in rows], float)
            apos = np.degrees(np.unwrap(np.radians(s * np.array([r[4] for r in rows], float))))
            out_rows += [(side, ti, p, av) for ti, p, av in zip(t, pos, apos)]
            if t.size < 20:
                print("  too few samples")
                continue
            res = analyse(t, pos)
            ap_ptp = float(apos.max() - apos.min())
            print(f"  encoder POS peak-to-peak {res['ptp_deg']:.0f} deg, APOS peak-to-peak {ap_ptp:.0f} deg")
            if np.isfinite(res["cluster_sep_deg"]):
                print(f"  dwell clusters separated by {res['cluster_sep_deg']:.2f} deg "
                      f"({100 * res['dwell_frac_lo']:.0f} % of the time in the lower half)")
            else:
                print("  no two clear dwell clusters (one end barely visited): rock it more slowly, pausing at each stop")
            consts[cm.key("backlash_ptp_deg", side)] = cm.const(res["ptp_deg"], None, "deg", 1, "none")
            consts[cm.key("backlash_cluster_deg", side)] = cm.const(res["cluster_sep_deg"], None, "deg", 1, "none")
            mm = cm.ask_float("  free play seen at the tire tread, mm (blank to skip): ")
            if mm is not None:
                deg = 360.0 * mm / (np.pi * WHEEL_DIAMETER_MM)
                consts[cm.key("backlash_visual_deg", side)] = cm.const(deg, None, "deg", 1, "none")
                print(f"  = {deg:.1f} deg of wheel rotation (for a {WHEEL_DIAMETER_MM:g} mm tire)")
                if res["ptp_deg"] < 0.3 * deg:
                    print("  the encoder saw much less than the eye did: the encoder probably sits before the last gear stage")
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["side", "t", "pos_deg_fwd", "apos_deg_fwd"])
            w.writerows(out_rows)
    meta = {"pwmparams": a.pwmparams, "vin": vin, "side": a.side, "wheel": "as built", "args": vars(a),
            "wheel_diameter_mm": WHEEL_DIAMETER_MM, "firmware": hat.version}
    print("wrote", base + ".csv", cm.write_fit_json(base, "backlash", consts, meta))
    hat.close()


if __name__ == "__main__":
    main()
