"""Slow-PWM period and whether a new `set` restarts it (B1b; DATASHEET 8.7). Hands-free.

    cd ~/legobot && robot/run_safe.sh robot.measure.pwm_probe --wheels-off --side left

WHEELS OFF THE GROUND. In slow PWM (pwmparams 1 0.01) the no-load speed at duty 0.1 ripples at about 10 Hz
(sd 3.6 rad/s, against 1.2 in the other modes), which suggests a PWM period near 100 ms. The balancing loop
sends a new duty every 5 ms, so what matters is whether each command restarts the cycle. For each duty in
--duties, three ways of holding it for --hold s:
  once      one `set`, nothing re-sent
  same      the same value re-sent every 5 ms (how motor_steps ran)
  dither    alternating d and d + 0.002 every 5 ms (a new value each time, as in balancing)
Reports the mean speed, the ripple sd (speed from position differences on a 100 Hz grid, after the first
0.5 s) and the main ripple frequency. If 'dither' ripples like 'once', the period is the firmware's own; if it
ripples like the other PWM modes (about 1.2 rad/s), each command restarts the cycle.
Files: logs/<stamp>_pwm_probe.csv (raw stream), _fit.json.
"""
import argparse
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm

WAYS = ("once", "same", "dither")


def ripple(t, pos_rad, t0, t1):
    m = (t > t0) & (t < t1)
    if m.sum() < 50:
        return float("nan"), float("nan"), float("nan")
    g = np.arange(t[m][0], t[m][-1], 0.01)
    w = np.diff(np.interp(g, t[m], pos_rad[m])) / 0.01
    F = np.abs(np.fft.rfft((w - w.mean()) * np.hanning(len(w))))
    f = np.fft.rfftfreq(len(w), 0.01)
    k = int(np.argmax(F[3:])) + 3
    return float(w.mean()), float(w.std()), float(f[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required: wheels clear of the ground")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--duties", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--pwmparams", type=float, nargs=2, default=[1.0, 0.01])
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.require_wheels_off(a)
    cm.check_duties(a.duties)
    side, port = a.side, cm.SIDE_PORT[a.side]
    hat, vin = cm.open_hat(a.pwmparams)
    base = cm.base_path(a.outdir, cm.stamp(), "pwm_probe")
    results = []
    try:
        for d in a.duties:
            for way in WAYS:
                cm.check_fault(hat)
                cm.stop(hat)
                cm.wait_rest(hat, side)
                time.sleep(0.3)
                t0 = time.perf_counter()
                cm.set_side(hat, side, d)
                n = 0
                nxt = t0
                while time.perf_counter() - t0 < a.hold:
                    nxt += 0.005
                    cm.sleep_until(nxt)
                    cm.check_fault(hat)
                    n += 1
                    if way == "same":
                        cm.set_side(hat, side, d)
                    elif way == "dither":
                        cm.set_side(hat, side, d + (0.002 if n % 2 else 0.0))
                t1 = time.perf_counter()
                cm.stop(hat)
                t, pos = cm.series(cm.rows_for(hat, side), port)
                w, sd, f = ripple(t, pos, t0 + 0.5, t1)
                results.append(dict(duty=d, way=way, mean=w, sd=sd, freq=f))
                print(f"  duty {d:.2f} {way:6}: speed {w:6.2f} rad/s, ripple sd {sd:5.2f} rad/s at {f:5.1f} Hz")
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        cm.write_raw_csv(base + ".csv", list(hat.data_log))
    consts = {cm.key(f"ripple_sd_{r['way']}_{r['duty']:g}", side): cm.const(r["sd"], None, "rad/s", 1, "none")
              for r in results}
    meta = {"pwmparams": a.pwmparams, "vin": vin, "side": side, "args": vars(a), "firmware": hat.version}
    print("wrote", base + ".csv", cm.write_fit_json(base, "pwm_probe", consts, meta, {"results": results}))
    hat.close()


if __name__ == "__main__":
    main()
