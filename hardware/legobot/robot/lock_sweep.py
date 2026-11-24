"""Choose the loop's command phase on the Build HAT's line grid (SOFTWARE_PLAN D2). Hands-free.

    cd ~/legobot && robot/run_safe.sh robot.lock_sweep --wheels-off              # about 3 minutes

WHEELS OFF THE GROUND. In slow PWM the HAT's 10 ms pulse ticks are locked to its 100 Hz data lines (C3,
measure/tick_lock.py), so a command written just before a tick is used at once and one written just
after waits nearly 10 ms. This steps one wheel from rest to --duty (1.0: a pulse on every tick) with the
first command written at a chosen phase after the line grid (control.LineClock, the lower envelope of the
line arrivals, the same estimate the loop locks to), and measures the delay to motion as
measure/latency.py does (fit.fit_spinup on the position stream, from the start of the write). Offsets
are interleaved in random order. The delay should be a sawtooth in the offset with a 10 ms drop just
after the tick; the offset with the least mean delay goes in loop_config.LINE_WRITE_PHASE_S.
Files: logs/<stamp>_lock_sweep.csv (one row per trial) and _summary.txt.
"""
import argparse
import csv
import math
import random
import signal
import statistics as st
import time

from . import control
from . import hw_config as cfg
from .measure import common as cm
from .measure import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required: wheels clear of the ground")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--ref-side", choices=("left", "right"), default="left",
                    help="whose lines set the grid (the loop locks to the left port's)")
    ap.add_argument("--offsets-ms", type=float, nargs="+", default=[float(k) for k in range(10)])
    ap.add_argument("--trials", type=int, default=8, help="per offset")
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=0.25)
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    cm.require_wheels_off(a)
    cm.check_duties([a.duty])
    port = cm.SIDE_PORT[a.side]
    ref_port = cm.SIDE_PORT[a.ref_side]

    hat, vin = cm.open_hat(cfg.PWMPARAMS)
    clock = control.LineClock(nominal_period=cfg.SELRATE_MS / 1000.0)
    time.sleep(1.0)                                   # 100 lines for the first lock
    if not clock.update(hat.line_times(ref_port)):
        hat.close()
        raise SystemExit("could not lock to the HAT's lines")
    other_port = cm.SIDE_PORT["right" if a.ref_side == "left" else "left"]
    lag = control.port_lag(hat.line_times(ref_port), hat.line_times(other_port), clock.period)
    lag_text = "unknown" if lag is None else f"{lag * 1e3:.2f} ms"
    plan = [o / 1000.0 for o in a.offsets_ms for _ in range(a.trials)]
    random.shuffle(plan)
    base = cm.base_path(a.outdir, cm.stamp(), "lock_sweep")
    rows = []
    print(f"vin {vin:.2f} V, pwmparams {cfg.PWMPARAMS}, line period {clock.period * 1e3:.4f} ms, "
          f"other port's lines lag the grid by {lag_text}; {len(plan)} trials")
    try:
        for n, offset in enumerate(plan):
            cm.check_fault(hat)
            cm.wait_rest(hat, a.side, timeout=4.0)
            clock.update(hat.line_times(ref_port))
            now = time.perf_counter()
            m = math.ceil((now + 0.003 - clock.t_ref - offset) / clock.period)
            cm.sleep_until(clock.t_ref + m * clock.period + offset)
            i0 = len(hat.data_log)
            t_cmd, _, _ = cm.hold_duty(hat, a.side, a.duty, a.hold)
            hat.coast()
            phase = clock.phase(t_cmd)
            t, pos = cm.series(cm.rows_for(hat, a.side, i0), port)
            try:
                f = fit.fit_spinup(t, pos, t_cmd, t_end=a.hold)
            except ValueError:
                continue
            valid = abs(f.omega) > 5 and 0.0 < f.t_d < 0.04
            rows.append((offset, phase, f.t_d, f.tau, f.omega, f.rms, valid))
            print(f"  {n + 1:3d}/{len(plan)}: offset {offset * 1e3:4.1f} ms (written at {phase * 1e3:5.2f})  "
                  f"t_d {f.t_d * 1e3:5.1f} ms{'' if valid else '  (invalid)'}", flush=True)
            time.sleep(0.2)
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        with open(base + ".csv", "w", newline="") as f_:
            w = csv.writer(f_)
            w.writerow(["offset_s", "write_phase_s", "t_d_s", "tau_s", "w_inf", "rms", "valid"])
            w.writerows(rows)

    lines = [f"lock sweep {a.side} (grid: {a.ref_side} port's lines), duty {a.duty}, pwmparams {cfg.PWMPARAMS}, vin {vin:.2f} V, "
             f"line period {clock.period * 1e3:.4f} ms, other port's lines lag the grid by {lag_text} (at start)",
             "offset ms   n   t_d mean   sd    (ms)"]
    best = None
    for o in sorted(set(r[0] for r in rows)):
        td = [r[2] for r in rows if r[0] == o and r[6]]
        if len(td) < 2:
            continue
        mean, sd = st.fmean(td), st.stdev(td)
        lines.append(f"  {o * 1e3:5.1f}  {len(td):3d}   {mean * 1e3:7.2f}  {sd * 1e3:5.2f}")
        if best is None or mean < best[1]:
            best = (o, mean)
    if best:
        lines.append(f"least delay at a write phase of {best[0] * 1e3:.1f} ms ({best[1] * 1e3:.2f} ms): "
                     f"set loop_config.LINE_WRITE_PHASE_S = {best[0]:.4f}")
    with open(base + "_summary.txt", "w") as f_:
        f_.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", base + ".csv")
    hat.close()


if __name__ == "__main__":
    main()
