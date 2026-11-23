"""Are the HAT's slow-PWM pulses locked to its 100 Hz data stream? (B1b follow-up; DATASHEET 8.7). Hands-free.

    cd ~/legobot && robot/run_safe.sh robot.measure.tick_lock --wheels-off

WHEELS OFF THE GROUND, robot secured. If one firmware loop both ticks the pulse-density modulation and sends
the motor data lines, every data line marks a tick at a fixed offset, and the robot loop can lock to the data
stream (SOFTWARE_PLAN D2) so the tick phase that the simulation found so important stops being random.

One wheel runs at --duty (default 0.05: an isolated 10 ms pulse every 200 ms) for --seconds while the IMU
is read as fast as possible with its filters opened (as pulse_imu) and every data line is timestamped. Then:
  * the HAT's clock: a least-squares grid t = t0 + n P through the data-line arrival times of the driven
    port (P near 10 ms; the residual sd is the reader thread's timestamp jitter);
  * pulse onsets: upward crossings of the vibration envelope (high-passed accelerometer and gyro magnitude,
    3 ms smoothing) above a threshold, at least 100 ms apart;
  * each onset's phase on the HAT grid, ((t - t0) / P) mod 1, as a fraction of a 10 ms tick.
A lock shows as phases bunched at one value (circular spread well under the 0.29 of uniform phases), and
no trend against time. The spread includes the onset detection's resolution (one IMU sample, about
0.7 ms = 0.07 of a tick) and the line jitter. Also reports the Pi-vs-HAT clock rate difference (ppm) from P.
Files: logs/<stamp>_tick_lock_imu.csv, _lines.csv, _fit.json.
"""
import argparse
import csv
import math
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm

NOMINAL_PERIOD = 0.010


def envelope(t, gyro, acc):
    """Vibration envelope on a 1 kHz grid: high-passed (50 ms moving mean removed) accelerometer and gyro
    magnitudes, each scaled by its sd, summed, rectified and smoothed over 3 ms."""
    tt = np.arange(t[0], t[-1], 0.001)
    sig = np.zeros(len(tt))
    for arr in (acc, gyro):
        x = np.column_stack([np.interp(tt, t, arr[:, k]) for k in range(3)])
        x = x - np.array([np.convolve(x[:, k], np.ones(51) / 51, "same") for k in range(3)]).T
        sig += np.linalg.norm(x, axis=1) / (np.std(x) + 1e-12)
    return tt, np.convolve(np.abs(sig), np.ones(3) / 3, "same")


def onsets(tt, env, refractory=0.1):
    core = env[100:-100]
    p50, p99 = np.percentile(core, [50, 99])
    thr = p50 + 0.5 * (p99 - p50)
    out, last = [], -1e9
    for i in range(101, len(env) - 100):
        if env[i - 1] < thr <= env[i] and tt[i] - last > refractory:
            # interpolate the crossing time between the two samples
            frac = (thr - env[i - 1]) / (env[i] - env[i - 1])
            out.append(tt[i - 1] + frac * (tt[i] - tt[i - 1]))
            last = tt[i]
    return np.array(out), thr


def grid_fit(t_lines):
    """t = t0 + n P through line arrivals. n counts lines (a gap of k periods counts k, so a lost line doesn't
    shift the rest); rounding to a nominal period instead mis-indexes them once the HAT's clock has drifted
    a period against the Pi's. The grid is then moved to the lower envelope (the earliest arrivals, least
    delayed by the reader thread). Returns t0, P, residual sd, residuals."""
    d = np.diff(t_lines)
    k = np.maximum(np.round(d / np.median(d)), 1)
    n = np.concatenate([[0.0], np.cumsum(k)])
    A = np.column_stack([np.ones_like(n), n])
    (t0, P), *_ = np.linalg.lstsq(A, t_lines, rcond=None)
    resid = t_lines - (t0 + P * n)
    t0 += float(np.percentile(resid, 5))
    return float(t0), float(P), float(resid.std()), resid


def circular(phases):
    """Mean phase (fraction of a tick) and circular spread sqrt(-2 ln R) / (2 pi)."""
    z = np.exp(2j * np.pi * phases).mean()
    return float((np.angle(z) / (2 * np.pi)) % 1.0), float(math.sqrt(max(-2 * math.log(max(abs(z), 1e-12)), 0)) / (2 * math.pi)), float(abs(z))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required: wheels clear of the ground")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--duty", type=float, default=0.05)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--pwmparams", type=float, nargs=2, default=[1.0, 0.01])
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.require_wheels_off(a)
    cm.check_duties([a.duty])
    side, port = a.side, cm.SIDE_PORT[a.side]

    cfg.GYRO_DLPFCFG, cfg.ACCEL_DLPFCFG = 0, 0
    from ..imu import Imu
    imu = Imu()
    hat, vin = cm.open_hat(a.pwmparams)
    base = cm.base_path(a.outdir, cm.stamp(), "tick_lock")
    rows = []
    try:
        cm.stop(hat)
        time.sleep(0.5)
        i0 = len(hat.data_log)
        cm.set_side(hat, side, a.duty)
        time.sleep(0.5)
        t_end = time.perf_counter() + a.seconds
        while time.perf_counter() < t_end:
            t, acc, gyro, _ = imu.read_raw()
            rows.append((t, *gyro, *acc))
            if hat.fault:
                break
        cm.stop(hat)
        lines = [r for r in hat.data_log[i0:]]
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        imu.close()
    with open(base + "_imu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "gx", "gy", "gz", "ax", "ay", "az"])
        w.writerows(rows)
    with open(base + "_lines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "port", "speed_raw", "pos_raw", "apos_raw"])
        w.writerows(lines)

    arr = np.array(rows)
    tt, env = envelope(arr[:, 0], arr[:, 1:4], arr[:, 4:7])
    on, thr = onsets(tt, env)
    tl = np.array([r[0] for r in lines if r[1] == port and arr[0, 0] <= r[0] <= arr[-1, 0]])
    t0, P, jitter, _ = grid_fit(tl)
    ph = ((on - t0) / P) % 1.0
    mean, spread, R = circular(ph)
    # trend: unwrap the phases around the mean and fit against time
    d = ((ph - mean + 0.5) % 1.0) - 0.5
    slope = float(np.polyfit(on - on[0], d, 1)[0]) if len(on) > 3 else float("nan")
    gaps = np.diff(on)
    print(f"IMU {len(arr) / (arr[-1, 0] - arr[0, 0]):.0f} Hz; {len(tl)} data lines; {len(on)} pulse onsets "
          f"(median gap {np.median(gaps) * 1e3:.0f} ms)")
    print(f"HAT line grid: period {P * 1e3:.5f} ms ({(P / NOMINAL_PERIOD - 1) * 1e6:+.0f} ppm against the Pi's clock), "
          f"arrival jitter sd {jitter * 1e3:.2f} ms")
    print(f"pulse phase on the line grid: mean {mean:.3f} of a tick ({mean * 10:.2f} ms after the grid), "
          f"circular spread {spread:.3f} (uniform would be about 0.29; R = {R:.2f}), drift {slope * 10:+.3f} ms/s")
    print("  phases:", " ".join(f"{p:.2f}" for p in ph))
    # The rate test is the strong one: onsets every round(1 / duty) ticks. If their period equals that many
    # line periods, both run on the HAT's clock (the Pi's is hundreds of ppm off). Onset detection jitter
    # (a few ms) blurs the phase spread, so it only has to be well below uniform's 0.29.
    per = round(1 / a.duty)
    k = np.round((on - on[0]) / (per * P))
    A = np.column_stack([np.ones_like(k), k])
    (c_, Tp), res_, *_ = np.linalg.lstsq(A, on, rcond=None)
    r_ = on - A @ np.array([c_, Tp])
    se_Tp = math.sqrt(r_ @ r_ / max(len(on) - 2, 1) / np.sum((k - k.mean()) ** 2))
    ppm = (Tp / (per * P) - 1) * 1e6
    print(f"pulse period {Tp * 1e3:.3f} +- {se_Tp * 1e3:.3f} ms against {per} x line period {per * P * 1e3:.3f} ms: "
          f"{ppm:+.0f} +- {se_Tp / (per * P) * 1e6:.0f} ppm")
    locked = abs(Tp - per * P) < 3 * se_Tp + 1e-6 and spread < 0.25
    print("  -> LOCKED: the pulses keep a fixed offset from the data lines" if locked else
          "  -> not clearly locked (see the phases)")
    consts = {"pulse_phase_mean": cm.const(mean, spread / math.sqrt(max(len(ph), 1)), "tick", len(ph), "normal"),
              "pulse_phase_spread": cm.const(spread, None, "tick", len(ph), "none"),
              "hat_line_period": cm.const(P, None, "s", len(tl), "none"),
              "line_jitter_sd": cm.const(jitter, None, "s", len(tl), "none")}
    meta = {"vin": vin, "side": side, "duty": a.duty, "pwmparams": a.pwmparams, "args": vars(a),
            "firmware": hat.version}
    print("wrote", base + "_imu.csv", base + "_lines.csv",
          cm.write_fit_json(base, "tick_lock", consts, meta, {"phases": ph.tolist(), "threshold": thr,
                                                              "drift_ms_per_s": slope * 10, "locked": locked}))
    hat.close()


if __name__ == "__main__":
    main()
