"""The Build HAT's PWM pulse train seen by the IMU at about 1 kHz (B1b follow-up; DATASHEET 8.7). Hands-free.

    cd ~/legobot && robot/run_safe.sh robot.measure.pulse_imu --wheels-off

WHEELS OFF THE GROUND, robot secured on its stand. The encoder only updates at 100 Hz, so it can't resolve
the slow-PWM pulses; the chassis can. Each pulse jolts the motor housing, which is bolted to the body. This
script opens the IMU's filters (gyro DLPF cfg 0, 197 Hz; accelerometer cfg 0, 246 Hz; ODR 1.125 kHz) and
reads it as fast as the I2C link allows (about 1.5 kHz, one 12-byte read per sample), while one wheel runs at
each duty in --duties, in each PWM mode in --modes, for --hold s (the duty sent once: re-sending makes no
difference, pwm_probe). A motors-off phase gives the floor.

Analysis per run: the vibration signal is the high-passed accelerometer magnitude plus gyro magnitude,
resampled to 1 kHz. Its envelope (rectified, 5 ms smoothing) is autocorrelated; the first peak between
15 and 500 ms is the pulse period T, and the share of each period the envelope spends above the
midpoint between its 10th and 90th percentiles is the on-fraction (compare with the duty). Also reports
the spectrum's top peaks. Files: logs/<stamp>_pulse_imu.csv (mode, duty, t, gx, gy, gz, ax, ay, az),
_fit.json.
"""
import argparse
import csv
import time

import numpy as np

from .. import hw_config as cfg
from . import common as cm


def analyse(t, g, a):
    """Pulse period (s), on-fraction, envelope contrast, and top spectral peaks (Hz) of one run."""
    tt = np.arange(t[0], t[-1], 0.001)
    sig = np.zeros(len(tt))
    for arr in (a, g):
        x = np.column_stack([np.interp(tt, t, arr[:, k]) for k in range(3)])
        x = x - np.array([np.convolve(x[:, k], np.ones(51) / 51, "same") for k in range(3)]).T   # high-pass ~20 Hz
        sig += np.linalg.norm(x, axis=1) / (np.std(x) + 1e-12)
    env = np.convolve(np.abs(sig), np.ones(5) / 5, "same")
    e = env[50:-50] - env[50:-50].mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac /= ac[0]
    lo, hi = 15, min(500, len(ac) - 1)
    k = lo + int(np.argmax(ac[lo:hi]))
    period = k / 1000.0
    p10, p90 = np.percentile(env, [10, 90])
    on = float(np.mean(env > (p10 + p90) / 2))
    F = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    f = np.fft.rfftfreq(len(e), 0.001)
    top = [round(float(f[i]), 1) for i in np.argsort(F[5:])[::-1][:4] + 5]
    return period, float(ac[k]), on, float(p90 / max(p10, 1e-9)), top


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wheels-off", action="store_true", help="required: wheels clear of the ground")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--duties", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--modes", nargs="+", default=["1,0.01", "0.65,0.01"], help="pwmparams as 'thresh,minpwm'")
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--outdir", default="logs")
    a = ap.parse_args()
    cm.require_wheels_off(a)
    cm.check_duties(a.duties)
    side, port = a.side, cm.SIDE_PORT[a.side]

    cfg.GYRO_DLPFCFG, cfg.ACCEL_DLPFCFG = 0, 0        # open the filters for this script only
    from ..imu import Imu
    imu = Imu()
    modes = [tuple(float(v) for v in m.split(",")) for m in a.modes]
    hat, vin = cm.open_hat(modes[0])
    base = cm.base_path(a.outdir, cm.stamp(), "pulse_imu")
    rows, results = [], []

    def record(label_mode, duty, seconds):
        out = []
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            t, acc, gyro, _ = imu.read_raw()
            out.append((t, *gyro, *acc))
        arr = np.array(out)
        rows.extend((label_mode, duty, *r) for r in out)
        return arr

    try:
        cm.stop(hat)
        time.sleep(0.5)
        floor = record("off", 0.0, a.hold)
        fs = (len(floor) - 1) / (floor[-1, 0] - floor[0, 0])
        print(f"IMU read rate {fs:.0f} Hz; motors off recorded")
        for mode in modes:
            hat.command(f"port {port} ; pwmparams {mode[0]} {mode[1]}", 0.05)
            for d in a.duties:
                cm.check_fault(hat)
                cm.stop(hat)
                cm.wait_rest(hat, side)
                time.sleep(0.3)
                cm.set_side(hat, side, d)
                time.sleep(0.5)                  # past the spin-up
                arr = record(f"{mode[0]:g},{mode[1]:g}", d, a.hold)
                cm.stop(hat)
                T, strength, on, contrast, top = analyse(arr[:, 0], arr[:, 1:4], arr[:, 4:7])
                results.append(dict(mode=list(mode), duty=d, period_s=T, ac_peak=strength, on_fraction=on,
                                    contrast=contrast, top_hz=top))
                print(f"  pwmparams {mode[0]:g} {mode[1]:g}  duty {d:.2f}: period {T * 1e3:5.0f} ms (autocorr {strength:.2f}), "
                      f"on-fraction {on:.2f}, envelope p90/p10 {contrast:4.1f}, spectrum peaks {top} Hz")
    finally:
        try:
            cm.stop(hat)
        except Exception:
            pass
        imu.close()
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode", "duty", "t", "gx", "gy", "gz", "ax", "ay", "az"])
            w.writerows(rows)
    consts = {cm.key(f"period_{r['mode'][0]:g}_{r['duty']:g}", side): cm.const(r["period_s"], None, "s", 1, "none")
              for r in results}
    meta = {"vin": vin, "side": side, "args": vars(a), "firmware": hat.version, "imu_dlpf": "cfg 0 (gyro 197 Hz)"}
    print("wrote", base + ".csv", cm.write_fit_json(base, "pulse_imu", consts, meta, {"results": results}))
    hat.close()


if __name__ == "__main__":
    main()
