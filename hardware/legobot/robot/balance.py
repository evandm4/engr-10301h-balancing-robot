"""The balancing loop (SOFTWARE_PLAN D1-D5).

    cd ~/legobot && robot/run_safe.sh robot.balance --bench --seconds 5        # wheels off the ground
    robot/run_safe.sh robot.balance --controller results/legobot/.../cma_es.json --seconds 30
    python -m robot.balance --sim                  # laptop, from hardware/legobot: closed loop with the simulator

Sequence: open the Build HAT and IMU; load calibration.json (balance-point pitch zero, from robot.calibrate);
hold the robot still for 2 s (gyro bias); then hold it upright at its balance point until the loop arms
(within ARM_TILT and ARM_RATE for ARM_HOLD_S); then it balances for --seconds, or until a trip. Every trip
coasts both motors and ends the run. Logs go to logs/<stamp>_<label>{_run.csv,_raw.csv,_summary.txt};
_run.csv is balance_sim.save_trajectory's format (python simulation/animate.py --csv <file> replays it),
with the robot's own estimates as the state.

Each tick (200 Hz, time.perf_counter): read the IMU -> filter -> newest motor data -> safety -> controller
-> feedforward duty per wheel -> one serial command for both motors. See control.py for what is shared
with the simulator, loop_config.py for the limits.

--inject tests one trip deliberately (D3): loop (one 50 ms tick), hang (the loop blocks 0.5 s: the
watchdog thread must coast), stale (motor data treated as old), vin (cutoff raised above the battery),
fault (a fake "Motor power fault"). The tilt and stall trips are tested by hand on the bench.
"""
import argparse
import gc
import math
import signal
import os
import statistics as st
import threading
import time

from . import control
from . import hw_config as hc
from . import loop_config as lc

LEFT, RIGHT = hc.LEFT_PORT, hc.RIGHT_PORT
RAW_COLUMNS = ("tick", "t", "late", "compute", "phase", "imu_t", "imu_read", "pitch_rate_raw", "acc_fwd", "acc_up",
               "tilt", "theta", "gyro", "x", "v", "w_left", "w_right", "pos_left", "pos_right", "age_left",
               "age_right", "torque", "duty_left", "duty_right", "pitch_ref", "goal", "vin", "locked", "cmd_phase")


# ------------------------------------------------------------------------------- real time (D2) ---
class RealClock:
    now = staticmethod(time.perf_counter)

    @staticmethod
    def wait_until(deadline):
        """Sleep to just before the deadline, then spin. Returns how late we are (s, >= 0)."""
        rest = deadline - time.perf_counter() - lc.SPIN_S
        if rest > 0:
            time.sleep(rest)
        while True:
            now = time.perf_counter()
            if now >= deadline:
                return now - deadline


def set_priority(prio, tid=0, cpu=None):
    """SCHED_FIFO at prio for a thread (0 = this one), else nice -10, else nothing. Returns what was set."""
    if cpu is not None:
        try:
            os.sched_setaffinity(tid, {cpu})
        except (AttributeError, OSError):
            pass
    try:
        os.sched_setscheduler(tid, os.SCHED_FIFO, os.sched_param(prio))
        return f"SCHED_FIFO {prio}"
    except (AttributeError, PermissionError, OSError):
        pass
    if tid == 0:
        try:
            os.nice(-10)
            return "nice -10"
        except (AttributeError, PermissionError, OSError):
            pass
    return "normal"


def lock_memory():
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc.mlockall(3) == 0          # MCL_CURRENT | MCL_FUTURE
    except OSError:
        return False


class Watchdog(threading.Thread):
    """Coasts the motors from its own thread if the loop stops ticking while they are driven (D3)."""

    def __init__(self, hat, timeout=lc.WATCHDOG_S):
        super().__init__(name="watchdog", daemon=True)
        self.hat, self.timeout = hat, timeout
        self.last = None              # None = not armed (motors not driven)
        self.fired = None
        self._stop = threading.Event()

    def kick(self):
        self.last = time.perf_counter()

    def run(self):
        while not self._stop.wait(0.01):
            last = self.last
            if last is not None and time.perf_counter() - last > self.timeout:
                try:
                    self.hat.emergency_coast()
                finally:
                    self.fired = time.perf_counter() - last
                # Keep coasting until the run ends: a command the stuck loop was writing could land after
                # the first coast and drive a motor again (review G11).
                while not self._stop.wait(lc.WATCHDOG_REPEAT_S):
                    self.hat.emergency_coast()
                return

    def stop(self):
        self._stop.set()


# ------------------------------------------------------------------------------------ the run ---
def calibrate_gyro(imu, clock, dt, say=lambda m: print(m, flush=True)):
    """D4: the robot still for GYRO_CAL_S; returns (bias, sd, g ratio). Retries each GYRO_CAL_S window that
    shows motion until GYRO_CAL_TIMEOUT_S, then raises the last ValueError."""
    t_give_up = clock.now() + lc.GYRO_CAL_TIMEOUT_S
    while True:
        samples = []
        t_end = clock.now() + lc.GYRO_CAL_S
        deadline = clock.now()
        while clock.now() < t_end:
            deadline += dt
            clock.wait_until(deadline)
            r = imu.read()
            samples.append((r.pitch_rate, r.acc_fwd, r.acc_up))
        try:
            return control.gyro_bias_from(samples)
        except ValueError as e:
            if clock.now() >= t_give_up:
                raise
            say(f"  {e}: keep it still, trying again")


def run(hat, imu, clock, brain, safety, seconds, watchdog=None, on_arm=None, drive=True, inject=None,
        inject_at=1.0, say=print, line_clock=None, pacer=None, rows=None, stats=None):
    """The loop. Returns (raw rows, stop reason, arm time from the start or None).

    rows (a list) and stats (a dict) are filled as the loop goes, so the caller still has them if the loop
    raises: stats gets "t_arm" (s from the start) when it arms and "IMU read errors" (bridged or not).

    pacer (control.Pacer) sets the tick times: locked to line_clock's HAT data-line grid when it has a lock,
    else every dt. line_clock is re-estimated from the left port's line arrivals every LINE_UPDATE_TICKS
    ticks, after that tick's command is out. cmd_phase in the log is when the command write started,
    in s after the grid's latest line (lower envelope of arrivals)."""
    dt = brain.dt
    c = brain.controller
    rows = [] if rows is None else rows
    stats = {} if stats is None else stats
    reason = None
    phase = 0                     # 0 arming, 1 running
    t_arm = None
    still_since = None
    duties = (0.0, 0.0)
    torque = 0.0
    last_vin_request = -1e9
    vin_used_t = None
    injected = False
    r_prev, imu_errors, imu_injected = None, 0, 0
    t0 = clock.now()
    deadline = t0
    tick = 0
    pacer = pacer or control.Pacer(dt)
    locked = False
    need_lock = line_clock is not None      # --no-line-lock runs unlocked, on purpose (not what the sim tuned for)
    say(f"hold the robot upright at its balance point to arm (|pitch| < {lc.ARM_TILT} rad, still for "
        f"{lc.ARM_HOLD_S} s)")
    while True:
        deadline, locked = pacer.next(deadline, clock.now())
        late = clock.wait_until(deadline)
        t = clock.now()
        if late > dt:             # missed whole periods: re-phase rather than run catch-up ticks
            deadline = t
        tick += 1

        # inject a fault (D3 tests), once, inject_at s after arming; a delay here counts as this tick's
        if inject and phase == 1 and not injected and t - t_arm >= inject_at:
            injected = True
            say(f"injecting {inject}")
            if inject == "loop":          # one 50 ms tick (busy, on the real clock)
                clock.wait_until(clock.now() + 0.05)
            elif inject == "hang":        # blocked 0.5 s: the watchdog thread must coast the motors
                time.sleep(0.5)
            elif inject == "vin":
                safety.vin_cutoff = 99.0
            elif inject == "fault":
                hat.fault = (t, "Motor power fault (injected)")
            # "imu": the next IMU_ERROR_LIMIT reads raise OSError (below)
            elif inject == "unlock" and line_clock is not None:
                line_clock.stale = -1.0       # locked() is false from now on
            if inject in ("loop", "hang"):
                late += clock.now() - t
                t = clock.now()
                deadline = t

        # the watchdog (it already coasted the motors) names the cause before the loop's own stall check
        if watchdog is not None and watchdog.fired is not None:
            reason = f"watchdog fired: loop silent {watchdog.fired * 1e3:.0f} ms"
        elif late > lc.STALL_LOOP_S and phase == 1:
            reason = f"loop stalled: tick {late * 1e3:.1f} ms late"

        # An I2C error (motor noise can NACK a read) is bridged with the last sample for up to IMU_ERROR_LIMIT - 1
        # ticks, then trips like any fault, so the motors coast and the logs are written (review G2).
        try:
            if inject == "imu" and injected and imu_injected < lc.IMU_ERROR_LIMIT:
                imu_injected += 1
                raise OSError(121, "Remote I/O error (injected)")
            r = imu.read()
            imu_errors = 0
        except OSError as e:
            imu_errors += 1
            stats["IMU read errors"] = stats.get("IMU read errors", 0) + 1
            if r_prev is None or imu_errors >= lc.IMU_ERROR_LIMIT:
                reason = reason or f"IMU read failed {imu_errors} times in a row: {e}"
                hat.coast()
                break
            r = r_prev
        r_prev = r
        pl, pr = hat.pair(LEFT), hat.pair(RIGHT)
        state, ex = brain.sense(r.pitch_rate, r.acc_fwd, r.acc_up, pl, pr)
        theta, gyro = state[2], state[3]
        ages = [None if p[1] is None else t - p[1].t for p in (pl, pr)]
        if inject == "stale" and injected:
            ages = [1.0, 1.0]

        # battery: ask every VIN_PERIOD_S; use each new reply for the duty scale
        if t - last_vin_request >= lc.VIN_PERIOD_S:
            hat.request_vin()
            last_vin_request = t
        vr = hat.vin_reading
        if vr is not None and vr[0] != vin_used_t:
            vin_used_t = vr[0]
            brain.feedforward.set_supply(vr[1])

        reason = reason or safety.check(t, theta, duties if drive else (0.0, 0.0), (ex["w_left"], ex["w_right"]),
                                        ages, vr, hat.fault, hat.reader_alive(), phase == 1)
        # The simulator the gains were tuned in assumes the loop is locked to the HAT's pulse ticks
        # (legobot_config ACTUATOR.pwm_tick_locked, review G6): unlocked, a command can wait up to a whole
        # extra tick. So no arming without the lock, and losing it while armed ends the run.
        if not reason and phase == 1 and need_lock and not locked:
            reason = "lost the lock to the HAT's line grid (the tuned gains assume it)"
        if reason:
            hat.coast()
        elif phase == 0:
            if abs(theta) < lc.ARM_TILT and abs(gyro) < lc.ARM_RATE and (locked or not need_lock):
                still_since = t if still_since is None else still_since
                if t - still_since >= lc.ARM_HOLD_S:
                    phase, t_arm = 1, t
                    stats["t_arm"] = t - t0
                    c.reset()
                    if on_arm is not None:
                        on_arm()
                    if watchdog is not None:
                        watchdog.kick()
                    say(f"armed at {t - t0:.2f} s (pitch {theta:+.3f} rad)")
            else:
                still_since = None
            if phase == 0 and tick % 200 == 0:     # once a second: how far from the arming window
                say(f"  waiting to arm: pitch {theta:+.3f} rad (need |pitch| < {lc.ARM_TILT}), "
                    f"rate {gyro:+.2f} rad/s (need < {lc.ARM_RATE})"
                    + ("" if locked or not need_lock else "; no lock to the HAT's lines yet"))
            if phase == 0 and t - t0 > lc.ARM_TIMEOUT_S:
                reason = f"not armed within {lc.ARM_TIMEOUT_S:.0f} s"
        t_cmd = None
        if phase == 1 and not reason:
            torque, dl, dr = brain.act(state, ex["w_left"], ex["w_right"])
            duties = (dl, dr)
            if drive:
                t_cmd = clock.now()
                hat.set_pwm(dl, dr)
            if watchdog is not None:
                watchdog.kick()

        rel = 0.0 if t_arm is None else t - t_arm
        rows.append((tick, t - t0, late, clock.now() - t, phase, r.t - t0, r.read_s, r.pitch_rate, r.acc_fwd,
                     r.acc_up, ex["tilt"], theta, gyro, state[0], state[1], ex["w_left"], ex["w_right"],
                     None if pl[1] is None else pl[1].pos, None if pr[1] is None else pr[1].pos,
                     ages[0], ages[1], torque if phase else 0.0, duties[0] if phase else 0.0,
                     duties[1] if phase else 0.0, c.pitch_reference() if phase else 0.0,
                     c.goal_velocity(rel) if phase else 0.0, None if vr is None else vr[1], int(locked),
                     None if t_cmd is None or line_clock is None or line_clock.t_ref is None
                     else line_clock.phase(t_cmd)))
        if line_clock is not None and tick % lc.LINE_UPDATE_TICKS == 0:
            line_clock.update(hat.line_times(LEFT))
        if reason:
            break
        if phase == 1 and t - t_arm >= seconds:
            reason = "done"
            hat.coast()
            break
    return rows, reason, (None if t_arm is None else t_arm - t0)


# ------------------------------------------------------------------------------------ logging (D5) ---
def stats_ms(xs):
    xs = sorted(x * 1e3 for x in xs if x is not None)
    if not xs:
        return "n/a"
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    return (f"mean {st.fmean(xs):.3f}  sd {sd:.3f}  p50 {p(.5):.3f}  p99 {p(.99):.3f}  max {xs[-1]:.3f} ms "
            f"(n={len(xs)})")


def write_logs(base, rows, reason, t_arm, meta, notes):
    """base_raw.csv (every tick), base_run.csv (from arming, save_trajectory format), base_summary.txt."""
    import csv
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    with open(base + "_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RAW_COLUMNS)
        w.writerows(rows)

    i = {name: k for k, name in enumerate(RAW_COLUMNS)}
    run_rows = [r for r in rows if r[i["phase"]] == 1]
    lines = [f"reason: {reason}", f"armed at: {t_arm}", *(f"{k}: {v}" for k, v in notes.items())]
    if run_rows:
        import numpy as np
        control.import_config()
        from balance_sim import SimResult, save_trajectory
        t = np.array([r[i["t"]] - t_arm for r in run_rows])
        states = np.array([[r[i["x"]] - run_rows[0][i["x"]], r[i["v"]], r[i["theta"]], r[i["gyro"]]]
                           for r in run_rows])
        fell = reason.startswith("tilt")
        result = SimResult(t=t, states=states, torques=np.array([r[i["torque"]] for r in run_rows]),
                           velocity_goal=np.array([r[i["goal"]] for r in run_rows]), fell=fell,
                           fall_time=float(t[-1]) if fell else None, duration=float(t[-1]),
                           pitch_reference=np.array([r[i["pitch_ref"]] for r in run_rows]))
        save_trajectory(result, base + "_run.csv", **meta, reason=reason)
        ticks = [r[i["t"]] for r in run_rows]
        periods = [b - a for a, b in zip(ticks, ticks[1:])]
        lates = [r[i["late"]] for r in run_rows]
        lines += [
            f"run: {t[-1]:.2f} s, {len(run_rows)} ticks",
            f"period: {stats_ms(periods)}",
            f"lateness: {stats_ms(lates)}",
            f"overruns (> {lc.OVERRUN_S * 1e3:.1f} ms late): {sum(1 for x in lates if x > lc.OVERRUN_S)}",
            f"compute (IMU read to command sent): {stats_ms([r[i['compute']] for r in run_rows])}",
            f"IMU read: {stats_ms([r[i['imu_read']] for r in run_rows])}",
            f"ticks locked to the HAT's lines: {sum(r[i['locked']] for r in run_rows)} of {len(run_rows)}",
            f"command write phase after the line grid, mod 5 ms: "
            f"{stats_ms([None if r[i['cmd_phase']] is None else r[i['cmd_phase']] % 0.005 for r in run_rows])}",
            f"motor data age at the tick, left: {stats_ms([r[i['age_left']] for r in run_rows])}",
            f"motor data age at the tick, right: {stats_ms([r[i['age_right']] for r in run_rows])}",
            f"pitch: mean {st.fmean(states[:, 2]):+.4f} sd {st.pstdev(states[:, 2]):.4f} rad; "
            f"|duty| max {max(max(abs(r[i['duty_left']]), abs(r[i['duty_right']])) for r in run_rows):.2f}",
        ]
    with open(base + "_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if hasattr(os, "sync"):
        os.sync()             # to the SD card now: a battery swap cuts the Pi's power
    return lines


def coast_check(hat, wait=1.0, window=0.1):
    """Wheel speeds (deg/s, forward-positive) `wait` s after the loop stopped: ~0 if the trip really coasted."""
    hat.coast()
    time.sleep(wait)
    p0 = [hat.latest(p) for p in (LEFT, RIGHT)]
    time.sleep(window)
    p1 = [hat.latest(p) for p in (LEFT, RIGHT)]
    return [None if a is None or b is None else round((b.pos - a.pos) / window, 1) for a, b in zip(p0, p1)]


# ------------------------------------------------------------------------------------ main ---
def deployed_version():
    try:
        with open("DEPLOYED") as f:
            return f.read().strip()
    except OSError:
        return "repo"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controller", default="hand",
                    help="cascaded PID gains: hand | model | path to a tuning result JSON (default hand)")
    ap.add_argument("--goal", choices=("still", "scenario"), default="still")
    ap.add_argument("--seconds", type=float, default=10.0, help="run time after arming")
    ap.add_argument("--bench", action="store_true", help=f"wheels off the ground: |duty| <= {lc.BENCH_DUTY_CAP}")
    ap.add_argument("--duty-cap", type=float)
    ap.add_argument("--ff-gain", type=float, help="feedforward gain (default: the gains file's, else the config's)")
    ap.add_argument("--no-drive", action="store_true", help="compute everything but send no motor commands")
    ap.add_argument("--calibration", default=lc.CALIBRATION_FILE)
    ap.add_argument("--no-calibration", action="store_true", help="pitch zero 0 (no calibration.json)")
    ap.add_argument("--pitch-zero", type=float, help="override the pitch zero (rad), e.g. the tilt on a bench stand")
    ap.add_argument("--no-line-lock", action="store_true",
                    help="tick on the Pi's clock, not the HAT's line grid (bench experiments only: tuned gains assume the lock)")
    ap.add_argument("--line-phase", type=float, help=f"command write phase after the line grid, s "
                    f"(default loop_config.LINE_WRITE_PHASE_S = {lc.LINE_WRITE_PHASE_S})")
    ap.add_argument("--inject", choices=("loop", "hang", "stale", "vin", "fault", "imu", "unlock"))
    ap.add_argument("--inject-at", type=float, default=1.0, help="s after arming")
    ap.add_argument("--sim", action="store_true", help="closed loop with the simulator instead of hardware")
    ap.add_argument("--sim-pitch", type=float, default=0.03, help="--sim: pitch the robot is held at (rad)")
    ap.add_argument("--sim-noise", action="store_true", help="--sim: legobot_config's sensor noise and offsets")
    ap.add_argument("--outdir", default=lc.LOG_DIR)
    ap.add_argument("--label", default="balance")
    a = ap.parse_args(argv)
    # run_safe.sh starts Python as a background job, which inherits SIGINT as ignored: restore Ctrl-C now,
    # so it works while the HAT opens; install_safety() replaces this handler once the HAT is open.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    cfg = control.load_robot_config()
    controller, desc = control.build_controller(cfg, a.controller, a.goal)
    duty_cap = a.duty_cap if a.duty_cap is not None else (lc.BENCH_DUTY_CAP if a.bench else lc.DUTY_CAP)
    if a.bench and duty_cap > lc.BENCH_DUTY_CAP:
        raise SystemExit(f"--bench caps |duty| at {lc.BENCH_DUTY_CAP}")
    ff_gain = a.ff_gain if a.ff_gain is not None else desc["feedforward_gain"]
    dt = cfg.timing.control_dt_s
    hat_period = hc.SELRATE_MS / 1000.0
    notes = {}

    if a.sim:
        from .sim_hw import SimWorld
        world = SimWorld(cfg, pitch=a.sim_pitch, noise=a.sim_noise)
        hat, imu, clock, on_arm = world.hat, world.imu, world.clock, world.release
        pitch_zero, cal = 0.0, {"source": "sim"}
        watchdog = None
    else:
        cal = None if a.no_calibration else control.load_calibration(a.calibration)
        if cal is None and not a.no_calibration:
            raise SystemExit(f"no {a.calibration}: run robot.calibrate first, or pass --no-calibration")
        cal = cal or {}
        if cal.get("port_sign") and {int(k): v for k, v in cal["port_sign"].items()} != hc.PORT_SIGN:
            raise SystemExit("hw_config.PORT_SIGN differs from the one the sign check confirmed: re-run "
                             "robot.calibrate signs")
        pitch_zero = float(cal.get("pitch_zero", 0.0)) if a.pitch_zero is None else a.pitch_zero
        from .buildhat_serial import BuildHat, install_safety
        from .imu import Imu
        hat = BuildHat()
        install_safety(hat)
        hat.configure(plimit=hc.RUN_PLIMIT)
        imu = Imu()
        clock, on_arm = RealClock, None
        watchdog = Watchdog(hat)
        watchdog.start()
        time.sleep(0.3)                       # both data streams running
        vin = hat.vin()
        if vin is None or vin < lc.VIN_CUTOFF:
            hat.close()
            raise SystemExit(f"battery {vin} V: below the {lc.VIN_CUTOFF} V cutoff, or no reply")
        notes["vin at start"] = vin
        # D2: watchdog above the loop, reader just below it, the loop pinned to one core
        notes["watchdog priority"] = set_priority(lc.RT_PRIORITY + 1, watchdog.native_id)
        notes["reader priority"] = set_priority(lc.RT_PRIORITY - 1, hat._reader.native_id)
        notes["loop priority"] = set_priority(lc.RT_PRIORITY, 0, lc.CPU)
        notes["memory locked"] = lock_memory()

    plimit = 1.0 if a.sim else hc.RUN_PLIMIT
    brain = control.Brain(cfg, controller, control.PitchEstimator(cfg, 0.0, pitch_zero),
                          control.Feedforward(cfg, ff_gain, plimit), duty_cap, hat_period)
    if not a.sim and (vin := notes.get("vin at start")):
        brain.feedforward.set_supply(vin)
    # stall threshold under the duty cap, so the stall trip can fire on the bench too
    safety = control.Safety(dt, stall_duty=min(lc.STALL_DUTY, 0.8 * duty_cap))

    print(f"controller {desc['kind']} {desc['spec']}: gains {[round(g, 4) for g in desc['gains']]}, goal {a.goal}, "
          f"duty cap {duty_cap}, feedforward gain {brain.feedforward.gain}, pitch zero {pitch_zero:+.4f} rad")
    print("hold the robot STILL for gyro calibration...", flush=True)
    try:
        bias, sd, g = calibrate_gyro(imu, clock, dt)
    except ValueError as e:
        if not a.sim:
            hat.close()
        raise SystemExit(f"gyro calibration failed: {e}")
    brain.estimator.gyro_bias = bias
    notes.update({"gyro bias": bias, "gyro sd at rest": sd, "accel |g| ratio": g})

    # D2: lock the ticks to the HAT's data-line grid (lines have been arriving through the gyro calibration)
    line_clock = pacer = None
    if lc.LINE_LOCK and not a.no_line_lock:
        line_clock = control.LineClock(nominal_period=hat_period)
        write_phase = lc.LINE_WRITE_PHASE_S if a.line_phase is None else a.line_phase
        pacer = control.Pacer(dt, line_clock, (write_phase - lc.COMPUTE_LEAD_S) % (0.5 * hat_period))
        if line_clock.update(hat.line_times(LEFT)):
            notes["line period at start"] = f"{line_clock.period * 1e3:.5f} ms ({(line_clock.period / hat_period - 1) * 1e6:+.0f} ppm vs nominal)"
            notes["line lock"] = f"command write phase {write_phase * 1e3:.2f} ms, tick offset {pacer.offset * 1e3:.2f} ms"
            lag = control.port_lag(hat.line_times(LEFT), hat.line_times(RIGHT), hat_period)
            notes["right lines lag left"] = "unknown" if lag is None else f"{lag * 1e3:.2f} ms"
            print(f"locked to the HAT's lines: period {line_clock.period * 1e3:.4f} ms; {notes['line lock']}")
        else:
            notes["line lock"] = "no lock at start (too few lines); ticks on the Pi's clock until it locks"
    else:
        notes["line lock"] = "off"
    print(f"gyro bias {bias:+.5f} rad/s (sd {sd:.4f}); |a| = {g:.3f} g")

    gc.collect()
    gc.freeze()
    gc.disable()
    rows, stats, crash = [], {}, None
    try:
        rows, reason, t_arm = run(hat, imu, clock, brain, safety, a.seconds, watchdog, on_arm,
                                  drive=not a.no_drive, inject=a.inject, inject_at=a.inject_at,
                                  line_clock=line_clock, pacer=pacer, rows=rows, stats=stats)
    except Exception as e:          # any bug or I/O error in the loop: still coast, then write the logs (review G2)
        crash = e
        reason, t_arm = f"crashed: {type(e).__name__}: {e}", stats.get("t_arm")
    finally:
        gc.enable()
        if not a.sim:
            hat.emergency_coast()   # first, before the watchdog stops, and without the write lock
        if watchdog is not None:
            watchdog.stop()
        if not a.sim:
            try:
                notes["wheels 1 s after stop (deg/s)"] = coast_check(hat)
            finally:
                hat.close()
                imu.close()
    print(f"stopped: {reason}")
    if stats.get("IMU read errors"):
        notes["IMU read errors"] = stats["IMU read errors"]
    if line_clock is not None and line_clock.t_ref is not None:
        notes["line period at end"] = (f"{line_clock.period * 1e3:.5f} ms "
                                       f"({(line_clock.period / hat_period - 1) * 1e6:+.0f} ppm vs nominal)")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(a.outdir, f"{stamp}_{a.label}")
    meta = {"label": f"{a.label} {stamp}", "source": "sim_hw" if a.sim else "legobot", "controller": desc,
            "duty_cap": duty_cap, "feedforward_gain": brain.feedforward.gain, "calibration": cal,
            "deployed": deployed_version(), "inject": a.inject,
            "note": "state = the robot's own estimates (encoders + complementary filter), not truth"}
    if a.sim:
        meta["sim_pitch"], meta["sim_noise"] = a.sim_pitch, a.sim_noise
    for line in write_logs(base, rows, reason, t_arm, meta, notes):
        print(line)
    print(f"logs: {base}_*")
    if crash is not None:
        raise crash
    return reason


if __name__ == "__main__":
    main()
