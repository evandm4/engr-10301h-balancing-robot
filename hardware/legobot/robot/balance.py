"""The balancing loop (SOFTWARE_PLAN D1-D5).

    cd ~/legobot && robot/run_safe.sh robot.balance --bench --seconds 5        # wheels off the ground
    robot/run_safe.sh robot.balance --controller results/legobot/.../cma_es.json --seconds 30
    robot/run_safe.sh robot.balance --controller policy:l_pid --seconds 20    # a policy bundle in ~/legobot/policies
    python -m robot.balance --sim                  # laptop, from hardware/legobot: closed loop with the simulator

Sequence: open the Build HAT and IMU; load calibration.json (balance-point pitch zero, from robot.calibrate);
for a policy, replay its bundle's golden vectors through it (self-test, refused on any torque more than
GOLDEN_TOL off) and time it; hold the robot still for 2 s (gyro bias); then hold it upright at its balance point until the loop arms
(within ARM_TILT and ARM_RATE for ARM_HOLD_S); then it balances for --seconds, or until a trip. Every trip
coasts both motors and ends the run. Logs go to logs/<stamp>_<label>{_run.csv,_raw.csv,_summary.txt};
_run.csv is balance_sim.save_trajectory's format (python simulation/animate.py --csv <file> replays it),
with the robot's own estimates as the state. For a policy, _raw.csv adds its raw observation vector, pre-clip
output, saturated and integrators-frozen flags and B1's residual torque (P4a), and the summary its envelope
excursions (P4c).

Each tick (200 Hz, time.perf_counter): read the IMU -> filter -> newest motor data -> safety -> controller
-> feedforward duty per wheel -> one serial command for both motors. See control.py for what is shared
with the simulator, loop_config.py for the limits.

--inject tests one trip deliberately (D3): loop (one 50 ms tick), hang (the loop blocks 0.5 s: the
watchdog thread must coast), stale (motor data treated as old), vin (cutoff raised above the battery),
fault (a fake "Motor power fault"), ood (policies: OOD_INJECT_OFFSET added to the measured velocity on every
tick from then on, for --envelope-trip, P5a). The tilt and stall trips are tested by hand on the bench.
"""
import argparse
import gc
import math
import signal
import os
import statistics as st
import sys
import threading
import time

from . import control
from . import hw_config as hc
from . import loop_config as lc

LEFT, RIGHT = hc.LEFT_PORT, hc.RIGHT_PORT
RAW_COLUMNS = ("tick", "t", "late", "compute", "phase", "imu_t", "imu_read", "pitch_rate_raw", "acc_fwd", "acc_up",
               "tilt", "theta", "gyro", "x", "v", "w_left", "w_right", "pos_left", "pos_right", "age_left",
               "age_right", "torque", "duty_left", "duty_right", "pitch_ref", "goal", "vin", "locked", "cmd_phase",
               "cmd_at", "issued_left", "issued_right")
# duty_left/right: the duty WRITTEN to the HAT that tick; issued_left/right: the duty the controller computed that
# tick (Brain.act). They differ only with --added-delay-ms k > 0: duty[n] = issued[n - k], 0 for the first k armed
# ticks. torque (and _run.csv's torque) is always the ISSUED torque, so replay_policy compares like with like.
POLICY_EXTRA_COLUMNS = ("pre_clip", "saturated", "integrators_frozen", "residual_torque")


def policy_columns(controller):
    """The extra _raw.csv columns of a policy run (P4a): obs_<name> for each raw observation (11, or 13 for a
    residual policy), then POLICY_EXTRA_COLUMNS. () for the PID."""
    if not (hasattr(controller, "last_raw") and hasattr(controller, "spec")):
        return ()
    control.import_config()
    from balance_sim.policy import OBSERVATIONS, RESIDUAL_INPUTS
    names = OBSERVATIONS + (RESIDUAL_INPUTS if controller.spec.residual else ())
    return tuple(f"obs_{n}" for n in names) + POLICY_EXTRA_COLUMNS


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
        self._halt = threading.Event()     # not _stop: Thread.join() calls its own _stop()

    def kick(self):
        self.last = time.perf_counter()

    def run(self):
        while not self._halt.wait(0.01):
            last = self.last
            if last is not None and time.perf_counter() - last > self.timeout:
                try:
                    self.hat.emergency_coast()
                finally:
                    self.fired = time.perf_counter() - last
                # Keep coasting until the run ends: a command the stuck loop was writing could land after
                # the first coast and drive a motor again (review G11).
                while not self._halt.wait(lc.WATCHDOG_REPEAT_S):
                    self.hat.emergency_coast()
                return

    def stop(self):
        self._halt.set()
        if threading.current_thread() is not self and self.is_alive():
            self.join(timeout=0.2)


# ------------------------------------------------------------------------------------ the run ---
def calibrate_gyro(imu, clock, dt, say=lambda m: print(m, flush=True)):
    """D4: the robot still for GYRO_CAL_S; returns (bias, sd, g ratio). Retries each GYRO_CAL_S window that
    shows motion until GYRO_CAL_TIMEOUT_S, then raises the last ValueError."""
    t_give_up = clock.now() + lc.GYRO_CAL_TIMEOUT_S
    imu_errors = 0
    while True:
        samples = []
        t_end = clock.now() + lc.GYRO_CAL_S
        deadline = clock.now()
        while clock.now() < t_end:
            deadline += dt
            clock.wait_until(deadline)
            try:
                r = imu.read()
            except OSError:
                # one transient I2C error (motor noise can NACK a read) is bridged by skipping the sample, as
                # run() does; IMU_ERROR_LIMIT in a row raises
                imu_errors += 1
                if imu_errors >= lc.IMU_ERROR_LIMIT:
                    raise
                continue
            imu_errors = 0
            samples.append((r.pitch_rate, r.acc_fwd, r.acc_up))
        try:
            return control.gyro_bias_from(samples)
        except ValueError as e:
            if clock.now() >= t_give_up:
                raise
            say(f"  {e}: keep it still, trying again")


def run(hat, imu, clock, brain, safety, seconds, watchdog=None, on_arm=None, drive=True, inject=None,
        inject_at=1.0, say=print, line_clock=None, pacer=None, rows=None, stats=None, envelope_monitor=None):
    """The loop. Returns (raw rows, stop reason, arm time from the start or None).

    rows (a list) and stats (a dict) are filled as the loop goes, so the caller still has them if the loop
    raises: stats gets "t_arm" (s from the start) when it arms, "IMU read errors" (bridged or not), and
    "acted_ticks" (rows whose torque the controller computed: all armed rows but a last one that tripped
    before the controller ran, or on a non-finite command, which logs torque and duties 0).

    pacer (control.Pacer) sets the tick times: locked to line_clock's HAT data-line grid when it has a lock,
    else every dt. line_clock is re-estimated from the left port's line arrivals every LINE_UPDATE_TICKS
    ticks, after that tick's command is out. cmd_phase in the log is when the command write started,
    in s after the grid's latest line (lower envelope of arrivals; driving runs only). cmd_at is the command
    point, s from the tick start to the write (in --no-drive, to where it would be): what the compute lead
    (P3b) aims at, without the watchdog kick and envelope check the compute column also counts. The goal
    column is the goal the controller used that tick (c.goal_velocity(c._time) just before it ran, its own
    clock as in the simulator; 0 where it didn't act), so a replay can feed it back exactly.

    A non-finite state on an armed tick trips before the controller runs; a non-finite torque or duty trips
    and coasts before the command is sent (a NaN would pass the tilt check and clamp to +cap).
    brain.delay (control.CommandDelay, --added-delay-ms) sits between brain.act and the HAT write: duty_left/right
    in the log (and what Safety's stall checks see next tick) are what was written, issued_left/right what the
    controller computed; trips coast at once, past it.
    For a policy, each row also carries policy_columns(controller) (None where it didn't act), and
    envelope_monitor (control.EnvelopeMonitor) sees each acted tick's raw observation right after the command
    write (so it adds nothing to the compute lead); its trip coasts at once."""
    dt = brain.dt
    c = brain.controller
    delay = getattr(brain, "delay", None)   # control.CommandDelay (--added-delay-ms), between act() and the HAT
    rows = [] if rows is None else rows
    stats = {} if stats is None else stats
    stats["acted_ticks"] = 0
    n_policy = len(policy_columns(c))
    no_policy = (None,) * n_policy
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
    stats["t0"] = t0              # the raw log's time origin (for --log-lines)
    deadline = t0
    tick = 0
    pacer = pacer or control.Pacer(dt)
    locked = was_locked = False
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
                line_clock.stale = -1.0       # locked() is false from now on: the loop falls back to the Pi's clock
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
        if inject == "ood" and injected:  # P5a: a velocity far outside any policy's envelope, logged as measured
            state[1] += lc.OOD_INJECT_OFFSET
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
            if 5.0 <= vr[1] <= 10.0:   # as Safety.check: an implausible reading doesn't rescale the duty
                brain.feedforward.set_supply(vr[1])

        reason = reason or safety.check(t, theta, duties if drive else (0.0, 0.0), (ex["w_left"], ex["w_right"]),
                                        ages, vr, hat.fault, hat.reader_alive(), phase == 1)
        # The line lock is optional. Bench B1 (2026-11-23) showed the HAT's pulse ticks wander against its
        # data lines, so locking can't fix the wait to a pulse, and the simulator draws that wait at random
        # (legobot_config: pwm_tick_locked False). The lock still keeps the motor data's age steady, so the
        # loop paces on it when it has it and falls back to the Pi's clock when it doesn't (Pacer.next).
        # Arming doesn't wait for it, and losing it is logged, not a trip.
        if line_clock is not None and phase == 1 and locked != was_locked:
            if not locked:
                stats["line lock lost while armed"] = stats.get("line lock lost while armed", 0) + 1
                say(f"  line lock lost at {t - t_arm:.2f} s: ticking on the Pi's clock")
            else:
                say(f"  line lock regained at {t - t_arm:.2f} s")
        was_locked = locked
        if reason:
            hat.coast()
        elif phase == 0:
            if abs(theta) < lc.ARM_TILT and abs(gyro) < lc.ARM_RATE:
                still_since = t if still_since is None else still_since
                if t - still_since >= lc.ARM_HOLD_S:
                    phase, t_arm = 1, t
                    stats["t_arm"] = t - t0
                    c.reset()
                    if delay is not None:
                        delay.reset()         # the FIFO starts empty (zeros) at arming, as simulate()'s queue
                    if on_arm is not None:
                        on_arm()
                    if watchdog is not None:
                        watchdog.kick()
                    say(f"armed at {t - t0:.2f} s (pitch {theta:+.3f} rad)")
            else:
                still_since = None
            if phase == 0 and tick % 200 == 0:     # once a second: how far from the arming window
                say(f"  waiting to arm: pitch {theta:+.3f} rad (need |pitch| < {lc.ARM_TILT}), "
                    f"rate {gyro:+.2f} rad/s (need < {lc.ARM_RATE})")
            if phase == 0 and t - t0 > lc.ARM_TIMEOUT_S:
                reason = f"not armed within {lc.ARM_TIMEOUT_S:.0f} s"
        t_cmd = None
        acted = False
        goal_used = 0.0
        issued = (0.0, 0.0)
        rel = 0.0 if t_arm is None else t - t_arm
        if phase == 1 and not reason and not all(math.isfinite(x) for x in state):
            reason = f"non-finite state {list(state)}"   # a NaN would pass the tilt check and clamp to +cap
            hat.coast()
        if phase == 1 and not reason:
            # the goal the controller uses this tick: its own clock (reset at arming, + dt per update), not rel
            goal_used = c.goal_velocity(getattr(c, "_time", rel))
            torque, dl, dr = brain.act(state, ex["w_left"], ex["w_right"])
            acted = True
            if not (math.isfinite(torque) and math.isfinite(dl) and math.isfinite(dr)):
                reason = f"non-finite command: torque {torque}, duties {dl}, {dr}"
                acted = False             # logged as not acted (torque, duties 0; the values are in the reason)
                duties = (0.0, 0.0)       # tripped on what the controller just computed: send nothing
                hat.coast()
            else:
                # --added-delay-ms: the issued duty enters the FIFO, the one issued k ticks ago is written now
                # (k = 0: the same tuple). The controller already kept what it issued (control.CommandDelay).
                issued = (dl, dr)
                duties = delay.push(issued) if delay is not None else issued
                t_cmd = clock.now()       # the command point (cmd_at), in --no-drive too
                if drive:
                    hat.set_pwm(duties[0], duties[1])
                if watchdog is not None:
                    watchdog.kick()
                if envelope_monitor is not None:   # after the write, so it adds nothing to the compute lead
                    reason = envelope_monitor.observe(c.last_raw)
                    if reason:
                        hat.coast()

        row = (tick, t - t0, late, clock.now() - t, phase, r.t - t0, r.read_s, r.pitch_rate, r.acc_fwd,
               r.acc_up, ex["tilt"], theta, gyro, state[0], state[1], ex["w_left"], ex["w_right"],
               None if pl[1] is None else pl[1].pos, None if pr[1] is None else pr[1].pos,
               ages[0], ages[1], torque if acted else 0.0, duties[0] if acted else 0.0,
               duties[1] if acted else 0.0, c.pitch_reference() if phase else 0.0,
               goal_used, None if vr is None else vr[1], int(locked),
               None if not drive or t_cmd is None or line_clock is None or line_clock.t_ref is None
               else line_clock.phase(t_cmd),
               None if t_cmd is None else t_cmd - t,
               issued[0] if acted else 0.0, issued[1] if acted else 0.0)
        if n_policy:
            row += ((*c.last_raw, float(c.last_pre_clip), int(c.last_saturated),
                     int(c.last_integrators_frozen), float(c.last_residual)) if acted else no_policy)
        rows.append(row)
        if acted:
            stats["acted_ticks"] += 1
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


def write_logs(base, rows, reason, t_arm, meta, notes, seconds=None, columns=RAW_COLUMNS, summary_extra=()):
    """base_raw.csv (every tick), base_run.csv (from arming, save_trajectory format), base_summary.txt.

    seconds: the requested run time, the run's SimResult.duration (as balance_sim.simulate's duration, which
    scales the fall penalty); default the logged run's length. columns: the raw rows' header, RAW_COLUMNS plus
    policy_columns() for a policy (its saturation and integrator-freeze fractions then go in the summary, P4c),
    which _run.csv leaves out. summary_extra: more summary lines (the envelope monitor's)."""
    import csv
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    with open(base + "_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
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
                           fall_time=float(t[-1]) if fell else None,
                           duration=float(seconds) if seconds is not None else float(t[-1]),
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
            f"command point (tick start to command write, cmd_at; compute lead target): "
            f"{stats_ms([r[i['cmd_at']] for r in run_rows])}",
            f"IMU read: {stats_ms([r[i['imu_read']] for r in run_rows])}",
            f"ticks locked to the HAT's lines: {sum(r[i['locked']] for r in run_rows)} of {len(run_rows)}",
            f"command write phase after the line grid, mod 5 ms: "
            f"{stats_ms([None if r[i['cmd_phase']] is None else r[i['cmd_phase']] % 0.005 for r in run_rows])}",
            *(f"write-to-tick wait, {side} (ESTIMATED: (loop_config.TICK_CLIFF_S - cmd_phase) mod "
              f"{hc.SELRATE_MS:g} ms; the pulse ticks wander against the lines, B1): "
              f"{stats_ms([None if r[i['cmd_phase']] is None else (cliff - r[i['cmd_phase']]) % (hc.SELRATE_MS / 1e3) for r in run_rows])}"
              for side, cliff in zip(("left", "right"), lc.TICK_CLIFF_S)),
            f"motor data age at the tick, left: {stats_ms([r[i['age_left']] for r in run_rows])}",
            f"motor data age at the tick, right: {stats_ms([r[i['age_right']] for r in run_rows])}",
            f"pitch: mean {st.fmean(states[:, 2]):+.4f} sd {st.pstdev(states[:, 2]):.4f} rad; "
            f"|duty| max {max(max(abs(r[i['duty_left']]), abs(r[i['duty_right']])) for r in run_rows):.2f}",
        ]
        k_delay = int(meta.get("added_delay_ticks") or 0)
        if k_delay and periods:
            mean_p = st.fmean(periods)
            lines.append(f"added delay realised: {k_delay} ticks x mean tick period {mean_p * 1e3:.4f} ms = "
                         f"{k_delay * mean_p * 1e3:.3f} ms (issued_left/right delayed into duty_left/right in _raw.csv)")
        if "pre_clip" in columns:
            k = columns.index("pre_clip")
            acted = [r for r in run_rows if len(r) > k and r[k] is not None]
            n = len(acted)
            sat = sum(r[columns.index("saturated")] for r in acted)
            frozen = sum(r[columns.index("integrators_frozen")] for r in acted)
            pct = lambda c: 100.0 * c / n if n else 0.0
            lines.append(f"saturated: {sat}/{n} ({pct(sat):.1f}%) ticks; integrators frozen: {frozen}/{n} "
                         f"({pct(frozen):.1f}%) ticks")
        lines += list(summary_extra)
    with open(base + "_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    if hasattr(os, "sync"):
        os.sync()             # to the SD card now: a battery swap cuts the Pi's power
    return lines


def write_line_log(base, data_log, t0):
    """--log-lines: base_lines.csv, one row per HAT data line as the reader thread stamped it: t (s, the raw
    log's time origin, so t - age_left at a tick is that tick's newest left line), port, and the raw speed,
    pos, apos (the HAT's signs). For re-checking the line clock (P7 step 1: stamps held for the GIL)."""
    import csv
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    lines = list(data_log)
    with open(base + "_lines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("t", "port", "speed", "pos", "apos"))
        w.writerows((t - t0, p, s, pos, apos) for t, p, s, pos, apos in lines)
    return len(lines)


def coast_check(hat, wait=1.0, window=0.1):
    """Wheel speeds (deg/s, forward-positive) `wait` s after the loop stopped: ~0 if the trip really coasted."""
    hat.coast()
    time.sleep(wait)
    p0 = [hat.latest(p) for p in (LEFT, RIGHT)]
    time.sleep(window)
    p1 = [hat.latest(p) for p in (LEFT, RIGHT)]
    return [None if a is None or b is None else round((b.pos - a.pos) / window, 1) for a, b in zip(p0, p1)]


# ------------------------------------------------------------------------------------ main ---
# ICM-20948 DLPFCFG -> 3 dB bandwidth, Hz (ICM-20948 datasheet, gyro and accel DLPF tables, FCHOICE = 1)
GYRO_DLPF_HZ = (196.6, 151.8, 119.5, 51.2, 23.9, 11.6, 5.7, 361.4)
ACCEL_DLPF_HZ = (246.0, 246.0, 111.4, 50.4, 23.9, 11.5, 5.7, 473.0)


def imu_dlpf_text(readback=None):
    """The IMU's low-pass config for the summary (a covariate: the DLPF's ~3 ms group delay is on the IMU
    path only). readback: Imu.config_readback (register values read back after configuring), if any."""
    text = (f"gyro DLPFCFG {hc.GYRO_DLPFCFG} ({GYRO_DLPF_HZ[hc.GYRO_DLPFCFG]:g} Hz 3 dB), accel DLPFCFG "
            f"{hc.ACCEL_DLPFCFG} ({ACCEL_DLPF_HZ[hc.ACCEL_DLPFCFG]:g} Hz 3 dB), ODR 1125/(1+div) Hz with div "
            f"{hc.GYRO_SMPLRT_DIV}/{hc.ACCEL_SMPLRT_DIV} (hw_config)")
    if readback:
        text += "; readback " + " ".join(f"{k}={v:#04x}" for k, v in readback.items())
    return text


def deployed_version():
    try:
        with open("DEPLOYED") as f:
            return f.read().strip()
    except OSError:
        return "repo"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controller", default="hand",
                    help="hand | model | path to a cascaded-PID tuning result JSON | policy:<name> (a bundle in "
                         f"~/legobot/{lc.POLICIES_DIR}) | a policy bundle folder | a bare policy .npz with "
                         "--allow-unchecked (default hand)")
    ap.add_argument("--goal", choices=("still", "scenario"), default="still")
    ap.add_argument("--seconds", type=float, default=10.0, help="run time after arming")
    ap.add_argument("--bench", action="store_true", help=f"wheels off the ground: |duty| <= {lc.BENCH_DUTY_CAP}")
    ap.add_argument("--duty-cap", type=float)
    ap.add_argument("--ff-gain", type=float, help="feedforward gain (default: the gains file's, else the config's; "
                                                  "refused for a policy, which runs its training config's)")
    ap.add_argument("--ff-speed", choices=control.FF_SPEED_MODES, default="per_wheel",
                    help="back-EMF feedforward speed: each wheel's own (per_wheel, default, as run through E3) or the "
                         "mean of the two (same total torque; damps the heading mode; post-E3 A/B)")
    ap.add_argument("--allow-unchecked", action="store_true",
                    help="allow a bare policy .npz (no manifest, sha256, config check, self-test or envelope)")
    ap.add_argument("--allow-config-mismatch", action="store_true",
                    help="run a policy whose training config differs from legobot_config (logged)")
    ap.add_argument("--reference-forward", action="store_true",
                    help="policies: the reference PolicyController instead of the (bit-identical, faster) "
                         "FastPolicyController, e.g. to time both on the Pi (P3c)")
    ap.add_argument("--envelope-trip", type=float, default=None, metavar="K",
                    help=f"policies: coast if any input stays past K x its envelope's max |value| for "
                         f"{lc.ENVELOPE_TRIP_S} s (default off: excursions are only logged)")
    ap.add_argument("--compute-lead", type=float, default=None,
                    help="tick start to command write, s (default: on the robot, a policy's self-test lead, "
                         f"else loop_config.COMPUTE_LEAD_PERF_S = {lc.COMPUTE_LEAD_PERF_S} under the performance "
                         f"governor or COMPUTE_LEAD_S = {lc.COMPUTE_LEAD_S}; --sim always the latter)")
    ap.add_argument("--no-drive", action="store_true", help="compute everything but send no motor commands")
    ap.add_argument("--calibration", default=lc.CALIBRATION_FILE)
    ap.add_argument("--no-calibration", action="store_true", help="pitch zero 0 (no calibration.json)")
    ap.add_argument("--pitch-zero", type=float, help="override the pitch zero (rad), e.g. the tilt on a bench stand")
    ap.add_argument("--no-line-lock", action="store_true",
                    help="tick on the Pi's clock only, never on the HAT's line grid")
    ap.add_argument("--line-phase", type=float, help=f"command write phase after the line grid, s "
                    f"(default loop_config.LINE_WRITE_PHASE_S = {lc.LINE_WRITE_PHASE_S})")
    ap.add_argument("--line-slope", choices=("lsq", "envelope"), default=None,
                    help=f"line clock period fit (default loop_config.LINE_SLOPE = {lc.LINE_SLOPE})")
    ap.add_argument("--log-lines", action="store_true",
                    help="log every HAT data line's arrival stamp (reader thread) to <log>_lines.csv")
    ap.add_argument("--switch-interval-us", type=float, default=None,
                    help="sys.setswitchinterval during the run, us, 10-10000 (default Python's 5000): how long the "
                         "loop may hold the GIL while the HAT reader waits to stamp a line (P7 step 1)")
    ap.add_argument("--inject", choices=("loop", "hang", "stale", "vin", "fault", "imu", "unlock", "ood"))
    ap.add_argument("--inject-at", type=float, default=1.0, help="s after arming")
    ap.add_argument("--added-delay-ms", type=float, default=0.0, metavar="N",
                    help="E3 Block D: add N ms of latency on the final duty command (after feedforward; the "
                         "simulator's `latency` location) as a FIFO of N / 5 whole loop ticks; N must be a "
                         "non-negative multiple of the 5 ms loop period (default 0). Safety trips bypass it.")
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
    try:
        controller, desc = control.build_controller(cfg, a.controller, a.goal, allow_unchecked=a.allow_unchecked,
                                                    allow_config_mismatch=a.allow_config_mismatch,
                                                    ff_gain_override=a.ff_gain,
                                                    fast_forward=not a.reference_forward)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"controller {a.controller!r} refused: {e}")
    duty_cap = a.duty_cap if a.duty_cap is not None else (lc.BENCH_DUTY_CAP if a.bench else lc.DUTY_CAP)
    if a.bench and duty_cap > lc.BENCH_DUTY_CAP:
        raise SystemExit(f"--bench caps |duty| at {lc.BENCH_DUTY_CAP}")
    ff_gain = a.ff_gain if a.ff_gain is not None else desc["feedforward_gain"]
    dt = cfg.timing.control_dt_s
    hat_period = hc.SELRATE_MS / 1000.0
    notes = {"deployed": deployed_version()}
    try:                                       # refused before any hardware opens
        delay_ticks = control.added_delay_ticks(a.added_delay_ms, dt)
    except ValueError as e:
        raise SystemExit(f"refused: {e}")
    delay_ms = delay_ticks * dt * 1e3
    notes["added delay"] = (f"{delay_ms:.1f} ms ({delay_ticks} ticks)" + (
        f": FIFO on the final duty after feedforward; the first {delay_ticks} armed ticks write duty 0; the "
        "controller keeps its issued commands" if delay_ticks else ""))
    notes["IMU DLPF"] = imu_dlpf_text() + ("; --sim: SimImu has no filter" if a.sim else "")
    notes["feedforward speed"] = a.ff_speed

    # policies (P2e, P4c, P5a): provenance, the envelope monitor, and the checks on the flags that need one
    bundle = getattr(controller, "bundle", None)
    envelope = None if bundle is None else bundle.get("envelope")
    if bundle is not None:
        notes["policy"] = f"{desc['bundle_name']} (kind={desc['policy_kind']}, sha256={desc['sha256']})"
        notes["policy forward pass"] = desc["forward"]
        if desc["unchecked"]:
            notes["policy check"] = ("UNCHECKED bare .npz (--allow-unchecked): no manifest sha256, config check, "
                                     "golden self-test or envelope")
        if desc["config_differences"]:
            notes["config mismatch"] = ", ".join(desc["config_differences"])
    if a.envelope_trip is not None and envelope is None:
        raise SystemExit("--envelope-trip needs a policy bundle with an envelope (envelope.json)")
    if a.inject == "ood":
        if envelope is None:
            raise SystemExit("--inject ood needs a policy bundle with an envelope (envelope.json)")
        if a.envelope_trip is None or not (a.sim or a.bench or a.no_drive):
            raise SystemExit("--inject ood needs --envelope-trip K and one of --sim, --bench or --no-drive: the "
                             f"policy acts on a {lc.OOD_INJECT_OFFSET} m/s phantom velocity, saturated until the trip")
    if a.compute_lead is not None and not (0.0 <= a.compute_lead <= dt):
        raise SystemExit(f"--compute-lead must be within 0..{dt} s, got {a.compute_lead}")
    if a.switch_interval_us is not None and not (math.isfinite(a.switch_interval_us)
                                                 and 10.0 <= a.switch_interval_us <= 10_000.0):
        raise SystemExit(f"--switch-interval-us must be finite and in [10, 10000] us, got {a.switch_interval_us}")
    envelope_monitor = None
    if envelope is not None:
        try:
            envelope_monitor = control.EnvelopeMonitor(envelope, dt, trip_k=a.envelope_trip,
                                                       active_names=bundle["spec"].input_names)
        except (ValueError, KeyError, TypeError) as e:
            raise SystemExit(f"envelope monitor: {e}")
        notes["envelope trip"] = ("off (logged only)" if a.envelope_trip is None else
                                  f"{a.envelope_trip:g} x max |value| for {lc.ENVELOPE_TRIP_S:g} s")

    if a.sim:
        from .sim_hw import SimWorld
        world = SimWorld(cfg, pitch=a.sim_pitch, noise=a.sim_noise)
        hat, imu, clock, on_arm = world.hat, world.imu, world.clock, world.release
        if a.log_lines:
            hat.data_log = []
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
        hat = BuildHat(log_lines=a.log_lines)   # --log-lines: the reader thread appends each line (no loop cost)
        install_safety(hat)
        hat.configure(plimit=hc.RUN_PLIMIT)
        imu = Imu()
        notes["IMU DLPF"] = imu_dlpf_text(getattr(imu, "config_readback", None))
        clock, on_arm = RealClock, None
        watchdog = Watchdog(hat)
        watchdog.start()
        time.sleep(0.3)                       # both data streams running
        vin = hat.vin()
        if vin is None or vin < lc.VIN_CUTOFF:
            watchdog.stop()
            hat.close()
            imu.close()
            raise SystemExit(f"battery {vin} V: below the {lc.VIN_CUTOFF} V cutoff, or no reply")
        notes["vin at start"] = vin
        # D2: watchdog above the loop, reader just below it, the loop pinned to one core
        try:
            notes["watchdog priority"] = set_priority(lc.RT_PRIORITY + 1, watchdog.native_id)
            notes["reader priority"] = set_priority(lc.RT_PRIORITY - 1, hat._reader.native_id)
            notes["loop priority"] = set_priority(lc.RT_PRIORITY, 0, lc.CPU)
            notes["memory locked"] = lock_memory()
        except BaseException:
            watchdog.stop()
            hat.close()
            imu.close()
            raise

    try:
        plimit = 1.0 if a.sim else hc.RUN_PLIMIT
        brain = control.Brain(cfg, controller, control.PitchEstimator(cfg, 0.0, pitch_zero),
                              control.Feedforward(cfg, ff_gain, plimit, speed=a.ff_speed), duty_cap, hat_period,
                              added_delay_ticks=delay_ticks)
        if not a.sim and (vin := notes.get("vin at start")):
            brain.feedforward.set_supply(vin)
        # stall threshold under the duty cap, so the stall trip can fire on the bench too; seed Safety.vin from
        # startup so the fixed 17 rad/s fallback isn't used before the first in-loop vin reply arrives (audit L9)
        safety = control.Safety(dt, stall_duty=min(lc.STALL_DUTY, 0.8 * duty_cap),
                                vin=hat.vin() if a.sim else notes.get("vin at start"))

        if bundle is not None:
            ctrl_text = (f"policy {desc['bundle_name']} (kind {desc['policy_kind']}, {desc['n_params']} params, "
                         f"sha256 {desc['sha256'][:12]})")
        else:
            ctrl_text = f"{desc['kind']} {desc['spec']}: gains {[round(g, 4) for g in desc['gains']]}"
        print(f"controller {ctrl_text}, goal {a.goal}, duty cap {duty_cap}, feedforward gain "
              f"{brain.feedforward.gain}, pitch zero {pitch_zero:+.4f} rad")
        # P2d, P3b: the golden vectors before gyro calibration (so a refusal costs no holding still), at the
        # loop's priority, so the timing is the loop's
        self_test = None
        # P7 steps 2-4: the compute lead depends on the CPU clock; the performance governor's pair is used only
        # when the loop core reads "performance" (control.lead_constants). --sim: unknown -> the fixed lead
        governor = None if a.sim else control.cpu_governor()
        notes["cpu governor"] = governor or "unknown"
        if bundle is not None and bundle.get("golden") is not None:
            self_test = control.self_test_policy(controller, bundle["golden"], cfg, governor=governor)
            notes["self-test"] = (f"{self_test['n_ticks']} golden ticks, max torque error "
                                  f"{self_test['max_torque_err']:.3g} N m (tol {lc.GOLDEN_TOL:g}); forward pass mean "
                                  f"{self_test['mean_s'] * 1e6:.1f} p50 {self_test['p50_s'] * 1e6:.1f} p99 "
                                  f"{self_test['p99_s'] * 1e6:.1f} max {self_test['max_s'] * 1e6:.1f} us; PID p50 "
                                  f"{self_test['pid_p50_s'] * 1e6:.1f} p99 {self_test['pid_p99_s'] * 1e6:.1f} us")
            print(f"policy self-test passed: {notes['self-test']}")
        elif bundle is not None:
            notes["self-test"] = "none (no golden vectors)"
        print("hold the robot STILL for gyro calibration...", flush=True)
        bias, sd, g = calibrate_gyro(imu, clock, dt)
    except BaseException as e:      # ValueError, a second I2C error in a row, Ctrl-C: release the hardware
        if not a.sim:
            watchdog.stop()
            hat.close()
            imu.close()
        if isinstance(e, ValueError):
            raise SystemExit(f"policy self-test, gyro calibration or setup failed: {e}")
        raise
    brain.estimator.gyro_bias = bias
    notes.update({"gyro bias": bias, "gyro sd at rest": sd, "accel |g| ratio": g})

    # P3b: where the tick starts before the command write. The self-test's lead on the robot; in --sim the
    # fixed one, so the virtual clock's tick times don't depend on how busy the laptop is
    if a.compute_lead is not None:
        compute_lead, lead_source = a.compute_lead, "--compute-lead"
    elif self_test is not None and not a.sim:
        compute_lead, lead_source = self_test["compute_lead_s"], "policy self-test"
    else:
        compute_lead = control.lead_constants(governor)[0]
        lead_source = ("loop_config.COMPUTE_LEAD_PERF_S (performance governor)" if governor == "performance"
                       else "loop_config.COMPUTE_LEAD_S")
    notes["compute lead"] = f"{compute_lead * 1e3:.3f} ms ({lead_source})"

    # D2: lock the ticks to the HAT's data-line grid (lines have been arriving through the gyro calibration)
    line_clock = pacer = None
    if lc.LINE_LOCK and not a.no_line_lock:
        line_clock = control.LineClock(nominal_period=hat_period, slope=a.line_slope)
        notes["line period fit"] = line_clock.slope
        write_phase = lc.LINE_WRITE_PHASE_S if a.line_phase is None else a.line_phase
        pacer = control.Pacer(dt, line_clock, (write_phase - compute_lead) % (0.5 * hat_period))
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
    old_switch = sys.getswitchinterval()
    try:
        if a.switch_interval_us is not None:    # inside the try, so the finally always restores it and coasts
            sys.setswitchinterval(a.switch_interval_us * 1e-6)
            notes["GIL switch interval"] = (f"{a.switch_interval_us:g} us requested, "
                                            f"{sys.getswitchinterval() * 1e6:.0f} us set "
                                            f"(--switch-interval-us; Python keeps whole us)")
        rows, reason, t_arm = run(hat, imu, clock, brain, safety, a.seconds, watchdog, on_arm,
                                  drive=not a.no_drive, inject=a.inject, inject_at=a.inject_at,
                                  line_clock=line_clock, pacer=pacer, rows=rows, stats=stats,
                                  envelope_monitor=envelope_monitor)
    except Exception as e:          # any bug or I/O error in the loop: still coast, then write the logs (review G2)
        crash = e
        reason, t_arm = f"crashed: {type(e).__name__}: {e}", stats.get("t_arm")
    finally:
        gc.enable()
        sys.setswitchinterval(old_switch)
        if not a.sim:
            hat.emergency_coast()   # first, before the watchdog stops, and without the write lock
        if watchdog is not None:
            watchdog.stop()
        try:                        # the right-line gap again at the end (a covariate; it can drift in a run)
            lag = control.port_lag(hat.line_times(LEFT), hat.line_times(RIGHT), hat_period)
            if lag is not None:
                notes["right lines lag left at end"] = f"{lag * 1e3:.2f} ms"
        except Exception:
            pass
        if not a.sim:
            try:
                notes["wheels 1 s after stop (deg/s)"] = coast_check(hat)
            finally:
                hat.close()
                imu.close()
    print(f"stopped: {reason}")
    if stats.get("IMU read errors"):
        notes["IMU read errors"] = stats["IMU read errors"]
    if stats.get("line lock lost while armed"):
        notes["line lock lost while armed"] = f"{stats['line lock lost while armed']} times (fell back to the Pi's clock)"
    if line_clock is not None and line_clock.t_ref is not None:
        notes["line period at end"] = (f"{line_clock.period * 1e3:.5f} ms "
                                       f"({(line_clock.period / hat_period - 1) * 1e6:+.0f} ppm vs nominal)")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(a.outdir, f"{stamp}_{a.label}")
    if getattr(hat, "data_log", None) is not None and "t0" in stats:
        n_lines = write_line_log(base, hat.data_log, stats["t0"])
        notes["raw line arrivals"] = f"{n_lines} lines in {os.path.basename(base)}_lines.csv (--log-lines)"
    meta = {"label": f"{a.label} {stamp}", "source": "sim_hw" if a.sim else "legobot", "controller": desc,
            "duty_cap": duty_cap, "feedforward_gain": brain.feedforward.gain, "calibration": cal,
            "deployed": notes["deployed"], "inject": a.inject, "acted_ticks": stats.get("acted_ticks", 0),
            "compute_lead_s": compute_lead,
            "ff_speed": a.ff_speed,
            "added_delay_ms": delay_ms, "added_delay_ticks": delay_ticks,
            "imu_dlpf": {"gyro_dlpfcfg": hc.GYRO_DLPFCFG, "accel_dlpfcfg": hc.ACCEL_DLPFCFG,
                         "gyro_smplrt_div": hc.GYRO_SMPLRT_DIV, "accel_smplrt_div": hc.ACCEL_SMPLRT_DIV,
                         "readback": None if a.sim else getattr(imu, "config_readback", None)},
            "note": "state = the robot's own estimates (encoders + complementary filter), not truth"}
    if bundle is not None:
        meta["self_test"] = self_test
        meta["envelope_trip"] = a.envelope_trip
    if a.sim:
        meta["sim_pitch"], meta["sim_noise"] = a.sim_pitch, a.sim_noise
    columns = RAW_COLUMNS + policy_columns(controller)
    extra = envelope_monitor.summary_lines() if envelope_monitor is not None else ()
    for line in write_logs(base, rows, reason, t_arm, meta, notes, seconds=a.seconds, columns=columns,
                           summary_extra=extra):
        print(line)
    print(f"logs: {base}_*")
    if crash is not None:
        raise crash
    return reason


if __name__ == "__main__":
    main()
