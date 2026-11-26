"""Sim-vs-real overlay for logged LEGO robot policy runs (E2): a re-synced "ghost" sim and a free run.

Usage (from the repo root, laptop only):
    python hardware/legobot/robot/sim_vs_real.py 20261126_011006 --mode resync --resync 1.0
    python hardware/legobot/robot/sim_vs_real.py <..._run.csv> --mode free --out C:/balancebot/reports/sim_vs_real

Two ways to put the nominal simulator next to a real run:

  resync (default)  every T s (--resync, default 1.0) a fresh nominal sim starts from the robot's logged state
                    at that tick and runs for T s under the same controller and the same logged goal. The
                    stitched pieces are the "ghost": how far the model drifts from the robot within T.
  free              one sim from the logged start state for the whole run: long-horizon divergence. The
                    wobble's phase desyncs within a few periods, because the PDM pulse train and the backlash
                    events are chaotic (tiny state differences flip which tick a pulse or an impact lands on),
                    so compare wobble amplitude / frequency, not the waveform.

How a window starts from the real state (start_window):
  * body state [x, v, theta, theta_dot]: the robot's own estimate at that tick (encoders + complementary
    filter, robot/control.Brain.sense), NOT truth. The sim then runs on true states; errors are sim truth
    minus real estimate, so they include the estimator's own error (e.g. its balance-point offset).
  * controller: a fresh PolicyController for the logged bundle, its internal state set from the raw log
    (init_controller): the leaky integrals from the previous tick's post-update obs_int_v / obs_int_theta,
    the command history from this tick's obs_cmd_1..4, and the anti-windup freeze from integrators_frozen.
    Its goal is the logged obs_goal sequence from this tick on. No controller or sim code is changed: the
    four attributes reset() sets are overwritten after reset() (tested: test_sim_vs_real.py).
  * actuator chain: the commands already in flight are reconstructed by replaying the LOGGED commands and
    duties (mean of the two wheels' duty) from the start of the run through simulate()'s own latency queue
    and PDM arithmetic (ActuatorChain), on the global step grid. That gives the pending queue, the torque /
    duty reaching the motor, the PDM's last output and its running sum at the window's first step.
  * not observable, so set by a documented default:
      - PWM / PDM phase: the config's pwm_tick_phase (0.25 for plain runs) counted from the run's first
        acted tick. The real loop runs unlocked against the HAT, so the true phase is unknown and drifts;
        the PDM running sum is the replayed estimate, not the HAT's.
      - backlash: --backlash-init closed (default) puts the play closed on the side of the duty reaching
        the motor (|gap| = b, motor side moving with the wheel, encoder shaft on the wheel); "open" uses
        Planar2D.complete_state (open and centred). Zero duty -> open and centred either way.
      - sensor (--sensor pipeline, default): the config's IMU + complementary filter + encoder model
        (360 counts, 10 ms samples, 5 ms delay) with every noise std and bias set to 0, so the controller
        sees what the robot's own estimator would compute, deterministically. Its state is set from the
        log (init_sensor): the filter's estimate is the logged pitch estimate of the previous tick, the
        latest encoder count is floor((x/r - theta) / resolution) at the logged tick one encoder period
        back (the real wheel angle the robot logged), and the previous count is set so the first measured
        speed is the whole count nearest the logged v/r - gyro (the robot averages two wheels, so its speed
        has half-count steps; residual <= ~0.03 m/s, first-command check ~0.02 N m on run 9), with the
        encoder grid restarting at the window. --noise adds the config's noise and biases (seed 7,300,000 + window index; the
        +0.015 rad accelerometer bias then shifts the balance point). --sensor ideal feeds the true sim
        state to the controller instead: that sim shows a ~3-3.8 Hz, ~2x larger wobble the real robot
        doesn't have (the policy was trained with the encoder's delayed, quantized speed), so it is not
        the default.
  * latency: cfg.timing.latency_s = 10 ms by default. The measured sample-to-HAT time is 9.3-9.4 ms typical
    (9.7 ms p99, POLICY_ON_ROBOT_PLAN P3d); simulate() applies a command at the first physics step at or
    after it is due, so on the 2.5 ms grid anything in (7.5, 10] ms is 4 steps = 10 ms. --dt 0.0005
    --latency 0.0095 runs it at 9.5 ms exactly (the PDM tick and control period stay whole step counts).
  * parameters: nominal legobot config (no randomization), supply voltage cfg.actuator.supply_voltage.
  * time: tick i of the log is i x control_dt in the sim (the logged period is 4.998 ms mean, 0.03 ms sd).

Outputs (in --out): <stamp>_<mode>[_T<T>].gif (MP4 if ffmpeg is installed), <stamp>_summary_T<T>.png
(error vs horizon, wobble amplitude and frequency), <stamp>_metrics_T<T>.json.
matplotlib is imported only inside the plotting functions: this module is in robot/, which the Pi-layout
test imports with matplotlib blocked.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import sys
import time as _wall
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

if __package__:
    from . import control
else:
    _hw_legobot = str(Path(__file__).resolve().parents[1])
    if _hw_legobot not in sys.path:
        sys.path.insert(0, _hw_legobot)
    from robot import control

DEFAULT_LOG_DIR = Path(r"C:\balancebot_training\legobot\pi_logs\e2_deploy")
DEFAULT_OUT = Path(r"C:\balancebot\reports\sim_vs_real")
NOISE_SEED_BASE = 7_300_000          # ad-hoc seeds >= 7.3M (project rule)
N_HISTORY = 4
REAL_COLOR, SIM_COLOR, FREE_COLOR = "tab:blue", "tab:orange", "tab:green"
WOBBLE_HIGHPASS_S = 1.0              # moving-average window removed before wobble stats (keeps > ~1 Hz)
WOBBLE_BAND_HZ = (0.7, 20.0)


def _bs():
    """balance_sim's pieces this module uses (imported lazily, after control.import_config puts the repo
    root on sys.path)."""
    control.import_config()
    from balance_sim import actuators, planar2d, simulator
    return actuators, planar2d, simulator


# ----------------------------------------------------------------------------------------- logs ---
@dataclass
class RunLog:
    """One acted run, per controller tick (the run.csv rows, with the raw log's policy internals)."""
    label: str
    state: np.ndarray        # (n, 4) robot estimate [x, v, theta, theta_dot] at each acted tick
    command: np.ndarray      # (n,) torque command
    drive: np.ndarray        # (n,) duty sent to the HAT (mean of the two wheels)
    goal: np.ndarray         # (n,) goal velocity the policy saw
    int_v: np.ndarray        # (n,) leaky integral of goal - v, after that tick's update
    int_theta: np.ndarray    # (n,) leaky integral of theta, after that tick's update
    history: np.ndarray      # (n, 4) obs_cmd_1..4 the policy saw (before that tick's update)
    frozen: np.ndarray       # (n,) integrators frozen on that tick (previous tick saturated)
    control_dt: float
    meta: dict
    t_logged: Optional[np.ndarray] = None   # (n,) the run.csv's own clock
    full_state: Optional[np.ndarray] = None  # (n, 8) hidden model states: synthetic runs only
    theta_meas: Optional[np.ndarray] = None  # (n,) the controller's pitch estimate if not state[:, 2] (synthetic)

    def pitch_estimate(self, k: int) -> float:
        return float((self.state[:, 2] if self.theta_meas is None else self.theta_meas)[k])

    @property
    def n(self) -> int:
        return len(self.command)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n) * self.control_dt


def resolve_run(target, log_dir=DEFAULT_LOG_DIR) -> tuple[Path, Path]:
    """(run_csv, raw_csv) from a path to either file / their base, or a stamp found in log_dir."""
    text = str(target)
    for suffix in ("_run.csv", "_raw.csv", "_summary.txt"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    run, raw = Path(text + "_run.csv"), Path(text + "_raw.csv")
    if not run.is_file():
        hits = sorted(glob.glob(str(Path(log_dir) / f"{Path(text).name}*_run.csv")))
        if len(hits) != 1:
            raise FileNotFoundError(f"{target!r}: {len(hits)} matching *_run.csv in {log_dir}")
        run = Path(hits[0])
        raw = Path(str(run)[:-len("_run.csv")] + "_raw.csv")
    if not raw.is_file():
        raise FileNotFoundError(f"{raw} is missing: the policy's internal state comes from the raw log")
    return run, raw


def _flag(text: str) -> bool:
    return text.strip().lower() in ("1", "true")


def load_run(target, log_dir=DEFAULT_LOG_DIR, control_dt: Optional[float] = None) -> RunLog:
    run_csv, raw_csv = resolve_run(target, log_dir)
    with open(run_csv, encoding="utf-8") as f:
        first = f.readline()
        meta = json.loads(first[1:]) if first.startswith("#") else {}
        rows = list(csv.DictReader(f if first.startswith("#") else open(run_csv, encoding="utf-8")))
    with open(raw_csv, encoding="utf-8") as f:
        raw = [r for r in csv.DictReader(f) if r["phase"] == "1" and r["pre_clip"] != ""]
    if len(raw) != len(rows):
        raise ValueError(f"{len(raw)} acted raw rows but {len(rows)} run rows")
    col = lambda rs, k: np.array([float(r[k]) for r in rs])
    state = np.column_stack([col(rows, k) for k in ("x", "x_dot", "theta", "theta_dot")])
    command = col(rows, "torque")
    if np.max(np.abs(col(raw, "torque") - command)) > 0 or np.max(np.abs(col(raw, "theta") - state[:, 2])) > 0:
        raise ValueError("raw and run logs disagree")
    if control_dt is None:
        control_dt = control.load_robot_config().timing.control_dt_s
    meta = dict(meta, run_csv=str(run_csv), raw_csv=str(raw_csv), vin_mean=float(col(raw, "vin").mean()))
    # --added-delay-ms runs: the ISSUED duty (issued_left/right) goes through the sim's latency queue with the added
    # delay on top (resync), which is the same as the written duty (duty_left/right) through the base latency
    issued = "issued_left" in raw[0] and float(meta.get("added_delay_ms") or 0.0) > 0
    dl, dr = ("issued_left", "issued_right") if issued else ("duty_left", "duty_right")
    return RunLog(
        label=run_csv.name[:-len("_run.csv")], state=state, command=command,
        drive=0.5 * (col(raw, dl) + col(raw, dr)), goal=col(raw, "obs_goal"),
        int_v=col(raw, "obs_int_v"), int_theta=col(raw, "obs_int_theta"),
        history=np.column_stack([col(raw, f"obs_cmd_{i}") for i in range(1, N_HISTORY + 1)]),
        frozen=np.array([_flag(r["integrators_frozen"]) for r in raw]), control_dt=float(control_dt),
        meta=meta, t_logged=col(rows, "t"))


def load_bundle(log: RunLog, cfg, bundle=None) -> dict:
    target = bundle
    ctrl = log.meta.get("controller") or {}
    if target is None:
        if ctrl.get("kind") not in (None, "policy"):
            raise ValueError(f"run controller is {ctrl.get('kind')!r}; this tool replays policy bundles")
        target = f"policy:{ctrl['bundle_name']}" if ctrl.get("bundle_name") else ctrl.get("spec")
    b = control.load_policy_bundle(target, cfg, allow_config_mismatch=True)
    if ctrl.get("sha256") and b.get("sha256") and ctrl["sha256"] != b["sha256"] and bundle is None:
        raise ValueError(f"bundle sha {b['sha256'][:12]} != logged {ctrl['sha256'][:12]}")
    return b


# --------------------------------------------------------------------------------- the sim side ---
def latency_steps(latency: float, dt: float) -> int:
    """simulate()'s rule (due at t_cmd + latency, applied at the first step t with due <= t + 1e-12) in steps."""
    return max(0, int(math.ceil((latency - 1e-12) / dt)))


class ActuatorChain:
    """simulate()'s actuator path (latency queue, then the Build HAT's PDM) as an object, so its state can be
    reconstructed from logged commands and handed to a window. Steps are GLOBAL physics steps from the run's
    first acted tick, so the PDM phase carries across windows."""

    def __init__(self, dynamics, dt: float, latency: float, pwm_phase: Optional[float] = None):
        act_mod, _, _ = _bs()
        act = getattr(dynamics, "actuator", None)
        self.delay = latency_steps(latency, dt)
        self.pdm_steps = act_mod.pwm_steps_per_tick(getattr(act, "pwm_tick", 0.0), dt)
        phase = getattr(act, "pwm_tick_phase", 0.25) if pwm_phase is None else pwm_phase
        self.pdm_offset = act_mod.pwm_tick_offset(phase, self.pdm_steps)
        self._pdm_update = act_mod.pdm_update
        self.pending = deque()   # (due step, torque command, drive signal)
        self.torque = 0.0        # command reaching the actuator
        self.drive = 0.0         # its drive signal (duty)
        self.applied = 0.0       # what reaches the motor this step (PDM output)
        self.pdm_total = 0.0

    def command(self, k: int, torque: float, drive: float) -> None:
        self.pending.append((k + self.delay, torque, drive))

    def advance(self, k: int) -> float:
        while self.pending and self.pending[0][0] <= k:
            _, self.torque, self.drive = self.pending.popleft()
        if not self.pdm_steps:
            self.applied = self.drive
        elif k >= self.pdm_offset and (k - self.pdm_offset) % self.pdm_steps == 0:
            self.pdm_total, self.applied = self._pdm_update(self.pdm_total, self.drive)
        return self.applied

    def copy(self) -> "ActuatorChain":
        c = copy.copy(self)
        c.pending = deque(self.pending)
        return c


@dataclass
class WindowResult:
    states: np.ndarray       # (steps + 1, dim) every physics step
    torques: np.ndarray      # (steps + 1,) command reaching the actuator (as SimResult.torques)
    commands: np.ndarray     # (ticks,) controller outputs
    fell: bool


def run_window(dynamics, controller, state, n_steps: int, dt: float, control_dt: float, chain: ActuatorChain,
               step0: int = 0, sensor=None, on_control: Optional[Callable] = None,
               reset_sensor: bool = True) -> WindowResult:
    """simulate()'s loop, step for step, from a given state, controller and actuator chain, all already set
    up: the controller is NOT reset (init_controller sets it), the chain is advanced in place, and the
    sensor is reset unless reset_sensor=False (init_sensor set it). step0 is the window's first global
    step (PDM phase and due times). With step0 = 0, a fresh chain, a reset controller and sensor this is
    bit-identical to simulate() (test_sim_vs_real.py). on_control(k, measured, command, drive) is called
    after every controller update."""
    _, _, sim = _bs()
    spc = sim.steps_per_control(control_dt, dt)
    backlash = getattr(dynamics, "has_backlash", False)
    state = np.array(state, dtype=float)
    if backlash:
        state = dynamics.complete_state(state)
    if sensor is not None and reset_sensor:
        sensor.reset()
    imu = sensor is not None and getattr(sensor, "needs_accelerations", False)
    encoder = sensor is not None and getattr(sensor, "has_encoder", False)
    to_drive = getattr(dynamics, "drive", None)
    states, torques, commands, fell = [state.copy()], [], [], False
    for k in range(n_steps):
        t = k * dt
        kg = step0 + k
        if encoder:
            sensor.observe(t, state)
        if k % spc == 0:
            if sensor is None:
                measured = state
            elif imu:
                sense_state = dynamics.begin_step(state, chain.applied) if backlash else state
                derivs = dynamics.derivatives(sense_state, chain.applied)
                measured = sensor.measure(state, (derivs[1], derivs[3]))
            else:
                measured = sensor.measure(state)
            command = controller.update(measured, control_dt)
            drive = command if to_drive is None else to_drive(command, measured)
            chain.command(kg, command, drive)
            commands.append(command)
            if on_control is not None:
                on_control(k, measured, command, drive)
        applied = chain.advance(kg)
        torques.append(chain.torque)
        state = dynamics.step(state, applied, dt) if backlash else sim._rk4_step(dynamics, state, applied, dt)
        states.append(state.copy())
        if dynamics.is_fallen(state):
            fell = True
            break
    torques.append(chain.torque)
    return WindowResult(np.array(states), np.array(torques), np.array(commands), fell)


def goal_from_log(goals: np.ndarray, k0: int, control_dt: float) -> Callable[[float], float]:
    """The logged goal sequence from tick k0 on, as a function of the controller's own clock (0 at reset)."""
    n = len(goals)
    return lambda t: float(goals[min(n - 1, k0 + int(round(t / control_dt)))])


def init_controller(controller, log: RunLog, k0: int) -> None:
    """Reset, then set the policy's internal state to the robot's just before tick k0's update."""
    controller.reset()
    if k0 > 0:
        controller._int_v = float(log.int_v[k0 - 1])
        controller._int_theta = float(log.int_theta[k0 - 1])
    controller._history = [float(h) for h in log.history[k0]]
    controller._saturated = bool(log.frozen[k0])


def build_sensor(cfg, kind: str = "pipeline", noise: bool = False, seed: int = NOISE_SEED_BASE):
    """None for "ideal"; else the config's sensor, with noise and biases zeroed unless noise."""
    if kind == "ideal":
        if noise:
            raise ValueError("--noise needs the sensor pipeline")
        return None
    if kind != "pipeline":
        raise ValueError(f"sensor {kind!r}")
    sc = cfg.sensor
    if not noise:
        sc = replace(sc, pitch_std=0.0, pitch_bias=0.0, pitch_rate_std=0.0, pitch_rate_bias=0.0, position_std=0.0,
                     velocity_std=0.0)
    return sc.build(seed, gravity=cfg.robot.gravity, control_dt=cfg.timing.control_dt_s,
                    wheel_radius=cfg.robot.wheel_radius)


def init_sensor(sensor, log: RunLog, k0: int) -> None:
    """Reset, then give the sensor the robot's estimator state just before tick k0: the complementary
    filter's estimate (the logged pitch estimate at k0 - 1) and the encoder's two latest counts (the logged
    wheel angle x/r - theta, or the shaft of a synthetic full state, one and two encoder periods back; the
    window's own samples start at its t = 0). At k0 = 0 it is a plain reset, as simulate() does."""
    sensor.reset()
    if k0 == 0:
        return
    filt = getattr(sensor, "_filter", None)
    if filt is not None:
        filt.estimate = log.pitch_estimate(k0 - 1)
    enc = getattr(sensor, "_encoder", None)
    if enc is None:
        return
    per = int(round(enc.period / log.control_dt))
    if per < 1 or abs(per * log.control_dt - enc.period) > 1e-9:
        raise ValueError("the encoder period must be a whole number of control ticks")

    def count(k):
        if k < 0:
            return None
        if log.full_state is not None and log.full_state.shape[1] > 6:
            rel = float(log.full_state[k, 6])
        else:
            rel = float(log.state[k, 0]) / enc.wheel_radius - float(log.state[k, 2])
        return float(np.floor(rel / enc.resolution))

    # samples at window time -m * period; those with -m * period + delay <= 0 have arrived, the rest are in flight
    m_latest = max(1, int(math.ceil(enc.delay / enc.period - 1e-9)))
    enc._latest, enc._previous = count(k0 - m_latest * per), count(k0 - (m_latest + 1) * per)
    if enc._latest is None:
        enc._previous = None
    elif log.full_state is None:
        # A real log: the robot averages two wheels (half-count speed steps) and the model has one, so the
        # floor-of-position difference can be a count off the robot's own speed. Choose the previous count so
        # the first measured speed is the nearest whole count to the logged v / r - gyro (residual <= half a
        # count, ~0.03 m/s); a synthetic log (full_state) keeps the exact counts.
        speed = float(log.state[k0, 1]) / enc.wheel_radius - float(log.state[k0, 3])
        enc._previous = enc._latest - float(round(speed / enc.speed_per_count))
    for m in range(m_latest - 1, 0, -1):
        c = count(k0 - m * per)
        if c is not None:
            enc._in_flight.append((-m * enc.period + enc.delay, c))
    enc._n_samples = 0


def actuator_snapshots(log: RunLog, dynamics, dt: float, latency: float, starts, pwm_phase=None) -> dict:
    """{k0: ActuatorChain at global step k0 * spc, before that tick's command}, from the logged commands and
    duties replayed through the chain from the run's start."""
    _, _, sim = _bs()
    spc = sim.steps_per_control(log.control_dt, dt)
    chain = ActuatorChain(dynamics, dt, latency, pwm_phase)
    wanted, snaps = set(int(s) for s in starts), {}
    end = max(wanted) * spc
    for K in range(end + 1):
        if K % spc == 0 and K // spc in wanted:
            snaps[K // spc] = chain.copy()
        if K == end:
            break
        if K % spc == 0:
            chain.command(K, float(log.command[K // spc]), float(log.drive[K // spc]))
        chain.advance(K)
    return snaps


def start_state(dynamics, log: RunLog, k0: int, chain: ActuatorChain, backlash_init: str = "closed"):
    """The window's initial model state from the logged estimate, the hidden backlash states by default."""
    _, p2d, _ = _bs()
    if backlash_init == "log":
        if log.full_state is None:
            raise ValueError("backlash_init='log' needs a synthetic log with full_state")
        return np.array(log.full_state[k0], dtype=float)
    s = np.array(log.state[k0], dtype=float)
    if not getattr(dynamics, "has_backlash", False):
        return s
    full = dynamics.complete_state(s)                 # open, centred, motor side at the wheel's speed
    side = math.copysign(1.0, chain.drive) if chain.drive != 0.0 else 0.0
    if backlash_init == "closed" and side:
        full[p2d.GAP] = side * dynamics.p.backlash
        full[p2d.CONTACT] = side
    elif backlash_init not in ("closed", "open"):
        raise ValueError(f"backlash_init {backlash_init!r}")
    return full


@dataclass
class Ghost:
    """A re-synced (or free) comparison: per window, the sim at every control tick."""
    T: float
    starts: list            # first tick of each window
    windows: list           # dicts: k0, states (ticks + 1, 4), commands (ticks,), fell
    state: np.ndarray       # (n, 4) stitched sim state per log tick (window start = logged state)
    command: np.ndarray     # (n,) stitched sim command
    first_command_err: float  # max |sim - logged command| at window starts (ideal sensor: ~0)
    settings: dict


def resync(log: RunLog, cfg, bundle: dict, T: Optional[float] = 1.0, dt: Optional[float] = None,
           latency: Optional[float] = None, pwm_phase: Optional[float] = None, backlash_init: str = "closed",
           noise: bool = False, dynamics=None, sensor: str = "pipeline") -> Ghost:
    """Re-sync every T s (T=None: one free run over the whole log). sensor: "pipeline" or "ideal"."""
    _, _, sim = _bs()
    dt = cfg.timing.dt_s if dt is None else dt
    latency = cfg.timing.latency_s if latency is None else latency
    added = 1e-3 * float(log.meta.get("added_delay_ms") or 0.0)   # balance.py --added-delay-ms (load_run: issued duty)
    latency = latency + added
    dynamics = cfg.build_dynamics() if dynamics is None else dynamics
    cdt, n = log.control_dt, log.n
    spc = sim.steps_per_control(cdt, dt)
    W = n - 1 if T is None else max(1, int(round(T / cdt)))
    starts = list(range(0, n - 1, W))
    snaps = actuator_snapshots(log, dynamics, dt, latency, starts, pwm_phase)
    g_state = np.full((n, 4), np.nan)
    g_cmd = np.full(n, np.nan)
    windows, first_err = [], 0.0
    for j, k0 in enumerate(starts):
        ticks = min(W, n - 1 - k0)
        chain = snaps[k0].copy()
        ctrl = cfg.build_policy_controller(bundle["spec"], bundle["params"], goal=goal_from_log(log.goal, k0, cdt),
                                           fast=True)
        init_controller(ctrl, log, k0)
        s0 = start_state(dynamics, log, k0, chain, backlash_init)
        sens = build_sensor(cfg, sensor, noise, NOISE_SEED_BASE + j)
        if sens is not None:
            init_sensor(sens, log, k0)
        res = run_window(dynamics, ctrl, s0, ticks * spc, dt, cdt, chain, step0=k0 * spc, sensor=sens,
                         reset_sensor=False)
        sc = res.states[::spc, :4]
        cmds = res.commands
        windows.append({"k0": k0, "states": sc, "commands": cmds, "fell": res.fell})
        last = j == len(starts) - 1
        m = len(sc) if last else min(len(sc), ticks)
        g_state[k0:k0 + m] = sc[:m]
        g_cmd[k0:k0 + len(cmds)] = cmds
        if len(cmds):
            first_err = max(first_err, abs(float(cmds[0]) - float(log.command[k0])))
    return Ghost(T=W * cdt, starts=starts, windows=windows, state=g_state, command=g_cmd,
                 first_command_err=first_err,
                 settings={"dt": dt, "latency": latency, "added_delay_s": added,
                           "latency_steps": latency_steps(latency, dt),
                           "pwm_phase": getattr(dynamics.actuator, "pwm_tick_phase", None) if pwm_phase is None
                           else pwm_phase, "backlash_init": backlash_init, "noise": noise, "sensor": sensor,
                           "noise_seed_base": NOISE_SEED_BASE if noise else None, "free": T is None})


# ---------------------------------------------------------------------------- synthetic runs ---
def synthesize_log(cfg, bundle: dict, dynamics, goals: np.ndarray, state0, dt: float, latency: float,
                   label: str = "synthetic", sensor=None) -> RunLog:
    """A "fake real" run from the sim itself, in the RunLog layout: the TRUE state in `state` (the sim
    starts from it), the controller's pitch estimate in theta_meas (the filter starts from it), the hidden
    model states in full_state. For the round-trip test: resync must reproduce it."""
    _, _, sim = _bs()
    cdt = cfg.timing.control_dt_s
    spc = sim.steps_per_control(cdt, dt)
    n = len(goals)
    ctrl = cfg.build_policy_controller(bundle["spec"], bundle["params"], goal=goal_from_log(goals, 0, cdt),
                                       fast=False)
    ctrl.reset()
    rec = {k: [] for k in ("command", "drive", "int_v", "int_theta", "history", "frozen", "theta_meas")}

    def on_control(k, measured, command, drive):
        rec["theta_meas"].append(float(measured[2]))
        rec["command"].append(command)
        rec["drive"].append(drive)
        rec["int_v"].append(ctrl._int_v)
        rec["int_theta"].append(ctrl._int_theta)
        rec["history"].append([float(h) for h in ctrl.last_raw[7:7 + N_HISTORY]])
        rec["frozen"].append(ctrl.last_integrators_frozen)

    chain = ActuatorChain(dynamics, dt, latency)
    res = run_window(dynamics, ctrl, state0, (n - 1) * spc + 1, dt, cdt, chain, sensor=sensor,
                     on_control=on_control)
    if res.fell:
        raise ValueError("the synthetic run fell")
    full = res.states[::spc][:n]
    return RunLog(label=label, state=full[:, :4].copy(), command=np.array(rec["command"]),
                  drive=np.array(rec["drive"]), goal=np.asarray(goals, dtype=float),
                  int_v=np.array(rec["int_v"]), int_theta=np.array(rec["int_theta"]),
                  history=np.array(rec["history"]), frozen=np.array(rec["frozen"]), control_dt=cdt,
                  meta={"synthetic": True}, full_state=full.copy(),
                  theta_meas=np.array(rec["theta_meas"]))


# ---------------------------------------------------------------------------------- metrics ---
def horizon_errors(log: RunLog, ghost: Ghost) -> dict:
    """Per horizon h ticks after a re-sync: RMS over windows of sim - real for theta, v and x."""
    W = max(len(w["states"]) for w in ghost.windows) - 1
    out = {"h": [], "t": [], "theta": [], "v": [], "x": [], "n": []}
    for h in range(1, W + 1):
        e = [w["states"][h] - log.state[w["k0"] + h] for w in ghost.windows if len(w["states"]) > h]
        if not e:
            continue
        e = np.array(e)
        out["h"].append(h)
        out["t"].append(h * log.control_dt)
        for i, name in ((2, "theta"), (1, "v"), (0, "x")):
            out[name].append(float(np.sqrt(np.mean(e[:, i] ** 2))))
        out["n"].append(len(e))
    return {k: np.array(v) for k, v in out.items()}


def horizon_table(err: dict, T: float, step: float = 0.1) -> list:
    rows = []
    for hz in np.arange(step, T + 1e-9, step):
        i = int(np.argmin(np.abs(err["t"] - hz)))
        rows.append({"horizon_s": round(float(err["t"][i]), 4), "theta_rms_deg": math.degrees(err["theta"][i]),
                     "v_rms_mps": float(err["v"][i]), "x_rms_m": float(err["x"][i]), "windows": int(err["n"][i])})
    return rows


def highpass(y: np.ndarray, dt: float, window_s: float = WOBBLE_HIGHPASS_S) -> np.ndarray:
    """y minus its centred moving average (reflect-padded); NaNs are not allowed."""
    m = max(1, int(round(window_s / dt)))
    pad = np.pad(y, (m, m), mode="reflect")
    ma = np.convolve(pad, np.ones(m) / m, mode="same")[m:-m]
    return y - ma


def psd(y: np.ndarray, dt: float, seg_s: float = 4.0):
    """Welch PSD (Hann, 50 % overlap). Returns (f, P)."""
    m = min(len(y), int(round(seg_s / dt)))
    w = np.hanning(m)
    step = max(1, m // 2)
    segs = [y[i:i + m] for i in range(0, len(y) - m + 1, step)]
    P = np.mean([np.abs(np.fft.rfft((s - s.mean()) * w)) ** 2 for s in segs], axis=0) * dt / np.sum(w ** 2)
    return np.fft.rfftfreq(m, dt), P


def wobble_stats(y: np.ndarray, dt: float) -> dict:
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    y = y[: np.argmin(ok)] if not ok.all() else y
    hp = highpass(y, dt)
    chunk = int(round(1.0 / dt))
    p2p = [np.ptp(hp[i:i + chunk]) for i in range(0, len(hp) - chunk + 1, chunk)]
    f, P = psd(hp, dt)
    band = (f >= WOBBLE_BAND_HZ[0]) & (f <= WOBBLE_BAND_HZ[1])
    fpk = float(f[band][np.argmax(P[band])])
    return {"rms": float(np.sqrt(np.mean(hp ** 2))), "p2p_median": float(np.median(p2p)) if p2p else float("nan"),
            "peak_hz": fpk, "f": f, "P": P}


def summarize(log: RunLog, ghost: Ghost, free: Ghost) -> dict:
    err = horizon_errors(log, ghost)
    dt = log.control_dt
    wob = {}
    for name, series in (("real", log.state), ("ghost", ghost.state), ("free", free.state)):
        wob[name] = {"theta": wobble_stats(series[:, 2], dt), "v": wobble_stats(series[:, 1], dt)}
    # free-run divergence, RMS per 1 s bin
    chunk = int(round(1.0 / dt))
    fe = free.state - log.state
    bins = [(i * dt, float(np.sqrt(np.nanmean(fe[i:i + chunk, 2] ** 2))), float(np.sqrt(np.nanmean(fe[i:i + chunk, 1] ** 2))),
             float(np.sqrt(np.nanmean(fe[i:i + chunk, 0] ** 2)))) for i in range(0, log.n - chunk + 1, chunk)]
    lean = lambda s: float(np.nanmean(s[:, 2]))
    return {
        "label": log.label, "n_ticks": log.n, "T": ghost.T, "settings": ghost.settings,
        "first_command_err": ghost.first_command_err, "free_first_command_err": free.first_command_err,
        "ghost_windows_fell": int(sum(w["fell"] for w in ghost.windows)), "free_fell": bool(free.windows[0]["fell"]),
        "horizon": horizon_table(err, ghost.T), "horizon_curve": {k: v.tolist() for k, v in err.items()},
        "wobble": {name: {q: {k: v for k, v in s.items() if k not in ("f", "P")} for q, s in d.items()}
                   for name, d in wob.items()},
        "wobble_ratio_real_over_ghost": {q: wob["real"][q]["rms"] / wob["ghost"][q]["rms"] for q in ("theta", "v")},
        "wobble_ratio_real_over_free": {q: wob["real"][q]["rms"] / wob["free"][q]["rms"] for q in ("theta", "v")},
        "mean_theta_deg": {"real": math.degrees(lean(log.state)), "ghost": math.degrees(lean(ghost.state)),
                           "free": math.degrees(lean(free.state))},
        "free_divergence_per_s": [{"t0": b[0], "theta_rms_deg": math.degrees(b[1]), "v_rms_mps": b[2], "x_rms_m": b[3]}
                                  for b in bins],
        "_psd": {name: {q: (s["f"], s["P"]) for q, s in d.items()} for name, d in wob.items()},
        "_err": err,
    }


# ---------------------------------------------------------------------------------- plotting ---
def _segments(ghost: Ghost, log: RunLog, col: int):
    """Ghost time series with a NaN break at every re-sync (so jumps are visible), and the re-sync points."""
    t, y = [], []
    for w in ghost.windows:
        tt = (w["k0"] + np.arange(len(w["states"]))) * log.control_dt
        t += list(tt) + [np.nan]
        y += list(w["states"][:, col]) + [np.nan]
    return np.array(t), np.array(y)


def _cmd_segments(ghost: Ghost, log: RunLog):
    t, y = [], []
    for w in ghost.windows:
        tt = (w["k0"] + np.arange(len(w["commands"]))) * log.control_dt
        t += list(tt) + [np.nan]
        y += list(w["commands"]) + [np.nan]
    return np.array(t), np.array(y)


CAVEAT = ("real state = the robot's own estimate (encoders + complementary filter); sim = nominal model, true state. "
          "PDM phase, backlash gap and sensor history are unobservable (defaults); the wobble phase desyncs fast "
          "because PDM pulses and backlash impacts are chaotic: compare amplitude/frequency, not waveform.")


def caption_lines(width: int) -> list:
    """CAVEAT wrapped to at most width characters per line (no matplotlib auto-wrap: it clips at the edges)."""
    import textwrap
    return textwrap.wrap(CAVEAT, width)


def _caption(fig, width: int, fontsize: float) -> None:
    fig.text(0.5, 0.008, "\n".join(caption_lines(width)), ha="center", va="bottom", fontsize=fontsize,
             color="0.35", linespacing=1.25)


def summary_figure(log: RunLog, ghost: Ghost, free: Ghost, s: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    err, dt = s["_err"], log.control_dt
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    T = ghost.T
    ax = axs[0, 0]
    ax.plot(err["t"], np.degrees(err["theta"]), color=SIM_COLOR)
    tab = s["horizon"]
    ax.plot([r["horizon_s"] for r in tab], [r["theta_rms_deg"] for r in tab], "o", color=SIM_COLOR)
    ax.axhline(s["wobble"]["real"]["theta"]["rms"] * 180 / math.pi, color=REAL_COLOR, ls=":",
               label="real wobble RMS (1 s high-pass)")
    ax.set(xlabel="time since re-sync (s)", ylabel="RMS pitch error (deg)", title=f"Pitch error vs horizon (T = {T:g} s)")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.plot(err["t"], err["v"], color=SIM_COLOR)
    ax.plot([r["horizon_s"] for r in tab], [r["v_rms_mps"] for r in tab], "o", color=SIM_COLOR)
    ax.axhline(s["wobble"]["real"]["v"]["rms"], color=REAL_COLOR, ls=":", label="real v wobble RMS")
    ax.set(xlabel="time since re-sync (s)", ylabel="RMS velocity error (m/s)", title="Velocity error vs horizon")
    ax.legend(fontsize=8)
    ax = axs[0, 2]
    fd = s["free_divergence_per_s"]
    t0 = [b["t0"] + 0.5 for b in fd]
    ax.plot(t0, [b["theta_rms_deg"] for b in fd], "o-", color=FREE_COLOR, label="pitch RMS (deg)")
    ax.plot(t0, [100 * b["v_rms_mps"] for b in fd], "s-", color="tab:purple", label="v RMS (cm/s)")
    ax.plot(t0, [100 * b["x_rms_m"] for b in fd], "^-", color="tab:gray", label="x RMS (cm)")
    ax.set(xlabel="run time (s)", title="Free run: error per 1 s bin")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    for name, color in (("real", REAL_COLOR), ("ghost", SIM_COLOR), ("free", FREE_COLOR)):
        f, P = s["_psd"][name]["theta"]
        ax.semilogy(f, np.degrees(np.sqrt(P)), color=color,
                    label=f"{name}: peak {s['wobble'][name]['theta']['peak_hz']:.2f} Hz")
    ax.set(xlim=(0, 15), xlabel="frequency (Hz)", ylabel="pitch ASD (deg/sqrt(Hz))", title="Pitch wobble spectrum")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    names = ["real", "ghost", "free"]
    x = np.arange(3)
    th = [math.degrees(s["wobble"][n]["theta"]["rms"]) for n in names]
    p2p = [math.degrees(s["wobble"][n]["theta"]["p2p_median"]) for n in names]
    ax.bar(x - 0.2, th, 0.4, color=[REAL_COLOR, SIM_COLOR, FREE_COLOR], label="RMS")
    ax.bar(x + 0.2, p2p, 0.4, color=[REAL_COLOR, SIM_COLOR, FREE_COLOR], alpha=0.45, label="median p-p per 1 s")
    ax.set_xticks(x, [f"{n}\nv RMS {100 * s['wobble'][n]['v']['rms']:.2f} cm/s" for n in names])
    rg = s["wobble_ratio_real_over_ghost"]["theta"]
    rf = s["wobble_ratio_real_over_free"]["theta"]
    ax.set(ylabel="pitch wobble (deg)", title=f"Wobble amplitude: real/ghost {rg:.2f}x, real/free {rf:.2f}x")
    ax.legend(fontsize=8)
    ax = axs[1, 2]
    t = log.t
    mid = min(log.n * dt, 8.0)
    sel = (t >= mid - 1.5) & (t <= mid + 1.5)
    ax.plot(t[sel], np.degrees(log.state[sel, 2]), color=REAL_COLOR, label="real (estimate)")
    tg, yg = _segments(ghost, log, 2)
    sg = (tg >= mid - 1.5) & (tg <= mid + 1.5) | np.isnan(tg)
    ax.plot(tg[sg], np.degrees(yg[sg]), color=SIM_COLOR, label=f"ghost (re-sync {T:g} s)")
    ax.plot(t[sel], np.degrees(free.state[sel, 2]), color=FREE_COLOR, alpha=0.7, label="free run")
    for k0 in ghost.starts:
        if mid - 1.5 <= k0 * dt <= mid + 1.5:
            ax.axvline(k0 * dt, color="0.6", ls=":", lw=1)
    ax.set(xlim=(mid - 1.5, mid + 1.5), xlabel="t (s)", ylabel="pitch (deg)", title="Pitch, 3 s zoom")
    ax.legend(fontsize=8)
    fig.suptitle(f"Sim vs real: {log.label}  (dt {ghost.settings['dt'] * 1e3:g} ms, latency "
                 f"{ghost.settings['latency_steps'] * ghost.settings['dt'] * 1e3:g} ms applied, backlash init "
                 f"{ghost.settings['backlash_init']}{', sensor ' + ghost.settings['sensor'] + (' + noise' if ghost.settings['noise'] else ' (no noise)')})",
                 fontsize=12)
    _caption(fig, width=190, fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def animate(log: RunLog, ghost: Ghost, cfg, path: Path, fps: int = 15, dpi: int = 72, window_s: float = 4.0,
            speed: float = 1.0, mode_label: str = "re-sync") -> Path:
    """Top: real robot (solid) and sim ghost (translucent) at their x and pitch, view centred on the real robot.
    Bottom: pitch, velocity and torque, real vs sim, re-sync markers, moving cursor (scrolling window_s)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation as mpl_animation
    from matplotlib.patches import Circle, Polygon

    robot = cfg.build_robot_params()
    r, l = robot.wheel_radius, robot.body_com_height
    bh, bw = 2 * l, 0.35 * 2 * l
    half = 0.25
    dt, n = log.control_dt, log.n
    t_end = (n - 1) * dt
    fig = plt.figure(figsize=(8, 8.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1, 1, 1], hspace=0.35, left=0.1, right=0.97, top=0.95,
                          bottom=0.115)
    top = fig.add_subplot(gs[0])
    top.set_aspect("equal")
    top.set_xlim(-half, half)
    top.set_ylim(-0.02, r + bh + 0.045)
    top.set_xticks([])
    top.set_yticks([])
    top.axhline(0, color="k", lw=1)
    ground = top.plot([], [], "k|", ms=8)[0]

    def parts(color, alpha, z):
        return {"body": top.add_patch(Polygon(np.zeros((4, 2)), closed=True, facecolor=color, edgecolor=color,
                                              alpha=alpha, zorder=z)),
                "wheel": top.add_patch(Circle((0, r), r, facecolor="0.3", edgecolor="k", alpha=alpha, zorder=z + 1)),
                "spoke": top.plot([], [], color="w", lw=2, alpha=alpha, zorder=z + 2)[0],
                "com": top.plot([], [], "o", color="k", ms=4, alpha=alpha, zorder=z + 2)[0]}

    real_p = parts(REAL_COLOR, 0.9, 2)
    sim_p = parts(SIM_COLOR, 0.35, 5)
    text = top.text(0.01, 0.98, "", transform=top.transAxes, va="top", fontsize=8.5, family="monospace")
    flash = top.text(0.99, 0.98, "", transform=top.transAxes, va="top", ha="right", fontsize=10,
                     fontweight="bold", color=SIM_COLOR)
    top.set_title(f"{log.label}: real (solid) vs sim {mode_label} ghost (translucent)", fontsize=10)

    def place(p, xoff, theta, x_abs):
        axle = np.array([xoff, r])
        up = np.array([math.sin(theta), math.cos(theta)])
        side = np.array([math.cos(theta), -math.sin(theta)]) * bw / 2
        tip = axle + up * bh
        p["body"].set_xy([axle - side, axle + side, tip + side, tip - side])
        p["wheel"].center = (xoff, r)
        spin = x_abs / r
        p["spoke"].set_data([xoff, xoff + r * math.sin(spin)], [r, r + r * math.cos(spin)])
        com = axle + up * l
        p["com"].set_data([com[0]], [com[1]])

    t = log.t
    axes, cursors = [], []
    specs = [("pitch (deg)", 2, math.degrees(1)), ("v (m/s)", 1, 1.0), ("torque (N m)", None, 1.0)]
    for i, (ylabel, col, scale) in enumerate(specs):
        ax = fig.add_subplot(gs[i + 1])
        if col is not None:
            ax.plot(t, log.state[:, col] * scale, color=REAL_COLOR, lw=1.1, label="real (estimate)")
            tg, yg = _segments(ghost, log, col)
            ax.plot(tg, yg * scale, color=SIM_COLOR, lw=1.1, label="sim")
            ax.plot([w["k0"] * dt for w in ghost.windows], [w["states"][0, col] * scale for w in ghost.windows],
                    "o", color=SIM_COLOR, ms=3)
        else:
            ax.plot(t, log.command, color=REAL_COLOR, lw=1.0, label="real command")
            tg, yg = _cmd_segments(ghost, log)
            ax.plot(tg, yg, color=SIM_COLOR, lw=1.0, label="sim command")
        if col == 1:
            ax.plot(t, log.goal, color="0.4", ls="--", lw=1, label="goal")
        for k0 in ghost.starts[1:]:
            ax.axvline(k0 * dt, color="0.75", ls=":", lw=0.8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        cursors.append(ax.axvline(0, color="0.3", lw=1))
        axes.append(ax)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
        if col == 1:
            ax.legend(fontsize=7, loc="upper right", ncol=3)
    axes[-1].set_xlabel("t (s)   dotted: re-sync", fontsize=9)
    for ax, (_, col, scale) in zip(axes, specs):   # fixed y limits from the real data (robust to a fallen ghost)
        y = log.state[:, col] * scale if col is not None else log.command
        lo, hi = np.percentile(y, [0.2, 99.8])
        pad = 0.25 * (hi - lo) + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
    _caption(fig, width=125, fontsize=7)

    n_frames = int(math.floor(t_end / speed * fps)) + 1
    starts_t = np.array(ghost.starts) * dt

    def draw(frame):
        time = min(frame * speed / fps, t_end)
        i = min(int(round(time / dt)), n - 1)
        xr, vr, thr = log.state[i, 0], log.state[i, 1], log.state[i, 2]
        gs_ = ghost.state[i]
        place(real_p, 0.0, thr, xr)
        if np.all(np.isfinite(gs_)):
            place(sim_p, gs_[0] - xr, gs_[2], gs_[0])
        first = math.floor((xr - half) / 0.05)
        marks = np.arange(first, first + 2 * half / 0.05 + 2) * 0.05 - xr
        ground.set_data(marks, np.zeros_like(marks))
        k_since = time - starts_t[starts_t <= time + 1e-9].max()
        text.set_text(f"t {time:5.2f} s   goal {log.goal[i]:+.2f} m/s\n"
                      f"real  pitch {math.degrees(thr):+5.2f} deg  v {vr:+.3f} m/s\n"
                      f"sim   pitch {math.degrees(gs_[2]):+5.2f} deg  v {gs_[1]:+.3f} m/s\n"
                      f"sim - real  dx {100 * (gs_[0] - xr):+5.1f} cm   {k_since:4.2f} s since sync")
        flash.set_text("RE-SYNC" if (ghost.settings["free"] is False and k_since < 0.15 and time > 0.01) else "")
        for ax, c in zip(axes, cursors):
            c.set_xdata([time, time])
            if window_s and window_s < t_end:
                lo = min(max(0.0, time - window_s / 2), t_end - window_s)
                ax.set_xlim(lo, lo + window_s)
        return []

    if not window_s or window_s >= t_end:
        for ax in axes:
            ax.set_xlim(0, t_end)
    anim = mpl_animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps, blit=False)
    path = Path(path)
    if path.suffix.lower() == ".mp4":
        writer = mpl_animation.FFMpegWriter(fps=fps)
    else:
        writer = mpl_animation.PillowWriter(fps=fps)
    anim.save(str(path), writer=writer, dpi=dpi)
    plt.close(fig)
    return path


def _strip(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a run stamp (searched in --log-dir), or a *_run.csv / *_raw.csv path")
    ap.add_argument("--mode", choices=("resync", "free"), default="resync", help="which one to animate")
    ap.add_argument("--resync", type=float, default=1.0, help="re-sync period T (s)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    ap.add_argument("--bundle", default=None, help="policy bundle (default: the one the log names)")
    ap.add_argument("--dt", type=float, default=None, help="physics step (default cfg, 2.5 ms)")
    ap.add_argument("--latency", type=float, default=None, help="sample-to-actuator latency (default cfg, 10 ms)")
    ap.add_argument("--pwm-phase", type=float, default=None, help="PDM tick phase in [0,1) (default cfg 0.25)")
    ap.add_argument("--backlash-init", choices=("closed", "open"), default="closed")
    ap.add_argument("--sensor", choices=("pipeline", "ideal"), default="pipeline",
                    help="pipeline: the config's IMU filter + encoder without noise (default); ideal: true state")
    ap.add_argument("--noise", action="store_true", help="add the config's sensor noise and biases (seed 7.3M + window)")
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--dpi", type=int, default=72)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--window", type=float, default=4.0, help="scrolling plot window (s); 0 = whole run")
    ap.add_argument("--format", choices=("auto", "gif", "mp4"), default="auto")
    args = ap.parse_args(argv)

    cfg = control.load_robot_config()
    log = load_run(args.run, args.log_dir, cfg.timing.control_dt_s)
    bundle = load_bundle(log, cfg, args.bundle)
    kw = dict(dt=args.dt, latency=args.latency, pwm_phase=args.pwm_phase, backlash_init=args.backlash_init,
              noise=args.noise, sensor=args.sensor)
    t0 = _wall.perf_counter()
    ghost = resync(log, cfg, bundle, args.resync, **kw)
    free = resync(log, cfg, bundle, None, **kw)
    t_sim = _wall.perf_counter() - t0
    s = summarize(log, ghost, free)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = log.label[:15]
    tag = f"T{args.resync:g}" + ("_noise" if args.noise else "") + ("_ideal" if args.sensor == "ideal" else "") + ("" if args.backlash_init == "closed" else "_open")
    s["files"] = {"summary": str(summary_figure(log, ghost, free, s, out / f"{stamp}_summary_{tag}.png"))}
    if not args.no_anim:
        import shutil
        fmt = args.format if args.format != "auto" else ("mp4" if shutil.which("ffmpeg") else "gif")
        g = ghost if args.mode == "resync" else free
        name = f"{stamp}_{args.mode}" + (f"_{tag}" if args.mode == "resync" else "") + f".{fmt}"
        t1 = _wall.perf_counter()
        s["files"]["animation"] = str(animate(log, g, cfg, out / name, fps=args.fps, dpi=args.dpi,
                                              window_s=args.window, speed=args.speed,
                                              mode_label="re-sync" if args.mode == "resync" else "free-run"))
        s["render_s"] = _wall.perf_counter() - t1
    s["sim_s"] = t_sim
    s["bundle"] = {"name": bundle.get("name"), "sha256": bundle.get("sha256")}
    path = out / f"{stamp}_metrics_{tag}.json"
    path.write_text(json.dumps(_strip(s), indent=1))
    print(f"{log.label}: T={ghost.T:g} s, {len(ghost.windows)} windows, sim {t_sim:.1f} s, "
          f"first-command check {ghost.first_command_err:.2e}")
    print("horizon(s)  theta_rms(deg)  v_rms(m/s)  x_rms(m)")
    for row in s["horizon"]:
        print(f"  {row['horizon_s']:5.2f}   {row['theta_rms_deg']:8.3f}     {row['v_rms_mps']:.4f}    {row['x_rms_m']:.4f}")
    for q in ("theta", "v"):
        w = s["wobble"]
        print(f"wobble {q}: rms real {w['real'][q]['rms']:.4g} ghost {w['ghost'][q]['rms']:.4g} free "
              f"{w['free'][q]['rms']:.4g}; peak Hz real {w['real'][q]['peak_hz']:.2f} ghost "
              f"{w['ghost'][q]['peak_hz']:.2f} free {w['free'][q]['peak_hz']:.2f}")
    print(f"mean pitch (deg): {s['mean_theta_deg']}")
    print(f"wrote {path}" + "".join(f"\n      {v}" for v in s["files"].values()))
    return s


if __name__ == "__main__":
    main()
