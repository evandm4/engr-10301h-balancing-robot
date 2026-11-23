"""
Derivation of the LEGO robot's scenario and cost limits (SOFTWARE_PLAN A3).

Repeats the method of config.py's ScenarioConfig and CostConfig docstrings, but
as code, so every number in legobot_config.py's scenario can be re-derived when
the datasheet values change (after the bench measurements, C4). The same
functions run on the Pololu robot as a check: they reproduce the numbers
config.py's docstrings quote.

Usage (from the repo root):
    python hardware/legobot/legobot_scenario.py              # LEGO robot, full derivation
    python hardware/legobot/legobot_scenario.py --check      # also re-derive the Pololu numbers

The "robust-scoring box" is every corner of RobustnessSettings: each randomize
spread at either end and each randomize_range at either end (2^11 = 2048
robots for both configs). Worst cases are taken over all corners.

A "full-torque response" (the catch test): the robot cruises upright at speed v,
a push of impulse J acts on the body's centre of mass for push_duration; the
motors give no torque until the latency has passed (friction keeps acting, so
this is slightly harsher than holding the cruise torque; it is the convention
that reproduces config.py's 0.149 N s), then the command goes to full torque
toward recovery (the actuator clips it to the motor's envelope and max_torque)
and is held. The robot is caught if the pitch comes back through zero before it
passes the fall angle. Start tilts: the same from rest, with no push. No
controller does better than this from the same state, so a push or tilt that
passes on every corner can always be saved: a fall is the controller's doing.
"""

import argparse
import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import POLOLU_CONFIG, ProjectConfig  # noqa: E402  (also puts simulation/ on the path)
from batch_sim.physics import actuator_torque, rk4_step, voltage_drive  # noqa: E402
from batch_sim.robot import BatchRobot  # noqa: E402

BLEND_PEAK = 2.1875   # step_profile: peak acceleration = 2.1875 x change / blend time


# ------------------------------------------------------------- robots ---
def nominal_robot(cfg: ProjectConfig, driver: str = None) -> BatchRobot:
    """driver: "voltage" or "current" (default: the config's actuator model). In voltage mode
    the robot code's constants stay nominal while corner_robots varies the true ones."""
    motor = cfg.actuator.build_motor_params()
    voltage = (cfg.actuator.model in ("voltage", "slow_pwm")) if driver is None else driver in ("voltage", "slow_pwm")
    slow = (cfg.actuator.model == "slow_pwm") if driver is None else driver == "slow_pwm"
    return BatchRobot.from_params(cfg.build_robot_params(), motor, voltage_estimate=motor if voltage else None,
                                  voltage_feedforward=cfg.actuator.voltage_feedforward,
                                  voltage_feedforward_gain=cfg.actuator.voltage_feedforward_gain,
                                  slow_pwm=slow and voltage)


def drive(R: BatchRobot, command, x_dot, theta_dot):
    """What the actuator gets for a torque command: the command itself, or in voltage mode the
    robot code's drive voltage from the wheel speed (taken as measured exactly, every step)."""
    return voltage_drive(R, command, x_dot, theta_dot) if R.voltage_mode else command


def corner_robots(cfg: ProjectConfig, driver: str = None) -> BatchRobot:
    """Every corner of the robust-scoring box, as one batch."""
    base = nominal_robot(cfg, driver)
    r = cfg.robustness
    n_rel, n_rng = len(r.randomize), len(r.randomize_range)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n_rel + n_rng)))
    changes = {name: getattr(base, name) * (1 + s * signs[:, j]) for j, (name, s) in enumerate(r.randomize)}
    changes.update({name: np.where(signs[:, n_rel + j] > 0, hi, lo)
                    for j, (name, lo, hi) in enumerate(r.randomize_range) if name not in NOT_IN_CORNERS})
    return replace(base, **changes)


# Robustness ranges the catch tests here don't see: the pulse-density tick phase only matters to a closed
# loop at the control rate, and these tests apply full torque. Held at the base value (both corner signs
# then give the same robot, so the batch has duplicates rather than fewer corners: harmless).
NOT_IN_CORNERS = ("pwm_tick_phase",)


def size(robot: BatchRobot) -> int:
    return int(np.size(robot.body_mass))


def coeffs(R: BatchRobot, theta: float = 0.0):
    """a, b(theta), d of the mass matrix (PHYSICS.md 2.5)."""
    ml = R.body_mass * R.body_com_height
    a = R.body_mass + R.wheel_mass + (R.wheel_inertia + R.rotor_inertia) / R.wheel_radius**2
    b = ml * math.cos(theta) - R.rotor_inertia / R.wheel_radius
    d = R.body_mass * R.body_com_height**2 + R.body_inertia + R.rotor_inertia
    return a, b, d


def friction(R: BatchRobot, wheel_speed):
    return R.wheel_friction * wheel_speed + R.coulomb_friction * np.tanh(wheel_speed / R.coulomb_friction_speed)


# ------------------------------------------------- lean and acceleration ---
def lean_acceleration(R: BatchRobot, lean: float):
    """Steady lean (PHYSICS.md 2.8): acceleration and the net wheel torque it takes."""
    a, b, _ = coeffs(R, lean)
    ml = R.body_mass * R.body_com_height
    xdd = ml * R.gravity * math.sin(lean) / (b + a * R.wheel_radius)
    return xdd, a * R.wheel_radius * xdd


def lean_gain(R: BatchRobot):
    """G = Mgl / (b0 + a r), m/s^2 per rad of lean (linearized)."""
    a, b, _ = coeffs(R)
    return R.body_mass * R.body_com_height * R.gravity / (b + a * R.wheel_radius)


def unstable_pole(R: BatchRobot):
    a, b, d = coeffs(R)
    return np.sqrt(a * R.body_mass * R.body_com_height * R.gravity / (a * d - b * b))


def net_forward_torque(R: BatchRobot, v):
    """Forward torque left after friction, upright at speed v (m/s)."""
    w = v / R.wheel_radius
    command = np.full(size(R), 1e6) if size(R) > 1 else 1e6
    full = actuator_torque(R, drive(R, command, v, 0.0 * w), w)
    return full - friction(R, w)


def top_speed(R: BatchRobot, torque_needed, v_max: float = 3.0, steps: int = 40):
    """Per robot: the highest speed at which net forward torque >= torque_needed (bisection;
    available torque falls monotonically with speed). -1 where even standstill is short."""
    n = size(R)
    need = np.broadcast_to(torque_needed, (n,)) if n > 1 else torque_needed
    lo, hi = np.zeros(n), np.full(n, v_max)
    feasible = net_forward_torque(R, lo) >= need
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        ok = net_forward_torque(R, mid) >= need
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    return np.where(feasible, lo, -1.0)


def no_load_speed(R: BatchRobot):
    return R.supply_voltage / (R.torque_constant * R.gear_ratio) * R.wheel_radius


# ------------------------------------------------------------ catch test ---
def caught(R: BatchRobot, *, speed, impulse=0.0, start_pitch=0.0, push_duration=0.05, latency=0.020,
           fall_angle=math.radians(60), dt=0.0005, horizon=3.0, hold_cruise_torque=False):
    """Full-torque response (module docstring). impulse > 0 is a push from behind
    (forward). Returns a bool per robot (arrays broadcast: impulse may be per robot)."""
    n = size(R)
    impulse = np.broadcast_to(np.asarray(impulse, float), (n,)).copy()
    force = impulse / push_duration
    direction = np.where(impulse != 0, np.sign(impulse), np.sign(start_pitch) if start_pitch else 1.0)
    state = (np.zeros(n), np.full(n, float(speed)), np.full(n, float(start_pitch)), np.zeros(n))
    hold = (friction(R, speed / R.wheel_radius) if hold_cruise_torque else 0.0) * np.ones(n)
    alive, done = np.ones(n, bool), np.zeros(n, bool)
    t = 0.0
    while t < horizon and not done.all():
        torque = hold if t < latency else direction * 1e6
        f = force if t < push_duration else 0.0
        state = _step_with_force(R, state, drive(R, torque, state[1], state[3]), dt, f)
        t += dt
        theta = state[2]
        fell = np.abs(theta) > fall_angle
        back = (t > max(latency, push_duration)) & (direction * theta <= 0)
        alive &= ~(fell & ~done)
        done |= fell | back
    return alive & done


def _step_with_force(R, state, torque, dt, force):
    """rk4_step takes one force for the whole batch; here it can differ per robot."""
    if np.ndim(force) == 0:
        return rk4_step(R, state, torque, dt, force=float(force))
    # The per-robot push added the way physics.derivatives adds a shared one.
    def f(s):
        x_dot, theta, theta_dot = s[1], s[2], s[3]
        w = x_dot / R.wheel_radius - theta_dot
        tau = actuator_torque(R, torque, w) - friction(R, w)
        a, _, d = coeffs(R)
        ml = R.body_mass * R.body_com_height
        b = ml * np.cos(theta) - R.rotor_inertia / R.wheel_radius   # per-robot theta, so not coeffs()
        rhs_x = tau / R.wheel_radius + ml * np.sin(theta) * theta_dot**2 + force
        rhs_t = -tau + ml * R.gravity * np.sin(theta) + force * R.body_com_height * np.cos(theta)
        det = a * d - b * b
        return x_dot, (d * rhs_x - b * rhs_t) / det, theta_dot, (a * rhs_t - b * rhs_x) / det

    k1 = f(state)
    k2 = f(tuple(s + 0.5 * dt * k for s, k in zip(state, k1)))
    k3 = f(tuple(s + 0.5 * dt * k for s, k in zip(state, k2)))
    k4 = f(tuple(s + dt * k for s, k in zip(state, k3)))
    return tuple(s + dt / 6 * (p + 2 * q + 2 * r + u) for s, p, q, r, u in zip(state, k1, k2, k3, k4))


def largest_catchable_impulse(R, speed, latency, push_duration=0.05, j_max=2.0, steps=18, sign=1.0):
    """Per robot, the largest push (N s) a full-torque response still catches."""
    n = size(R)
    lo, hi = np.zeros(n), np.full(n, j_max)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        ok = caught(R, speed=speed, impulse=sign * mid, push_duration=push_duration, latency=latency)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    return lo


def largest_catchable_tilt(R, latency, steps=16, max_tilt=1.2):
    n = size(R)
    lo, hi = np.zeros(n), np.full(n, max_tilt)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        ok = _tilt_batch(R, mid, latency)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    return lo


def _tilt_batch(R, tilts, latency, dt=0.0005, horizon=3.0, fall_angle=math.radians(60)):
    n = size(R)
    state = (np.zeros(n), np.zeros(n), tilts.copy(), np.zeros(n))
    alive, done = np.ones(n, bool), np.zeros(n, bool)
    t = 0.0
    while t < horizon and not done.all():
        torque = np.zeros(n) if t < latency else np.full(n, 1e6)
        state = rk4_step(R, state, drive(R, torque, state[1], state[3]), dt)
        t += dt
        fell = np.abs(state[2]) > fall_angle
        back = (t > latency) & (state[2] <= 0)
        alive &= ~(fell & ~done)
        done |= fell | back
    return alive & done


def describe_worst(cfg, R, index):
    """Which end of each spread the worst corner sits at."""
    base = nominal_robot(cfg)
    out = []
    for name, _ in cfg.robustness.randomize:
        out.append(f"{name} {'+' if np.ravel(getattr(R, name))[index] > getattr(base, name) else '-'}")
    for name, lo, hi in cfg.robustness.randomize_range:
        if name in NOT_IN_CORNERS:
            continue
        out.append(f"{name} {'hi' if np.ravel(getattr(R, name))[index] > 0.5 * (lo + hi) else 'lo'}")
    return ", ".join(out)


# ----------------------------------------------------------- reports ---
def check_pololu():
    """Re-derive the numbers config.py's docstrings quote for the Pololu robot."""
    cfg = POLOLU_CONFIG
    N, R = nominal_robot(cfg), corner_robots(cfg)
    lean = cfg.controller.max_lean_rad
    xdd, tau = lean_acceleration(N, lean)
    print("Pololu check (quoted values in brackets)")
    print(f"  G = {float(lean_gain(N)):.2f} m/s^2/rad [7.15]; lean-cap accel {float(xdd):.2f} [3.24], torque {float(tau):.3f} [0.135]")
    print(f"  unstable pole {float(unstable_pole(N)):.2f} rad/s [8.44]")
    vt = top_speed(R, float(tau))
    print(f"  top speed with the nominal lean-cap torque on every corner motor: {vt.min():.3f} m/s [1.3 chosen; weakest corner 1.16]")
    j = largest_catchable_impulse(R, 1.3, 0.020)
    k = int(np.argmin(j))
    print(f"  largest catchable forward push at 1.3 m/s, 20 ms: {j.min():.3f} N s [0.149] ({describe_worst(cfg, R, k)})")
    print(f"     nominal, 10 ms: {float(largest_catchable_impulse(N, 1.3, 0.010)[0]):.3f} N s [0.85]")
    tilt = largest_catchable_tilt(R, 0.020)
    print(f"  largest catchable start tilt, 20 ms: {tilt.min():.3f} rad [0.39]; nominal 10 ms "
          f"{float(largest_catchable_tilt(N, 0.010)[0]):.3f} [0.79]")


# The Pololu scenario's proportions, which the LEGO scenario keeps (config.py docstrings).
POLOLU_LEAN_SHARES = {            # lean each goal change needs, as a share of pitch_limit_rad (0.30)
    "aggressive": 0.25 / 0.30,    # 0 -> top speed: 0.25 rad
    "brake": 1.31 / 7.15 / 0.30,  # top -> reverse: 1.31 m/s^2 at G = 7.15, 0.183 rad
    "gentle": 0.06 / 0.30,        # reverse -> 0: 0.06 rad
}
POLOLU_PITCH_LIMIT_OF_TILT = 0.30 / 0.391     # pitch_limit = 77% of the catchable start tilt
POLOLU_START_TILT_SHARES = (0.05 / 0.30, -0.15 / 0.30, 1.0)   # of pitch_limit
POLOLU_REVERSE_SHARE = 0.5 / 1.3              # reverse cruise speed / top speed
POLOLU_VELOCITY_LIMIT_SHARE = 0.2 / 1.3       # velocity_limit / top speed
POLOLU_TORQUE_LIMIT_OF_CAP = 0.1 / 0.135      # torque_limit / nominal lean-cap torque
POLOLU_LEAN_CAP_DEG = 25.0                    # config.py's lean cap
POLOLU_PITCH_LIMIT_OF_CAP = 0.30 / math.radians(25.0)   # pitch_limit = 69% of the lean cap

# Decided 2026-11-23 (the user), after the derived 0.47 m/s made every baseline fall: to accelerate, a
# balancing robot drives its wheels well past the goal speed first (+66-79% for the model-based PID), and
# this voltage-limited motor loses torque linearly with speed, so the top-speed rule above (lean-cap torque
# AT the cruise speed) leaves no headroom. The cruise is set to 0.40 m/s (Pololu's 53% of no-load), and the
# speed changes are softened: each asks for LEAN_SOFTENING x Pololu's lean share (longer blends), the least
# softening at which the model-based PID, an untuned reference design, has no falls on the robust training
# set (0.65: 19 of 150 fall, 0.6: 13, 0.55: 2, 0.5: none; A4_BASELINES.md).
TOP_SPEED_CHOICE = 0.40
LEAN_SOFTENING = 0.5
POLOLU_PUSH_SHARE = 0.145 / 0.149             # push / largest catchable on every corner
# What a full run of the catch simulations gives for legobot_config.py as of 2026-11-22
# (DATASHEET 2 values with the measured body, voltage-mode driver), for --quick. Re-run without --quick after changing the config.
RECORDED_CATCH = dict(forward=0.0325, backward=0.1352, tilt=0.233, nominal_forward=0.105, nominal_tilt=0.450)


def lean_cap_table(N, R, caps_deg):
    """Per lean cap: nominal accel and torque, corner torque range, top speed, v x a on the limiting corner."""
    rows = []
    for cap in caps_deg:
        L = math.radians(cap)
        xn, tn = lean_acceleration(N, L)
        xc, tc = lean_acceleration(R, L)
        vt = top_speed(R, tc)
        k = int(np.argmin(vt))
        rows.append((float(cap), float(xn), float(tn), tc.min(), tc.max(), vt.min(), vt.min() * xc[k]))
    return rows


def derive_legobot(full=True):
    """The LEGO scenario and cost limits, each from a number of the robot. Prints the
    derivation and returns the values legobot_config.py uses."""
    from legobot_config import LEGOBOT_CONFIG as cfg

    N, R = nominal_robot(cfg), corner_robots(cfg)
    lat = cfg.robustness.latency_range_s[1]
    p = float(unstable_pole(N))
    print("LEGO robot")
    print(f"  G = {float(lean_gain(N)):.2f} m/s^2/rad nominal ({np.min(lean_gain(R)):.2f}-{np.max(lean_gain(R)):.2f} over corners); "
          f"unstable pole {p:.2f} rad/s = {1000 / p:.0f} ms (fastest corner {1000 / np.max(unstable_pole(R)):.0f} ms)")
    print(f"  no-load speed nominal {float(no_load_speed(N)):.3f} m/s, corners {np.min(no_load_speed(R)):.3f}-{np.max(no_load_speed(R)):.3f}")
    print(f"  net forward torque at standstill: nominal {float(net_forward_torque(N, 0.0)):.3f} N m, weakest corner "
          f"{np.min(net_forward_torque(R, np.zeros(size(R)))):.3f}")

    # 1. Lean cap and top speed together. config.py fixed the cap (25 deg) and took the top speed
    # as the highest at which every motor still has the lean-cap torque; here that is 0.017 m/s at
    # 25 deg. So both are chosen: the cap that maximizes top speed x lean-cap acceleration on the
    # limiting corner, i.e. the scenario asks for the weakest robot's peak mechanical power.
    print("\n  1. Lean cap vs top speed (each corner needs its own lean-cap torque, from its own motor, at the top speed):")
    print("     cap deg | accel nom | torque nom | torque corners | top speed (min corner) | v x a (limiting corner)")
    for cap, xn, tn, tlo, thi, v, va in lean_cap_table(N, R, (25, 20, 17.5, 15, 14, 13, 12.5, 12, 11, 10, 7.5)):
        print(f"     {cap:7.1f} | {xn:5.2f}     | {tn:6.3f}     | {tlo:.3f}-{thi:.3f}    | {v:6.3f}                 | {va:.3f}")
    # The search runs up to config.py's own 25 deg: when the weakest robot's v x a still rises at 25 deg,
    # the LEGO robot can afford the Pololu cap, and the top speed then comes from config.py's own rule
    # (the highest speed at which every corner still has the lean-cap torque). Before the motor was
    # measured the optimum was 12-15 deg; the search stopped at 15 deg, which hid that the measured motor
    # no longer needs a smaller cap.
    fine = lean_cap_table(N, R, np.arange(10.0, POLOLU_LEAN_CAP_DEG + 0.01, 0.25))
    best = max(fine, key=lambda row: row[-1])
    cap_deg = min(round(best[0] * 2) / 2, POLOLU_LEAN_CAP_DEG)
    cap = math.radians(cap_deg)
    v_limit = [row for row in fine if abs(row[0] - cap_deg) < 1e-9][0][5]
    derived_top = math.floor(v_limit * 100) / 100
    top = min(derived_top, TOP_SPEED_CHOICE)
    print(f"     top speed: derived {derived_top} m/s; chosen {top} m/s (TOP_SPEED_CHOICE: headroom for the speed "
          f"overshoot of the acceleration step)")
    _, tau_cap = lean_acceleration(N, cap)
    tau_cap = float(tau_cap)
    print(f"     v x a peaks at {best[0]:.2f} deg -> lean cap {cap_deg} deg ({cap:.3f} rad); top speed {v_limit:.3f} "
          f"on the limiting corner -> cruise {top:.2f} m/s ({top / float(no_load_speed(N)):.0%} of nominal no-load; Pololu 53%)")

    # 2. Catch tests at the top speed and the top of the robustness latency range.
    if full:
        jf_all = largest_catchable_impulse(R, top, lat)
        jb = largest_catchable_impulse(R, top, lat, sign=-1.0).min()
        tilt_all = largest_catchable_tilt(R, lat)
        jf, tilt = jf_all.min(), tilt_all.min()
        jn = float(largest_catchable_impulse(N, top, 0.010)[0])
        tn = float(largest_catchable_tilt(N, 0.010)[0])
        worst_push = describe_worst(cfg, R, int(np.argmin(jf_all)))
        worst_tilt = describe_worst(cfg, R, int(np.argmin(tilt_all)))
    else:
        c = RECORDED_CATCH
        jf, jb, tilt, jn, tn = c["forward"], c["backward"], c["tilt"], c["nominal_forward"], c["nominal_tilt"]
        worst_push = worst_tilt = "recorded; run without --quick"
    print(f"\n  2. Catch tests at {top} m/s, {lat * 1000:.0f} ms latency (top of the robustness range), 50 ms push:")
    print(f"     forward push: {jf:.4f} N s on every corner (worst: {worst_push}); nominal at 10 ms {jn:.3f}")
    print(f"     backward push: {jb:.4f} N s on every corner")
    print(f"     start tilt: {tilt:.3f} rad on every corner (worst: {worst_tilt}); nominal at 10 ms {tn:.3f}")

    # 3. Cost limits, start tilts and pushes, in Pololu's proportions.
    # Pololu's pitch_limit is both 77% of its catchable tilt and 69% of its lean cap. When the two disagree,
    # take the smaller: above the cap's share, the goal profile (whose leans are shares of pitch_limit) would
    # ask for more lean than the controller is allowed.
    by_tilt = round(POLOLU_PITCH_LIMIT_OF_TILT * tilt * 100) / 100
    by_cap = round(POLOLU_PITCH_LIMIT_OF_CAP * cap * 100) / 100
    pitch_limit = min(by_tilt, by_cap)
    tilts = tuple(round(s * pitch_limit, 4) for s in POLOLU_START_TILT_SHARES)
    velocity_limit = round(POLOLU_VELOCITY_LIMIT_SHARE * top * 100) / 100
    torque_limit = round(POLOLU_TORQUE_LIMIT_OF_CAP * tau_cap * 100) / 100
    push_force = round(POLOLU_PUSH_SHARE * jf / 0.05, 2)
    print(f"\n  3. pitch_limit_rad: 77% of the catchable tilt {tilt:.3f} -> {by_tilt}; 69% of the lean cap -> {by_cap}; "
          f"the smaller: {pitch_limit} ({pitch_limit / cap:.0%} of the cap, {pitch_limit / tilt:.0%} of the tilt)")
    print(f"     start tilts {tilts} rad (Pololu's shares of pitch_limit: 1/6, -1/2, 1)")
    print(f"     velocity_limit: {POLOLU_VELOCITY_LIMIT_SHARE:.3f} x top speed -> {velocity_limit} m/s")
    print(f"     torque_limit: {POLOLU_TORQUE_LIMIT_OF_CAP:.2f} x nominal lean-cap torque {tau_cap:.3f} -> {torque_limit} N m "
          f"({torque_limit / cfg.robot.max_torque:.0%} of max_torque; Pololu 29%)")
    print(f"     pushes: {push_force} N x 50 ms = {push_force * 0.05:.3f} N s, {push_force * 0.05 / jf:.0%} of the catchable "
          f"{jf:.4f}; 50 ms is under the {1000 / np.max(unstable_pole(R)):.0f} ms fastest unstable time constant")

    # 4. Goal profile: config.py's start times; each blend time from the lean its change needs.
    reverse = -round(POLOLU_REVERSE_SHARE * top, 2)
    changes = {"aggressive": top, "brake": top - reverse, "gentle": -reverse}
    blends, accels, leans = {}, {}, {}
    for name, dv in changes.items():
        leans[name] = LEAN_SOFTENING * POLOLU_LEAN_SHARES[name] * pitch_limit
        xdd, _ = lean_acceleration(N, leans[name])
        blends[name] = round(BLEND_PEAK * dv / float(xdd) * 20) / 20     # to 0.05 s
        accels[name] = BLEND_PEAK * dv / blends[name]
    print("\n  4. Goal profile (start times as config.py):")
    print(f"     t=2.0  0 -> {top} in {blends['aggressive']} s: peak {accels['aggressive']:.2f} m/s^2, lean {leans['aggressive']:.3f} rad")
    print(f"     t=9.5  {top} -> {reverse} in {blends['brake']} s: peak {accels['brake']:.2f} m/s^2, lean {leans['brake']:.3f} rad")
    print(f"     t=14.0 {reverse} -> 0 in {blends['gentle']} s: peak {accels['gentle']:.2f} m/s^2, lean {leans['gentle']:.3f} rad")

    # 5. Fall-penalty bound (PHYSICS.md 8.3), as test_fall_penalty_exceeds_any_non_fall_cost computes it.
    T = cfg.scenario.duration_s
    vw, ew = (pitch_limit / velocity_limit) ** 2, (pitch_limit / torque_limit) ** 2
    v_max = float(np.max(no_load_speed(R)))
    terms = (cfg.timing.fall_angle_rad ** 2 * T, ew * cfg.robot.max_torque ** 2 * T, vw * (v_max + top) ** 2 * T)
    print(f"\n  5. Weights: velocity {vw:.2f}, effort {ew:.2f}. Worst non-fall cost over {T:.0f} s: pitch {terms[0]:.1f} + "
          f"effort {terms[1]:.1f} + velocity {terms[2]:.1f} = {sum(terms):.0f} (speed bound {v_max:.3f} + {top} m/s)")

    return dict(max_lean_deg=cap_deg, top_speed=top, reverse=reverse, blends=blends, push_force=push_force,
                start_tilts_rad=tilts, pitch_limit_rad=pitch_limit, velocity_limit=velocity_limit,
                torque_limit=torque_limit, worst_non_fall_cost=sum(terms))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="also re-derive the Pololu numbers")
    ap.add_argument("--quick", action="store_true", help="skip the catch simulations (use the recorded values)")
    args = ap.parse_args()
    if args.check:
        check_pololu()
        print()
    derive_legobot(full=not args.quick)


if __name__ == "__main__":
    main()
