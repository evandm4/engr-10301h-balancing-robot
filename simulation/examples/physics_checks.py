"""Reproduce every number quoted in PHYSICS.md (the derivation of the simulation's physics).

Each block prints the values one section of PHYSICS.md relies on, under the
same section number. Robot, motor, timing and scenario all come from config.py
(the datasheet values since 2026-11-08, rotor inertia and gearbox efficiency
included); the tuned gains (TUNED) are the CMA-ES result tuned for that robot,
named below. Several sections also show TUNED_1106, the gains tuned for the
robot as it was before 2026-11-08, to show what the realistic constants change.
Takes under a minute.

From the project folder:   python simulation/examples/physics_checks.py
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balance_sim import CascadedPIDController, Planar2D, PIDController, pitch_error_signal, simulate
from balance_sim.actuators import Actuator
from balance_sim.simulator import _rk4_step
from batch_sim import BatchRobot, actuator_torque, derivatives
from config import CONFIG

TUNED = (1.6351051516678046, 1.765803347356468, 0.29857655444040127, 0.6197806802419683)  # CMA-ES, results/batch_tuning/20261108-153813 (20 s scenario with pushes)
TUNED_1106 = (0.659938589993706, 0.4980220464276531, 0.11153595895763543, 0.7956961244475451)  # CMA-ES, 2026-11-06, earlier robot
HAND = tuple(CONFIG.controller.hand_tuned_gains)

P = CONFIG.build_robot_params()
MOTOR = CONFIG.actuator.build_motor_params()
M, L, IB, MW, R, IW, G = (P.body_mass, P.body_com_height, P.body_inertia, P.wheel_mass,
                          P.wheel_radius, P.wheel_inertia, P.gravity)
J = P.rotor_inertia              # motor rotors reflected to the axle, n N^2 J_r
ML = M * L
A = M + MW + IW / R**2 + J / R**2   # effective translating mass
B0 = ML - J / R                  # mass-matrix coupling at theta = 0
D = M * L**2 + IB + J            # body (plus rotor) inertia about the axle
DET0 = A * D - B0**2             # mass-matrix determinant at theta = 0
C = A + B0 / R                   # torque-to-restoring-moment factor (J cancels)
E = D / R + B0                   # torque-to-forward-acceleration factor (times 1/Delta_0)
ETA = MOTOR.gear_efficiency
C_M = ETA * MOTOR.n_motors * MOTOR.gear_ratio**2 * MOTOR.torque_constant**2 / MOTOR.resistance   # back-EMF damping


def clip(x, limit):
    return max(-limit, min(limit, x))


def linear_accelerations():
    """(x'', theta'') per unit theta and per unit torque, from the upright linearization (PHYSICS.md 3.1)."""
    minv = np.linalg.inv([[A, B0], [B0, D]])
    return minv @ [0, ML * G], minv @ [1 / R, -1]


def heading(text):
    print(f"\n=== {text} ===")


def controller(gains, goal=0.0, torque_limit=None):
    return CascadedPIDController.from_gains(
        gains, torque_limit=P.max_torque if torque_limit is None else torque_limit, pitch_integral_limit=CONFIG.controller.pitch_integral_limit,
        goal=goal, max_pitch_command=CONFIG.controller.max_lean_rad,
        integral_limit=CONFIG.controller.velocity_integral_limit)


def scenario_run(gains, tilt, seed, dynamics=None, dt=None, latency=None, torque_limit=None):
    """One run of the tuner's scenario (goal profile, pushes, noise, latency) with the reference simulator."""
    dyn = CONFIG.build_dynamics() if dynamics is None else dynamics
    return simulate(dyn, controller(gains, CONFIG.build_goal(), torque_limit),
                    duration=CONFIG.scenario.duration_s, dt=CONFIG.timing.dt_s if dt is None else dt,
                    initial_state=dyn.initial_state(tilt), control_dt=CONFIG.timing.control_dt_s,
                    sensor=CONFIG.build_sensor(seed),
                    latency=CONFIG.timing.latency_s if latency is None else latency,
                    disturbance=CONFIG.build_disturbance())


def validation_runs(gains, dynamics=None, torque_limit=None):
    return [scenario_run(gains, tilt, seed, dynamics, torque_limit=torque_limit)
            for seed in CONFIG.scenario.validation_seeds for tilt in CONFIG.scenario.start_pitches_rad]


# ---- 1-2: constants and the equations of motion -------------------------------------------

def constants():
    heading("1. Derived constants")
    print(f"M l = {ML:.4f} kg m    I_w/r^2 = {IW / R**2:.6f} kg    J = {J:.3e} kg m^2    J/r^2 = {J / R**2:.4f} kg"
          f"    J/r = {J / R:.5f} kg m")
    print(f"a = M + m_w + (I_w + J)/r^2 = {A:.6f} kg    b0 = M l - J/r = {B0:.5f} kg m    d = M l^2 + I_b + J = {D:.5f} kg m^2")
    print(f"Delta_0 = a d - b0^2 = {DET0:.6e} kg^2 m^2    c = a + b0/r = {C:.4f} kg (= M + m_w + I_w/r^2 + M l/r)"
          f"    e = d/r + b0 = {E:.5f} kg m")
    print(f"smallest det over all theta = a d - (M l + J/r)^2 = {A * D - (ML + J / R)**2:.6e} > 0 (at theta = pi)")
    print(f"gear efficiency {ETA}; with J = 0: a = {M + MW + IW / R**2:.4f}, d = {M * L**2 + IB:.5f}")


def power_balance():
    heading("2.7 Power balance: E(T) - E(0) vs integral of (tau - b w) w dt")
    p = P.with_changes(wheel_friction=0.001)
    dyn = Planar2D(p, CONFIG.timing.fall_angle_rad)
    res = simulate(dyn, controller(TUNED, CONFIG.build_goal()), duration=10.0, dt=1e-4,
                   initial_state=dyn.initial_state(0.1), control_dt=CONFIG.timing.control_dt_s,
                   latency=CONFIG.timing.latency_s)
    s, t = res.states, res.t
    w = s[:, 1] / R - s[:, 3]
    tau = np.clip(res.torques[:-1], -P.max_torque, P.max_torque)   # held over each step
    power_left = tau * w[:-1] - p.wheel_friction * w[:-1] ** 2
    power_right = tau * w[1:] - p.wheel_friction * w[1:] ** 2
    work = np.sum(0.5 * (power_left + power_right) * np.diff(t))
    change = dyn.energy(s[-1]) - dyn.energy(s[0])
    print(f"energy change {change:.6f} J, work in {work:.6f} J, difference {change - work:.1e} J")


def steady_lean():
    heading("2.8 Steady lean <-> steady acceleration")
    gain = ML * G / (B0 + A * R)
    lean = CONFIG.controller.max_lean_rad
    accel = ML * G * math.sin(lean) / (ML * math.cos(lean) - J / R + A * R)
    print(f"linear: x'' = {gain:.3f} theta m/s^2 (point-mass cart would give g = {G})")
    print(f"at the {math.degrees(lean):.0f} deg lean cap: x'' = {accel:.3f} m/s^2, needs tau = a r x'' = {A * R * accel:.4f} N m")


# ---- 3: linearization -------------------------------------------------------------------

def linear_model():
    heading("3. Linearized model")
    pole = math.sqrt(A * ML * G / DET0)
    zero = math.sqrt(ML * G / (D + B0 * R))
    print(f"A[2,3] = -b0 M l g / Delta_0 = {-B0 * ML * G / DET0:.4f}    A[4,3] = a M l g / Delta_0 = {A * ML * G / DET0:.4f}")
    print(f"B[2] = e / Delta_0 = {E / DET0:.4f}    B[4] = -c / Delta_0 = {-C / DET0:.4f}")
    print(f"open-loop poles 0, 0, +/-{pole:.3f} rad/s (time constant {1000 / pole:.1f} ms, "
          f"doubling time {1000 * math.log(2) / pole:.1f} ms); wheels locked: {math.sqrt(ML * G / D):.3f} rad/s")
    dyn = CONFIG.build_dynamics(ideal_actuator=True)
    res = simulate(dyn, PIDController(0.0), duration=3.0, initial_state=dyn.initial_state(0.05))
    print(f"uncontrolled fall from 0.05 rad: {res.fall_time:.3f} s (linear estimate "
          f"{math.acosh(CONFIG.timing.fall_angle_rad / 0.05) / pole:.3f} s)")
    print(f"RHP zero of X/T at {zero:.3f} rad/s")
    run = scenario_run(TUNED, CONFIG.scenario.start_pitches_rad[0], 100)
    goal = CONFIG.build_goal()
    for step in CONFIG.scenario.goal_steps[:2]:      # the speed-up and the brake-and-reverse
        start, span = step[0], 0.4
        pick = np.min if step[1] > goal(start - 1e-3) else np.max   # wrong way = opposite to the change
        i = np.searchsorted(run.t, start)
        window = run.states[i:i + int(round(span / CONFIG.timing.dt_s)), 1]
        print(f"  tuned, seed 100: goal step at {start} s, v = {run.states[i, 1]:.3f} m/s, then "
              f"{'min' if pick is np.min else 'max'} {pick(window):.3f} m/s within {span} s (wrong way first)")
    kp_crit = A * ML * G / C
    print(f"critical Kp = a M l g / c = {kp_crit:.4f} N m/rad (M g l = {ML * G:.4f}; c/a = {C / A:.3f})")
    for name, (kp, ki, kd, _) in (("hand-tuned", HAND), ("tuned", TUNED), ("tuned 2026-11-06", TUNED_1106)):
        roots = np.real_if_close(np.sort_complex(np.roots([DET0, C * kd, C * kp - A * ML * G, C * ki])))
        bound = kp_crit + DET0 * ki / (C * kd)
        print(f"{name}: pitch-loop poles {np.round(roots, 2)} rad/s; "
              f"Routh-Hurwitz needs kp > {bound:.4f}; kp/kp_crit = {kp / kp_crit:.2f}")


# ---- 4: actuator ------------------------------------------------------------------------

def actuator_envelope():
    heading("4. DC motor envelope")
    n, N, kt, res, vs, imax = (MOTOR.n_motors, MOTOR.gear_ratio, MOTOR.torque_constant,
                               MOTOR.resistance, MOTOR.supply_voltage, MOTOR.current_limit)
    corner = (vs - res * imax) / (kt * N)
    print(f"stall torque {MOTOR.stall_torque_limit:.3f} N m = n N eta k_t I_max (V/R = {vs / res:.3f} A > limit {imax} A); "
          f"torque cap max_torque = {P.max_torque} N m")
    print(f"no-load speed {MOTOR.no_load_speed:.1f} rad/s = {MOTOR.no_load_speed * R:.2f} m/s; "
          f"corner speed {corner:.1f} rad/s = {corner * R:.2f} m/s")
    for v in (0.8, 1.0, 1.4, 2.0):
        w = v / R
        tau = n * N * ETA * kt * min(imax, (vs - kt * N * w) / res)
        print(f"  {v} m/s: back-EMF {kt * N * w:.2f} V, forward torque available {tau:.3f} N m")
    print(f"copper loss = R / (n N^2 eta^2 k^2) = {_copper_loss_factor():.0f} W per (N m)^2 of torque")
    print(f"back-EMF damping of a voltage-mode driver c_m = eta n N^2 k^2 / R = {C_M:.4f} N m s/rad")


# ---- 5: sensor --------------------------------------------------------------------------

def sensor_noise():
    heading("5. Sensor noise as torque noise (per control tick, 1 sigma)")
    s = CONFIG.sensor
    for name, (kp, _, kd, kv) in (("hand-tuned", HAND), ("tuned", TUNED)):
        parts = (kp * s.pitch_std, kd * s.pitch_rate_std, kp * kv * s.velocity_std)
        print(f"{name}: pitch {parts[0]:.4f}, gyro {parts[1]:.4f}, velocity {parts[2]:.4f}, "
              f"combined {math.hypot(*parts):.4f} N m; gyro bias offset {kd * s.pitch_rate_bias:.5f} N m; "
              f"PD-only velocity error {kd * s.pitch_rate_bias / (kp * kv):.4f} m/s; "
              f"pitch offset {s.pitch_bias} rad: {kp * s.pitch_bias:.5f} N m, steady velocity error with or "
              f"without ki b_theta / kv = {s.pitch_bias / kv:.4f} m/s")


# ---- 6-7: control, timing and integration -----------------------------------------------

def _cascade_polynomial(gains):
    """Characteristic polynomial of the continuous cascade with no delay (PHYSICS.md 6.3), highest power first."""
    kp, ki, kd, kv = gains
    e, h = E, ML * G / R
    return [DET0, C * kd - e * kp * kv, C * kp - A * ML * G - e * ki * kv, C * ki + h * kp * kv, h * ki * kv]


def _hurwitz(coeffs):
    a4, a3, a2, a1, a0 = coeffs
    return all(x > 0 for x in coeffs) and a3 * a2 > a4 * a1 and a3 * a2 * a1 > a4 * a1**2 + a3**2 * a0


def cascade_stability():
    heading("6.3 Cascade stability: how much outer-loop gain the inner loop allows")
    e = E
    print(f"e = d/r + b0 = {e:.5f} kg m, h = M g l / r = {ML * G / R:.4f} N")
    limits = {}
    for name, gains in (("hand-tuned", HAND), ("tuned", TUNED)):
        kp, ki, kd, kv = gains
        lo, hi = 0.0, 2.0   # bisect for the largest kv_p that passes Routh-Hurwitz
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _hurwitz(_cascade_polynomial((kp, ki, kd, mid))) else (lo, mid)
        roots = np.sort_complex(np.roots(_cascade_polynomial(gains)))
        # Same loop as a state matrix over (v, theta, theta_dot, integral), to check the polynomial.
        acc_theta, acc_tau = linear_accelerations()
        error_row = np.array([kv, 1.0, 0.0, 0.0])
        torque_row = kp * error_row + [0, 0, kd, ki]
        loop = np.array([[0, acc_theta[0], 0, 0], [0, 0, 1, 0], [0, acc_theta[1], 0, 0], error_row])
        loop[0] += acc_tau[0] * torque_row
        loop[2] += acc_tau[1] * torque_row
        gap = np.max(np.abs(roots - np.sort_complex(np.linalg.eigvals(loop))))
        limits[name] = lo
        print(f"{name}: kd/kp = {kd / kp:.4f} s; kv_p < c kd / (e kp) = {C * kd / (e * kp):.4f} (s^3 coefficient), "
              f"Routh-Hurwitz limit {lo:.4f}; poles at kv_p = {kv:.3g}: {np.round(np.real_if_close(roots), 2)} "
              f"(state-matrix eigenvalues differ by {gap:.1e})")
    dyn = CONFIG.build_dynamics(ideal_actuator=True)
    near = round(limits["hand-tuned"], 2)
    for kv in (near - 0.03, near + 0.03):   # hand-tuned inner loop either side of its limit, no latency
        res = simulate(dyn, controller(HAND[:3] + (kv,)), duration=10.0, dt=0.0005,
                       initial_state=dyn.initial_state(0.02), control_dt=CONFIG.timing.control_dt_s)
        outcome = (f"fell at {res.fall_time:.2f} s" if res.fell
                   else f"max |theta| in last 1 s = {np.max(np.abs(res.states[res.t > 9, 2])):.1e} rad")
        print(f"  nonlinear, hand-tuned inner loop, kv_p = {kv:.2f}, no latency: {outcome}")
    G_lean = ML * G / (B0 + A * R)
    for name, kv in (("hand-tuned", HAND[3]), ("tuned", TUNED[3])):
        print(f"{name}: outer-loop bandwidth estimate G kv_p = {G_lean * kv:.2f} rad/s (time constant {1 / (G_lean * kv):.2f} s)")
    print(f"RHP-zero rule of thumb: bandwidth < z/2 = {math.sqrt(ML * G / (D + B0 * R)) / 2:.2f} rad/s")


def _discrete_loop(gains, latency, cascade=True):
    """Closed-loop transition matrix of the linear plant under the sampled, delayed controller.

    Exact for the linear model: torque is held between ticks and a command
    reaches the plant `latency` seconds after its tick. Plant state is
    (v, theta, theta_dot); position is left out (nothing depends on it).
    """
    kp, ki, kd, kv = gains
    T = CONFIG.timing.control_dt_s
    acc_theta, acc_tau = linear_accelerations()
    a_mat = np.array([[0, acc_theta[0], 0], [0, 0, 1], [0, acc_theta[1], 0]])
    b_vec = np.array([acc_tau[0], 0, acc_tau[1]])

    def hold(h):   # (e^{A h}, integral of e^{A s} B over [0, h])
        z = np.zeros((4, 4))
        z[:3, :3], z[:3, 3] = a_mat, b_vec
        e = expm(z * h)
        return e[:3, :3], e[:3, 3]

    q = int(math.floor(latency / T + 1e-12))
    early = latency - q * T                      # time each period still runs the older command
    phi, _ = hold(T)
    inputs = {q: hold(T - early)[1]}             # u_{k-q}, for the end of the period
    if early > 1e-12:
        inputs[q + 1] = hold(T - early)[0] @ hold(early)[1]   # u_{k-q-1}, for the start
    slots = max(inputs)
    error_row = np.array([kv if cascade else 0.0, 1.0, 0.0])            # e = theta + kv v (goal 0)
    gain_row = (kp + ki * T) * error_row + kd * np.array([0, 0, 1.0])   # u_k = gain_row s + ki I_{k-1}
    n = 3 + 1 + slots
    m = np.zeros((n, n))
    m[:3, :3] = phi
    if 0 in inputs:
        m[:3, :3] += np.outer(inputs[0], gain_row)
        m[:3, 3] += inputs[0] * ki
    for j, b in inputs.items():
        if j >= 1:
            m[:3, 3 + j] += b
    m[3, :3], m[3, 3] = T * error_row, 1.0
    if slots >= 1:
        m[4, :3], m[4, 3] = gain_row, ki
        for j in range(2, slots + 1):
            m[3 + j, 2 + j] = 1.0
    eig = np.linalg.eigvals(m)
    if not cascade:   # pitch-only control leaves velocity undriven: drop its eigenvalue at exactly 1
        eig = np.delete(eig, np.argmin(np.abs(eig - 1)))
    return eig


def _delay_margin(gains, cascade=True):
    """Smallest latency (0.25 ms grid, up to 80 ms) at which the sampled linear loop is unstable."""
    for latency in np.arange(0.0, 0.08, 0.00025):
        if np.max(np.abs(_discrete_loop(gains, latency, cascade))) >= 1:
            return latency
    return math.inf


def delay_margins():
    heading("7.4 Delay margins (linear, exact for sampled + delayed control)")
    T = CONFIG.timing.control_dt_s
    cases = (("hand-tuned pitch-only (4, 0.5, 0.15)", HAND[:3] + (0.0,), False),
             ("hand-tuned cascade", HAND, True),
             ("tuned cascade", TUNED, True),
             ("tuned 2026-11-06 cascade", TUNED_1106, True))
    margin_of = {}
    for name, gains, cascade in cases:
        margin_of[name] = _delay_margin(gains, cascade)
        s = np.log(_discrete_loop(gains, CONFIG.timing.latency_s, cascade).astype(complex)) / T
        s = s[np.abs(s.imag) < math.pi / T - 1e-6]   # drop real negative z (no continuous-time twin)
        osc = s[s.imag > 1e-6]
        worst = osc[np.argmin(-osc.real / np.abs(osc))]
        print(f"{name}: unstable from {margin_of[name] * 1000:.2f} ms; at 10 ms the least-damped mode is "
              f"{abs(worst):.1f} rad/s ({abs(worst) / (2 * math.pi):.1f} Hz), damping ratio {-worst.real / abs(worst):.3f}")
    for name, inner in (("hand-tuned", HAND[:3]), ("tuned", TUNED[:3])):
        margins = []
        for kv in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            margin = _delay_margin(tuple(inner) + (kv,))
            margins.append(f"{kv}: " + ("unstable" if margin == 0 else f"{margin * 1000:.2f} ms"))
        print(f"{name} inner loop, delay margin by kv_p: " + ", ".join(margins))

    print("nonlinear check (ideal actuator, no noise, 0.02 rad start, 0.5 ms physics step):")
    dyn = CONFIG.build_dynamics(ideal_actuator=True)
    around = lambda m: tuple(t for t in (0.5 * m, m - 0.001, m, m + 0.001) if t >= 0)   # either side of the linear margin
    for name, make, latencies, duration in (
            ("hand-tuned pitch-only", lambda: CONFIG.build_pitch_controller(),
             around(margin_of["hand-tuned pitch-only (4, 0.5, 0.15)"]), 6.0),
            ("tuned cascade", lambda: controller(TUNED), around(margin_of["tuned cascade"]), 10.0)):
        for lat in latencies:
            res = simulate(dyn, make(), duration=duration, dt=0.0005, initial_state=dyn.initial_state(0.02),
                           control_dt=CONFIG.timing.control_dt_s, latency=lat)
            last = "fell" if res.fell else f"max |theta| in last 1 s = {np.max(np.abs(res.states[res.t > duration - 1, 2])):.1e} rad"
            print(f"  {name}, {lat * 1000:.1f} ms: {last}")


def integration_accuracy():
    heading("7.1 RK4 order: relative energy error, free fall from 0.3 rad for 0.3 s")
    dyn = CONFIG.build_dynamics(ideal_actuator=True)
    previous = None
    for dt in (0.01, 0.005, 0.0025, 0.00125, 0.000625):
        state = dyn.initial_state(0.3)
        e0 = dyn.energy(state)
        for _ in range(int(round(0.3 / dt))):
            state = _rk4_step(dyn, state, 0.0, dt)
        err = abs(dyn.energy(state) - e0) / e0
        print(f"  dt {dt * 1000:.3f} ms: {err:.2e}" + (f"  (ratio {previous / err:.1f})" if previous else ""))
        previous = err
    acc_theta, acc_tau = linear_accelerations()
    for damping in (0.0, C_M):   # torque = -damping * (x_dot / r - theta_dot): back-EMF once the voltage saturates
        torque_row = np.array([0, -damping / R, 0, damping])
        plant = np.zeros((4, 4))
        plant[0, 1] = plant[2, 3] = 1
        plant[1] = acc_tau[0] * torque_row + [0, 0, acc_theta[0], 0]
        plant[3] = acc_tau[1] * torque_row + [0, 0, acc_theta[1], 0]
        fastest = np.max(np.abs(np.linalg.eigvals(plant)))
        print(f"  plant with {damping:.4f} N m s/rad damping: fastest |lambda| {fastest:.1f} rad/s, "
              f"|lambda| dt = {fastest * CONFIG.timing.dt_s:.3f} (RK4 stable to about 2.8)")


def cost_accuracy():
    heading("7.5 Cost vs physics step (seed 100, start tilt 5.73 deg)")
    tilt, jumps = CONFIG.scenario.start_pitches_rad[0], np.diff([0.0] + [step[1] for step in CONFIG.scenario.goal_steps])
    for name, gains in (("tuned", TUNED), ("hand-tuned", HAND)):
        fine = scenario_run(gains, tilt, 100, dt=0.00025)
        coarse = scenario_run(gains, tilt, 100)
        c_fine, c_coarse = CONFIG.score(fine), CONFIG.score(coarse)
        t_fine, t_coarse = _trapezoid_cost(fine), _trapezoid_cost(coarse)
        gap = np.max(np.abs(fine.states[::10] - coarse.states), axis=0)   # at the shared 2.5 ms instants
        print(f"{name}: left sum {c_fine:.5f} (0.25 ms) vs {c_coarse:.5f} (2.5 ms), {(c_coarse / c_fine - 1) * 100:+.2f}%;"
              f" trapezoid {t_fine:.5f} vs {t_coarse:.5f}, {(t_coarse / t_fine - 1) * 100:+.4f}%")
        print(f"  largest state gap: v {gap[1]:.1e} m/s, theta {gap[2]:.1e} rad")
    if CONFIG.scenario.goal_transition_s == 0:
        print(f"predicted left-sum bias (dt/2) sum(dv_goal^2) = {CONFIG.timing.dt_s / 2 * np.sum(jumps**2):.5f}")


def cost_breakdown():
    heading("8. Cost terms, mean over the 30 validation runs (nominal robot, configured cost)")
    c = CONFIG.cost
    for name, gains in (("tuned", TUNED), ("tuned 2026-11-06", TUNED_1106), ("hand-tuned", HAND)):
        terms = []
        for res in validation_runs(gains):
            dt, s = np.diff(res.t), res.states
            terms.append((np.sum(pitch_error_signal(res, c.pitch_reference)[:-1] ** 2 * dt),
                          c.effort_weight * np.sum(res.torques[:-1] ** 2 * dt),
                          c.velocity_weight * np.sum((s[:-1, 1] - res.velocity_goal[:-1]) ** 2 * dt)))
        pitch, effort, velocity = np.mean(terms, axis=0)
        total = pitch + effort + velocity
        print(f"{name}: pitch {pitch:.4f} ({pitch / total:.1%}), effort {effort:.5f} ({effort / total:.2%}), "
              f"velocity {velocity:.4f} ({velocity / total:.1%}), total {total:.4f}; "
              f"copper loss {effort / c.effort_weight * _copper_loss_factor():.1f} J per run")
    lean = CONFIG.controller.max_lean_rad
    accel = ML * G * math.sin(lean) / (ML * math.cos(lean) - J / R + A * R)
    jumps = np.abs(np.diff([0.0] + [step[1] for step in CONFIG.scenario.goal_steps]))
    if CONFIG.scenario.goal_transition_s > 0:
        return   # the floor below is for instant goal jumps; a smooth goal has none
    print(f"velocity term if each goal step were met at the steady {math.degrees(lean):.0f} deg-lean acceleration "
          f"({accel:.2f} m/s^2): sum |dv|^3 / (3 a) = {c.velocity_weight * np.sum(jumps**3) / (3 * accel):.3f}")


def _copper_loss_factor():
    m = MOTOR
    return m.resistance / (m.n_motors * m.gear_ratio**2 * m.gear_efficiency**2 * m.torque_constant**2)


def _trapezoid_cost(res):
    """The tuner's cost with trapezoidal integration (references held over each step, as in the run)."""
    c, s, dt = CONFIG.cost, res.states, np.diff(res.t)
    pitch_err, goal = pitch_error_signal(res, c.pitch_reference), res.velocity_goal[:-1]
    trap = lambda left, right: np.sum(0.5 * (left**2 + right**2) * dt)
    # The reference is held over each step, as in the run, so only the state moves between ends.
    pitch_ref = s[:-1, 2] - pitch_err[:-1]
    pitch = trap(pitch_err[:-1], s[1:, 2] - pitch_ref)
    # A step goal jumps at a step boundary, so it is held over each step. A smooth goal
    # moves within the step like the state, so it is taken at both ends (holding it would
    # add a first-order error of its own).
    goal_end = res.velocity_goal[1:] if CONFIG.scenario.goal_transition_s > 0 else goal
    velocity = trap(s[:-1, 1] - goal, s[1:, 1] - goal_end)
    return pitch + c.effort_weight * np.sum(res.torques[:-1] ** 2 * dt) + c.velocity_weight * velocity


# ---- 9: assumption audit ----------------------------------------------------------------

def assumption_audit():
    heading("9.1-9.5 Assumption audit over the 30 validation runs")
    _, (xdd_per_tau, thdd_per_tau) = linear_accelerations()
    k = ((M + MW) * xdd_per_tau + ML * thdd_per_tau) / ((M + MW) * G)   # Newton, whole robot, upright at rest
    print(f"upright, at rest: |F|/N = {k:.3f} per N m, so traction caps torque at mu / {k:.3f} "
          f"({0.8 / k:.3f} N m for mu = 0.8)")
    static = brentq(lambda th: A * ML * G * math.sin(th) - P.max_torque * (A + (ML * math.cos(th) - J / R) / R),
                    0.1, math.pi / 2)
    print(f"static hold limit (theta_dot = 0): {math.degrees(static):.1f} deg")
    robot = BatchRobot.from_params(P, MOTOR)
    for name, gains in (("tuned", TUNED), ("hand-tuned", HAND)):
        runs = validation_runs(gains)
        s = np.concatenate([r.states[:-1] for r in runs])
        u = np.concatenate([r.torques[:-1] for r in runs])
        x, v, th, om = s.T
        _, xdd, _, thdd = derivatives(robot, (x, v, th, om), u)
        w = v / R - om
        delivered = actuator_torque(robot, u, w)
        friction_force = (M + MW) * xdd + M * L * (np.cos(th) * thdd - np.sin(th) * om**2)   # Newton, whole robot
        # wheel spin equation: the rotors take J (x''/r - theta'') of the motor torque
        wheel_force = (delivered - P.wheel_friction * w - IW * xdd / R - J * (xdd / R - thdd)) / R
        normal = (M + MW) * G - M * L * (np.sin(th) * thdd + np.cos(th) * om**2)
        ratio = np.abs(friction_force) / normal
        short = np.abs(delivered - np.clip(u, -P.max_torque, P.max_torque)) > 1e-9
        print(f"{name}: Newton vs Lagrange ground force max diff {np.max(np.abs(friction_force - wheel_force)):.1e} N")
        print(f"  max |F|/N {ratio.max():.2f}; time above 0.6: {np.mean(ratio > 0.6) * 100:.2f}%, above 0.8: "
              f"{np.mean(ratio > 0.8) * 100:.2f}%; min N {normal.min():.2f} N")
        print(f"  max |x''| {np.abs(xdd).max():.1f} m/s^2 (raw accelerometer tilt error up to "
              f"{math.degrees(math.atan(np.abs(xdd).max() / G)):.0f} deg); max |v| {np.abs(v).max():.2f} m/s; "
              f"max |theta| {math.degrees(np.abs(th).max()):.1f} deg")
        print(f"  motor delivers less than the clipped command {np.mean(short) * 100:.2f}% of the time; "
              f"falls {sum(r.fell for r in runs)}/{len(runs)}")


class VoltageModeActuator(Actuator):
    """PWM duty proportional to the torque command, with no back-EMF compensation.

    command_limit: the largest torque command (N m). P.max_torque by default; the
    command that maps to the full supply voltage, FULL_SUPPLY_COMMAND, lets the duty reach 100%.
    """

    def __init__(self, command_limit=None):
        self.command_limit = P.max_torque if command_limit is None else command_limit

    def torque(self, command, relative_wheel_speed):
        m = MOTOR
        command = clip(command, self.command_limit)
        per_amp = m.n_motors * m.gear_ratio * m.gear_efficiency * m.torque_constant
        volts = clip(m.resistance * command / per_amp, m.supply_voltage)
        current = (volts - m.torque_constant * m.gear_ratio * relative_wheel_speed) / m.resistance
        return per_amp * clip(current, m.current_limit)


def friction():
    heading("9.4 Friction sensitivity (one run: seed 100, 5.73 deg start)")
    tilt = CONFIG.scenario.start_pitches_rad[0]
    for b in (0.0, 0.001, 0.002, 0.003, 0.005):
        p = P.with_changes(wheel_friction=b)
        row = []
        for name, gains in (("tuned", TUNED), ("tuned 2026-11-06", TUNED_1106), ("hand-tuned", HAND)):
            res = scenario_run(gains, tilt, 100, Planar2D(p, CONFIG.timing.fall_angle_rad, CONFIG.build_actuator(p)))
            row.append(f"{name} " + (f"fell at {res.fall_time:.2f} s after reaching {np.max(res.states[:, 1]):.2f} m/s"
                                     if res.fell else f"cost {CONFIG.score(res):.2f}"))
        print(f"b = {b}: " + ", ".join(row) + f"  (friction at 1.4 m/s: {b * 1.4 / R:.3f} N m)")

    heading("9.4 Load torque: speed shift before the integrator catches up")
    goal = 1.4
    for name, (kp, ki, kd, kv) in (("tuned", TUNED), ("tuned 2026-11-06", TUNED_1106), ("hand-tuned", HAND)):
        threshold = R * kp * kv
        shifts = ", ".join(f"b = {b:.4f}: {goal / (1 - b / threshold):.3f} m/s" if b < threshold
                           else f"b = {b:.4f}: no steady speed" for b in (0.001, 0.003, C_M))
        print(f"{name}: runaway threshold r kp kv = {threshold:.5f} N m s/rad; quasi-steady speed for a "
              f"{goal} m/s goal with ki = 0: {shifts}")
        p = P.with_changes(wheel_friction=0.003)
        dyn = Planar2D(p, CONFIG.timing.fall_angle_rad)   # ideal actuator: friction is the only load
        res = simulate(dyn, controller((kp, 0.0, kd, kv), goal), duration=20.0, dt=CONFIG.timing.dt_s,
                       initial_state=dyn.initial_state(0.0), control_dt=CONFIG.timing.control_dt_s,
                       latency=CONFIG.timing.latency_s)
        print(f"  simulated, b = 0.003, ki = 0, ideal actuator: v after 20 s = {res.states[-1, 1]:.3f} m/s")
    m, b = MOTOR, 0.003   # n N eta k (V_s - k N w) / R = b w, solved for w
    headroom_zero = ETA * m.n_motors * m.gear_ratio * m.torque_constant * m.supply_voltage / (
        m.resistance * b + ETA * m.n_motors * m.gear_ratio**2 * m.torque_constant**2)
    print(f"with b = 0.003 the DC motor's forward headroom reaches zero at {headroom_zero:.1f} rad/s = {headroom_zero * R:.2f} m/s")


def driver_modes():
    heading("9.3 Motor driver: current-mode (simulated) vs voltage-mode, 30 validation runs")
    m = MOTOR
    per_amp = m.n_motors * m.gear_ratio * m.gear_efficiency * m.torque_constant
    full = per_amp * m.supply_voltage / m.resistance   # the command whose voltage is the whole supply
    print(f"voltage at the {P.max_torque} N m command cap: {m.resistance * P.max_torque / per_amp:.2f} V of "
          f"{m.supply_voltage} V; full supply = a {full:.3f} N m command")
    capped = Planar2D(P, CONFIG.timing.fall_angle_rad, VoltageModeActuator())
    full_robot = P.with_changes(max_torque=full)
    full_duty = Planar2D(full_robot, CONFIG.timing.fall_angle_rad, VoltageModeActuator(full))
    for name, gains in (("tuned", TUNED), ("tuned 2026-11-06", TUNED_1106), ("hand-tuned", HAND)):
        for label, dyn, limit in (("current-mode", None, None), ("voltage-mode, duty capped at max_torque", capped, None),
                                  ("voltage-mode, duty up to the full supply", full_duty, full)):
            runs = validation_runs(gains, dyn, torque_limit=limit)
            print(f"{name}, {label}: falls {sum(r.fell for r in runs)}/30, "
                  f"mean cost {np.mean([CONFIG.score(r) for r in runs]):.3f}")



def main():
    constants()
    power_balance()
    steady_lean()
    linear_model()
    actuator_envelope()
    sensor_noise()
    cascade_stability()
    integration_accuracy()
    delay_margins()
    cost_accuracy()
    cost_breakdown()
    assumption_audit()
    driver_modes()
    friction()


if __name__ == "__main__":
    main()
