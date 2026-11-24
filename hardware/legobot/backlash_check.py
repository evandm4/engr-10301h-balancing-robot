"""
Checks the step-event backlash model (balance_sim.Planar2D, PHYSICS.md 10.3; SOFTWARE_PLAN B5) against the
stiff tooth contact of the first B5 test, on the LEGO robot's plain scenario (nominal robot, seeds 1-3 x
3 start tilts, reference engine).

    python hardware/legobot/backlash_check.py              # about 10 minutes on 8 cores
    python hardware/legobot/backlash_check.py --quick      # one seed

The stiff model is the B5 test's: the motor side carries J, the motor torque and the friction, and drives the
wheel through a spring-damper that acts only past the dead zone (a 200 Hz tooth contact against J, critically
damped, one-sided so it never pulls), integrated at dt 0.1 ms. The step-event model (the one in the engines)
runs at the tuning's 2.5 ms, and at 0.1 ms to separate its own error from the step size. Both read the encoder
on the shaft between the two plays. The play cases: the B5 test's single play with the encoder on the motor
side (all of it in the coupling, +-0.5 and +-1.5 deg), and LEGO's measured split (gearbox / coupling).
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ["ROBOT"] = "legobot"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "simulation"), str(ROOT / "optimization")]
import numpy as np

from balance_sim.planar2d import CONTACT, GAP, MOTOR_SPEED, SHAFT, Planar2D
from config import CONFIG as C
from tuning import Evaluator

assert C.name == "legobot"
FINE_DT = 1e-4


class StiffBacklash(Planar2D):
    """The B5 test's contact: the play never 'closes' as an event; past +-b a one-sided spring-damper
    transmits T = k (gap -+ b) + c gap_rate between motor side and wheel. contact stays 0."""

    def __init__(self, params, fall_angle, actuator, frequency_hz=200.0, damping_ratio=1.0):
        super().__init__(params, fall_angle, actuator)
        w = 2 * math.pi * frequency_hz
        self.k = params.rotor_inertia * w * w
        self.c = 2 * damping_ratio * params.rotor_inertia * w

    def _backlash_terms(self, state, torque, force=0.0):
        p = self.p
        x_dot, theta, theta_dot = float(state[1]), float(state[2]), float(state[3])
        gap, motor_speed = float(state[GAP]), float(state[MOTOR_SPEED])
        wheel_speed = x_dot / p.wheel_radius - theta_dot
        tau = self.actuator.torque(torque, motor_speed)
        tau -= p.wheel_friction * motor_speed
        if p.coulomb_friction:
            tau -= p.coulomb_friction * float(np.tanh(motor_speed / p.coulomb_friction_speed))
        gap_rate = motor_speed - wheel_speed
        b = p.backlash
        contact = 0.0
        if gap > b:
            contact = max(0.0, self.k * (gap - b) + self.c * gap_rate)
        elif gap < -b:
            contact = min(0.0, self.k * (gap + b) + self.c * gap_rate)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        ml = p.body_mass * p.body_com_height
        a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
        bb = ml * cos_t
        d = p.body_mass * p.body_com_height**2 + p.body_inertia
        rhs_x = contact / p.wheel_radius + ml * sin_t * theta_dot**2
        rhs_t = -contact + ml * p.gravity * sin_t
        if force:
            rhs_x += force
            rhs_t += force * p.body_com_height * cos_t
        det = a * d - bb * bb
        x_ddot = (d * rhs_x - bb * rhs_t) / det
        theta_ddot = (a * rhs_t - bb * rhs_x) / det
        motor_accel = (tau - contact) / p.rotor_inertia
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot, gap_rate, motor_accel, 0.0, 0.0]), tau

    def begin_step(self, state, torque, force=0.0):
        return state

    def end_step(self, state):
        """Only the encoder's shaft (play operator); the gap itself may overshoot b while the spring is loaded."""
        p = self.p
        state = state.copy()
        gap = min(max(state[GAP], -p.backlash), p.backlash)
        wheel_angle = state[0] / p.wheel_radius - state[2]
        low = wheel_angle + max(gap - p.gearbox_backlash, -p.coupling_backlash)
        high = wheel_angle + min(gap + p.gearbox_backlash, p.coupling_backlash)
        state[SHAFT] = min(max(state[SHAFT], low), high)
        state[CONTACT] = 0.0
        return state


class StiffEvaluator(Evaluator):
    def dynamics(self):
        d = super().dynamics()
        return StiffBacklash(d.p, d.fall_angle, d.actuator)


def score(job):
    """job: (model, dt, gearbox deg, coupling deg, gains name, seeds) -> (mean cost, median cost, falls)."""
    model, dt, gearbox, coupling, gains_name, seeds = job
    base = C.build_eval_config()
    params = base.params.with_changes(gearbox_backlash=math.radians(gearbox), coupling_backlash=math.radians(coupling))
    config = replace(base, params=params, dt=dt, train_seeds=seeds)
    ev = (StiffEvaluator if model == "stiff" else Evaluator)(config, C.build_search_space())
    gains = C.controller.hand_tuned_gains if gains_name == "hand-tuned PID" else C.model_based_gains()
    r = ev.evaluate_gains(gains)
    return r.mean_cost, float(np.median(r.costs)), r.n_fell


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="seed 1 only")
    a = ap.parse_args()
    seeds = (1,) if a.quick else (1, 2, 3)
    plays = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.5), (2.6, 0.0), (2.8, 1.5), (3.0, 3.0)]
    models = [("event", C.timing.dt_s), ("event", FINE_DT), ("stiff", FINE_DT)]
    names = ("hand-tuned PID", "model-based PID")
    jobs = [(m, dt, g, c, n, seeds) for n in names for g, c in plays for m, dt in models]
    with ProcessPoolExecutor() as pool:
        results = dict(zip(jobs, pool.map(score, jobs)))
    print(f"Plain scenario, nominal LEGO robot, seeds {seeds} x {len(C.scenario.start_pitches_rad)} tilts: "
          f"mean cost (falls), and the change in the median from no play (a fall moves the mean by ~100)")
    header = "".join(f"{f'{m} {dt * 1e3:g} ms':>24}" for m, dt in models)
    for n in names:
        print(f"\n{n}\n{'play gearbox/coupling (deg)':<30}{header}")
        for g, c in plays:
            cells = []
            for m, dt in models:
                cost, median, fell = results[(m, dt, g, c, n, seeds)]
                base = results[(m, dt, 0.0, 0.0, n, seeds)][1]
                cells.append(f"{cost:8.2f} ({fell}) {100 * (median / base - 1):+6.0f}%")
            print(f"{f'+-{g:g} / +-{c:g}':<30}" + "".join(f"{s:>24}" for s in cells))


if __name__ == "__main__":
    main()
