"""Observation ranges of the LEGO robot's closed loop, to size legobot_config.LEARNING's scales
(docs/ml-baselines-plan.md 13). Re-run when the reference PID changes (after E2).

Runs, in the one-robot reference simulator with the LEGO models (IMU filter, slow-PWM voltage drive):
  * the robot's reference PID (config.reference_pid_gains: the model-based PID since E2.3), recording
    its pitch setpoint and pitch integral each tick;
  * its PID-equivalent linear policy (leaky, freeze-on-saturation integrators), recording the
    raw OBSERVATIONS each tick.
Dev seeds only (5000-...), the 60 s validation scenario, the nominal robot. Runs that fall are counted but left\r
out of the statistics (E2.4: one L-PID dev run falls; its last ticks would set the maxima).

    python hardware/legobot/legobot_learning_scales.py [N_DEV_SEEDS]      (default 50: about 2.5 min)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
sys.path[:0] = [str(REPO / "simulation"), str(REPO / "optimization"), str(REPO)]

import numpy as np

from config import load_config
from balance_sim import pid_equivalent_weights
from balance_sim.policy import OBSERVATIONS
from tuning import Evaluator

LEGO = load_config("legobot")
gains = np.array(LEGO.reference_pid_gains())
ev = Evaluator(LEGO.build_eval_config(validation=True), LEGO.build_search_space())
seeds = tuple(LEGO.build_robustness().dev_seeds[:N_SEEDS])


class Rec:
    """Wraps a controller; calls `grab(inner)` after every update."""

    def __init__(self, inner, grab):
        self.inner, self.grab, self.rows = inner, grab, []

    def reset(self):
        self.inner.reset()
        self.rows = []

    def goal_velocity(self, t):
        return self.inner.goal_velocity(t)

    def pitch_reference(self):
        return self.inner.pitch_reference()

    def update(self, state, dt):
        u = self.inner.update(state, dt)
        self.rows.append(self.grab(self.inner, u))
        return u


# 1. The reference PID: setpoint, pitch integral, torque; falls.
pid_rows, pid_falls = [], 0
for pitch, seed in ev.scenarios(seeds):
    r = Rec(ev.controller(gains), lambda c, u: (c.pitch_pid.setpoint, c.pitch_pid._integral, u))
    run = ev.run_controller(r, pitch, seed)
    pid_falls += run.fell
    if not run.fell:          # a fall's last ticks are not the closed loop's range
        pid_rows += r.rows
pid_rows = np.abs(np.array(pid_rows))

# 2. Its PID-equivalent linear policy, recording the raw observations it computes.
spec = LEGO.build_policy_spec("linear")
weights = pid_equivalent_weights(spec, gains)


def grab_obs(c, u):
    return list(c._last_raw) + [abs(c.last_pre_clip)]


obs_rows, l_falls = [], 0
for pitch, seed in ev.scenarios(seeds):
    c = LEGO.build_policy_controller(spec, weights)
    original = c.observation

    def observation(state, dt, c=c, original=original):
        raw = original(state, dt)
        c._last_raw = raw
        return raw

    c.observation = observation
    r = Rec(c, grab_obs)
    run = ev.run_controller(r, pitch, seed)
    l_falls += run.fell
    if not run.fell:
        obs_rows += r.rows
obs = np.abs(np.array(obs_rows))

print(f"LEGO, {len(ev.scenarios(seeds))} runs of {ev.config.duration:g} s (dev seeds {seeds[0]}-{seeds[-1]}), "
      f"nominal robot; falls: PID {pid_falls}, L-PID {l_falls} (statistics over the runs that stay up)")
print(f"{'observation':<16}{'max':>10}{'p99':>10}{'p90':>10}")
for i, name in enumerate(OBSERVATIONS):
    print(f"{name:<16}{obs[:, i].max():>10.4f}{np.quantile(obs[:, i], 0.99):>10.4f}{np.quantile(obs[:, i], 0.90):>10.4f}")
print(f"{'|pre-clip out|':<16}{obs[:, -1].max():>10.4f}{np.quantile(obs[:, -1], 0.99):>10.4f}{np.quantile(obs[:, -1], 0.90):>10.4f}")
for j, name in enumerate(("pid_setpoint", "pid_int_theta", "pid_torque")):
    print(f"{name:<16}{pid_rows[:, j].max():>10.4f}{np.quantile(pid_rows[:, j], 0.99):>10.4f}"
          f"{np.quantile(pid_rows[:, j], 0.90):>10.4f}")
c = LEGO.cost
print(f"cost limits: pitch {c.pitch_limit_rad}, velocity {c.velocity_limit}, torque {c.torque_limit}; "
      f"max_torque {LEGO.robot.max_torque}; goal peak {max(abs(s[1]) for s in LEGO.scenario.goal_steps)}")
