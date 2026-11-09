"""Knock-out test: put one of L's features back the PID's way (no re-tuning), score on dev at 60 s.

Produced the table in docs/ml-baselines-plan.md 12.1. From the project folder (needs a GPU):
    python results/comparison/20261109-dev-L-seed0/l_knockouts.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "simulation"), str(REPO / "optimization"), str(REPO)]

import numpy as np
from config import CONFIG
from balance_sim import load_policy, pid_equivalent_weights, LinearController, stable_latency_range
from batch_tuning import BatchEvaluator

spec, w, _ = load_policy(REPO / "results" / "neuroevolution" / "linear" / "20261109-133801-seed0" / "final.npz")
pid_w = pid_equivalent_weights(spec, CONFIG.reference_pid_gains())
idx = {n: i for i, n in enumerate(spec.input_names)}
cmds = [idx[f"cmd_{k}"] for k in range(1, 5)]

def variant(**changes):
    x = w.copy()
    for name, value in changes.items():
        x[idx[name]] = value
    return x

def raw(x, name):   # weight in N m per raw unit, so v and goal can be compared directly
    i = idx[name]
    return spec.output_scale * x[i] / spec.scales[spec.inputs[i]]

# "Error-only velocity": the PID acts on goal - v with one gain. Give v the opposite of L's goal
# weight in raw units (keeping L's goal weight), so velocity enters only as goal - v.
error_only = w.copy()
k_goal = raw(w, "goal")
error_only[idx["v_meas"]] = -k_goal * spec.scales[spec.inputs[idx["v_meas"]]] / spec.output_scale
no_history = w.copy(); no_history[cmds] = 0.0
pid_integrators = variant(int_v=pid_w[idx["int_v"]], int_theta=pid_w[idx["int_theta"]])
pid_damping = variant(theta_dot_meas=pid_w[idx["theta_dot_meas"]])
variants = {
    "L as trained": w,
    "no command history (cmd_1-4 = 0)": no_history,
    "velocity as goal - v only": error_only,
    "PID's integrator weights": pid_integrators,
    "PID's pitch-rate weight": pid_damping,
    "PID-equivalent (all of the above)": pid_w,
}
robust = CONFIG.build_robustness()
dev = CONFIG.build_eval_config(validation=True)
space = CONFIG.build_search_space()
pid = BatchEvaluator(dev, space, robustness=robust, backend="torch").evaluate_population(
    [CONFIG.reference_pid_gains()], robust.dev_seeds)
res = BatchEvaluator(dev, space, robustness=robust, controller="policy", policy=spec, backend="torch") \
    .evaluate_policies(np.array(list(variants.values())), robust.dev_seeds)
P, T = CONFIG.build_robot_params(), CONFIG.timing.control_dt_s
print(f"{'variant':<36}{'median vs PID':>14}{'mean cost':>11}{'falls':>7}{'dU/tick':>9}{'stable to':>11}")
print(f"{'PID, ML-tuned':<36}{1.0:>14.3f}{pid.mean_cost[0]:>11.3f}{pid.n_fell[0]:>7}"
      f"{np.median(np.sqrt(pid.torque_rate_sq[0] / pid.alive_ticks[0])):>9.4f}{'37.25 ms':>11}")
for i, (name, x) in enumerate(variants.items()):
    ratio = np.median(res.costs[i] / pid.costs[0])
    rough = np.median(np.sqrt(res.torque_rate_sq[i] / np.maximum(res.alive_ticks[i], 1)))
    window = stable_latency_range(P, LinearController.linear_policy(spec, x, T), T, around=0.0)
    stable = "unstable" if window is None else f"{1000 * window[1]:.2f} ms"
    print(f"{name:<36}{ratio:>14.3f}{res.mean_cost[i]:>11.3f}{res.n_fell[i]:>7}{rough:>9.4f}{stable:>11}")
print("raw weights (N m per unit): v %.3f, goal %.3f; PID-equivalent v %.3f, goal %.3f"
      % (raw(w, "v_meas"), raw(w, "goal"), raw(pid_w, "v_meas"), raw(pid_w, "goal")))
