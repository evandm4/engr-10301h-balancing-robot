"""E2.4 gate (b) for the LEGO robot: the linear policy with the PID-equivalent weights (L-PID, policy
integrators) against the frozen reference PID, NumPy float64 on the CPU.

The Pololu gate (simulation/tests/test_policy.py) asserts zero falls; the LEGO reference PID falls on a few
scenarios, so this compares the cost on the scenarios where BOTH stay up (mean and median of the per-scenario
ratio, and ratio of means), and reports falls separately. Bound stated before measuring: 1%.

Tiers: training (default train seeds, robust scoring) and dev (5000-5049, 60 s). Never touches validation
(3000-3099) or fall-rate (6000-6999) seeds.

    ROBOT=legobot python hardware/legobot/e2_gate_b.py [--gains kp,ki,kd,kv_p] [--out file.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("ROBOT", "legobot")
sys.path[:0] = [str(REPO), str(REPO / "simulation"), str(REPO / "optimization")]

import numpy as np
from balance_sim import pid_equivalent_weights
from batch_tuning import BatchEvaluator
from config import CONFIG


def compare(config, seeds, gains, robust, spec):
    space = CONFIG.build_search_space()
    pid = BatchEvaluator(config, space, robustness=robust).evaluate_population([gains], seeds)
    lin = BatchEvaluator(config, space, robustness=robust, controller="policy",
                         policy=spec).evaluate_policies([pid_equivalent_weights(spec, gains)], seeds)
    up = ~pid.fell[0] & ~lin.fell[0]
    ratio = lin.costs[0, up] / pid.costs[0, up]
    return {
        "scenarios": int(pid.costs.shape[1]),
        "pid_falls": int(pid.n_fell[0]), "l_pid_falls": int(lin.n_fell[0]),
        "falls_differ_on": int((pid.fell[0] != lin.fell[0]).sum()),
        "both_up": int(up.sum()),
        "pid_objective": float(pid.objective[0]), "l_pid_objective": float(lin.objective[0]),
        "objective_ratio_minus_1": float(lin.objective[0] / pid.objective[0] - 1.0),
        "both_up_ratio_of_means_minus_1": float(lin.costs[0, up].mean() / pid.costs[0, up].mean() - 1.0),
        "both_up_mean_ratio_minus_1": float(ratio.mean() - 1.0),
        "both_up_median_ratio_minus_1": float(np.median(ratio) - 1.0),
        "both_up_max_abs_ratio_minus_1": float(np.abs(ratio - 1.0).max()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains", help="kp,ki,kd,kv_p (default: CONFIG.reference_pid_gains())")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    assert CONFIG.name == "legobot", CONFIG.name
    gains = (np.array([float(x) for x in args.gains.split(",")]) if args.gains
             else np.asarray(CONFIG.reference_pid_gains(), float))
    robust = CONFIG.build_robustness()
    spec = CONFIG.build_policy_spec("linear")
    t0 = time.perf_counter()
    report = {"gains": gains.tolist(), "reference_pid_file": CONFIG.learning.reference_pid_file,
              "reference_pid_entry": CONFIG.learning.reference_pid_entry, "bound": 0.01}
    for name, config, seeds in (("train", CONFIG.build_eval_config(), None),
                                ("dev", CONFIG.build_eval_config(validation=True), tuple(robust.dev_seeds))):
        report[name] = compare(config, seeds, gains, robust, spec)
        print(name, json.dumps(report[name], indent=1), flush=True)
    report["pass"] = all(abs(report[t]["both_up_ratio_of_means_minus_1"]) < 0.01 for t in ("train", "dev"))
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    print("PASS" if report["pass"] else "FAIL", f"({report['elapsed_s']} s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
