"""Time one CMA-ES generation of each learned policy, to fix budgets (docs/ml-baselines-plan.md 3.2, 4.2).

Scores a whole population on the robust training scenarios (50 seeds x 3 tilts plus chaos twins,
300 simulations per candidate, 20 s), the way run_neuroevolution.py does, at max_batch 200k (the
default: 666 candidates per chunk) and 400k (1,332 in one chunk). The weights balance the robot,
so every simulation runs its full length (a population that falls early would stop the batch
early and look faster than training really is):

    L         the PID-equivalent weights, plus small noise                 (population 256)
    A-size    A-warm near the PID-equivalent weights (492 parameters; A's 481 cost the same per tick)
    B1        the residual with a zero output layer: exactly the reference PID   (population 1,332)

Reports seconds per generation, microseconds per simulation, and hours for each method's full
budget (LearningConfig.budget_robot_seconds). Saved to results/benchmarks/policies_<time>.json.

From the project folder (GPU; plug the laptop in for representative numbers):
    python optimization/benchmark_policies.py
    python optimization/benchmark_policies.py --batches 200000 --repeats 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from config import CONFIG
from balance_sim import pid_equivalent_weights
from batch_tuning import BatchEvaluator
from tuning import provenance


def populations(rng):
    learning = CONFIG.learning
    gains = CONFIG.reference_pid_gains()
    linear = CONFIG.build_policy_spec("linear")
    warm = CONFIG.build_policy_spec("warm")
    residual = CONFIG.build_policy_spec("residual")
    l_pid = pid_equivalent_weights(linear, gains)
    full = dict(zip(linear.inputs, l_pid))
    warm_x = np.zeros((learning.mlp_popsize, warm.n_params))
    warm_x[:, :warm.n_inputs] = [full.get(i, 0.0) for i in warm.inputs]
    warm_x[:, warm.n_inputs:] = rng.normal(0.0, 0.01, (learning.mlp_popsize, warm.n_params - warm.n_inputs))
    res_x = rng.normal(0.0, 0.3, (learning.mlp_popsize, residual.n_params))
    res_x[:, -17:] = 0.0
    return {
        "linear": (linear, l_pid * (1 + 0.01 * rng.standard_normal((learning.linear_popsize, linear.n_params)))),
        "mlp (as warm)": (warm, warm_x),
        "residual": (residual, res_x),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, nargs="+", default=[200_000, 400_000])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    learning = CONFIG.learning
    budgets = dict(learning.budget_robot_seconds)
    budget_key = {"linear": "linear", "mlp (as warm)": "mlp", "residual": "residual"}
    config, robust, space = CONFIG.build_eval_config(), CONFIG.build_robustness(), CONFIG.build_search_space()
    rows = []
    for name, (spec, params) in populations(np.random.default_rng(0)).items():
        for max_batch in args.batches:
            ev = BatchEvaluator(config, space, dtype=np.float32, max_batch=max_batch, backend="torch",
                                device=args.device, robustness=robust, controller="policy", policy=spec)
            ev.evaluate_policies(params)                       # capture the CUDA graphs
            times = []
            for _ in range(args.repeats):
                t = time.perf_counter()
                result = ev.evaluate_policies(params)
                times.append(time.perf_counter() - t)
            per_gen = min(times)
            n_sims = len(params) * ev.simulations_per_evaluation()
            robot_s = n_sims * config.duration
            hours = budgets[budget_key[name]] / robot_s * per_gen / 3600
            rows.append({"method": name, "n_params": spec.n_params, "population": len(params),
                         "max_batch": max_batch, "chunks": -(-len(params) // ev.candidates_per_chunk()),
                         "seconds_per_generation": per_gen, "microseconds_per_simulation": 1e6 * per_gen / n_sims,
                         "robot_seconds_per_generation": robot_s, "budget_robot_seconds": budgets[budget_key[name]],
                         "hours_for_budget": hours, "falls": int(result.n_fell.sum())})
            print(f"{name:<15}{spec.n_params:>5} params  pop {len(params):>5}  max_batch {max_batch:>7}: "
                  f"{per_gen:6.2f} s/gen, {1e6 * per_gen / n_sims:5.2f} us/sim, "
                  f"{hours:5.1f} h for {budgets[budget_key[name]]:.2g} robot-s  (falls {rows[-1]['falls']})")

    out = ROOT / "results" / "benchmarks" / f"policies_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "provenance": provenance()}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
