"""Phase E1 Step 1, out of sample: every full-budget Step 1 candidate re-scored on seeds 5050-5149 against the
untuned model-based PID (300 scenarios = 100 seeds x 3 start tilts, 60 s, robust, randomized 7.5-12.5 ms latency,
NumPy float64).

Backs E1_NOTES.md section 1 finding 1 and section 3's out-of-sample takeaway (PR #15 validation, V2). The full-budget
candidates are DE, CMA-ES best, CMA-ES (mean) and random search from the two midpoint runs, and CMA-ES best and
CMA-ES (mean) from the two model-based warm-start runs; the warm runs' 72-eval DE stubs and their repeated random
search rows are left out. Gains are the full-precision values in each run's comparison.json.

For each candidate: mean / median / worst cost, falls, the paired per-scenario median cost ratio against model-based
with a 95% bootstrap CI (e1_ablations.paired_ratio_bootstrap; < 1 favours the candidate), and the exact McNemar test
on falls. For each method with two runs (seeds 0 and 1), the same ratio over both runs' 600 pooled scenario ratios.

Produces:
  - results/legobot/comparison/e1-fresh-5050-5149-step1/per_scenario.csv
  - results/legobot/comparison/e1-fresh-5050-5149-step1/summary.json

Command line:
  C:\\venvs\\balancebot\\Scripts\\python.exe hardware/legobot/e1_fresh_step1.py

Runtime: about 1 min on CPU.

Scores seeds 5050-5149 (RobustnessSettings.dev_extended_seeds) only: never the frozen validation (3000-3099) or
fall-rate (6000-6999) sets.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["ROBOT"] = "legobot"
sys.path[:0] = [str(REPO), str(REPO / "simulation"), str(REPO / "optimization"), str(REPO / "hardware" / "legobot")]

import numpy as np
from batch_tuning import BatchEvaluator
from compare_controllers import mcnemar_exact
from config import load_config
from e1_ablations import paired_ratio_bootstrap
from tuning import provenance

RUNS = {   # E1_NOTES.md Step 1 table: folder -> (run label, entries scored)
    "20261124-032414": ("mid s0", ("differential evolution", "cma-es", "cma-es (mean)", "random search")),
    "20261124-034505": ("mid s1", ("differential evolution", "cma-es", "cma-es (mean)", "random search")),
    "20261124-040107": ("warm s0", ("cma-es", "cma-es (mean)")),
    "20261124-041714": ("warm s1", ("cma-es", "cma-es (mean)")),
}
METHOD = {"differential evolution": "DE", "cma-es": "CMA-ES best", "cma-es (mean)": "CMA-ES (mean)",
          "random search": "Random"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=REPO / "results" / "legobot" / "comparison" / "e1-fresh-5050-5149-step1")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    cfg = load_config("legobot")
    space = cfg.build_search_space()
    rob = cfg.build_robustness()
    seeds = tuple(rob.dev_extended_seeds)
    assert seeds == tuple(range(5050, 5150))
    for held in (rob.validation_seeds, rob.fall_rate_seeds, rob.retired_seeds, rob.dev_seeds, rob.train_seeds):
        assert not set(seeds) & set(held)

    controllers = {"model-based": np.asarray(cfg.model_based_gains(), float),
                   "hand-tuned": np.asarray(cfg.controller.hand_tuned_gains, float)}
    method_of = {}
    for folder, (run, entries) in RUNS.items():
        results = json.loads((REPO / "results" / "legobot" / "batch_tuning" / folder / "comparison.json")
                             .read_text())["results"]
        for entry in entries:
            label = f"{METHOD[entry]} ({run})"
            controllers[label] = np.array([results[entry]["gains"][k] for k in space.names], float)
            method_of[label] = f"{METHOD[entry]} ({run.split()[0]})"

    labels = list(controllers)
    ev = BatchEvaluator(cfg.build_eval_config(validation=True), space, robustness=rob, backend="numpy",
                        dtype=np.float64)
    res = ev.evaluate_population(np.vstack([controllers[k] for k in labels]), seeds=seeds)
    ref = labels.index("model-based")

    per_controller = {}
    for i, label in enumerate(labels):
        costs, fell = res.costs[i], res.fell[i].astype(bool)
        per_controller[label] = {
            "gains": space.as_dict(controllers[label]),
            "mean_cost": float(np.mean(costs)), "median_cost": float(np.median(costs)),
            "worst_cost": float(np.max(costs)), "falls": int(fell.sum()), "n_scenarios": int(costs.size),
            "median_ratio_vs_model_based": paired_ratio_bootstrap(costs, res.costs[ref]),
            "mcnemar_vs_model_based": mcnemar_exact(fell, res.fell[ref].astype(bool)),
        }
        s = per_controller[label]
        r = s["median_ratio_vs_model_based"]
        print(f"{label:28s} mean {s['mean_cost']:7.2f} median {s['median_cost']:6.2f} falls {s['falls']:2d}/300 "
              f"ratio {r['median_ratio']:.3f} [{r['low_95']:.3f}, {r['high_95']:.3f}] "
              f"McNemar p {s['mcnemar_vs_model_based']['p_value']:.3f}", flush=True)

    pooled = {}
    for method in sorted(set(method_of.values())):
        idx = [labels.index(lbl) for lbl, m in method_of.items() if m == method]
        if len(idx) < 2:
            continue
        a = np.concatenate([res.costs[i] for i in idx])
        b = np.concatenate([res.costs[ref]] * len(idx))
        pooled[method] = {"runs": [labels[i] for i in idx], "n_ratios": int(a.size),
                          "median_ratio_vs_model_based": paired_ratio_bootstrap(a, b)}
        r = pooled[method]["median_ratio_vs_model_based"]
        print(f"pooled {method:21s} ratio {r['median_ratio']:.3f} [{r['low_95']:.3f}, {r['high_95']:.3f}]", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    pitches = [float(p) for p in cfg.build_eval_config(validation=True).start_pitches]
    scenarios = [(seed, pitch) for seed in seeds for pitch in pitches]   # the evaluator's order, as e1_fresh_check
    assert len(scenarios) == res.costs.shape[1]
    with open(args.out / "per_scenario.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "seed", "start_pitch", "cost", "fell"])
        for i, label in enumerate(labels):
            for (seed, pitch), c, fl in zip(scenarios, res.costs[i], res.fell[i]):
                w.writerow([label, seed, pitch, float(c), int(fl)])
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(time.perf_counter() - t0, 2),
        "provenance": provenance(), "robot": "legobot", "backend": "numpy", "dtype": "float64",
        "seeds": list(seeds), "duration_s": 60.0, "latency_range_s": list(rob.latency_range),
        "start_pitches_rad": pitches, "ratio_convention": "median(candidate / model-based) per scenario; < 1 favours candidate",
        "controllers": per_controller, "pooled_two_run_methods": pooled, "config": cfg.snapshot(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {args.out} in {summary['elapsed_s']} s")


if __name__ == "__main__":
    main()
