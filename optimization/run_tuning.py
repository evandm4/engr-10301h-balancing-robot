"""Tune the controller gains and compare search methods against hand-picked gains.

All settings, robot parameters, cost weights, delays, and search ranges come
from config.py at the project root — edit that file to change what's being
tuned for, not this script.

The robot is scored under the conditions set there (by default: DC motors,
sensor noise, 10 ms latency) while following a changing goal velocity. Every
candidate is scored on several starting tilts and noise seeds, then the
winners are re-scored on noise seeds the search never saw to check for
overfitting.

From the optimization/ folder:
    python run_tuning.py                          differential evolution + random search
    python run_tuning.py --bayesian               also run Bayesian optimization (needs
                                                    `pip install scikit-optimize`)
    python run_tuning.py --iters 10               faster, rougher differential evolution
    python run_tuning.py --workers -1              use every CPU core for differential evolution

Results (JSON, CSV history, plot) are saved under ../results/tuning/<timestamp>/.
Angles are shown in degrees; all calculations stay in radians.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "simulation"))
sys.path.insert(0, str(HERE.parent))

from config import CONFIG
from tuning import (
    Evaluator,
    bayesian_optimization_search,
    differential_evolution_search,
    gains_cells,
    gains_header,
    random_search,
    save_example_runs,
    save_result,
    tuning_summary_figure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=20, help="differential evolution generations")
    parser.add_argument("--pop", type=int, default=8, help="population multiplier (population = pop x 5 gains)")
    parser.add_argument("--workers", type=int, default=1, help="CPU cores for differential evolution (-1 = all)")
    parser.add_argument("--bayesian", action="store_true", help="also run Bayesian optimization")
    parser.add_argument("--bayesian-calls", type=int, default=150, help="evaluations for Bayesian optimization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    hand_tuned = np.array(CONFIG.controller.hand_tuned_gains)
    validation_seeds = CONFIG.scenario.validation_seeds
    space = CONFIG.build_search_space()
    evaluator = Evaluator(CONFIG.build_eval_config(), space)

    print(f"Search space: {', '.join(f'{p.name} [{p.low:.3g}, {p.high:.3g}]' for p in space.parameters)}")
    print("Running differential evolution ...")
    de = differential_evolution_search(evaluator, max_iter=args.iters, pop_multiplier=args.pop,
                                       seed=args.seed, workers=args.workers)
    print(f"  done: {de.n_evals} evaluations in {de.elapsed_s:.0f} s")

    print(f"Running random search with the same budget ({de.n_evals} evaluations) ...")
    rs = random_search(evaluator, n_evals=de.n_evals, seed=args.seed)
    print(f"  done in {rs.elapsed_s:.0f} s")

    results = [("hand-tuned", hand_tuned), ("random search", rs.gains), ("differential evolution", de.gains)]
    searches = {"random search": rs, "differential evolution": de}
    histories = [("differential evolution", de.history), ("random search", rs.history)]

    if args.bayesian:
        print(f"Running Bayesian optimization ({args.bayesian_calls} evaluations) ...")
        try:
            bo = bayesian_optimization_search(evaluator, n_calls=args.bayesian_calls, seed=args.seed)
            print(f"  done in {bo.elapsed_s:.0f} s")
            results.append(("bayesian optimization", bo.gains))
            searches["bayesian optimization"] = bo
            histories.append(("bayesian optimization", bo.history))
        except ImportError as e:
            print(f"  skipped: {e}")
            bo = None
    else:
        bo = None

    n_train_runs = len(evaluator.scenarios())
    n_val_runs = len(evaluator.scenarios(validation_seeds))
    print(f"\nEach score averages {n_train_runs} runs (training) or {n_val_runs} runs (validation);"
          f" pitch error is measured against the {CONFIG.cost.pitch_reference!r} reference.")

    comparison = {}
    print()
    print(f"{'method':<24}{gains_header(space.names)}{'evals':>7}{'train':>10}{'validation':>12}{'falls':>9}")
    for name, gains in results:
        train = evaluator.evaluate_gains(gains)
        val = evaluator.evaluate_gains(gains, seeds=validation_seeds)
        n_evals = searches[name].n_evals if name in searches else None
        comparison[name] = {"gains": space.as_dict(gains), "n_evals": n_evals,
                            "train_cost": train.mean_cost, "validation_cost": val.mean_cost,
                            "validation_falls": val.n_fell, "validation_runs": val.n_runs,
                            "at_search_bound": space.at_bounds(gains) if name in searches else []}
        print(f"{name:<24}{gains_cells(gains)}{'-' if n_evals is None else n_evals:>7}"
              f"{train.mean_cost:>10.4f}{val.mean_cost:>12.4f}{val.n_fell:>5}/{val.n_runs}")

    # Warn about any search whose gains landed on a search bound: the true optimum may lie outside.
    for name in searches:
        if comparison[name]["at_search_bound"]:
            print(f"\nNote: {name}'s result is at the edge of the search range for "
                  f"{', '.join(comparison[name]['at_search_bound'])}. Consider widening that range in config.py.")

    # The methods above don't all get the same budget (Bayesian optimization is
    # far slower per evaluation). Compare them fairly at the smallest budget.
    equal_budget = None
    common = min(s.n_evals for s in searches.values())
    if any(s.n_evals != common for s in searches.values()):
        equal_budget = {"budget_evals": common, "best_train_cost": {}}
        print(f"\nEqual-budget comparison (best training cost within {common} evaluations):")
        for name, search in searches.items():
            at = search.best_cost_at(common)
            equal_budget["best_train_cost"][name] = None if at is None else {"evals": at[0], "cost": at[1]}
            shown = "no result that early" if at is None else f"{at[1]:.4f}  (after {at[0]} evals)"
            print(f"  {name:<24}{shown}")

    best_name, best_gains = min(results[1:], key=lambda nc: comparison[nc[0]]["train_cost"])

    folder = HERE.parent / "results" / "tuning" / time.strftime("%Y%m%d-%H%M%S")
    # Every file written below carries the full config, so any one of them can
    # be traced back to exactly what produced it.
    config_snapshot = CONFIG.snapshot()
    save_result(de, folder, extra={"config": config_snapshot})
    save_result(rs, folder, extra={"config": config_snapshot})
    if bo is not None:
        save_result(bo, folder, extra={"config": config_snapshot})
    (folder / "comparison.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_arguments": vars(args),
        "budget_evals": de.n_evals,
        "runs_per_score": {"train": n_train_runs, "validation": n_val_runs},
        "results": comparison,
        "equal_budget": equal_budget,
        "config": config_snapshot}, indent=2))
    # Example runs on an unseen seed: plotted below, and saved for simulation/animate.py.
    scenario = dict(start_pitch=CONFIG.scenario.start_pitches_rad[0], seed=validation_seeds[0])
    runs = {"hand-tuned": evaluator.run(hand_tuned, **scenario),
            best_name: evaluator.run(best_gains, **scenario)}
    save_example_runs(folder, runs, {"hand-tuned": hand_tuned, best_name: best_gains}, **scenario)
    print(f"\nSaved results to {folder}")

    if args.no_plot:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see plots)")
        return

    fig = tuning_summary_figure(dict(histories), comparison["hand-tuned"]["train_cost"], runs)
    fig.savefig(folder / "tuning_summary.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
