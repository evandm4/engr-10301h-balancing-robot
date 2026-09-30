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
    random_search,
    save_result,
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
    histories = [("differential evolution", de.history), ("random search", rs.history)]

    if args.bayesian:
        print(f"Running Bayesian optimization ({args.bayesian_calls} evaluations) ...")
        try:
            bo = bayesian_optimization_search(evaluator, n_calls=args.bayesian_calls, seed=args.seed)
            print(f"  done in {bo.elapsed_s:.0f} s")
            results.append(("bayesian optimization", bo.gains))
            histories.append(("bayesian optimization", bo.history))
        except ImportError as e:
            print(f"  skipped: {e}")
            bo = None
    else:
        bo = None

    comparison = {}
    print()
    print(f"{'method':<24}{'kp':>7}{'ki':>7}{'kd':>7}{'kv_p':>7}{'kv_i':>7}{'train':>10}{'validation':>12}{'falls':>7}")
    for name, gains in results:
        train = evaluator.evaluate_gains(gains)
        val = evaluator.evaluate_gains(gains, seeds=validation_seeds)
        comparison[name] = {"gains": space.as_dict(gains), "train_cost": train.mean_cost,
                            "validation_cost": val.mean_cost, "validation_falls": val.n_fell}
        g = gains
        print(f"{name:<24}{g[0]:>7.2f}{g[1]:>7.2f}{g[2]:>7.2f}{g[3]:>7.3f}{g[4]:>7.3f}"
              f"{train.mean_cost:>10.4f}{val.mean_cost:>12.4f}{val.n_fell:>4}/{len(validation_seeds)}")

    # Warn if the winning method's gains landed on a search bound: the true optimum may lie outside.
    best_name, best_gains = min(results[1:], key=lambda nc: comparison[nc[0]]["train_cost"])
    unit = space.encode(best_gains)
    at_bound = [n for n, u in zip(space.names, unit) if u < 0.01 or u > 0.99]
    if at_bound:
        print(f"\nNote: {best_name}'s result is at the edge of the search range for {', '.join(at_bound)}."
              " Consider widening that range in config.py.")

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
        "results": comparison,
        "config": config_snapshot}, indent=2))
    print(f"\nSaved results to {folder}")

    if args.no_plot:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see plots)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    for name, history in histories:
        ax.plot(*zip(*history), label=name)
    ax.axhline(comparison["hand-tuned"]["train_cost"], color="gray", ls="--", label="hand-tuned")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best training cost so far")
    ax.set_yscale("log")
    ax.set_title("Convergence")
    ax.legend()

    scenario = dict(start_pitch=CONFIG.scenario.start_pitches_rad[0], seed=validation_seeds[0])
    runs = {"hand-tuned": evaluator.run(hand_tuned, **scenario),
            best_name: evaluator.run(best_gains, **scenario)}
    for name, r in runs.items():
        axes[0, 1].plot(r.t, r.states[:, 1], label=name)
        axes[1, 0].plot(r.t, np.degrees(r.states[:, 2]), label=name)
        axes[1, 1].plot(r.t, r.torques, label=name)
    first = next(iter(runs.values()))
    axes[0, 1].plot(first.t, first.velocity_goal, "k--", label="goal")
    axes[0, 1].set_ylabel("velocity (m/s)")
    axes[0, 1].set_title("Tracking on an unseen noise seed")
    axes[0, 1].legend()
    axes[1, 0].set_ylabel("pitch (deg)")
    axes[1, 1].set_ylabel("torque (N m)")
    for a in (axes[1, 0], axes[1, 1]):
        a.set_xlabel("time (s)")
    axes[0, 1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(folder / "tuning_summary.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
