"""Tune the pivot robot's controller (side project; see PIVOT.md).

Searches the seven gains of the cascaded PID + pivot loop
[kp, ki, kd, kv_p, k_level, k_phi_p, k_phi_d] with differential evolution, on
the same scenario and cost as the base tuner (config.py), applied to the
combined center of mass, plus the pivot terms in pivot_config.py. The winner
is compared with:

    hand-tuned (held)    config.py's hand-tuned wheel gains, pivot held straight
    hand-tuned (level)   the same wheel gains, upper body kept vertical
    LQR                  the model-based optimal benchmark (pivot_sim.PivotLQRController)
    LQR (pivot held)     the same LQR design with the servo left at 0
    DE (pivot held)      with --compare-held: the same search over the four wheel
                         gains only, pivot held straight, at the same budget.
The gaps "LQR vs LQR (pivot held)" and "DE vs DE (pivot held)" are what the
pivot is worth to each kind of controller.

Two costs are reported. "train"/"validation" are the tuner's cost, whose pitch
term follows config.py (by default measured from the lean the controller
commands). The LQR commands no lean, so for comparing across controller types
"upright" re-scores the validation runs with pitch error from vertical.

From the project folder:
    python optimization/run_pivot_tuning.py --workers -1
    python optimization/run_pivot_tuning.py --workers -1 --compare-held
    python optimization/run_pivot_tuning.py --iters 5 --pop 4         quick and rough

Results go to results/pivot_tuning/<timestamp>/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dataclasses import replace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pivot_config import PIVOT_CONFIG  # noqa: E402  (imports config.py, which sets up the paths)
from pivot_sim.plotting import runs_figure  # noqa: E402
from pivot_tuning import PivotEvaluator  # noqa: E402
from tuning import SearchSpace, differential_evolution_search, gains_cells, gains_header, save_result  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=20, help="differential evolution generations")
    parser.add_argument("--pop", type=int, default=6, help="population multiplier (population = pop x 7 gains)")
    parser.add_argument("--workers", type=int, default=1, help="CPU cores for differential evolution (-1 = all)")
    parser.add_argument("--compare-held", action="store_true",
                        help="also tune the wheel gains alone with the pivot held straight, same budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    cfg = PIVOT_CONFIG
    evaluator = PivotEvaluator(cfg)
    space = evaluator.space
    validation_seeds = cfg.base.scenario.validation_seeds

    print(f"Search space: {', '.join(f'{p.name} [{p.low:.3g}, {p.high:.3g}]' for p in space.parameters)}")
    print("Running differential evolution over all 7 gains ...")
    de = differential_evolution_search(evaluator, max_iter=args.iters, pop_multiplier=args.pop,
                                       seed=args.seed, workers=args.workers)
    print(f"  done: {de.n_evals} evaluations in {de.elapsed_s:.0f} s")
    searches = {"DE (pivot active)": (de, de.gains)}

    if args.compare_held:
        held_space = SearchSpace(space.parameters[:4])
        held_evaluator = PivotEvaluator(cfg, held_space)
        # Same number of evaluations: population scaled up for the smaller space.
        pop = max(1, round(args.pop * 7 / 4))
        print("Running differential evolution over the 4 wheel gains, pivot held ...")
        held = differential_evolution_search(held_evaluator, max_iter=args.iters, pop_multiplier=pop,
                                             seed=args.seed, workers=args.workers)
        held.method = "differential_evolution_pivot_held"
        print(f"  done: {held.n_evals} evaluations in {held.elapsed_s:.0f} s")
        searches["DE (pivot held)"] = (held, np.concatenate([held.gains, np.zeros(3)]))

    controllers = {
        "hand-tuned (held)": lambda: cfg.build_controller(cfg.hand_tuned_gains()),
        "hand-tuned (level)": lambda: cfg.build_controller(cfg.level_gains()),
        "LQR": cfg.build_lqr,
        "LQR (pivot held)": lambda: cfg.build_lqr(pivot_active=False),
    }
    gains_of = {"hand-tuned (held)": np.array(cfg.hand_tuned_gains()),
                "hand-tuned (level)": np.array(cfg.level_gains())}
    for name, (_, gains) in searches.items():
        controllers[name] = (lambda g: (lambda: cfg.build_controller(g)))(gains)
        gains_of[name] = gains

    upright = PivotEvaluator(replace(cfg, base=replace(cfg.base, cost=replace(cfg.base.cost, pitch_reference="upright"))))
    n_train, n_val = len(evaluator.scenarios()), len(evaluator.scenarios(validation_seeds))
    print(f"\nEach score averages {n_train} runs (training) or {n_val} runs (validation).")
    print(f"{'controller':<22}{gains_header(cfg.gain_names)}{'train':>10}{'validation':>12}{'upright':>10}{'falls':>9}")
    comparison = {}
    for name, make in controllers.items():
        train = evaluator.evaluate_controller(make)
        val = evaluator.evaluate_controller(make, validation_seeds)
        val_upright = upright.evaluate_controller(make, validation_seeds)
        gains = gains_of.get(name)
        comparison[name] = {
            "gains": None if gains is None else dict(zip(cfg.gain_names, map(float, gains))),
            "train_cost": train.mean_cost, "validation_cost": val.mean_cost,
            "validation_cost_upright": val_upright.mean_cost,
            "validation_falls": val.n_fell, "validation_runs": val.n_runs,
        }
        if name in searches:
            search, _ = searches[name]
            comparison[name]["n_evals"] = search.n_evals
            comparison[name]["at_search_bound"] = SearchSpace(space.parameters[:len(search.gains)]).at_bounds(search.gains)
        if name.startswith("LQR"):
            comparison[name]["gain_matrix"] = make().K.tolist()
        cells = gains_cells(gains) if gains is not None else " " * len(gains_header(cfg.gain_names))
        print(f"{name:<22}{cells}{train.mean_cost:>10.4f}{val.mean_cost:>12.4f}{val_upright.mean_cost:>10.4f}"
              f"{val.n_fell:>5}/{val.n_runs}")
    print(f"\ntrain/validation: pitch error from the commanded lean ({cfg.base.cost.pitch_reference!r}); the LQR"
          " commands none, so its are from upright. upright: every controller's validation runs scored from upright.")

    folder = HERE.parent / "results" / "pivot_tuning" / time.strftime("%Y%m%d-%H%M%S")
    snapshot = cfg.snapshot()
    for search, _ in searches.values():
        save_result(search, folder, extra={"pivot_config": snapshot})
    (folder / "comparison.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_arguments": vars(args),
        "runs_per_score": {"train": n_train, "validation": n_val},
        "results": comparison,
        "pivot_config": snapshot}, indent=2))
    print(f"\nSaved results to {folder}")

    if args.no_plot:
        return
    try:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see plots)")
        return
    start_pitch, seed = cfg.base.scenario.start_pitches_rad[0], validation_seeds[0]
    runs = {name: cfg.run(make(), start_pitch, seed) for name, make in controllers.items()}
    histories = {name: search.history for name, (search, _) in searches.items()}
    fig = runs_figure(runs, histories, comparison["hand-tuned (held)"]["train_cost"],
                      title="Pivot robot: example runs on an unseen noise seed")
    fig.savefig(folder / "tuning_summary.png", dpi=150)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
