"""Tune the controller gains with batched searches, scored for robustness.

The batched counterpart of run_tuning.py: same scenario, cost, and search
space (all from config.py), but whole populations are scored at once with
batch_sim, on the CPU (NumPy) or the GPU (PyTorch). By default every candidate
is scored the robust way set in config.py's RobustnessSettings (see
batch_tuning/robust.py):
  * on 50 noise seeds instead of 3,
  * with a randomized robot in every scenario,
  * with a penalty for chaotic gains, measured by re-running each scenario
    with the start tilt nudged by 1e-4 rad.
That's 300 simulations per candidate instead of 9, which is what the batching
is for. --plain scores exactly like run_tuning.py instead.

From the project folder:
    python optimization/run_batch_tuning.py --backend torch --float32     robust, on the GPU (about 70 s)
    python optimization/run_batch_tuning.py                               robust, NumPy on the CPU (roughly 20 min)
    python optimization/run_batch_tuning.py --plain --backend torch --float32 --de-pop 2000 --cma-pop 10000
                                                                          run_tuning.py-style scoring, huge budgets
Options: --seeds N (training seeds), --no-randomize, --sensitivity-weight W (0 = no chaos penalty).

The searches can run in float32 for speed. Every finalist is re-scored in
float64 with NumPy, both the reference way (nominal robot, the scenario's
validation seeds, so comparable with results/tuning/) and the robust way on
unseen seeds and unseen robots.

Results are saved under results/batch_tuning/<timestamp>/: the same
per-method files run_tuning.py writes, comparison.json, and the same
tuning_summary.png plot. Robust runs add robustness_summary.png. --no-show
saves the plots without opening a window; --no-plot skips them.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # config.py; importing it puts simulation/ and optimization/ on the path

import numpy as np

from config import CONFIG
from batch_tuning import (
    BatchEvaluator,
    cma_es_search,
    differential_evolution_search,
    random_search,
)
from tuning import Evaluator, gains_cells, gains_header, save_example_runs, save_result, tuning_summary_figure


def example_runs(gains_by_label):
    """Hand-tuned vs best on one unseen seed, for the summary plot and simulation/animate.py.

    They use the reference simulator and the nominal robot, so they're exactly
    what run_tuning.py's simulator produces for these gains.
    """
    reference = Evaluator(CONFIG.build_eval_config(), CONFIG.build_search_space())
    scenario = dict(start_pitch=CONFIG.scenario.start_pitches_rad[0], seed=CONFIG.scenario.validation_seeds[0])
    runs = {label: reference.run(np.asarray(g), **scenario) for label, g in gains_by_label.items()}
    return runs, scenario


def plot_summary(searches, comparison, runs, objective_label):
    """run_tuning.py's figure: convergence, plus the example runs."""
    return tuning_summary_figure(
        {name: s.history for name, s in searches.items()}, comparison["hand-tuned"]["train_objective"], runs,
        objective_label=objective_label, log_evaluations=True,   # budgets here run to the hundreds of thousands
        example_title="Tracking on an unseen noise seed (nominal robot)")


def plot_robustness(check: BatchEvaluator, validation, names):
    """Per-scenario validation costs on unseen seeds and robots: spread, and dependence on the robot."""
    import matplotlib.pyplot as plt

    robots = check.robots(check.validation_seeds)
    # M*l sets the gravity torque the controller has to fight; relative to nominal.
    gravity_factor = (robots.body_mass * robots.body_com_height) / (check.robot.body_mass * check.robot.body_com_height)
    fig, (hist, scatter) = plt.subplots(1, 2, figsize=(12, 4.5))
    costs = {name: validation[name].costs[0] for name in names}
    bins = np.geomspace(min(c.min() for c in costs.values()), max(c.max() for c in costs.values()), 40)
    for name, c in costs.items():
        hist.hist(c, bins=bins, histtype="step", lw=1.5, label=f"{name} (median {np.median(c):.3f})")
        scatter.scatter(gravity_factor, c, s=8, alpha=0.6, label=name)
    hist.set_xscale("log")
    hist.set_xlabel("cost of one validation scenario")
    hist.set_ylabel("scenarios")
    hist.set_title(f"Spread over {len(gravity_factor)} unseen seeds and robots")
    hist.legend(fontsize=8)
    scatter.set_yscale("log")
    scatter.set_xlabel("robot's body mass x COM height, relative to nominal")
    scatter.set_ylabel("cost")
    scatter.set_title("Cost vs how top-heavy the robot is")
    scatter.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--de-iters", type=int, default=30, help="differential evolution generations")
    parser.add_argument("--de-pop", type=int, default=24, help="DE population multiplier (x 5 gains)")
    parser.add_argument("--cma-iters", type=int, default=30, help="CMA-ES generations")
    parser.add_argument("--cma-pop", type=int, default=120, help="CMA-ES population")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plain", action="store_true", help="score like run_tuning.py (no robustness)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="training noise seeds in robust mode (default: config.py)")
    parser.add_argument("--no-randomize", action="store_true", help="robust mode without randomized robots")
    parser.add_argument("--sensitivity-weight", type=float, default=None,
                        help="weight of the chaos penalty (default: config.py; 0 turns it off)")
    parser.add_argument("--no-plot", action="store_true", help="skip the plots")
    parser.add_argument("--no-show", action="store_true", help="save the plots without opening a window")
    args = parser.parse_args()

    robustness = None
    if not args.plain:
        robustness = CONFIG.build_robustness()
        if args.seeds is not None:
            robustness = replace(robustness, train_seeds=tuple(range(1, args.seeds + 1)))
        if args.no_randomize:
            robustness = replace(robustness, randomize=())
        if args.sensitivity_weight is not None:
            robustness = replace(robustness, sensitivity_weight=args.sensitivity_weight)

    space = CONFIG.build_search_space()
    eval_config = CONFIG.build_eval_config()
    evaluator = BatchEvaluator(eval_config, space, backend=args.backend, robustness=robustness,
                               dtype=np.float32 if args.float32 else np.float64)
    sims_per_eval = evaluator.simulations_per_evaluation()
    if robustness:
        print(f"Robust scoring: {len(robustness.train_seeds)} seeds x {len(eval_config.start_pitches)} tilts, "
              f"{'randomized robots' if robustness.randomize else 'nominal robot'}, "
              f"chaos penalty weight {robustness.sensitivity_weight:g} "
              f"-> {sims_per_eval} simulations per evaluation")

    searches = {}
    for name, run in [
        ("differential evolution", lambda: differential_evolution_search(
            evaluator, max_iter=args.de_iters, pop_multiplier=args.de_pop, seed=args.seed)),
        ("cma-es", lambda: cma_es_search(evaluator, max_iter=args.cma_iters, popsize=args.cma_pop, seed=args.seed)),
    ]:
        print(f"Running {name} ...")
        searches[name] = run()
        s = searches[name]
        print(f"  {s.n_evals} evaluations ({s.n_evals * sims_per_eval:,} simulations) in {s.elapsed_s:.1f} s")
    budget = max(s.n_evals for s in searches.values())
    print(f"Running random search with the largest budget ({budget} evaluations) ...")
    searches["random search"] = random_search(evaluator, n_evals=budget, seed=args.seed)
    print(f"  done in {searches['random search'].elapsed_s:.1f} s")

    # Final scores always in float64 with NumPy. "reference" = exactly
    # run_tuning.py's scoring; "robust" = unseen seeds and unseen robots.
    ref_check = BatchEvaluator(eval_config, space)
    robust_check = BatchEvaluator(eval_config, space, robustness=robustness) if robustness else None
    train_check = robust_check or ref_check
    ref_val_seeds = CONFIG.scenario.validation_seeds
    candidates = {"hand-tuned": np.array(CONFIG.controller.hand_tuned_gains),
                  **{name: s.gains for name, s in searches.items()}}
    comparison, robust_validation = {}, {}
    header = f"\n{'method':<24}{gains_header(space.names)}{'evals':>8}{'train':>9}{'ref val':>9}{'falls':>7}"
    if robustness:
        header += f"{'robust val':>12}{'worst':>8}{'chaos':>9}{'falls':>9}"
    print(header)
    for name, g in candidates.items():
        train = train_check.evaluate_population([g])
        ref_val = ref_check.evaluate_population([g], seeds=ref_val_seeds)
        n_evals = searches[name].n_evals if name in searches else None
        comparison[name] = {
            "gains": space.as_dict(g), "n_evals": n_evals,
            "train_objective": float(train.objective[0]), "train_cost": float(train.mean_cost[0]),
            "reference_validation_cost": float(ref_val.mean_cost[0]),
            "reference_validation_falls": int(ref_val.n_fell[0]),
            "reference_validation_runs": ref_val.costs.shape[1],
            "at_search_bound": space.at_bounds(g) if name in searches else [],
        }
        line = (f"{name:<24}{gains_cells(g)}{'-' if n_evals is None else n_evals:>8}{train.objective[0]:>9.4f}"
                f"{ref_val.mean_cost[0]:>9.4f}{ref_val.n_fell[0]:>3}/{ref_val.costs.shape[1]:<3}")
        if robustness:
            val = robust_validation[name] = robust_check.evaluate_population([g], seeds=robustness.validation_seeds)
            comparison[name].update({
                "robust_validation_cost": float(val.mean_cost[0]),
                "robust_validation_sensitivity": float(val.sensitivity[0]),
                "robust_validation_objective": float(val.objective[0]),
                "robust_validation_worst_cost": float(val.costs[0].max()),
                "robust_validation_falls": int(val.n_fell[0]),
                "robust_validation_runs": val.costs.shape[1],
            })
            line += (f"{val.mean_cost[0]:>12.4f}{val.costs[0].max():>8.4f}{val.sensitivity[0]:>9.1e}"
                     f"{val.n_fell[0]:>5}/{val.costs.shape[1]}")
        print(line)
    print("\ntrain = what the searches minimized"
          + (" (mean cost + chaos penalty, robust scenarios)" if robustness else " (mean cost)")
          + f"; ref val = run_tuning.py's scoring on its {len(ref_val_seeds)} validation seeds")
    if robustness:
        print(f"robust val = mean cost on {len(robustness.validation_seeds)} unseen seeds with unseen robots; "
              "worst = its worst single scenario;\nchaos = mean cost change when the start tilt is nudged by "
              f"{robustness.sensitivity_nudge:.1e} rad ({np.degrees(robustness.sensitivity_nudge):.4f} deg)")
    for name, c in comparison.items():
        if c["at_search_bound"]:
            print(f"\nNote: {name}'s result is at the edge of the search range for "
                  f"{', '.join(c['at_search_bound'])}. Consider widening that range in config.py.")

    folder = ROOT / "results" / "batch_tuning" / time.strftime("%Y%m%d-%H%M%S")
    snapshot = CONFIG.snapshot()
    robust_snapshot = asdict(robustness) if robustness else None
    for s in searches.values():
        save_result(s, folder, extra={"config": snapshot, "robustness": robust_snapshot,
                                      "backend": args.backend, "dtype": np.dtype(evaluator.dtype).name})
    (folder / "comparison.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_arguments": vars(args),
        "simulations_per_evaluation": sims_per_eval,
        "results": comparison,
        "robustness": robust_snapshot,
        "config": snapshot}, indent=2))
    best_name = min(searches, key=lambda n: comparison[n]["train_objective"])
    gains = {"hand-tuned": candidates["hand-tuned"], best_name: searches[best_name].gains}
    runs, scenario = example_runs(gains)
    save_example_runs(folder, runs, gains, **scenario)   # for simulation/animate.py
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
    label = "best training objective so far" if robustness else "best training cost so far"
    figures = {"tuning_summary.png": plot_summary(searches, comparison, runs, label)}
    if robustness:
        figures["robustness_summary.png"] = plot_robustness(robust_check, robust_validation,
                                                            ["hand-tuned", best_name])
    for filename, fig in figures.items():
        fig.savefig(folder / filename, dpi=150)
        print(f"Saved plot to {folder / filename}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
