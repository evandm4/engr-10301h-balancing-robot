"""Compare the cascaded PID with LQR, hand-designed and ML-tuned, on one common score.

Six controllers:

    PID, hand-tuned         config.py's hand_tuned_gains
    PID, ML-tuned           the 4 gains [kp, ki, kd, kv_p], searched
    LQR, cost-matched       Q and R are the cost function's own weights, designed for the
                            configured latency. Model-based, no tuning, no simulation.
    LQR, latency ignored    the same weights, designed as if there were no latency (the
                            textbook design), run with the real latency
    LQR, ML-tuned weights   the 4 LQR weights searched; every candidate is an LQR design
    LQR, ML-tuned gains     the 5 LQR gains searched directly, starting from the
                            cost-matched design and keeping the nominal linear loop stable
                            (no integral gain: LQR.md explains why)

The common score is config.py's cost with pitch error measured from upright. The LQR
commands no lean, so the default ("commanded") would score the two kinds of controller
against different references. Every search minimizes that same score with the robust
scoring of config.py's RobustnessSettings (50 seeds x 3 tilts, randomized robots, chaos
penalty), runs both CMA-ES and differential evolution at the same budget, and keeps the
better result. The three searches get the same budget whatever their dimension.

Then, for every controller:
  * validation on 100 unseen seeds with unseen robots, in float64 with NumPy: mean cost,
    its terms, falls, and each scenario's cost relative to the ML-tuned PID's on the same
    scenario (same noise, same robot);
  * nominal linear stability: the latencies over which the sampled linear loop is stable
    (balance_sim.stability; the delay margin of PHYSICS.md 7.4);
  * a latency sweep: the controller as tuned or designed for the configured latency, run
    on validation scenarios at other latencies;
  * one example run on an unseen seed with the reference simulator, saved for
    simulation/animate.py (--csv <folder>/trajectories/<name>.csv).

From the project folder:
    python optimization/run_lqr_comparison.py --backend torch --float32     on the GPU (about 10 min)
    python optimization/run_lqr_comparison.py --quick --backend torch       tiny budgets, to check it runs
    python optimization/run_lqr_comparison.py --replot results/lqr_comparison/<run>   redraw a run's figures
NumPy works too (--backend numpy, the default) but the searches take hours.

Results go to results/lqr_comparison/<timestamp>/: comparison.json, lqr_comparison.png,
example_runs.png, each search's <method>.json and history CSV, and trajectories/.
"""

import argparse
import csv
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
from balance_sim import LinearController, spectral_radius, stable_latency_range
from batch_tuning import BatchEvaluator, cma_es_search, differential_evolution_search
from tuning import Evaluator, lqr_weights, save_example_runs, save_result

PID_HAND, PID_ML = "PID, hand-tuned", "PID, ML-tuned"
LQR_COST, LQR_NO_DELAY = "LQR, cost-matched", "LQR, latency ignored"
LQR_WEIGHTS, LQR_GAINS = "LQR, ML-tuned weights", "LQR, ML-tuned gains"
NAMES = (PID_HAND, PID_ML, LQR_COST, LQR_NO_DELAY, LQR_WEIGHTS, LQR_GAINS)
SEARCHED = (PID_ML, LQR_WEIGHTS, LQR_GAINS)

# Categorical slots 1-6 of the dataviz reference palette, in its validated order; one
# color per controller in every figure. Text stays in the ink colors.
COLORS = dict(zip(NAMES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SWEEP_LATENCIES_MS = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0)
MAX_STABLE_LATENCY = 0.08   # s, how far the linear stability search looks
SLUGS = {PID_ML: "pid_ml", LQR_WEIGHTS: "lqr_ml_weights", LQR_GAINS: "lqr_ml_gains"}   # result file names


def is_lqr(name: str) -> bool:
    return name.startswith("LQR")


def pad(gains_by_name: dict) -> np.ndarray:
    """Stack LQR gains of different widths (the latency-ignored design has no commands in
    flight) by appending zero gains, which change nothing."""
    width = max(len(g) for g in gains_by_name.values())
    return np.array([np.concatenate([g, np.zeros(width - len(g))]) for g in gains_by_name.values()])


def run_search(evaluator: BatchEvaluator, args) -> dict:
    """CMA-ES and differential evolution at the same budget; returns both, best first."""
    dim = evaluator.space.dim
    results = {
        "cma_es": cma_es_search(evaluator, max_iter=args.cma_iters, popsize=args.pop, seed=args.seed),
        # DE's population is pop_multiplier x dim; pick the multiplier that gives about the same population.
        "differential_evolution": differential_evolution_search(
            evaluator, max_iter=args.de_iters, pop_multiplier=max(1, round(args.pop / dim)), seed=args.seed),
    }
    return dict(sorted(results.items(), key=lambda kv: kv[1].best_cost))


def evaluate(check: dict, gains_by_name: dict, seeds) -> dict:
    """Score each controller on `seeds` with the matching evaluator; returns name -> BatchEvalResult (one row)."""
    out = {}
    for kind, names in (("cascaded", [n for n in gains_by_name if not is_lqr(n)]),
                        ("lqr", [n for n in gains_by_name if is_lqr(n)])):
        if not names:
            continue
        stacked = pad({n: gains_by_name[n] for n in names}) if kind == "lqr" else \
            np.array([gains_by_name[n] for n in names])
        result = check[kind].evaluate_population(stacked, seeds=seeds)
        for i, name in enumerate(names):
            out[name] = {field: (value[i] if value is not None else None)
                         for field, value in vars(result).items()}
    return out


def example_runs(cfg, gains: dict):
    """Each controller on one unseen seed with the reference simulator and the nominal robot."""
    reference = Evaluator(cfg.build_eval_config(), cfg.build_search_space())
    scenario = dict(start_pitch=cfg.scenario.start_pitches_rad[0], seed=cfg.scenario.validation_seeds[0])
    controllers = {name: cfg.build_lqr_controller(gain=g) if is_lqr(name) else cfg.build_controller(g)
                   for name, g in gains.items()}
    return {name: reference.run_controller(c, **scenario) for name, c in controllers.items()}, scenario


def comparison_cfg():
    """The project config with the common score: pitch error measured from upright."""
    return replace(CONFIG, cost=replace(CONFIG.cost, pitch_reference="upright"))


def linear_model(name: str, gains) -> LinearController:
    T = CONFIG.timing.control_dt_s
    return LinearController.lqr(gains) if is_lqr(name) else LinearController.cascaded_pid(gains, T)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cma-iters", type=int, default=40, help="CMA-ES generations")
    parser.add_argument("--de-iters", type=int, default=40, help="differential evolution generations")
    parser.add_argument("--pop", type=int, default=120, help="population per generation, for both optimizers")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--float32", action="store_true", help="search in float32 (validation is always float64)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=None, help="training noise seeds (default: config.py)")
    parser.add_argument("--sweep-seeds", type=int, default=20,
                        help="validation seeds per latency in the latency sweep (x 3 tilts)")
    parser.add_argument("--quick", action="store_true", help="tiny budgets, to check the script runs")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="save the plots without opening a window")
    parser.add_argument("--replot", metavar="FOLDER", help="redraw the figures of an earlier run, without rerunning it")
    args = parser.parse_args()
    if args.replot:
        replot(Path(args.replot), show=not args.no_show)
        return
    if args.quick:
        args.cma_iters, args.de_iters, args.pop, args.sweep_seeds = 3, 2, 16, 2
        args.seeds = args.seeds or 4

    # ---- the common score and the evaluators ----
    cfg = comparison_cfg()
    eval_config = cfg.build_eval_config()
    robustness = cfg.build_robustness()
    if args.seeds is not None:
        robustness = replace(robustness, train_seeds=tuple(range(1, args.seeds + 1)))
    if args.quick:
        robustness = replace(robustness, validation_seeds=robustness.validation_seeds[:10])
    dtype = np.float32 if args.float32 else np.float64
    search_kw = dict(backend=args.backend, dtype=dtype, robustness=robustness)
    evaluators = {
        PID_ML: BatchEvaluator(eval_config, cfg.build_search_space(), **search_kw),
        LQR_WEIGHTS: BatchEvaluator(eval_config, cfg.build_lqr_space(), controller="lqr",
                                    to_gains=cfg.build_lqr_designer(), **search_kw),
        LQR_GAINS: BatchEvaluator(eval_config, cfg.build_direct_lqr_space(), controller="lqr",
                                  to_gains=cfg.build_direct_lqr_designer(), **search_kw),
    }
    sims = evaluators[PID_ML].simulations_per_evaluation()
    print(f"Score: config.py's cost, pitch from upright. Robust scoring: {len(robustness.train_seeds)} seeds x "
          f"{len(eval_config.start_pitches)} tilts, randomized robots, chaos penalty weight "
          f"{robustness.sensitivity_weight:g} -> {sims} simulations per evaluation")

    # ---- searches ----
    searches, params, gains = {}, {}, {}
    for name, evaluator in evaluators.items():
        print(f"Tuning {name} ({evaluator.space.dim} parameters: {', '.join(evaluator.space.names)}) ...")
        searches[name] = run_search(evaluator, args)
        for method, s in searches[name].items():
            print(f"  {method:<24} best objective {s.best_cost:.4f}  ({s.n_evals} evaluations, {s.elapsed_s:.0f} s)")
        best = next(iter(searches[name].values()))
        params[name] = best.gains
        gains[name] = evaluator.gains_for([best.gains])[0]

    gains[PID_HAND] = np.array(cfg.controller.hand_tuned_gains, dtype=float)
    gains[LQR_COST] = cfg.build_lqr_gain()
    gains[LQR_NO_DELAY] = cfg.build_lqr_gain(design_latency=0.0)
    gains = {name: gains[name] for name in NAMES}

    # ---- validation, float64 NumPy, unseen seeds and robots ----
    print("Validating on unseen seeds and robots (float64, NumPy) ...")
    space_of = {"cascaded": cfg.build_search_space(), "lqr": cfg.build_lqr_space()}
    check = {kind: BatchEvaluator(eval_config, space_of[kind], robustness=robustness, controller=kind)
             for kind in ("cascaded", "lqr")}
    train = evaluate(check, gains, robustness.train_seeds)
    val = evaluate(check, gains, robustness.validation_seeds)

    # ---- linear stability ----
    params_nominal = cfg.build_robot_params()
    T, latency = cfg.timing.control_dt_s, cfg.timing.latency_s
    stability = {}
    for name, g in gains.items():
        model = linear_model(name, g)
        radius = spectral_radius(params_nominal, model, T, latency)
        window = stable_latency_range(params_nominal, model, T, around=latency, max_latency=MAX_STABLE_LATENCY)
        stability[name] = {"spectral_radius": radius, "stable_latency_range_s": window}

    # ---- latency sweep ----
    sweep_seeds = robustness.validation_seeds[:args.sweep_seeds]
    sweep_robust = replace(robustness, validation_seeds=sweep_seeds, sensitivity_weight=0.0)
    sweep_backend = dict(backend=args.backend, dtype=np.float64)
    sweep = {name: {"latency_ms": [], "mean_cost": [], "falls": []} for name in NAMES}
    latencies = (0.0, 10.0, 20.0) if args.quick else SWEEP_LATENCIES_MS
    print(f"Latency sweep: {len(latencies)} latencies x {len(sweep_seeds) * len(eval_config.start_pitches)} "
          "validation scenarios ...")
    for ms in latencies:
        config_at = replace(eval_config, latency=ms / 1000)
        check_at = {kind: BatchEvaluator(config_at, space_of[kind], robustness=sweep_robust, controller=kind,
                                         **sweep_backend) for kind in ("cascaded", "lqr")}
        scores = evaluate(check_at, gains, sweep_seeds)
        for name in NAMES:
            sweep[name]["latency_ms"].append(ms)
            sweep[name]["mean_cost"].append(float(scores[name]["mean_cost"]))
            sweep[name]["falls"].append(int(scores[name]["n_fell"]))

    # ---- table ----
    w = cfg.cost
    ref_costs = val[PID_ML]["costs"]
    comparison = {}
    print(f"\n{'controller':<24}{'train':>9}{'val cost':>10}{'falls':>9}{'worst':>8}{'vs ML PID':>11}"
          f"{'pitch':>8}{'velocity':>10}{'effort':>9}{'stable latency (ms)':>22}")
    for name in NAMES:
        v = val[name]
        n_runs = len(v["costs"])
        pitch, effort, velocity = (float(np.mean(v[k])) for k in ("pitch_error", "effort", "velocity_error"))
        ratio = v["costs"] / ref_costs
        diff = v["costs"] - ref_costs
        window = stability[name]["stable_latency_range_s"]
        entry = {
            "kind": "lqr" if is_lqr(name) else "cascaded_pid",
            "gains": [float(g) for g in gains[name]],
            "train_objective": float(train[name]["objective"]),
            "validation_cost": float(v["mean_cost"]),
            "validation_sensitivity": float(v["sensitivity"]),
            "validation_worst_cost": float(np.max(v["costs"])),
            "validation_falls": int(v["n_fell"]),
            "validation_runs": n_runs,
            "validation_costs": [float(c) for c in v["costs"]],   # per scenario, the same order for everyone
            "validation_terms": {"pitch": pitch, "effort_weighted": w.effort_weight * effort,
                                 "velocity_weighted": w.velocity_weight * velocity,
                                 "fall_penalty": float(v["mean_cost"]) - pitch - w.effort_weight * effort
                                 - w.velocity_weight * velocity},
            "vs_ml_pid": {"median_ratio": float(np.median(ratio)),
                          "mean_difference": float(diff.mean()),
                          "mean_difference_95ci": float(1.96 * diff.std(ddof=1) / np.sqrt(n_runs)),
                          "scenarios_better": int(np.sum(diff < 0))},
            "linear_stability": {"spectral_radius_at_design_latency": stability[name]["spectral_radius"],
                                 "stable_latency_range_ms": None if window is None else [1000 * x for x in window]},
            "latency_sweep": sweep[name],
        }
        if name in SEARCHED:
            space = evaluators[name].space
            entry["parameters"] = space.as_dict(params[name])
            entry["at_search_bound"] = space.at_bounds(params[name])
            entry["searches"] = {m: {"best_objective": s.best_cost, "n_evals": s.n_evals,
                                     "parameters": s.gains_dict()} for m, s in searches[name].items()}
            entry["chosen_search"] = next(iter(searches[name]))
        if name == LQR_WEIGHTS:
            entry["lqr_weights"] = asdict(lqr_weights(params[name]))
        if name in (LQR_COST, LQR_NO_DELAY):
            entry["lqr_weights"] = asdict(cfg.cost_matched_lqr_weights())
            entry["design_latency_s"] = latency if name == LQR_COST else 0.0
        comparison[name] = entry
        stable = "unstable" if window is None else f"{1000 * window[0]:.2f}-{1000 * window[1]:.2f}"
        print(f"{name:<24}{entry['train_objective']:>9.4f}{v['mean_cost']:>10.4f}{v['n_fell']:>5}/{n_runs:<3}"
              f"{np.max(v['costs']):>8.3f}{np.median(ratio):>10.3f}x{pitch:>8.4f}{w.velocity_weight * velocity:>10.4f}"
              f"{w.effort_weight * effort:>9.5f}{stable:>22}")
    print("\ntrain = what the searches minimized (mean cost + chaos penalty); val cost = mean over "
          f"{len(ref_costs)} unseen seeds x tilts with unseen robots; vs ML PID = median per-scenario cost ratio;"
          "\npitch / velocity / effort = the cost's terms (weighted); stable latency = where the nominal "
          "sampled linear loop is stable, around the configured "
          f"{1000 * latency:g} ms ({1000 * MAX_STABLE_LATENCY:g} ms = end of the search)")
    for name in SEARCHED:
        if comparison[name]["at_search_bound"]:
            print(f"Note: {name} ended at the edge of its search range for "
                  f"{', '.join(comparison[name]['at_search_bound'])}.")

    # ---- save ----
    folder = ROOT / "results" / "lqr_comparison" / time.strftime("%Y%m%d-%H%M%S")
    snapshot = CONFIG.snapshot()
    for name in SEARCHED:
        for method, s in searches[name].items():
            s.method = f"{SLUGS[name]}_{method}"
            save_result(s, folder, extra={"controller": name, "backend": args.backend,
                                          "dtype": np.dtype(dtype).name})
    (folder / "comparison.json").write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_arguments": vars(args),
        "score": "config.py cost with pitch_reference='upright' (same for every controller)",
        "simulations_per_evaluation": sims,
        "results": comparison,
        "robustness": asdict(robustness),
        "config": snapshot}, indent=2))
    runs, scenario = example_runs(cfg, gains)
    save_example_runs(folder, {n.replace(", ", "_"): r for n, r in runs.items()},
                      {n.replace(", ", "_"): gains[n] for n in NAMES}, **scenario)
    print(f"\nSaved results to {folder}")

    if not args.no_plot:
        histories = {name: {method: s.history for method, s in searches[name].items()} for name in SEARCHED}
        save_figures(folder, comparison, histories, runs, scenario, cfg, show=not args.no_show)


def replot(folder: Path, show: bool):
    """Redraw a run's figures from its comparison.json and history CSVs. The example runs
    are re-simulated (they're deterministic). A run saved before comparison.json kept the
    per-scenario validation costs gets them recomputed (float64 NumPy) and written back."""
    data = json.loads((folder / "comparison.json").read_text())
    comparison = data["results"]
    cfg = comparison_cfg()
    gains = {name: np.array(comparison[name]["gains"]) for name in NAMES}
    if any("validation_costs" not in comparison[name] for name in NAMES):
        print("Recomputing the per-scenario validation costs (float64, NumPy) ...")
        saved = data["robustness"]
        robustness = replace(cfg.build_robustness(), train_seeds=tuple(saved["train_seeds"]),
                             validation_seeds=tuple(saved["validation_seeds"]),
                             randomize=tuple(tuple(r) for r in saved["randomize"]), robot_seed=saved["robot_seed"],
                             sensitivity_weight=saved["sensitivity_weight"], sensitivity_nudge=saved["sensitivity_nudge"])
        check = {kind: BatchEvaluator(cfg.build_eval_config(), cfg.build_lqr_space() if kind == "lqr"
                                      else cfg.build_search_space(), robustness=robustness, controller=kind)
                 for kind in ("cascaded", "lqr")}
        val = evaluate(check, gains, robustness.validation_seeds)
        for name in NAMES:
            assert np.isclose(np.mean(val[name]["costs"]), comparison[name]["validation_cost"], rtol=1e-12)
            comparison[name]["validation_costs"] = [float(c) for c in val[name]["costs"]]
        (folder / "comparison.json").write_text(json.dumps(data, indent=2))
    histories = {}
    for name in SEARCHED:
        histories[name] = {}
        for method in comparison[name]["searches"]:
            with open(folder / f"{SLUGS[name]}_{method}_history.csv") as f:
                rows = list(csv.reader(f))[1:]
            histories[name][method] = [(int(n), float(c)) for n, c in rows]
    runs, scenario = example_runs(cfg, gains)
    save_figures(folder, comparison, histories, runs, scenario, cfg, show)


def save_figures(folder, comparison, histories, runs, scenario, cfg, show: bool):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see plots)")
        return
    figures = {"lqr_comparison.png": comparison_figure(comparison, histories, cfg.timing.latency_s),
               "example_runs.png": example_runs_figure(runs, cfg, scenario)}
    for filename, fig in figures.items():
        fig.savefig(folder / filename, dpi=150, facecolor="white")
        print(f"Saved plot to {folder / filename}")
    if show:
        plt.show()


# ---- figures ------------------------------------------------------------------------

def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _plain_log_ticks(axis, candidates, low, high, suffix=""):
    """Log axis ticks at round values within [low, high], written plainly ("1.5x", not "1.5 x 10^0")."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    axis.set_major_locator(FixedLocator([c for c in candidates if low <= c <= high]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}{suffix}"))


def _title(ax, text, sub):
    ax.set_title(text, loc="left", fontsize=10, color=INK, fontweight="bold", pad=16)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")


def comparison_figure(comparison, histories, latency):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))
    rows = np.arange(len(NAMES))[::-1]          # first controller at the top

    # (a) mean validation cost
    ax = axes[0, 0]
    costs = [comparison[n]["validation_cost"] for n in NAMES]
    ax.barh(rows, costs, height=0.55, color=[COLORS[n] for n in NAMES])
    for y, n, c in zip(rows, NAMES, costs):
        falls = comparison[n]["validation_falls"]
        ax.text(c + max(costs) * 0.01, y, f"{c:.3f}" + (f"  ({falls} falls)" if falls else ""),
                va="center", fontsize=8, color=INK)
    ax.set_yticks(rows, NAMES, fontsize=9, color=INK)
    ax.set_xlim(0, max(costs) * 1.18)
    ax.set_xlabel("mean cost (lower is better)")
    _style(ax, "x")
    _title(ax, "Validation cost", f"{comparison[PID_ML]['validation_runs']} unseen scenarios: new noise seeds "
                                  "and new randomized robots")

    # (b) per-scenario cost relative to the ML-tuned PID
    ax = axes[0, 1]
    others = [n for n in NAMES if n != PID_ML]
    ref = np.array(comparison[PID_ML]["validation_costs"])
    ratios = [np.array(comparison[n]["validation_costs"]) / ref for n in others]
    pos = np.arange(len(others))[::-1]
    parts = ax.boxplot(ratios, positions=pos, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                       medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK_2, lw=0.8),
                       capprops=dict(color=INK_2, lw=0.8),
                       flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_2, markeredgecolor="none",
                                       alpha=0.5))
    for patch, n in zip(parts["boxes"], others):
        patch.set_facecolor(COLORS[n])
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
    ax.axvline(1.0, color=COLORS[PID_ML], lw=1.5)
    ax.text(1.0, pos[-1] - 0.45, " ML-tuned PID = 1", color=INK_2, fontsize=8, va="top")
    for y, r in zip(pos, ratios):
        ax.text(np.max(r) * 1.04, y, f"median {np.median(r):.3f}",
                va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(pos, others, fontsize=9, color=INK)
    ax.set_ylim(pos[-1] - 0.8, pos[0] + 0.5)
    low, high = min(np.min(r) for r in ratios) * 0.95, max(np.max(r) for r in ratios) * 1.6
    ax.set_xlim(low, high)
    _plain_log_ticks(ax.xaxis, (0.5, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 5, 10, 20, 50, 100), low, high, "x")
    ax.set_xlabel("cost / ML-tuned PID's cost on the same scenario (log)")
    _style(ax, "x")
    _title(ax, "Scenario by scenario", "same noise and same robot for every controller; left of the line is better")

    # (c) latency sweep
    ax = axes[1, 0]
    for n in NAMES:
        s = comparison[n]["latency_sweep"]
        lat, cost, falls = np.array(s["latency_ms"]), np.array(s["mean_cost"]), np.array(s["falls"])
        ax.plot(lat, cost, color=COLORS[n], lw=1.6, ls="--" if is_lqr(n) else "-", label=n,
                marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8)
        if falls.any():
            ax.plot(lat[falls > 0], cost[falls > 0], ls="none", marker="X", markersize=7, color=COLORS[n],
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(1000 * latency, color=INK_2, lw=0.8)
    ax.text(1000 * latency, 1.0, " tuned / designed for", transform=ax.get_xaxis_transform(), fontsize=8,
            color=INK_2, va="top")
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    _plain_log_ticks(ax.yaxis, (0.5, 1, 2, 5, 10, 20, 50, 100, 200), low, high)
    ax.set_xlabel("actual latency (ms)")
    ax.set_ylabel("mean cost (log)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="none", marker="X", color=INK_2, markersize=7))
    labels.append("some runs fell")
    ax.legend(handles, labels, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    _style(ax)
    _title(ax, "Latency it wasn't tuned for", "same controllers, validation scenarios at other latencies "
                                               "(solid: PID, dashed: LQR)")

    # (d) search convergence
    ax = axes[1, 1]
    for n in SEARCHED:
        for method, history in histories[n].items():
            evals, best = zip(*history)
            ax.plot(evals, best, color=COLORS[n], lw=1.6 if method == "cma_es" else 1.2,
                    ls="-" if method == "cma_es" else ":",
                    label=f"{n} ({'CMA-ES' if method == 'cma_es' else 'DE'})")
    # Zoom on where the searches are: the first generation's best down to the best found.
    firsts = [h[0][1] for n in SEARCHED for h in histories[n].values()]
    low = min(h[-1][1] for n in SEARCHED for h in histories[n].values()) * 0.99
    untuned_lqr = comparison[LQR_COST]["train_objective"]
    top = max(min(max(firsts), 2 * untuned_lqr), untuned_lqr) * 1.02   # a bad first generation doesn't squash the rest
    off_chart = []
    for n in (PID_HAND, LQR_COST):
        objective = comparison[n]["train_objective"]
        if objective <= top:
            ax.axhline(objective, color=COLORS[n], lw=1.0, ls="--", label=f"{n} (untuned)")
        else:
            off_chart.append(f"{n} at {objective:.2f}")
    ax.set_ylim(low, top)
    ax.set_yscale("log")
    ax.set_xscale("log")
    _plain_log_ticks(ax.yaxis, (0.9, 0.92, 0.94, 0.96, 0.98, 1, 1.05, 1.1, 1.2, 1.5, 2), low, top)
    _plain_log_ticks(ax.xaxis, (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000),
                     *ax.get_xlim())
    ax.set_xlabel("candidates scored (log)")
    ax.set_ylabel("best training objective so far (log)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    _style(ax)
    _title(ax, "Search convergence", "each candidate scored on the robust training scenarios"
           + (f" (off the top: {'; '.join(off_chart)})" if off_chart else ""))

    fig.suptitle("Cascaded PID vs LQR, hand-designed and ML-tuned (cost with pitch measured from upright)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0, w_pad=3.0)
    return fig


def example_runs_figure(runs, cfg, scenario):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(NAMES), figsize=(17, 8), sharex=True, sharey="row")
    limit = cfg.robot.max_torque
    for j, name in enumerate(NAMES):
        r = runs[name]
        color = COLORS[name]
        axes[0, j].plot(r.t, r.velocity_goal, color=INK_2, lw=0.9, ls="--")
        axes[0, j].plot(r.t, r.states[:, 1], color=color, lw=1.4)
        axes[1, j].plot(r.t, np.degrees(r.states[:, 2]), color=color, lw=1.1)
        axes[2, j].axhline(limit, color=GRID, lw=1.0)
        axes[2, j].axhline(-limit, color=GRID, lw=1.0)
        axes[2, j].plot(r.t, r.torques, color=color, lw=0.6)
        axes[0, j].set_title(name + ("  (fell)" if r.fell else ""), fontsize=9, color=INK, loc="left")
        for i in range(3):
            _style(axes[i, j])
        axes[2, j].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("velocity (m/s)\n(dashed: goal)")
    axes[1, 0].set_ylabel("pitch (deg)")
    axes[2, 0].set_ylabel(f"torque (N m)\n(gray: +-{limit:g} limit)")
    fig.suptitle(f"One unseen scenario, nominal robot: start tilt {np.degrees(scenario['start_pitch']):.2f} deg, "
                 f"noise seed {scenario['seed']}", x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


if __name__ == "__main__":
    main()
