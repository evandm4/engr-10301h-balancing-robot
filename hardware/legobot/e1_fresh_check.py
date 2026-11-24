"""Phase E1 fresh out-of-sample dev evaluations on seeds 5050-5149 (300 scenarios = 100 seeds x 3 start tilts).

Reproduces and saves all deterministic NumPy float64 evaluations on seeds 5050-5149 referenced in E1_NOTES.md
and A4_BASELINES.md:
  1. E1_NOTES.md section 3 ("Fresh Out-of-Sample Confirmation"): 60 s robust runs on seeds 5050-5149 under
     randomized latency (7.5-12.5 ms) and fixed latencies (7.5, 10.0, 12.5, 15.0 ms) for model-based,
     CMA-mid / seed0, CMA-warm (mean) / seed1, CMA-mid (mean) / seed0, and CMA-lat10-15 / seed0.
  2. E1_NOTES.md sections 1-2: the 5050-5149 20 s slice of dev450_20s across all 15 ablation cross-evaluation
     controllers (hand-tuned, model-based, step1_best, and the 12 Step 2 variants), plus the reverse
     wider_play_bounds sensitivity check (full_model, wider_play_bounds, model-based scored under wider play).
  3. A4_BASELINES.md:520: post-H9 vs pre-H9 (before non-mutating backlash_release preceded IMU derivatives)
     hand-tuned and model-based 20 s baselines on seeds 5050-5149.

Produces:
  - results/legobot/comparison/e1-fresh-5050-5149/per_scenario.csv
  - results/legobot/comparison/e1-fresh-5050-5149/summary.json

Command line used:
  C:\\venvs\\balancebot\\Scripts\\python.exe hardware/legobot/e1_fresh_check.py

Runtime:
  ~104 s on CPU (deterministic NumPy float64).

Strictly evaluates dev seeds (5050-5149, plus 5000-5049 only for the 450-scenario wider_play_bounds aggregate check).
Frozen validation (3000-3099) and fall-rate (6000-6999) seed sets are NEVER touched.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["ROBOT"] = "legobot"
sys.path[:0] = [str(REPO), str(REPO / "simulation"), str(REPO / "optimization"), str(REPO / "hardware" / "legobot")]

import numpy as np
import batch_sim.simulate as sim_mod
import batch_tuning.evaluation as eval_mod
from batch_tuning import BatchEvaluator
from config import load_config
from e1_ablations import build_variants
from tuning import provenance

FRESH_SEEDS = tuple(range(5050, 5150))
DEV150_SEEDS = tuple(range(5000, 5050))
FIXED_LATENCIES_MS = (7.5, 10.0, 12.5, 15.0)
H9_LINE = "sense_contact = backlash_release(robot, state, applied, contact, np, force) if backlash else None"
PRE_H9_LINE = "sense_contact = contact if backlash else None"


def _step1_gains(folder_name: str, entry_name: str, space_names: tuple[str, ...]) -> np.ndarray:
    comp_path = REPO / "results" / "legobot" / "batch_tuning" / folder_name / "comparison.json"
    g_dict = json.loads(comp_path.read_text())["results"][entry_name]["gains"]
    return np.array([g_dict[k] for k in space_names], float)


def _summarize_row(result, idx: int) -> dict:
    costs = result.costs[idx]
    return {
        "mean_cost": float(result.mean_cost[idx]),
        "median_cost": float(np.median(costs)),
        "worst_cost": float(np.max(costs)),
        "falls": int(result.n_fell[idx]),
        "n_scenarios": int(costs.shape[0]),
    }


def _append_csv_rows(rows: list, suite: str, labels: list[str], duration_s: float, latency_mode: str,
                     seeds: tuple[int, ...], start_pitches: tuple[float, ...], result) -> None:
    scenarios = [(seed, pitch) for seed in seeds for pitch in start_pitches]
    for i, label in enumerate(labels):
        for (seed, pitch), c, fell, ft in zip(scenarios, result.costs[i], result.fell[i], result.fall_time[i]):
            rows.append([
                suite,
                label,
                duration_s,
                latency_mode,
                seed,
                pitch,
                float(c),
                int(fell),
                "" if np.isnan(ft) else float(ft),
            ])


def _evaluate_pre_h9(ec20, space, rob, gains_arr: np.ndarray, seeds: tuple[int, ...]):
    """Temporarily compile batch_sim.simulate with pre-H9 contact sensing, restoring the module afterwards."""
    src = Path(sim_mod.__file__).read_text(encoding="utf-8")
    if H9_LINE not in src:
        raise RuntimeError(f"Expected H9 line not found in {sim_mod.__file__}")
    pre_src = src.replace(H9_LINE, PRE_H9_LINE, 1)
    ns = dict(sim_mod.__dict__)
    exec(compile(pre_src, sim_mod.__file__, "exec"), ns)
    orig_sim_batch = eval_mod.simulate_batch
    try:
        eval_mod.simulate_batch = ns["simulate_batch"]
        ev = BatchEvaluator(ec20, space, robustness=rob, backend="numpy", dtype=np.float64)
        return ev.evaluate_population(gains_arr, seeds=seeds)
    finally:
        eval_mod.simulate_batch = orig_sim_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results" / "legobot" / "comparison" / "e1-fresh-5050-5149",
        help="output directory for per_scenario.csv and summary.json",
    )
    parser.add_argument(
        "--verify-npy",
        type=Path,
        default=None,
        help="optional path to saved r5_fresh_costs.npy (3x300) to verify exact float64 equality",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    cfg = load_config("legobot")
    space = cfg.build_search_space()
    rob = cfg.build_robustness()

    # Enforce strict seed-tier hygiene: never touch frozen validation (3000-3099) or fall-rate (6000-6999)
    for checked_seeds in (FRESH_SEEDS, DEV150_SEEDS):
        assert not (set(checked_seeds) & set(rob.validation_seeds))
        assert not (set(checked_seeds) & set(rob.fall_rate_seeds))
        assert not (set(checked_seeds) & set(rob.retired_seeds))

    abl_summary = json.loads((REPO / "results" / "legobot" / "e1_ablations" / "ablation_summary.json").read_text())
    cross_eval_ref = abl_summary["cross_evaluation_on_full_lego"]

    ec20 = cfg.build_eval_config(validation=False)
    ec60 = cfg.build_eval_config(validation=True)
    start_pitches = tuple(float(p) for p in ec20.start_pitches)

    all_gains_catalog = {}
    csv_rows = []

    # 1. E1_NOTES.md Section 3: 60 s Fresh Out-of-Sample Confirmation (randomized + fixed latencies)
    fresh60_controllers = {
        "PID, model-based": np.asarray(cfg.model_based_gains(), float),
        "PID, CMA-mid / seed0": _step1_gains("20261124-032414", "cma-es", space.names),
        "PID, CMA-warm (mean) / seed1": _step1_gains("20261124-041714", "cma-es (mean)", space.names),
        "PID, CMA-mid (mean) / seed0": _step1_gains("20261124-032414", "cma-es (mean)", space.names),
        "PID, CMA-lat10-15 / seed0": np.array(
            [cross_eval_ref["wider_latency_10_15ms"]["gains"][k] for k in space.names], float
        ),
    }
    for k, g in fresh60_controllers.items():
        all_gains_catalog[k] = space.as_dict(g)

    fresh60_labels = list(fresh60_controllers.keys())
    fresh60_arr = np.vstack([fresh60_controllers[k] for k in fresh60_labels])

    print("=== 1. E1_NOTES.md §3: 60 s Fresh Out-of-Sample Confirmation (seeds 5050-5149) ===", flush=True)
    ev60_rand = BatchEvaluator(ec60, space, robustness=rob, backend="numpy", dtype=np.float64)
    res60_rand = ev60_rand.evaluate_population(fresh60_arr, seeds=FRESH_SEEDS)
    _append_csv_rows(csv_rows, "fresh_60s", fresh60_labels, 60.0, "randomized_7.5_12.5", FRESH_SEEDS, start_pitches, res60_rand)

    fresh60_summary = {"randomized_7.5_12.5ms": {}, "fixed_latency_ms": {f"{ms:.1f}": {} for ms in FIXED_LATENCIES_MS}}
    for i, lbl in enumerate(fresh60_labels):
        fresh60_summary["randomized_7.5_12.5ms"][lbl] = _summarize_row(res60_rand, i)
        s = fresh60_summary["randomized_7.5_12.5ms"][lbl]
        print(f"  [60s rand] {lbl:28s} mean={s['mean_cost']:7.2f} med={s['median_cost']:6.2f} falls={s['falls']:2d}/300", flush=True)

    fixed_rob = replace(rob, latency_range=None, sensitivity_weight=0.0)
    for ms in FIXED_LATENCIES_MS:
        ev60_fix = BatchEvaluator(replace(ec60, latency=ms / 1000.0), space, robustness=fixed_rob, backend="numpy", dtype=np.float64)
        res60_fix = ev60_fix.evaluate_population(fresh60_arr, seeds=FRESH_SEEDS)
        _append_csv_rows(csv_rows, "fresh_60s", fresh60_labels, 60.0, f"{ms:.1f}", FRESH_SEEDS, start_pitches, res60_fix)
        for i, lbl in enumerate(fresh60_labels):
            fresh60_summary["fixed_latency_ms"][f"{ms:.1f}"][lbl] = _summarize_row(res60_fix, i)
        print(
            f"  [60s {ms:4.1f}ms] "
            + " | ".join(f"{lbl}: {int(res60_fix.n_fell[i])}" for i, lbl in enumerate(fresh60_labels)),
            flush=True,
        )

    # 2. E1_NOTES.md Sections 1-2: 20 s 5050-5149 slice of dev450_20s + wider_play_bounds reverse check
    print("\n=== 2. 20 s Fresh Slice (seeds 5050-5149) of dev450_20s ===", flush=True)
    dev20_labels = list(cross_eval_ref.keys())
    dev20_arr = np.vstack([[cross_eval_ref[lbl]["gains"][k] for k in space.names] for lbl in dev20_labels])
    for lbl, g in zip(dev20_labels, dev20_arr):
        all_gains_catalog[lbl] = space.as_dict(g)

    ev20_full = BatchEvaluator(ec20, space, robustness=rob, backend="numpy", dtype=np.float64)
    res20_fresh = ev20_full.evaluate_population(dev20_arr, seeds=FRESH_SEEDS)
    _append_csv_rows(csv_rows, "dev450_20s_fresh_slice", dev20_labels, 20.0, "randomized_7.5_12.5", FRESH_SEEDS, start_pitches, res20_fresh)

    dev20_fresh_summary = {}
    for i, lbl in enumerate(dev20_labels):
        dev20_fresh_summary[lbl] = _summarize_row(res20_fresh, i)
        s = dev20_fresh_summary[lbl]
        print(f"  [20s fresh] {lbl:22s} mean={s['mean_cost']:7.2f} med={s['median_cost']:6.2f} falls={s['falls']:3d}/300", flush=True)

    # Reverse wider_play_bounds check (E1_NOTES.md §2 K3)
    wider_play_cfg = dict((k, v_cfg) for k, _, v_cfg in build_variants(cfg))["wider_play_bounds"]
    wider_play_rob = wider_play_cfg.build_robustness()
    rev_labels = ["full_model", "wider_play_bounds", "model-based"]
    rev_arr = np.vstack([[cross_eval_ref[lbl]["gains"][k] for k in space.names] for lbl in rev_labels])
    ev20_wider_play = BatchEvaluator(wider_play_cfg.build_eval_config(validation=False), space, robustness=wider_play_rob, backend="numpy", dtype=np.float64)
    res_rev_fresh = ev20_wider_play.evaluate_population(rev_arr, seeds=FRESH_SEEDS)
    res_rev_dev150 = ev20_wider_play.evaluate_population(rev_arr, seeds=DEV150_SEEDS)
    _append_csv_rows(csv_rows, "wider_play_bounds_reverse_20s", rev_labels, 20.0, "randomized_7.5_12.5", FRESH_SEEDS, start_pitches, res_rev_fresh)

    wider_play_reverse_summary = {"fresh_5050_5149_300": {}, "combined_dev450_5000_5149": {}}
    print("\n=== 2b. Reverse wider_play_bounds Check (20 s) ===", flush=True)
    for i, lbl in enumerate(rev_labels):
        wider_play_reverse_summary["fresh_5050_5149_300"][lbl] = _summarize_row(res_rev_fresh, i)
        comb_costs = np.concatenate([res_rev_dev150.costs[i], res_rev_fresh.costs[i]])
        comb_falls = int(res_rev_dev150.n_fell[i] + res_rev_fresh.n_fell[i])
        wider_play_reverse_summary["combined_dev450_5000_5149"][lbl] = {
            "mean_cost": float(np.mean(comb_costs)),
            "median_cost": float(np.median(comb_costs)),
            "worst_cost": float(np.max(comb_costs)),
            "falls": comb_falls,
            "n_scenarios": int(comb_costs.shape[0]),
        }
        c450 = wider_play_reverse_summary["combined_dev450_5000_5149"][lbl]
        print(
            f"  [wider_play 5000-5149] {lbl:20s} mean={c450['mean_cost']:6.2f} med={c450['median_cost']:6.2f} falls={c450['falls']:2d}/450 "
            f"(5050-5149: {int(res_rev_fresh.n_fell[i])}/300)",
            flush=True,
        )

    # 3. A4_BASELINES.md:520: Post-H9 vs Pre-H9 baselines on seeds 5050-5149 (20 s)
    print("\n=== 3. A4_BASELINES.md:520: Post-H9 vs Pre-H9 Baselines (20 s, seeds 5050-5149) ===", flush=True)
    base_labels = ["hand-tuned", "model-based"]
    base_arr = np.vstack([
        np.asarray(cfg.controller.hand_tuned_gains, float),
        np.asarray(cfg.model_based_gains(), float),
    ])
    res_post_h9 = ev20_full.evaluate_population(base_arr, seeds=FRESH_SEEDS)
    res_pre_h9 = _evaluate_pre_h9(ec20, space, rob, base_arr, FRESH_SEEDS)

    _append_csv_rows(
        csv_rows, "h9_attribution_post_h9_20s", base_labels, 20.0, "randomized_7.5_12.5", FRESH_SEEDS, start_pitches, res_post_h9
    )
    _append_csv_rows(
        csv_rows, "h9_attribution_pre_h9_20s", base_labels, 20.0, "randomized_7.5_12.5", FRESH_SEEDS, start_pitches, res_pre_h9
    )

    h9_summary = {"post_h9": {}, "pre_h9": {}}
    for i, lbl in enumerate(base_labels):
        h9_summary["post_h9"][lbl] = _summarize_row(res_post_h9, i)
        h9_summary["pre_h9"][lbl] = _summarize_row(res_pre_h9, i)
        print(
            f"  {lbl:12s} | post-H9: {res_post_h9.mean_cost[i]:6.2f} ({int(res_post_h9.n_fell[i])}/300) "
            f"| pre-H9: {res_pre_h9.mean_cost[i]:6.2f} ({int(res_pre_h9.n_fell[i])}/300)",
            flush=True,
        )

    # Optional verification against saved r5_fresh_costs.npy
    npy_verification = None
    if args.verify_npy is not None and args.verify_npy.exists():
        saved = np.load(args.verify_npy)
        idx_fm = dev20_labels.index("full_model")
        idx_lat = dev20_labels.index("wider_latency_10_15ms")
        warm_4dp = np.array([2.0173, 3.9543, 0.1641, 0.5388], float)
        res_warm_4dp = ev20_full.evaluate_population(warm_4dp[None, :], seeds=FRESH_SEEDS)
        npy_verification = {
            "shape": list(saved.shape),
            "full_model_max_abs_diff": float(np.max(np.abs(saved[0] - res20_fresh.costs[idx_fm]))),
            "wider_latency_10_15ms_max_abs_diff": float(np.max(np.abs(saved[1] - res20_fresh.costs[idx_lat]))),
            "warm_mean_s1_4dp_max_abs_diff": float(np.max(np.abs(saved[2] - res_warm_4dp.costs[0]))),
        }
        print(f"\n=== Verified against {args.verify_npy}: {npy_verification} ===", flush=True)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_scenario.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "controller", "duration_s", "latency_ms", "seed", "start_pitch", "cost", "fell", "fall_time_s"])
        w.writerows(csv_rows)

    summary_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "provenance": provenance(),
        "robot": cfg.name,
        "backend": "numpy",
        "dtype": "float64",
        "seeds": list(FRESH_SEEDS),
        "seed_range": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
        "n_seeds": len(FRESH_SEEDS),
        "start_pitches_rad": list(start_pitches),
        "n_scenarios_per_eval": len(FRESH_SEEDS) * len(start_pitches),
        "durations_s": {
            "fresh_60s": 60.0,
            "dev450_20s_fresh_slice": 20.0,
            "wider_play_bounds_reverse_20s": 20.0,
            "h9_attribution_20s": 20.0,
        },
        "latencies_ms": {
            "randomized_range_ms": [float(x * 1000.0) for x in rob.latency_range],
            "fixed_sweep_ms": list(FIXED_LATENCIES_MS),
        },
        "gains": all_gains_catalog,
        "fresh_60s_confirmation": fresh60_summary,
        "dev450_20s_fresh_slice": dev20_fresh_summary,
        "wider_play_bounds_reverse_20s": wider_play_reverse_summary,
        "h9_attribution_20s": h9_summary,
        "config": cfg.snapshot(),
    }
    if npy_verification is not None:
        summary_payload["saved_npy_verification"] = npy_verification

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"\n[E1 Fresh Check] Saved {len(csv_rows)} scenario rows to {csv_path} and summary to {summary_path} in {summary_payload['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
