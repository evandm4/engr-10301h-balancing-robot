"""Phase E1 Step 2: Comprehensive Model Ablations (re-scoring + re-tuning + paired cross-evaluation on full LEGO model).

Produces:
  - results/legobot/e1_ablations/ablation_summary.json (12 variants re-scored, re-tuned with float64 GPU CMA-ES,
    picked on variant dev150_20s, and cross-evaluated on the full LEGO model on dev150_20s, dev150_60s, and dev450_20s
    with paired bootstrap CIs and exact McNemar tests)
  - results/legobot/e1_ablations/variants.json (exact per-variant config overrides, robustness settings,
    seed sets for tuning/picking/cross-evaluation, optimizer seed, and CONFIG.snapshot())

Command lines used:
  Full Step 2 ablation run:
    C:\\venvs\\balancebot\\Scripts\\python.exe hardware/legobot/e1_ablations.py
  Variant metadata export only (no re-tuning; leaves ablation_summary.json untouched):
    C:\\venvs\\balancebot\\Scripts\\python.exe hardware/legobot/e1_ablations.py --variants-only

Runtime:
  ~3,081 s (~51.4 min) for the full 12-variant GPU float64 ablation + CPU float64 cross-evaluation
  on an NVIDIA RTX 5070 Ti Laptop GPU capped at 33 W; < 1 s for --variants-only.

Strictly uses training (1-50) and dev (5000-5049, plus extended dev 5050-5149 for 450-scenario paired resolution) seeds ONLY.
Frozen validation (3000-3099) and fall-rate (6000-6999) tiers are NEVER touched.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["ROBOT"] = "legobot"
sys.path[:0] = [str(REPO), str(REPO / "simulation"), str(REPO / "optimization")]

import numpy as np
from config import load_config
from batch_tuning import BatchEvaluator, cma_es_search
from compare_controllers import clopper_pearson, mcnemar_exact
from power import ghelper_mode, keep_awake, power_state
from tuning import provenance


def paired_ratio_bootstrap(costs_a: np.ndarray, costs_ref: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired per-scenario bootstrap 95% CI on median(costs_a / costs_ref)."""
    ratios = costs_a / costs_ref
    point = float(np.median(ratios))
    rng = np.random.default_rng(seed)
    n = len(ratios)
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(ratios[idx], axis=1)
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return {"median_ratio": point, "low_95": float(lo), "high_95": float(hi)}


def build_variants(base_cfg):
    """Return ordered list of (key, description, modified ProjectConfig), preserving CRN draw positions wherever possible."""
    rb = base_cfg.robustness

    def replace_range(items, replacements: dict, drop: set = frozenset()):
        out = []
        for name, lo, hi in items:
            if name in drop:
                continue
            if name in replacements:
                lo, hi = replacements[name]
            out.append((name, lo, hi))
        return tuple(out)

    def replace_spread(items, zero_names: set):
        return tuple((name, 0.0 if name in zero_names else sp) for name, sp in items)

    variants = [
        ("full_model", "Full LEGO model (control)", base_cfg),
        (
            "no_backlash",
            "Rigid drivetrain (gearbox_backlash=0, coupling_backlash=0)",
            replace(
                base_cfg,
                robot=replace(base_cfg.robot, gearbox_backlash=0.0, coupling_backlash=0.0),
                robustness=replace(rb, randomize_range=replace_range(rb.randomize_range, {}, {"gearbox_backlash", "coupling_backlash"})),
            ),
        ),
        (
            "gearbox_play_only",
            "Unseen gearbox play only (+-1.1..1.6 deg, coupling=0)",
            replace(
                base_cfg,
                robot=replace(base_cfg.robot, coupling_backlash=0.0),
                robustness=replace(rb, randomize_range=replace_range(rb.randomize_range, {"coupling_backlash": (0.0, 0.0)})),
            ),
        ),
        (
            "coupling_play_only",
            "Downstream coupling play only (gearbox=0, coupling 0.01..3.0 deg)",
            replace(
                base_cfg,
                robot=replace(base_cfg.robot, gearbox_backlash=0.0),
                robustness=replace(rb, randomize_range=replace_range(rb.randomize_range, {"gearbox_backlash": (0.0, 0.0), "coupling_backlash": (math.radians(0.01), math.radians(3.0))})),
            ),
        ),
        (
            "wider_play_bounds",
            "Unadjusted play bounds sensitivity check (gearbox +-1.0..2.0 deg, coupling 0..4.35 deg; I1/J5)",
            replace(
                base_cfg,
                robot=replace(base_cfg.robot, gearbox_backlash=math.radians(1.5), coupling_backlash=math.radians(2.175)),
                robustness=replace(rb, max_total_backlash=None, randomize_range=replace_range(rb.randomize_range, {"gearbox_backlash": (math.radians(1.0), math.radians(2.0)), "coupling_backlash": (0.0, math.radians(4.35))})),
            ),
        ),
        (
            "ideal_pitch_sensor",
            "Measured pitch + white noise instead of IMU specific-force + complementary filter",
            replace(base_cfg, sensor=replace(base_cfg.sensor, model="noise")),
        ),
        (
            "continuous_velocity",
            "Continuous velocity + noise instead of 360-count 100 Hz 5 ms delayed encoder",
            replace(base_cfg, sensor=replace(base_cfg.sensor, encoder_counts=0)),
        ),
        (
            "averaged_slow_pwm",
            "Averaged slow-PWM (pwm_tick_s=0) instead of 10 ms pulse-density modulation",
            replace(
                base_cfg,
                actuator=replace(base_cfg.actuator, pwm_tick_s=0.0),
                robustness=replace(rb, randomize_range=replace_range(rb.randomize_range, {}, {"pwm_tick_phase"})),
            ),
        ),
        (
            "fixed_tick_phase",
            "10 ms PDM pulses with fixed tick phase (0.25) instead of uniform [0, 1) per scenario",
            replace(
                base_cfg,
                robustness=replace(rb, randomize_range=replace_range(rb.randomize_range, {"pwm_tick_phase": (0.25, 0.25)})),
            ),
        ),
        (
            "exact_feedforward",
            "Zero motor/supply mismatch on feedforward (k, R, V, eta spreads = 0)",
            replace(
                base_cfg,
                robustness=replace(rb, randomize=replace_spread(rb.randomize, {"torque_constant", "resistance", "supply_voltage", "gear_efficiency"})),
            ),
        ),
        (
            "zero_pitch_offset",
            "Zero pitch-bias spread (pitch_bias = 0 instead of +-0.015 rad)",
            replace(
                base_cfg,
                sensor=replace(base_cfg.sensor, pitch_bias=0.0),
                robustness=replace(rb, randomize_range=replace_range(rb.randomize_range, {"pitch_bias": (0.0, 0.0)})),
            ),
        ),
        (
            "wider_latency_10_15ms",
            "Wider sample-to-HAT latency (10.0-15.0 ms -> 10.0-17.5 ms loop delay; H4)",
            replace(base_cfg, robustness=replace(rb, latency_range_s=(0.010, 0.015))),
        ),
    ]
    return variants


def _config_overrides(base_cfg, var_cfg) -> dict:
    """Exact field-by-field diff of var_cfg relative to base_cfg."""
    norm = lambda v: json.loads(json.dumps(asdict(v)))
    base_d = norm(base_cfg)
    var_d = norm(var_cfg)
    overrides = {}
    for section, b_sec in base_d.items():
        v_sec = var_d.get(section)
        if isinstance(b_sec, dict) and isinstance(v_sec, dict):
            sec_diff = {k: v_sec[k] for k in v_sec if v_sec[k] != b_sec.get(k)}
            if sec_diff:
                overrides[section] = sec_diff
        elif v_sec != b_sec:
            overrides[section] = v_sec
    return overrides


def write_variants_json(base_cfg, variants, dev_seeds_150, dev_seeds_450, out_dir: Path) -> Path:
    """Record exact per-variant config overrides, robustness settings, seeds, optimizer seed, and CONFIG.snapshot()."""
    space = base_cfg.build_search_space()
    mb_gains = np.asarray(base_cfg.model_based_gains(), float)
    x0_mb = np.clip(space.encode(mb_gains), 0.0, 1.0)
    n_tilts = len(base_cfg.scenario.start_pitches_deg)

    variant_entries = {}
    for key, desc, v_cfg in variants:
        v_rob = v_cfg.build_robustness()
        v_snap = v_cfg.snapshot()
        variant_entries[key] = {
            "key": key,
            "description": desc,
            "optimizer_seed": 0,
            "config_overrides": _config_overrides(base_cfg, v_cfg),
            "scoring_differences_vs_base": base_cfg.scoring_differences(v_snap),
            "robustness": {
                "latency_range_s": list(v_rob.latency_range) if v_rob.latency_range is not None else None,
                "randomize": [[name, float(sp)] for name, sp in v_rob.randomize],
                "randomize_range": [[name, float(lo), float(hi)] for name, lo, hi in v_rob.randomize_range],
                "sensitivity_weight": float(v_rob.sensitivity_weight),
                "sensitivity_nudge_rad": float(v_rob.sensitivity_nudge),
                "robot_seed": int(v_rob.robot_seed),
                "tuning_seeds": list(v_rob.train_seeds),
                "fixed_rescore_seeds": list(dev_seeds_150),
                "picking_seeds": list(dev_seeds_150),
                "cross_evaluation_seeds": {
                    "dev150_20s": list(dev_seeds_150),
                    "dev150_60s": list(dev_seeds_150),
                    "dev450_20s": list(dev_seeds_450),
                },
            },
            "config": v_snap,
        }

    payload = {
        "provenance": provenance(),
        "robot": base_cfg.name,
        "optimizer": {
            "method": "cma_es",
            "optimizer_seed": 0,
            "max_iter": 25,
            "popsize": 96,
            "n_evals": 2400,
            "x0": "model-based",
            "x0_gains": space.as_dict(mb_gains),
            "x0_encoded": [float(x) for x in x0_mb],
            "backend": "torch",
            "dtype": "float64",
        },
        "seed_sets": {
            "tuning": {
                "seed_range": [1, base_cfg.robustness.n_train_seeds],
                "seeds": list(range(1, base_cfg.robustness.n_train_seeds + 1)),
                "n_scenarios": base_cfg.robustness.n_train_seeds * n_tilts,
                "duration_s": float(base_cfg.scenario.duration_s),
                "evaluated_on": "variant_model",
                "backend": "torch",
                "dtype": "float64",
            },
            "fixed_rescore": {
                "seed_range": [dev_seeds_150[0], dev_seeds_150[-1]],
                "seeds": list(dev_seeds_150),
                "n_scenarios": len(dev_seeds_150) * n_tilts,
                "duration_s": float(base_cfg.scenario.duration_s),
                "evaluated_on": "variant_model",
                "backend": "numpy",
                "dtype": "float64",
            },
            "picking": {
                "seed_range": [dev_seeds_150[0], dev_seeds_150[-1]],
                "seeds": list(dev_seeds_150),
                "n_scenarios": len(dev_seeds_150) * n_tilts,
                "duration_s": float(base_cfg.scenario.duration_s),
                "evaluated_on": "variant_model",
                "backend": "numpy",
                "dtype": "float64",
                "rule": "mean if mean_cost(mean_gains) <= mean_cost(best_sample_gains) else best_sample",
            },
            "cross_evaluation": {
                "dev150_20s": {
                    "seed_range": [dev_seeds_150[0], dev_seeds_150[-1]],
                    "seeds": list(dev_seeds_150),
                    "n_scenarios": len(dev_seeds_150) * n_tilts,
                    "duration_s": float(base_cfg.scenario.duration_s),
                    "evaluated_on": "full_model",
                    "backend": "numpy",
                    "dtype": "float64",
                },
                "dev150_60s": {
                    "seed_range": [dev_seeds_150[0], dev_seeds_150[-1]],
                    "seeds": list(dev_seeds_150),
                    "n_scenarios": len(dev_seeds_150) * n_tilts,
                    "duration_s": float(base_cfg.scenario.validation_duration_s),
                    "evaluated_on": "full_model",
                    "backend": "numpy",
                    "dtype": "float64",
                    "bootstrap_samples": 10000,
                    "bootstrap_seed": 0,
                },
                "dev450_20s": {
                    "seed_range": [dev_seeds_450[0], dev_seeds_450[-1]],
                    "seeds": list(dev_seeds_450),
                    "n_scenarios": len(dev_seeds_450) * n_tilts,
                    "duration_s": float(base_cfg.scenario.duration_s),
                    "evaluated_on": "full_model",
                    "backend": "numpy",
                    "dtype": "float64",
                    "bootstrap_samples": 10000,
                    "bootstrap_seed": 0,
                },
            },
        },
        "variant_order": [key for key, _, _ in variants],
        "variants": variant_entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "variants.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _resolve_step1_comparison(folder_str: str) -> Path:
    direct = Path(folder_str) / "comparison.json"
    if direct.exists():
        return direct
    return REPO / "results" / "legobot" / "batch_tuning" / Path(folder_str).name / "comparison.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variants-only",
        action="store_true",
        help="write results/legobot/e1_ablations/variants.json without re-tuning or modifying ablation_summary.json",
    )
    args = parser.parse_args(argv)

    base_cfg = load_config("legobot")
    space = base_cfg.build_search_space()
    mb_gains = np.asarray(base_cfg.model_based_gains(), float)
    ht_gains = np.asarray(base_cfg.controller.hand_tuned_gains, float)
    x0_mb = np.clip(space.encode(mb_gains), 0.0, 1.0)

    out_dir = REPO / "results" / "legobot" / "e1_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_seeds_150 = tuple(base_cfg.build_robustness().dev_seeds)        # 5000-5049 (150 scenarios)
    dev_seeds_450 = tuple(range(5000, 5150))                            # 5000-5149 (450 scenarios, strictly in dev region; frozen are 3000-3099 and 6000-6999)
    assert not (set(dev_seeds_450) & set(base_cfg.build_robustness().validation_seeds))
    assert not (set(dev_seeds_450) & set(base_cfg.build_robustness().fall_rate_seeds))
    assert not (set(dev_seeds_450) & set(base_cfg.build_robustness().retired_seeds))

    variants = build_variants(base_cfg)
    variants_path = write_variants_json(base_cfg, variants, dev_seeds_150, dev_seeds_450, out_dir)
    print(f"[E1 Step 2 Ablations] Saved variant input records to {variants_path}", flush=True)
    if args.variants_only:
        return

    # Load Step 1 best E1 gains if manifest exists
    step1_manifest_path = REPO / "results" / "legobot" / "e1_step1_manifest.json"
    e1_best_gains = None
    e1_best_label = None
    if step1_manifest_path.exists():
        manifest = json.loads(step1_manifest_path.read_text())
        best_val = math.inf
        for r in manifest.get("runs", []):
            comp_path = _resolve_step1_comparison(r["folder"])
            if comp_path.exists():
                comp = json.loads(comp_path.read_text())
                for m_name, entry in comp["results"].items():
                    if m_name in ("hand-tuned", "model-based"):
                        continue
                    v_obj = entry.get("robust_validation_objective", math.inf)
                    if v_obj < best_val:
                        best_val = v_obj
                        g_dict = entry["gains"]
                        e1_best_gains = np.array([g_dict[k] for k in space.names], float)
                        e1_best_label = f"{r['label']}:{m_name}"

    ablation_records = []

    with keep_awake(display=True) as awake_ok:
        print(f"[E1 Step 2 Ablations] keep_awake={awake_ok}, GPU={power_state()}, Step1 best={e1_best_label}", flush=True)

        # Part A & B: For each variant, re-score fixed controllers on that variant's dev_150 (20s) AND re-tune CMA-ES (25x96, x0=model-based)
        fixed_pop = [ht_gains, mb_gains] + ([e1_best_gains] if e1_best_gains is not None else [])
        fixed_names = ["hand-tuned", "model-based"] + (["step1_best"] if e1_best_gains is not None else [])
        fixed_arr = np.vstack(fixed_pop)

        for key, desc, v_cfg in variants:
            t0 = time.perf_counter()
            ec20 = v_cfg.build_eval_config(validation=False)
            rob = v_cfg.build_robustness()

            # 1. Re-score fixed controllers on THIS variant's dev_150 (20s) in NumPy float64
            ev_v_np = BatchEvaluator(ec20, space, robustness=rob, backend="numpy", dtype=np.float64)
            res_fixed = ev_v_np.evaluate_population(fixed_arr, seeds=dev_seeds_150)
            rescore_dict = {
                fname: {
                    "dev20_mean_cost": float(res_fixed.mean_cost[i]),
                    "dev20_falls": int(res_fixed.n_fell[i]),
                    "dev20_median_cost": float(np.median(res_fixed.costs[i])),
                }
                for i, fname in enumerate(fixed_names)
            }

            # 2. Re-tune CMA-ES on THIS variant's train_seeds (1-50) in GPU float64 (25 gens x 96 pop = 2,400 evals, warm-started at model-based)
            ev_v_gpu = BatchEvaluator(ec20, space, robustness=rob, backend="torch", dtype=np.float64)
            t_tune0 = time.perf_counter()
            cma_res = cma_es_search(ev_v_gpu, max_iter=25, popsize=96, x0=x0_mb, seed=0)
            tune_s = time.perf_counter() - t_tune0

            # Pick better of best_sample vs xfavorite (mean_gains) on THIS variant's own dev_150 (NumPy float64)
            both_g = np.vstack([cma_res.gains, cma_res.mean_gains])
            both_dev = ev_v_np.evaluate_population(both_g, seeds=dev_seeds_150)
            pick_idx = 1 if both_dev.mean_cost[1] <= both_dev.mean_cost[0] else 0
            picked_gains = both_g[pick_idx]
            picked_from = "mean" if pick_idx == 1 else "best_sample"

            elapsed = time.perf_counter() - t0
            rec = {
                "key": key,
                "description": desc,
                "tune_elapsed_s": round(tune_s, 2),
                "total_elapsed_s": round(elapsed, 2),
                "gpu_power": power_state(),
                "rescore_on_variant_dev150_20s": rescore_dict,
                "tuned_best_sample_gains": space.as_dict(cma_res.gains),
                "tuned_mean_gains": space.as_dict(cma_res.mean_gains),
                "picked_source": picked_from,
                "picked_gains": space.as_dict(picked_gains),
                "picked_gains_vec": [float(x) for x in picked_gains],
                "own_dev150_20s_cost": float(both_dev.mean_cost[pick_idx]),
                "own_dev150_20s_falls": int(both_dev.n_fell[pick_idx]),
            }
            ablation_records.append(rec)
            print(
                f"  [{key:22s}] {elapsed:5.1f}s | rescore(mb)={rescore_dict['model-based']['dev20_mean_cost']:6.2f} ({rescore_dict['model-based']['dev20_falls']:2d}f) "
                f"| tuned({picked_from:11s})={[round(x,3) for x in picked_gains]} own_dev={rec['own_dev150_20s_cost']:6.2f} ({rec['own_dev150_20s_falls']:2d}f)",
                flush=True,
            )

        # Part C: Evaluate ALL ablation-tuned gain sets + fixed baselines TOGETHER on the TRUE FULL REALISTIC LEGO MODEL
        # in NumPy float64 on: (1) dev_150 at 20s, (2) dev_150 at 60s, AND (3) extended dev_450 (seeds 5000-5149) at 20s!
        print("\n=== Evaluating all ablation-tuned controllers on the TRUE FULL REALISTIC LEGO MODEL (NumPy float64) ===", flush=True)
        all_labels = ["hand-tuned", "model-based"] + (["step1_best"] if e1_best_gains is not None else []) + [r["key"] for r in ablation_records]
        all_gains = np.vstack(fixed_pop + [np.array(r["picked_gains_vec"], float) for r in ablation_records])

        ev_full_20 = BatchEvaluator(base_cfg.build_eval_config(validation=False), space, robustness=base_cfg.build_robustness(), backend="numpy", dtype=np.float64)
        ev_full_60 = BatchEvaluator(base_cfg.build_eval_config(validation=True), space, robustness=base_cfg.build_robustness(), backend="numpy", dtype=np.float64)

        full_dev150_20 = ev_full_20.evaluate_population(all_gains, seeds=dev_seeds_150)
        full_dev150_60 = ev_full_60.evaluate_population(all_gains, seeds=dev_seeds_150)
        full_dev450_20 = ev_full_20.evaluate_population(all_gains, seeds=dev_seeds_450)

        ctrl_idx = all_labels.index("full_model")
        cross_eval = {}
        for i, lbl in enumerate(all_labels):
            ci_150_60 = paired_ratio_bootstrap(full_dev150_60.costs[i], full_dev150_60.costs[ctrl_idx])
            mc_150_60 = mcnemar_exact(full_dev150_60.fell[i], full_dev150_60.fell[ctrl_idx])
            ci_450_20 = paired_ratio_bootstrap(full_dev450_20.costs[i], full_dev450_20.costs[ctrl_idx])
            mc_450_20 = mcnemar_exact(full_dev450_20.fell[i], full_dev450_20.fell[ctrl_idx])
            cp_lo, cp_hi = clopper_pearson(int(full_dev450_20.n_fell[i]), 450)
            cross_eval[lbl] = {
                "gains": space.as_dict(all_gains[i]),
                "dev150_20s": {
                    "mean_cost": float(full_dev150_20.mean_cost[i]),
                    "median_cost": float(np.median(full_dev150_20.costs[i])),
                    "falls": int(full_dev150_20.n_fell[i]),
                },
                "dev150_60s": {
                    "mean_cost": float(full_dev150_60.mean_cost[i]),
                    "median_cost": float(np.median(full_dev150_60.costs[i])),
                    "falls": int(full_dev150_60.n_fell[i]),
                    "paired_ratio_vs_full_model": ci_150_60,
                    "mcnemar_vs_full_model": mc_150_60,
                },
                "dev450_20s": {
                    "mean_cost": float(full_dev450_20.mean_cost[i]),
                    "median_cost": float(np.median(full_dev450_20.costs[i])),
                    "falls": int(full_dev450_20.n_fell[i]),
                    "fall_rate": float(full_dev450_20.n_fell[i] / 450.0),
                    "fall_rate_95ci": [cp_lo, cp_hi],
                    "paired_ratio_vs_full_model": ci_450_20,
                    "mcnemar_vs_full_model": mc_450_20,
                },
            }
            print(
                f"  {lbl:22s} | dev150_60s: mean={full_dev150_60.mean_cost[i]:7.2f} med={np.median(full_dev150_60.costs[i]):6.2f} falls={full_dev150_60.n_fell[i]:2d}/150 "
                f"ratio={ci_150_60['median_ratio']:.3f} [{ci_150_60['low_95']:.3f},{ci_150_60['high_95']:.3f}] "
                f"| dev450_20s: mean={full_dev450_20.mean_cost[i]:7.2f} falls={full_dev450_20.n_fell[i]:2d}/450 (McNemar p={mc_450_20['p_value']:.4f})",
                flush=True,
            )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ghelper": ghelper_mode(),
        "step1_best_label": e1_best_label,
        "ablation_budget": {"method": "cma_es", "max_iter": 25, "popsize": 96, "n_evals": 2400, "x0": "model-based", "dtype": "float64"},
        "ablations": ablation_records,
        "cross_evaluation_on_full_lego": cross_eval,
    }
    out_json = out_dir / "ablation_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[E1 Step 2 Ablations] Saved complete results to {out_json}", flush=True)


if __name__ == "__main__":
    main()
