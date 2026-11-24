"""Stage Phase E1 per-method PID gain JSON files for compare_controllers.py --gain-runs.

Produces:
  - results/legobot/comparison/e1_staged_gains/
    (de/seed{0,1}/de.json, cma_mid/seed{0,1}/cma.json, cma_mid_mean/seed{0,1}/cma_es_mean.json,
     random/seed{0,1}/rs.json, cma_warm/seed{0,1}/cma.json, cma_warm_mean/seed{0,1}/cma_es_mean.json,
     cma_lat10_15/seed0/cma.json)

Provenance guarantees:
  - Each staged file embeds the exact config (snapshot()) and robustness settings it was tuned under
    (in particular, cma_lat10_15/seed0/cma.json embeds wider_latency_10_15ms's
    latency_range_s = [0.01, 0.015] so compare_controllers.py emits its config-drift warning).
  - *_mean/seed*/cma_es_mean.json drop the single-sample best_training_cost and instead record
    mean_train_cost and mean_train_objective from the run's comparison.json["results"]["cma-es (mean)"].
  - All gain values remain bit-for-bit identical to the Step 1 and Step 2 outputs.

Command line used:
  C:\\venvs\\balancebot\\Scripts\\python.exe hardware/legobot/e1_stage_gains.py

Runtime:
  < 1 s.
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["ROBOT"] = "legobot"
sys.path[:0] = [str(REPO), str(REPO / "simulation"), str(REPO / "optimization"), str(REPO / "hardware" / "legobot")]

from batch_tuning import checking_tier
from config import load_config
from e1_ablations import build_variants
from tuning import provenance


def _resolve_step1_folder(folder_str: str) -> Path:
    direct = Path(folder_str)
    if direct.exists():
        return direct
    return REPO / "results" / "legobot" / "batch_tuning" / direct.name


def write_mean_json(cma_json_path: Path, dest_dir: Path) -> Path:
    data = json.loads(cma_json_path.read_text())
    comp = json.loads((cma_json_path.parent / "comparison.json").read_text())
    mean_entry = comp["results"]["cma-es (mean)"]

    mean_data = {
        "method": data["method"] + "_mean",
        "gains": data["mean_gains"],
        "mean_train_cost": mean_entry["train_cost"],
        "mean_train_objective": mean_entry["train_objective"],
        "n_evals": data["n_evals"],
        "elapsed_s": data["elapsed_s"],
        "provenance": data["provenance"],
        "config": data["config"],
        "robustness": data["robustness"],
        "backend": data["backend"],
        "dtype": data["dtype"],
    }
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "cma_es_mean.json"
    out.write_text(json.dumps(mean_data, indent=2))
    return out


def stage_gains(stage_dir: Path | None = None) -> Path:
    cfg = load_config("legobot")
    manifest = json.loads((REPO / "results" / "legobot" / "e1_step1_manifest.json").read_text())
    abl_summary = json.loads((REPO / "results" / "legobot" / "e1_ablations" / "ablation_summary.json").read_text())

    by_label = {r["label"]: _resolve_step1_folder(r["folder"]) for r in manifest["runs"]}
    if stage_dir is None:
        stage_dir = REPO / "results" / "legobot" / "comparison" / "e1_staged_gains"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Stage clean per-run folders so compare_controllers groups each method cleanly across seeds 0 and 1
    for seed_k, run_lbl in enumerate(("full_seed0_midpoint", "full_seed1_midpoint")):
        f = by_label[run_lbl]
        d_de = stage_dir / "de" / f"seed{seed_k}"
        d_de.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f / "differential_evolution.json", d_de / "de.json")

        d_cma = stage_dir / "cma_mid" / f"seed{seed_k}"
        d_cma.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f / "cma_es.json", d_cma / "cma.json")

        write_mean_json(f / "cma_es.json", stage_dir / "cma_mid_mean" / f"seed{seed_k}")

        d_rs = stage_dir / "random" / f"seed{seed_k}"
        d_rs.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f / "random_search.json", d_rs / "rs.json")

    for seed_k, run_lbl in enumerate(("warm_seed0_model_based", "warm_seed1_model_based")):
        f = by_label[run_lbl]
        d_w = stage_dir / "cma_warm" / f"seed{seed_k}"
        d_w.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f / "cma_es.json", d_w / "cma.json")

        write_mean_json(f / "cma_es.json", stage_dir / "cma_warm_mean" / f"seed{seed_k}")

    # Stage Step 2's wider_latency_10_15ms tuned controller with the exact config & robustness it was tuned under
    variants_by_key = {k: v_cfg for k, _, v_cfg in build_variants(cfg)}
    wider_cfg = variants_by_key["wider_latency_10_15ms"]
    wider_rob, _ = checking_tier(wider_cfg.build_robustness(), final=False)
    step1_prov = json.loads((by_label["full_seed0_midpoint"] / "cma_es.json").read_text()).get("provenance", provenance())

    lat_rec = [r for r in abl_summary["ablations"] if r["key"] == "wider_latency_10_15ms"][0]
    d_lat = stage_dir / "cma_lat10_15" / "seed0"
    d_lat.mkdir(parents=True, exist_ok=True)
    (d_lat / "cma.json").write_text(json.dumps({
        "method": "cma_lat10_15",
        "gains": lat_rec["picked_gains"],
        "picked_source": lat_rec["picked_source"],
        "own_dev150_20s_cost": lat_rec["own_dev150_20s_cost"],
        "own_dev150_20s_falls": lat_rec["own_dev150_20s_falls"],
        "n_evals": 2400,
        "optimizer_seed": 0,
        "cma_x0": "model-based",
        "elapsed_s": lat_rec["tune_elapsed_s"],
        "tuned_timestamp": abl_summary["timestamp"],
        "provenance": step1_prov,
        "config": wider_cfg.snapshot(),
        "robustness": asdict(wider_rob),
        "backend": "torch",
        "dtype": "float64",
    }, indent=2))
    return stage_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="override output directory (default: results/legobot/comparison/e1_staged_gains)")
    args = parser.parse_args(argv)
    out = stage_gains(args.out)
    print(f"[E1 Stage Gains] Staged 13 per-method gain files in {out}", flush=True)


if __name__ == "__main__":
    main()
