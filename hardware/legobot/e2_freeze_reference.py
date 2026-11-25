"""E2.3: freeze the LEGO robot's reference PID (E2_PLAN.md D3) as results/legobot/reference_pid/comparison.json.

Reads an E2.2 selection (hardware/legobot/e2_select.py: results/legobot/e2_selection/<ts>/selection.json) and
writes the adopted reference, with the hand-tuned baseline alongside, in read_pid_comparison's layout (the
run_lqr_comparison.py style: entries keyed by label, "kind": "cascaded_pid", a gains list). Each entry carries:

  * train_objective / train_falls: BatchEvaluator, NumPy float64, robust scoring on the default training seeds
    (CONFIG.build_eval_config(), CONFIG.build_robustness()), which is what the policy gates check against;
  * dev: copied from the selection (5000-5049 x 3 tilts, 60 s); confirm: the D2 slice, for the reference only;
  * linear_stability: spectral radius at the design latency and the stable latency window
    (balance_sim.stable_latency_range, around the design latency, up to 80 ms), as in the Pololu reference.

No validation (3000-3099) or fall-rate (6000-6999) seed is touched: they stay frozen until E2.8.

    ROBOT=legobot python hardware/legobot/e2_freeze_reference.py [results/legobot/e2_selection/<ts>]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ["ROBOT"] = "legobot"
sys.path[:0] = [str(REPO), str(REPO / "simulation"), str(REPO / "optimization")]

import numpy as np
from balance_sim import LinearController, spectral_radius, stable_latency_range
from batch_tuning import BatchEvaluator
from config import CONFIG
from tuning.reporting import provenance

OUT = REPO / "results" / "legobot" / "reference_pid" / "comparison.json"
NAMES = ("kp", "ki", "kd", "kv_p")


def _latest_selection() -> Path:
    runs = sorted((REPO / "results" / "legobot" / "e2_selection").glob("*/selection.json"))
    if not runs:
        raise SystemExit("no E2.2 selection under results/legobot/e2_selection")
    return runs[-1].parent


def linear_stability(gains):
    T, latency, P = CONFIG.timing.control_dt_s, CONFIG.timing.latency_s, CONFIG.build_robot_params()
    model = LinearController.cascaded_pid(list(gains), T)
    window = stable_latency_range(P, model, T, around=latency, max_latency=0.08)
    return {"spectral_radius_at_design_latency": spectral_radius(P, model, T, latency),
            "stable_latency_range_ms": [1000 * x for x in window]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selection", nargs="?", type=Path, help="E2.2 selection folder (default: the latest)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    assert CONFIG.name == "legobot", CONFIG.name
    folder = (args.selection or _latest_selection()).resolve()
    sel = json.loads((folder / "selection.json").read_text())
    ref = sel["reference"]
    label = ref["label"]
    gains_of = {lbl: list(map(float, g)) for lbl, g in sel["baseline_gains"].items()}
    gains_of[label] = [float(ref["gains"][k]) for k in NAMES]
    labels = [label] + [lbl for lbl in ("PID, model-based", "PID, hand-tuned") if lbl != label]

    robust = CONFIG.build_robustness()
    ev = BatchEvaluator(CONFIG.build_eval_config(), CONFIG.build_search_space(), robustness=robust)
    t0 = time.perf_counter()
    train = ev.evaluate_population([gains_of[lbl] for lbl in labels])
    train_s = time.perf_counter() - t0

    results = {}
    for i, lbl in enumerate(labels):
        g = gains_of[lbl]
        dev = sel["dev"]["stats"][lbl]
        entry = {
            "kind": "cascaded_pid",
            "gains": g,
            "parameters": dict(zip(NAMES, g)),
            "role": "reference" if lbl == label else "baseline",
            "train_objective": float(train.objective[i]),
            "train_mean_cost": float(train.mean_cost[i]),
            "train_falls": int(train.n_fell[i]),
            "train_scenarios": int(train.costs.shape[1]),
            "dev": {k: dev[k] for k in ("objective", "mean_cost", "median_cost", "worst_cost", "falls", "scenarios",
                                         "median_ratio_vs_model_based", "ratio_95ci") if k in dev},
            "linear_stability": linear_stability(g),
        }
        if lbl == label and label in sel.get("confirm", {}).get("stats", {}):
            entry["confirm"] = sel["confirm"]["stats"][label]
        results[lbl] = entry

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(),
        "what": "E2.3 frozen reference PID for the LEGO robot (E2_PLAN.md D3)",
        "reference": label,
        "selection": {"folder": folder.relative_to(REPO).as_posix(), "rule": "D3",
                      "verdict": sel["decision"]["why"],
                      "adopted_is_model_based": sel["decision"]["adopted_is_model_based"],
                      "finalists": sel.get("finalists"), "e2_1_commits": sel["settings"]["commits"]},
        "score": {"train": "BatchEvaluator numpy float64, robust scoring, default training seeds "
                           f"{robust.train_seeds[0]}-{robust.train_seeds[-1]} ({train_s:.1f} s)",
                  "dev": "copied from the selection: seeds 5000-5049 x 3 tilts, 60 s, numpy float64"},
        "results": results,
        "config": CONFIG.snapshot(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for lbl, e in results.items():
        print(f"{lbl:20s} gains {np.round(e['gains'], 5).tolist()}  train obj {e['train_objective']:.4f} "
              f"falls {e['train_falls']}/{e['train_scenarios']}  dev obj {e['dev']['objective']:.2f} "
              f"falls {e['dev']['falls']}  rho {e['linear_stability']['spectral_radius_at_design_latency']:.5f} "
              f"window {e['linear_stability']['stable_latency_range_ms']} ms")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
