"""Phase E1 Step 1 driver: default robust float64 GPU tuning on the LEGO robot.

Produces:
  - results/legobot/e1_step1_manifest.json
  - results/legobot/batch_tuning/<timestamp>/ (4 runs: 2 midpoint seeds + 2 model-based warm-start seeds)

Command line used:
  C:\\venvs\\balancebot\\Scripts\\python.exe hardware/legobot/e1_step1.py

Runtime:
  4,428.8 s (73.8 min) across all 4 runs on an NVIDIA RTX 5070 Ti Laptop GPU capped at 33 W
  (1,255.5 s + 1,248.1 s for full DE/CMA-ES/RS midpoint seeds 0-1;
   960.4 s + 964.9 s for model-based warm-start CMA-ES seeds 0-1).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
sys.path.insert(0, str(REPO / "optimization"))
from power import ghelper_mode, keep_awake, power_state

RUNS = [
    ("full_seed0_midpoint", ["--robot", "legobot", "--backend", "torch", "--no-show", "--seed", "0", "--cma-x0", "midpoint"]),
    ("full_seed1_midpoint", ["--robot", "legobot", "--backend", "torch", "--no-show", "--seed", "1", "--cma-x0", "midpoint"]),
    ("warm_seed0_model_based", ["--robot", "legobot", "--backend", "torch", "--no-show", "--seed", "0", "--cma-x0", "model-based", "--de-iters", "2", "--de-pop", "6"]),
    ("warm_seed1_model_based", ["--robot", "legobot", "--backend", "torch", "--no-show", "--seed", "1", "--cma-x0", "model-based", "--de-iters", "2", "--de-pop", "6"]),
]


def main():
    env = os.environ.copy()
    env["ROBOT"] = "legobot"
    manifest = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "ghelper": ghelper_mode(), "runs": []}
    bt_root = REPO / "results" / "legobot" / "batch_tuning"
    bt_root.mkdir(parents=True, exist_ok=True)

    with keep_awake(display=True) as awake_ok:
        manifest["keep_awake_ok"] = awake_ok
        print(f"[E1 Step 1] keep_awake={awake_ok}, initial GPU={power_state()}", flush=True)
        for label, args in RUNS:
            before_dirs = {p.name for p in bt_root.iterdir() if p.is_dir()}
            pwr_before = power_state()
            t0 = time.perf_counter()
            cmd = [PY, str(REPO / "optimization" / "run_batch_tuning.py"), *args]
            print(f"\n=== Starting {label}: {' '.join(args)} ===", flush=True)
            proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
            elapsed = time.perf_counter() - t0
            pwr_after = power_state()
            print(proc.stdout, flush=True)
            if proc.stderr:
                print("STDERR:", proc.stderr, flush=True)
            if proc.returncode != 0:
                raise RuntimeError(f"{label} failed with exit code {proc.returncode}")
            after_dirs = sorted({p.name for p in bt_root.iterdir() if p.is_dir()} - before_dirs)
            folder = str(bt_root / after_dirs[-1]) if after_dirs else None
            manifest["runs"].append({
                "label": label,
                "args": args,
                "folder": folder,
                "elapsed_s": round(elapsed, 2),
                "power_before": pwr_before,
                "power_after": pwr_after,
            })
            (REPO / "results" / "legobot" / "e1_step1_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"=== Finished {label} in {elapsed:.1f}s -> {folder} ===", flush=True)
            time.sleep(2.0)  # ensure unique timestamp directory names

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (REPO / "results" / "legobot" / "e1_step1_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n[E1 Step 1] ALL 4 RUNS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
