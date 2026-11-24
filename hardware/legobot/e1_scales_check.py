"""Phase E1 Step 3 (wide latency sweep & formal dev comparison) + Post-E1 Dev Roadmap (LEARNING scales & policy preview).

Produces:
  - results/legobot/comparison/e1_staged_gains/ (via hardware/legobot/e1_stage_gains.py)
  - results/legobot/comparison/e1-dev-sweep/ (compare_controllers.py --tier dev --latency-sweep, 7.5..40 ms)
  - results/legobot/e1_learning_scales_check.json (closed-loop observation quantiles across all 50 dev seeds x 3 tilts,
    60 s nominal, comparing hand-tuned vs best E1-tuned PID against legobot_config.LEARNING)
  - results/legobot/e1_policy_dev_preview/ (linear, mlp, residual trained with run_neuroevolution.py --profile quick --float64)
  - results/legobot/comparison/e1-dev-with-policies-preview/ (dev comparison of quick preview policies vs E1-best PID)

Command lines used:
  Full Step 3 + scales check + policy preview pipeline:
    C:\\venvs\\balancebot\\Scripts\\python.exe hardware/legobot/e1_scales_check.py
  LEARNING observation scales check only (reads existing e1-dev-sweep/comparison.json):
    C:\\venvs\\balancebot\\Scripts\\python.exe hardware/legobot/e1_scales_check.py --scales-only

Runtime:
  ~47 min total on an NVIDIA RTX 5070 Ti Laptop GPU capped at 33 W
  (263.0 s CPU float64 for Step 3 dev comparison + wide latency sweep;
   ~8 s CPU for e1_learning_scales_check.json;
   2,547.6 s GPU float64 + CPU scoring for the 3 quick learned-policy previews).

NEVER touches frozen validation (3000-3099) or fall-rate (6000-6999) tiers.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
os.environ["ROBOT"] = "legobot"
sys.path[:0] = [str(REPO), str(REPO / "simulation"), str(REPO / "optimization"), str(REPO / "hardware" / "legobot")]

import numpy as np
from balance_sim import pid_equivalent_weights
from balance_sim.policy import OBSERVATIONS
from config import load_config
from e1_stage_gains import stage_gains
from power import keep_awake, power_state
from tuning import Evaluator


def measure_scales_for_gains(cfg, gains: np.ndarray, seeds):
    ev = Evaluator(cfg.build_eval_config(validation=True), cfg.build_search_space())

    class Rec:
        def __init__(self, inner, grab):
            self.inner, self.grab, self.rows = inner, grab, []

        def reset(self):
            self.inner.reset()
            self.rows = []

        def goal_velocity(self, t):
            return self.inner.goal_velocity(t)

        def pitch_reference(self):
            return self.inner.pitch_reference()

        def update(self, state, dt):
            u = self.inner.update(state, dt)
            self.rows.append(self.grab(self.inner, u))
            return u

    pid_rows, pid_falls = [], 0
    for pitch, seed in ev.scenarios(seeds):
        r = Rec(ev.controller(gains), lambda c, u: (c.pitch_pid.setpoint, c.pitch_pid._integral, u))
        run = ev.run_controller(r, pitch, seed)
        pid_falls += int(run.fell)
        pid_rows += r.rows
    pid_rows = np.abs(np.array(pid_rows))

    spec = cfg.build_policy_spec("linear")
    weights = pid_equivalent_weights(spec, gains)
    obs_rows, l_falls = [], 0
    for pitch, seed in ev.scenarios(seeds):
        c = cfg.build_policy_controller(spec, weights)
        orig = c.observation

        def observation(state, dt, c=c, orig=orig):
            raw = orig(state, dt)
            c._last_raw = raw
            return raw

        c.observation = observation
        r = Rec(c, lambda ctrl, u: list(ctrl._last_raw) + [abs(ctrl.last_pre_clip)])
        run = ev.run_controller(r, pitch, seed)
        l_falls += int(run.fell)
        obs_rows += r.rows
    obs = np.abs(np.array(obs_rows))

    stats = {"pid_falls_of_150": pid_falls, "l_pid_falls_of_150": l_falls, "observations": {}, "pid_internals": {}}
    for i, name in enumerate(OBSERVATIONS):
        stats["observations"][name] = {
            "max": float(obs[:, i].max()),
            "p99": float(np.quantile(obs[:, i], 0.99)),
            "p90": float(np.quantile(obs[:, i], 0.90)),
        }
    stats["observations"]["|pre-clip out|"] = {
        "max": float(obs[:, -1].max()),
        "p99": float(np.quantile(obs[:, -1], 0.99)),
        "p90": float(np.quantile(obs[:, -1], 0.90)),
    }
    for j, name in enumerate(("pid_setpoint", "pid_int_theta", "pid_torque")):
        stats["pid_internals"][name] = {
            "max": float(pid_rows[:, j].max()),
            "p99": float(np.quantile(pid_rows[:, j], 0.99)),
            "p90": float(np.quantile(pid_rows[:, j], 0.90)),
        }
    return stats


def _resolve_staged_source(source_str: str, stage_dir: Path) -> Path:
    direct = Path(source_str)
    if direct.exists():
        return direct
    parts = direct.parts
    return stage_dir / parts[-3] / parts[-2] / parts[-1]


def run_scales_check(cfg, sweep_out: Path, stage_dir: Path) -> tuple[str, Path, Path]:
    cmp_report = json.loads((sweep_out / "comparison.json").read_text())
    best_e1_run_lbl = min(
        (k for k in cmp_report["runs"] if k not in ("PID, hand-tuned", "PID, model-based")),
        key=lambda k: cmp_report["runs"][k]["mean_cost"],
    )
    best_src = _resolve_staged_source(cmp_report["runs"][best_e1_run_lbl]["source"], stage_dir)
    best_g_dict = json.loads(best_src.read_text())["gains"]
    space_names = cfg.build_search_space().names
    best_e1_gains = np.array([best_g_dict[n] for n in space_names], float)
    print(f"Best E1 PID on 60s dev: {best_e1_run_lbl} -> gains={best_g_dict}", flush=True)

    print("\n=== Post-E1 Step A: Checking LEARNING observation scales (hand-tuned vs E1-best PID) ===", flush=True)
    dev_seeds = tuple(cfg.build_robustness().dev_seeds)
    scales_ht = measure_scales_for_gains(cfg, np.array(cfg.controller.hand_tuned_gains, float), dev_seeds)
    scales_e1 = measure_scales_for_gains(cfg, best_e1_gains, dev_seeds)
    scales_out = REPO / "results" / "legobot" / "e1_learning_scales_check.json"
    scales_out.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_e1_label": best_e1_run_lbl,
        "best_e1_gains": best_g_dict,
        "configured_scales": {
            "speed_scale": cfg.learning.speed_scale,
            "velocity_error_scale": cfg.learning.velocity_error_scale,
            "pitch_scale": cfg.learning.pitch_scale,
            "pitch_rate_scale": cfg.learning.pitch_rate_scale,
            "int_v_scale": cfg.learning.int_v_scale,
            "int_theta_scale": cfg.learning.int_theta_scale,
            "command_scale": cfg.learning.command_scale,
            "output_scale": cfg.learning.output_scale,
            "residual_setpoint_scale": cfg.learning.residual_setpoint_scale,
            "residual_int_theta_scale": cfg.learning.residual_int_theta_scale,
        },
        "hand_tuned_closed_loop": scales_ht,
        "e1_best_closed_loop": scales_e1,
    }, indent=2))
    print(f"Saved learning scales check to {scales_out}", flush=True)
    return best_e1_run_lbl, best_src, scales_out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scales-only",
        action="store_true",
        help="run only the LEARNING observation scales check using existing e1-dev-sweep/comparison.json",
    )
    args = parser.parse_args(argv)

    cfg = load_config("legobot")
    stage_dir = REPO / "results" / "legobot" / "comparison" / "e1_staged_gains"
    sweep_out = REPO / "results" / "legobot" / "comparison" / "e1-dev-sweep"

    if args.scales_only:
        if not stage_dir.exists():
            stage_gains(stage_dir)
        run_scales_check(cfg, sweep_out, stage_dir)
        return

    stage_gains(stage_dir)
    env = os.environ.copy()
    env["ROBOT"] = "legobot"

    with tempfile.TemporaryDirectory(prefix="e1_training_") as tmp_train_root, keep_awake(display=True):
        env.setdefault("BALANCEBOT_TRAINING_DIR", tmp_train_root)

        # 1. Formal E1 Step 3: Dev comparison (60s) + Wide Latency Sweep (7.5..40 ms) in NumPy float64
        sweep_out.mkdir(parents=True, exist_ok=True)
        cmd_cmp = [
            PY, str(REPO / "optimization" / "compare_controllers.py"),
            "--tier", "dev",
            "--backend", "numpy",
            "--latency-sweep",
            "--latencies", "7.5,10,12.5,15,17.5,20,25,30,35,40",
            "--gain-runs", f"PID, DE={stage_dir.as_posix()}/de/seed*/de.json",
            "--gain-runs", f"PID, CMA-mid={stage_dir.as_posix()}/cma_mid/seed*/cma.json",
            "--gain-runs", f"PID, CMA-mid (mean)={stage_dir.as_posix()}/cma_mid_mean/seed*/cma_es_mean.json",
            "--gain-runs", f"PID, CMA-warm={stage_dir.as_posix()}/cma_warm/seed*/cma.json",
            "--gain-runs", f"PID, CMA-warm (mean)={stage_dir.as_posix()}/cma_warm_mean/seed*/cma_es_mean.json",
            "--gain-runs", f"PID, Random={stage_dir.as_posix()}/random/seed*/rs.json",
            "--gain-runs", f"PID, CMA-lat10-15={stage_dir.as_posix()}/cma_lat10_15/seed0/cma.json",
            "--out", str(sweep_out),
        ]
        print("=== Running E1 Step 3: compare_controllers.py --tier dev --latency-sweep ===", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd_cmp, cwd=str(REPO), env=env, capture_output=True, text=True)
        print(proc.stdout, flush=True)
        if proc.stderr:
            print("STDERR:", proc.stderr, flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"compare_controllers failed: {proc.returncode}")
        print(f"=== Step 3 completed in {time.perf_counter() - t0:.1f}s -> {sweep_out} ===", flush=True)

        # 2. Post-E1 Step A: Re-check LEARNING observation scales on all 50 dev seeds (60 s nominal)
        _, best_src, _ = run_scales_check(cfg, sweep_out, stage_dir)

        # 3. Post-E1 Step B: Dev-only float64 learned-controller preview (linear, mlp, residual, --profile quick --float64)
        print("\n=== Post-E1 Step B: Running dev-only float64 learned-controller quick preview (linear, mlp, residual) ===", flush=True)
        policy_preview_dir = REPO / "results" / "legobot" / "e1_policy_dev_preview"
        policy_preview_dir.mkdir(parents=True, exist_ok=True)

        policy_runs_info = []
        for method in ("linear", "mlp", "residual"):
            p_out = Path(tmp_train_root) / "legobot" / "e1_preview" / method / "seed0"
            if p_out.exists():
                shutil.rmtree(p_out)
            cmd_pol = [
                PY, str(REPO / "optimization" / "run_neuroevolution.py"),
                "--method", method,
                "--profile", "quick",
                "--float64",
                "--run-seed", "0",
                "--backend", "torch",
                "--out", str(p_out),
            ]
            t_p0 = time.perf_counter()
            print(f"  Training {method} (--profile quick --float64)...", flush=True)
            pr = subprocess.run(cmd_pol, cwd=str(REPO), env=env, capture_output=True, text=True)
            el_p = time.perf_counter() - t_p0
            print(pr.stdout[-600:] if len(pr.stdout) > 600 else pr.stdout, flush=True)
            if pr.returncode != 0:
                print("STDERR:", pr.stderr, flush=True)
                raise RuntimeError(f"run_neuroevolution {method} failed: {pr.returncode}")
            dest_p = policy_preview_dir / method / "seed0"
            dest_p.mkdir(parents=True, exist_ok=True)
            for fn in ("final.npz", "snapshot.json", "summary.json", "checkpoints.csv"):
                if (p_out / fn).exists():
                    shutil.copyfile(p_out / fn, dest_p / fn)
            policy_runs_info.append({"method": method, "elapsed_s": round(el_p, 2), "summary": json.loads((p_out / "summary.json").read_text())})

        # Score the preview policies against the E1-best PID and baselines on dev seeds (60s NumPy float64)
        pol_cmp_out = REPO / "results" / "legobot" / "comparison" / "e1-dev-with-policies-preview"
        pol_cmp_out.mkdir(parents=True, exist_ok=True)
        cmd_pol_cmp = [
            PY, str(REPO / "optimization" / "compare_controllers.py"),
            "--tier", "dev",
            "--backend", "numpy",
            "--gain-runs", f"PID, E1-best={best_src.as_posix()}",
            "--policies", f"Policy, Linear (quick)={(policy_preview_dir / 'linear' / 'seed0' / 'final.npz').as_posix()}",
            "--policies", f"Policy, MLP (quick)={(policy_preview_dir / 'mlp' / 'seed0' / 'final.npz').as_posix()}",
            "--policies", f"Policy, Residual (quick)={(policy_preview_dir / 'residual' / 'seed0' / 'final.npz').as_posix()}",
            "--out", str(pol_cmp_out),
        ]
        pr_cmp = subprocess.run(cmd_pol_cmp, cwd=str(REPO), env=env, capture_output=True, text=True)
        print(pr_cmp.stdout, flush=True)
        if pr_cmp.returncode != 0:
            print("STDERR:", pr_cmp.stderr, flush=True)
            raise RuntimeError(f"policy preview compare_controllers failed: {pr_cmp.returncode}")

        (policy_preview_dir / "preview_manifest.json").write_text(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_power": power_state(),
            "runs": policy_runs_info,
            "comparison_dir": str(pol_cmp_out.relative_to(REPO)),
        }, indent=2))
        print("\n[E1 Step 3 + Post-E1 Dev Roadmap] COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
