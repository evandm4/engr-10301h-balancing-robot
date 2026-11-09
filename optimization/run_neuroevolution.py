"""Train a learned controller with CMA-ES: the linear policy (L), the MLP (A), the MLP warm-started
from L (A-warm), or the residual on the reference PID (B1). docs/ml-baselines-plan.md 4-5.

Every candidate is scored the robust way, exactly like the PID it is compared against
(RobustnessSettings: randomized robots and latencies, the chaos penalty). At log-spaced budgets
of simulated robot-seconds (LearningConfig.checkpoints_robot_seconds, plus the reference PID's own
search budget) the run saves both the best candidate seen and CMA's mean, and scores both in
float64 on the development tier: at 60 s (what selects policies) and at 20 s (the training-vs-dev
gap, which shows overfitting). At the end the better of the two by dev score is saved as
final.npz. Dev scores never feed back into the search itself, only into stopping and selection;
the validation and fall-rate tiers are never touched here.

From the project folder:
    python optimization/run_neuroevolution.py --method linear                    quick profile (default)
    python optimization/run_neuroevolution.py --method mlp --profile full --run-seed 3
    python optimization/run_neuroevolution.py --method warm --warm-from C:/balancebot_training/neuroevolution/linear/<run>/final.npz
    python optimization/run_neuroevolution.py --method residual --authority 0.5
    python optimization/run_neuroevolution.py --resume C:/balancebot_training/neuroevolution/mlp/<run>

One run seed per process, so a sweep maps onto a job array (scripts/slurm_neuroevolution.sh).
State is saved after every generation (state.pkl); --resume continues exactly where it stopped,
including CMA-ES's random draws. Runs on the GPU when PyTorch sees one, else NumPy on the CPU.

While it runs, Windows is asked to keep the system and the screen on (optimization/power.py): on
this laptop the screen-off idle state is when the GPU's power budget gets cut. Every generation
logs its own wall-clock, the GPU's enforced power limit, draw and clock, and the G-Helper mode, so
throttling (or a quieter profile chosen mid-run) shows. The script never changes the mode.

Output: <training folder>/neuroevolution/<method>/<timestamp>-seed<k>/, where the training folder
is ProjectConfig.training_output_root(): LearningConfig.output_dir (C:/balancebot_training, outside
the synced repository), or $BALANCEBOT_TRAINING_DIR, or results/ where neither exists. --out
overrides it. Copy a finished run into results/ to keep it with the code.
    snapshot.json      config, arguments, profile, policy spec, hardware
    history.csv        one row per generation (with its seconds and the GPU's power state)
    checkpoints.csv    one row per checkpoint: train objective, dev scores, falls, gap
    checkpoints/       best_<robot-seconds>.npz and mean_<robot-seconds>.npz
    final.npz          the selected policy;  summary.json  the run's numbers
"""

import argparse
import csv
import json
import math
import pickle
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # config.py; importing it puts simulation/ and optimization/ on the path

import numpy as np

from config import CONFIG
from balance_sim import load_policy, save_policy, unpack
from batch_tuning import BatchEvaluator, PolicySearch
from power import POWER_FIELDS, ghelper_mode, keep_awake, power_state
from tuning import provenance

METHODS = ("linear", "mlp", "warm", "residual")


# ---- setup ----------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, help="L (linear), A (mlp), A-warm (warm), B1 (residual)")
    parser.add_argument("--profile", default="quick", help="LearningConfig run profile: quick or full")
    parser.add_argument("--run-seed", type=int, default=0, help="seeds the initial weights and CMA-ES")
    parser.add_argument("--hidden", type=str, default=None, help="hidden sizes, e.g. 32,32 (default: config)")
    parser.add_argument("--authority", type=float, default=None, help="residual authority (x max torque)")
    parser.add_argument("--warm-from", type=str, default=None, help="A-warm: the linear policy (.npz) to start at")
    parser.add_argument("--popsize", type=int, default=None)
    parser.add_argument("--sigma0", type=float, default=None)
    parser.add_argument("--budget", type=float, default=None, help="robot-seconds cap (default: config)")
    parser.add_argument("--generations", type=int, default=None, help="generation cap (default: profile)")
    parser.add_argument("--train-seeds", type=int, default=None, help="number of training seeds (default: profile)")
    parser.add_argument("--dev-seeds", type=int, default=None, help="use only the first N seeds of the dev tier")
    parser.add_argument("--backend", choices=("auto", "torch", "numpy"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--float64", action="store_true", help="train in float64 (default: the profile's float32)")
    parser.add_argument("--out", type=str, default=None, help="run folder (default: a new timestamped one)")
    parser.add_argument("--resume", type=str, default=None, help="continue the run saved in this folder")
    parser.add_argument("--allow-screen-off", action="store_true",
                        help="let the screen turn off while it runs (sleep stays blocked either way)")
    # For tests and smoke checks: shorter scenarios, and stopping early as if interrupted.
    parser.add_argument("--train-duration", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dev-duration", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stop-after", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.resume is None and args.method is None:
        parser.error("--method is required (or --resume)")
    return args


def resolve_backend(args) -> str:
    if args.backend != "auto":
        return args.backend
    try:
        import torch
        return "torch" if torch.cuda.is_available() else "numpy"
    except ImportError:
        return "numpy"


def build_spec(args):
    hidden = None if args.hidden is None else tuple(int(h) for h in args.hidden.split(","))
    return CONFIG.build_policy_spec(args.method, hidden=hidden, authority=args.authority)


class Setup:
    """The evaluators and settings a run needs, rebuilt identically on resume from the saved args."""

    def __init__(self, args):
        learning = CONFIG.learning
        self.args = args
        self.profile = learning.profile(args.profile)
        self.spec = build_spec(args)
        self.backend = resolve_backend(args)
        robust = CONFIG.build_robustness()
        n_train = args.train_seeds or self.profile.n_train_seeds
        self.robust = replace(robust, train_seeds=tuple(range(1, n_train + 1)))
        tier = robust.smoke_seeds if self.profile.dev_tier == "smoke" else robust.dev_seeds
        self.dev_seeds = tuple(tier[:args.dev_seeds] if args.dev_seeds else tier)
        space = CONFIG.build_search_space()
        train_cfg = CONFIG.build_eval_config()
        dev_cfg = CONFIG.build_eval_config(validation=True)
        if args.train_duration:
            train_cfg = replace(train_cfg, duration=args.train_duration)
        if args.dev_duration:
            dev_cfg = replace(dev_cfg, duration=args.dev_duration)
        common = dict(robustness=self.robust, controller="policy", policy=self.spec, backend=self.backend,
                      device=args.device)
        train_dtype = np.float64 if (args.float64 or not self.profile.float32) else np.float32
        self.train = BatchEvaluator(train_cfg, space, dtype=train_dtype, max_batch=args.max_batch, **common)
        # Dev scores are float64 (fairness contract 7-8), on whichever backend is training.
        self.dev20 = BatchEvaluator(train_cfg, space, dtype=np.float64, **common)
        self.dev60 = BatchEvaluator(dev_cfg, space, dtype=np.float64, **common) if self.profile.dev_60s else None
        self.train_dtype = train_dtype

        self.popsize = args.popsize or (learning.linear_popsize if args.method == "linear" else learning.mlp_popsize)
        self.sigma0 = args.sigma0 or dict(learning.sigma0)[args.method]
        self.budget = args.budget or dict(learning.budget_robot_seconds)[args.method]
        self.max_generations = args.generations or self.profile.max_generations
        per_eval = self.train.simulations_per_evaluation() * train_cfg.duration
        pid_budget = learning.pid_reference_evaluations * per_eval
        marks = sorted(set(learning.checkpoints_robot_seconds) | {pid_budget})
        self.checkpoints = [m for m in marks if m < self.budget] + [self.budget]
        self.pid_budget = pid_budget


def initial_point(setup: Setup, rng: np.random.Generator):
    """(x0, stds or None, robot-seconds already charged) for the method (plan 4.1-5)."""
    spec, learning, args = setup.spec, CONFIG.learning, setup.args
    x0 = np.zeros(spec.n_params)
    linear, layers = unpack(spec, x0)            # views into x0
    for k, (W, b) in enumerate(layers):
        out, inp = W.shape
        if k < len(layers) - 1:
            W[...] = rng.normal(0.0, math.sqrt(2.0 / (inp + out)), W.shape)   # Xavier normal
        elif args.method == "mlp":
            W[...] = rng.normal(0.0, learning.init_output_std, W.shape)
        # warm and residual: the output layer stays at zero, so the start is exactly L-best / the PID
    stds, charged = None, 0.0
    if args.method == "linear":
        linear[...] = rng.normal(0.0, 1.0, linear.shape)          # a random start, not the PID's weights
    elif args.method == "warm":
        if not args.warm_from:
            raise SystemExit("--method warm needs --warm-from <linear policy .npz>")
        l_spec, l_params, _ = load_policy(args.warm_from)
        if l_spec.hidden or not l_spec.linear_path:
            raise SystemExit("--warm-from must be a linear policy")
        weights = dict(zip(l_spec.inputs, l_params))
        linear[...] = [weights.get(i, 0.0) for i in spec.inputs]
        stds = np.ones(spec.n_params)
        stds[:spec.n_inputs] = learning.warm_linear_std
        summary = Path(args.warm_from).with_name("summary.json")
        if summary.exists():                                       # L's robot-seconds count against A-warm
            charged = float(json.loads(summary.read_text())["robot_seconds_total"])
    return x0, stds, charged


def hardware() -> dict:
    info = provenance()
    info["cpu"] = platform.processor() or platform.machine()
    info.update(ghelper_mode())            # the laptop's power profile at the start (also logged per generation)
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:   # older NumPy
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as features
        except ImportError:
            features = {}
    info["cpu_simd"] = sorted(k for k, v in features.items() if v)
    torch = sys.modules.get("torch")
    if torch is not None:
        info["cuda"] = torch.version.cuda
        info["tf32"] = {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32,
                        "float32_matmul_precision": torch.get_float32_matmul_precision()}
    return info


# ---- checkpoints ----------------------------------------------------------------------------

CHECKPOINT_FIELDS = ["robot_seconds", "generation", "which", "train_objective", "dev20_objective",
                     "dev20_falls", "dev60_objective", "dev60_falls", "gap20", "file"]


def score_checkpoint(setup: Setup, search: PolicySearch, folder: Path):
    """Save best and mean, dev-score both. Returns the rows and the better candidate's dev score."""
    rs = search.robot_seconds
    tag = f"{rs:.4g}".replace("+", "")
    rows = []
    for which, params in (("best", search.best_params), ("mean", search.mean_params)):
        path = folder / "checkpoints" / f"{which}_{tag}.npz"
        save_policy(path, setup.spec, params, CONFIG.robot.max_torque, robot_seconds=rs)
        train = setup.train.evaluate_policies(params[None, :]).objective[0] if which == "mean" else search.best_cost
        d20 = setup.dev20.evaluate_policies(params[None, :], setup.dev_seeds)
        row = {"robot_seconds": rs, "generation": search.generation, "which": which,
               "train_objective": float(train), "dev20_objective": float(d20.objective[0]),
               "dev20_falls": int(d20.n_fell[0]), "dev60_objective": "", "dev60_falls": "",
               "gap20": float(d20.objective[0] / train - 1.0), "file": path.relative_to(folder).as_posix()}
        if setup.dev60 is not None:
            d60 = setup.dev60.evaluate_policies(params[None, :], setup.dev_seeds)
            row.update(dev60_objective=float(d60.objective[0]), dev60_falls=int(d60.n_fell[0]))
        rows.append(row)
    key = "dev60_objective" if setup.dev60 is not None else "dev20_objective"
    winner = min(rows, key=lambda r: r[key])
    return rows, winner, winner[key]


# ---- the run --------------------------------------------------------------------------------

def run(argv=None) -> Path:
    """Parse the arguments and run, keeping the machine (and the GPU's power budget) awake meanwhile."""
    args = parse_args(argv)
    with keep_awake(display=not args.allow_screen_off):
        return _run(args)


def _run(args) -> Path:
    if args.resume:
        folder = Path(args.resume)
        with open(folder / "state.pkl", "rb") as f:
            saved = pickle.load(f)
        stop_after = args.stop_after
        args = argparse.Namespace(**saved["args"])
        args.stop_after = stop_after
        setup = Setup(args)
        search = PolicySearch.restore(setup.train, saved["search"])
        run_state = saved["run"]
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = Path(args.out) if args.out else CONFIG.training_output_root() / "neuroevolution" / \
            args.method / f"{stamp}-seed{args.run_seed}"
        (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
        setup = Setup(args)
        rng = np.random.default_rng([args.run_seed, 7])
        x0, stds, charged = initial_point(setup, rng)
        search = PolicySearch(setup.train, x0, setup.sigma0, setup.popsize, setup.max_generations,
                              seed=args.run_seed, stds=stds, full_cma_max_params=CONFIG.learning.full_cma_max_params)
        run_state = {"next_checkpoint": 0, "best_dev": math.inf, "stagnant": 0, "selected": None,
                     "charged_before": charged, "stop_reason": None}
        snapshot = {"config": CONFIG.snapshot(), "args": vars(args), "profile": asdict(setup.profile),
                    "spec": asdict(setup.spec), "n_params": setup.spec.n_params, "backend": setup.backend,
                    "train_dtype": np.dtype(setup.train_dtype).name, "train_seeds": list(setup.robust.train_seeds),
                    "dev_seeds": list(setup.dev_seeds), "popsize": setup.popsize, "sigma0": setup.sigma0,
                    "budget_robot_seconds": setup.budget, "checkpoints_robot_seconds": setup.checkpoints,
                    "pid_reference_budget_robot_seconds": setup.pid_budget, "method": search.method,
                    "hardware": hardware()}
        (folder / "snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str))
        with open(folder / "history.csv", "w", newline="") as f:
            csv.writer(f).writerow(["generation", "n_evals", "robot_seconds", "best_train_objective",
                                    "generation_min", "generation_median", "sigma", "elapsed_s",
                                    "generation_s", *POWER_FIELDS])
        with open(folder / "checkpoints.csv", "w", newline="") as f:
            csv.DictWriter(f, CHECKPOINT_FIELDS).writeheader()
        search.start()
        print(f"{args.method}: {setup.spec.n_params} parameters, {search.method}, population {setup.popsize}, "
              f"{setup.backend}; start point scores {search.best_cost:.4f}")

    def save_state():
        tmp = folder / "state.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"args": vars(args), "search": search.state(), "run": run_state}, f)
        tmp.replace(folder / "state.pkl")

    generations_this_call = 0
    while True:
        # Checkpoint when one or more marks have been passed, and always at the end.
        over_budget = (search.n_evals + setup.popsize) * search.per_eval > setup.budget
        capped = search.generation >= setup.max_generations
        at_end = over_budget or capped or search.stopped()
        passed = run_state["next_checkpoint"]
        while passed < len(setup.checkpoints) and search.robot_seconds >= setup.checkpoints[passed]:
            passed += 1
        if (passed > run_state["next_checkpoint"] or at_end) and run_state.get("scored_at") != search.n_evals:
            rows, winner, dev = score_checkpoint(setup, search, folder)
            with open(folder / "checkpoints.csv", "a", newline="") as f:
                csv.DictWriter(f, CHECKPOINT_FIELDS).writerows(rows)
            print(f"  checkpoint at {search.robot_seconds:.3g} robot-s: dev {dev:.4f} ({winner['which']})")
            if dev < run_state["best_dev"]:
                run_state.update(best_dev=dev, stagnant=0, selected=winner)
            else:
                run_state["stagnant"] += 1
            run_state["scored_at"] = search.n_evals
            limit = setup.profile.stagnation_checkpoints
            if limit and run_state["stagnant"] >= limit:
                run_state["stop_reason"] = f"dev score stagnant for {limit} checkpoints"
        run_state["next_checkpoint"] = passed
        if run_state["stop_reason"] is None and at_end:
            run_state["stop_reason"] = ("budget" if over_budget else "generation cap" if capped else
                                        "cma-es stop: " + str(search.es.stop()))
        if run_state["stop_reason"] is not None:
            save_state()
            break
        if args.stop_after is not None and generations_this_call >= args.stop_after:
            save_state()
            return folder            # as if interrupted; --resume continues
        before = search.elapsed_s
        costs = search.step()
        generations_this_call += 1
        # The GPU's power state right after the generation: a lowered limit or clock explains a slow one.
        power = power_state() if setup.backend == "torch" else {}
        with open(folder / "history.csv", "a", newline="") as f:
            csv.writer(f).writerow([search.generation, search.n_evals, search.robot_seconds, search.best_cost,
                                    float(np.min(costs)), float(np.median(costs)), search.sigma,
                                    round(search.elapsed_s, 2), round(search.elapsed_s - before, 3),
                                    *(power.get(k, "") for k in POWER_FIELDS)])
        save_state()
        if search.generation % 10 == 0:
            watts = power.get("gpu_power_limit_w")
            print(f"  generation {search.generation}: best {search.best_cost:.4f}, "
                  f"{search.robot_seconds:.3g} robot-s, {search.elapsed_s:.0f} s"
                  + (f", GPU limit {watts:.0f} W" if watts else "")
                  + (f" ({power['ghelper_mode']})" if power.get("ghelper_mode") else ""))

    finish(setup, search, run_state, folder)
    return folder


def finish(setup: Setup, search: PolicySearch, run_state: dict, folder: Path):
    selected = run_state["selected"]
    spec, max_torque = setup.spec, CONFIG.robot.max_torque
    params = load_policy(folder / selected["file"])[1]
    save_policy(folder / "final.npz", spec, params, max_torque)
    # Trained in float32: re-score in float64 on the training scenarios and flag a disagreement
    # over 1% (fairness contract 8).
    train64 = BatchEvaluator(setup.train.config, CONFIG.build_search_space(), dtype=np.float64,
                             robustness=setup.robust, controller="policy", policy=spec,
                             backend=setup.backend, device=setup.args.device)
    float64_train = float(train64.evaluate_policies(params[None, :]).objective[0])
    summary = {
        "method": setup.args.method, "search": search.method, "run_seed": setup.args.run_seed,
        "profile": setup.profile.name, "n_params": spec.n_params, "stop_reason": run_state["stop_reason"],
        "generations": search.generation, "n_evals": search.n_evals,
        "robot_seconds": search.robot_seconds,
        "robot_seconds_total": search.robot_seconds + run_state["charged_before"],
        "charged_before": run_state["charged_before"],
        "elapsed_s": search.elapsed_s, "selected": selected,
        "train_objective_float32": selected["train_objective"], "train_objective_float64": float64_train,
        "float32_float64_disagree": abs(float64_train / selected["train_objective"] - 1.0) > 0.01,
        "best_train_objective": search.best_cost,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done ({run_state['stop_reason']}): selected {selected['which']} at {selected['robot_seconds']:.3g} "
          f"robot-s, dev {run_state['best_dev']:.4f}; saved {folder}")


if __name__ == "__main__":
    run()
