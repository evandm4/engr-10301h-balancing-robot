"""Export a learned policy as a bundle the robot can load (hardware/legobot/POLICY_ON_ROBOT_PLAN.md P1).

A bundle is one folder, results/<robot>/policies/<name>/, holding everything needed to run the
policy on the robot and to prove afterwards that it ran what was trained:

    final.npz       the policy (balance_sim.save_policy): spec, weights, torque limit.
                    From a training run: a byte-for-byte copy of the run's final.npz.
    snapshot.json   the config it was trained under (the run's own snapshot.json), which the
                    robot compares against its deployed config (P2b)
    summary.json    the run's summary.json
    manifest.json   kind, n_params, multiply-adds per tick, sha256 of final.npz, source run,
                    git commit, the run's dev score (P1a)
    golden.npz      golden vectors (P1b): one dev-seed scenario (2 s, 400 ticks: the 0.30 rad
                    start, a cruise, a push) through the reference simulator, with every state
                    given to update(), the goal at that tick, and the torque returned. A fresh
                    PolicyController fed the same states and goals must return the same torques;
                    the export checks that bit for bit, and the robot checks it at startup (P2d).
    envelope.json   input envelope (P1c): per raw observation, the 0.1 / 99.9 percentiles and the
                    largest |value| over the policy's own 60 s dev runs (reference engine,
                    --envelope-seeds dev seeds x the 3 start tilts), for the robot's logging and
                    out-of-envelope monitor (P4c, P5a)

Synthetic policies (no training needed; their behaviour is known exactly, for P2-P6's tests):
    l_pid         the linear policy with the reference PID's equivalent weights (L-PID)
    a_warm_zero   A-warm: l_pid's weights on the linear path, Xavier hidden layers, a zero output
                  layer, so it returns l_pid's torques bit for bit
    b1_zero       B1: the residual on the reference PID with a zero output layer, so it returns
                  the reference PID's torques bit for bit

From the project folder:
    python optimization/export_policy.py <run folder> [--name NAME]
    python optimization/export_policy.py --synthetic l_pid
    options: --out-root DIR (default results/<robot>/policies), --robot (default legobot),
             --envelope-seeds N (default 5)
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # config.py; importing it puts simulation/ and optimization/ on the path

import numpy as np

from config import ROBOTS, load_config, results_dir
from balance_sim import load_policy, pid_equivalent_weights, push_profile, save_policy, simulate, step_profile, unpack
from balance_sim.policy import OBSERVATIONS, RESIDUAL_INPUTS, PolicySpec
from tuning import Evaluator, provenance

KNOWN_POLICY_KINDS = ("linear", "mlp", "warm", "residual")
SYNTHETIC = ("l_pid", "a_warm_zero", "b1_zero")
DEFAULT_ROBOT = "legobot"
DEFAULT_ENVELOPE_SEEDS = 5
GOLDEN_DURATION_S = 2.0
BUNDLE_FILES = ("final.npz", "snapshot.json", "summary.json", "manifest.json", "golden.npz", "envelope.json")
# What save_policy writes, plus the metadata run_neuroevolution.py adds to its checkpoints. Anything
# else (a recurrent state, a future policy type's arrays) is refused: load_policy would ignore it.
NPZ_KEYS = frozenset({"params", "inputs", "hidden", "linear_path", "scales", "output_scale", "t_leak",
                      "integral_limits", "max_torque", "residual_gains", "residual_scales", "observations"})
NPZ_EXTRA_KEYS = frozenset({"robot_seconds"})


# ---- policy kinds ---------------------------------------------------------------------------

def classify_policy_kind(spec: PolicySpec) -> str:
    """'linear' (L), 'mlp' (A), 'warm' (A-warm) or 'residual' (B1); ValueError for anything else."""
    if spec.residual:
        if spec.hidden and not spec.linear_path:
            return "residual"
    elif spec.linear_path and not spec.hidden:
        return "linear"
    elif not spec.linear_path and spec.hidden:
        return "mlp"
    elif spec.linear_path and spec.hidden:
        return "warm"
    raise ValueError(f"unsupported policy spec (residual={spec.residual}, hidden={spec.hidden}, "
                     f"linear_path={spec.linear_path}); known kinds: {KNOWN_POLICY_KINDS}")


def multiply_adds_per_tick(spec: PolicySpec) -> int:
    """Multiply-adds of the linear path and the MLP's matrix-vector products per tick (not the
    residual's base PID, the tanh, or the input scaling)."""
    return (spec.n_inputs if spec.linear_path else 0) + sum(o * i for o, i in spec.layer_shapes)


def observation_names(spec: PolicySpec) -> list:
    """Names of PolicyController.last_raw's entries: OBSERVATIONS, then RESIDUAL_INPUTS for a residual."""
    return list(OBSERVATIONS + (RESIDUAL_INPUTS if spec.residual else ()))


# ---- synthetic policies ---------------------------------------------------------------------

def _xavier_hidden(layers, rng: np.random.Generator) -> None:
    """Hidden layers Xavier-normal, biases and the output layer left at zero (run_neuroevolution's
    initial_point for A-warm and B1)."""
    for W, _ in layers[:-1]:
        out, inp = W.shape
        W[...] = rng.normal(0.0, math.sqrt(2.0 / (inp + out)), W.shape)


def synthetic_policy(synthetic: str, cfg, seed: int = 0):
    """(spec, params) of a synthetic policy (module docstring)."""
    if synthetic not in SYNTHETIC:
        raise ValueError(f"unknown synthetic policy {synthetic!r}; choose from {SYNTHETIC}")
    gains = cfg.reference_pid_gains()
    l_spec = cfg.build_policy_spec("linear")
    l_params = pid_equivalent_weights(l_spec, gains)
    if synthetic == "l_pid":
        return l_spec, l_params
    rng = np.random.default_rng(seed)
    if synthetic == "a_warm_zero":
        spec = cfg.build_policy_spec("warm")
        params = np.zeros(spec.n_params)
        linear, layers = unpack(spec, params)          # views into params
        weights = dict(zip(l_spec.inputs, l_params))
        linear[...] = [weights.get(i, 0.0) for i in spec.inputs]
    else:
        spec = cfg.build_policy_spec("residual", base_gains=tuple(gains))
        params = np.zeros(spec.n_params)
        _, layers = unpack(spec, params)
    _xavier_hidden(layers, rng)
    return spec, params


# ---- golden vectors (P1b) -------------------------------------------------------------------

class _Recorder:
    """Wraps a PolicyController (or a PID) for simulate(); calls on_update(state, goal, torque,
    controller) after every update, with the goal the controller saw at that tick."""

    def __init__(self, inner, on_update):
        self.inner, self.on_update = inner, on_update

    def reset(self):
        self.inner.reset()

    def goal_velocity(self, t):
        return self.inner.goal_velocity(t)

    def pitch_reference(self):
        return self.inner.pitch_reference()

    def update(self, state, dt):
        goal = self.inner.goal_velocity(self.inner._time)
        torque = self.inner.update(state, dt)
        self.on_update(state, goal, torque, self.inner)
        return torque


def golden_scenario(cfg) -> dict:
    """The golden scenario's settings: one dev seed, the largest start tilt, and a compact goal and
    push so the start, a cruise and a push all fall within GOLDEN_DURATION_S."""
    s = cfg.scenario
    cruise_v = float(s.goal_steps[0][1])
    push_f = float(s.pushes[0][2]) if s.pushes else 0.0
    return {"seed": int(cfg.robustness.dev_seeds[0]), "start_pitch": float(s.start_pitches_rad[-1]),
            "duration": GOLDEN_DURATION_S, "goal_steps": ((0.4, cruise_v, 0.6),),
            "pushes": ((1.2, 0.05, push_f),)}


def run_golden_scenario(cfg, controller, on_update):
    """Run `controller` (built with the golden goal; golden_goal) through the golden scenario in the
    reference simulator, calling on_update after every tick. Returns the SimResult."""
    g = golden_scenario(cfg)
    return simulate(cfg.build_dynamics(), _Recorder(controller, on_update), duration=g["duration"],
                    dt=cfg.timing.dt_s, initial_state=np.array([0.0, 0.0, g["start_pitch"], 0.0]),
                    control_dt=cfg.timing.control_dt_s, sensor=cfg.build_sensor(seed=g["seed"]),
                    latency=cfg.timing.latency_s, disturbance=push_profile(g["pushes"]))


def golden_goal(cfg):
    """The golden scenario's goal velocity as a function of time."""
    return step_profile(golden_scenario(cfg)["goal_steps"], cfg.scenario.goal_transition_s)


def replay_golden(cfg, spec: PolicySpec, params, golden) -> np.ndarray:
    """The torques a fresh PolicyController returns for the golden states and goals (as the robot's
    self-test replays them: the goal set each tick, on the base PID too for a residual)."""
    controller = cfg.build_policy_controller(spec, params)
    controller.reset()
    dt = float(golden["dt"])
    torques = []
    for state, goal in zip(np.asarray(golden["states"]), np.asarray(golden["goals"])):
        controller.goal = float(goal)
        if controller.base is not None:
            controller.base.goal = float(goal)
        torques.append(controller.update(state, dt))
    return np.array(torques, dtype=float)


def golden_vectors(cfg, spec: PolicySpec, params) -> dict:
    """golden.npz's arrays, checked to replay bit for bit through a fresh controller."""
    g = golden_scenario(cfg)
    states, goals, torques = [], [], []

    def record(state, goal, torque, _):
        states.append(np.array(state[:4], dtype=float))
        goals.append(float(goal))
        torques.append(float(torque))

    run_golden_scenario(cfg, cfg.build_policy_controller(spec, params, goal=golden_goal(cfg)), record)
    golden = {"states": np.array(states, dtype=float).reshape(-1, 4), "torques": np.array(torques, dtype=float),
              "goals": np.array(goals, dtype=float), "dt": np.float64(cfg.timing.control_dt_s),
              "seed": np.int64(g["seed"]), "start_pitch": np.float64(g["start_pitch"])}
    replayed = replay_golden(cfg, spec, params, golden)
    if not np.array_equal(replayed, golden["torques"]):
        worst = float(np.max(np.abs(replayed - golden["torques"])))
        raise RuntimeError(f"golden vectors don't replay bit for bit through a fresh controller (worst {worst:.3g} N m)")
    return golden


# ---- input envelope (P1c) -------------------------------------------------------------------

def input_envelope(cfg, spec: PolicySpec, params, envelope_seeds: int = DEFAULT_ENVELOPE_SEEDS) -> dict:
    """envelope.json's content: raw observation ranges over the 60 s dev scenarios (reference engine,
    nominal robot), the first `envelope_seeds` dev seeds x every start tilt."""
    dev = tuple(cfg.robustness.dev_seeds)
    if not 1 <= envelope_seeds <= len(dev):
        raise ValueError(f"envelope_seeds must be 1..{len(dev)}, got {envelope_seeds}")
    seeds = tuple(int(s) for s in dev[:envelope_seeds])
    ev = Evaluator(cfg.build_eval_config(validation=True), cfg.build_search_space())
    raws, falls, costs = [], 0, []
    record = lambda state, goal, torque, c: raws.append(np.array(c.last_raw, dtype=float))
    for pitch, seed in ev.scenarios(seeds):
        result = ev.run_controller(_Recorder(cfg.build_policy_controller(spec, params), record), pitch, seed)
        falls += int(result.fell)
        costs.append(float(ev.score(result)))
    raws = np.array(raws)
    names = observation_names(spec)
    observations = {}
    for i, name in enumerate(names):
        lo, hi = np.percentile(raws[:, i], [0.1, 99.9])
        observations[name] = {"p00_1": float(lo), "p99_9": float(hi), "max_abs": float(np.max(np.abs(raws[:, i])))}
    return {"observation_names": names, "n_seeds": len(seeds), "seeds": list(seeds),
            "start_pitches_rad": [float(p) for p in cfg.scenario.start_pitches_rad],
            "duration_s": float(ev.config.duration), "n_runs": len(costs), "n_falls": falls,
            "n_ticks": int(raws.shape[0]),
            "nominal_cost_mean": float(np.mean(costs)),   # nominal robot, fixed latency: not the robust dev score
            "observations": observations}


# ---- the bundle -----------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    info = provenance()
    commit = info.get("git_commit")
    if commit and info.get("git_dirty"):
        commit += "-dirty"
    return commit


def _source_run(folder: Path, method: str) -> str:
    """A machine-independent label for the run: the repo-relative path when the run is in the repo,
    else neuroevolution/<method>/<run folder> (the layout run_neuroevolution.py writes)."""
    folder = folder.resolve()
    try:
        return folder.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        parts = folder.parts
        if "neuroevolution" in parts:
            return "/".join(parts[len(parts) - 1 - parts[::-1].index("neuroevolution"):])
        return f"neuroevolution/{method}/{folder.name}"


def _check_npz_keys(path: Path) -> None:
    with np.load(path) as f:
        keys = set(f.files)
    missing, unknown = NPZ_KEYS - keys, keys - NPZ_KEYS - NPZ_EXTRA_KEYS
    if missing or unknown:
        raise ValueError(f"{path} is not a policy file this exporter knows: "
                         + (f"missing {sorted(missing)} " if missing else "")
                         + (f"unknown keys {sorted(unknown)}" if unknown else ""))


def _dev_score(summary: dict, snapshot: dict) -> dict:
    selected = summary.get("selected") or {}
    blank = lambda v: None if v == "" else v
    score = {k: blank(selected.get(k)) for k in ("which", "robot_seconds", "generation", "dev20_objective",
                                                 "dev20_falls", "dev60_objective", "dev60_falls")}
    score.update(profile=summary.get("profile"), dev_seeds=snapshot.get("dev_seeds"))
    return score


def _json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def export_bundle(run_folder=None, *, synthetic: Optional[str] = None, name: Optional[str] = None, out_root=None,
                  cfg=None, envelope_seeds: int = DEFAULT_ENVELOPE_SEEDS, seed: int = 0,
                  source_label: Optional[str] = None, checkpoint: Optional[str] = None) -> Path:
    """Write the bundle of a training run's final.npz (run_folder) or of a synthetic policy;
    returns the bundle folder. Everything is computed and checked before anything is written.

    checkpoint: a policy file inside run_folder (e.g. "checkpoints/mean_7.488e08.npz", the E2.6 selection) to export
    instead of final.npz. It is still copied byte for byte and still named final.npz in the bundle (the robot loads
    that name); the manifest records it as source_checkpoint. The run's summary.json dev score describes final.npz,
    so dev_score is left null for a checkpoint that is not byte-identical to final.npz.
    """
    if (run_folder is None) == (synthetic is None):
        raise ValueError("give either a run folder or synthetic=..., not both or neither")
    if checkpoint is not None and run_folder is None:
        raise ValueError("checkpoint= needs a run folder")
    cfg = load_config(DEFAULT_ROBOT) if cfg is None else cfg
    max_torque = float(cfg.robot.max_torque)
    source = None
    policy_file = None
    if synthetic is not None:
        spec, params = synthetic_policy(synthetic, cfg, seed)
        kind = classify_policy_kind(spec)
        name = synthetic if name is None else name
    else:
        source = Path(run_folder)
        for file in ("final.npz", "snapshot.json", "summary.json"):
            if not (source / file).is_file():
                raise FileNotFoundError(f"{source / file} not found: not a finished training run")
        policy_file = source / "final.npz"
        if checkpoint is not None:
            policy_file = source / checkpoint
            try:
                policy_file.resolve().relative_to(source.resolve())
            except ValueError:
                raise ValueError(f"checkpoint {checkpoint!r} is outside the run folder") from None
            if not policy_file.is_file() or policy_file.suffix != ".npz":
                raise FileNotFoundError(f"{policy_file} is not a policy .npz file")
        _check_npz_keys(policy_file)
        spec, params, saved_torque = load_policy(policy_file)
        kind = classify_policy_kind(spec)
        snapshot = json.loads((source / "snapshot.json").read_text())
        summary = json.loads((source / "summary.json").read_text())
        if snapshot.get("robot", cfg.name) != cfg.name:
            raise ValueError(f"the run is for robot {snapshot['robot']!r}, not {cfg.name!r} (use --robot)")
        if abs(saved_torque - max_torque) > 1e-12:
            raise ValueError(f"the policy's max_torque {saved_torque} differs from {cfg.name}'s {max_torque}")
        if summary.get("method", kind) != kind:
            raise ValueError(f"summary.json says method {summary['method']!r} but final.npz is a {kind!r} policy")
        name = f"{kind}-{source.resolve().name}" if name is None else name
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"bad bundle name {name!r}")
    bundle = Path(results_dir("policies", robot=cfg.name) if out_root is None else out_root) / name
    if bundle.exists() and any(bundle.iterdir()) and not (bundle / "manifest.json").is_file():
        raise FileExistsError(f"{bundle} exists and isn't a policy bundle; not overwriting it")

    if not np.all(np.isfinite(np.asarray(params, dtype=float))):
        raise ValueError("the policy has non-finite parameters")
    golden = golden_vectors(cfg, spec, params)
    envelope = input_envelope(cfg, spec, params, envelope_seeds)
    if not all(np.all(np.isfinite(golden[k])) for k in ("states", "torques", "goals")):
        raise ValueError("the golden vectors are not all finite")
    if envelope["n_ticks"] <= 0 or not all(np.isfinite(v) for o in envelope["observations"].values()
                                           for v in o.values()):
        raise ValueError("the input envelope is empty or not finite")

    bundle.mkdir(parents=True, exist_ok=True)
    for file in BUNDLE_FILES:
        (bundle / file).unlink(missing_ok=True)
    if source is not None:
        shutil.copyfile(policy_file, bundle / "final.npz")
        shutil.copyfile(source / "snapshot.json", bundle / "snapshot.json")
        shutil.copyfile(source / "summary.json", bundle / "summary.json")
        if sha256_file(bundle / "final.npz") != sha256_file(policy_file):
            raise RuntimeError(f"{policy_file.name} changed while being copied")
        dev_score = _dev_score(summary, snapshot)
        if checkpoint is not None and sha256_file(policy_file) != sha256_file(source / "final.npz"):
            dev_score = None   # summary.json's dev score is final.npz's, not this checkpoint's
        config_differences = cfg.scoring_differences(snapshot["config"]) if "config" in snapshot else None
        source_commit = (snapshot.get("hardware") or {}).get("git_commit")
    else:
        save_policy(bundle / "final.npz", spec, params, max_torque)
        _json(bundle / "snapshot.json", {"config": cfg.snapshot(), "robot": cfg.name,
                                         "reference_provisional": cfg.reference_is_provisional,
                                         "synthetic": synthetic, "spec": asdict(spec), "n_params": spec.n_params,
                                         "hardware": provenance()})
        _json(bundle / "summary.json", {"method": kind, "synthetic": synthetic, "n_params": spec.n_params,
                                        "reference_gains": [float(g) for g in cfg.reference_pid_gains()]})
        dev_score, config_differences, source_commit = None, [], None
    # What the robot will load must be what the golden vectors were made with.
    spec2, params2, _ = load_policy(bundle / "final.npz")
    if spec2 != spec or not np.array_equal(params2, np.asarray(params, dtype=float)):
        raise RuntimeError("final.npz doesn't load back as the exported policy")
    np.savez(bundle / "golden.npz", **golden)
    _json(bundle / "envelope.json", envelope)
    manifest = {
        "name": name, "robot": cfg.name, "kind": kind, "synthetic": synthetic, "n_params": spec.n_params,
        "multiply_adds_per_tick": multiply_adds_per_tick(spec), "sha256": sha256_file(bundle / "final.npz"),
        "source_run": None if source is None else (source_label or _source_run(source, kind)),
        "git_commit": _git_commit(),
        "dev_score": dev_score,
        # Extras for people reading the bundle; the robot's loader needs only the fields above.
        "source_path": None if (source is None or source_label is not None) else str(source.resolve()),
        "source_checkpoint": None if source is None else policy_file.resolve().relative_to(
            source.resolve()).as_posix(),
        "max_torque": max_torque, "input_names": list(spec.input_names),
        "observation_names": observation_names(spec), "source_git_commit": source_commit,
        "config_differences": config_differences, "golden_ticks": int(golden["torques"].size),
        "envelope_runs": envelope["n_runs"], "envelope_falls": envelope["n_falls"],
        "golden_saturated_ticks": int(np.sum(np.abs(golden["torques"]) >= max_torque - 1e-12)),
        # sha256 of every other bundle file, so an edited snapshot/golden/envelope is caught too.
        "files_sha256": {f: sha256_file(bundle / f) for f in BUNDLE_FILES if f != "manifest.json"},
    }
    _json(bundle / "manifest.json", manifest)
    return bundle


# ---- command line ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder", nargs="?", default=None, help="a finished run_neuroevolution.py run folder")
    parser.add_argument("--synthetic", choices=SYNTHETIC, default=None, help="export a synthetic policy instead")
    parser.add_argument("--name", default=None, help="bundle name (default: the synthetic's, or <kind>-<run folder>)")
    parser.add_argument("--out-root", default=None, help="where bundles go (default: results/<robot>/policies)")
    parser.add_argument("--robot", choices=ROBOTS, default=DEFAULT_ROBOT, help=f"default {DEFAULT_ROBOT}")
    parser.add_argument("--envelope-seeds", type=int, default=DEFAULT_ENVELOPE_SEEDS,
                        help="dev seeds (x 3 start tilts) of 60 s runs for the input envelope")
    parser.add_argument("--seed", type=int, default=0, help="hidden-layer init of a_warm_zero / b1_zero")
    parser.add_argument("--source-run", "--source-label", dest="source_label", default=None,
                        help="manifest source_run for a run folder (default: repo-relative path, "
                             "else neuroevolution/<method>/<folder>)")
    parser.add_argument("--checkpoint", default=None,
                        help="export this policy file inside the run folder (e.g. checkpoints/mean_7.488e08.npz, "
                             "an E2.6 selection) instead of final.npz")
    args = parser.parse_args(argv)
    if (args.run_folder is None) == (args.synthetic is None):
        parser.error("give a run folder or --synthetic, not both")
    if args.source_label is not None and args.synthetic is not None:
        parser.error("--source-label is for run folders only")
    if args.checkpoint is not None and args.synthetic is not None:
        parser.error("--checkpoint is for run folders only")
    return args


def main(argv=None) -> Path:
    args = parse_args(argv)
    bundle = export_bundle(args.run_folder, synthetic=args.synthetic, name=args.name, out_root=args.out_root,
                           cfg=load_config(args.robot), envelope_seeds=args.envelope_seeds, seed=args.seed,
                           source_label=args.source_label, checkpoint=args.checkpoint)
    m = json.loads((bundle / "manifest.json").read_text())
    print(f"{bundle}: {m['kind']}, {m['n_params']} parameters, {m['multiply_adds_per_tick']} multiply-adds/tick, "
          f"sha256 {m['sha256'][:12]}..., golden {m['golden_ticks']} ticks, envelope {m['envelope_runs']} runs "
          f"({m['envelope_falls']} falls)")
    if m["config_differences"]:
        print("NOTE: trained under a config that differs from today's in: " + ", ".join(m["config_differences"]))
    return bundle


if __name__ == "__main__":
    main()
