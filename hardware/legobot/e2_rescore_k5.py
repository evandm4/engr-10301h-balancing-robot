"""Phase E2.6: pre-registered K=5 CPU float64 realization re-score for LEGO E2 linear (L) runs.

Pre-registered specification (hardware/legobot/E2_NOTES.md, "Notes for E2.5-E2.6 selection",
approved 2026-11-25 16:44, before any E2.6 re-score ran; also hardware/legobot/E2_PLAN.md E2.6):

    - K = 5 NumPy f64 realizations of the dev tier (5000-5049 x 3, 60 s). Realization k adds
      k x 1e-15 rad to every scenario's initial tilt, k = 0...4; k = 0 is the plain dev tier.
      Same machine, CUDA_VISIBLE_DEVICES=-1.
    - Scored this way: every candidate (each run's final.npz and its saved checkpoint candidates)
      and the frozen MB PID baseline.
    - Reported per candidate:
        * the mean over realizations of the mean dev cost;
        * the median cost (median over all K x N scenario runs);
        * the both-up cost (per realization, the mean over scenarios where the candidate and the
          MB PID both stay up, then averaged over realizations);
        * total falls / (K x N), with N = 150. Falls per realization are listed too.
    - Selection key: the mean over realizations of the mean dev cost, as before. Ties (equal at
      the reported 3-decimal precision) go to fewer total falls.
    - E2.8's frozen tiers are unchanged: one realization, as already specified.
    - Per-run machine and stop_reason are reported alongside, for disclosure.

Code identity:
    The E2 L runs were trained at commit 4582fe7. Before scoring, this script checks
    `git diff 4582fe7 -- simulation/ optimization/batch_tuning config.py legobot_config.py`
    (and committed diff `4582fe7 HEAD`) in the code tree it imports from (--code-root, default
    this repository). If non-empty, it refuses to run unless --allow-code-drift is passed.
    To run against the pinned training tree without modifying it, pass:
        --code-root C:\\balancebot_training\\e2_L_worktree

Numerical invariance:
    On NumPy float64, simulation/batch_sim/policy.py computes forward() using elementwise
    multiply-adds in fixed input order (never BLAS matmul), so per-scenario costs and falls
    are bit-identical across --batch-size (1, 8, 16, 32) and --workers (1 vs N).

Outputs, in results/legobot/e2_selection/<timestamp>-e26-k5/ (with -dry appended when
--seeds/--duration/--k shorten it):
    selection.json          runs & machine provenance, skipped/missing runs, code identity,
                            every candidate's K-realization metrics, ranking, selected winner,
                            per-run bests, and machine/parity summary;
    dev_per_scenario.csv    one row per (candidate, realization, seed, start_pitch);
    e2_notes_tables.md      paste-ready Markdown tables for E2_NOTES.md;
    cache/                  per-unique-candidate .npz caches + manifest.json (for --resume).

Usage (from the repository root, using the venv):
    ROBOT=legobot CUDA_VISIBLE_DEVICES=-1 python hardware/legobot/e2_rescore_k5.py --list-runs
    ROBOT=legobot CUDA_VISIBLE_DEVICES=-1 python hardware/legobot/e2_rescore_k5.py --require-all \\
        --code-root C:\\balancebot_training\\e2_L_worktree
    ROBOT=legobot CUDA_VISIBLE_DEVICES=-1 python hardware/legobot/e2_rescore_k5.py \\
        --resume results/legobot/e2_selection/<stamp>-e26-k5

Seeds: dev (5000-5049) only, checked against every held-out/frozen/training tier at start;
never touches validation (3000-3099), fall-rate (6000-6999), confirm (6,000,000-6,000,099),
dev_extended, or retired (100-109) seeds.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Enforce CPU-only before any numpy/torch/project import.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO = Path(__file__).resolve().parents[2]
_HW_LEGOBOT = Path(__file__).resolve().parent
if str(_HW_LEGOBOT) not in sys.path:
    sys.path.insert(0, str(_HW_LEGOBOT))

from e2_select import MB, RUN_SEEDS  # reuse E2.2 selection constants

DEFAULT_RUNS_ROOT = Path("C:/balancebot_training/legobot/neuroevolution/linear")
DEFAULT_EXPECTED_COMMIT = "4582fe7"
DEFAULT_TRAIN_SEEDS = 150
DEFAULT_K = 5
DEFAULT_NUDGE_RAD = 1e-15
DEFAULT_BATCH_SIZE = 16
CODE_IDENTITY_PATHS = ("simulation/", "optimization/batch_tuning", "config.py", "legobot_config.py")
PRIORITY_CLASSES = {"normal": 0x00000020, "below-normal": 0x00004000, "idle": 0x00000040}

PREREG_SPEC = (
    "Pre-registered K-realization re-score (E2_NOTES.md, approved 2026-11-25 16:44, before any E2.6 "
    "re-score ran): K = 5 NumPy float64 realizations of the dev tier (seeds 5000-5049 x 3 start tilts "
    "= N 150, 60 s); realization k adds k x 1e-15 rad to every scenario's initial tilt (k = 0..4, "
    "k = 0 is the plain dev tier); same machine, CUDA_VISIBLE_DEVICES=-1. Candidates: every run's "
    "final.npz and its saved checkpoint candidates, plus the frozen MB PID baseline. Selection key: "
    "mean over realizations of the mean dev cost; ties equal at the reported 3-decimal precision go "
    "to fewer total falls. Per-run machine and stop_reason reported for disclosure."
)


# ---- process & code-identity helpers -------------------------------------------------------------

def _configure_threads(threads: Optional[int]) -> None:
    if threads is not None and threads > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(threads)


def _set_process_priority(level: str) -> bool:
    """Set Windows process priority ('normal', 'below-normal', 'idle') to avoid starving GPU training."""
    if sys.platform != "win32" or level == "normal":
        return False
    cls = PRIORITY_CLASSES.get(level)
    if cls is None:
        return False
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass.restype = ctypes.c_int
        return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), cls))
    except Exception:
        return False


def _add_code_paths(code_root: Path) -> None:
    """Put code_root (and its simulation/, optimization/, hardware/legobot/) at the front of sys.path."""
    resolved = Path(code_root).resolve()
    for p in (resolved / "hardware" / "legobot", resolved / "optimization", resolved / "simulation", resolved):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


def check_code_identity(code_root: Path, expected_commit: str = DEFAULT_EXPECTED_COMMIT,
                        paths: Sequence[str] = CODE_IDENTITY_PATHS) -> dict:
    """Inspect git diff vs expected_commit on the scoring/config paths in code_root.

    Checks both committed drift (`git diff expected HEAD -- paths`) and working-tree drift
    (`git diff expected -- paths`), plus untracked files under `paths`.
    """
    code_root = Path(code_root).resolve()
    info = {
        "code_root": str(code_root),
        "expected_commit": expected_commit,
        "paths": list(paths),
        "head_commit": None,
        "committed_diff_files": [],
        "worktree_diff_files": [],
        "untracked_files": [],
        "has_drift": True,
        "error": None,
    }
    try:
        head = subprocess.run(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        if head.returncode != 0:
            info["error"] = (head.stderr or head.stdout or "git rev-parse HEAD failed").strip()
            return info
        info["head_commit"] = head.stdout.strip()

        def _git_lines(*args: str) -> Tuple[int, List[str], str]:
            proc = subprocess.run(
                ["git", "-C", str(code_root), *args],
                capture_output=True, text=True, check=False, timeout=30,
            )
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            return proc.returncode, lines, (proc.stderr or "").strip()

        rc1, committed, err1 = _git_lines("diff", "--name-status", expected_commit, "HEAD", "--", *paths)
        if rc1 != 0:
            info["error"] = err1 or f"git diff {expected_commit} HEAD failed (rc={rc1})"
            return info
        rc2, worktree, err2 = _git_lines("diff", "--name-status", expected_commit, "--", *paths)
        if rc2 != 0:
            info["error"] = err2 or f"git diff {expected_commit} failed (rc={rc2})"
            return info
        rc3, untracked, _ = _git_lines("ls-files", "--others", "--exclude-standard", "--", *paths)
        info["committed_diff_files"] = committed
        info["worktree_diff_files"] = worktree
        info["untracked_files"] = untracked if rc3 == 0 else []
        info["has_drift"] = bool(committed or worktree or info["untracked_files"])
        return info
    except OSError as exc:
        info["error"] = f"git invocation failed: {exc}"
        return info


# ---- run & candidate collection ------------------------------------------------------------------

@dataclass
class LRun:
    seed: int
    folder: Path
    status: str = "missing"   # "finished" | "in_progress" | "missing" | "invalid"
    commit: Optional[str] = None
    dirty: Optional[bool] = None
    hostname: Optional[str] = None
    gpu: Optional[str] = None
    cpu: Optional[str] = None
    method: Optional[str] = None
    profile: Optional[str] = None
    train_dtype: Optional[str] = None
    n_train_seeds: Optional[int] = None
    stop_reason: Optional[str] = None
    generations: Optional[int] = None
    n_evals: Optional[int] = None
    robot_seconds: Optional[float] = None
    elapsed_s: Optional[float] = None
    selected_file: Optional[str] = None
    selected_in_run_dev60: Optional[float] = None
    selected_in_run_dev60_falls: Optional[int] = None
    n_checkpoints: int = 0
    flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def dirty_reading(self) -> str:
        if self.dirty is None:
            return "unknown"
        return "dirty" if self.dirty else "clean"

    def summary(self) -> dict:
        return {
            "run_seed": self.seed,
            "folder": self.name,
            "path": str(self.folder),
            "status": self.status,
            "git_commit": self.commit,
            "git_dirty": self.dirty,
            "git_dirty_reading": self.dirty_reading,
            "hostname": self.hostname,
            "gpu": self.gpu,
            "cpu": self.cpu,
            "method": self.method,
            "profile": self.profile,
            "train_dtype": self.train_dtype,
            "n_train_seeds": self.n_train_seeds,
            "stop_reason": self.stop_reason,
            "generations": self.generations,
            "n_evals": self.n_evals,
            "robot_seconds": self.robot_seconds,
            "elapsed_s": self.elapsed_s,
            "selected_file": self.selected_file,
            "selected_in_run_dev60": self.selected_in_run_dev60,
            "selected_in_run_dev60_falls": self.selected_in_run_dev60_falls,
            "n_checkpoints": self.n_checkpoints,
            "flags": list(self.flags),
            "notes": list(self.notes),
        }


@dataclass
class PolicyCandidate:
    label: str
    kind: str                       # "cascaded" (MB PID) or "policy"
    run_seed: Optional[int] = None  # None for MB PID
    run_folder: Optional[str] = None
    checkpoint_rel: Optional[str] = None  # "final.npz" or "checkpoints/..."
    is_final: bool = False
    which: Optional[str] = None     # "best" | "mean" | "final" | "baseline"
    generation: Optional[int] = None
    robot_seconds: Optional[float] = None
    in_run_train_objective: Optional[float] = None
    in_run_dev60_objective: Optional[float] = None
    in_run_dev60_falls: Optional[int] = None
    source_path: Optional[Path] = None
    params_sha256: Optional[str] = None
    cache_key: str = ""
    matches_final: bool = False
    gains: Optional[object] = None  # np.ndarray
    spec: Optional[object] = None   # PolicySpec for kind == "policy"

    def meta(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "run_seed": self.run_seed,
            "run_folder": self.run_folder,
            "checkpoint_rel": self.checkpoint_rel,
            "is_final": self.is_final,
            "matches_final": self.matches_final,
            "which": self.which,
            "generation": self.generation,
            "robot_seconds": self.robot_seconds,
            "in_run_train_objective": self.in_run_train_objective,
            "in_run_dev60_objective": self.in_run_dev60_objective,
            "in_run_dev60_falls": self.in_run_dev60_falls,
            "source_path": str(self.source_path) if self.source_path else None,
            "params_sha256": self.params_sha256,
            "cache_key": self.cache_key,
        }


def read_l_run(folder: Path, expected_seed: int, expected_commit: str = DEFAULT_EXPECTED_COMMIT,
               expected_train_seeds: int = DEFAULT_TRAIN_SEEDS, cfg=None) -> LRun:
    """Inspect one E2 L run folder (read-only); distinguish missing, in-progress, and finished."""
    run = LRun(seed=expected_seed, folder=Path(folder))
    if not run.folder.exists():
        run.status = "missing"
        run.notes.append("run folder does not exist yet")
        return run

    snap_path = run.folder / "snapshot.json"
    summ_path = run.folder / "summary.json"
    final_path = run.folder / "final.npz"
    ck_dir = run.folder / "checkpoints"
    if ck_dir.exists():
        run.n_checkpoints = len(list(ck_dir.glob("*.npz")))

    snap = None
    if snap_path.exists():
        try:
            snap = json.loads(snap_path.read_text())
        except (OSError, ValueError) as exc:
            run.flags.append(f"unreadable snapshot.json ({type(exc).__name__})")
    if isinstance(snap, dict):
        hw = snap.get("hardware") or {}
        args = snap.get("args") or {}
        run.commit = hw.get("git_commit")
        run.dirty = hw.get("git_dirty")
        run.hostname = hw.get("hostname")
        run.gpu = hw.get("gpu")
        run.cpu = hw.get("cpu")
        run.method = args.get("method")
        run.profile = args.get("profile")
        run.train_dtype = snap.get("train_dtype")
        t_seeds = snap.get("train_seeds")
        run.n_train_seeds = len(t_seeds) if isinstance(t_seeds, list) else None
        if cfg is not None and isinstance(snap.get("config"), dict):
            drift = cfg.scoring_differences(snap["config"])
            if drift:
                run.flags.append("config drift vs current scoring config: " + ", ".join(drift))

    if not summ_path.exists() or not final_path.exists():
        run.status = "in_progress"
        missing_parts = [name for name, p in (("summary.json", summ_path), ("final.npz", final_path)) if not p.exists()]
        run.notes.append(f"unfinished run (missing {', '.join(missing_parts)})")
        return run

    try:
        summ = json.loads(summ_path.read_text())
    except (OSError, ValueError) as exc:
        run.status = "invalid"
        run.flags.append(f"unreadable summary.json ({type(exc).__name__})")
        return run

    run.status = "finished"
    run.stop_reason = summ.get("stop_reason")
    run.generations = summ.get("generations")
    run.n_evals = summ.get("n_evals")
    run.robot_seconds = summ.get("robot_seconds")
    run.elapsed_s = summ.get("elapsed_s")
    if run.method is None:
        run.method = summ.get("method")
    if run.profile is None:
        run.profile = summ.get("profile")
    sel = summ.get("selected") or {}
    run.selected_file = sel.get("file")
    run.selected_in_run_dev60 = sel.get("dev60_objective")
    run.selected_in_run_dev60_falls = sel.get("dev60_falls")

    # Provenance checks
    if summ.get("run_seed") != expected_seed:
        run.flags.append(f"summary.json run_seed={summ.get('run_seed')!r} != expected {expected_seed}")
    if expected_commit and (not run.commit or not str(run.commit).startswith(expected_commit)):
        run.flags.append(f"git_commit {str(run.commit)[:9]!r} != expected {expected_commit!r}")
    if run.dirty is True:
        run.flags.append("trained from a dirty git tree")
    elif run.dirty is None:
        run.flags.append("git_dirty unknown")
    if run.method != "linear":
        run.flags.append(f"method {run.method!r} != 'linear'")
    if run.profile != "full":
        run.flags.append(f"profile {run.profile!r} != 'full'")
    if run.train_dtype is not None and run.train_dtype != "float64":
        run.flags.append(f"train_dtype {run.train_dtype!r} != 'float64'")
    if run.n_train_seeds is not None and run.n_train_seeds != expected_train_seeds:
        run.flags.append(f"n_train_seeds {run.n_train_seeds} != {expected_train_seeds}")
    if run.n_checkpoints == 0:
        run.flags.append("no checkpoint .npz files under checkpoints/")
    return run


def collect_l_runs(runs_root: Path, run_seeds: Sequence[int] = RUN_SEEDS, prefix: str = "e2-seed",
                   expected_commit: str = DEFAULT_EXPECTED_COMMIT,
                   expected_train_seeds: int = DEFAULT_TRAIN_SEEDS,
                   cfg=None) -> Tuple[List[LRun], List[LRun]]:
    """Return (finished_runs, skipped_runs) across expected E2 L run seeds."""
    runs_root = Path(runs_root)
    finished: List[LRun] = []
    skipped: List[LRun] = []
    for seed in run_seeds:
        run = read_l_run(runs_root / f"{prefix}{seed}", seed, expected_commit, expected_train_seeds, cfg=cfg)
        if run.status == "finished":
            finished.append(run)
        else:
            skipped.append(run)
    return finished, skipped


def policy_params_hash(spec, params) -> str:
    """Stable SHA-256 over the PolicySpec architecture/scales and float64 parameter bytes."""
    import numpy as np
    from dataclasses import asdict

    h = hashlib.sha256()
    spec_payload = json.dumps(asdict(spec), sort_keys=True, default=str).encode("utf-8")
    h.update(spec_payload)
    h.update(b"\0")
    h.update(np.ascontiguousarray(params, dtype=np.float64).tobytes())
    return h.hexdigest()


def _read_checkpoints_csv(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    out: Dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rel = (row.get("file") or "").replace("\\", "/")
            if rel:
                out[rel] = row
    return out


def _opt_float(val) -> Optional[float]:
    if val in (None, ""):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _opt_int(val) -> Optional[int]:
    if val in (None, ""):
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def collect_candidates(runs: Sequence[LRun], mb_gains, load_policy_fn,
                       expected_max_torque: Optional[float] = None) -> List[PolicyCandidate]:
    """Build the ordered candidate list: frozen MB PID baseline + every run's final.npz and checkpoints."""
    import numpy as np

    mb_arr = np.asarray(mb_gains, dtype=np.float64)
    mb_hash = hashlib.sha256(np.ascontiguousarray(mb_arr, dtype=np.float64).tobytes()).hexdigest()
    candidates: List[PolicyCandidate] = [
        PolicyCandidate(
            label=MB,
            kind="cascaded",
            which="baseline",
            params_sha256=mb_hash,
            cache_key=f"mb_pid_{mb_hash[:16]}",
            gains=mb_arr,
        )
    ]

    for run in runs:
        ck_rows = _read_checkpoints_csv(run.folder / "checkpoints.csv")
        final_path = run.folder / "final.npz"
        spec_f, params_f, mt_f = load_policy_fn(final_path)
        if expected_max_torque is not None and not math.isclose(float(mt_f), float(expected_max_torque), rel_tol=0, abs_tol=1e-12):
            run.flags.append(f"final.npz max_torque {mt_f} != config {expected_max_torque}")
        params_f = np.asarray(params_f, dtype=np.float64)
        final_hash = policy_params_hash(spec_f, params_f)

        sel_meta = ck_rows.get((run.selected_file or "").replace("\\", "/"), {})
        candidates.append(
            PolicyCandidate(
                label=f"L / s{run.seed} / final",
                kind="policy",
                run_seed=run.seed,
                run_folder=run.name,
                checkpoint_rel="final.npz",
                is_final=True,
                matches_final=True,
                which="final",
                generation=_opt_int(sel_meta.get("generation")),
                robot_seconds=_opt_float(sel_meta.get("robot_seconds")),
                in_run_train_objective=_opt_float(sel_meta.get("train_objective")),
                in_run_dev60_objective=run.selected_in_run_dev60,
                in_run_dev60_falls=run.selected_in_run_dev60_falls,
                source_path=final_path,
                params_sha256=final_hash,
                cache_key=f"policy_{final_hash[:16]}",
                gains=params_f,
                spec=spec_f,
            )
        )

        # Preserve checkpoints.csv order first, then any extra .npz files sorted by name.
        ck_dir = run.folder / "checkpoints"
        disk_files = sorted(ck_dir.glob("*.npz")) if ck_dir.exists() else []
        disk_by_rel = {f"checkpoints/{p.name}": p for p in disk_files}
        ordered_rels: List[str] = []
        for rel in ck_rows:
            if rel in disk_by_rel:
                ordered_rels.append(rel)
        for rel in sorted(disk_by_rel):
            if rel not in ordered_rels:
                ordered_rels.append(rel)

        for rel in ordered_rels:
            p = disk_by_rel[rel]
            spec_c, params_c, _ = load_policy_fn(p)
            params_c = np.asarray(params_c, dtype=np.float64)
            h = policy_params_hash(spec_c, params_c)
            meta = ck_rows.get(rel, {})
            which = meta.get("which") or (p.stem.split("_", 1)[0] if "_" in p.stem else "checkpoint")
            candidates.append(
                PolicyCandidate(
                    label=f"L / s{run.seed} / {p.stem}",
                    kind="policy",
                    run_seed=run.seed,
                    run_folder=run.name,
                    checkpoint_rel=rel,
                    is_final=False,
                    matches_final=(h == final_hash),
                    which=which,
                    generation=_opt_int(meta.get("generation")),
                    robot_seconds=_opt_float(meta.get("robot_seconds")),
                    in_run_train_objective=_opt_float(meta.get("train_objective")),
                    in_run_dev60_objective=_opt_float(meta.get("dev60_objective")),
                    in_run_dev60_falls=_opt_int(meta.get("dev60_falls")),
                    source_path=p,
                    params_sha256=h,
                    cache_key=f"policy_{h[:16]}",
                    gains=params_c,
                    spec=spec_c,
                )
            )

    return candidates


# ---- K-realization scoring & resume cache --------------------------------------------------------

def realization_start_pitches(base_pitches: Sequence[float], k: int,
                              nudge_rad: float = DEFAULT_NUDGE_RAD) -> Tuple[float, ...]:
    """Initial start tilts for realization k: base_pitch + k * nudge_rad (k=0 is plain dev tier)."""
    offset = float(k) * float(nudge_rad)
    return tuple(float(p) + offset for p in base_pitches)


def _manifest_payload(seeds: Sequence[int], base_pitches: Sequence[float], duration_s: float,
                      k_realizations: int, nudge_rad: float) -> dict:
    return {
        "backend": "numpy",
        "dtype": "float64",
        "seeds": [int(s) for s in seeds],
        "start_pitches_rad": [float(p) for p in base_pitches],
        "duration_s": float(duration_s),
        "k_realizations": int(k_realizations),
        "nudge_rad": float(nudge_rad),
    }


def init_or_verify_cache(cache_dir: Path, manifest: dict) -> None:
    """Create cache_dir/manifest.json on fresh run, or verify exact match on resume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpath = cache_dir / "manifest.json"
    if mpath.exists():
        existing = json.loads(mpath.read_text())
        if existing != manifest:
            raise SystemExit(
                f"resume cache manifest mismatch in {mpath}:\n"
                f"  existing: {existing}\n"
                f"  current:  {manifest}\n"
                "Refusing to mix cached results from different scoring settings."
            )
    else:
        tmp = cache_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(mpath)


def save_candidate_cache(cache_dir: Path, cache_key: str, costs, fell, fall_time) -> Path:
    """Atomically persist one unique candidate's (K, N) arrays."""
    import numpy as np

    target = cache_dir / f"{cache_key}.npz"
    tmp = cache_dir / f"{cache_key}.tmp.npz"
    np.savez(
        tmp,
        costs=np.asarray(costs, dtype=np.float64),
        fell=np.asarray(fell, dtype=bool),
        fall_time=np.asarray(fall_time, dtype=np.float64),
    )
    tmp.replace(target)
    return target


def load_candidate_cache(cache_dir: Path, cache_key: str, expected_shape: Tuple[int, int]) -> Optional[dict]:
    """Load cached (K, N) arrays if present and shape-valid; else None."""
    import numpy as np

    target = cache_dir / f"{cache_key}.npz"
    if not target.exists():
        return None
    with np.load(target) as data:
        costs = np.asarray(data["costs"], dtype=np.float64)
        fell = np.asarray(data["fell"], dtype=bool)
        fall_time = np.asarray(data["fall_time"], dtype=np.float64)
    if costs.shape != expected_shape or fell.shape != expected_shape or fall_time.shape != expected_shape:
        raise SystemExit(
            f"cached file {target} has shape {costs.shape}, expected {expected_shape}; "
            "remove the corrupted cache file or use a new --out directory."
        )
    return {"costs": costs, "fell": fell, "fall_time": fall_time}


_WORKER_STATE: dict = {}


def _init_worker(code_root_str: str, duration_s: float, seeds: Tuple[int, ...],
                 k_realizations: int, nudge_rad: float, priority: str) -> None:
    """Initialize one worker process with CPU-only, 1-thread BLAS, requested priority, and evaluators."""
    os.environ["ROBOT"] = "legobot"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    _configure_threads(1)
    _set_process_priority(priority)
    _add_code_paths(Path(code_root_str))
    import numpy as np
    from batch_tuning import BatchEvaluator
    from config import CONFIG

    robust = CONFIG.build_robustness()
    eval_cfg = replace(CONFIG.build_eval_config(validation=True), duration=float(duration_s))
    base_pitches = tuple(float(p) for p in eval_cfg.start_pitches)
    space = CONFIG.build_search_space()
    _WORKER_STATE.clear()
    _WORKER_STATE.update({
        "robust": robust,
        "eval_cfg": eval_cfg,
        "base_pitches": base_pitches,
        "space": space,
        "seeds": tuple(int(s) for s in seeds),
        "k_realizations": int(k_realizations),
        "nudge_rad": float(nudge_rad),
        "evs": {},  # (controller, spec_key) -> list[BatchEvaluator] of len K
        "BatchEvaluator": BatchEvaluator,
        "np": np,
    })


def _worker_score_chunk(task: dict) -> List[Tuple[str, object, object, object]]:
    """Score one chunk of same-kind, same-spec candidates across all K realizations in a worker."""
    st = _WORKER_STATE
    np = st["np"]
    controller = task["controller"]
    spec = task["spec"]
    keys = task["keys"]
    arr = np.asarray(task["params"], dtype=np.float64)
    ev_key = (controller, None if spec is None else repr(spec))
    evs = st["evs"].get(ev_key)
    if evs is None:
        BatchEvaluator = st["BatchEvaluator"]
        evs = [
            BatchEvaluator(
                replace(st["eval_cfg"],
                        start_pitches=realization_start_pitches(st["base_pitches"], rk, st["nudge_rad"])),
                st["space"],
                dtype=np.float64,
                backend="numpy",
                robustness=st["robust"],
                controller=controller,
                policy=spec,
            )
            for rk in range(st["k_realizations"])
        ]
        st["evs"][ev_key] = evs

    n_sc = len(st["seeds"]) * len(st["base_pitches"])
    K = st["k_realizations"]
    costs = np.empty((len(keys), K, n_sc), dtype=np.float64)
    fell = np.empty((len(keys), K, n_sc), dtype=bool)
    ft = np.empty((len(keys), K, n_sc), dtype=np.float64)
    for rk, ev in enumerate(evs):
        res = ev.evaluate_population(arr, st["seeds"]) if controller == "cascaded" else ev.evaluate_policies(arr, st["seeds"])
        costs[:, rk, :] = res.costs
        fell[:, rk, :] = res.fell
        ft[:, rk, :] = res.fall_time
    return [(keys[i], costs[i], fell[i], ft[i]) for i in range(len(keys))]


def score_all_candidates(
    candidates: Sequence[PolicyCandidate],
    eval_config,
    robust,
    seeds: Sequence[int],
    k_realizations: int = DEFAULT_K,
    nudge_rad: float = DEFAULT_NUDGE_RAD,
    cache_dir: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dedup: bool = True,
    evaluator_factory=None,
    workers: int = 1,
    code_root: Path = REPO,
    priority: str = "normal",
) -> Tuple[Dict[str, dict], dict]:
    """Score every candidate across K realizations in NumPy float64, with optional cache/resume, dedup,
    and multiprocessing (--workers > 1; bit-identical to single-process execution).

    Returns (scores_by_label, timing_info) where each scores_by_label[label] holds
    {"costs": (K, N), "fell": (K, N), "fall_time": (K, N)}.
    """
    import numpy as np

    seeds = tuple(int(s) for s in seeds)
    base_pitches = tuple(float(p) for p in eval_config.start_pitches)
    n_scenarios = len(seeds) * len(base_pitches)
    expected_shape = (k_realizations, n_scenarios)
    bs = max(1, int(batch_size))

    if cache_dir is not None:
        init_or_verify_cache(
            cache_dir,
            _manifest_payload(seeds, base_pitches, eval_config.duration, k_realizations, nudge_rad),
        )

    # Deduplicate simulation work by cache_key when dedup=True (bit-identical params -> bit-identical result).
    unique_items: List[Tuple[str, PolicyCandidate]] = []
    seen_keys = set()
    for idx, c in enumerate(candidates):
        key = c.cache_key if dedup else f"{idx:04d}_{c.cache_key}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_items.append((key, c))

    results_by_key: Dict[str, dict] = {}
    to_compute: List[Tuple[str, PolicyCandidate]] = []
    for key, c in unique_items:
        cached = load_candidate_cache(cache_dir, key, expected_shape) if cache_dir is not None else None
        if cached is not None:
            results_by_key[key] = cached
        else:
            to_compute.append((key, c))

    n_cached = len(results_by_key)
    n_to_compute = len(to_compute)
    t_start = time.perf_counter()

    if to_compute:
        cascaded_items = [(k, c) for k, c in to_compute if c.kind == "cascaded"]
        policy_groups: Dict[object, List[Tuple[str, PolicyCandidate]]] = {}
        for key, c in to_compute:
            if c.kind == "policy":
                policy_groups.setdefault(c.spec, []).append((key, c))

        if workers > 1 and evaluator_factory is None:
            tasks = []
            for start in range(0, len(cascaded_items), bs):
                chunk = cascaded_items[start:start + bs]
                tasks.append({
                    "controller": "cascaded",
                    "spec": None,
                    "keys": [k for k, _ in chunk],
                    "params": np.array([c.gains for _, c in chunk], dtype=np.float64),
                })
            for spec, group in policy_groups.items():
                for start in range(0, len(group), bs):
                    chunk = group[start:start + bs]
                    tasks.append({
                        "controller": "policy",
                        "spec": spec,
                        "keys": [k for k, _ in chunk],
                        "params": np.array([c.gains for _, c in chunk], dtype=np.float64),
                    })
            done_count = 0
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(workers),
                initializer=_init_worker,
                initargs=(str(Path(code_root).resolve()), float(eval_config.duration),
                          seeds, int(k_realizations), float(nudge_rad), str(priority)),
            ) as ex:
                futures = [ex.submit(_worker_score_chunk, t) for t in tasks]
                for fut in concurrent.futures.as_completed(futures):
                    chunk_out = fut.result()
                    for key, c_arr, f_arr, ft_arr in chunk_out:
                        results_by_key[key] = {"costs": c_arr, "fell": f_arr, "fall_time": ft_arr}
                        if cache_dir is not None:
                            save_candidate_cache(cache_dir, key, c_arr, f_arr, ft_arr)
                    done_count += len(chunk_out)
                    elapsed = time.perf_counter() - t_start
                    rate = done_count / elapsed if elapsed > 0 else 0.0
                    rem = (n_to_compute - done_count) / rate if rate > 0 else 0.0
                    print(
                        f"  scored {done_count}/{n_to_compute} unique candidates "
                        f"(+{n_cached} cached, {workers} workers) in {elapsed:.1f}s (est. {rem:.1f}s left)",
                        flush=True,
                    )
        else:
            if evaluator_factory is None:
                from batch_tuning import BatchEvaluator
                from config import CONFIG

                space = CONFIG.build_search_space()

                def evaluator_factory(cfg_k, controller: str, policy=None):
                    return BatchEvaluator(
                        cfg_k,
                        space,
                        dtype=np.float64,
                        backend="numpy",
                        robustness=robust,
                        controller=controller,
                        policy=policy,
                    )

            if cascaded_items:
                evs = [
                    evaluator_factory(
                        replace(eval_config, start_pitches=realization_start_pitches(base_pitches, rk, nudge_rad)),
                        "cascaded",
                        None,
                    )
                    for rk in range(k_realizations)
                ]
                for start in range(0, len(cascaded_items), bs):
                    chunk = cascaded_items[start:start + bs]
                    gains = np.array([c.gains for _, c in chunk], dtype=np.float64)
                    costs = np.empty((len(chunk), k_realizations, n_scenarios), dtype=np.float64)
                    fell = np.empty((len(chunk), k_realizations, n_scenarios), dtype=bool)
                    ft = np.empty((len(chunk), k_realizations, n_scenarios), dtype=np.float64)
                    for rk, ev in enumerate(evs):
                        res = ev.evaluate_population(gains, seeds)
                        costs[:, rk, :] = res.costs
                        fell[:, rk, :] = res.fell
                        ft[:, rk, :] = res.fall_time
                    for i, (key, _) in enumerate(chunk):
                        results_by_key[key] = {"costs": costs[i], "fell": fell[i], "fall_time": ft[i]}
                        if cache_dir is not None:
                            save_candidate_cache(cache_dir, key, costs[i], fell[i], ft[i])

            done_count = len(cascaded_items)
            for spec, group in policy_groups.items():
                evs = [
                    evaluator_factory(
                        replace(eval_config, start_pitches=realization_start_pitches(base_pitches, rk, nudge_rad)),
                        "policy",
                        spec,
                    )
                    for rk in range(k_realizations)
                ]
                for start in range(0, len(group), bs):
                    chunk = group[start:start + bs]
                    params = np.array([c.gains for _, c in chunk], dtype=np.float64)
                    costs = np.empty((len(chunk), k_realizations, n_scenarios), dtype=np.float64)
                    fell = np.empty((len(chunk), k_realizations, n_scenarios), dtype=bool)
                    ft = np.empty((len(chunk), k_realizations, n_scenarios), dtype=np.float64)
                    for rk, ev in enumerate(evs):
                        res = ev.evaluate_policies(params, seeds)
                        costs[:, rk, :] = res.costs
                        fell[:, rk, :] = res.fell
                        ft[:, rk, :] = res.fall_time
                    for i, (key, _) in enumerate(chunk):
                        results_by_key[key] = {"costs": costs[i], "fell": fell[i], "fall_time": ft[i]}
                        if cache_dir is not None:
                            save_candidate_cache(cache_dir, key, costs[i], fell[i], ft[i])
                    done_count += len(chunk)
                    elapsed = time.perf_counter() - t_start
                    rate = done_count / elapsed if elapsed > 0 else 0.0
                    rem = (n_to_compute - done_count) / rate if rate > 0 else 0.0
                    print(
                        f"  scored {done_count}/{n_to_compute} unique candidates "
                        f"(+{n_cached} cached) in {elapsed:.1f}s (est. {rem:.1f}s left)",
                        flush=True,
                    )

    elapsed_total = time.perf_counter() - t_start
    scores_by_label: Dict[str, dict] = {}
    for idx, c in enumerate(candidates):
        key = c.cache_key if dedup else f"{idx:04d}_{c.cache_key}"
        scores_by_label[c.label] = results_by_key[key]

    timing = {
        "n_candidates_total": len(candidates),
        "n_unique_candidates": len(unique_items),
        "n_loaded_from_cache": n_cached,
        "n_computed_now": n_to_compute,
        "workers": int(workers),
        "elapsed_s": round(elapsed_total, 3),
        "simulations_computed_now": int(n_to_compute * k_realizations * n_scenarios),
    }
    return scores_by_label, timing


# ---- pre-registered statistics & ranking ---------------------------------------------------------

def summarize_candidate(costs_kn, fell_kn, mb_costs_kn, mb_fell_kn) -> dict:
    """Compute the pre-registered K-realization metrics for one candidate against MB PID.

    costs_kn, fell_kn, mb_costs_kn, mb_fell_kn all have shape (K, N).
    """
    import numpy as np

    costs = np.asarray(costs_kn, dtype=np.float64)
    fell = np.asarray(fell_kn, dtype=bool)
    mb_costs = np.asarray(mb_costs_kn, dtype=np.float64)
    mb_fell = np.asarray(mb_fell_kn, dtype=bool)
    if costs.ndim != 2 or costs.shape != fell.shape or costs.shape != mb_fell.shape:
        raise ValueError(f"shape mismatch: costs={costs.shape}, fell={fell.shape}, mb_fell={mb_fell.shape}")

    K, N = costs.shape
    per_real_mean = [float(costs[k].mean()) for k in range(K)]
    mean_of_means = float(np.mean(per_real_mean))
    median_cost = float(np.median(costs))
    falls_per_real = [int(fell[k].sum()) for k in range(K)]
    total_falls = int(fell.sum())
    total_scenarios = int(K * N)

    both_up_counts: List[int] = []
    per_real_both_up: List[Optional[float]] = []
    per_real_mb_both_up: List[Optional[float]] = []
    for k in range(K):
        up = (~fell[k]) & (~mb_fell[k])
        cnt = int(up.sum())
        both_up_counts.append(cnt)
        if cnt > 0:
            per_real_both_up.append(float(costs[k, up].mean()))
            per_real_mb_both_up.append(float(mb_costs[k, up].mean()))
        else:
            per_real_both_up.append(None)
            per_real_mb_both_up.append(None)

    if all(v is not None for v in per_real_both_up):
        both_up_cost: Optional[float] = float(np.mean(per_real_both_up))
        mb_both_up_cost: Optional[float] = float(np.mean(per_real_mb_both_up))
        both_up_ratio_vs_mb: Optional[float] = (
            float(both_up_cost / mb_both_up_cost) if mb_both_up_cost and mb_both_up_cost > 0 else None
        )
    else:
        both_up_cost = None
        mb_both_up_cost = None
        both_up_ratio_vs_mb = None

    return {
        "mean_of_means": mean_of_means,
        "mean_of_means_3dp": f"{mean_of_means:.3f}",
        "per_realization_mean": per_real_mean,
        "median_cost": median_cost,
        "both_up_cost": both_up_cost,
        "per_realization_both_up_cost": per_real_both_up,
        "per_realization_both_up_counts": both_up_counts,
        "mb_both_up_cost_on_shared": mb_both_up_cost,
        "both_up_ratio_vs_mb": both_up_ratio_vs_mb,
        "total_falls": total_falls,
        "total_scenarios": total_scenarios,
        "fall_rate": float(total_falls / total_scenarios) if total_scenarios > 0 else 0.0,
        "falls_per_realization": falls_per_real,
        "k_realizations": K,
        "n_scenarios_per_realization": N,
    }


def selection_sort_key(stats: dict, label: str, is_final: bool = False) -> Tuple:
    """Pre-registered selection key:
    1. mean over realizations of mean dev cost, rounded to the reported 3-decimal precision;
    2. ties at 3 decimals -> fewer total falls;
    3. deterministic secondary tie-breaks: exact unrounded mean_of_means, median_cost,
       final.npz before its identical checkpoint copy, then label.
    """
    m = float(stats["mean_of_means"])
    m_3dp = Decimal(f"{m:.3f}")
    return (
        m_3dp,
        int(stats["total_falls"]),
        m,
        float(stats["median_cost"]),
        0 if is_final else 1,
        label,
    )


def rank_candidates(stats_by_label: Dict[str, dict],
                    meta_by_label: Optional[Dict[str, dict]] = None) -> List[str]:
    """Rank candidate labels using the pre-registered selection key (lowest first)."""
    meta_by_label = meta_by_label or {}
    return sorted(
        stats_by_label.keys(),
        key=lambda lbl: selection_sort_key(
            stats_by_label[lbl],
            lbl,
            is_final=bool(meta_by_label.get(lbl, {}).get("is_final", False)),
        ),
    )


def summarize_all(candidates: Sequence[PolicyCandidate], scores_by_label: Dict[str, dict],
                  runs: Sequence[LRun]) -> dict:
    """Compute per-candidate stats, overall ranking, selected L candidate, per-run bests, and machine parity."""
    mb_score = scores_by_label[MB]
    stats_by_label: Dict[str, dict] = {}
    meta_by_label: Dict[str, dict] = {}
    for c in candidates:
        s = scores_by_label[c.label]
        stats_by_label[c.label] = summarize_candidate(
            s["costs"], s["fell"], mb_score["costs"], mb_score["fell"]
        )
        meta_by_label[c.label] = c.meta()

    ranking_all = rank_candidates(stats_by_label, meta_by_label)
    l_labels = [c.label for c in candidates if c.kind == "policy"]
    ranking_l = [lbl for lbl in ranking_all if lbl in set(l_labels)]
    selected_l = ranking_l[0] if ranking_l else None

    per_run_best: Dict[str, dict] = {}
    run_by_seed = {r.seed: r for r in runs}
    for r in runs:
        mine = [lbl for lbl in ranking_l if meta_by_label[lbl]["run_seed"] == r.seed]
        if mine:
            best_lbl = mine[0]
            final_lbl = f"L / s{r.seed} / final"
            per_run_best[str(r.seed)] = {
                "run_seed": r.seed,
                "folder": r.name,
                "hostname": r.hostname,
                "gpu": r.gpu,
                "stop_reason": r.stop_reason,
                "best_label": best_lbl,
                "best_checkpoint_rel": meta_by_label[best_lbl]["checkpoint_rel"],
                "best_matches_final": meta_by_label[best_lbl]["matches_final"],
                "best_stats": stats_by_label[best_lbl],
                "final_label": final_lbl,
                "final_stats": stats_by_label.get(final_lbl),
            }

    # Machine & even/odd parity disclosure (E2_NOTES.md lines 303, 321, 342-343)
    by_host: Dict[str, List[int]] = {}
    by_parity: Dict[str, List[int]] = {"even": [], "odd": []}
    for r in runs:
        host = r.hostname or "unknown"
        by_host.setdefault(host, []).append(r.seed)
        by_parity["even" if r.seed % 2 == 0 else "odd"].append(r.seed)

    def _group_summary(seed_list: Sequence[int]) -> dict:
        bests = [per_run_best[str(s)]["best_stats"]["mean_of_means"] for s in seed_list if str(s) in per_run_best]
        finals = [per_run_best[str(s)]["final_stats"]["mean_of_means"] for s in seed_list
                  if str(s) in per_run_best and per_run_best[str(s)]["final_stats"]]
        import numpy as np
        return {
            "run_seeds": list(seed_list),
            "n_runs": len(seed_list),
            "best_mean_of_means_median": float(np.median(bests)) if bests else None,
            "best_mean_of_means_min": float(min(bests)) if bests else None,
            "best_mean_of_means_max": float(max(bests)) if bests else None,
            "final_mean_of_means_median": float(np.median(finals)) if finals else None,
        }

    parity = {
        "by_hostname": {host: _group_summary(ss) for host, ss in sorted(by_host.items())},
        "by_seed_parity": {k: _group_summary(ss) for k, ss in by_parity.items()},
        "per_seed_machine": {
            str(r.seed): {"hostname": r.hostname, "gpu": r.gpu, "stop_reason": r.stop_reason}
            for r in sorted(run_by_seed.values(), key=lambda x: x.seed)
        },
    }

    return {
        "stats": stats_by_label,
        "ranking_all": ranking_all,
        "ranking_l": ranking_l,
        "selected": {
            "label": selected_l,
            "meta": meta_by_label[selected_l] if selected_l else None,
            "stats": stats_by_label[selected_l] if selected_l else None,
            "beats_mb_pid": (
                bool(selection_sort_key(stats_by_label[selected_l], selected_l,
                                        meta_by_label[selected_l]["is_final"])
                     < selection_sort_key(stats_by_label[MB], MB))
                if selected_l else False
            ),
        },
        "per_run_best": per_run_best,
        "parity": parity,
    }


# ---- output writers ------------------------------------------------------------------------------

def write_per_scenario_csv(path: Path, candidates: Sequence[PolicyCandidate], scores_by_label: Dict[str, dict],
                           seeds: Sequence[int], base_pitches: Sequence[float],
                           k_realizations: int, nudge_rad: float) -> None:
    """Write every (candidate, realization, seed, start_pitch) outcome to CSV."""
    import numpy as np

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "controller", "run_seed", "checkpoint", "realization", "nudge_rad",
            "seed", "start_pitch", "cost", "fell", "fall_time_s",
        ])
        for c in candidates:
            s = scores_by_label[c.label]
            for rk in range(k_realizations):
                pitches_k = realization_start_pitches(base_pitches, rk, nudge_rad)
                scenarios_k = [(seed, p) for seed in seeds for p in pitches_k]
                nudge = float(rk) * float(nudge_rad)
                for (seed, pitch), cost, fell, ft in zip(
                    scenarios_k, s["costs"][rk], s["fell"][rk], s["fall_time"][rk]
                ):
                    w.writerow([
                        c.label,
                        "" if c.run_seed is None else c.run_seed,
                        c.checkpoint_rel or "baseline",
                        rk,
                        nudge,
                        seed,
                        pitch,
                        float(cost),
                        int(fell),
                        "" if np.isnan(ft) else float(ft),
                    ])


def _fmt_opt(val: Optional[float], fmt: str = ".3f") -> str:
    return "—" if val is None else format(val, fmt)


def render_markdown(selection: dict, top_n: int = 25) -> str:
    """Render E2_NOTES.md-ready Markdown tables."""
    st = selection["settings"]
    runs = selection["runs"]
    skipped = selection["skipped_runs"]
    stats = selection["results"]["stats"]
    meta_by_lbl = {c["label"]: c for c in selection["candidates"]}
    per_run = selection["results"]["per_run_best"]
    sel = selection["results"]["selected"]
    parity = selection["results"]["parity"]

    lines: List[str] = []
    if st["dry_run"]:
        lines.append(
            f"> DRY RUN (K={st['k_realizations']}, {st['n_seeds']} dev seeds, "
            f"{st['duration_s']:g} s): not E2.6's final selection.\n"
        )
    if skipped:
        lines.append(
            f"> PARTIAL RUN SET ({len(runs)} finished, {len(skipped)} skipped/missing: "
            + ", ".join(f"s{r['run_seed']}={r['status']}" for r in skipped) + ").\n"
        )

    lines += [
        f"### E2.6 L run provenance ({len(runs)} finished runs; K = {st['k_realizations']} realizations, "
        f"N = {st['n_scenarios_per_realization']} dev scenarios, {st['duration_s']:g} s, NumPy float64)",
        "",
        "| Run seed | Folder | Host | GPU | Commit | Dirty | Stop reason | Gens | Robot-s | Selected (in-run) | Ckpts |",
        "|---:|---|---|---|---|---|---|---:|---:|---|---:|",
    ]
    for r in runs:
        rs = "—" if r["robot_seconds"] is None else f"{r['robot_seconds']:.3e}"
        lines.append(
            f"| {r['run_seed']} | `{r['folder']}` | {r['hostname'] or '—'} | {r['gpu'] or '—'} | "
            f"`{str(r['git_commit'])[:7]}` | {r['git_dirty_reading']} | {r['stop_reason'] or '—'} | "
            f"{r['generations'] if r['generations'] is not None else '—'} | {rs} | "
            f"`{r['selected_file'] or '—'}` | {r['n_checkpoints']} |"
        )

    lines += [
        "",
        "### E2.6 per-run best candidate (plus frozen MB PID baseline)",
        "",
        "| Seed | Host | Best candidate | Matches final | Mean-of-means | Median (KxN) | Both-up | Falls/(KxN) | Falls/realization |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    mb_s = stats[MB]
    lines.append(
        f"| — | — | `{MB}` | — | {mb_s['mean_of_means']:.3f} | {mb_s['median_cost']:.3f} | "
        f"{_fmt_opt(mb_s['both_up_cost'])} | {mb_s['total_falls']}/{mb_s['total_scenarios']} | "
        f"`{mb_s['falls_per_realization']}` |"
    )
    for seed_str in sorted(per_run.keys(), key=lambda x: int(x)):
        pr = per_run[seed_str]
        bs = pr["best_stats"]
        lines.append(
            f"| {pr['run_seed']} | {pr['hostname'] or '—'} | `{pr['best_label']}` | "
            f"{'yes' if pr['best_matches_final'] else 'no'} | {bs['mean_of_means']:.3f} | "
            f"{bs['median_cost']:.3f} | {_fmt_opt(bs['both_up_cost'])} | "
            f"{bs['total_falls']}/{bs['total_scenarios']} | `{bs['falls_per_realization']}` |"
        )

    lines += [
        "",
        f"### E2.6 top {top_n} candidates overall (plus `{MB}`)",
        "",
        "| Rank | Candidate | Run seed | Mean-of-means | Median (KxN) | Both-up | Falls/(KxN) | Falls/realization |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    ranking = selection["results"]["ranking_all"]
    shown = list(ranking[:top_n])
    if MB not in shown and MB in stats:
        shown.append(MB)
    for lbl in shown:
        rank = ranking.index(lbl) + 1
        s = stats[lbl]
        rs = meta_by_lbl[lbl]["run_seed"]
        lines.append(
            f"| {rank} | `{lbl}` | {'—' if rs is None else rs} | {s['mean_of_means']:.3f} | "
            f"{s['median_cost']:.3f} | {_fmt_opt(s['both_up_cost'])} | "
            f"{s['total_falls']}/{s['total_scenarios']} | `{s['falls_per_realization']}` |"
        )

    lines += [
        "",
        "### Machine & seed-parity disclosure",
        "",
        "| Group | Seeds | Runs | Best mean-of-means median | Range |",
        "|---|---|---:|---:|---|",
    ]
    for host, gs in parity["by_hostname"].items():
        rng_str = (
            "—" if gs["best_mean_of_means_min"] is None
            else f"{gs['best_mean_of_means_min']:.3f}–{gs['best_mean_of_means_max']:.3f}"
        )
        lines.append(
            f"| host `{host}` | {gs['run_seeds']} | {gs['n_runs']} | "
            f"{_fmt_opt(gs['best_mean_of_means_median'])} | {rng_str} |"
        )
    for par, gs in parity["by_seed_parity"].items():
        rng_str = (
            "—" if gs["best_mean_of_means_min"] is None
            else f"{gs['best_mean_of_means_min']:.3f}–{gs['best_mean_of_means_max']:.3f}"
        )
        lines.append(
            f"| {par} seeds | {gs['run_seeds']} | {gs['n_runs']} | "
            f"{_fmt_opt(gs['best_mean_of_means_median'])} | {rng_str} |"
        )

    if sel["label"]:
        ss = sel["stats"]
        lines += [
            "",
            f"**Selected E2.6 L candidate:** `{sel['label']}` "
            f"(mean-of-means {ss['mean_of_means']:.3f}, median {ss['median_cost']:.3f}, "
            f"both-up {_fmt_opt(ss['both_up_cost'])}, falls {ss['total_falls']}/{ss['total_scenarios']} "
            f"{ss['falls_per_realization']}).",
        ]
    lines += ["", f"*{PREREG_SPEC}*"]
    return "\n".join(lines) + "\n"


# ---- CLI -----------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                    help=f"directory containing e2-seed0..9 folders (default: {DEFAULT_RUNS_ROOT})")
    ap.add_argument("--run-prefix", default="e2-seed", help="run folder prefix (default: e2-seed)")
    ap.add_argument("--run-seeds", type=str, default="0-9",
                    help="run seeds to include, e.g. '0-9' or '0,1,3,5' (default: 0-9)")
    ap.add_argument("--expected-commit", default=DEFAULT_EXPECTED_COMMIT,
                    help=f"git commit the L runs were trained at (default: {DEFAULT_EXPECTED_COMMIT})")
    ap.add_argument("--code-root", type=Path, default=REPO,
                    help="code tree to import simulation/config from and check git diff against "
                         "(e.g. C:\\balancebot_training\\e2_L_worktree)")
    ap.add_argument("--allow-code-drift", action="store_true",
                    help="allow scoring even if git diff against --expected-commit is non-empty")
    ap.add_argument("--require-all", action="store_true",
                    help="refuse to run if any expected run seed is missing/unfinished or has provenance flags")
    ap.add_argument("--list-runs", action="store_true",
                    help="inventory runs & candidates only; score nothing")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"number of tilt-nudge realizations (default: {DEFAULT_K}; != {DEFAULT_K} marks dry run)")
    ap.add_argument("--nudge-rad", type=float, default=DEFAULT_NUDGE_RAD,
                    help=f"initial-tilt increment per realization in rad (default: {DEFAULT_NUDGE_RAD:g})")
    ap.add_argument("--seeds", type=int, default=None,
                    help="dry run: only the first N dev seeds (default: all 50 dev seeds 5000-5049)")
    ap.add_argument("--duration", type=float, default=None,
                    help="dry run: scenario duration in s (default: 60 s)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"unique policy candidates scored per NumPy batch & cache flush (default: {DEFAULT_BATCH_SIZE})")
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes for parallel batch scoring (default: 1; leave >=4 logical CPUs free)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="disable parameter-hash deduplication across identical checkpoints/final.npz")
    ap.add_argument("--threads", type=int, default=None,
                    help="cap OMP/MKL/OPENBLAS/NUMEXPR thread count")
    ap.add_argument("--priority", choices=("normal", "below-normal", "idle"), default="normal",
                    help="Windows process priority (default: normal; use 'idle' when training queues run at BelowNormal)")
    ap.add_argument("--below-normal", action="store_true",
                    help="alias for --priority below-normal")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: results/legobot/e2_selection/<stamp>-e26-k5)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume an interrupted re-score in this output directory")
    return ap.parse_args(argv)


def _parse_seed_list(spec: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(dict.fromkeys(out))


def main(argv=None) -> Optional[Path]:
    args = parse_args(argv)
    priority = "below-normal" if args.below_normal and args.priority == "normal" else args.priority
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    _configure_threads(args.threads if args.workers == 1 else 1)
    _set_process_priority(priority)

    os.environ["ROBOT"] = "legobot"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    code_root = Path(args.code_root).resolve()
    code_id = check_code_identity(code_root, args.expected_commit)

    # Enforce code identity before importing/scoring (allowed to proceed with --list-runs for inspection).
    if code_id["has_drift"] and not args.list_runs:
        drift_summary = (
            code_id["error"]
            or f"committed diff ({len(code_id['committed_diff_files'])} files), "
               f"worktree diff ({len(code_id['worktree_diff_files'])} files), "
               f"untracked ({len(code_id['untracked_files'])} files)"
        )
        if not args.allow_code_drift:
            detail_lines = []
            if code_id["error"]:
                detail_lines.append(f"  git error: {code_id['error']}")
            if code_id["worktree_diff_files"]:
                detail_lines.append("  diff vs " + args.expected_commit + ":\n    " +
                                    "\n    ".join(code_id["worktree_diff_files"][:25]))
            if code_id["untracked_files"]:
                detail_lines.append("  untracked:\n    " + "\n    ".join(code_id["untracked_files"][:25]))
            raise SystemExit(
                f"REFUSING to score: code drift detected in {code_root} against {args.expected_commit} "
                f"({drift_summary}).\n" + "\n".join(detail_lines) + "\n"
                "Run with --code-root C:\\balancebot_training\\e2_L_worktree (pinned tree) "
                "or pass --allow-code-drift to override explicitly."
            )
        print(f"WARNING: --allow-code-drift set; proceeding despite code drift ({drift_summary})", flush=True)

    _add_code_paths(code_root)
    import numpy as np
    from balance_sim import load_policy
    from batch_tuning import first_train_seeds
    from compare_controllers import tier_seeds
    from config import CONFIG, load_config
    from tuning import provenance

    if CONFIG.name != "legobot":
        raise SystemExit(f"config was already loaded for {CONFIG.name}; run this script with ROBOT=legobot")
    cfg = load_config("legobot")

    # Verify dev tier and disjointness from every frozen / held-out / training tier (matches e2_select.py).
    robust = CONFIG.build_robustness()
    dev_seeds = tier_seeds(robust, "dev")
    assert dev_seeds == tuple(range(5000, 5050)), f"unexpected dev seeds: {dev_seeds[:5]}..{dev_seeds[-5:]}"
    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if args.seeds is not None:
        if not (1 <= args.seeds <= len(dev_seeds)):
            raise SystemExit(f"--seeds must be between 1 and {len(dev_seeds)}")
        dev_seeds = dev_seeds[:args.seeds]
    dry = bool(args.seeds is not None or args.duration is not None or args.k != DEFAULT_K)

    held = (
        set(robust.validation_seeds)
        | set(robust.fall_rate_seeds)
        | set(robust.confirm_seeds)
        | set(robust.retired_seeds)
        | set(robust.dev_extended_seeds)
        | set(first_train_seeds(DEFAULT_TRAIN_SEEDS, robust.held_out_seeds))
    )
    assert not (set(dev_seeds) & held), "a scored seed lies in a frozen, confirm, retired, or training tier"

    eval_config = CONFIG.build_eval_config(validation=True)
    if args.duration is not None:
        eval_config = replace(eval_config, duration=args.duration)
    base_pitches = tuple(float(p) for p in eval_config.start_pitches)

    # 1. Collect runs
    run_seeds = _parse_seed_list(args.run_seeds)
    runs, skipped = collect_l_runs(
        args.runs_root, run_seeds=run_seeds, prefix=args.run_prefix,
        expected_commit=args.expected_commit, expected_train_seeds=DEFAULT_TRAIN_SEEDS, cfg=cfg,
    )
    print(f"E2.6 L runs under {args.runs_root}: {len(runs)} finished, {len(skipped)} missing/unfinished")
    print(f"{'seed':>4}  {'folder':<12}{'status':<12}{'host':<14}{'commit':<9}{'dirty':<7}{'ckpts':>5}  stop_reason")
    for r in sorted([*runs, *skipped], key=lambda x: x.seed):
        print(
            f"{r.seed:>4}  {r.name:<12}{r.status:<12}{str(r.hostname or '—'):<14}"
            f"{str(r.commit or '—')[:7]:<9}{r.dirty_reading:<7}{r.n_checkpoints:>5}  "
            f"{r.stop_reason or '; '.join(r.notes) or '—'}"
            + (f"  FLAGS: {'; '.join(r.flags)}" if r.flags else "")
        )
    for r in skipped:
        print(f"WARNING: skipping {r.name} ({r.status}): {'; '.join(r.notes) or '; '.join(r.flags)}", flush=True)

    problems: List[str] = []
    for r in skipped:
        problems.append(f"{r.name}: {r.status}")
    for r in runs:
        for fl in r.flags:
            problems.append(f"{r.name}: {fl}")

    # 2. Collect candidates
    mb_gains = tuple(map(float, CONFIG.reference_pid_gains()))
    candidates = collect_candidates(runs, mb_gains, load_policy, expected_max_torque=CONFIG.robot.max_torque)
    unique_keys = {c.cache_key for c in candidates}
    print(
        f"Candidates: {len(candidates)} total (1 MB PID + {len(candidates) - 1} L policies; "
        f"{len(unique_keys)} unique parameter vectors)"
    )

    if args.list_runs:
        print("--list-runs: inventory only, nothing scored")
        return None
    if args.require_all and problems:
        raise SystemExit(
            "REFUSING (--require-all): run set is incomplete or has provenance flags:\n  "
            + "\n  ".join(problems)
        )
    if not runs:
        raise SystemExit("no finished E2 L runs to score")

    # 3. Output / resume folder
    if args.resume is not None and args.out is not None and args.resume.resolve() != args.out.resolve():
        raise SystemExit("pass either --resume or --out (or the same directory for both)")
    if args.resume is not None:
        out_dir = args.resume
    elif args.out is not None:
        out_dir = args.out
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S") + "-e26-k5" + ("-dry" if dry else "")
        out_dir = REPO / "results" / "legobot" / "e2_selection" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"

    # 4. Score K realizations
    print(
        f"\nScoring {len(candidates)} candidates ({len(unique_keys)} unique) x K={args.k} realizations "
        f"x {len(dev_seeds) * len(base_pitches)} dev scenarios x {eval_config.duration:g} s "
        f"(numpy float64, CUDA_VISIBLE_DEVICES=-1) ...",
        flush=True,
    )
    scores_by_label, timing = score_all_candidates(
        candidates,
        eval_config,
        robust,
        dev_seeds,
        k_realizations=args.k,
        nudge_rad=args.nudge_rad,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        dedup=not args.no_dedup,
        workers=args.workers,
        code_root=code_root,
        priority=priority,
    )

    # 5. Summarize & rank
    results = summarize_all(candidates, scores_by_label, runs)
    sel_info = results["selected"]
    print(f"\nFinished scoring in {timing['elapsed_s']:.1f}s "
          f"(computed {timing['n_computed_now']}, cached {timing['n_loaded_from_cache']}).")
    if sel_info["label"]:
        s = sel_info["stats"]
        print(
            f"Selected L candidate: {sel_info['label']}  mean-of-means={s['mean_of_means']:.3f}  "
            f"median={s['median_cost']:.3f}  both-up={_fmt_opt(s['both_up_cost'])}  "
            f"falls={s['total_falls']}/{s['total_scenarios']} {s['falls_per_realization']}"
        )
    mb_s = results["stats"][MB]
    print(
        f"Baseline MB PID:      {MB}  mean-of-means={mb_s['mean_of_means']:.3f}  "
        f"median={mb_s['median_cost']:.3f}  both-up={_fmt_opt(mb_s['both_up_cost'])}  "
        f"falls={mb_s['total_falls']}/{mb_s['total_scenarios']} {mb_s['falls_per_realization']}"
    )

    # 6. Write outputs
    selection = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec": PREREG_SPEC,
        "provenance": provenance(),
        "code_identity": {**code_id, "allow_code_drift": bool(args.allow_code_drift)},
        "settings": {
            "runs_root": str(args.runs_root),
            "run_seeds_requested": list(run_seeds),
            "expected_commit": args.expected_commit,
            "require_all": bool(args.require_all),
            "backend": "numpy",
            "dtype": "float64",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "k_realizations": int(args.k),
            "nudge_rad": float(args.nudge_rad),
            "seeds": [int(dev_seeds[0]), int(dev_seeds[-1])],
            "n_seeds": len(dev_seeds),
            "start_pitches_rad": list(base_pitches),
            "n_scenarios_per_realization": len(dev_seeds) * len(base_pitches),
            "duration_s": float(eval_config.duration),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "priority": priority,
            "dedup": not args.no_dedup,
            "dry_run": dry,
            "argv": sys.argv[1:] if argv is None else list(argv),
        },
        "timing": timing,
        "runs": [r.summary() for r in runs],
        "skipped_runs": [r.summary() for r in skipped],
        "problems": problems,
        "baseline_gains": {MB: list(mb_gains)},
        "candidates": [c.meta() for c in candidates],
        "results": results,
        "config": CONFIG.snapshot(),
    }

    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=float) + "\n")
    write_per_scenario_csv(
        out_dir / "dev_per_scenario.csv", candidates, scores_by_label,
        dev_seeds, base_pitches, args.k, args.nudge_rad,
    )
    (out_dir / "e2_notes_tables.md").write_text(render_markdown(selection), encoding="utf-8")
    print(f"saved {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
