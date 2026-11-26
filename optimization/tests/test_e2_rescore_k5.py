"""Tests for hardware/legobot/e2_rescore_k5.py:
- K-realization initial-tilt offsets (k=0 plain dev tier, k=1..4 exact k*1e-15 rad shift)
- both-up cost logic (per-realization mask where candidate AND MB PID stay up, averaged over K;
  also handles realizations with zero both-up scenarios)
- pre-registered selection key & tie-break (mean-of-means rounded to 3 decimals -> fewer total falls)
- run collection (finished vs missing vs in-progress, provenance flags, --require-all)
- code-identity drift guard (refuses when diff non-empty unless --allow-code-drift)
- per-candidate resume cache (interrupted run resumes without re-scoring cached candidates,
  produces identical results, and rejects mismatched manifests)
- tiny end-to-end subprocess test on synthetic runs with real NumPy float64 simulation.
"""

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "simulation", REPO / "optimization", REPO / "hardware" / "legobot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import e2_rescore_k5 as rk5  # noqa: E402


# ---- 1. Realization initial-tilt offsets ---------------------------------------------------------

def test_realization_start_pitches_k0_is_exact_and_k_adds_k_times_nudge():
    base = (-0.15, 0.05, 0.15)
    p0 = rk5.realization_start_pitches(base, 0)
    assert p0 == base
    for k in range(1, 5):
        pk = rk5.realization_start_pitches(base, k)
        assert len(pk) == len(base)
        for b, shifted in zip(base, pk):
            assert shifted != b
            assert shifted - b == pytest.approx(k * 1e-15, rel=0.05, abs=5e-17)
    # Strictly increasing across k=0..4 for every pitch
    for i in range(len(base)):
        vals = [rk5.realization_start_pitches(base, k)[i] for k in range(5)]
        assert vals == sorted(vals) and len(set(vals)) == 5


# ---- 2. Both-up logic & K-realization summary ----------------------------------------------------

def test_summarize_candidate_both_up_per_realization_then_averaged():
    # K = 2 realizations, N = 4 scenarios each.
    # Realization 0:
    #   MB fell:   [False, True,  False, False]  -> up at idx 0, 2, 3
    #   Cand fell: [False, False, True,  False]  -> up at idx 0, 1, 3
    #   Both up:   idx 0, 3 -> cand costs [10.0, 20.0] -> mean 15.0 (MB [30.0, 50.0] -> mean 40.0)
    # Realization 1:
    #   MB fell:   [False, False, False, True]   -> up at idx 0, 1, 2
    #   Cand fell: [False, False, False, False]  -> up at idx 0, 1, 2, 3
    #   Both up:   idx 0, 1, 2 -> cand costs [30.0, 30.0, 60.0] -> mean 40.0 (MB [60.0, 60.0, 60.0] -> mean 60.0)
    # Note: pooled average over all 5 both-up scenarios would be (10+20+30+30+60)/5 = 30.0,
    # whereas per-realization mean then averaged over realizations is (15.0 + 40.0) / 2 = 27.5!
    mb_costs = np.array([[30.0, 1500.0, 40.0, 50.0], [60.0, 60.0, 60.0, 1400.0]])
    mb_fell = np.array([[False, True, False, False], [False, False, False, True]])
    cand_costs = np.array([[10.0, 12.0, 1200.0, 20.0], [30.0, 30.0, 60.0, 70.0]])
    cand_fell = np.array([[False, False, True, False], [False, False, False, False]])

    s = rk5.summarize_candidate(cand_costs, cand_fell, mb_costs, mb_fell)
    assert s["per_realization_mean"] == pytest.approx([310.5, 47.5])
    assert s["mean_of_means"] == pytest.approx(179.0)
    assert s["median_cost"] == pytest.approx(np.median(cand_costs))
    assert s["per_realization_both_up_counts"] == [2, 3]
    assert s["per_realization_both_up_cost"] == pytest.approx([15.0, 40.0])
    assert s["both_up_cost"] == pytest.approx(27.5)
    assert s["mb_both_up_cost_on_shared"] == pytest.approx(50.0)
    assert s["both_up_ratio_vs_mb"] == pytest.approx(27.5 / 50.0)
    assert s["total_falls"] == 1
    assert s["total_scenarios"] == 8
    assert s["fall_rate"] == pytest.approx(1 / 8)
    assert s["falls_per_realization"] == [1, 0]


def test_summarize_candidate_handles_zero_both_up_realization():
    mb_costs = np.array([[20.0, 20.0], [20.0, 20.0]])
    mb_fell = np.array([[False, False], [False, False]])
    cand_costs = np.array([[1500.0, 1500.0], [10.0, 20.0]])
    cand_fell = np.array([[True, True], [False, False]])

    s = rk5.summarize_candidate(cand_costs, cand_fell, mb_costs, mb_fell)
    assert s["per_realization_both_up_counts"] == [0, 2]
    assert s["per_realization_both_up_cost"] == [None, pytest.approx(15.0)]
    assert s["both_up_cost"] is None
    assert s["both_up_ratio_vs_mb"] is None
    assert s["total_falls"] == 2
    assert s["falls_per_realization"] == [2, 0]


# ---- 3. Selection key & 3-decimal tie-break ------------------------------------------------------

def test_tie_break_at_3_decimal_precision_prefers_fewer_falls():
    # A and B both round to 19.812 at 3 decimals; B has slightly lower unrounded mean (19.8121 vs 19.8124)
    # but MORE falls (2 vs 1), so A MUST win under the pre-registered 3-decimal tie rule.
    # C rounds to 19.811 with 5 falls -> NOT tied at 3 decimals -> C beats both A and B.
    # D ties A on 3dp (19.812) AND on falls (1), with lower exact mean (19.8122 < 19.8124) -> D beats A.
    stats = {
        "A": {"mean_of_means": 19.8124, "total_falls": 1, "median_cost": 19.5},
        "B": {"mean_of_means": 19.8121, "total_falls": 2, "median_cost": 19.0},
        "C": {"mean_of_means": 19.8114, "total_falls": 5, "median_cost": 20.0},
        "D": {"mean_of_means": 19.8122, "total_falls": 1, "median_cost": 19.6},
    }
    ranked = rk5.rank_candidates(stats)
    assert ranked == ["C", "D", "A", "B"]


def test_exact_tie_prefers_final_over_identical_checkpoint():
    stats = {
        "L / s1 / mean_8.986e08": {"mean_of_means": 19.806, "total_falls": 0, "median_cost": 19.6},
        "L / s1 / final": {"mean_of_means": 19.806, "total_falls": 0, "median_cost": 19.6},
    }
    meta = {
        "L / s1 / mean_8.986e08": {"is_final": False},
        "L / s1 / final": {"is_final": True},
    }
    assert rk5.rank_candidates(stats, meta) == ["L / s1 / final", "L / s1 / mean_8.986e08"]


# ---- 4. Synthetic run collection & missing/in-progress handling ----------------------------------

def _make_fake_run(root: Path, seed: int, *, finished: bool = True, host: str = "Evans-G16",
                   gpu: str = "RTX 5070 Ti", commit: str = "4582fe7386c5", dirty: bool = False,
                   stop_reason: str = "budget", n_ckpts: int = 2, spec=None,
                   base_params: np.ndarray = None) -> Path:
    from balance_sim import save_policy
    from config import load_config

    cfg = load_config("legobot")
    spec = spec or cfg.build_policy_spec("linear")
    base_params = np.zeros(spec.n_params) if base_params is None else np.asarray(base_params, float)

    folder = root / f"e2-seed{seed}"
    ck_dir = folder / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    snap = {
        "config": cfg.snapshot(),
        "args": {"method": "linear", "profile": "full", "run_seed": seed, "float64": True, "train_seeds": 150},
        "train_dtype": "float64",
        "train_seeds": list(range(1, 151)),
        "hardware": {"git_commit": commit, "git_dirty": dirty, "hostname": host, "gpu": gpu, "cpu": "test-cpu"},
    }
    (folder / "snapshot.json").write_text(json.dumps(snap))

    rows = []
    ck_paths = []
    for i in range(n_ckpts):
        p_vec = base_params + (seed * 0.01 + i * 0.001)
        rel = f"checkpoints/mean_{1000 * (i + 1)}.npz"
        save_policy(folder / rel, spec, p_vec, cfg.robot.max_torque)
        ck_paths.append((rel, p_vec))
        rows.append({
            "robot_seconds": 1000.0 * (i + 1), "generation": i + 1, "which": "mean",
            "train_objective": 10.0 - i, "dev20_objective": 10.0 - i, "dev20_falls": 0,
            "dev60_objective": 20.0 - i, "dev60_falls": 0, "gap20": 0.0, "file": rel,
        })
    with open(folder / "checkpoints.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if finished:
        sel_rel, sel_vec = ck_paths[-1]
        save_policy(folder / "final.npz", spec, sel_vec, cfg.robot.max_torque)
        summ = {
            "method": "linear", "search": "cma_es", "run_seed": seed, "profile": "full",
            "n_params": spec.n_params, "stop_reason": stop_reason, "generations": 10,
            "n_evals": 100, "robot_seconds": 2000.0, "elapsed_s": 5.0,
            "selected": {"file": sel_rel, "dev60_objective": 19.0, "dev60_falls": 0},
        }
        (folder / "summary.json").write_text(json.dumps(summ))
    return folder


def test_collect_l_runs_handles_missing_and_in_progress(tmp_path):
    _make_fake_run(tmp_path, 0, finished=True, host="Pavilion", stop_reason="budget")
    _make_fake_run(tmp_path, 1, finished=True, host="Evans-G16", stop_reason="cma-es stop: {'tolupsigma': 1e+20}")
    _make_fake_run(tmp_path, 2, finished=False, host="Evans-G16")   # in progress (no summary/final)
    # seed 3 missing

    finished, skipped = rk5.collect_l_runs(tmp_path, run_seeds=(0, 1, 2, 3))
    assert [r.seed for r in finished] == [0, 1]
    assert [(r.seed, r.status) for r in skipped] == [(2, "in_progress"), (3, "missing")]
    assert finished[0].hostname == "Pavilion" and finished[0].stop_reason == "budget"
    assert finished[1].hostname == "Evans-G16" and "tolupsigma" in finished[1].stop_reason
    assert finished[0].flags == [] and finished[1].flags == []


# ---- 5. Per-candidate resume & fake evaluator test -----------------------------------------------

@dataclass
class _FakeEvalResult:
    costs: np.ndarray
    fell: np.ndarray
    fall_time: np.ndarray


class _FakeEvaluator:
    """Deterministic fast evaluator that depends on start_pitches (so tilt nudges change costs)
    and counts how many candidates were actually simulated."""
    calls: list = []
    base_pitches: tuple = ()

    def __init__(self, cfg_k, controller: str, policy=None):
        self.cfg_k = cfg_k
        self.controller = controller
        self.policy = policy

    def _run(self, params, seeds):
        params = np.atleast_2d(np.asarray(params, dtype=np.float64))
        pitches = np.asarray(self.cfg_k.start_pitches, dtype=np.float64)
        base_p = np.asarray(self.base_pitches or self.cfg_k.start_pitches, dtype=np.float64)
        n_sc = len(seeds) * len(pitches)
        self.calls.append((self.controller, len(params), tuple(pitches)))
        # Encode pitch shift (k * 1e-15 rad) amplified to ~k so tests can assert exact realization dependence.
        delta = np.tile(pitches - base_p, len(seeds))
        base = np.sum(np.abs(params), axis=1, keepdims=True) + 10.0
        seed_vals = np.repeat(np.asarray(seeds, dtype=np.float64) - 5000.0, len(pitches))[None, :]
        costs = base + 0.1 * seed_vals + 1e15 * delta[None, :]
        fell = np.zeros((len(params), n_sc), dtype=bool)
        fall_time = np.full((len(params), n_sc), np.nan)
        return _FakeEvalResult(costs=costs, fell=fell, fall_time=fall_time)

    evaluate_population = _run
    evaluate_policies = _run


def test_resume_reuses_cached_candidates_and_matches_uninterrupted(tmp_path):
    from balance_sim import load_policy
    from config import load_config

    cfg = load_config("legobot")
    spec = cfg.build_policy_spec("linear")
    _make_fake_run(tmp_path / "runs", 0, finished=True, n_ckpts=3, spec=spec)
    _make_fake_run(tmp_path / "runs", 1, finished=True, n_ckpts=3, spec=spec)
    runs, _ = rk5.collect_l_runs(tmp_path / "runs", run_seeds=(0, 1))
    cands = rk5.collect_candidates(runs, cfg.reference_pid_gains(), load_policy)
    # 1 MB PID + 2 runs * (1 final + 3 ckpts) = 9 candidates; 2 finals duplicate last ckpt -> 7 unique.
    assert len(cands) == 9
    assert len({c.cache_key for c in cands}) == 7

    eval_cfg = cfg.build_eval_config(validation=True)
    seeds = (5000, 5001)
    _FakeEvaluator.base_pitches = tuple(float(p) for p in eval_cfg.start_pitches)

    # Full uninterrupted run
    _FakeEvaluator.calls = []
    full_scores, full_timing = rk5.score_all_candidates(
        cands, eval_cfg, robust=None, seeds=seeds, k_realizations=5,
        cache_dir=tmp_path / "full_cache", batch_size=2, evaluator_factory=_FakeEvaluator,
    )
    assert full_timing["n_loaded_from_cache"] == 0
    assert full_timing["n_computed_now"] == 7
    # Realization tilt offsets must produce distinct costs across k=0..4 (diff ~ k)
    c0 = full_scores[cands[1].label]["costs"]
    assert c0.shape == (5, 6)
    for k in range(1, 5):
        assert np.allclose(c0[k] - c0[0], float(k), atol=0.1)

    # Simulated interrupted run: pre-populate cache with first 3 unique candidates (MB + 2 policies), then resume.
    partial_cache = tmp_path / "partial_cache"
    rk5.score_all_candidates(
        cands[:3], eval_cfg, robust=None, seeds=seeds, k_realizations=5,
        cache_dir=partial_cache, batch_size=2, evaluator_factory=_FakeEvaluator,
    )
    _FakeEvaluator.calls = []
    resumed_scores, resumed_timing = rk5.score_all_candidates(
        cands, eval_cfg, robust=None, seeds=seeds, k_realizations=5,
        cache_dir=partial_cache, batch_size=2, evaluator_factory=_FakeEvaluator,
    )
    # Only the remaining unique candidates should be computed!
    assert resumed_timing["n_loaded_from_cache"] == 3
    assert resumed_timing["n_computed_now"] == 4
    # No cascaded evaluator calls on resume because MB PID was already cached
    assert all(ctrl == "policy" for ctrl, _, _ in _FakeEvaluator.calls)
    for c in cands:
        np.testing.assert_array_equal(resumed_scores[c.label]["costs"], full_scores[c.label]["costs"])
        np.testing.assert_array_equal(resumed_scores[c.label]["fell"], full_scores[c.label]["fell"])

    # Mismatched manifest (e.g. different K or seeds) must be rejected on resume.
    with pytest.raises(SystemExit, match="manifest mismatch"):
        rk5.score_all_candidates(
            cands, eval_cfg, robust=None, seeds=(5000,), k_realizations=5,
            cache_dir=partial_cache, evaluator_factory=_FakeEvaluator,
        )


# ---- 6. Code-identity guard & tiny end-to-end CLI test -------------------------------------------

def test_code_identity_refuses_drift_without_flag(tmp_path):
    # Create a tiny git repo with a drifted config.py relative to commit 1.
    repo = tmp_path / "mini_repo"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (repo / "config.py").write_text("X = 1\n")
    git("add", "config.py")
    git("commit", "-m", "base")
    base_sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()

    clean = rk5.check_code_identity(repo, base_sha, paths=("config.py",))
    assert not clean["has_drift"] and clean["error"] is None

    (repo / "config.py").write_text("X = 2\n")
    dirty = rk5.check_code_identity(repo, base_sha, paths=("config.py",))
    assert dirty["has_drift"] and any("config.py" in line for line in dirty["worktree_diff_files"])

    git("add", "config.py")
    git("commit", "-m", "drift")
    committed = rk5.check_code_identity(repo, base_sha, paths=("config.py",))
    assert committed["has_drift"] and any("config.py" in line for line in committed["committed_diff_files"])


def test_end_to_end_tiny_cli_and_resume(tmp_path):
    from balance_sim import pid_equivalent_weights
    from config import load_config

    cfg = load_config("legobot")
    spec = cfg.build_policy_spec("linear")
    l_pid_w = pid_equivalent_weights(spec, cfg.reference_pid_gains())

    runs_dir = tmp_path / "runs"
    _make_fake_run(runs_dir, 0, finished=True, host="Pavilion", n_ckpts=2, spec=spec, base_params=l_pid_w)
    _make_fake_run(runs_dir, 1, finished=False, host="Evans-G16", n_ckpts=1, spec=spec, base_params=l_pid_w)
    out_dir = tmp_path / "out"

    env = {**os.environ, "ROBOT": "legobot", "CUDA_VISIBLE_DEVICES": "-1"}
    script = str(REPO / "hardware" / "legobot" / "e2_rescore_k5.py")

    # 0) Without --allow-code-drift against 4582fe7 on this branch (which has non-empty diff in config.py/simulation/),
    #    the CLI must refuse to score!
    p_drift = subprocess.run(
        [sys.executable, script, "--runs-root", str(runs_dir), "--run-seeds", "0",
         "--k", "2", "--seeds", "1", "--duration", "0.5", "--out", str(tmp_path / "drift_out")],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    assert p_drift.returncode != 0 and "code drift" in (p_drift.stdout + p_drift.stderr).lower()

    # 1) --require-all must fail because seed 1 is unfinished.
    p_req = subprocess.run(
        [sys.executable, script, "--runs-root", str(runs_dir), "--run-seeds", "0,1",
         "--allow-code-drift", "--require-all", "--k", "2", "--seeds", "1", "--duration", "0.5",
         "--out", str(out_dir)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    assert p_req.returncode != 0 and "require-all" in (p_req.stdout + p_req.stderr)

    # 2) Without --require-all, skips seed 1 with warning and scores seed 0 + MB PID.
    p_run = subprocess.run(
        [sys.executable, script, "--runs-root", str(runs_dir), "--run-seeds", "0,1",
         "--allow-code-drift", "--k", "2", "--seeds", "1", "--duration", "0.5",
         "--out", str(out_dir)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    assert p_run.returncode == 0, p_run.stdout + p_run.stderr
    assert "WARNING: skipping e2-seed1 (in_progress)" in p_run.stdout
    sel = json.loads((out_dir / "selection.json").read_text())
    assert sel["settings"]["k_realizations"] == 2
    assert sel["settings"]["dry_run"] is True
    assert sel["settings"]["cuda_visible_devices"] == "-1"
    assert [r["run_seed"] for r in sel["runs"]] == [0]
    assert sel["runs"][0]["hostname"] == "Pavilion"
    assert [r["run_seed"] for r in sel["skipped_runs"]] == [1]
    # 1 MB PID + 1 final + 2 checkpoints = 4 candidates; final matches last ckpt -> 3 unique
    assert sel["timing"]["n_candidates_total"] == 4
    assert sel["timing"]["n_unique_candidates"] == 3
    assert sel["timing"]["n_computed_now"] == 3
    assert sel["results"]["selected"]["label"] in {c["label"] for c in sel["candidates"] if c["kind"] == "policy"}
    assert (out_dir / "e2_notes_tables.md").exists()
    csv_lines = (out_dir / "dev_per_scenario.csv").read_text().splitlines()
    # header + 4 candidates * 2 realizations * (1 seed * 3 pitches) = 1 + 24 = 25 lines
    assert len(csv_lines) == 1 + 4 * 2 * 3

    # 3) Resume into the same directory should compute 0 new candidates and load all 3 from cache.
    p_res = subprocess.run(
        [sys.executable, script, "--runs-root", str(runs_dir), "--run-seeds", "0,1",
         "--allow-code-drift", "--k", "2", "--seeds", "1", "--duration", "0.5",
         "--resume", str(out_dir)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    assert p_res.returncode == 0, p_res.stdout + p_res.stderr
    sel2 = json.loads((out_dir / "selection.json").read_text())
    assert sel2["timing"]["n_loaded_from_cache"] == 3
    assert sel2["timing"]["n_computed_now"] == 0
    assert sel2["results"]["stats"] == sel["results"]["stats"]

    # 4) Parallel --workers 2 must be bit-identical to --workers 1.
    par_dir = tmp_path / "par_out"
    p_par = subprocess.run(
        [sys.executable, script, "--runs-root", str(runs_dir), "--run-seeds", "0,1",
         "--allow-code-drift", "--k", "2", "--seeds", "1", "--duration", "0.5",
         "--batch-size", "1", "--workers", "2", "--priority", "idle", "--out", str(par_dir)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    assert p_par.returncode == 0, p_par.stdout + p_par.stderr
    sel_par = json.loads((par_dir / "selection.json").read_text())
    assert sel_par["timing"]["workers"] == 2
    assert sel_par["results"]["stats"] == sel["results"]["stats"]
    assert (par_dir / "dev_per_scenario.csv").read_text() == (out_dir / "dev_per_scenario.csv").read_text()


def test_batch_size_invariance_is_bit_identical(tmp_path):
    """Per-scenario costs, fell flags, and fall_time must be bit-identical across batch_size (1, 2, 4)
    in NumPy float64 (batch_sim.policy uses elementwise multiply-adds in fixed order, no BLAS matmul)."""
    from dataclasses import replace as dc_replace
    from balance_sim import load_policy, pid_equivalent_weights
    from config import load_config

    cfg = load_config("legobot")
    spec = cfg.build_policy_spec("linear")
    l_pid_w = pid_equivalent_weights(spec, cfg.reference_pid_gains())

    runs_dir = tmp_path / "runs"
    _make_fake_run(runs_dir, 0, finished=True, n_ckpts=4, spec=spec, base_params=l_pid_w)
    runs, _ = rk5.collect_l_runs(runs_dir, run_seeds=(0,))
    cands = rk5.collect_candidates(runs, cfg.reference_pid_gains(), load_policy)
    assert len({c.cache_key for c in cands}) == 5  # MB PID + 4 unique L policies

    robust = cfg.build_robustness()
    eval_cfg = dc_replace(cfg.build_eval_config(validation=True), duration=2.0)
    seeds = (5000, 5001)  # 6 scenarios x K=2 = 12 sims/candidate

    ref_scores, _ = rk5.score_all_candidates(
        cands, eval_cfg, robust, seeds, k_realizations=2, batch_size=1, workers=1,
    )
    for bs in (2, 4):
        sc, _ = rk5.score_all_candidates(
            cands, eval_cfg, robust, seeds, k_realizations=2, batch_size=bs, workers=1,
        )
        for c in cands:
            np.testing.assert_array_equal(sc[c.label]["costs"], ref_scores[c.label]["costs"])
            np.testing.assert_array_equal(sc[c.label]["fell"], ref_scores[c.label]["fell"])
            np.testing.assert_array_equal(sc[c.label]["fall_time"], ref_scores[c.label]["fall_time"])
