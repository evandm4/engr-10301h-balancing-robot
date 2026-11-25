"""hardware/legobot/e2_select.py: E2.1 run collection / filtering, candidates, and the D3 decision on synthetic
statistics (pure functions, no simulation), plus one tiny end-to-end run on synthetic run folders in a
subprocess (the script loads the legobot config itself, which this pytest process may not have)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hardware" / "legobot"))

import e2_select as es  # noqa: E402

MID = {"cma_x0": "midpoint", "de_iters": 30, "de_pop": 24}
WARM = {"cma_x0": "model-based", "de_iters": 2, "de_pop": 6}
GAINS = {"kp": 2.0, "ki": 5.0, "kd": 0.15, "kv_p": 0.5}


def make_run(root: Path, name: str, recipe: dict, seed: int, *, commit="ecb666e0123456789", dirty=False,
             n_train=150, dtype="float64", hostname="laptop", gpu="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
             robust_on="dev", objectives=None, **arg_overrides) -> Path:
    """A run folder as run_batch_tuning writes it (only the fields e2_select reads)."""
    objectives = objectives or {"differential evolution": 30.0, "cma-es": 20.0, "random search": 40.0,
                                "cma-es (mean)": 21.0}
    args = {**es.COMMON_ARGS, **recipe, "seed": seed, "seeds": n_train, "no_plot": True, "no_show": True}
    args.update(arg_overrides)
    results = {"hand-tuned": {"gains": {"kp": 2.5, "ki": 1.5, "kd": 0.2, "kv_p": 0.5}},
               "model-based": {"gains": dict(GAINS)}}
    for i, (entry, obj) in enumerate(objectives.items()):
        g = dict(GAINS, kp=GAINS["kp"] + 0.01 * (i + 1) + 0.001 * seed)
        results[entry] = {"gains": g, "train_objective": obj, "robust_validation_objective": obj / 4,
                          "robust_validation_falls": 0}
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "comparison.json").write_text(json.dumps({
        "timestamp": name, "provenance": {"git_commit": commit, "git_dirty": dirty, "hostname": hostname, "gpu": gpu},
        "run_arguments": args, "robust_checked_on": robust_on,
        "robustness": {"train_seeds": list(range(n_train))}, "results": results}))
    (folder / "cma_es.json").write_text(json.dumps({"dtype": dtype}))
    return folder


def full_design(root: Path):
    for s in range(10):
        make_run(root, f"20261125-{s:02d}0000", MID, s)
        make_run(root, f"20261125-{s:02d}3000", WARM, s)


def test_collects_the_twenty_e21_runs_and_explains_the_rest(tmp_path):
    full_design(tmp_path)
    make_run(tmp_path, "20261124-032414", MID, 0, commit="59065a348abc", dirty=True, n_train=50, hostname=None,
             seeds=None)                                                      # an E1 run
    make_run(tmp_path, "x-float32", MID, 3, float32=True, dtype="float32")
    make_run(tmp_path, "x-pololu", MID, 3, robot="pololu")
    make_run(tmp_path, "x-recipe", {"cma_x0": "model-based", "de_iters": 30, "de_pop": 24}, 3)
    make_run(tmp_path, "x-val", MID, 3, robust_on="validation")
    make_run(tmp_path, "x-final", MID, 3, final=True)
    (tmp_path / "x-empty").mkdir()

    kept, excluded = es.collect_runs(tmp_path)
    assert len(kept) == 20 and es.design_problems(kept) == []
    assert {(r.recipe, r.seed) for r in kept} == {(k, s) for k in ("mid", "warm") for s in range(10)}
    why = {r.name: " ".join(r.reasons) for r in excluded}
    assert set(why) == {"20261124-032414", "x-float32", "x-pololu", "x-recipe", "x-val", "x-final", "x-empty"}
    assert "commit 59065a348" in why["20261124-032414"] and "50 training seeds" in why["20261124-032414"]
    assert "float32" in why["x-float32"] and "robot pololu" in why["x-pololu"]
    assert "not an E2.1 recipe" in why["x-recipe"] and "not dev" in why["x-val"]
    assert "--final" in why["x-final"] and "comparison.json" in why["x-empty"]


def test_missing_duplicate_dirty_and_drifted_runs_are_design_problems(tmp_path):
    full_design(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "20261125-030000")                                 # mid seed 3 missing
    make_run(tmp_path, "20261125-993000", WARM, 5)                              # warm seed 5 twice
    make_run(tmp_path, "20261125-990000", MID, 3, dirty=True)                   # mid seed 3, dirty
    kept, _ = es.collect_runs(tmp_path)
    problems = es.design_problems(kept)
    assert "21 E2.1 runs found, expected 20" in problems
    assert any(p.startswith("2 runs for warm seed 5") for p in problems)
    assert any("20261125-990000: trained from a dirty tree" in p for p in problems)
    assert not any("missing" in p for p in problems)

    class DriftedConfig:
        @staticmethod
        def scoring_differences(snapshot, ignore):
            return ["robot.coulomb_friction"]

    for name in os.listdir(tmp_path):                                           # give every run a config snapshot
        p = tmp_path / name / "comparison.json"
        d = json.loads(p.read_text())
        d["config"] = {"name": "legobot"}
        p.write_text(json.dumps(d))
    kept, _ = es.collect_runs(tmp_path, cfg=DriftedConfig())
    assert any("different config: robot.coulomb_friction" in p for p in es.design_problems(kept))


def test_commit_and_seed_filters_are_options(tmp_path):
    make_run(tmp_path, "a", MID, 0, commit="59065a348abc", n_train=50, hostname=None, seeds=None)
    assert es.collect_runs(tmp_path)[0] == []
    kept, _ = es.collect_runs(tmp_path, commits=("59065a3",), n_train_seeds=50)
    assert [r.name for r in kept] == ["a"] and kept[0].flags == ["no hostname in provenance (pre-ecb666e run)"]
    assert es.design_problems(kept)[0] == "1 E2.1 runs found, expected 20"   # the hostname flag isn't a problem


def test_candidates_mean_and_cma_best_sample_by_default(tmp_path):
    make_run(tmp_path, "a", MID, 0, objectives={"differential evolution": 10.0, "cma-es": 20.0,
                                                "random search": 40.0, "cma-es (mean)": 21.0})
    make_run(tmp_path, "b", WARM, 4)
    kept, _ = es.collect_runs(tmp_path)
    a = kept[0]
    assert es.run_entries(a) == ["cma-es (mean)", "cma-es"]
    assert es.run_entries(a, best="overall") == ["cma-es (mean)", "differential evolution"]   # DE won training
    assert es.run_entries(a, include=["random search", "nope"]) == ["cma-es (mean)", "cma-es", "random search"]
    cands = es.candidates_of(kept)
    assert [c.label for c in cands] == ["PID, CMA-mid (mean) / s0", "PID, CMA-mid / s0",
                                        "PID, CMA-warm (mean) / s4", "PID, CMA-warm / s4"]
    assert cands[1].gains == pytest.approx((2.02, 5.0, 0.15, 0.5)) and cands[1].train_objective == 20.0
    dup = es.candidates_of([a, a])
    assert len({c.label for c in dup}) == 2 and all("[a]" in c.label for c in dup)


# ---- D3 on synthetic statistics ---------------------------------------------------------------------

def stat(objective, ci_high, only_this=0, only_ref=0, p=1.0, ci_low=None):
    return {"objective": objective, "median_cost": objective, "ratio_95ci": [ci_low or ci_high - 0.1, ci_high],
            "mcnemar_vs_model_based": {"only_this_fell": only_this, "only_reference_fell": only_ref, "p_value": p}}


def test_shows_more_falls_needs_both_direction_and_significance():
    assert es.shows_more_falls({"only_this_fell": 9, "only_reference_fell": 1, "p_value": 0.02})
    assert not es.shows_more_falls({"only_this_fell": 9, "only_reference_fell": 1, "p_value": 0.05})
    assert not es.shows_more_falls({"only_this_fell": 4, "only_reference_fell": 1, "p_value": 0.375})
    assert not es.shows_more_falls({"only_this_fell": 1, "only_reference_fell": 9, "p_value": 0.02})  # fewer


def test_d3_adopts_a_finalist_only_when_ci_below_one_and_no_more_falls():
    conf = {"A": stat(1, 0.99), "B": stat(1, 1.0), "C": stat(1, 0.9, 9, 1, 0.02), "D": stat(1, 0.95, 4, 1, 0.375)}
    d = es.d3_decide(conf, ["B", "C", "A", "D"])
    v = d["verdicts"]
    assert [v[k]["qualifies"] for k in "BCAD"] == [False, False, True, True]
    assert v["B"]["why_not"] == ["ratio CI upper 1.0000 >= 1"]                  # CI touching 1 fails
    assert "more falls (9 vs 1" in v["C"]["why_not"][0]
    assert d["qualifying_in_dev_order"] == ["A", "D"] and d["adopted"] == "A" and not d["adopted_is_model_based"]
    assert d["rule"] == es.D3_RULE and d["n_finalists_tested"] == 4


def test_d3_several_qualify_dev_order_wins_over_a_better_confirm_ratio():
    conf = {"dev1": stat(9.0, 0.98), "dev2": stat(1.0, 0.50, ci_low=0.4)}    # dev2 far better on confirm
    d = es.d3_decide(conf, ["dev1", "dev2"])
    assert d["adopted"] == "dev1" and "best on dev" in d["why"]


def test_d3_none_qualifies_means_model_based():
    d = es.d3_decide({"A": stat(1, 1.2), "B": stat(1, 0.8, 12, 2, 0.01)}, ["A", "B"])
    assert d["adopted"] == es.MB and d["adopted_is_model_based"] and d["qualifying_in_dev_order"] == []


def test_dev_ranking_and_finalists_skip_the_baselines():
    stats = {es.MB: {"objective": 3.0, "median_cost": 3}, es.HT: {"objective": 5.0, "median_cost": 5},
             "t1": {"objective": 2.0, "median_cost": 2.5}, "t2": {"objective": 2.0, "median_cost": 2.1},
             "t3": {"objective": 4.0, "median_cost": 1}, "t4": {"objective": 6.0, "median_cost": 1}}
    ranked = es.rank_dev(stats)
    assert ranked == ["t2", "t1", es.MB, "t3", es.HT, "t4"]                     # tie -> lower median
    assert es.pick_finalists(ranked, ["t1", "t2", "t3", "t4"], 3) == ["t2", "t1", "t3"]


# ---- end to end (tiny, CPU) -------------------------------------------------------------------------

def test_end_to_end_dry_run_on_synthetic_runs(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, "r-mid", MID, 0)
    make_run(runs, "r-warm", WARM, 0)
    make_run(runs, "r-e1", MID, 1, commit="59065a348", n_train=50)
    out = tmp_path / "out"
    env = {k: v for k, v in os.environ.items() if k != "ROBOT"}
    proc = subprocess.run([sys.executable, str(REPO / "hardware" / "legobot" / "e2_select.py"), "--runs-root",
                           str(runs), "--allow-partial", "--seeds", "1", "--duration", "1", "--bootstrap", "200",
                           "--out", str(out)], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    sel = json.loads((out / "selection.json").read_text())
    assert [r["folder"] for r in sel["runs"]] == ["r-mid", "r-warm"]
    assert sel["excluded_runs"][0]["folder"] == "r-e1"
    assert sel["runs"][0]["hostname"] == "laptop" and sel["runs"][0]["gpu"].startswith("NVIDIA")
    assert sel["settings"]["dry_run"] and sel["dev"]["seeds"] == [5000, 5000]
    assert sel["confirm"]["seeds"] == [5049, 5049]          # a dry run never touches the D2 slice
    assert not sel["confirm"]["is_d2_slice"]
    assert len(sel["candidates"]) == 4 and len(sel["dev"]["ranking"]) == 6 and len(sel["finalists"]) == 3
    assert set(sel["confirm"]["stats"]) == {*sel["finalists"], es.MB}
    assert sel["decision"]["adopted"] in [*sel["finalists"], es.MB]
    assert sel["reference"]["label"] == sel["decision"]["adopted"]
    assert (out / "e2_notes_tables.md").read_text(encoding="utf-8").startswith("> DRY RUN")
    assert len((out / "dev_per_scenario.csv").read_text().splitlines()) == 1 + 6 * 3
