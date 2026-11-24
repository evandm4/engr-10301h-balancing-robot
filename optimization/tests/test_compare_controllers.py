"""optimization/compare_controllers.py: its statistics, and one tiny end-to-end comparison."""

import json

import numpy as np
import pytest
from scipy import stats

from balance_sim import pid_equivalent_weights, save_policy
from compare_controllers import (clopper_pearson, crossed_bootstrap, first_latency_losing, first_latency_with_cost,
                                 first_latency_with_falls, main, mcnemar_exact, order_statistic_interval,
                                 paired_method_bootstrap)
from config import CONFIG


def test_clopper_pearson_known_values():
    assert clopper_pearson(0, 1000) == (0.0, pytest.approx(0.003682, abs=1e-6))
    low, high = clopper_pearson(4, 1000)
    assert 0 < low < 0.004 < high
    assert clopper_pearson(5, 5)[1] == 1.0


def test_four_falls_in_a_thousand_is_where_the_upper_bound_passes_one_percent():
    assert clopper_pearson(3, 999)[1] < 0.0105 and clopper_pearson(4, 999)[1] > 0.01


def test_mcnemar_counts_only_discordant_pairs():
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 0, 0, 0], bool)
    r = mcnemar_exact(a, b)
    assert (r["only_this_fell"], r["only_reference_fell"]) == (2, 0)
    assert r["p_value"] == pytest.approx(0.5)
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_order_statistic_interval_for_ten_runs_is_second_to_ninth():
    r = order_statistic_interval(np.arange(10.0)[::-1])
    assert (r["low"], r["high"], r["k"]) == (1.0, 8.0, 2)
    assert r["coverage"] == pytest.approx(1 - 2 * stats.binom.cdf(1, 10, 0.5))
    assert order_statistic_interval([1.0, 2.0, 3.0])["low"] is None


def test_crossed_bootstrap_keeps_the_shared_scenario_variance():
    """Runs that differ only by noise but share hard and easy scenarios: resampling the shared
    scenarios once per replicate gives a wider interval than resampling them per run would."""
    rng = np.random.default_rng(0)
    difficulty = rng.lognormal(0.0, 0.3, 300)
    ratios = difficulty[None, :] * rng.lognormal(0.0, 0.02, (10, 300))
    crossed = crossed_bootstrap(ratios, n_boot=2000)
    independent = []
    for _ in range(2000):
        runs = rng.integers(0, 10, 10)
        independent.append(np.median([np.median(ratios[r, rng.integers(0, 300, 300)]) for r in runs]))
    lo, hi = np.quantile(independent, [0.025, 0.975])
    assert crossed["high"] - crossed["low"] >= 0.9 * (hi - lo)
    assert crossed["low"] < np.median(np.median(ratios, axis=1)) < crossed["high"]


def test_sweep_metric_is_the_first_latency_reaching_the_threshold():
    assert first_latency_with_falls([15, 17.5, 20, 22.5], [0, 4, 1, 9]) == 17.5   # not monotone: first wins
    assert first_latency_with_falls([15, 20], [0, 3]) is None


def test_cost_metrics_see_oscillation_without_falls():
    """The review's case: past its delay margin L oscillates instead of falling, so only cost shows it."""
    ms = [10, 15, 20, 25, 30, 35]
    l_cost = [1.90, 1.93, 2.07, 4.97, 18.3, 194]
    assert first_latency_with_falls(ms, [0, 0, 0, 0, 1, 84]) == 35
    assert first_latency_with_cost(ms, l_cost) == 25                       # 1.5 x 1.90 = 2.85
    assert first_latency_losing(ms, [0.83, 0.84, 0.88, 1.9, 4.3, 8.7]) == 25
    assert first_latency_losing(ms, [1.2, 0.9, 1.1, 1.1, 1.1, 1.1]) is None   # never beat it at the baseline
    assert first_latency_with_cost(ms, [2.0] * 6) is None


def test_paired_method_bootstrap_sees_a_difference_overlapping_intervals_hide():
    """Two methods measured on the same hard and easy scenarios: their own intervals overlap, but the
    paired interval for their ratio excludes 1, because the shared difficulty cancels."""
    rng = np.random.default_rng(1)
    difficulty = rng.lognormal(0.0, 0.4, 300)
    a = difficulty[None, :] * 0.97 * rng.lognormal(0.0, 0.01, (10, 300))
    b = difficulty[None, :] * rng.lognormal(0.0, 0.01, (10, 300))
    ca, cb = crossed_bootstrap(a, n_boot=1000), crossed_bootstrap(b, n_boot=1000)
    assert ca["high"] > cb["low"]                                         # the per-method intervals overlap
    pair = paired_method_bootstrap(a, b, n_boot=1000)
    assert pair["high"] < 1.0 and abs(pair["ratio"] - 0.97) < 0.01
    with pytest.raises(ValueError):
        paired_method_bootstrap(a, b[:, :100])


def test_frozen_tiers_need_final(tmp_path):
    with pytest.raises(SystemExit):
        main(["--tier", "validation", "--out", str(tmp_path)])


@pytest.mark.skipif(CONFIG.reference_is_provisional, reason="needs a tuned reference PID (LEGO: after E2); "
                    "test_legobot_learning.py runs compare_controllers under legobot")
def test_end_to_end_on_smoke(tmp_path):
    spec = CONFIG.build_policy_spec("linear")
    gains = CONFIG.reference_pid_gains()
    for k, scale in enumerate((1.0, 1.02)):
        (tmp_path / f"run{k}").mkdir()
        save_policy(tmp_path / f"run{k}" / "final.npz", spec, pid_equivalent_weights(spec, gains) * scale,
                    CONFIG.robot.max_torque)
    folder = main(["--tier", "smoke", "--seeds", "2", "--duration", "1.0", "--bootstrap", "200",
                   "--policies", f"L-PID={tmp_path}/run*/final.npz", "--latency-sweep", "--latencies", "15,40",
                   "--out", str(tmp_path / "out")])
    report = json.loads((folder / "comparison.json").read_text())
    ref = report["runs"]["PID, ML-tuned"]
    assert ref["median_ratio"] == 1.0 and ref["mcnemar_vs_reference"]["p_value"] == 1.0
    l_pid = report["methods"]["L-PID"]
    assert len(l_pid["runs"]) == 2 and l_pid["bootstrap_95ci"] is not None
    assert abs(l_pid["headline_median_ratio"] - 1.0) < 0.1
    assert set(report["latency_sweep"]["by_method"]) >= {"L-PID", "PID, ML-tuned"}
    lines = (folder / "per_scenario.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 9 * 6        # 7 existing + 2 policies, 2 seeds x 3 tilts
    # Weight sensitivity: the configured variant reproduces the headline; every variant scores the
    # reference at exactly 1; the variants include halved/doubled limits, effort off and thermal.
    sens = report["weight_sensitivity"]
    assert sens["configured"]["headline_median_ratio"]["L-PID"] == pytest.approx(l_pid["headline_median_ratio"],
                                                                                 rel=1e-12)   # summed in another order
    assert all(v["headline_median_ratio"]["PID, ML-tuned"] == 1.0 for v in sens.values())
    assert len(sens) == 1 + 6 + 2 and any("thermal" in k for k in sens)
    side = ref["side_metrics"]
    assert side["torque_rms_change_per_tick_n_m"]["median"] > 0
    assert 0 < side["time_at_torque_limit_fraction"]["worst"] < 1   # the 0.30 rad start saturates
    assert side["peak_pitch_deg"]["worst"] >= 17.0                  # it starts at 0.30 rad = 17.2 deg
    # Fall ordering checked per weighting; method pairs only where a method has several runs; the
    # policies were saved without a run snapshot, which is flagged.
    assert all("falls_rank_below_every_non_fall" in v for v in sens.values())
    assert sens["configured"]["falls_rank_below_every_non_fall"]
    assert set(report["method_pairs"]) == {f"{m} vs L-PID" for m in report["methods"] if m != "L-PID"}
    assert any("no snapshot.json" in w for w in report["runs"]["L-PID / run0"]["warnings"])
    assert report["runs"]["PID, ML-tuned"]["warnings"] == []
    curve = report["latency_sweep"]["curves"]["PID, ML-tuned"]
    assert report["latency_sweep"]["baseline_ms"] == 15.0 and curve["median_ratio_vs_reference"] == [1.0, 1.0]
    assert curve["first_latency_losing_to_reference"] is None       # the reference never beats itself
    assert set(report["latency_sweep"]["by_method"]["L-PID"]) >= {"first_latency_with_cost",
                                                                  "first_latency_with_falls"}


def _save_cma_result(folder, extra):
    from tuning import TuningResult, save_result
    names = CONFIG.build_search_space().names
    gains = np.asarray(CONFIG.controller.hand_tuned_gains, float)
    result = TuningResult("cma-es", gains, names, 1.0, 10, 0.1, [(10, 1.0)], mean_gains=gains * 1.01)
    return save_result(result, folder, extra)


def test_gain_runs_reads_the_dict_gains_save_result_writes(tmp_path):
    """H8/L7: <method>.json from the tuning scripts stores gains as {kp, ki, ...}; the matching config is embedded, so
    no snapshot.json is needed to check comparability. mean_gains is gated off on Pololu so its JSON keys stay unchanged."""
    from compare_controllers import gain_runs
    path = _save_cma_result(tmp_path, {"config": CONFIG.snapshot()})
    saved = json.loads(path.read_text())
    if CONFIG.name == "pololu":
        assert "mean_gains" not in saved
        lego_saved = json.loads(_save_cma_result(tmp_path / "lego", {"config": {**CONFIG.snapshot(), "name": "legobot"}}).read_text())
        assert set(lego_saved["mean_gains"]) == set(lego_saved["gains"])
    else:
        assert set(saved["mean_gains"]) == set(saved["gains"])      # the optional mean is saved as a dict on LEGO
    (entry,) = gain_runs(f"CMA={path}")
    np.testing.assert_allclose(entry.gains, CONFIG.controller.hand_tuned_gains)
    assert entry.kind == "cascaded" and entry.warnings == []


def test_embedded_config_drift_is_a_warning_not_silence(tmp_path):
    """H8: an old (pre-B5 / tick-locked) result has no snapshot.json but embeds its config: a difference in anything
    scored is reported; a result with no config at all still says it cannot be checked."""
    from compare_controllers import existing_controllers, gain_runs
    drifted = json.loads(json.dumps(CONFIG.snapshot(), default=str))
    drifted["timing"]["latency_s"] = CONFIG.timing.latency_s + 0.0123
    path = _save_cma_result(tmp_path, {"config": drifted})
    (entry,) = gain_runs(f"CMA={path}")
    assert any("trained under a different config" in w and "timing.latency_s" in w for w in entry.warnings)

    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"config": drifted, "results": {"cma-es": {"gains": json.loads(path.read_text())["gains"]}}}))
    entries = existing_controllers(comparison)
    assert entries and all(any("timing.latency_s" in w for w in e.warnings) for e in entries)
    comparison.write_text(json.dumps({"config": CONFIG.snapshot(), "results": {"cma-es": {"gains": json.loads(path.read_text())["gains"]}}}, default=str))
    assert all(e.warnings == [] for e in existing_controllers(comparison))

    bare = tmp_path / "bare"
    bare.mkdir()
    (entry,) = gain_runs(f"X={_save_cma_result(bare, None)}")
    assert any("no snapshot.json" in w for w in entry.warnings)


def test_learning_scale_drift_flags_policies_but_not_gains(tmp_path):
    """PR #15 validation V1: re-measured LEARNING scales change what a learned policy's inputs are divided by, but
    not how a PID or LQR is tuned or scored, so only policies warn; a real scoring drift still warns for gains."""
    from compare_controllers import existing_controllers, gain_runs, policy_runs
    drifted = json.loads(json.dumps(CONFIG.snapshot(), default=str))
    drifted["learning"]["int_v_scale"] = CONFIG.learning.int_v_scale * 2
    path = _save_cma_result(tmp_path / "pid", {"config": drifted})
    (entry,) = gain_runs(f"CMA={path}")
    assert entry.warnings == []
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"config": drifted, "results": {"cma-es": {"gains": json.loads(path.read_text())["gains"]}}}))
    assert all(e.warnings == [] for e in existing_controllers(comparison))

    drifted["robustness"]["latency_range_s"] = [0.010, 0.015]        # the E1 cma_lat10_15 case: still reported
    (entry,) = gain_runs(f"CMA={_save_cma_result(tmp_path / 'lat', {'config': drifted})}")
    assert len(entry.warnings) == 1 and "robustness.latency_range_s" in entry.warnings[0]
    assert "learning" not in entry.warnings[0]

    spec = CONFIG.build_policy_spec("linear")
    run = tmp_path / "policy"
    run.mkdir()
    save_policy(run / "final.npz", spec, pid_equivalent_weights(spec, CONFIG.reference_pid_gains()),
                CONFIG.robot.max_torque)
    (run / "snapshot.json").write_text(json.dumps({"config": drifted}))
    (entry,) = policy_runs(f"L={run}/final.npz")
    assert any("learning.int_v_scale" in w for w in entry.warnings)


def test_gain_runs_end_to_end_on_smoke_under_any_robot(tmp_path):
    """H12: a tiny comparison of the tuning scripts' own <method>.json (dict gains) against the reference runs under
    both robots (the LEGO one scores against its provisional baseline)."""
    for k in range(2):
        _save_cma_result(tmp_path / f"run{k}", {"config": CONFIG.snapshot()})
    folder = main(["--tier", "smoke", "--seeds", "2", "--duration", "1.0", "--bootstrap", "200",
                   "--gain-runs", f"CMA={tmp_path}/run*/cma-es.json", "--out", str(tmp_path / "out")])
    report = json.loads((folder / "comparison.json").read_text())
    assert len(report["methods"]["CMA"]["runs"]) == 2
    assert all(report["runs"][label]["warnings"] == [] for label in ("CMA / run0", "CMA / run1"))
    assert report["reference_provisional"] == CONFIG.reference_is_provisional
