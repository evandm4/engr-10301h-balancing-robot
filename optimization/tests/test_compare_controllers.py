"""optimization/compare_controllers.py: its statistics, and one tiny end-to-end comparison."""

import json

import numpy as np
import pytest
from scipy import stats

from balance_sim import pid_equivalent_weights, save_policy
from compare_controllers import (clopper_pearson, crossed_bootstrap, first_latency_with_falls, main,
                                 mcnemar_exact, order_statistic_interval)
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


def test_frozen_tiers_need_final(tmp_path):
    with pytest.raises(SystemExit):
        main(["--tier", "validation", "--out", str(tmp_path)])


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
    assert sens["configured"]["headline_median_ratio"]["L-PID"] == l_pid["headline_median_ratio"]
    assert all(v["headline_median_ratio"]["PID, ML-tuned"] == 1.0 for v in sens.values())
    assert len(sens) == 1 + 6 + 2 and any("thermal" in k for k in sens)
    side = ref["side_metrics"]
    assert side["torque_rms_change_per_tick_n_m"]["median"] > 0
    assert 0 < side["time_at_torque_limit_fraction"]["worst"] < 1   # the 0.30 rad start saturates
    assert side["peak_pitch_deg"]["worst"] >= 17.0                  # it starts at 0.30 rad = 17.2 deg
