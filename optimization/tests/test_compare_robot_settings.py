"""compare_controllers.py's robot-specific settings (config.ComparisonConfig; the 2026-11-25 review before
LEGO's E2.8): the Pololu robot's are exactly the constants the script hard-coded before, and the LEGO
robot's are its own (sweep grid with its 7.5-12.5 ms training range, a named 10 ms baseline, the relative
fall metric, its 0.50 N m thermal rating). Both robots are built with load_config, so these run the same
under any ROBOT. Simulations here use ad-hoc seeds >= 7,000,000, never the frozen tiers."""

import json
from dataclasses import replace

import numpy as np
import pytest

import compare_controllers
from compare_controllers import (cost_variants, excess_ceiling_censored, first_latency_with_excess_falls,
                                 first_latency_with_falls, mcnemar_excess, sweep_by_method, sweep_grid)
from config import ComparisonConfig, load_comparison_config, load_config

POLOLU, LEGO = load_config("pololu"), load_config("legobot")
POLOLU_CMP, LEGO_CMP = load_comparison_config("pololu"), load_comparison_config("legobot")
AD_HOC_SEEDS = (7_200_000, 7_200_001)   # never validation (3000-3099), fall-rate (6000-6999) or confirm

# compare_controllers.py's constants before 2026-11-25 (commit 4582fe7).
OLD_SWEEP_LATENCIES_MS = (10.0,) + tuple(15.0 + 2.5 * i for i in range(13))
OLD_SWEEP_FALL_RATE_SEEDS, OLD_SWEEP_FALL_THRESHOLD, OLD_SWEEP_COST_FACTOR = 333, 4, 1.5
OLD_CONTINUOUS_CURRENT_A = 1.5


def _old_cost_variants(cfg):
    """compare_controllers.cost_variants as it was before the fix."""
    c, a = cfg.cost, cfg.actuator
    base = (c.pitch_limit_rad, c.velocity_limit, c.torque_limit)
    thermal = a.torque_constant * OLD_CONTINUOUS_CURRENT_A * a.gear_ratio * a.gear_efficiency * a.n_motors
    variants = {"configured": base}
    for i, name in enumerate(("pitch", "velocity", "torque")):
        for factor in (0.5, 2.0):
            limits = list(base)
            limits[i] *= factor
            variants[f"{name} limit x{factor:g}"] = tuple(limits)
    variants["effort off"] = (base[0], base[1], float("inf"))
    variants[f"torque limit {thermal:.3f} (thermal)"] = (base[0], base[1], thermal)
    return variants


# ---- Pololu: unchanged -----------------------------------------------------------------------

def test_pololu_settings_are_the_old_constants():
    assert POLOLU_CMP == ComparisonConfig()
    assert POLOLU_CMP.sweep_latencies_ms == OLD_SWEEP_LATENCIES_MS
    assert POLOLU_CMP.baseline_ms(POLOLU) == 10.0
    assert sweep_grid(POLOLU_CMP.sweep_latencies_ms, POLOLU_CMP.baseline_ms(POLOLU)) == OLD_SWEEP_LATENCIES_MS
    assert (POLOLU_CMP.sweep_fall_rate_seeds, POLOLU_CMP.sweep_fall_threshold, POLOLU_CMP.sweep_cost_factor) == \
        (OLD_SWEEP_FALL_RATE_SEEDS, OLD_SWEEP_FALL_THRESHOLD, OLD_SWEEP_COST_FACTOR)
    a = POLOLU.actuator   # bit for bit: the same product in the same order
    assert POLOLU_CMP.thermal_torque(a) == a.torque_constant * 1.5 * a.gear_ratio * a.gear_efficiency * a.n_motors
    assert round(POLOLU_CMP.thermal_torque(a), 3) == 0.253


def test_pololu_cost_variants_are_unchanged(monkeypatch):
    monkeypatch.setattr(compare_controllers, "CONFIG", POLOLU)
    new, old = cost_variants(), _old_cost_variants(POLOLU)
    assert list(new) == list(old) and new == old
    assert "torque limit 0.253 (thermal)" in new


def test_pololu_default_sweep_keeps_its_10ms_baseline(tmp_path, monkeypatch):
    """Pololu's default call is unchanged, and the old positional reading of --latencies is gone: the
    baseline stays 10 ms (added to the grid) unless --baseline-latency names another."""
    monkeypatch.setattr(compare_controllers, "CONFIG", POLOLU)
    monkeypatch.setattr(compare_controllers, "tier_seeds", lambda robust, tier: AD_HOC_SEEDS[:1])
    folder = compare_controllers.main(["--tier", "smoke", "--duration", "0.5", "--bootstrap", "50",
                                       "--latency-sweep", "--latencies", "15,40", "--out", str(tmp_path / "o")])
    sweep = json.loads((folder / "comparison.json").read_text())["latency_sweep"]
    assert sweep["baseline_ms"] == 10.0 and sweep["latencies_ms"] == [10.0, 15.0, 40.0]
    assert (sweep["threshold_falls"], sweep["cost_factor"]) == (4, 1.5)
    ref = sweep["curves"][POLOLU.learning.reference_pid_entry]
    assert ref["latency_ms"] == [10.0, 15.0, 40.0] and ref["falls_at_baseline"] == ref["falls"][0]
    assert ref["excess_falls_vs_baseline"][0] == 0 and ref["mcnemar_p_vs_baseline"][0] == 1.0


# ---- LEGO: its own -------------------------------------------------------------------------------

def test_lego_sweep_covers_its_training_range_with_a_named_10ms_baseline():
    assert LEGO_CMP.baseline_ms(LEGO) == 10.0 == round(LEGO.timing.latency_s * 1000, 9)
    low, high = (round(s * 1000, 9) for s in LEGO.robustness.latency_range_s)
    assert (low, high) == (7.5, 12.5)                              # E2_PLAN D4
    grid = sweep_grid(LEGO_CMP.sweep_latencies_ms, LEGO_CMP.baseline_ms(LEGO))
    assert {low, 10.0, high} <= set(grid) and grid == LEGO_CMP.sweep_latencies_ms
    assert grid[0] == 7.5 and grid[-1] == 45.0 and all(b - a == 2.5 for a, b in zip(grid, grid[1:]))
    assert set(OLD_SWEEP_LATENCIES_MS) <= set(grid)                # Pololu's grid is part of it
    # The thresholds are Pololu's; the relative metric is the one LEGO's falling baseline leaves readable.
    assert (LEGO_CMP.sweep_fall_threshold, LEGO_CMP.sweep_excess_falls, LEGO_CMP.sweep_excess_alpha) == (4, 5, 0.05)


def test_lego_thermal_variant_is_its_rating(monkeypatch):
    monkeypatch.setattr(compare_controllers, "CONFIG", LEGO)
    variants = cost_variants()
    (name,) = [k for k in variants if "thermal" in k]
    assert name == "torque limit 0.500 (thermal)" and variants[name][2] == 0.50
    assert list(variants)[:-1] == list(_old_cost_variants(LEGO))[:-1]   # only the thermal limit changed
    pololu_rule = replace(LEGO_CMP, continuous_torque_n_m=None, continuous_current_a=1.5)
    assert pololu_rule.thermal_torque(LEGO.actuator) == pytest.approx(0.772, abs=5e-4)   # the old, wrong value


def test_settings_follow_the_selected_robot_and_stay_out_of_the_snapshot(monkeypatch):
    for cfg, settings in ((POLOLU, POLOLU_CMP), (LEGO, LEGO_CMP)):
        monkeypatch.setattr(compare_controllers, "CONFIG", cfg)
        assert compare_controllers.comparison_settings() is settings
        snap = json.dumps(cfg.snapshot(derived=False), default=str)
        assert "sweep_" not in snap and "continuous_torque" not in snap and "continuous_current" not in snap
        assert cfg.scoring_differences(cfg.snapshot(derived=False)) == []
    with pytest.raises(ValueError):
        load_comparison_config("nope")


# ---- the sweep's baseline and metrics ----------------------------------------------------------

def test_baseline_is_named_never_positional():
    assert sweep_grid((7.5, 10, 15), 10.0) == (7.5, 10.0, 15.0)    # 7.5 first, baseline still 10
    assert sweep_grid((15.0, 40.0), 10.0) == (10.0, 15.0, 40.0)   # missing baseline: added
    assert sweep_grid((40, 15, 15), 15) == (15.0, 40.0)            # sorted, no duplicates
    with pytest.raises(ValueError):
        sweep_grid((-2.5, 10.0), 10.0)


def test_mcnemar_excess_is_one_sided_and_paired():
    base = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    assert mcnemar_excess(base, base) == {"only_here": 0, "only_at_baseline": 0, "excess": 0, "p_value": 1.0}
    more = base.copy()
    more[2:7] = True                                                # 5 new falls, none recovered
    r = mcnemar_excess(more, base)
    assert (r["only_here"], r["only_at_baseline"], r["excess"]) == (5, 0, 5)
    assert r["p_value"] == pytest.approx(0.5 ** 5)
    assert mcnemar_excess(base, more)["p_value"] == 1.0            # fewer falls: never significant


def test_excess_metric_reads_through_a_falling_baseline():
    """LEGO's case: ~43 / 999 falls already at the 10 ms baseline. The absolute metric is reached at the
    baseline (left-censored) for every controller; the relative one finds where latency adds falls, and
    ignores borderline flips with no significant net excess."""
    rng = np.random.default_rng(7_300_000)
    n = 999
    base = np.zeros(n, bool)
    base[rng.choice(n, 43, replace=False)] = True
    up = np.flatnonzero(~base)

    def with_changes(new, recovered):
        fell = base.copy()
        fell[up[:new]] = True
        fell[np.flatnonzero(base)[:recovered]] = False
        return fell

    flags = {10.0: base, 12.5: with_changes(10, 6), 15.0: with_changes(3, 0), 17.5: with_changes(8, 1)}
    ms = list(flags)
    tests = [mcnemar_excess(flags[m], base) for m in ms]
    excess, p = [t["excess"] for t in tests], [t["p_value"] for t in tests]
    assert excess == [0, 4, 3, 7]
    assert p[1] > 0.2 and p[3] == pytest.approx(10 / 512)            # 10 vs 6 flips: noise; 8 vs 1: not
    assert first_latency_with_falls(ms, [int(flags[m].sum()) for m in ms]) == 10.0   # censored at the baseline
    assert first_latency_with_excess_falls(ms[1:], excess[1:], p[1:]) == 17.5
    assert first_latency_with_excess_falls(ms[1:], excess[1:], p[1:], min_excess=8) is None


def test_lego_latency_sweep_end_to_end(tmp_path, monkeypatch):
    """A tiny LEGO run of main with the sweep (ad-hoc seeds, 0.5 s): the baseline is 10 ms although 7.5 ms
    comes first, the relative metric and its settings are reported, and the thermal variant is 0.50 N m."""
    monkeypatch.setattr(compare_controllers, "CONFIG", LEGO)
    monkeypatch.setattr(compare_controllers, "tier_seeds", lambda robust, tier: AD_HOC_SEEDS)
    folder = compare_controllers.main(["--tier", "smoke", "--duration", "0.5", "--bootstrap", "50",
                                       "--latency-sweep", "--latencies", "7.5,12.5", "--out", str(tmp_path / "o")])
    report = json.loads((folder / "comparison.json").read_text())
    assert report["robot"] == "legobot" and "torque limit 0.500 (thermal)" in report["weight_sensitivity"]
    sweep = report["latency_sweep"]
    assert sweep["baseline_ms"] == 10.0 and sweep["latencies_ms"] == [7.5, 10.0, 12.5]
    assert sweep["excess_falls"]["min_excess"] == 5 and sweep["excess_falls"]["alpha"] == 0.05
    assert sweep["excess_falls"]["alpha_scope"] == "pointwise_unadjusted"
    assert sweep["excess_falls"]["n_tested_latencies"] == 1                  # only 12.5 is above 10 ms
    REF = LEGO.learning.reference_pid_entry
    assert set(sweep["curves"]) == {"PID, model-based", "PID, hand-tuned"}     # no LQR on this robot
    for label, c in sweep["curves"].items():
        assert c["latency_ms"] == [7.5, 10.0, 12.5] and c["scenarios"] == 6
        assert c["falls_at_baseline"] == c["falls"][1]
        assert c["excess_falls_vs_baseline"][1] == 0 and c["mcnemar_p_vs_baseline"][1] == 1.0
        assert c["excess_falls_vs_baseline"] == [f - c["falls"][1] for f in c["falls"]]
        assert c["first_latency_with_excess_falls"] in (None, 12.5)   # never at or below the baseline
        # 6 scenarios: the metric can fire only if at least 5 survive the baseline (0.5 ** 5 < 0.05).
        assert c["excess_falls_ceiling_censored"] == (6 - c["falls"][1] < 5)
    assert sweep["curves"][REF]["median_ratio_vs_reference"] == [1.0, 1.0, 1.0]
    assert "first_latency_with_excess_falls" in sweep["by_method"][REF]
    lines = (folder / "latency_sweep.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 2 * 3


# ---- 2026-11-25 review of d6268ce ------------------------------------------------------------------

def test_excess_threshold_is_reachable_at_its_alpha():
    """At alpha 0.05, 4 new falls with none recovered give p = 0.0625 (can never pass); 5 give 0.03125."""
    assert POLOLU_CMP.sweep_excess_falls == LEGO_CMP.sweep_excess_falls == 5
    base = np.zeros(20, bool)
    four, five = base.copy(), base.copy()
    four[:4], five[:5] = True, True
    t4, t5 = mcnemar_excess(four, base), mcnemar_excess(five, base)
    assert (t4["excess"], t4["only_at_baseline"]) == (4, 0) and t4["p_value"] == pytest.approx(0.0625)
    assert (t5["excess"], t5["only_at_baseline"]) == (5, 0) and t5["p_value"] == pytest.approx(0.03125)
    for settings in (POLOLU_CMP, LEGO_CMP):
        args = (settings.sweep_excess_falls, settings.sweep_excess_alpha)
        assert first_latency_with_excess_falls([12.5], [4], [t4["p_value"]], *args) is None
        assert first_latency_with_excess_falls([12.5], [5], [t5["p_value"]], *args) == 12.5
        # Even with the size threshold lowered to 4, (4, 0) misses alpha: 4 was unreachable.
        assert first_latency_with_excess_falls([12.5], [4], [t4["p_value"]], 4, settings.sweep_excess_alpha) is None


def test_excess_ceiling_censoring():
    """When too few scenarios survive the baseline for even all of them falling to pass, the metric is
    censored: flagged, and its None is unknown rather than "never" (no inf in the method median)."""
    assert excess_ceiling_censored(4, 5, 0.05) and not excess_ceiling_censored(5, 5, 0.05)
    assert excess_ceiling_censored(5, 5, 0.01) and not excess_ceiling_censored(7, 5, 0.01)   # 0.5**7 < 0.01
    assert not excess_ceiling_censored(999 - 43, 5, 0.05)                                    # LEGO's fall-rate set

    class E:
        def __init__(self, label, method):
            self.label, self.method = label, method

    def curve(first, censored):
        return {"first_latency_with_falls": None, "first_latency_with_cost": 20.0,
                "first_latency_losing_to_reference": None, "first_latency_with_excess_falls": first,
                "excess_falls_ceiling_censored": censored, "falls_at_or_below_baseline": False}

    entries = [E("a1", "A"), E("a2", "A"), E("a3", "A"), E("b1", "B"), E("b2", "B")]
    curves = {"a1": curve(15.0, False), "a2": curve(None, True), "a3": curve(None, False),
              "b1": curve(15.0, False), "b2": curve(None, False)}
    out = sweep_by_method(entries, curves)
    a = out["A"]["first_latency_with_excess_falls"]
    assert a == {"per_run": [15.0, None, None], "median": None, "censored_runs": 1}
    assert out["A"]["excess_falls_ceiling_censored"] == [False, True, False]
    b = out["B"]["first_latency_with_excess_falls"]   # uncensored None: "never", ranked above every latency
    assert b == {"per_run": [15.0, None], "median": None} and "censored_runs" not in b
    curves["b2"] = curve(25.0, False)
    assert sweep_by_method(entries, curves)["B"]["first_latency_with_excess_falls"]["median"] == 20.0
    # A censored run that fired anyway (impossible in practice) is not unknown.
    curves["a2"] = curve(17.5, True)
    assert "censored_runs" not in sweep_by_method(entries, curves)["A"]["first_latency_with_excess_falls"]


def test_latency_sweep_ignores_latencies_below_a_non_first_baseline(monkeypatch):
    """Grid [7.5, 10, 12.5, 15], baseline 10 (index 1). At 7.5 controller X falls 6 times at a low cost
    and loses to the reference: if the first grid point were the baseline, all four metrics would read
    differently. Every one must ignore 7.5, and 10 ms must be the cost baseline. Synthetic scores."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Cfg:
        latency: float = 0.01
        start_pitches: tuple = (0.0,)

    @dataclass(frozen=True)
    class Robust:
        latency_range: object = (0.0075, 0.0125)
        sensitivity_weight: float = 1.0

    class E:
        def __init__(self, label):
            self.label = label

    n = 20
    falls = {7.5: 6, 10.0: 0, 12.5: 0, 15.0: 6}                         # X's falls (scenarios 0..k-1)
    x_cost = {7.5: 0.5, 10.0: 1.0, 12.5: 1.2, 15.0: 2.0}               # X's per-scenario cost
    ref_cost = {7.5: 0.4, 10.0: 1.1, 12.5: 1.1, 15.0: 1.1}             # the reference's (never falls)

    def fake_score(entries, config, robust, seeds, backend, device):
        assert robust.latency_range is None and robust.sensitivity_weight == 0.0
        ms = round(config.latency * 1000, 9)
        out = {}
        for label, cost, k in (("ref", ref_cost[ms], 0), ("X", x_cost[ms], falls[ms])):
            fell = np.zeros(n, bool)
            fell[:k] = True
            costs = np.full(n, cost)
            out[label] = {"costs": costs, "fell": fell, "n_fell": int(k), "mean_cost": float(costs.mean())}
        return out

    monkeypatch.setattr(compare_controllers, "score", fake_score)
    settings = ComparisonConfig(sweep_latencies_ms=(7.5, 10.0, 12.5, 15.0))
    curves = compare_controllers.latency_sweep([E("ref"), E("X")], Cfg(), Robust(), list(range(n)), "ref",
                                               baseline_ms=10.0, settings=settings)
    x = curves["X"]
    assert x["latency_ms"] == [7.5, 10.0, 12.5, 15.0] and x["falls"] == [6, 0, 0, 6]
    assert x["falls_at_baseline"] == 0 and x["falls_at_or_below_baseline"] is False
    assert x["first_latency_with_falls"] == 15.0              # not 7.5 (6 >= 4 falls there)
    assert x["first_latency_with_excess_falls"] == 15.0       # not 7.5 (6 more falls than at 10 ms there too)
    assert x["excess_falls_vs_baseline"] == [6, 0, 0, 6] and x["mcnemar_p_vs_baseline"][3] == pytest.approx(0.5 ** 6)
    assert x["first_latency_with_cost"] == 15.0               # 2.0 >= 1.5 x 1.0 (10 ms); from 7.5's 0.5, 12.5
    assert x["first_latency_losing_to_reference"] == 12.5     # beats it at 10 ms (0.91); at 7.5 it lost (1.25)
    assert x["median_cost"][1] == 1.0 and x["excess_falls_ceiling_censored"] is False
    ref = curves["ref"]
    assert ref["first_latency_with_falls"] is None and ref["first_latency_with_excess_falls"] is None
