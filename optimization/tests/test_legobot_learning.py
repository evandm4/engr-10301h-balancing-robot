"""The learned-controller setup for the LEGO robot (legobot_config.LEARNING), before it has a tuned
reference PID (E1/E2): its own observation scales, the provisional reference (its hand-tuned PID),
the comparison reader for both tuning scripts' layouts, and compare_controllers under legobot.
Tests run under the default Pololu robot, so the LEGO config is built with load_config."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import compare_controllers
from balance_sim import pid_equivalent_weights, save_policy
from config import load_config, read_pid_comparison

ROOT = Path(__file__).resolve().parents[2]
LEGO = load_config("legobot")
CONFIG = load_config("pololu")      # the Pololu robot, whatever ROBOT is set to (these compare the two)


def test_lego_has_its_own_learning_scales():
    assert LEGO.learning != CONFIG.learning
    assert LEGO.learning.speed_scale == 0.4                  # its 0.40 m/s cruise, not the Pololu 1 m/s
    assert LEGO.learning.command_scale == LEGO.learning.output_scale == 0.4
    # The search settings are shared: only the scales and the reference differ.
    assert LEGO.learning.hidden == CONFIG.learning.hidden
    assert LEGO.learning.mlp_popsize == CONFIG.learning.mlp_popsize
    for kind in ("linear", "mlp", "warm", "residual"):
        spec = LEGO.build_policy_spec(kind)
        assert spec.scales == LEGO.learning.scales
    residual = LEGO.build_policy_spec("residual")
    np.testing.assert_allclose(residual.output_scale, LEGO.learning.residual_authority * LEGO.robot.max_torque)
    assert residual.residual_scales == (0.25, 0.06)


def test_lego_reference_is_its_hand_tuned_pid_and_provisional():
    assert LEGO.reference_is_provisional and not CONFIG.reference_is_provisional
    assert LEGO.reference_pid_gains() == tuple(LEGO.controller.hand_tuned_gains)
    assert set(LEGO.reference_controllers()) == {"PID, hand-tuned", "PID, model-based"}
    assert all(kind == "cascaded" for kind, _ in LEGO.reference_controllers().values())   # no LQR on this robot
    assert LEGO.build_policy_spec("residual").residual_gains == tuple(LEGO.controller.hand_tuned_gains)


def test_reader_handles_both_tuning_scripts_layouts():
    lqr_run = read_pid_comparison(ROOT / CONFIG.learning.reference_pid_file)
    assert lqr_run["PID, ML-tuned"][0] == "cascaded" and lqr_run["LQR, ML-tuned gains"][0] == "lqr"
    batch_file = sorted((ROOT / "results" / "pololu" / "batch_tuning").glob("*/comparison.json"))[-1]
    batch_run = read_pid_comparison(batch_file)
    assert {"PID, cma-es", "PID, hand-tuned", "PID, model-based"} <= set(batch_run)
    raw = json.loads(batch_file.read_text())["results"]["cma-es"]["gains"]
    assert batch_run["PID, cma-es"] == ("cascaded", tuple(raw[k] for k in ("kp", "ki", "kd", "kv_p")))
    # Once E2 exists, pointing the reference at a run_batch_tuning.py result works the same way.
    tuned = replace(LEGO, learning=replace(LEGO.learning, reference_pid_file=str(batch_file.relative_to(ROOT)),
                                           reference_pid_entry="PID, cma-es"))
    assert not tuned.reference_is_provisional
    assert tuned.reference_pid_gains() == batch_run["PID, cma-es"][1]
    with pytest.raises(KeyError):
        replace(LEGO, learning=replace(LEGO.learning, reference_pid_entry="PID, ML-tuned")).reference_pid_gains()


def test_compare_controllers_under_legobot_uses_its_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_controllers, "CONFIG", LEGO)
    entries = compare_controllers.existing_controllers()
    assert [e.label for e in entries] == ["PID, hand-tuned", "PID, model-based"]
    spec = LEGO.build_policy_spec("linear")
    (tmp_path / "run0").mkdir()
    save_policy(tmp_path / "run0" / "final.npz", spec, pid_equivalent_weights(spec, LEGO.reference_pid_gains()),
                LEGO.robot.max_torque)
    folder = compare_controllers.main(["--tier", "smoke", "--seeds", "1", "--duration", "1.0", "--bootstrap", "50",
                                       "--policies", f"L-PID={tmp_path}/run*/final.npz",
                                       "--out", str(tmp_path / "out")])
    report = json.loads((folder / "comparison.json").read_text())
    assert report["robot"] == "legobot" and report["reference"] == "PID, hand-tuned"
    assert report["reference_provisional"] is True
    assert set(report["methods"]) == {"PID, hand-tuned", "PID, model-based", "L-PID"}
    # Explicitly no existing controllers: then the reference must come from --policies.
    with pytest.raises(SystemExit):
        compare_controllers.main(["--tier", "smoke", "--seeds", "1", "--duration", "1.0", "--existing", "",
                                  "--policies", f"L-PID={tmp_path}/run*/final.npz", "--out", str(tmp_path / "o2")])
