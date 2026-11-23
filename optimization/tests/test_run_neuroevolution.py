"""optimization/run_neuroevolution.py end to end, on tiny NumPy settings: a run finishes and saves
what it should, and a run stopped part-way and resumed matches one that was never interrupted
(docs/ml-baselines-plan.md 3.3 and 3.7)."""

import csv
import json
import pickle

import numpy as np
import pytest

from balance_sim import load_policy
from config import CONFIG
from power import POWER_FIELDS
from run_neuroevolution import Setup, parse_args, run

TINY = ["--backend", "numpy", "--train-duration", "0.4", "--dev-duration", "0.4", "--train-seeds", "1",
        "--dev-seeds", "1", "--popsize", "6", "--generations", "4", "--run-seed", "3"]


def _history(folder):
    with open(folder / "history.csv") as f:
        # Wall-clock and the GPU's power state differ between runs; everything else must not.
        skip = {"elapsed_s", "generation_s", *POWER_FIELDS}
        return [{k: v for k, v in row.items() if k not in skip} for row in csv.DictReader(f)]


@pytest.mark.parametrize("method", ["linear", "residual"])
def test_run_saves_policies_checkpoints_and_summary(tmp_path, method):
    folder = run(["--method", method, *TINY, "--out", str(tmp_path / method)])
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["generations"] == 4 and summary["stop_reason"] == "generation cap"
    assert summary["n_evals"] == 1 + 4 * 6
    spec, params, _ = load_policy(folder / "final.npz")
    assert spec.n_params == params.size == (10 if method == "linear" else 513)
    snapshot = json.loads((folder / "snapshot.json").read_text())
    assert snapshot["dev_seeds"] == [5000] and snapshot["train_seeds"] == [1]
    assert len(_history(folder)) == 4


def test_resumed_run_matches_an_uninterrupted_one(tmp_path):
    whole = run(["--method", "linear", *TINY, "--out", str(tmp_path / "whole")])
    part = run(["--method", "linear", *TINY, "--out", str(tmp_path / "part"), "--stop-after", "2"])
    assert not (part / "summary.json").exists()
    assert len(_history(part)) == 2
    run(["--resume", str(part)])
    assert _history(part) == _history(whole)
    np.testing.assert_array_equal(load_policy(part / "final.npz")[1], load_policy(whole / "final.npz")[1])


def test_warm_start_begins_exactly_at_the_linear_policy(tmp_path):
    linear = run(["--method", "linear", *TINY, "--out", str(tmp_path / "linear")])
    warm = run(["--method", "warm", *TINY, "--generations", "1", "--warm-from", str(linear / "final.npz"),
                "--out", str(tmp_path / "warm")])
    summary = json.loads((warm / "summary.json").read_text())
    assert summary["n_params"] == 492
    assert summary["charged_before"] == json.loads((linear / "summary.json").read_text())["robot_seconds_total"]
    # Generation 0 is L's final policy plus an MLP whose output layer is zero: the same torques,
    # so exactly the same training score.
    with open(warm / "state.pkl", "rb") as f:
        start_cost = pickle.load(f)["search"]["history"][0][1]
    l_setup = Setup(parse_args(["--method", "linear", *TINY]))
    l_cost = l_setup.train.evaluate_policies(load_policy(linear / "final.npz")[1][None, :]).objective[0]
    assert start_cost == l_cost


def test_training_output_root(monkeypatch, tmp_path):
    """Runs write outside the synced repository: $BALANCEBOT_TRAINING_DIR first, then the
    configured folder where its drive exists, else results/."""
    monkeypatch.setenv("BALANCEBOT_TRAINING_DIR", str(tmp_path))
    assert CONFIG.training_output_root() == tmp_path
    monkeypatch.delenv("BALANCEBOT_TRAINING_DIR")
    root = CONFIG.training_output_root()
    assert root.name in ("balancebot_training", "results")
    assert "OneDrive" not in str(root) or root.name == "results"
