"""optimization/export_policy.py: policy bundles for the robot (hardware/legobot/POLICY_ON_ROBOT_PLAN.md P1).

Synthetic bundles are exported once per module with one envelope seed (3 runs of 60 s each), so the
whole file runs in about a minute.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "optimization", ROOT / "simulation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from balance_sim import load_policy, save_policy  # noqa: E402
from config import load_config  # noqa: E402
from export_policy import (  # noqa: E402
    BUNDLE_FILES, KNOWN_POLICY_KINDS, SYNTHETIC, _Recorder, classify_policy_kind, export_bundle, golden_goal,
    golden_scenario, main, multiply_adds_per_tick, observation_names, parse_args, replay_golden, run_golden_scenario,
    synthetic_policy)

LEGO = load_config("legobot")
QUICK_L_RUN = ROOT / "results" / "legobot" / "e1_policy_dev_preview" / "linear" / "seed0"
COMMITTED = ROOT / "results" / "legobot" / "policies"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_golden(bundle):
    with np.load(bundle / "golden.npz") as f:
        return {k: f[k] for k in f.files}


def _manifest(bundle):
    return json.loads((bundle / "manifest.json").read_text())


def _envelope(bundle):
    return json.loads((bundle / "envelope.json").read_text())


@pytest.fixture(scope="module")
def out_root(tmp_path_factory):
    return tmp_path_factory.mktemp("policies")


@pytest.fixture(scope="module")
def bundles(out_root):
    return {s: export_bundle(synthetic=s, out_root=out_root, cfg=LEGO, envelope_seeds=1) for s in SYNTHETIC}


@pytest.fixture(scope="module")
def run_bundle(tmp_path_factory):
    if not (QUICK_L_RUN / "final.npz").is_file():
        pytest.skip("the quick L run isn't in results/")
    return export_bundle(QUICK_L_RUN, out_root=tmp_path_factory.mktemp("run"), cfg=LEGO, envelope_seeds=1)


def _fake_run(folder, spec=None, params=None, snapshot=None, summary=None, max_torque=None, **extra):
    """A minimal finished-run folder holding l_pid (or the given policy)."""
    if spec is None:
        spec, params = synthetic_policy("l_pid", LEGO)
    folder.mkdir(parents=True, exist_ok=True)
    save_policy(folder / "final.npz", spec, params, LEGO.robot.max_torque if max_torque is None else max_torque,
                **extra)
    (folder / "snapshot.json").write_text(json.dumps(snapshot if snapshot is not None else
                                                     {"robot": "legobot", "config": LEGO.snapshot(),
                                                      "dev_seeds": [5000]}, default=str))
    (folder / "summary.json").write_text(json.dumps(summary if summary is not None else
                                                    {"method": classify_policy_kind(spec), "profile": "quick",
                                                     "selected": {"which": "best", "robot_seconds": 1.0,
                                                                  "dev20_objective": 12.5, "dev20_falls": 0,
                                                                  "dev60_objective": "", "dev60_falls": ""}}))
    return folder


# ---- kinds and sizes ------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KNOWN_POLICY_KINDS)
def test_classify_policy_kind_of_every_config_spec(kind):
    assert classify_policy_kind(LEGO.build_policy_spec(kind)) == kind


def test_classify_policy_kind_refuses_unknown_shapes():
    from dataclasses import replace
    residual = LEGO.build_policy_spec("residual")
    with pytest.raises(ValueError):
        classify_policy_kind(replace(residual, linear_path=True))          # residual with a linear path
    with pytest.raises(ValueError):
        classify_policy_kind(replace(residual, hidden=(), linear_path=True))   # residual without an MLP


def test_multiply_adds_per_tick():
    spec = lambda k: LEGO.build_policy_spec(k)
    mlp = 11 * 16 + 16 * 16 + 16 * 1
    assert multiply_adds_per_tick(spec("linear")) == 10
    assert multiply_adds_per_tick(spec("mlp")) == mlp
    assert multiply_adds_per_tick(spec("warm")) == 11 + mlp
    assert multiply_adds_per_tick(spec("residual")) == 13 * 16 + 16 * 16 + 16


def test_cli_defaults_and_argument_errors():
    args = parse_args(["--synthetic", "l_pid"])
    assert args.robot == "legobot" and args.envelope_seeds == 5 and args.name is None and args.out_root is None
    assert parse_args(["some/run"]).run_folder == "some/run"
    for bad in ([], ["some/run", "--synthetic", "l_pid"], ["--synthetic", "gru"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


# ---- synthetic policies ---------------------------------------------------------------------

def test_synthetic_policies_are_what_they_claim():
    l_spec, l_params = synthetic_policy("l_pid", LEGO)
    w_spec, w_params = synthetic_policy("a_warm_zero", LEGO, seed=0)
    b_spec, b_params = synthetic_policy("b1_zero", LEGO, seed=0)
    assert classify_policy_kind(w_spec) == "warm" and classify_policy_kind(b_spec) == "residual"
    from balance_sim import unpack
    linear, layers = unpack(w_spec, w_params)
    weights = dict(zip(l_spec.inputs, l_params))
    assert list(linear) == [weights.get(i, 0.0) for i in w_spec.inputs]
    for spec, params in ((w_spec, w_params), (b_spec, b_params)):
        _, layers = unpack(spec, params)
        assert all(np.any(W != 0) for W, _ in layers[:-1])                  # Xavier hidden layers
        assert not np.any(layers[-1][0]) and not np.any(layers[-1][1])      # zero output layer
    assert b_spec.residual_gains == tuple(LEGO.reference_pid_gains())
    # Same seed, same weights; another seed, other hidden weights.
    assert np.array_equal(synthetic_policy("b1_zero", LEGO, seed=0)[1], b_params)
    assert not np.array_equal(synthetic_policy("b1_zero", LEGO, seed=1)[1], b_params)
    with pytest.raises(ValueError):
        synthetic_policy("gru", LEGO)


# ---- synthetic bundles ----------------------------------------------------------------------

@pytest.mark.parametrize("synthetic, kind", [("l_pid", "linear"), ("a_warm_zero", "warm"), ("b1_zero", "residual")])
def test_synthetic_bundle_files_and_manifest(bundles, out_root, synthetic, kind):
    bundle = bundles[synthetic]
    assert bundle == out_root / synthetic
    assert all((bundle / f).is_file() for f in BUNDLE_FILES)
    m = _manifest(bundle)
    spec, params, max_torque = load_policy(bundle / "final.npz")
    assert m["name"] == synthetic and m["robot"] == "legobot" and m["kind"] == kind == classify_policy_kind(spec)
    assert m["synthetic"] == synthetic and m["source_run"] is None and m["dev_score"] is None
    assert m["n_params"] == spec.n_params == params.size
    assert m["multiply_adds_per_tick"] == multiply_adds_per_tick(spec)
    assert m["sha256"] == _sha(bundle / "final.npz")
    assert m["git_commit"] is None or isinstance(m["git_commit"], str)
    assert max_torque == LEGO.robot.max_torque
    exp_spec, exp_params = synthetic_policy(synthetic, LEGO)
    assert spec == exp_spec and np.array_equal(params, exp_params)
    snapshot = json.loads((bundle / "snapshot.json").read_text())
    assert snapshot["robot"] == "legobot" and snapshot["synthetic"] == synthetic
    assert LEGO.scoring_differences(snapshot["config"]) == []           # what the robot's config check needs
    summary = json.loads((bundle / "summary.json").read_text())
    assert summary["method"] == kind and summary["reference_gains"] == list(LEGO.reference_pid_gains())


@pytest.mark.parametrize("synthetic", SYNTHETIC)
def test_golden_vectors_replay_bit_for_bit(bundles, synthetic):
    bundle = bundles[synthetic]
    golden = _load_golden(bundle)
    g = golden_scenario(LEGO)
    assert golden["states"].shape == (400, 4) and golden["torques"].shape == golden["goals"].shape == (400,)
    assert float(golden["dt"]) == LEGO.timing.control_dt_s == 0.005
    assert int(golden["seed"]) == LEGO.robustness.dev_seeds[0] == 5000
    assert float(golden["start_pitch"]) == pytest.approx(0.30, abs=1e-5)
    # (The measured pitch starts near 0: the IMU's complementary filter converges from 0 at reset.)
    assert np.max(np.abs(golden["states"][:, 3])) > 0.5                 # recovering from the tilt
    # The three regimes: standing at goal 0, then the cruise goal, with a push at 1.2 s in between.
    assert golden["goals"][0] == 0.0 and golden["goals"][-1] == pytest.approx(g["goal_steps"][0][1])
    assert np.all(np.abs(golden["torques"]) <= LEGO.robot.max_torque) and np.any(golden["torques"] != 0)
    spec, params, _ = load_policy(bundle / "final.npz")
    assert np.array_equal(replay_golden(LEGO, spec, params, golden), golden["torques"])
    # A perturbed weight is caught.
    bumped = params.copy()
    # The last parameter is an output-layer bias (warm/residual: their hidden weights can't move
    # the torque while the output layer is zero) or, for L, the cmd_4 weight (command history is live).
    bumped[-1] += 1e-6
    assert not np.array_equal(replay_golden(LEGO, spec, bumped, golden), golden["torques"])


def test_golden_replay_by_hand_through_a_fresh_controller(bundles):
    """The robot's self-test, written out: set the goal (on the base PID too), call update."""
    bundle = bundles["b1_zero"]
    golden = _load_golden(bundle)
    spec, params, _ = load_policy(bundle / "final.npz")
    c = LEGO.build_policy_controller(spec, params, goal=0.0)
    c.reset()
    out = []
    for s, g in zip(golden["states"], golden["goals"]):
        c.goal = c.base.goal = float(g)
        out.append(c.update(s, float(golden["dt"])))
    assert np.array_equal(np.array(out), golden["torques"])


def test_a_warm_zero_equals_l_pid_bit_for_bit(bundles):
    l, w = _load_golden(bundles["l_pid"]), _load_golden(bundles["a_warm_zero"])
    for key in ("states", "torques", "goals"):
        assert np.array_equal(l[key], w[key]), key
    el, ew = _envelope(bundles["l_pid"]), _envelope(bundles["a_warm_zero"])
    assert el["observations"] == ew["observations"] and el["n_ticks"] == ew["n_ticks"]


def test_b1_zero_equals_the_reference_pid_bit_for_bit(bundles):
    golden = _load_golden(bundles["b1_zero"])
    # Closed loop: the reference PID through the same golden scenario gives the same states and torques.
    states, torques = [], []
    pid = LEGO.build_controller(LEGO.reference_pid_gains(), goal=golden_goal(LEGO))
    run_golden_scenario(LEGO, pid, lambda s, g, u, c: (states.append(np.array(s[:4])), torques.append(u)))
    assert np.array_equal(np.array(states), golden["states"])
    assert np.array_equal(np.array(torques), golden["torques"])
    # Open loop: the PID fed b1_zero's golden states and goals.
    pid = LEGO.build_controller(LEGO.reference_pid_gains(), goal=0.0)
    pid.reset()
    out = []
    for s, g in zip(golden["states"], golden["goals"]):
        pid.goal = float(g)
        out.append(pid.update(s, float(golden["dt"])))
    assert np.array_equal(np.array(out), golden["torques"])


def test_b1_zero_differs_from_l_pid(bundles):
    """Sanity: the comparisons above aren't vacuous (L-PID is close to the PID, not equal)."""
    assert not np.array_equal(_load_golden(bundles["l_pid"])["torques"], _load_golden(bundles["b1_zero"])["torques"])


@pytest.mark.parametrize("synthetic", SYNTHETIC)
def test_envelope_schema_and_bounds(bundles, synthetic):
    env = _envelope(bundles[synthetic])
    spec, _, _ = load_policy(bundles[synthetic] / "final.npz")
    names = observation_names(spec)
    assert env["observation_names"] == names and len(names) == (13 if synthetic == "b1_zero" else 11)
    assert env["n_seeds"] == 1 and env["seeds"] == [5000] and env["duration_s"] == 60.0
    assert env["n_runs"] == 3 and env["n_falls"] == 0
    assert env["n_ticks"] == 3 * round(60.0 / LEGO.timing.control_dt_s)
    assert set(env["observations"]) == set(names)
    for name, o in env["observations"].items():
        assert set(o) >= {"p00_1", "p99_9", "max_abs"}
        assert o["p00_1"] <= o["p99_9"] and o["max_abs"] >= max(abs(o["p00_1"]), abs(o["p99_9"])) >= 0.0, name
    obs = env["observations"]
    goal_peak = max(abs(s[1]) for s in LEGO.scenario.goal_steps)
    assert obs["goal"]["max_abs"] == pytest.approx(goal_peak)
    assert obs["theta_dot_meas"]["max_abs"] > 0.5                       # recovering from the start tilts
    for k in range(1, 5):
        assert obs[f"cmd_{k}"]["max_abs"] <= LEGO.robot.max_torque
    assert obs["int_v"]["max_abs"] <= spec.integral_limits[0] and obs["int_theta"]["max_abs"] <= spec.integral_limits[1]


def test_reexport_overwrites_a_bundle_but_not_other_folders(out_root, bundles):
    sha = _manifest(bundles["l_pid"])["sha256"]
    again = export_bundle(synthetic="l_pid", out_root=out_root, cfg=LEGO, envelope_seeds=1)
    assert _manifest(again)["sha256"] == sha                            # deterministic bytes
    stranger = out_root / "not_a_bundle"
    stranger.mkdir()
    (stranger / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        export_bundle(synthetic="l_pid", name="not_a_bundle", out_root=out_root, cfg=LEGO, envelope_seeds=1)
    assert (stranger / "notes.txt").read_text() == "mine"


# ---- a training run's bundle ----------------------------------------------------------------

def test_trained_run_bundle(run_bundle):
    m = _manifest(run_bundle)
    assert run_bundle.name == "linear-seed0" and m["name"] == "linear-seed0"
    for f in ("final.npz", "snapshot.json", "summary.json"):
        assert (run_bundle / f).read_bytes() == (QUICK_L_RUN / f).read_bytes(), f
    spec, params, _ = load_policy(run_bundle / "final.npz")
    assert m["kind"] == "linear" and m["synthetic"] is None
    assert m["n_params"] == 10 and m["multiply_adds_per_tick"] == 10
    assert m["sha256"] == _sha(QUICK_L_RUN / "final.npz")
    assert m["source_run"] == "results/legobot/e1_policy_dev_preview/linear/seed0"
    summary = json.loads((QUICK_L_RUN / "summary.json").read_text())
    snapshot = json.loads((QUICK_L_RUN / "snapshot.json").read_text())
    sel = summary["selected"]
    assert m["dev_score"]["dev20_objective"] == sel["dev20_objective"]
    assert m["dev_score"]["which"] == sel["which"] and m["dev_score"]["dev_seeds"] == snapshot["dev_seeds"]
    assert m["dev_score"]["dev60_objective"] == (None if sel["dev60_objective"] == "" else sel["dev60_objective"])
    assert isinstance(m["config_differences"], list)
    golden = _load_golden(run_bundle)
    assert golden["torques"].shape == (400,)
    assert np.array_equal(replay_golden(LEGO, spec, params, golden), golden["torques"])
    assert _envelope(run_bundle)["observation_names"] == observation_names(spec)


def test_run_folder_export_via_cli_with_name(tmp_path):
    run = _fake_run(tmp_path / "runs" / "20990101-000000-seed0")
    bundle = main([str(run), "--name", "fake", "--out-root", str(tmp_path / "out"), "--envelope-seeds", "1"])
    assert bundle == tmp_path / "out" / "fake"
    m = _manifest(bundle)
    assert m["kind"] == "linear" and m["source_run"] == "neuroevolution/linear/20990101-000000-seed0"
    assert m["source_path"] == str(run.resolve())
    assert m["dev_score"]["dev20_objective"] == 12.5 and m["dev_score"]["dev60_objective"] is None
    assert m["config_differences"] == []
    assert (bundle / "final.npz").read_bytes() == (run / "final.npz").read_bytes()
    # Exactly what the synthetic l_pid export would have made of the same policy.
    with np.load(bundle / "golden.npz") as f:
        golden = {k: f[k] for k in f.files}
    spec, params, _ = load_policy(bundle / "final.npz")
    assert np.array_equal(replay_golden(LEGO, spec, params, golden), golden["torques"])


def test_source_run_labels(tmp_path):
    run = _fake_run(tmp_path / "train" / "legobot" / "neuroevolution" / "linear" / "quick_linear_seed0")
    b = export_bundle(run, name="a", out_root=tmp_path / "out", cfg=LEGO, envelope_seeds=1)
    assert _manifest(b)["source_run"] == "neuroevolution/linear/quick_linear_seed0"
    b = main([str(run), "--name", "b", "--out-root", str(tmp_path / "out"), "--envelope-seeds", "1",
              "--source-label", "my/label"])
    assert _manifest(b)["source_run"] == "my/label"
    with pytest.raises(SystemExit):
        parse_args(["--synthetic", "l_pid", "--source-label", "x"])

def test_checkpoint_metadata_key_is_accepted(tmp_path):
    run = _fake_run(tmp_path / "run", robot_seconds=5.0)
    export_bundle(run, out_root=tmp_path / "out", cfg=LEGO, envelope_seeds=1)


def test_export_a_checkpoint_instead_of_final(tmp_path):
    """E2.6 can select a checkpoint that is not final.npz; the bundle must hold that checkpoint, byte for byte."""
    run = _fake_run(tmp_path / "run")
    m0 = _manifest(export_bundle(run, name="final", out_root=tmp_path / "out", cfg=LEGO, envelope_seeds=1))
    assert m0["source_checkpoint"] == "final.npz" and m0["dev_score"] is not None
    spec, params = synthetic_policy("l_pid", LEGO)
    other = np.asarray(params, dtype=float) * 1.01
    (run / "checkpoints").mkdir()
    save_policy(run / "checkpoints" / "mean_1e08.npz", spec, other, LEGO.robot.max_torque, robot_seconds=1e8)
    b = main([str(run), "--name", "ck", "--out-root", str(tmp_path / "out"), "--envelope-seeds", "1",
              "--checkpoint", "checkpoints/mean_1e08.npz"])
    m = _manifest(b)
    assert (b / "final.npz").read_bytes() == (run / "checkpoints" / "mean_1e08.npz").read_bytes()
    assert m["sha256"] == _sha(run / "checkpoints" / "mean_1e08.npz") != m0["sha256"]
    assert m["source_checkpoint"] == "checkpoints/mean_1e08.npz"
    assert m["dev_score"] is None                       # summary.json's score is final.npz's, not this one's
    s2, p2, _ = load_policy(b / "final.npz")
    assert np.array_equal(p2, other)
    golden = _load_golden(b)
    assert np.array_equal(replay_golden(LEGO, s2, p2, golden), golden["torques"])
    # A checkpoint byte-identical to final.npz keeps the run's dev score.
    shutil.copyfile(run / "final.npz", run / "checkpoints" / "same.npz")
    m3 = _manifest(export_bundle(run, name="same", out_root=tmp_path / "out", cfg=LEGO, envelope_seeds=1,
                                 checkpoint="checkpoints/same.npz"))
    assert m3["dev_score"] == m0["dev_score"]
    for bad in ("../outside.npz", "checkpoints/missing.npz", "summary.json"):
        with pytest.raises((ValueError, FileNotFoundError)):
            export_bundle(run, name="bad", out_root=tmp_path / "out", cfg=LEGO, envelope_seeds=1, checkpoint=bad)
    with pytest.raises(ValueError):
        export_bundle(synthetic="l_pid", out_root=tmp_path / "out", cfg=LEGO, envelope_seeds=1, checkpoint="x.npz")
    with pytest.raises(SystemExit):
        parse_args(["--synthetic", "l_pid", "--checkpoint", "x.npz"])


@pytest.mark.parametrize("problem", ["unknown_key", "robot", "method", "max_torque", "missing_summary", "nan_param"])
def test_run_folder_refusals(tmp_path, problem):
    folder = tmp_path / "run"
    if problem == "unknown_key":
        _fake_run(folder, hidden_state=np.zeros(4))
    elif problem == "nan_param":
        spec, params = synthetic_policy("l_pid", LEGO)
        params = params.copy()
        params[1] = np.nan
        _fake_run(folder, spec=spec, params=params)
    elif problem == "robot":
        _fake_run(folder, snapshot={"robot": "pololu", "config": {}})
    elif problem == "method":
        _fake_run(folder, summary={"method": "mlp"})
    elif problem == "max_torque":
        _fake_run(folder, max_torque=LEGO.robot.max_torque * 2)
    else:
        _fake_run(folder)
        (folder / "summary.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        export_bundle(folder, out_root=tmp_path / "out", cfg=LEGO, envelope_seeds=1)
    assert not (tmp_path / "out").exists()                              # nothing written


def test_export_bundle_argument_errors(tmp_path):
    with pytest.raises(ValueError):
        export_bundle(out_root=tmp_path)
    with pytest.raises(ValueError):
        export_bundle(QUICK_L_RUN, synthetic="l_pid", out_root=tmp_path)
    with pytest.raises(ValueError):
        export_bundle(synthetic="gru", out_root=tmp_path)
    with pytest.raises(ValueError):
        export_bundle(synthetic="l_pid", name="../escape", out_root=tmp_path)
    with pytest.raises(ValueError):
        export_bundle(synthetic="l_pid", out_root=tmp_path, cfg=LEGO, envelope_seeds=0)


# ---- the bundles committed under results/legobot/policies -----------------------------------

@pytest.mark.parametrize("synthetic", SYNTHETIC)
def test_committed_synthetic_bundles_are_consistent(synthetic):
    bundle = COMMITTED / synthetic
    if not (bundle / "manifest.json").is_file():
        pytest.skip(f"{bundle} not exported")
    m = _manifest(bundle)
    assert all((bundle / f).is_file() for f in BUNDLE_FILES)
    assert m["sha256"] == _sha(bundle / "final.npz") and m["synthetic"] == synthetic
    spec, params, _ = load_policy(bundle / "final.npz")
    exp_spec, exp_params = synthetic_policy(synthetic, LEGO)
    assert spec == exp_spec and np.array_equal(params, exp_params)
    golden = _load_golden(bundle)
    assert np.array_equal(replay_golden(LEGO, spec, params, golden), golden["torques"])
    env = _envelope(bundle)
    assert env["n_seeds"] == 5 and env["n_runs"] == 15 and env["observation_names"] == observation_names(spec)
    assert m["files_sha256"] == {f: _sha(bundle / f) for f in BUNDLE_FILES if f != "manifest.json"}


@pytest.mark.parametrize("synthetic", SYNTHETIC)
def test_manifest_hashes_every_bundle_file(bundles, synthetic):
    bundle = bundles[synthetic]
    m = _manifest(bundle)
    assert set(m["files_sha256"]) == set(BUNDLE_FILES) - {"manifest.json"}
    assert m["files_sha256"] == {f: _sha(bundle / f) for f in m["files_sha256"]}
    assert m["files_sha256"]["final.npz"] == m["sha256"]
    golden = _load_golden(bundle)
    assert m["golden_saturated_ticks"] == int(np.sum(np.abs(golden["torques"]) >= LEGO.robot.max_torque - 1e-12))
    assert all(np.all(np.isfinite(golden[k])) for k in ("states", "torques", "goals"))
