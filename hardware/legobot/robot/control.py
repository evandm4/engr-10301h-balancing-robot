"""The balancing loop's hardware-free core (SOFTWARE_PLAN D1, D3, D4): estimator, controller, feedforward, safety.

Everything the simulator also models comes from the simulator's own code and constants, so the robot
runs what was tuned:
  - the pitch estimate is balance_sim.ComplementaryFilter with legobot_config's time constant and the
    200 Hz control period (B2);
  - wheel speed is the difference of the two newest encoder counts over the HAT's 10 ms sample period,
    as balance_sim.Encoder computes it (B3), and velocity = r (wheel speed + gyro rate) as SensorNoise does;
  - the controller is balance_sim's CascadedPIDController, built by legobot_config's
    ProjectConfig with its torque cap, lean cap and integral limits; or a learned policy, balance_sim's own
    PolicyController (not a port) from an exported bundle, checked against the deployed config and its golden
    vectors before it runs (POLICY_ON_ROBOT_PLAN P2-P5: load_policy_bundle, self_test_policy, EnvelopeMonitor);
  - the duty is the config's actuator model's own drive() (B1, B1b) with its nominal constants and the
    measured battery voltage, per wheel with that wheel's speed: SlowPwmActuator's inverse of the HAT's slow
    PWM ("slow_pwm"), or VoltageModeActuator's voltage divided by vin ("voltage").

Nothing here touches the hardware, so robot.sim_hw can run it in closed loop with the simulator.

Robot code on the Pi imports config.py and legobot_config.py from ~/legobot and balance_sim from
~/legobot/simulation (hardware/legobot/deploy.sh copies them); in the repo they are found from here.
"""
import json
from collections import deque
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

from . import loop_config as lc


DRIVER_MODELS = ("slow_pwm", "voltage")   # actuator models whose drive() the robot can send to the HAT


def import_config():
    """The `config` module (config.py), from the working directory on the Pi or the repo root here."""
    try:
        import config
    except ImportError:
        root = Path(__file__).resolve().parents[3]
        if not (root / "config.py").is_file():
            raise
        sys.path.insert(0, str(root))
        import config
    return config


def load_robot_config():
    """legobot_config's ProjectConfig. Refuses an actuator model the Build HAT can't run (ideal, dc_motor)."""
    cfg = import_config().load_config("legobot")
    if cfg.actuator.model not in DRIVER_MODELS:
        raise ValueError(f"legobot actuator model is {cfg.actuator.model!r}; the robot code runs {DRIVER_MODELS}")
    return cfg


# ------------------------------------------------------------------------------------ controller ---
def load_gains_file(path):
    """A cascaded-PID tuning result JSON: {"gains": {"kp", "ki", "kd", "kv_p"[, "kv_i"]}} (or a list in that
    order). An optional "feedforward_gain" is the robot-code feedforward gain tuned with these gains
    (A4_BASELINES). The LEGO robot runs the cascaded PID only (the LQR was dropped, 2026-11-23)."""
    with open(path) as f:
        data = json.load(f)
    out = {"source": str(path), "feedforward_gain": data.get("feedforward_gain")}
    if "K" in data:
        raise ValueError(f"{path} holds LQR gains; the LEGO robot runs the cascaded PID only")
    g = data["gains"]
    if isinstance(g, dict):
        missing = {"kp", "ki", "kd", "kv_p"} - set(g)
        if missing:
            raise ValueError(f"{path}: gains lack {sorted(missing)} (not a cascaded-PID result?)")
        g = [g["kp"], g["ki"], g["kd"], g["kv_p"]] + ([g["kv_i"]] if "kv_i" in g else [])
    out["kind"], out["gains"] = "pid", [float(x) for x in g]
    return out


# --------------------------------------------------------------------------- learned policies (P2) ---
KNOWN_POLICY_KINDS = ("linear", "mlp", "warm", "residual")
BUNDLE_FILES = ("manifest.json", "final.npz", "snapshot.json", "golden.npz", "envelope.json")
# .npz keys (or key fragments) of a policy with hidden state, which balance_sim.PolicyController can't run
RECURRENT_KEY_FRAGMENTS = ("gru", "lstm", "rnn", "recurrent", "hidden_state")


def classify_policy_kind(spec):
    """'linear' (L), 'mlp' (A), 'warm' (A-warm) or 'residual' (B1) for a balance_sim.PolicySpec; ValueError
    for any other combination (e.g. a residual policy with a linear path)."""
    hidden, linear = bool(spec.hidden), bool(spec.linear_path)
    if spec.residual:
        if hidden and not linear:
            return "residual"
    elif linear and not hidden:
        return "linear"
    elif hidden and not linear:
        return "mlp"
    elif hidden and linear:
        return "warm"
    raise ValueError(f"unsupported policy spec (residual={spec.residual}, hidden={tuple(spec.hidden)}, "
                     f"linear_path={spec.linear_path}); this robot code runs {KNOWN_POLICY_KINDS}")


def is_policy_spec(spec):
    """Whether a --controller value names a policy: policy:<name>, a bundle folder, or a .npz path."""
    s = str(spec)
    return bool(s) and (s.startswith("policy:") or s.lower().endswith(".npz") or Path(s).is_dir())


def resolve_policy_path(path):
    """policy:<name> -> lc.POLICIES_DIR/<name> (~/legobot/policies on the Pi) if it exists, else
    <repo>/results/legobot/policies/<name>; FileNotFoundError if neither. Any other value is a plain path."""
    s = str(path)
    if not s.startswith("policy:"):
        return Path(s)
    name = s[len("policy:"):].strip()
    if not name:
        raise ValueError("policy:<name> needs a name")
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name) or name.endswith("."):
        raise ValueError(f"policy name {name!r} may only use letters, digits, '_', '.' and '-', not start or end "
                         f"with '.' (no paths)")
    candidates = [Path(lc.POLICIES_DIR) / name,
                  Path(__file__).resolve().parents[3] / "results" / "legobot" / "policies" / name]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"policy {name!r} not found in {' or '.join(str(c) for c in candidates)}")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_policy_npz(path):
    """(spec, params, max_torque, kind) from a save_policy .npz, refusing corrupt files, a different
    observation list, recurrent (hidden-state) keys, an unknown or inconsistent "kind" key, a spec kind
    this code doesn't know, and non-finite weights. All refusals are ValueError."""
    import zipfile
    import numpy as np
    import_config()
    from balance_sim import load_policy
    try:
        with np.load(path, allow_pickle=False) as f:
            keys = list(f.files)
            bad = [k for k in keys if any(frag in k.lower() for frag in RECURRENT_KEY_FRAGMENTS)]
            if bad:
                raise ValueError(f"has recurrent-policy keys {bad}: hidden-state policies are not supported")
            stored_kind = str(f["kind"]) if "kind" in keys else None
        spec, params, max_torque = load_policy(path)
    except ValueError as e:
        raise ValueError(f"{path}: refused: {e}") from e
    except (KeyError, OSError, EOFError, zipfile.BadZipFile, TypeError, AttributeError, IndexError) as e:
        raise ValueError(f"{path}: corrupt or not a policy file ({type(e).__name__}: {e})") from e
    if stored_kind is not None and stored_kind not in KNOWN_POLICY_KINDS:
        raise ValueError(f"{path}: unknown policy kind {stored_kind!r}; this robot code runs {KNOWN_POLICY_KINDS}")
    kind = classify_policy_kind(spec)
    if stored_kind is not None and stored_kind != kind:
        raise ValueError(f"{path}: says kind {stored_kind!r} but its spec is {kind!r}")
    params = np.asarray(params, dtype=float)
    if not np.all(np.isfinite(params)) or not math.isfinite(max_torque):
        raise ValueError(f"{path}: non-finite weights or max_torque")
    return spec, params, float(max_torque), kind


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except ValueError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e


def _check_max_torque(path, max_torque, cfg):
    if cfg is not None and abs(max_torque - cfg.robot.max_torque) > 1e-12:
        raise ValueError(f"{path}: policy max_torque {max_torque} N m differs from legobot_config's "
                         f"{cfg.robot.max_torque} N m (refused even with --allow-config-mismatch)")


def _load_golden(path):
    import zipfile
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as f:
            golden = {k: np.array(f[k]) for k in f.files}
    except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile) as e:
        raise ValueError(f"{path}: corrupt golden vectors ({type(e).__name__}: {e})") from e
    missing = {"states", "torques", "goals", "dt"} - set(golden)
    if missing:
        raise ValueError(f"{path}: golden vectors lack {sorted(missing)}")
    n = len(golden["torques"]) if golden["torques"].ndim == 1 else -1
    if n < lc.GOLDEN_MIN_TICKS or golden["states"].shape != (n, 4) or golden["goals"].shape != (n,) or \
            golden["dt"].size != 1 or not float(golden["dt"]) > 0:
        raise ValueError(f"{path}: malformed golden vectors (states {golden['states'].shape}, torques "
                         f"{golden['torques'].shape}, goals {golden['goals'].shape}, dt {golden['dt']}; "
                         f"need >= {lc.GOLDEN_MIN_TICKS} ticks)")
    if not all(np.all(np.isfinite(golden[k])) for k in ("states", "torques", "goals", "dt")):
        raise ValueError(f"{path}: golden vectors hold NaN or inf")
    return golden


def _check_files_sha256(p, manifest):
    """The manifest must hash every bundle file but itself (files_sha256, H5), and each must still match:
    otherwise an edited snapshot, golden or envelope would slip past the config check, self-test or trip."""
    files = manifest.get("files_sha256")
    if not isinstance(files, dict):
        raise ValueError(f"{p}: manifest has no files_sha256 mapping (re-export with optimization/export_policy.py)")
    missing = [f for f in BUNDLE_FILES if f != "manifest.json" and f not in files]
    if missing:
        raise ValueError(f"{p}: manifest files_sha256 does not cover {missing}")
    for name, want in files.items():
        f = p / name
        if Path(name).name != name or not f.is_file():
            raise ValueError(f"{p}: manifest hashes {name!r}, which is not a file in the bundle")
        got = _sha256(f)
        if got != want:
            raise ValueError(f"{f}: sha256 {got} does not match the manifest's {want} (corrupt or edited)")


def _check_envelope(path, envelope, spec):
    import_config()
    from balance_sim.policy import OBSERVATIONS, RESIDUAL_INPUTS
    names = list(OBSERVATIONS + (RESIDUAL_INPUTS if spec.residual else ()))
    if list(envelope.get("observation_names", ())) != names:
        raise ValueError(f"{path}: envelope observation list {envelope.get('observation_names')} is not {names}")
    obs = envelope.get("observations", {})
    for n in names:
        e = obs.get(n)
        if not isinstance(e, dict) or any(not isinstance(e.get(k), (int, float)) or isinstance(e.get(k), bool)
                                          or not math.isfinite(e.get(k)) for k in ("p00_1", "p99_9", "max_abs")):
            raise ValueError(f"{path}: envelope entry for {n!r} lacks finite numeric p00_1, p99_9, max_abs")
        if not (e["p00_1"] <= e["p99_9"] and e["max_abs"] >= 0):
            raise ValueError(f"{path}: envelope entry for {n!r} is inconsistent ({e})")
    if not (isinstance(envelope.get("n_ticks"), (int, float)) and envelope["n_ticks"] > 0):
        raise ValueError(f"{path}: envelope has no ticks (n_ticks {envelope.get('n_ticks')!r})")


def load_policy_bundle(path, cfg=None, *, allow_unchecked=False, allow_config_mismatch=False):
    """A policy bundle (P1: a folder with BUNDLE_FILES, from optimization/export_policy.py), policy:<name>, or a
    bare .npz (only with allow_unchecked). Checks (P2a, P2b), each a ValueError with the reason:
      - final.npz loads with balance_sim.load_policy (same observation list), has no recurrent keys, and its
        spec is a kind this code knows, the manifest's kind;
      - sha256(final.npz) is the manifest's;
      - with cfg: max_torque is legobot_config's (always), and cfg.scoring_differences(snapshot config) is
        empty unless allow_config_mismatch (the differences are returned, for the log).
    Returns a dict: kind, spec, params, max_torque, name, sha256, bundle_dir, manifest, snapshot, golden,
    envelope, config_differences (None if not checked), feedforward_gain (snapshot's; None if unknown),
    unchecked."""
    p = resolve_policy_path(path)
    if p.is_dir():
        missing = [f for f in BUNDLE_FILES if not (p / f).is_file()]
        if missing:
            raise ValueError(f"{p}: not a policy bundle, missing {missing}")
        manifest = _read_json(p / "manifest.json")
        kind = manifest.get("kind")
        if kind not in KNOWN_POLICY_KINDS:
            raise ValueError(f"{p}: unknown policy kind {kind!r}; this robot code runs {KNOWN_POLICY_KINDS}")
        npz = p / "final.npz"
        sha = _sha256(npz)
        if sha != manifest.get("sha256"):
            raise ValueError(f"{npz}: sha256 {sha} does not match the manifest's {manifest.get('sha256')} "
                             f"(corrupt or edited weights)")
        _check_files_sha256(p, manifest)
        spec, params, max_torque, spec_kind = _load_policy_npz(npz)
        if spec_kind != kind:
            raise ValueError(f"{p}: manifest says kind {kind!r} but final.npz holds a {spec_kind!r} policy")
        _check_max_torque(npz, max_torque, cfg)
        snapshot = _read_json(p / "snapshot.json")
        if not isinstance(snapshot.get("config"), dict):
            raise ValueError(f"{p / 'snapshot.json'}: no training config ('config')")
        diffs = None
        if cfg is not None:
            diffs = list(cfg.scoring_differences(snapshot["config"]))
            if diffs and not allow_config_mismatch:
                raise ValueError(f"{p}: trained under a config that differs from legobot_config in {diffs} "
                                 f"(--allow-config-mismatch to run it anyway; logged)")
        try:
            ff_gain = float(snapshot["config"]["actuator"]["voltage_feedforward_gain"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{p / 'snapshot.json'}: no actuator.voltage_feedforward_gain ({e})") from e
        golden = _load_golden(p / "golden.npz")
        if cfg is not None and abs(float(golden["dt"]) - cfg.timing.control_dt_s) > 1e-12:
            raise ValueError(f"{p / 'golden.npz'}: golden dt {float(golden['dt'])} s is not legobot_config's "
                             f"control dt {cfg.timing.control_dt_s} s")
        envelope = _read_json(p / "envelope.json")
        _check_envelope(p / "envelope.json", envelope, spec)
        return {"kind": kind, "spec": spec, "params": params, "max_torque": max_torque,
                "name": manifest.get("name") or p.name, "sha256": sha, "bundle_dir": p, "manifest": manifest,
                "snapshot": snapshot, "golden": golden, "envelope": envelope, "config_differences": diffs,
                "feedforward_gain": ff_gain, "unchecked": False}
    if not p.exists():
        raise FileNotFoundError(f"no policy at {p}")
    if p.suffix.lower() != ".npz":
        raise ValueError(f"{p}: not a policy bundle folder or .npz")
    if not allow_unchecked:
        raise ValueError(f"{p}: a bare .npz has no manifest, sha256, training config, golden vectors or envelope; "
                         f"export a bundle (optimization/export_policy.py) or pass --allow-unchecked")
    spec, params, max_torque, kind = _load_policy_npz(p)
    _check_max_torque(p, max_torque, cfg)
    return {"kind": kind, "spec": spec, "params": params, "max_torque": max_torque, "name": p.stem,
            "sha256": _sha256(p), "bundle_dir": None, "manifest": None, "snapshot": None, "golden": None,
            "envelope": None, "config_differences": None, "feedforward_gain": None, "unchecked": True}


def build_controller(cfg, spec="hand", goal="still", *, allow_unchecked=False, allow_config_mismatch=False,
                     ff_gain_override=None):
    """The controller and a small JSON-able description of it. spec:
      "hand" / "model"   the cascaded PID with legobot's hand-tuned / model-based gains;
      policy:<name>, a bundle folder or a .npz   a learned policy (load_policy_bundle), built by
                         cfg.build_policy_controller; the bundle is attached as controller.bundle;
      anything else      a cascaded-PID gains JSON path (load_gains_file).
    goal: "still" (hold 0 m/s) or "scenario" (the tuning scenario's goal profile, from arming). It is passed
    when building, so a residual policy's base PID gets it too (plan problem 1). ff_gain_override (--ff-gain)
    is refused for a policy: the feedforward gain is part of the plant it was trained on (P2c)."""
    if goal not in ("still", "scenario"):
        raise ValueError(f"goal must be 'still' or 'scenario', not {goal!r}")
    goal_val = 0.0 if goal == "still" else cfg.build_goal()
    if spec not in ("hand", "model") and is_policy_spec(spec):
        if ff_gain_override is not None:
            raise ValueError("--ff-gain is refused for a policy: it runs its training config's feedforward gain")
        bundle = load_policy_bundle(spec, cfg, allow_unchecked=allow_unchecked,
                                    allow_config_mismatch=allow_config_mismatch)
        controller = cfg.build_policy_controller(bundle["spec"], bundle["params"], goal=goal_val)
        controller.bundle = bundle
        desc = {
            "spec": str(spec),
            "kind": "policy",
            "policy_kind": bundle["kind"],
            "bundle_name": bundle["name"],
            "bundle_dir": str(bundle["bundle_dir"]) if bundle["bundle_dir"] else None,
            "sha256": bundle["sha256"],
            "n_params": bundle["spec"].n_params,
            "goal": goal,
            "feedforward_gain": bundle["feedforward_gain"],
            "config_differences": bundle["config_differences"],
            "unchecked": bundle["unchecked"],
        }
        return controller, desc
    ff_gain = None
    if spec == "hand":
        gains = list(cfg.controller.hand_tuned_gains)
    elif spec == "model":
        gains = list(cfg.model_based_gains())
    else:
        loaded = load_gains_file(spec)
        gains, ff_gain = loaded["gains"], loaded["feedforward_gain"]
    controller = cfg.build_controller(gains, goal=goal_val)
    return controller, {"spec": spec, "kind": "pid", "gains": gains, "goal": goal, "feedforward_gain": ff_gain}


def _quantile(sorted_xs, q):
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


def _time_updates(controller, states, dt, goals=None, base=None):
    """(torques, per-tick seconds) of controller.update over states, goals set per tick if given. gc is off
    while timing, as it is in the loop."""
    import gc
    import time
    torques, times = [], []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for k in range(len(states)):
            if goals is not None:
                controller.goal = goals[k]
                if base is not None:
                    base.goal = goals[k]
            s = states[k]
            t0 = time.perf_counter()
            u = controller.update(s, dt)
            times.append(time.perf_counter() - t0)
            torques.append(u)
    finally:
        if was_enabled:
            gc.enable()
    return torques, times


def self_test_policy(controller, golden, cfg=None, tol=lc.GOLDEN_TOL):
    """P2d, P3b: replay the bundle's golden vectors (states and goals into update(), the torques the laptop's
    reference run returned) through the controller; ValueError if any torque differs by more than tol N m (a
    NaN counts as a failure). The controller is reset and its goal (and a residual base PID's) restored
    afterwards, pass or fail. Also times the forward pass, and with cfg a fresh cascaded PID over the same
    states, for the compute lead: lc.COMPUTE_LEAD_S + max(0, policy p99 - PID p99)."""
    import numpy as np
    states = np.asarray(golden["states"], dtype=float)
    expected = np.asarray(golden["torques"], dtype=float)
    goals = [float(g) for g in np.asarray(golden["goals"], dtype=float)]
    dt = float(golden["dt"])
    n = len(expected)
    if n == 0 or states.shape != (n, 4) or len(goals) != n:
        raise ValueError(f"malformed golden vectors: states {states.shape}, torques {expected.shape}, goals {len(goals)}")
    base = getattr(controller, "base", None)
    orig_goal = controller.goal
    orig_base_goal = base.goal if base is not None else None
    try:
        controller.reset()
        torques, times = _time_updates(controller, states, dt, goals, base)
    finally:
        controller.goal = orig_goal
        if base is not None:
            base.goal = orig_base_goal
        controller.reset()
    errs = [abs(float(u) - float(e)) for u, e in zip(torques, expected)]
    bad = [k for k, err in enumerate(errs) if not err <= tol]          # catches NaN
    max_err = math.inf if any(not math.isfinite(x) for x in errs) else max(errs)
    if bad:
        k = bad[0]
        raise ValueError(f"policy self-test failed: {len(bad)} of {n} golden torques off by more than {tol:g} N m "
                         f"(max {max_err:.3g}); first at tick {k}: {float(torques[k])!r} vs {float(expected[k])!r} "
                         f"(a last-bit tanh difference at the saturation edge can flip the integrator freeze; "
                         f"anything else means the weights or the code differ from the laptop's)")
    ts = sorted(times)
    p99 = _quantile(ts, 0.99)
    if p99 > lc.POLICY_MAX_P99_FRACTION * dt:
        raise ValueError(f"policy self-test: forward pass p99 {p99 * 1e3:.3f} ms exceeds "
                         f"{lc.POLICY_MAX_P99_FRACTION:g} x dt ({dt * 1e3:g} ms): too slow for this loop")
    pid_p99 = None
    if cfg is not None:
        _, pid_times = _time_updates(cfg.build_controller(goal=0.0), states, dt)
        pid_p99 = _quantile(sorted(pid_times), 0.99)
    lead = lc.COMPUTE_LEAD_S + (max(0.0, p99 - pid_p99) if pid_p99 is not None else p99)
    return {"n_ticks": n, "max_torque_err": float(max_err), "mean_s": sum(ts) / n, "p50_s": _quantile(ts, 0.5),
            "p99_s": p99, "max_s": ts[-1], "pid_p99_s": pid_p99, "compute_lead_s": lead}


class EnvelopeMonitor:
    """P4c, P5a: each acted tick's raw policy observation (PolicyController.last_raw) against the bundle's
    envelope (envelope.json, P1c). Counts, per input, ticks outside [p0.1, p99.9], ticks past max |value|,
    non-finite values, and the worst |value|. An input whose envelope max |value| is 0 (never seen nonzero in
    the reference runs) counts any nonzero value as outside and past max. With trip_k, observe() returns a trip
    reason once an input the policy reads (active_names, default all; spec.input_names) has stayed past
    trip_k x its max |value| for trip_s (rounded to ticks, at least one), or at once on a NaN/inf in one;
    otherwise None."""

    def __init__(self, envelope, dt, trip_k=None, trip_s=lc.ENVELOPE_TRIP_S, active_names=None):
        if trip_k is not None and not (math.isfinite(trip_k) and trip_k > 0):
            raise ValueError(f"envelope trip K must be a positive number, got {trip_k}")
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.envelope = envelope
        self.names = list(envelope["observation_names"])
        obs = envelope["observations"]
        self.lo = [float(obs[n]["p00_1"]) for n in self.names]
        self.hi = [float(obs[n]["p99_9"]) for n in self.names]
        self.max_abs = [float(obs[n]["max_abs"]) for n in self.names]
        if active_names is None:
            active_names = self.names
        unknown = set(active_names) - set(self.names)
        if unknown:
            raise ValueError(f"policy inputs {sorted(unknown)} are not in the envelope")
        self.active = [n in set(active_names) for n in self.names]
        self.trip_k, self.trip_s, self.dt = trip_k, trip_s, dt
        self.trip_ticks = max(1, int(round(trip_s / dt)))
        self._limit = [None if trip_k is None or not act else trip_k * m
                       for m, act in zip(self.max_abs, self.active)]
        m = len(self.names)
        self.n_ticks = 0
        self.n_outside = [0] * m
        self.n_past_max = [0] * m
        self.n_nonfinite = [0] * m
        self.worst_abs = [0.0] * m
        self.worst_val = [0.0] * m
        self.consecutive = [0] * m
        self.tripped = None

    def observe(self, raw):
        vals = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if len(vals) != len(self.names):
            raise ValueError(f"observation has {len(vals)} values, the envelope {len(self.names)}")
        self.n_ticks += 1
        reason = None
        for i, v in enumerate(vals):
            v = float(v)
            finite = math.isfinite(v)
            a = abs(v)
            if not finite:
                self.n_nonfinite[i] += 1
            if not (self.lo[i] <= v <= self.hi[i]):
                self.n_outside[i] += 1
            if not a <= self.max_abs[i]:                 # max 0: any nonzero value; NaN counts too
                self.n_past_max[i] += 1
            if not finite:
                if self.worst_abs[i] != math.inf:
                    self.worst_abs[i], self.worst_val[i] = math.inf, v
            elif a > self.worst_abs[i]:
                self.worst_abs[i], self.worst_val[i] = a, v
            if self.trip_k is None or not self.active[i]:
                continue
            if not finite:
                self.consecutive[i] = self.trip_ticks
                if reason is None:
                    reason = f"input {self.names[i]} out of envelope (non-finite value {v})"
                continue
            lim = self._limit[i]
            if lim is not None and a > lim:
                self.consecutive[i] += 1
                if reason is None and self.consecutive[i] >= self.trip_ticks:
                    reason = (f"input {self.names[i]} out of envelope ({v:+.4g}, > {self.trip_k:g} x max "
                              f"{self.max_abs[i]:.4g} for {self.trip_s:g} s)")
            else:
                self.consecutive[i] = 0
        if reason is not None and self.tripped is None:
            self.tripped = reason
        return reason

    def summary_lines(self):
        e = self.envelope
        n = self.n_ticks
        pct = lambda c: 100.0 * c / n if n else 0.0
        trip = "off" if self.trip_k is None else f"{self.trip_k:g} x max |value| for {self.trip_s:g} s"
        lines = [f"envelope ({e.get('n_runs', '?')} reference runs x {e.get('duration_s', '?')} s, seeds "
                 f"{e.get('seeds', '?')}): {n} acted ticks checked; trip {trip}"]
        for i, name in enumerate(self.names):
            m = self.max_abs[i]
            ratio = f"{self.worst_abs[i] / m:.2f} x max" if m > 0 else "max 0"
            nonfinite = f"; {self.n_nonfinite[i]} non-finite" if self.n_nonfinite[i] else ""
            lines.append(f"  {name}: outside [{self.lo[i]:+.4g}, {self.hi[i]:+.4g}] {pct(self.n_outside[i]):.2f}% "
                         f"of ticks; past max |{m:.4g}| {pct(self.n_past_max[i]):.2f}%; worst {self.worst_val[i]:+.4g} "
                         f"({ratio}){nonfinite}")
        if self.tripped:
            lines.append(f"envelope trip: {self.tripped}")
        return lines


# ------------------------------------------------------------------------------------- sensing ---
def wheel_speed(pair, period):
    """Relative wheel speed (rad/s) from a (previous, latest) MotorReading pair: the count difference over
    one HAT sample period, as balance_sim.Encoder.speed. 0 until two readings exist.

    Consecutive data lines are consecutive HAT samples (the HAT samples every `period`); arrival times
    jitter by up to 20 ms when the reader thread is late, so they are not used to scale the difference."""
    prev, latest = pair
    if prev is None or latest is None:
        return 0.0
    return math.radians(latest.pos - prev.pos) / period


def wheel_angle(pair):
    latest = pair[1]
    return 0.0 if latest is None else math.radians(latest.pos)


def accel_tilt(acc_fwd, acc_up):
    """Accelerometer tilt, theta > 0 leaning forward (imu.pitch_from_accel)."""
    return math.atan2(-acc_fwd, acc_up)


class PitchEstimator:
    """Gyro bias and balance-point zero removed, then the simulator's complementary filter (B2)."""

    def __init__(self, cfg, gyro_bias=0.0, pitch_zero=0.0):
        sim = import_config()  # noqa: F841  (makes balance_sim importable)
        from balance_sim import ComplementaryFilter
        self.filter = ComplementaryFilter(cfg.sensor.filter_time_constant, cfg.timing.control_dt_s)
        self.gyro_bias = gyro_bias
        self.pitch_zero = pitch_zero

    def update(self, pitch_rate, acc_fwd, acc_up):
        """Returns (pitch estimate, bias-corrected gyro rate, zeroed accelerometer tilt)."""
        gyro = pitch_rate - self.gyro_bias
        tilt = accel_tilt(acc_fwd, acc_up) - self.pitch_zero
        return self.filter.update(gyro, tilt), gyro, tilt


# ---------------------------------------------------------------------------------- actuation ---
class Feedforward:
    """Torque command -> duty per wheel, by the config actuator model's drive() with the config's nominal
    motor constants, the robot's torque cap, and the config's feedforward on/off and gain. Per wheel, with
    a^ = eta^ k^ / R^ and w that wheel's measured speed:

      "slow_pwm" (B1b, pwmparams 1 0.01): d = (tau_c / n) / (a^ (V -+ gain k^ w)), - for tau >= 0, + below
      "voltage"  (B1):  V_drive = R^ tau_c / (n eta^ k^) + gain k^ w, clipped to +-V;  d = V_drive / V

    Each wheel gets half the torque (the n) and its own back-EMF. V is the battery voltage the HAT reports
    (set_supply) times port_plimit, so the duty uses the real voltage as the sim's true supply does."""

    def __init__(self, cfg, gain=None, plimit=1.0):
        sim = import_config()  # noqa: F841
        import balance_sim
        self.model = cfg.actuator.model
        if self.model not in DRIVER_MODELS:
            raise ValueError(f"no robot-side driver for actuator model {self.model!r}")
        self._cls = balance_sim.SlowPwmActuator if self.model == "slow_pwm" else balance_sim.VoltageModeActuator
        self._per_volt = self.model == "voltage"     # its drive() returns volts, not a duty
        self.motor = cfg.actuator.build_motor_params()
        self.max_torque = cfg.build_robot_params().max_torque
        self.enabled = cfg.actuator.voltage_feedforward
        self.gain = cfg.actuator.voltage_feedforward_gain if gain is None else float(gain)
        self.plimit = plimit
        self.set_supply(cfg.actuator.supply_voltage)

    def set_supply(self, vin):
        """Use battery voltage vin for the duty scale; a non-finite or non-positive reading is ignored (the
        last good one stays). Returns whether it was used."""
        try:
            vin = float(vin)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(vin) and vin > 0):
            return False
        self.supply = vin * self.plimit
        m = replace(self.motor, supply_voltage=self.supply)
        self.actuator = self._cls(m, m, max_torque=self.max_torque, feedforward=self.enabled,
                                  feedforward_gain=self.gain)
        return True

    def duties(self, torque, w_left, w_right):
        d = self.actuator.drive
        if self._per_volt:
            return d(torque, w_left) / self.supply, d(torque, w_right) / self.supply
        return d(torque, w_left), d(torque, w_right)


# ------------------------------------------------------------------------------------- the brain ---
class Brain:
    """One control tick: sense() turns raw readings into the controller's state, act() into duties."""

    def __init__(self, cfg, controller, estimator, feedforward, duty_cap, hat_period):
        self.cfg = cfg
        self.dt = cfg.timing.control_dt_s
        self.r = cfg.robot.wheel_radius
        self.controller = controller
        self.estimator = estimator
        self.feedforward = feedforward
        self.duty_cap = duty_cap
        self.hat_period = hat_period

    def sense(self, pitch_rate, acc_fwd, acc_up, pair_left, pair_right):
        """Returns (state [x, v, theta, theta_dot], extras dict) in the simulator's conventions."""
        theta, gyro, tilt = self.estimator.update(pitch_rate, acc_fwd, acc_up)
        wl, wr = wheel_speed(pair_left, self.hat_period), wheel_speed(pair_right, self.hat_period)
        angle = 0.5 * (wheel_angle(pair_left) + wheel_angle(pair_right))
        v = self.r * (0.5 * (wl + wr) + gyro)
        x = self.r * (angle + theta)
        return [x, v, theta, gyro], {"tilt": tilt, "w_left": wl, "w_right": wr}

    def act(self, state, w_left, w_right):
        """Returns (torque command, duty_left, duty_right), duties clamped to duty_cap. A non-finite duty is
        returned as is (clamping would turn a NaN into +duty_cap), for the loop to trip on."""
        torque = self.controller.update(state, self.dt)
        dl, dr = self.feedforward.duties(torque, w_left, w_right)
        cap = self.duty_cap
        clamp = lambda d: max(-cap, min(cap, d)) if math.isfinite(d) else d
        return torque, clamp(dl), clamp(dr)


# ------------------------------------------------------------------------- HAT line clock (D2) ---
class LineClock:
    """The Build HAT's 100 Hz data-line grid, estimated from line arrival times (perf_counter s).

    The HAT samples and sends a port's line on its own clock (about 880 ppm off the Pi's), and in slow PWM
    its pulse ticks are locked to the same grid (C3, tick_lock.py). Each arrival is the grid time plus a
    transit delay plus non-negative jitter (sd about 0.65 ms; up to 20 ms when the reader thread is late),
    so the grid is the LOWER envelope of the arrivals:

        period  = least-squares slope of arrival time against line index (index by rounding to the
                  previous period estimate, so dropped lines don't shift it);
        t_ref   = the `envelope` quantile of arrival - period x index, placed at the newest line's index.

    locked() is true once `min_lines` lines are in and the newest line at the last update is under `stale` s
    old: longer than the loop's update interval (100 ms), since stale motor data trips the run at 50 ms anyway
    (Safety). Pure Python
    and cheap enough (about 0.2 ms for 256 lines on the Pi 3B) to update every few ticks."""

    def __init__(self, nominal_period=0.010, window=256, min_lines=50, envelope=0.10, stale=0.25):
        self.period = nominal_period
        self.window, self.min_lines, self.envelope, self.stale = window, min_lines, envelope, stale
        self.smoothing = 0.1                  # each window's slope is good to about +-55 ppm (jitter sd 0.65 ms)
        self.t_ref = None
        self.newest = None
        self.n = 0

    def update(self, arrivals):
        a = arrivals[-self.window:]
        if len(a) < self.min_lines:
            return False
        p = self.period
        ref = a[0] if self.t_ref is None else self.t_ref
        # index each line by rounding against the median residual, so one late line can't shift the indices
        k0 = [round((t - ref) / p) for t in a]
        r0 = sorted(t - ref - p * k for k, t in zip(k0, a))
        ref += r0[len(r0) // 2]
        ks = [round((t - ref) / p) for t in a]
        n = len(a)
        km = sum(ks) / n
        tm = sum(a) / n
        skk = sum((k - km) ** 2 for k in ks)
        if skk <= 0:
            return False
        period = sum((k - km) * (t - tm) for k, t in zip(ks, a)) / skk
        if not (0.98 * p < period < 1.02 * p):
            return False                      # nonsense (e.g. a stream restart): keep the old estimate
        if self.t_ref is not None:            # after the first lock, smooth the period across updates
            period = p + self.smoothing * (period - p)
        resid = sorted(t - period * k for k, t in zip(ks, a))
        base = resid[min(n - 1, int(self.envelope * n))]
        self.period, self.t_ref, self.newest, self.n = period, base + period * ks[-1], a[-1], n
        return True

    def locked(self, now):
        return self.t_ref is not None and now - self.newest < self.stale

    def phase(self, t):
        """Time since the latest grid point at or before t (0 <= phase < period)."""
        return (t - self.t_ref) % self.period


def port_lag(ref_times, other_times, nominal_period):
    """How far the other port's line grid lags the reference port's, in [0, period) s; None if too few lines.

    Each port streams on its own grid, started by configure(), so the lag changes from run to run (1.5-8 ms
    in the 2026-11-23 logs). Logged as a diagnostic only: the pulse ticks don't follow their own port's grid,
    and neither port's tick holds a fixed phase between runs (B1 follow-up and cable-swap sweeps, 2026-11-23:
    port A 8.8 ms, port B 7.2 / 4.3 ms, then both 3.2 ms after re-plugging the motors, one power-up)."""
    a, b = LineClock(nominal_period), LineClock(nominal_period)
    if not (a.update(ref_times) and b.update(other_times)):
        return None
    return (b.t_ref - a.t_ref) % a.period


class Pacer:
    """The loop's tick times. Locked: two ticks per HAT line period, at `offset` s after each half-period
    point of the LineClock grid, so every command lands at the same phase of the HAT's pulse ticks.
    Unlocked (no lock yet, or lines stale): ticks every dt, as before."""

    def __init__(self, dt, line_clock=None, offset=0.0):
        self.dt, self.lc, self.offset = dt, line_clock, offset

    def next(self, prev, now):
        lc = self.lc
        if lc is None or not lc.locked(now):
            return prev + self.dt, False
        half = 0.5 * lc.period
        base = lc.t_ref + self.offset
        m = math.ceil((prev + 0.5 * half - base) / half)   # the first grid tick more than half a tick after prev
        return base + m * half, True


# --------------------------------------------------------------------------------------- safety ---
def runaway_speed_for_vin(vin):
    """Wheel speed (rad/s) at which the runaway trip fires: RUNAWAY_FRACTION of the no-load speed at this
    supply. In slow PWM at duty 1 the no-load speed is (a V - tau_c) / (a k + b) with a = 0.0597 N m/V,
    k = 0.348 V s/rad, b = 0.00085 N m s/rad and the Coulomb term 0.021 N m per motor (DATASHEET 8.7): 17.8 rad/s
    at 6.8 V, 20.0 at 7.6, 22.2 at 8.4. A fixed threshold is too high on a sagging battery and too low on a
    full one (review H6). Falls back to lc.RUNAWAY_SPEED when vin is unknown."""
    if vin is None:
        return lc.RUNAWAY_SPEED
    vin = max(float(vin), lc.VIN_CUTOFF)    # a glitched low reading (0 V) must not push the threshold to zero or below
    return lc.RUNAWAY_FRACTION * (0.0597 * vin - 0.021) / (0.0597 * 0.348 + 0.00085)


class Safety:
    """The per-tick trips of D3 that need no clock of their own. check() returns a reason string or None.
    The loop itself handles the loop-stall trip and the watchdog thread (balance.py)."""

    def __init__(self, dt, tilt_limit=lc.TILT_LIMIT, vin_cutoff=lc.VIN_CUTOFF, stall_duty=lc.STALL_DUTY, vin=None):
        self.dt = dt
        self.stall_duty = stall_duty
        self.tilt_limit = tilt_limit
        self.vin_cutoff = vin_cutoff
        n = max(1, int(round(lc.STALL_S / dt)))
        self._stall = [deque(maxlen=n), deque(maxlen=n)]   # speed in the duty's direction, while |duty| is high
        self.wrong_way_ticks = [0, 0]
        m = max(1, int(round(lc.RUNAWAY_S / dt)))
        self._runaway = [deque(maxlen=m), deque(maxlen=m)]  # each wheel's signed speed over the last RUNAWAY_S
        self.vin_low = 0
        self.vin = float(vin) if vin is not None and 5.0 <= vin <= 10.0 else None   # latest vin, for runaway threshold
        self._last_vin_t = None
        self.first_vin_t = None

    def check(self, t, theta, duties, speeds, motor_ages, vin_reading, fault, reader_alive, running):
        if fault is not None:
            return f"motor power fault: {fault[1]!r}"
        if not reader_alive:
            return "serial reader thread died"
        for side, age in zip(("left", "right"), motor_ages):
            if age is None or age > lc.MOTOR_STALE_S:
                return f"{side} motor data stale ({'none' if age is None else f'{age * 1e3:.0f} ms'})"
        if vin_reading is not None and 5.0 <= vin_reading[1] <= 10.0:   # ignore an implausible reading here
            self.vin = vin_reading[1]
        if vin_reading is not None and vin_reading[0] != self._last_vin_t:
            self._last_vin_t = vin_reading[0]
            self.vin_low = self.vin_low + 1 if vin_reading[1] < self.vin_cutoff else 0
            if self.vin_low >= lc.VIN_LOW_COUNT:
                return f"battery low: {vin_reading[1]:.2f} V < {self.vin_cutoff} V"
        if self.first_vin_t is None:
            self.first_vin_t = t
        last_vin = self.first_vin_t if vin_reading is None else max(vin_reading[0], self.first_vin_t)
        if t - last_vin > lc.VIN_STALE_S:
            return "no vin reply from the HAT"
        if not running:
            return None
        if abs(theta) > self.tilt_limit:
            return f"tilt {theta:+.2f} rad past {self.tilt_limit} rad"
        for i, side in enumerate(("left", "right")):
            d, w = duties[i], speeds[i]
            # Stall: |duty| high on every tick of the last STALL_S, and the wheel's MEAN speed in the duty's
            # direction under STALL_SPEED. A mean, because a held wheel still creeps or is back-driven by single
            # encoder counts (1.75 rad/s each), which reset any per-tick test (bench, 2026-11-23).
            q = self._stall[i]
            if abs(d) >= self.stall_duty:
                q.append(w if d > 0 else -w)
                if len(q) == q.maxlen:
                    mean = sum(q) / len(q)
                    if mean < lc.STALL_SPEED:
                        return (f"{side} wheel stalled (duty {d:+.2f}, mean {mean:+.2f} rad/s in its direction "
                                f"over {lc.STALL_S} s)")
            else:
                q.clear()
            # Encoder against drive: a strong duty and the wheel turning fast the other way means a sign
            # error, which makes the feedforward positive feedback (D4's sign check, at run time).
            self.wrong_way_ticks[i] = (self.wrong_way_ticks[i] + 1
                                       if abs(d) >= self.stall_duty and d * w < 0 and abs(w) > 5.0 else 0)
            if self.wrong_way_ticks[i] * self.dt >= 0.2:
                return f"{side} wheel turns against its duty (duty {d:+.2f}, {w:+.2f} rad/s): sign error?"
        # Runaway: both wheels' MEAN speed over the last RUNAWAY_S near the no-load speed, the same way. On the
        # floor that is past 0.59 m/s for 1.5 s, beyond anything the scenario asks; lifted while armed (held
        # upright, so no tilt trip) the free wheels spin up to the no-load speed and the stall trip can't see it
        # (review G11). A mean, because one encoder count per tick is 3.5 rad/s: at 19.4 rad/s, 4.4% of ticks read
        # under 17 and reset a per-tick counter, which delayed the trip to ~4 s (bench A5, 2026-11-23). The speed is
        # 0.85 of the no-load speed at the measured vin (runaway_speed_for_vin), not a fixed 17 rad/s.
        for q, w in zip(self._runaway, speeds):
            q.append(w)
        ql, qr = self._runaway
        if len(ql) == ql.maxlen:
            ml, mr = sum(ql) / len(ql), sum(qr) / len(qr)
            if min(abs(ml), abs(mr)) >= runaway_speed_for_vin(self.vin) and ml * mr > 0:
                return f"wheels running away (mean {ml:+.1f}, {mr:+.1f} rad/s over {lc.RUNAWAY_S} s): lifted?"
        return None


# ------------------------------------------------------------------------------------ calibration ---
def load_calibration(path):
    """{"pitch_zero": rad, "port_sign": {...}, ...} written by robot.calibrate; None if absent."""
    p = Path(path)
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def gyro_bias_from(samples):
    """samples: (pitch_rate, acc_fwd, acc_up) with the robot still. Returns (bias, sd, |a|/g mean) or
    raises ValueError if the robot moved (D4)."""
    n = len(samples)
    if n < 50:
        raise ValueError(f"only {n} IMU samples for the gyro bias")
    rates = [s[0] for s in samples]
    mean = sum(rates) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rates) / (n - 1))
    g = sum(math.hypot(s[1], s[2]) for s in samples) / n / 9.80665
    if sd > lc.GYRO_CAL_MAX_SD:
        raise ValueError(f"robot moved during gyro calibration (pitch rate sd {sd:.4f} rad/s)")
    if abs(mean) > lc.GYRO_CAL_MAX_BIAS:
        raise ValueError(f"gyro bias {mean:.4f} rad/s is implausibly large (robot turning?)")
    if abs(g - 1.0) > lc.ACCEL_G_TOLERANCE:
        raise ValueError(f"accelerometer reads {g:.3f} g in the pitch plane (robot moving or not upright-ish)")
    return mean, sd, g
