"""Offline tick-by-tick replay verification for logged LEGO robot policy runs (POLICY_ON_ROBOT_PLAN P4b).

Feeds the logged state rows from a hardware or sim_hw policy run through a fresh
``balance_sim.PolicyController`` built from the same policy bundle, checking that every acted
tick reproduces the logged torque (and, when ``_raw.csv`` is present, the raw observation
vector, pre-clip output, saturation/freeze flags, and residual torque) within ``tol``.

Usage:
    python -m robot.replay_policy logs/<stamp>_<label> [--bundle <dir_or_npz>] [--tol 1e-12]
    python hardware/legobot/robot/replay_policy.py logs/<stamp>_<label>_run.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__:
    from . import control
else:
    _hw_legobot = str(Path(__file__).resolve().parents[1])
    if _hw_legobot not in sys.path:
        sys.path.insert(0, _hw_legobot)
    from robot import control


def _resolve_log_paths(log_path) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve ``<base>``, ``<base>_run.csv``, or ``<base>_raw.csv`` to ``(run_csv, raw_csv)``."""
    p = Path(log_path)
    text = str(p)
    base: Optional[str] = None
    for suffix in ("_run.csv", "_raw.csv", "_summary.txt"):
        if text.endswith(suffix):
            base = text[:-len(suffix)]
            break
    if base is None:
        base = text

    run_csv = Path(base + "_run.csv")
    raw_csv = Path(base + "_raw.csv")
    has_run = run_csv.is_file()
    has_raw = raw_csv.is_file()

    if not has_run and not has_raw and p.is_file():
        with open(p, encoding="utf-8") as f:
            first = f.readline().strip()
        if first.startswith("#") or "x_dot" in first:
            return p, None
        if first.startswith("tick,"):
            return None, p
        return p, None

    if not has_run and not has_raw:
        raise FileNotFoundError(f"neither {run_csv} nor {raw_csv} exists for {log_path!r}")

    return (run_csv if has_run else None), (raw_csv if has_raw else None)


def _resolve_bundle_target(bundle_path, ctrl_meta: dict):
    if bundle_path is not None:
        return bundle_path

    if not ctrl_meta:
        raise ValueError(
            "cannot resolve policy bundle without controller metadata in _run.csv; pass --bundle"
        )

    ctrl_kind = ctrl_meta.get("kind")
    if ctrl_kind is not None and ctrl_kind != "policy":
        raise ValueError(f"log controller kind is {ctrl_kind!r}, not 'policy'")

    bundle_dir = ctrl_meta.get("bundle_dir")
    if bundle_dir and Path(bundle_dir).exists():
        return bundle_dir

    bundle_name = ctrl_meta.get("bundle_name")
    if bundle_name and not ctrl_meta.get("unchecked"):
        return f"policy:{bundle_name}"

    spec_str = ctrl_meta.get("spec")
    if spec_str:
        return spec_str

    if bundle_name:
        return f"policy:{bundle_name}"

    raise ValueError("log controller metadata lacks bundle_dir, bundle_name, and spec; pass --bundle")


def _parse_bool_flag(raw_value: str, field_name: str, tick: int) -> bool:
    text = raw_value.strip().lower()
    if text in ("1", "true"):
        return True
    if text in ("0", "false"):
        return False
    raise ValueError(f"tick {tick}: invalid boolean value {raw_value!r} in column {field_name!r}")


def replay_log(
    log_path,
    *,
    bundle_path=None,
    tol: float = 1e-12,
    allow_unchecked: bool = False,
    allow_config_mismatch: bool = True,
) -> dict:
    """Replay every acted tick of a logged policy run through a fresh PolicyController.

    Accepts ``<base>``, ``<base>_run.csv``, or ``<base>_raw.csv`` (resolving sibling
    ``_run.csv`` and ``_raw.csv`` automatically when present). Loads the policy via
    ``control.load_policy_bundle`` and raises ``ValueError`` if the bundle fails
    verification, the log has 0 acted ticks, or any replayed value disagrees with the
    logged value by more than ``tol``.
    """
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tol must be finite and >= 0, got {tol!r}")

    run_csv, raw_csv = _resolve_log_paths(log_path)
    cfg = control.load_robot_config()
    control.import_config()
    from balance_sim import OBSERVATIONS, RESIDUAL_INPUTS, load_trajectory

    result = None
    meta: dict = {}
    if run_csv is not None:
        result, meta = load_trajectory(run_csv)

    ctrl_meta = meta.get("controller") if isinstance(meta.get("controller"), dict) else {}
    target = _resolve_bundle_target(bundle_path, ctrl_meta)
    bundle = control.load_policy_bundle(
        target,
        cfg,
        allow_unchecked=allow_unchecked,
        allow_config_mismatch=allow_config_mismatch,
    )

    expected_sha = ctrl_meta.get("sha256")
    if expected_sha is not None and bundle["sha256"] != expected_sha:
        raise ValueError(
            f"bundle sha256 mismatch: loaded {bundle['sha256']}, log recorded {expected_sha}"
        )
    expected_kind = ctrl_meta.get("policy_kind")
    if expected_kind is not None and bundle["kind"] != expected_kind:
        raise ValueError(
            f"policy kind mismatch: loaded {bundle['kind']!r}, log recorded {expected_kind!r}"
        )

    spec = bundle["spec"]
    obs_names = tuple(OBSERVATIONS) + (tuple(RESIDUAL_INPUTS) if spec.residual else ())

    acted_raw_rows: Optional[list[dict]] = None
    if raw_csv is not None:
        with open(raw_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or ())
            required_cols = [f"obs_{n}" for n in obs_names] + [
                "pre_clip",
                "saturated",
                "integrators_frozen",
                "residual_torque",
            ]
            missing_cols = [c for c in required_cols if c not in fieldnames]
            if missing_cols:
                raise ValueError(f"{raw_csv} is missing policy columns: {missing_cols}")

            acted_raw_rows = []
            seen_non_acted_in_phase1 = False
            for row in reader:
                if row.get("phase") != "1":
                    continue
                if row.get("pre_clip", "") != "":
                    if seen_non_acted_in_phase1:
                        raise ValueError(
                            f"{raw_csv} has an acted phase-1 row after a non-acted phase-1 row"
                        )
                    acted_raw_rows.append(row)
                else:
                    seen_non_acted_in_phase1 = True

    if result is not None:
        if "acted_ticks" in meta and meta["acted_ticks"] is not None:
            n_acted = int(meta["acted_ticks"])
            if n_acted < 0 or n_acted > len(result.t):
                raise ValueError(
                    f"invalid acted_ticks={n_acted} for _run.csv with {len(result.t)} rows"
                )
            if acted_raw_rows is not None and len(acted_raw_rows) != n_acted:
                raise ValueError(
                    f"acted_ticks mismatch: _run.csv meta has {n_acted}, "
                    f"_raw.csv has {len(acted_raw_rows)} acted rows"
                )
        elif acted_raw_rows is not None:
            n_acted = len(acted_raw_rows)
            if n_acted > len(result.t):
                raise ValueError(
                    f"_raw.csv has {n_acted} acted rows, more than _run.csv's {len(result.t)} rows"
                )
        else:
            n_acted = len(result.t)
            reason = meta.get("reason")
            # Without acted_ticks or _raw.csv, a pre-act trip leaves one trailing 0-torque row.
            if n_acted > 0 and reason is not None and reason != "done" and float(result.torques[-1]) == 0.0:
                n_acted -= 1
    else:
        assert acted_raw_rows is not None
        n_acted = len(acted_raw_rows)

    if n_acted == 0:
        raise ValueError(f"log {log_path!r} has 0 acted ticks; nothing to replay")

    if not np.all(np.isfinite(bundle["params"])):
        raise ValueError("loaded policy bundle contains non-finite parameters")

    controller = cfg.build_policy_controller(spec, bundle["params"], goal=0.0)
    controller.reset()
    dt = cfg.timing.control_dt_s

    max_torque_diff = 0.0
    max_obs_diff: Optional[float] = 0.0 if acted_raw_rows is not None else None
    max_pre_clip_diff: Optional[float] = 0.0 if acted_raw_rows is not None else None
    max_residual_diff: Optional[float] = 0.0 if acted_raw_rows is not None else None
    first_failure: Optional[str] = None

    for k in range(n_acted):
        if result is not None:
            state = np.asarray(result.states[k], dtype=float)
            goal_k = (
                float(acted_raw_rows[k]["obs_goal"])
                if acted_raw_rows is not None
                else float(result.velocity_goal[k])
            )
        else:
            r = acted_raw_rows[k]
            state = np.array(
                [float(r["x"]), float(r["v"]), float(r["theta"]), float(r["gyro"])],
                dtype=float,
            )
            goal_k = float(r["obs_goal"])

        if not np.all(np.isfinite(state)) or not math.isfinite(goal_k):
            raise ValueError(f"tick {k}: non-finite input state={state.tolist()} or goal={goal_k!r}")

        controller.goal = goal_k
        if controller.base is not None:
            controller.base.goal = goal_k

        replayed_torque = float(controller.update(state, dt))
        if not math.isfinite(replayed_torque):
            raise ValueError(f"tick {k}: replayed torque is non-finite ({replayed_torque!r})")

        if result is not None:
            logged_torque = float(result.torques[k])
            diff = abs(replayed_torque - logged_torque)
            if not (math.isfinite(diff) and diff <= tol):
                if first_failure is None:
                    first_failure = (
                        f"tick {k}: torque mismatch (replayed={replayed_torque:.15g}, "
                        f"logged={logged_torque:.15g}, diff={diff:.3e} > tol={tol:.3e})"
                    )
            max_torque_diff = math.inf if not math.isfinite(diff) else max(max_torque_diff, diff)

        if acted_raw_rows is not None:
            r = acted_raw_rows[k]

            raw_torque = float(r["torque"])
            raw_diff = abs(replayed_torque - raw_torque)
            if not (math.isfinite(raw_diff) and raw_diff <= tol):
                if first_failure is None:
                    first_failure = (
                        f"tick {k}: _raw.csv torque mismatch (replayed={replayed_torque:.15g}, "
                        f"logged={raw_torque:.15g}, diff={raw_diff:.3e} > tol={tol:.3e})"
                    )
            max_torque_diff = math.inf if not math.isfinite(raw_diff) else max(max_torque_diff, raw_diff)

            for j, name in enumerate(obs_names):
                logged_obs = float(r[f"obs_{name}"])
                replayed_obs = float(controller.last_raw[j])
                odiff = abs(replayed_obs - logged_obs)
                if not (math.isfinite(odiff) and odiff <= tol):
                    if first_failure is None:
                        first_failure = (
                            f"tick {k}: obs_{name} mismatch (replayed={replayed_obs:.15g}, "
                            f"logged={logged_obs:.15g}, diff={odiff:.3e} > tol={tol:.3e})"
                        )
                max_obs_diff = math.inf if not math.isfinite(odiff) else max(max_obs_diff, odiff)

            logged_pre = float(r["pre_clip"])
            pdiff = abs(float(controller.last_pre_clip) - logged_pre)
            if not (math.isfinite(pdiff) and pdiff <= tol):
                if first_failure is None:
                    first_failure = (
                        f"tick {k}: pre_clip mismatch (replayed={controller.last_pre_clip:.15g}, "
                        f"logged={logged_pre:.15g}, diff={pdiff:.3e} > tol={tol:.3e})"
                    )
            max_pre_clip_diff = math.inf if not math.isfinite(pdiff) else max(max_pre_clip_diff, pdiff)

            logged_sat = _parse_bool_flag(r["saturated"], "saturated", k)
            if bool(controller.last_saturated) != logged_sat:
                if first_failure is None:
                    note = (
                        " (near saturation boundary; cross-arch tanh rounding can flip this flag)"
                        if abs(abs(float(controller.last_pre_clip)) - controller.max_torque) <= tol
                        else ""
                    )
                    first_failure = (
                        f"tick {k}: saturated mismatch (replayed={int(controller.last_saturated)}, "
                        f"logged={int(logged_sat)}){note}"
                    )

            logged_frozen = _parse_bool_flag(r["integrators_frozen"], "integrators_frozen", k)
            if bool(controller.last_integrators_frozen) != logged_frozen:
                if first_failure is None:
                    first_failure = (
                        f"tick {k}: integrators_frozen mismatch "
                        f"(replayed={int(controller.last_integrators_frozen)}, logged={int(logged_frozen)})"
                    )

            logged_res = float(r["residual_torque"])
            rdiff = abs(float(controller.last_residual) - logged_res)
            if not (math.isfinite(rdiff) and rdiff <= tol):
                if first_failure is None:
                    first_failure = (
                        f"tick {k}: residual_torque mismatch (replayed={controller.last_residual:.15g}, "
                        f"logged={logged_res:.15g}, diff={rdiff:.3e} > tol={tol:.3e})"
                    )
            max_residual_diff = math.inf if not math.isfinite(rdiff) else max(max_residual_diff, rdiff)

    if first_failure is not None:
        raise ValueError(f"replay verification failed: {first_failure}")

    return {
        "n_ticks": n_acted,
        "max_torque_diff": max_torque_diff,
        "max_obs_diff": max_obs_diff,
        "max_pre_clip_diff": max_pre_clip_diff,
        "max_residual_diff": max_residual_diff,
        "bundle_name": bundle["name"],
        "sha256": bundle["sha256"],
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", help="log base path, _run.csv, or _raw.csv")
    ap.add_argument("--bundle", default=None, help="policy bundle directory, policy:<name>, or bare .npz")
    ap.add_argument("--tol", type=float, default=1e-12, help="max allowed absolute torque/obs diff (default 1e-12)")
    ap.add_argument("--allow-unchecked", action="store_true", help="allow replaying a bare .npz policy")
    ap.add_argument(
        "--strict-config",
        action="store_true",
        help="refuse replay if the bundle's snapshot config differs from legobot_config",
    )
    args = ap.parse_args(argv)

    try:
        report = replay_log(
            args.log,
            bundle_path=args.bundle,
            tol=args.tol,
            allow_unchecked=args.allow_unchecked,
            allow_config_mismatch=not args.strict_config,
        )
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print(f"REPLAY FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

    parts = [
        f"REPLAY OK: {report['bundle_name']} (sha256={report['sha256'][:12]}...)",
        f"ticks={report['n_ticks']}",
        f"max_torque_diff={report['max_torque_diff']:.3e}",
    ]
    if report["max_obs_diff"] is not None:
        parts.append(f"max_obs_diff={report['max_obs_diff']:.3e}")
        parts.append(f"max_pre_clip_diff={report['max_pre_clip_diff']:.3e}")
    print("  ".join(parts))
    return report


if __name__ == "__main__":
    main()
