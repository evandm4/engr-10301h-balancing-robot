"""E4 comparison hooks: score real/sim_hw runs with the simulator's cost and generate counterbalanced run sheets (POLICY_ON_ROBOT_PLAN P8).

Usage:
    python -m robot.score_run logs/*_run.csv [--duration 20] [--json scores.json]
    python -m robot.score_run --run-sheet hand policy:l_pid policy:b1_zero --blocks 4 --goal still --seconds 20 [--out sheet.md]
    python hardware/legobot/robot/score_run.py logs/*_run.csv
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics as st
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

if __package__:
    from . import control
else:
    _hw_legobot = str(Path(__file__).resolve().parents[1])
    if _hw_legobot not in sys.path:
        sys.path.insert(0, _hw_legobot)
    from robot import control


def _resolve_run_csv(path) -> Path:
    """Resolve ``<base>``, ``<base>_run.csv``, or ``<base>_raw.csv`` to a ``_run.csv`` file."""
    p = Path(path)
    text = str(p)
    if text.endswith("_raw.csv"):
        sibling = Path(text[:-len("_raw.csv")] + "_run.csv")
        if sibling.is_file():
            return sibling
    if text.endswith("_summary.txt"):
        sibling = Path(text[:-len("_summary.txt")] + "_run.csv")
        if sibling.is_file():
            return sibling
    if p.is_file():
        return p
    candidate = Path(text + "_run.csv")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"run CSV not found for {path!r}")


def _controller_name(meta: dict) -> str:
    ctrl = meta.get("controller")
    if isinstance(ctrl, dict):
        if ctrl.get("kind") == "policy":
            return str(ctrl.get("bundle_name") or ctrl.get("spec") or "policy")
        return str(ctrl.get("spec") or ctrl.get("kind") or "unknown")
    if ctrl is not None:
        return str(ctrl)
    return "unknown"


def score_trajectory(path, cfg=None, *, duration: Optional[float] = None) -> dict:
    """Score a logged ``_run.csv`` trajectory using ``cfg.score(result)`` and ``balance_sim.compute_metrics``."""
    run_path = _resolve_run_csv(path)
    if cfg is None:
        cfg = control.load_robot_config()
    control.import_config()
    from balance_sim import compute_metrics, load_trajectory

    result, meta = load_trajectory(run_path)
    if len(result.t) == 0:
        raise ValueError(f"trajectory {run_path} has 0 rows")

    if duration is not None:
        duration_f = float(duration)
        if not math.isfinite(duration_f) or duration_f <= 0.0:
            raise ValueError(f"duration must be finite and positive, got {duration!r}")
        result = replace(result, duration=duration_f)
    elif result.fell and (not math.isfinite(result.duration) or result.duration <= 0.0):
        raise ValueError(
            f"fallen trajectory {run_path} has non-positive duration={result.duration!r}; pass --duration"
        )

    metrics = compute_metrics(result)
    run_cost = float(cfg.score(result))
    reason = meta.get("reason")
    complete = bool(result.fell or reason == "done")

    return {
        "path": str(run_path),
        "label": str(meta.get("label", run_path.stem)),
        "controller": _controller_name(meta),
        "reason": reason,
        "fell": bool(metrics.fell),
        "complete": complete,
        "fall_time": None if metrics.fall_time is None else float(metrics.fall_time),
        "duration": float(result.duration),
        "cost": run_cost,
        "rms_pitch": float(metrics.rms_pitch),
        "max_abs_pitch": float(metrics.max_abs_pitch),
        "velocity_rms_error": float(metrics.velocity_rms_error),
        "effort": float(metrics.effort),
        "drift": float(metrics.drift),
        "settling_time": None if metrics.settling_time is None else float(metrics.settling_time),
    }


def _safe_label(controller: str, block: int, goal: str, run_index: int) -> str:
    name = controller
    if name.startswith("policy:"):
        name = name[len("policy:"):]
    else:
        p = Path(name)
        if "/" in name or "\\" in name or name.endswith((".json", ".npz")):
            name = p.stem if p.suffix in (".json", ".npz") else p.name
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "ctrl"
    return f"r{run_index:02d}_b{block}_{slug}_{goal}"


def generate_run_sheet(
    controllers: Sequence[str],
    *,
    blocks: int = 4,
    goal: str = "still",
    seconds: float = 20.0,
    alternate: bool = True,
) -> list[dict]:
    """Generate a counterbalanced E4 comparison run sheet within one battery charge.

    With ``alternate=True``, odd 0-based blocks reverse controller order (ABBA: block 1
    ``A, B, C``, block 2 ``C, B, A``, block 3 ``A, B, C``, block 4 ``C, B, A``, ...) so
    a linear/monotonic battery voltage sag cancels exactly across controllers over an even
    number of blocks (default ``blocks=4``). Using an odd block count with ``alternate=True``
    and multiple controllers emits a ``UserWarning`` because linear trend cancellation is
    only exact for even block counts.
    """
    ctrl_list = [str(c) for c in controllers]
    if not ctrl_list:
        raise ValueError("controllers must be non-empty")
    if blocks < 1:
        raise ValueError(f"blocks must be >= 1, got {blocks!r}")
    if goal not in ("still", "scenario"):
        raise ValueError(f"goal must be 'still' or 'scenario', got {goal!r}")
    seconds_f = float(seconds)
    if not math.isfinite(seconds_f) or seconds_f <= 0.0:
        raise ValueError(f"seconds must be finite and positive, got {seconds!r}")
    if alternate and blocks % 2 == 1 and len(ctrl_list) > 1:
        warnings.warn(
            f"generate_run_sheet: blocks={blocks} is odd with alternate=True; "
            f"ABBA order only cancels a linear battery-sag trend exactly over an even number of blocks",
            UserWarning,
            stacklevel=2,
        )

    sheet: list[dict] = []
    run_index = 0
    for b in range(blocks):
        block_num = b + 1
        order = list(reversed(ctrl_list)) if (alternate and (b % 2 == 1)) else list(ctrl_list)
        for ctrl in order:
            run_index += 1
            label = _safe_label(ctrl, block_num, goal, run_index)
            cmd = (
                f"robot/run_safe.sh robot.balance --controller {shlex.quote(ctrl)} "
                f"--goal {goal} --seconds {seconds_f:g} --label {label}"
            )
            sheet.append(
                {
                    "run_index": run_index,
                    "block": block_num,
                    "controller": ctrl,
                    "label": label,
                    "goal": goal,
                    "seconds": seconds_f,
                    "command": cmd,
                }
            )
    return sheet


def format_scores_table(scores: Sequence[dict]) -> str:
    headers = (
        "label",
        "controller",
        "reason",
        "status",
        "dur(s)",
        "cost",
        "rms_th",
        "max_th",
        "v_rms",
        "effort",
        "drift",
    )
    rows = []
    for s in scores:
        if s["fell"]:
            status = "FELL"
        elif not s.get("complete", True):
            status = "INCOMPLETE"
        else:
            status = "ok"
        rows.append(
            (
                str(s["label"]),
                str(s["controller"]),
                str(s["reason"] or "-"),
                status,
                f"{s['duration']:.2f}",
                f"{s['cost']:.4f}",
                f"{s['rms_pitch']:.4f}",
                f"{s['max_abs_pitch']:.4f}",
                f"{s['velocity_rms_error']:.4f}",
                f"{s['effort']:.4f}",
                f"{s['drift']:+.3f}",
            )
        )
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" if i < 4 else f"{{:>{w}}}" for i, w in enumerate(widths))
    lines = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt.format(*r) for r in rows)

    by_ctrl: dict[str, list[dict]] = {}
    for s in scores:
        by_ctrl.setdefault(s["controller"], []).append(s)
    if len(scores) > 1 and any(len(v) > 1 for v in by_ctrl.values()):
        lines.append("")
        lines.append("Per-controller summary:")
        for ctrl, items in by_ctrl.items():
            falls = sum(1 for x in items if x["fell"])
            incompletes = sum(1 for x in items if not x.get("complete", True))
            complete_items = [x for x in items if x.get("complete", True)]
            if complete_items:
                costs = [x["cost"] for x in complete_items]
                cost_str = f"mean_cost={st.fmean(costs):.4f}, median_cost={st.median(costs):.4f}"
            else:
                cost_str = "mean_cost=n/a (all incomplete)"
            extra = f", incomplete={incompletes}" if incompletes else ""
            lines.append(f"  {ctrl}: n={len(items)}, falls={falls}{extra}, {cost_str}")
    return "\n".join(lines)


def format_run_sheet_markdown(sheet: Sequence[dict]) -> str:
    if not sheet:
        return ""
    goal = sheet[0]["goal"]
    seconds = sheet[0]["seconds"]
    blocks = max(e["block"] for e in sheet)
    lines = [
        f"# E4 Comparison Run Sheet ({blocks} blocks, goal={goal}, {seconds:g} s/run)",
        "",
        "| Run | Block | Controller | Label | Command |",
        "|---:|---:|---|---|---|",
    ]
    for e in sheet:
        lines.append(
            f"| {e['run_index']} | {e['block']} | `{e['controller']}` | `{e['label']}` | `{e['command']}` |"
        )

    by_ctrl: dict[str, list[int]] = {}
    for e in sheet:
        by_ctrl.setdefault(e["controller"], []).append(e["run_index"])
    lines.append("")
    lines.append("Mean run index per controller (battery-sag balance):")
    means = []
    for ctrl, idxs in by_ctrl.items():
        m = st.fmean(idxs)
        means.append(m)
        lines.append(f"- `{ctrl}`: mean={m:.2f} (runs {idxs})")
    if len(by_ctrl) > 1 and max(means) - min(means) > 1e-9:
        lines.append(
            f"\n> Note: blocks={blocks} is unbalanced (mean run index spread "
            f"{max(means) - min(means):.2f}); use an even block count with alternating order "
            f"to cancel a linear battery-sag trend exactly."
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="_run.csv file(s) or log base path(s) to score")
    ap.add_argument("--duration", type=float, default=None, help="override requested run duration (s) for fall penalty")
    ap.add_argument("--json", dest="json_out", default=None, help="write scored metrics list as JSON")
    ap.add_argument(
        "--run-sheet",
        nargs="+",
        metavar="CTRL",
        default=None,
        help="generate an alternating-order comparison run sheet for these controllers",
    )
    ap.add_argument("--blocks", type=int, default=4, help="number of blocks for --run-sheet (default 4, even for exact linear balance)")
    ap.add_argument("--goal", choices=("still", "scenario"), default="still", help="goal for --run-sheet")
    ap.add_argument("--seconds", type=float, default=20.0, help="seconds per run for --run-sheet (default 20)")
    ap.add_argument("--no-alternate", action="store_true", help="keep fixed controller order across blocks")
    ap.add_argument("--out", default=None, help="write --run-sheet output to .md or .json file")
    args = ap.parse_args(argv)

    if args.run_sheet and args.runs:
        ap.error("cannot combine positional run CSVs with --run-sheet")
    if not args.run_sheet and not args.runs:
        ap.error("provide one or more _run.csv files to score, or --run-sheet <ctrl> ...")

    try:
        if args.run_sheet:
            sheet = generate_run_sheet(
                args.run_sheet,
                blocks=args.blocks,
                goal=args.goal,
                seconds=args.seconds,
                alternate=not args.no_alternate,
            )
            md = format_run_sheet_markdown(sheet)
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.suffix.lower() == ".json":
                    out_path.write_text(json.dumps(sheet, indent=2) + "\n", encoding="utf-8")
                else:
                    out_path.write_text(md, encoding="utf-8")
            print(md, end="")
            return sheet

        cfg = control.load_robot_config()
        scores = [score_trajectory(p, cfg=cfg, duration=args.duration) for p in args.runs]
        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
        print(format_scores_table(scores))
        return scores
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
