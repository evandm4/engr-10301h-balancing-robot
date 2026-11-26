"""E3 session metrics: the numbers pre-registered in hardware/legobot/E3_SESSION1_PREREG.md, from the robot's logs.

Every metric here is defined in the prereg's section 2 ("Notation" and the H tables) and section 4 (analysis plan).
This script only computes them; it does not change a definition. If a definition and this code disagree, the doc wins.

Per run (section 2 notation; all from the robot's own estimates, not truth):
  - theta_mean  (theta-bar): mean of `theta` over the armed run (`_run.csv`).
  - v_mean      (v-bar):     net change of `x` divided by the run's duration (t_last - t_first).
  - cost, fell:              `score_run.py`'s `cost` and `fell` (imported, not reimplemented).
  - theta p-p:               peak-to-peak theta in consecutive 2 s windows (H-B4), in degrees.
  - plateaus:                each maximal run of constant `velocity_goal` longer than 0.5 s; error = mean `x_dot`
                             minus the goal over the whole plateau (H-C1).
  - wheel ratio:             (pos_right change) / (pos_left change) over the armed rows of `_raw.csv` (H-C2). The
                             HAT reader already applies hw_config.PORT_SIGN, so both are forward-positive. Also given
                             for the cruise plateau (the largest positive goal).
  - band fraction:           theta power in 3.3-4.3 Hz over power above 0.25 Hz, per consecutive 4 s window (mean
                             removed, Hann window), plus the dominant frequency (above 0.25 Hz) per window (H-C3).

Hypothesis tests (`--block A|B|C`, thresholds copied from the section 2 tables):
  A: H-A1 theta-bar(A1) in [+0.007, +0.017]; H-A2 |v-bar| < 0.005 and |theta-bar| < 0.004 in A2 (A5 reported);
     H-A3 OLS of v-bar on theta-bar over A1, A2, A3+, A3-, A5, slope in [-1.10, -0.60] and v-bar(A3+) > 0 > v-bar(A3-);
     H-A4 delta = theta-bar(A5) - theta-bar(A2): |delta| > 0.01 supported, < 0.005 not supported, else inconclusive;
     H-A5 0.0008 rad/degC x the largest SoC temperature change < |delta| / 2 (needs temp_before/temp_after columns).
  B: H-B1/H-B2 exact one-sided Mann-Whitney L < P and L < M, Holm at alpha = 0.05, plus median(L) < median(other);
     only once every controller has n >= 6 (before that: interim descriptives, no test, section 0);
     H-B3 median(P) / median(M) vs [0.85, 1.15] (descriptive); H-B4 pooled median theta p-p (descriptive);
     run-to-run sd per controller.
  C: H-C1 |plateau error| <= 0.05 m/s on every plateau of every L run; H-C2 wheel ratio < 1 in every L run (Evan's
     visual right curve is the other half of the verdict); H-C3 band fraction <= 0.20 in every window and no cruise
     window (4-8, 8-12 s) peaking in 3.3-4.3 Hz; H-C4 median `wobble_ratio_real_over_free.theta` from sim_vs_real's
     `<stamp>_metrics_T1.json` in [1.3, 1.7] (weak: [1.0, 2.0]); H-C5 M runs fall-free (cost comparison descriptive).

Usage:
    python hardware/legobot/robot/session_metrics.py <stamp-or-label ...> [--logdir DIR] [--sheet runsheet.csv]
        [--block A|B|C] [--json out.json] [--svr-dir DIR]
Run sheet (CSV, header row): `stamp` required; optional `label`, `controller` (L/P/M), `run` (e.g. A3p), `block`,
`t_max` (crop the run to its first t_max s, theta/x/raw metrics only), `excluded` (non-empty = the logged exclusion
reason, section 4.4), `temp_before`, `temp_after` (degC). With --sheet and no positional runs, every sheet row is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

if __package__:
    from . import score_run
else:
    _hw_legobot = str(Path(__file__).resolve().parents[1])
    if _hw_legobot not in sys.path:
        sys.path.insert(0, _hw_legobot)
    from robot import score_run

DEFAULT_LOGDIR = Path(r"C:\balancebot_training\legobot\pi_logs")
DEFAULT_SVR_DIR = Path(r"C:\balancebot\reports\sim_vs_real")

# Thresholds, prereg section 2.
H_A1_RANGE = (0.007, 0.017)
H_A2_V, H_A2_TH = 0.005, 0.004
H_A3_SLOPE = (-1.10, -0.60)
H_A4_SUPPORT, H_A4_NULL = 0.01, 0.005
H_A5_TEMPCO = 0.0008            # rad per degC
ALPHA = 0.05
B_FULL_N = 6
H_B3_BAND = (0.85, 1.15)
H_C1_TOL = 0.05
H_C3_MAX = 0.20
H_C4_PASS, H_C4_WEAK = (1.3, 1.7), (1.0, 2.0)
# Metric definitions, prereg section 2.
PP_WINDOW_S = 2.0
BAND_WINDOW_S = 4.0
BAND_HZ = (3.3, 4.3)
TOTAL_MIN_HZ = 0.25
PLATEAU_MIN_S = 0.5
CRUISE_WINDOWS = ((4.0, 8.0), (8.0, 12.0))

CONTROLLER_CODES = {"l_e2_s1": "L", "l_pid": "P", "model": "M"}
NAME_RE = re.compile(r"^(\d{8}_\d{6})_(.+)_run\.csv$")


# ---------------------------------------------------------------------------------------------- loading ---
def read_run_csv(path):
    """(meta dict, structured float64 array) from a `_run.csv` with its `# {json}` header line."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    meta = json.loads(first[1:]) if first.startswith("#") else {}
    data = np.genfromtxt(path, delimiter=",", names=True, skip_header=1 if meta else 0, dtype=np.float64)
    return meta, np.atleast_1d(data)


def read_raw_armed(run_path):
    """Armed rows (phase == 1) of the sibling `_raw.csv`: dict of t (from arming), pos_left, pos_right, goal."""
    raw_path = Path(str(run_path)[: -len("_run.csv")] + "_raw.csv")
    if not raw_path.is_file():
        return None
    a = np.atleast_1d(np.genfromtxt(raw_path, delimiter=",", names=True, dtype=np.float64,
                                    usecols=("t", "phase", "pos_left", "pos_right", "goal")))
    m = a["phase"] == 1
    if not m.any():
        return None
    t = a["t"][m]
    return {"t": t - t[0], "pos_left": a["pos_left"][m], "pos_right": a["pos_right"][m], "goal": a["goal"][m]}


def index_logs(logdir):
    """{(stamp, label): path} for every `_run.csv` under logdir (recursive); the first copy of a duplicate wins."""
    out = {}
    for p in sorted(Path(logdir).rglob("*_run.csv")):
        m = NAME_RE.match(p.name)
        if m and (m.group(1), m.group(2)) not in out:
            out[(m.group(1), m.group(2))] = p
    return out


def find_runs(keys, logdir):
    """Resolve stamps (`20261126_011006` or `011006`), labels, `<stamp>_<label>` or file paths to `_run.csv` paths."""
    index, found = None, []
    for key in keys:
        if Path(key).is_file():
            found.append(Path(key))
            continue
        index = index if index is not None else index_logs(logdir)
        hits = [p for (stamp, label), p in index.items()
                if key in (stamp, label, f"{stamp}_{label}") or stamp.endswith("_" + key)]
        if not hits:
            raise FileNotFoundError(f"no _run.csv for {key!r} under {logdir}")
        if len(hits) > 1:
            print(f"note: {key!r} matches {len(hits)} runs; using all of them", file=sys.stderr)
        found.extend(hits)
    return found


def controller_code(meta, label):
    """L / P / M from the meta header (bundle name or spec), checked against the label's last token."""
    name = score_run._controller_name(meta)
    code = CONTROLLER_CODES.get(name)
    token = label.split("_")[-1] if label else ""
    token = token if token in ("L", "P", "M") else None
    if code and token and code != token:
        raise ValueError(f"label {label!r} says {token} but the meta header says {name} ({code})")
    return code or token or name


def run_id(label):
    """`e3_A3p_pz0.0243` -> `A3+`; `e3_B1_M` -> `B1`. Labels not in the e3_<block><run>_... form give ''."""
    tok = label.split("_")
    if len(tok) < 2 or tok[0] != "e3":
        return ""
    rid = tok[1]
    return re.sub(r"^(A3)(p|plus|\+)$", r"\1+", re.sub(r"^(A3)(m|n|minus|-)$", r"\1-", rid))


# ---------------------------------------------------------------------------------------------- metrics ---
def mean_theta(theta):
    return float(np.mean(theta))


def mean_velocity(t, x):
    """v-bar = net x change / duration."""
    return float((x[-1] - x[0]) / (t[-1] - t[0]))


def _windows(t, width):
    """Start times of consecutive complete windows [k, k + width) from t[0] (1 ms tolerance at the end)."""
    return np.arange(t[0], t[-1] - width + 1e-3, width)


def theta_pp_windows(t, theta, width=PP_WINDOW_S):
    return [float(np.ptp(theta[(t >= k) & (t < k + width)])) for k in _windows(t, width)]


def band_windows(t, theta, width=BAND_WINDOW_S, band=BAND_HZ, f_min=TOTAL_MIN_HZ):
    """Per window: 3.3-4.3 Hz power fraction of theta (mean removed, Hann, total above 0.25 Hz) and peak frequency.

    The sample rate is the nominal loop rate (median tick rounded to 0.1 ms, i.e. 200 Hz), as in the reference
    prereg_scen.py. With a 4 s window the bins fall on multiples of 0.25 Hz, so "above 0.25 Hz" drops the 0.25 Hz
    bin only if the bins are exact; the measured 4.998 ms mean tick would move that bin to 0.2501 Hz and include it.
    """
    fs = 1.0 / round(float(np.median(np.diff(t))), 4)
    out = []
    for k in _windows(t, width):
        y = theta[(t >= k) & (t < k + width)]
        y = (y - y.mean()) * np.hanning(len(y))
        f = np.fft.rfftfreq(len(y), 1.0 / fs)
        p = np.abs(np.fft.rfft(y)) ** 2
        tot = f > f_min
        inband = (f >= band[0]) & (f <= band[1])
        out.append({"t0": float(k - t[0]), "t1": float(k - t[0] + width),
                    "fraction": float(p[inband].sum() / p[tot].sum()), "peak_hz": float(f[tot][np.argmax(p[tot])])})
    return out


def plateau_spans(t, goal, min_s=PLATEAU_MIN_S):
    """(i0, i1) index spans of each maximal run of constant goal lasting longer than min_s."""
    edges = np.r_[0, np.flatnonzero(np.diff(goal) != 0) + 1, len(goal)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if t[b - 1] - t[a] > min_s]


def plateau_errors(t, v, goal):
    return [{"t0": float(t[a]), "t1": float(t[b - 1]), "goal": float(goal[a]), "mean_v": float(np.mean(v[a:b])),
             "error": float(np.mean(v[a:b]) - goal[a])} for a, b in plateau_spans(t, goal)]


def wheel_ratio(pos_left, pos_right):
    """Delta pos_right / delta pos_left between the first and last finite samples (forward-positive degrees)."""
    m = np.isfinite(pos_left) & np.isfinite(pos_right)
    dl, dr = pos_left[m][-1] - pos_left[m][0], pos_right[m][-1] - pos_right[m][0]
    return float(dr / dl) if dl != 0 else float("nan")


def ols(x, y):
    """OLS of y on x: slope, intercept, Pearson r, residual sd (n - 2 dof)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(icpt), "r": float(np.corrcoef(x, y)[0, 1]),
            "resid_sd": float(np.sqrt(np.sum(resid ** 2) / (len(x) - 2))) if len(x) > 2 else float("nan")}


def holm(pvalues, alpha=ALPHA):
    """Holm step-down: (adjusted p-values, reject flags) in the input order."""
    p = np.asarray(pvalues, np.float64)
    order, m = np.argsort(p), len(p)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist(), (adj <= alpha).tolist()


def run_metrics(path, cfg=None, sheet_row=None, with_cost=True):
    """Every per-run metric for one `_run.csv` (sheet_row: optional run-sheet dict for overrides)."""
    row = sheet_row or {}
    meta, a = read_run_csv(path)
    m = NAME_RE.match(Path(path).name)
    stamp, file_label = (m.group(1), m.group(2)) if m else (Path(path).stem, Path(path).stem)
    label = row.get("label") or file_label
    t, x, v, th, goal = a["t"], a["x"], a["x_dot"], a["theta"], a["velocity_goal"]
    t_max = float(row["t_max"]) if row.get("t_max") else None
    if t_max is not None:
        keep = t - t[0] <= t_max
        t, x, v, th, goal = t[keep], x[keep], v[keep], th[keep], goal[keep]
    out = {"stamp": stamp, "label": label, "path": str(path),
           "controller": row.get("controller") or controller_code(meta, label),
           "run": row.get("run") or run_id(label), "t_max": t_max, "excluded": row.get("excluded") or None,
           "pitch_zero_file": (meta.get("calibration") or {}).get("pitch_zero"),
           "added_delay_ms": float(meta.get("added_delay_ms") or 0.0),   # balance.py --added-delay-ms (Block D)
           "duration": float(t[-1] - t[0]), "net_dx": float(x[-1] - x[0]),
           "theta_mean": mean_theta(th), "v_mean": mean_velocity(t, x)}
    out["block"] = row.get("block") or (out["run"][:1] if out["run"] else "")
    pp = np.degrees(theta_pp_windows(t, th))
    out["theta_pp_deg"] = pp.tolist()
    out["theta_pp_median_deg"] = float(np.median(pp)) if len(pp) else float("nan")
    out["band"] = band_windows(t, th) if t[-1] - t[0] >= BAND_WINDOW_S - 1e-3 else []
    out["plateaus"] = plateau_errors(t, v, goal)
    raw = read_raw_armed(path)
    if raw is not None:
        if t_max is not None:
            raw = {k: val[raw["t"] <= t_max] for k, val in raw.items()}
        out["wheel_ratio"] = wheel_ratio(raw["pos_left"], raw["pos_right"])
        spans = plateau_spans(raw["t"], raw["goal"])
        pos = [(a, b) for a, b in spans if raw["goal"][a] > 0]
        if pos:
            a0, b0 = max(pos, key=lambda s: raw["goal"][s[0]])
            out["wheel_ratio_cruise"] = wheel_ratio(raw["pos_left"][a0:b0], raw["pos_right"][a0:b0])
    if with_cost:
        s = score_run.score_trajectory(path, cfg=cfg)
        out.update(cost=s["cost"], fell=s["fell"], reason=s["reason"])
    return out


# ------------------------------------------------------------------------------------------- hypotheses ---
def _h(hid, value, verdict, rule, **extra):
    return {"H": hid, "value": value, "verdict": verdict, "rule": rule, **extra}


def analyze_block_a(runs, temps=None):
    """runs: metric dicts (with 'run' = A1, A2, A3+, A3-, A4, A5). temps: {run: (before, after)} in degC."""
    by = {r["run"]: r for r in runs if not r.get("excluded")}
    out = []
    if "A1" in by:
        th = by["A1"]["theta_mean"]
        ok = H_A1_RANGE[0] <= th <= H_A1_RANGE[1]
        pz = by["A1"].get("pitch_zero_file")
        out.append(_h("H-A1", th, "pass" if ok else "fail", "theta-bar(A1) in [+0.007, +0.017] rad",
                      trim_pitch_zero_new=None if pz is None else pz + th))
    else:
        out.append(_h("H-A1", None, "not run", "theta-bar(A1) in [+0.007, +0.017] rad"))
    for rid in ("A2", "A5"):
        if rid not in by:
            out.append(_h(f"H-A2 ({rid})", None, "not run", "|v-bar| < 0.005 m/s and |theta-bar| < 0.004 rad"))
            continue
        v, th = by[rid]["v_mean"], by[rid]["theta_mean"]
        verdict = ("pass" if abs(v) < H_A2_V and abs(th) < H_A2_TH else
                   "fail" if abs(v) >= H_A2_V else "inconclusive (|v| ok, |theta| >= 0.004; doc gives no verdict)")
        out.append(_h(f"H-A2 ({rid})", {"v_mean": v, "theta_mean": th},
                      verdict if rid == "A2" else f"reported: {verdict} (counts only if H-A4 shows no shift)",
                      "|v-bar| < 0.005 m/s and |theta-bar| < 0.004 rad; fail if |v-bar| >= 0.005"))
    fit_ids = [k for k in ("A1", "A2", "A3+", "A3-", "A5") if k in by]
    if len(fit_ids) >= 3:
        fit = ols([by[k]["theta_mean"] for k in fit_ids], [by[k]["v_mean"] for k in fit_ids])
        signs = ("A3+" in by and "A3-" in by and by["A3+"]["v_mean"] > 0 > by["A3-"]["v_mean"])
        ok = H_A3_SLOPE[0] <= fit["slope"] <= H_A3_SLOPE[1] and signs
        out.append(_h("H-A3", fit, "pass" if ok else "fail",
                      "slope in [-1.10, -0.60] (m/s)/rad and v-bar(A3+) > 0 > v-bar(A3-)", runs=fit_ids,
                      a3_signs_ok=bool(signs)))
    else:
        out.append(_h("H-A3", None, "not run", "needs >= 3 of A1, A2, A3+, A3-, A5"))
    delta = None
    if "A2" in by and "A5" in by:
        delta = by["A5"]["theta_mean"] - by["A2"]["theta_mean"]
        verdict = ("supported" if abs(delta) > H_A4_SUPPORT else "not supported" if abs(delta) < H_A4_NULL
                   else "inconclusive")
        out.append(_h("H-A4", delta, verdict, "|theta-bar(A5) - theta-bar(A2)| > 0.01 supported, < 0.005 not",
                      forward_sign=bool(delta > 0)))
    else:
        out.append(_h("H-A4", None, "not run", "needs A2 and A5"))
    vals = [x for pair in (temps or {}).values() for x in pair if x is not None]
    if len(vals) >= 2 and delta is not None:
        dT = max(vals) - min(vals)
        ok = H_A5_TEMPCO * dT < 0.5 * abs(delta)
        out.append(_h("H-A5", {"max_dT_soc": dT, "tempco_x_dT": H_A5_TEMPCO * dT, "half_abs_delta": 0.5 * abs(delta)},
                      "supported" if ok else "not supported",
                      "0.0008 rad/degC x largest dT_SoC < |delta theta-bar (H-A4)| / 2 (SoC temp is a proxy)",
                      theta_vs_temp=[(k, (temps or {}).get(k, (None,))[0], by[k]["theta_mean"]) for k in by]))
    else:
        out.append(_h("H-A5", None, "not run", "needs temp_before/temp_after in the sheet and H-A4's delta"))
    return out


def _costs(runs, code):
    return [r["cost"] for r in runs if r["controller"] == code and not r.get("excluded")]


def analyze_block_b(runs, full_n=B_FULL_N):
    from scipy.stats import mannwhitneyu
    groups = {c: _costs(runs, c) for c in ("L", "P", "M")}
    desc = {c: {"n": len(v), "median": float(np.median(v)) if v else None,
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "costs": v} for c, v in groups.items()}
    out = [_h("costs", desc, "descriptive", "score_run cost per controller; sd (ddof 1) sets the next N")]
    final = all(len(v) >= full_n for v in groups.values())
    if final:
        p = [mannwhitneyu(groups["L"], groups[o], alternative="less", method="exact").pvalue for o in ("P", "M")]
        adj, rej = holm(p)
        for (hid, o), pi, ai, ri in zip((("H-B1", "P"), ("H-B2", "M")), p, adj, rej):
            ok = ri and desc["L"]["median"] < desc[o]["median"]
            out.append(_h(hid, {"p": float(pi), "p_holm": ai, "median_L": desc["L"]["median"],
                                f"median_{o}": desc[o]["median"]}, "pass" if ok else "fail",
                          f"exact one-sided Mann-Whitney L < {o}, Holm alpha 0.05, and median(L) < median({o})"))
    else:
        for hid in ("H-B1", "H-B2"):
            out.append(_h(hid, None, f"interim (n = {[len(v) for v in groups.values()]} < {full_n}): no test",
                          "tested only after all 18 Block B runs (section 0)"))
    if desc["P"]["median"] and desc["M"]["median"]:
        ratio = desc["P"]["median"] / desc["M"]["median"]
        inband = H_B3_BAND[0] <= ratio <= H_B3_BAND[1]
        out.append(_h("H-B3", ratio, f"descriptive ({'in' if inband else 'outside'} [0.85, 1.15])",
                      "median(P) / median(M)"))
    pp = {c: [w for r in runs if r["controller"] == c and not r.get("excluded") for w in r["theta_pp_deg"]]
          for c in ("L", "P", "M")}
    med = {c: float(np.median(v)) if v else None for c, v in pp.items()}
    lower = None if None in med.values() else bool(med["L"] < med["P"] and med["L"] < med["M"])
    out.append(_h("H-B4", med, f"descriptive (L below both PIDs: {lower})",
                  "median theta p-p (deg) over all 2 s windows pooled per controller",
                  range_deg={c: [min(v), max(v)] if v else None for c, v in pp.items()}))
    return out


def load_svr_ratio(stamp, svr_dir):
    p = Path(svr_dir) / f"{stamp}_metrics_T1.json"
    if not p.is_file():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["wobble_ratio_real_over_free"]["theta"])


def analyze_block_c(runs, svr_dir=DEFAULT_SVR_DIR):
    use = [r for r in runs if not r.get("excluded")]
    L, M = [r for r in use if r["controller"] == "L"], [r for r in use if r["controller"] == "M"]
    out = []
    if not L:
        return [_h(h, None, "not run", "no L runs") for h in ("H-C1", "H-C2", "H-C3", "H-C4")]
    errs = {r["stamp"]: [round(p["error"], 4) for p in r["plateaus"]] for r in L}
    ok = all(abs(e) <= H_C1_TOL for v in errs.values() for e in v)
    out.append(_h("H-C1", errs, "pass" if ok else "fail", "|mean x_dot - goal| <= 0.05 m/s on every plateau > 0.5 s"))
    ratios = {r["stamp"]: {"run": r.get("wheel_ratio"), "cruise": r.get("wheel_ratio_cruise")} for r in L}
    ok = all(v["run"] is not None and v["run"] < 1 for v in ratios.values())
    out.append(_h("H-C2", ratios, f"{'pass' if ok else 'fail'} on the metric (also needs Evan's visual right curve)",
                  "Delta pos_right / Delta pos_left < 1 over the armed run, every L run"))
    band = {}
    ok = True
    for r in L:
        cruise_peaks = [w["peak_hz"] for w in r["band"] if (w["t0"], w["t1"]) in CRUISE_WINDOWS]
        cruise_hit = any(BAND_HZ[0] <= f <= BAND_HZ[1] for f in cruise_peaks)
        fr = [w["fraction"] for w in r["band"]]
        ok &= bool(fr) and max(fr) <= H_C3_MAX and not cruise_hit and len(cruise_peaks) == len(CRUISE_WINDOWS)
        band[r["stamp"]] = {"fractions": [round(f, 4) for f in fr], "peaks_hz": [w["peak_hz"] for w in r["band"]],
                            "cruise_peak_in_band": cruise_hit}
    out.append(_h("H-C3", band, "pass" if ok else "fail",
                  "band fraction <= 0.20 in every 4 s window and no 4-8 / 8-12 s window peaking in 3.3-4.3 Hz"))
    svr = {r["stamp"]: load_svr_ratio(r["stamp"], svr_dir) for r in L}
    have = [v for v in svr.values() if v is not None]
    if have:
        med = float(np.median(have))
        verdict = ("pass" if H_C4_PASS[0] <= med <= H_C4_PASS[1] else
                   "weak support" if H_C4_WEAK[0] <= med <= H_C4_WEAK[1] else "fail")
        if len(have) < len(L):
            verdict += f" (only {len(have)}/{len(L)} L runs have sim_vs_real metrics)"
        out.append(_h("H-C4", {"median": med, "per_run": svr}, verdict, "median wobble ratio real/free in [1.3, 1.7]"))
    else:
        out.append(_h("H-C4", None, "not run (run sim_vs_real.py <stamp> --no-anim first)", "median in [1.3, 1.7]"))
    if M:
        fell = [r["stamp"] for r in M if r.get("fell")]
        out.append(_h("H-C5", {"fell": fell, "median_cost_M": float(np.median([r["cost"] for r in M])),
                               "median_cost_L": float(np.median([r["cost"] for r in L]))},
                      "fail" if fell else "pass", "M completes with no fall; cost comparison descriptive (n = 2)"))
    else:
        out.append(_h("H-C5", None, "not run (no M runs)", "M completes with no fall"))
    return out


# ---------------------------------------------------------------------------------------------- output ---
def _fmt(x):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (
        f"{x:+.4f}" if isinstance(x, float) else str(x))


def format_table(rows):
    lines = [f"{'stamp':16s} {'label':28s} {'ctl':4s} {'dur':>6s} {'theta_bar':>9s} {'v_bar':>8s} {'cost':>7s} "
             f"{'pp_med':>6s} {'wheelR':>7s}  plateau errors / band fractions"]
    for r in rows:
        extra = " ".join(f"{p['goal']:+.2f}:{p['error']:+.3f}" for p in r["plateaus"])
        if r["band"]:
            extra += "  band " + " ".join(f"{w['fraction']:.3f}@{w['peak_hz']:.2f}" for w in r["band"])
        flag = f"  EXCLUDED: {r['excluded']}" if r.get("excluded") else ""
        lines.append(f"{r['stamp']:16s} {r['label'][:28]:28s} {str(r['controller'])[:4]:4s} {r['duration']:6.2f} "
                     f"{r['theta_mean']:+9.4f} {r['v_mean']:+8.4f} {r.get('cost', float('nan')):7.3f} "
                     f"{r['theta_pp_median_deg']:6.1f} {_fmt(r.get('wheel_ratio')):>7s}  {extra}{flag}")
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def read_sheet(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k.strip().lower(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="stamps, labels or _run.csv paths")
    ap.add_argument("--logdir", default=str(DEFAULT_LOGDIR), help="searched recursively for <stamp>_<label>_run.csv")
    ap.add_argument("--sheet", default=None, help="run sheet CSV (stamp, label, controller, run, block, t_max, ...)")
    ap.add_argument("--block", choices=("A", "B", "C"), default=None, help="apply that block's hypothesis tests")
    ap.add_argument("--svr-dir", default=str(DEFAULT_SVR_DIR), help="sim_vs_real --out folder (H-C4)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)
    sheet = read_sheet(args.sheet) if args.sheet else []
    keys = args.runs or [r["stamp"] for r in sheet
                         if not args.block or (r.get("block") or run_id(r.get("label", ""))[:1]) == args.block]
    if not keys:
        ap.error("give runs or a --sheet")
    by_stamp = {r["stamp"]: r for r in sheet}
    cfg = score_run.control.load_robot_config()
    rows = []
    for p in find_runs(keys, args.logdir):
        m = NAME_RE.match(p.name)
        stamp = m.group(1) if m else ""
        srow = by_stamp.get(stamp) or next((v for k, v in by_stamp.items() if stamp.endswith("_" + k)), None)
        rows.append(run_metrics(p, cfg=cfg, sheet_row=srow))
    if args.block:
        rows_b = [r for r in rows if r["block"] == args.block] or rows
    print(format_table(rows))
    result = {"runs": rows}
    if len(rows) >= 3:
        result["fit_all"] = ols([r["theta_mean"] for r in rows if not r.get("excluded")],
                                [r["v_mean"] for r in rows if not r.get("excluded")])
        f = result["fit_all"]
        print(f"\nOLS v-bar on theta-bar over these runs (descriptive): n {f['n']} slope {f['slope']:+.3f} (m/s)/rad, "
              f"intercept {f['intercept']:+.4f} m/s, r {f['r']:+.3f}, residual sd {f['resid_sd']:.4f} m/s")
    if args.block:
        if args.block == "A":
            temps = {}
            for r in rows_b:
                s = by_stamp.get(r["stamp"], {})
                tb, ta = (float(s[k]) if s.get(k) else None for k in ("temp_before", "temp_after"))
                temps[r["run"]] = (tb, ta)
            hyps = analyze_block_a(rows_b, temps)
        elif args.block == "B":
            hyps = analyze_block_b(rows_b)
        else:
            hyps = analyze_block_c(rows_b, args.svr_dir)
        result["hypotheses"] = hyps
        print(f"\nBlock {args.block} (prereg section 2):")
        for h in hyps:
            print(f"  {h['H']:12s} {h['verdict']}\n      rule: {h['rule']}\n      value: {json.dumps(_jsonable(h['value']))}")
    excl = [r for r in rows if r.get("excluded")]
    if excl:
        print("\nExcluded (section 4.4): " + "; ".join(f"{r['stamp']} {r['label']}: {r['excluded']}" for r in excl))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
