"""Plots and a side-view animation for pivot-robot runs (needs matplotlib)."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .params import PivotRobotParams
from .simulator import PivotSimResult

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def runs_figure(runs: Dict[str, PivotSimResult], histories: Optional[Dict[str, Sequence[Tuple[int, float]]]] = None,
                baseline_cost: Optional[float] = None, title: str = ""):
    """2 x 3 panels: velocity, COM pitch, wheel torque, pivot angle, upper body
    pitch, and either the tuning convergence (if histories are given) or the
    servo torque."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
    panels = [
        (axes[0, 0], "velocity (m/s)", lambda r: r.states[:, 1]),
        (axes[0, 1], "COM pitch (deg)", lambda r: np.degrees(r.com_pitch)),
        (axes[0, 2], "wheel torque (N m)", lambda r: r.torques),
        (axes[1, 0], "pivot angle (deg)", lambda r: np.degrees(r.states[:, 4])),
        (axes[1, 1], "upper body pitch (deg)", lambda r: np.degrees(r.upper_pitch)),
    ]
    if not histories:
        panels.append((axes[1, 2], "servo torque (N m)", lambda r: r.servo_torques))
    for ax, label, signal in panels:
        for i, (name, r) in enumerate(runs.items()):
            ax.plot(r.t, signal(r), label=name, color=_color(i), lw=1)
        ax.set_ylabel(label)
        ax.set_xlabel("time (s)")
    first = next(iter(runs.values()))
    axes[0, 0].plot(first.t, first.velocity_goal, "k--", lw=1, label="goal")
    axes[0, 0].legend(fontsize=8)
    if histories:
        ax = axes[1, 2]
        for name, history in histories.items():
            ax.plot(*zip(*history), label=name)
        if baseline_cost is not None:
            ax.axhline(baseline_cost, color="gray", ls="--", label="hand-tuned (held)")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("best training cost so far")
        ax.set_yscale("log")
        ax.set_title("Convergence")
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def animate_runs(runs: Dict[str, PivotSimResult], params: PivotRobotParams, speed: float = 1.0,
                 fps: int = 30, save: Optional[str] = None):
    """Side view of each run, side by side: wheel, lower body to the pivot,
    upper body, and the combined COM (black dot). The view follows each robot;
    ground ticks every 5 cm show it moving. save: a .gif (or .mp4 with ffmpeg)
    path to write instead of showing."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    r_w, h = params.wheel_radius, params.pivot_height
    upper_len = 2 * params.upper_com_offset if params.upper_com_offset > 0 else 0.05
    half_width, top = 0.3, r_w + h + upper_len + 0.05
    duration = max(r.t[-1] for r in runs.values())
    frame_times = np.arange(0.0, duration, speed / fps)

    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 2.6), squeeze=False)
    artists = []
    for i, (ax, (name, run)) in enumerate(zip(axes[0], runs.items())):
        color = _color(i)
        ax.set_aspect("equal")
        ax.set_ylim(-0.03, top)
        ax.set_title(name)
        ax.set_yticks([])
        ground, = ax.plot([], [], "k|", ms=6)
        ax.axhline(0.0, color="k", lw=1)
        wheel = Circle((0, r_w), r_w, fill=False, lw=2, color="0.3")
        ax.add_patch(wheel)
        spoke, = ax.plot([], [], color="0.3", lw=1)
        lower, = ax.plot([], [], color=color, lw=6, solid_capstyle="round")
        upper, = ax.plot([], [], color=color, lw=4, alpha=0.6, solid_capstyle="round")
        com, = ax.plot([], [], "ko", ms=5)
        clock = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", fontsize=8)
        artists.append((ax, run, ground, wheel, spoke, lower, upper, com, clock))

    def draw(t):
        out = []
        for ax, run, ground, wheel, spoke, lower, upper, com, clock in artists:
            k = min(np.searchsorted(run.t, t), len(run.t) - 1)
            x, theta1, phi = run.states[k, 0], run.states[k, 2], run.states[k, 4]
            theta2 = theta1 + phi
            ax.set_xlim(x - half_width, x + half_width)
            marks = np.arange(np.floor((x - half_width) / 0.05) * 0.05, x + half_width, 0.05)
            ground.set_data(marks, np.zeros_like(marks))
            wheel.center = (x, r_w)
            roll = -x / r_w
            spoke.set_data([x, x + r_w * np.sin(roll)], [r_w, r_w + r_w * np.cos(roll)])
            pivot = (x + h * np.sin(theta1), r_w + h * np.cos(theta1))
            lower.set_data([x, pivot[0]], [r_w, pivot[1]])
            upper.set_data([pivot[0], pivot[0] + upper_len * np.sin(theta2)],
                           [pivot[1], pivot[1] + upper_len * np.cos(theta2)])
            com_len = (params.b1 ** 2 + params.b2 ** 2 + 2 * params.b1 * params.b2 * np.cos(phi)) ** 0.5 / params.total_mass
            com.set_data([x + com_len * np.sin(run.com_pitch[k])], [r_w + com_len * np.cos(run.com_pitch[k])])
            fell = " (fell)" if run.fell and t >= run.t[-1] else ""
            clock.set_text(f"t = {run.t[k]:.2f} s{fell}")
            out += [ground, wheel, spoke, lower, upper, com, clock]
        return out

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=frame_times, interval=1000 / fps, blit=False)
    if save:
        anim.save(save, fps=fps)
    return fig, anim
