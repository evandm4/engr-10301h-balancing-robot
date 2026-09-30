"""Watch the robot: an animated side view of one or more runs, with live plots.

Each run is a trajectory, either simulated here with the reference simulator
(same scenario, robot, noise and latency as the tuner, all from config.py) or
replayed from a CSV saved earlier (balance_sim.save_trajectory; the tuning
scripts save their example runs to <results folder>/trajectories/). Physics
runs before playback, so what you see is exactly what the tuner scored.

From the project folder:
    python simulation/animate.py                          hand-tuned vs the latest tuning result
    python simulation/animate.py --gains 1.03 0.20 0.10 0.33 --label tuned
    python simulation/animate.py --gains 4 0.5 0.15 0.1 --gains 1.03 0.2 0.1 0.33 --tilt -4.58 --seed 105
    python simulation/animate.py --results results/batch_tuning/20260928-145838
    python simulation/animate.py --csv results/batch_tuning/<run>/trajectories/cma-es.csv
    python simulation/animate.py --speed 0.25             slow motion
    python simulation/animate.py --save robot.gif         save instead of showing (.gif, or .mp4 with ffmpeg)
    python simulation/animate.py --export out/            also save the shown runs as CSV

What's drawn: the wheel (its spoke shows it rolling), the body leaning at its
pitch, its center of mass, and a dashed line for the lean the controller is
asking for. The view follows the robot; the ground marks (every 5 cm) show it
moving. Below: pitch, velocity and torque over the whole run with a cursor.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # config.py; importing it puts simulation/ and optimization/ on the path

from config import CONFIG  # noqa: E402
from balance_sim import SimResult, load_trajectory, save_trajectory  # noqa: E402

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
GROUND_MARK_SPACING = 0.05   # m
VIEW_HALF_WIDTH = 0.25       # m, robot view extends this far either side of the robot


# ---- where runs come from ----------------------------------------------------

def simulate_runs(gains_by_label: Dict[str, List[float]], tilt_rad: float, seed: int) -> Dict[str, SimResult]:
    """Run the reference simulator on the tuner's scenario for each set of gains."""
    from tuning import Evaluator

    evaluator = Evaluator(CONFIG.build_eval_config(), CONFIG.build_search_space())
    return {label: evaluator.run(np.asarray(g, dtype=float), tilt_rad, seed) for label, g in gains_by_label.items()}


def latest_results_folder() -> Optional[Path]:
    """The most recent folder under results/tuning or results/batch_tuning that has a comparison.json."""
    found = [p.parent for sub in ("tuning", "batch_tuning") for p in (ROOT / "results" / sub).glob("*/comparison.json")]
    return max(found, key=lambda p: p.name) if found else None


def gains_from_results(folder: Path) -> Dict[str, List[float]]:
    """Hand-tuned and best gains from a results folder's comparison.json."""
    results = json.loads((folder / "comparison.json").read_text())["results"]
    score = lambda c: c.get("train_objective", c.get("train_cost"))
    best = min((name for name in results if name != "hand-tuned"), key=lambda n: score(results[n]))
    return {"hand-tuned": list(results["hand-tuned"]["gains"].values()),
            best: list(results[best]["gains"].values())}


def runs_from_csv(paths: List[Path]) -> Dict[str, SimResult]:
    runs = {}
    for path in paths:
        result, meta = load_trajectory(path)
        runs[meta.get("label", Path(path).stem)] = result
    return runs


# ---- the animation -------------------------------------------------------------

class RunAnimation:
    """Side view of each run, plus pitch / velocity / torque plots with a moving cursor."""

    def __init__(self, runs: Dict[str, SimResult], robot=None, fps: int = 30, speed: float = 1.0,
                 title: str = ""):
        import matplotlib.pyplot as plt

        if not runs:
            raise ValueError("nothing to animate")
        self.runs = runs
        self.fps, self.speed = fps, speed
        robot = robot if robot is not None else CONFIG.build_robot_params()
        self.r = robot.wheel_radius
        self.l = robot.body_com_height
        self.body_height = 2 * self.l            # draw the body as a block with its COM at mid-height
        self.body_width = 0.35 * self.body_height
        self.end_time = max(float(r.t[-1]) for r in runs.values())
        # Play the run, then hold the last frame for a second.
        self.n_frames = int(math.ceil((self.end_time / speed + 1.0) * fps)) + 1

        n = len(runs)
        self.fig = plt.figure(figsize=(max(10, 4 * n), 8), layout="constrained")
        grid = self.fig.add_gridspec(2, 1, height_ratios=[1.25, 1])
        top, bottom = grid[0].subgridspec(1, n), grid[1].subgridspec(1, 3)
        if title:
            self.fig.suptitle(title)

        self.artists = []
        self.robots = []
        for i, (label, run) in enumerate(runs.items()):
            ax = self.fig.add_subplot(top[0, i])
            self.robots.append(self._robot_view(ax, label, COLORS[i % len(COLORS)]))

        self.cursors = []
        plots = [self.fig.add_subplot(bottom[0, j]) for j in range(3)]
        for i, (label, run) in enumerate(runs.items()):
            color = COLORS[i % len(COLORS)]
            plots[0].plot(run.t, np.degrees(run.states[:, 2]), color=color, label=label)
            if run.pitch_reference is not None:
                plots[0].plot(run.t, np.degrees(run.pitch_reference), color=color, ls="--", lw=0.8)
            plots[1].plot(run.t, run.states[:, 1], color=color, label=label)
            plots[2].plot(run.t, run.torques, color=color, lw=0.8, label=label)
        first = next(iter(runs.values()))
        plots[0].plot([], [], color="gray", ls="--", lw=0.8, label="commanded")
        plots[1].plot(first.t, first.velocity_goal, "k--", lw=1, label="goal")
        for ax, title in zip(plots, ("pitch (deg)", "velocity (m/s)", "torque (N m)")):
            ax.set_xlabel("time (s)")
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, self.end_time)
            self.cursors.append(ax.axvline(0, color="gray", lw=1))
        plots[0].legend(fontsize=8, loc="lower left")
        plots[1].legend(fontsize=8, loc="lower left")
        self.artists += self.cursors

    def _robot_view(self, ax, label, color):
        from matplotlib.patches import Circle, Polygon

        ax.set_aspect("equal")
        ax.set_xlim(-VIEW_HALF_WIDTH, VIEW_HALF_WIDTH)
        ax.set_ylim(-0.03, self.r + self.body_height + 0.06)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, color=color)
        ax.axhline(0, color="k", lw=1)
        parts = {
            "ground": ax.plot([], [], "k|", ms=8)[0],
            "ghost": ax.plot([], [], ls="--", color="gray", lw=1.5)[0],
            "body": ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, color=color, alpha=0.85)),
            "wheel": ax.add_patch(Circle((0, self.r), self.r, facecolor="0.3", edgecolor="k")),
            "spoke": ax.plot([], [], color="w", lw=2)[0],
            "com": ax.plot([], [], "o", color="k", ms=5)[0],
            "text": ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace"),
            "color": color,
        }
        self.artists += [parts[k] for k in ("ground", "ghost", "body", "wheel", "spoke", "com", "text")]
        return parts

    def _index(self, run: SimResult, time: float) -> int:
        return min(int(np.searchsorted(run.t, time, side="right")) - 1, len(run.t) - 1)

    def draw(self, frame: int):
        """Update every artist for one frame (also usable without the animation, e.g. in tests)."""
        time = min(frame * self.speed / self.fps, self.end_time)
        for run, parts in zip(self.runs.values(), self.robots):
            k = max(self._index(run, time), 0)
            x, x_dot, theta = run.states[k, 0], run.states[k, 1], run.states[k, 2]
            axle = np.array([0.0, self.r])
            up = np.array([math.sin(theta), math.cos(theta)])        # along the body, leaning forward = +x
            side = np.array([math.cos(theta), -math.sin(theta)]) * self.body_width / 2
            top = axle + up * self.body_height
            parts["body"].set_xy([axle - side, axle + side, top + side, top - side])
            com = axle + up * self.l
            parts["com"].set_data([com[0]], [com[1]])
            spin = x / self.r                                        # rolling without slip
            parts["spoke"].set_data([0, self.r * math.sin(spin)], [self.r, self.r + self.r * math.cos(spin)])
            ref = run.pitch_reference[k] if run.pitch_reference is not None else 0.0
            ghost = axle + np.array([math.sin(ref), math.cos(ref)]) * self.body_height
            parts["ghost"].set_data([axle[0], ghost[0]], [axle[1], ghost[1]])
            first = math.floor((x - VIEW_HALF_WIDTH) / GROUND_MARK_SPACING)
            marks = np.arange(first, first + 2 * VIEW_HALF_WIDTH / GROUND_MARK_SPACING + 2) * GROUND_MARK_SPACING - x
            parts["ground"].set_data(marks, np.zeros_like(marks))

            fallen = run.fell and run.fall_time is not None and time >= run.fall_time
            parts["body"].set_color("red" if fallen else parts["color"])
            status = f"FELL at {run.fall_time:.2f} s" if fallen else ""
            parts["text"].set_text(
                f"t {time:5.2f} s\nv {x_dot:+.2f} m/s (goal {run.velocity_goal[k]:+.2f})\n"
                f"pitch {math.degrees(theta):+5.1f} deg\nx {x:+.2f} m\n{status}")
        for cursor in self.cursors:
            cursor.set_xdata([time, time])
        return self.artists

    def animation(self):
        from matplotlib.animation import FuncAnimation

        class ClosingSafeAnimation(FuncAnimation):
            """FuncAnimation that stops cleanly when its window closes.

            On close, matplotlib swaps the figure's canvas for a bare
            FigureCanvasBase straight away, but close_event (which normally
            stops the animation) only arrives when Tk destroys the window a
            moment later. A timer tick in between would try to draw on the
            bare canvas and raise AttributeError ('restore_region'). So each
            tick first checks the figure still has the canvas it started on.
            """

            def __init__(self, fig, *args, **kwargs):
                self._live_canvas = fig.canvas
                super().__init__(fig, *args, **kwargs)

            def _step(self, *args):
                if self._fig.canvas is not self._live_canvas:
                    if self.event_source is not None:
                        self.event_source.stop()
                    return False   # tells the timer to drop this callback
                return super()._step(*args)

        return ClosingSafeAnimation(self.fig, self.draw, frames=self.n_frames, interval=1000 / self.fps, blit=True)

    def save(self, path, dpi: int = 72):
        """Save as .gif (Pillow, always available) or .mp4 (needs ffmpeg on the PATH).

        The default dpi keeps a 6 s GIF to a few MB; raise it for sharper video.
        """
        from matplotlib import animation as mpl_animation

        path = Path(path)
        if path.suffix.lower() == ".gif":
            writer = mpl_animation.PillowWriter(fps=self.fps)
        elif path.suffix.lower() == ".mp4":
            if not mpl_animation.writers.is_available("ffmpeg"):
                raise RuntimeError("saving .mp4 needs ffmpeg on the PATH; save a .gif instead")
            writer = mpl_animation.FFMpegWriter(fps=self.fps)
        else:
            raise ValueError("save as .gif or .mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.animation().save(path, writer=writer, dpi=dpi,
                              progress_callback=lambda i, n: print(f"\r  frame {i + 1}/{n}", end=""))
        print()
        return path


# ---- command line ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gains", type=float, nargs="+", action="append",
                        help="kp ki kd kv_p [kv_i]; repeat to compare several")
    parser.add_argument("--label", action="append", help="a name for each --gains, in order")
    parser.add_argument("--results", help="a results folder (or 'latest'): replay its saved trajectories, "
                                          "or re-simulate its hand-tuned and best gains")
    parser.add_argument("--csv", nargs="+", help="trajectory CSV files to replay")
    parser.add_argument("--tilt", type=float, default=CONFIG.scenario.start_pitches_deg[0],
                        help="starting tilt in degrees for simulated runs")
    parser.add_argument("--seed", type=int, default=CONFIG.scenario.validation_seeds[0],
                        help="sensor-noise seed for simulated runs")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (0.25 = slow motion)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save", help="save to this .gif / .mp4 instead of showing a window")
    parser.add_argument("--dpi", type=int, default=72, help="resolution when saving")
    parser.add_argument("--export", help="also save the runs shown as trajectory CSVs in this folder")
    args = parser.parse_args()

    title = ""
    if args.csv:
        runs = runs_from_csv([Path(p) for p in args.csv])
        title = "replayed from CSV"
    elif args.gains:
        labels = args.label or []
        gains = {(labels[i] if i < len(labels) else f"gains {i + 1}"): g for i, g in enumerate(args.gains)}
        runs = simulate_runs(gains, math.radians(args.tilt), args.seed)
        title = f"simulated: tilt {args.tilt:g} deg, noise seed {args.seed}"
    else:
        folder = None
        if args.results and args.results != "latest":
            folder = Path(args.results)
        elif args.results == "latest" or not args.results:
            folder = latest_results_folder()
        if folder is None:
            print("No tuning results yet; showing the hand-tuned gains.")
            runs = simulate_runs({"hand-tuned": list(CONFIG.controller.hand_tuned_gains)},
                                 math.radians(args.tilt), args.seed)
            title = f"simulated: tilt {args.tilt:g} deg, noise seed {args.seed}"
        elif (folder / "trajectories").is_dir() and any((folder / "trajectories").glob("*.csv")):
            runs = runs_from_csv(sorted((folder / "trajectories").glob("*.csv")))
            title = f"replayed from {folder.name}"
        else:
            runs = simulate_runs(gains_from_results(folder), math.radians(args.tilt), args.seed)
            title = f"gains from {folder.name}, re-simulated: tilt {args.tilt:g} deg, noise seed {args.seed}"
    print(f"{title}: {', '.join(runs)}")

    if args.export:
        for label, run in runs.items():
            path = save_trajectory(run, Path(args.export) / f"{label.replace(' ', '_')}.csv", label=label)
            print(f"  saved {path}")

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viewer = RunAnimation(runs, fps=args.fps, speed=args.speed, title=title)
    if args.save:
        print(f"Saving {args.save} ...")
        print(f"Saved {viewer.save(args.save, dpi=args.dpi)}")
    else:
        anim = viewer.animation()   # keep a reference, or matplotlib stops the animation
        plt.show()
        del anim


if __name__ == "__main__":
    main()
