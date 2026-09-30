"""Trajectory CSVs and the animation viewer (simulation/animate.py)."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import animate  # noqa: E402  (simulation/animate.py)
from balance_sim import load_trajectory, save_trajectory  # noqa: E402
from config import CONFIG  # noqa: E402
from tuning import Evaluator  # noqa: E402

SHORT = replace(CONFIG.build_eval_config(), duration=1.0)
HAND_TUNED = list(CONFIG.controller.hand_tuned_gains)
TOO_WEAK = [0.1, 0.0, 0.05, 0.0]   # Kp below the critical gain: falls


@pytest.fixture(scope="module")
def runs():
    ev = Evaluator(SHORT, CONFIG.build_search_space())
    return {"hand-tuned": ev.run(HAND_TUNED, 0.1, 1), "too weak": ev.run(TOO_WEAK, 0.1, 1)}


def test_trajectory_round_trip_is_exact(tmp_path, runs):
    for label, run in runs.items():
        path = save_trajectory(run, tmp_path / f"{label}.csv", label=label, gains=[1.0, 2.0])
        loaded, meta = load_trajectory(path)
        for field in ("t", "states", "torques", "velocity_goal", "pitch_reference"):
            np.testing.assert_array_equal(getattr(loaded, field), getattr(run, field))
        assert (loaded.fell, loaded.fall_time, loaded.duration) == (run.fell, run.fall_time, run.duration)
        assert meta["label"] == label and meta["gains"] == [1.0, 2.0]
    assert runs["too weak"].fell and not runs["hand-tuned"].fell


def test_trajectory_csv_is_a_plain_table(tmp_path, runs):
    path = save_trajectory(runs["hand-tuned"], tmp_path / "run.csv")
    lines = path.read_text().splitlines()
    assert lines[0].startswith("# ") and json.loads(lines[0][2:])["fell"] is False
    assert lines[1] == "t,x,x_dot,theta,theta_dot,torque,pitch_reference,velocity_goal"
    assert len(lines) == 2 + len(runs["hand-tuned"].t)


def test_viewer_draws_the_robot_where_the_simulation_says(runs):
    viewer = animate.RunAnimation(runs, fps=20)
    assert viewer.n_frames == math.ceil((viewer.end_time + 1.0) * 20) + 1
    frame = 10   # 0.5 s in
    viewer.draw(frame)
    run, parts = runs["hand-tuned"], viewer.robots[0]
    k = viewer._index(run, 0.5)
    theta = run.states[k, 2]
    corners = np.asarray(parts["body"].get_xy())[:4]
    top_middle = (corners[2] + corners[3]) / 2
    axle = np.array([0.0, viewer.r])
    assert top_middle == pytest.approx(axle + viewer.body_height * np.array([math.sin(theta), math.cos(theta)]))
    com_x, com_y = parts["com"].get_data()
    assert (com_x[0], com_y[0]) == pytest.approx(tuple(axle + viewer.l * np.array([math.sin(theta), math.cos(theta)])))


def test_viewer_marks_a_fall(runs):
    viewer = animate.RunAnimation(runs, fps=20)
    fell = viewer.robots[1]
    viewer.draw(0)
    assert "FELL" not in fell["text"].get_text()
    viewer.draw(viewer.n_frames - 1)   # past the fall: frozen at the fall, drawn red
    assert "FELL" in fell["text"].get_text()
    assert matplotlib.colors.to_hex(fell["body"].get_facecolor()) == matplotlib.colors.to_hex("red")


def test_timer_tick_while_the_window_closes_stops_quietly(runs):
    """Closing a window swaps the figure's canvas for a bare one right away, but
    close_event (which normally stops the animation) only arrives once Tk
    destroys the window a moment later. A timer tick in that gap used to crash
    with "'FigureCanvasBase' object has no attribute ...". Recreate the gap."""
    viewer = animate.RunAnimation(runs, fps=20)
    anim = viewer.animation()
    anim._step()                        # a normal tick while the window is open
    viewer.fig._set_base_canvas()       # what matplotlib does when a window starts closing
    assert anim._step() is False        # the next tick stops the animation instead of drawing
    anim._step()                        # and any later stray tick is harmless too


def test_viewer_saves_a_gif(tmp_path, runs):
    short = {label: run for label, run in runs.items() if label == "too weak"}   # falls quickly: few frames
    path = animate.RunAnimation(short, fps=10, speed=4.0).save(tmp_path / "robot.gif", dpi=40)
    assert path.exists() and path.stat().st_size > 0


def test_gains_from_results_picks_hand_tuned_and_best(tmp_path):
    (tmp_path / "comparison.json").write_text(json.dumps({"results": {
        "hand-tuned": {"gains": {"kp": 4.0, "ki": 0.5, "kd": 0.15, "kv_p": 0.1}, "train_objective": 0.3},
        "cma-es": {"gains": {"kp": 1.0, "ki": 0.2, "kd": 0.1, "kv_p": 0.33}, "train_objective": 0.107},
        "random search": {"gains": {"kp": 0.9, "ki": 0.7, "kd": 0.08, "kv_p": 0.37}, "train_objective": 0.115},
    }}))
    assert animate.gains_from_results(tmp_path) == {"hand-tuned": [4.0, 0.5, 0.15, 0.1],
                                                    "cma-es": [1.0, 0.2, 0.1, 0.33]}
