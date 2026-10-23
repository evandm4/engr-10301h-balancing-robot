"""Pivot robot demo (side project; see PIVOT.md): the robot with a servo-driven
pivot and a second body on top, following the tuner's goal-velocity profile.

Compares, on the same scenario (DC motors, sensor noise, latency from config.py):
    held     config.py's hand-tuned wheel gains, pivot held straight
    level    the same wheel gains, pivot keeping the upper body vertical
    LQR      the model-based optimal controller (wheels and pivot together)
    tuned    the best gains from the latest optimization/run_pivot_tuning.py
             run, if there is one (or any 7 gains given with --gains)

From the project folder:
    python simulation/examples/run_pivot_demo.py
    python simulation/examples/run_pivot_demo.py --animate             side view, live
    python simulation/examples/run_pivot_demo.py --animate --save pivot.gif
    python simulation/examples/run_pivot_demo.py --gains 4 0.5 0.15 0.1 0 -1 0
    python simulation/examples/run_pivot_demo.py --ideal                no noise, delay or motor limits
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pivot_config import PIVOT_CONFIG as PC  # noqa: E402


def latest_tuned_gains():
    found = sorted((ROOT / "results" / "pivot_tuning").glob("*/comparison.json"))
    if not found:
        return None
    results = json.loads(found[-1].read_text())["results"]
    best = results.get("DE (pivot active)")
    return None if best is None else (list(best["gains"].values()), found[-1].parent.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gains", type=float, nargs=7, metavar="G",
                        help="kp ki kd kv_p k_level k_phi_p k_phi_d for a 'custom' run")
    parser.add_argument("--tilt", type=float, default=PC.base.scenario.start_pitches_deg[0], help="start tilt (deg)")
    parser.add_argument("--seed", type=int, default=PC.base.scenario.validation_seeds[0], help="sensor noise seed")
    parser.add_argument("--ideal", action="store_true", help="ideal motors, perfect sensing, no latency")
    parser.add_argument("--animate", action="store_true", help="show an animated side view")
    parser.add_argument("--speed", type=float, default=1.0, help="animation playback speed")
    parser.add_argument("--save", help="save the animation (.gif, or .mp4 with ffmpeg) instead of showing it")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    controllers = {
        "held": lambda: PC.build_controller(PC.hand_tuned_gains()),
        "level": lambda: PC.build_controller(PC.level_gains()),
        "LQR": lambda: PC.build_lqr(ideal_actuator=args.ideal, latency=0.0 if args.ideal else None),
    }
    if args.gains:
        controllers["custom"] = lambda: PC.build_controller(args.gains)
    else:
        tuned = latest_tuned_gains()
        if tuned:
            print(f"'tuned' uses the best gains from results/pivot_tuning/{tuned[1]}")
            controllers["tuned"] = lambda: PC.build_controller(tuned[0])

    conditions = "ideal" if args.ideal else (f"{PC.base.actuator.model} motors, sensor noise (seed {args.seed}), "
                                             f"{PC.base.timing.latency_s * 1000:.0f} ms latency")
    print(f"Start tilt {args.tilt:.2f} deg, {conditions}, servo at {1 / PC.servo.update_period_s:.0f} Hz\n")
    print(f"{'controller':<10}{'fell':>6}{'cost':>9}{'vel RMS':>9}{'max COM':>9}{'max pivot':>11}"
          f"{'upper RMS':>11}{'servo sat':>11}")
    runs = {}
    for name, make in controllers.items():
        r = PC.run(make(), math.radians(args.tilt), seed=None if args.ideal else args.seed,
                   realistic=not args.ideal, dt=PC.base.timing.demo_dt_s)
        m = PC.metrics(r)
        runs[name] = r
        print(f"{name:<10}{str(r.fell):>6}{PC.score(r):>9.3f}{m.base.velocity_rms_error:>9.3f}"
              f"{math.degrees(m.base.max_abs_pitch):>8.1f}°{math.degrees(m.max_abs_pivot):>10.1f}°"
              f"{math.degrees(m.rms_upper_pitch):>10.1f}°{100 * m.servo_saturated:>10.1f}%")
    print("\ncost: config.py's cost on the combined COM, plus servo effort (pivot_config.py)."
          "\nvel RMS in m/s; max COM = largest combined-COM tilt; upper RMS = upper body tilt from vertical;"
          "\nservo sat = share of the run the servo was at its torque limit.")

    if args.no_plot:
        return
    try:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see plots)")
        return
    from pivot_sim.plotting import animate_runs, runs_figure

    if args.animate or args.save:
        _, anim = animate_runs(runs, PC.build_params(), speed=args.speed, save=args.save)
        if args.save:
            print(f"Saved animation to {args.save}")
            return
    runs_figure(runs, title="Pivot robot following the goal velocity")
    plt.show()


if __name__ == "__main__":
    main()
