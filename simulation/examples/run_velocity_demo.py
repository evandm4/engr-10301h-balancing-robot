"""Cascaded velocity control: the robot follows the tuner's goal-velocity profile.

All gains, robot parameters, sensor noise, latency, and the goal profile come
from config.py at the project root, so this demo runs the exact scenario the
tuner scores against. Edit config.py to change the conditions, not this script.
Angles are shown in degrees; calculations stay in radians.

From the project folder:   python simulation/examples/run_velocity_demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balance_sim import compute_metrics, simulate
from config import CONFIG


def run(label, dyn, **conditions):
    controller = CONFIG.build_controller()   # hand-tuned gains, same setup the tuner scores
    result = simulate(
        dyn,
        controller,
        duration=CONFIG.scenario.duration_s,
        dt=CONFIG.timing.demo_dt_s,
        initial_state=dyn.initial_state(pitch=CONFIG.scenario.start_pitches_rad[0]),
        control_dt=CONFIG.timing.control_dt_s,
        **conditions,
    )
    m = compute_metrics(result)
    c = CONFIG.score(result)
    print(label)
    print(f"  fell: {m.fell}   velocity RMS error: {m.velocity_rms_error:.3f} m/s"
          f"   max |pitch|: {np.degrees(m.max_abs_pitch):.2f} deg   cost: {c:.4f}")
    return result


def main():
    ideal = CONFIG.build_dynamics(ideal_actuator=True)   # ideal actuator
    real = CONFIG.build_dynamics()                       # actuator per config

    runs = {
        "ideal": run("Ideal conditions", ideal),
        "harder": run(f"Harder ({CONFIG.actuator.model} motors, sensor noise, "
                      f"{CONFIG.timing.latency_s * 1000:.0f} ms latency)", real,
                      sensor=CONFIG.build_sensor(), latency=CONFIG.timing.latency_s),
    }

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to see plots)")
        return

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    first = next(iter(runs.values()))
    axes[0].plot(first.t, first.velocity_goal, "k--", label="goal")
    for name, r in runs.items():
        axes[0].plot(r.t, r.states[:, 1], label=name)
        axes[1].plot(r.t, np.degrees(r.states[:, 2]), label=name)
        axes[2].plot(r.t, r.states[:, 0], label=name)
    axes[0].set_ylabel("velocity (m/s)")
    axes[1].set_ylabel("pitch (deg)")
    axes[2].set_ylabel("position (m)")
    axes[2].set_xlabel("time (s)")
    axes[0].legend()
    fig.suptitle("Cascaded PID following a goal velocity")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
