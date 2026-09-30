"""Compare an ideal run against a harder one, print metrics, and plot both.

All robot parameters, sensor noise, latency, and cost weights come from
config.py at the project root — edit that file to change the conditions, not
this script. Angles are shown in degrees; calculations stay in radians.

From the project folder:   python simulation/examples/run_demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balance_sim import compute_metrics, simulate
from config import CONFIG


def run(label, dyn, **overrides):
    conditions = dict(
        duration=CONFIG.scenario.duration_s,
        dt=CONFIG.timing.demo_dt_s,
        initial_state=dyn.initial_state(pitch=CONFIG.scenario.start_pitches_rad[0]),
        control_dt=CONFIG.timing.control_dt_s,
    )
    conditions.update(overrides)
    controller = CONFIG.build_pitch_controller()   # pitch-only demo: the hand-tuned kp, ki, kd
    result = simulate(dyn, controller, **conditions)
    m = compute_metrics(result)
    c = CONFIG.score(result)
    print(label)
    print(f"  fell: {m.fell}   settling: {m.settling_time}   max |pitch|: {np.degrees(m.max_abs_pitch):.2f} deg"
          f"   drift: {m.drift:.2f} m   cost: {c:.4f}")
    return result


def main():
    ideal = CONFIG.build_dynamics(ideal_actuator=True)   # ideal actuator regardless of config.actuator
    real = CONFIG.build_dynamics()                       # actuator per config.actuator.model

    runs = {
        "ideal": run("Ideal (no noise, no latency, ideal motors)", ideal),
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
    for name, r in runs.items():
        axes[0].plot(r.t, np.degrees(r.states[:, 2]), label=name)
        axes[1].plot(r.t, r.states[:, 0], label=name)
        axes[2].plot(r.t, r.torques, label=name)
    axes[0].set_ylabel("pitch (deg)")
    axes[1].set_ylabel("position (m)")
    axes[2].set_ylabel("torque (N m)")
    axes[2].set_xlabel("time (s)")
    axes[0].legend()
    fig.suptitle("Same PID gains, easy vs harder conditions")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
