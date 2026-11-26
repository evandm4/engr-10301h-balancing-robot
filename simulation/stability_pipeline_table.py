"""Delay-margin table for the LEGO robot: L (a learned linear policy) vs the model-based cascaded PID, under three
sensing models (balance_sim.stability_pipeline; notes in simulation/balance_sim/STABILITY_PIPELINE.md):

  ideal          stability.py's loop: the controller sees the true state (Pololu-like sensing)
  sim pipeline   the loop the LEGO policies are trained and evaluated in: complementary filter, delayed and
                 differenced encoder, back-EMF feedforward, Build HAT PDM (the sim lumps the IMU DLPF into latency)
  hardware-like  sim pipeline + the ~3 ms DLPF on the IMU only + the right wheel acting 2.5 ms late

Margins are the TOTAL sample -> HAT delay (the sim's TimingConfig latency; for hardware-like: actuation latency +
the IMU-only delay) up to which the linear loop is stable, worst-best over the free PDM tick phase. The robot's
own base delay is ~9.5 +- 2.5 ms of that total. "unst." = unstable at every latency (MB at standstill with the
CF: a bounded ~0.5 Hz limit cycle in the nonlinear sim, not a fall risk); "80+" = stable to the scan limit.

From the project folder:
    python simulation/stability_pipeline_table.py
    python simulation/stability_pipeline_table.py --phase-step 10 --json margins.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # config.py; importing it puts simulation/ and optimization/ on the path

from config import load_config  # noqa: E402
from balance_sim import LinearController, load_policy  # noqa: E402
from balance_sim.stability_pipeline import format_table, margin_table  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="results/legobot/policies/l_e2_s1/final.npz",
                    help="learned linear policy (relative to the project folder)")
    ap.add_argument("--phase-step", type=int, default=4,
                    help="PDM phase step in 0.25 ms sub-steps (4 = 1 ms, the default; 10 = the sim's 2.5 ms grid)")
    ap.add_argument("--imu-delay", type=float, default=0.003, help="hardware-like IMU-only (DLPF) delay, s")
    ap.add_argument("--right-lag", type=float, default=0.0025, help="hardware-like right-wheel actuation lag, s")
    ap.add_argument("--right-enc-age", type=float, default=None,
                    help="hardware-like right encoder age, s (default: same as the left's, 5 ms)")
    ap.add_argument("--max-latency", type=float, default=0.08, help="scan limit, s")
    ap.add_argument("--json", default=None, help="also write the rows here")
    args = ap.parse_args(argv)

    cfg = load_config("legobot")
    T = cfg.timing.control_dt_s
    spec, params, _ = load_policy(ROOT / args.policy)
    controllers = {"L": LinearController.linear_policy(spec, params, T),
                   "MB": LinearController.cascaded_pid(tuple(float(g) for g in cfg.reference_pid_gains()), T)}
    t0 = time.time()
    rows = margin_table(cfg, controllers, cfg.build_robot_params(), pdm_phases=range(0, 40, args.phase_step),
                        max_latency=args.max_latency, imu_delay=args.imu_delay, right_lag=args.right_lag,
                        right_enc_age=args.right_enc_age)
    print(f"Delay margins, ms (total sample -> HAT; worst-best over PDM phase in {args.phase_step * 0.25:g} ms "
          f"steps; corner = COM height and rotor inertia +8 %; cruise = 0.40 m/s)\n")
    print(format_table(rows))
    print(f"\n({time.time() - t0:.0f} s)")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
