"""Does the Build HAT's slow-PWM pulsing matter for balancing? Reference-engine experiment (B1b follow-up).

    python hardware/legobot/pwm_pulse_check.py            # period fit, then the baselines
    python hardware/legobot/pwm_pulse_check.py --quick    # fewer scenarios

The bench (DATASHEET 8.7, robot/measure/pwm_probe.py) shows that at duty 0.05-0.15 slow PWM makes a free
motor's speed ripple by 3-4 rad/s (sd) at 10-15 Hz, whatever the command timing: the firmware drives in
pulses. SlowPwmActuator averages them away. This script:

1. Models the pulses as full-voltage on-time |d| T at the start of every period T, free-running (a new
   duty takes effect within the current period), coasting otherwise: per motor, on-torque
   N eta k (copysign(V, d) - k N w) / R, the same as SlowPwmActuator at |d| = 1. Over a physics step the
   torque is that times the fraction of the step that is on.
2. Fits T to the bench ripple. It simulates the free motor (J = rotor + one wheel, the config's friction)
   at duties 0.05-0.2, samples the position at 100 Hz with 1-degree counts, and compares the ripple sd of
   10 ms speed differences with pwm_probe's.
3. Scores the baselines on the LEGO scenario, averaged and pulsed, at dt = 0.5 ms for both so pulses as
   short as 2.5 ms are resolved.
"""

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "optimization"))

from balance_sim import MotorParams, Planar2D, SlowPwmActuator, push_profile, simulate  # noqa: E402
from legobot_config import LEGOBOT_CONFIG as C  # noqa: E402
from tuning import Evaluator  # noqa: E402

# pwm_probe, robot secured (20261123_155652): ripple sd, rad/s, left wheel, mean of the three command styles
BENCH_RIPPLE = {0.05: 2.86, 0.10: 3.11, 0.15: 4.12, 0.20: 1.72}
BASELINE_RIPPLE = 1.2   # rad/s: the other PWM modes and full duty (friction around the turn, quantization)


class PulsedSlowPwmActuator(SlowPwmActuator):
    """SlowPwmActuator with the duty delivered as pulses of period `period` (s). `clock` is the start of
    the current physics step and `step` its length; the simulation loop sets clock every step."""

    def __init__(self, *args, period=0.1, step=0.0005, **kwargs):
        super().__init__(*args, **kwargs)
        self.period, self.step, self.clock = period, step, 0.0

    def on_fraction(self, duty):
        """Share of [clock, clock + step] inside the on-windows [n T, n T + |d| T)."""
        T, a, b = self.period, self.clock, self.clock + self.step
        width = min(abs(duty), 1.0) * T
        on = 0.0
        for n in range(int(math.floor(a / T)), int(math.floor(b / T)) + 1):
            lo, hi = n * T, n * T + width
            on += max(0.0, min(b, hi) - max(a, lo))
        return on / self.step

    def torque(self, command, relative_wheel_speed):
        if command == 0.0:
            return 0.0
        frac = self.on_fraction(command)
        return frac * super().torque(math.copysign(1.0, command), relative_wheel_speed) if frac else 0.0


class PdmSlowPwmActuator(SlowPwmActuator):
    """Pulse-density modulation, the pattern pulse_imu found (DATASHEET 8.7): time is cut into ticks of
    `period` (10 ms, the HAT's 100 Hz loop); at each tick a running sum adds the duty in effect, and the
    tick is a full-voltage pulse of the sum's sign when |sum| > 0.5 (the sum then moves 1 toward zero),
    coasting otherwise. Small duties become sparse pulses (one per 200 ms at 0.05) but the delivered
    impulse tracks the command to within one pulse. The sum persists across commands (pwm_probe)."""

    def __init__(self, *args, period=0.01, step=0.0005, **kwargs):
        super().__init__(*args, **kwargs)
        self.period, self.step, self.clock = period, step, 0.0
        self.tick, self.acc, self.out = -1, 0.0, 0.0

    def torque(self, command, relative_wheel_speed):
        tick = int(math.floor(self.clock / self.period + 1e-9))
        if tick != self.tick:                     # first call in a new tick: decide this tick's output
            self.tick = tick
            self.acc += max(-1.0, min(1.0, command))
            self.out = 1.0 if self.acc > 0.5 else -1.0 if self.acc < -0.5 else 0.0
            self.acc -= self.out
        return super().torque(self.out, relative_wheel_speed) if self.out else 0.0


MODEL = {"block": PulsedSlowPwmActuator, "pdm": PdmSlowPwmActuator}
PULSED = PulsedSlowPwmActuator


def free_motor_ripple(period, duty, seconds=3.0, dt=0.0001, seed=0):
    """Bare motor on the stand: ripple sd of the speed seen through 1-degree counts at 100 Hz."""
    a = C.actuator
    m = MotorParams(gear_ratio=1.0, torque_constant=a.torque_constant, resistance=a.resistance,
                    supply_voltage=7.8, current_limit=10.0, n_motors=1, gear_efficiency=a.gear_efficiency)
    J = a.rotor_inertia + C.robot.wheel_inertia / 2
    tc, b, we = C.robot.coulomb_friction / 2, C.robot.wheel_friction / 2, C.robot.coulomb_friction_speed
    act = PULSED(m, period=period, step=dt) if period else SlowPwmActuator(m)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, period) if period else 0.0
    w = th = 0.0
    samples = []
    n = int(seconds / dt)
    every = int(round(0.01 / dt))
    for i in range(n):
        if period:
            act.clock = i * dt + phase
        acc = (act.torque(duty, w) - tc * math.tanh(w / we) - b * w) / J
        w += acc * dt
        th += w * dt
        if i % every == 0:
            samples.append(math.floor(math.degrees(th)))
    pos = np.radians(np.array(samples[int(0.5 / 0.01):], float))
    sp = np.diff(pos) / 0.01
    return float(sp.mean()), float(sp.std())


def run(gains_or_controller, evaluator, start_pitch, seed, period, dt):
    """One scenario in the reference engine with averaged (period None) or pulsed slow PWM."""
    c = evaluator.config
    motor = c.motor
    if period:
        act = PULSED(motor, c.voltage_estimate, max_torque=c.params.max_torque, period=period, step=dt,
                                    feedforward=c.voltage_feedforward, feedforward_gain=c.voltage_feedforward_gain)
    else:
        act = SlowPwmActuator(motor, c.voltage_estimate, max_torque=c.params.max_torque,
                              feedforward=c.voltage_feedforward, feedforward_gain=c.voltage_feedforward_gain)
    dyn = Planar2D(c.params, fall_angle=c.fall_angle, actuator=act)
    push = push_profile(c.pushes) if c.pushes else (lambda t: 0.0)

    def disturbance(t):
        act.clock = t
        return push(t)

    controller = (evaluator.controller(gains_or_controller) if isinstance(gains_or_controller, np.ndarray)
                  else gains_or_controller)
    sensor = replace(c.sensor, seed=seed)
    if getattr(sensor, "needs_accelerations", False):
        sensor = replace(sensor, control_dt=c.control_dt)
    res = simulate(dyn, controller, duration=c.duration, dt=dt, initial_state=dyn.initial_state(start_pitch),
                   control_dt=c.control_dt, sensor=sensor, latency=c.latency, disturbance=disturbance)
    return evaluator.score(res), res.fell, float(np.std(res.states[:, 3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--periods", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--model", choices=sorted(MODEL), default="block",
                    help="block: on-time |d| T at the start of each period T; pdm: pulse-density, ticks of T")
    a = ap.parse_args()
    global PULSED
    PULSED = MODEL[a.model]
    print(f"pulse model: {a.model}")
    assert C.actuator.model == "slow_pwm", "legobot_config should use the slow_pwm actuator"

    print("1. Free-motor ripple sd (rad/s) through the 100 Hz, 1-degree encoder; bench (pwm_probe) in the last row")
    duties = sorted(BENCH_RIPPLE)
    print("   period  " + "".join(f"  d={d:<5}" for d in duties))
    fits = {}
    for T in [None] + a.periods:
        sds = [free_motor_ripple(T, d)[1] for d in duties]
        label = "averaged" if T is None else f"{T * 1e3:4.0f} ms"
        print(f"   {label:8}" + "".join(f"  {s:7.2f}" for s in sds))
        if T is not None:
            # the bench adds about BASELINE_RIPPLE of its own (friction around the turn), in quadrature
            model = np.sqrt(np.array(sds) ** 2 + BASELINE_RIPPLE ** 2)
            fits[T] = float(np.sqrt(np.mean((model - np.array([BENCH_RIPPLE[d] for d in duties])) ** 2)))
    print("   bench   " + "".join(f"  {BENCH_RIPPLE[d]:7.2f}" for d in duties))
    best = min(fits, key=fits.get)
    print("   rms misfit after adding the bench's 1.2 rad/s floor: "
          + ", ".join(f"{T * 1e3:.0f} ms {v:.2f}" for T, v in fits.items()) + f"  -> best {best * 1e3:.0f} ms")

    print("\n2. Baselines on the LEGO scenario (nominal robot, sensor noise), dt 0.5 ms, averaged vs pulsed")
    ev = Evaluator(C.build_eval_config(), C.build_search_space())
    seeds = (1,) if a.quick else (1, 2, 3)
    scen = [(p, s) for p in ev.config.start_pitches for s in seeds]
    ctrls = [("hand-tuned PID", np.array(C.controller.hand_tuned_gains)),
             ("model-based PID", np.array(C.model_based_gains()))]   # LQR dropped for this robot (A4_BASELINES)
    periods = [None] + sorted(set([best] + list(a.periods[-2:])))
    print(f"   {len(scen)} runs each; cost (falls), and the sd of the pitch rate (rad/s)")
    print("   " + " " * 18 + "".join(f"{'averaged' if T is None else f'pulsed {T * 1e3:.0f} ms':>26}" for T in periods))
    for name, g in ctrls:
        cells = []
        for T in periods:
            costs, falls, rates = [], 0, []
            for p, s in scen:
                ctrl = g
                cst, fell, rate_sd = run(ctrl, ev, p, s, T, 0.0005)
                costs.append(cst)
                falls += fell
                rates.append(rate_sd)
            cells.append(f"{np.mean(costs):9.3f} ({falls}) {np.mean(rates):6.3f}")
        print(f"   {name:18}" + "".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    main()
