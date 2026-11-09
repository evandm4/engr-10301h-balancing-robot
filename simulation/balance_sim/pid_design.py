"""A cascaded PID designed from the linear model alone: the model-based baseline.

The textbook sequential cascade design (PHYSICS.md 3.4, 3.5, 6.2, 6.3), with no
simulation and no tuning:

1. Inner pitch loop. Its characteristic polynomial (PHYSICS.md 3.5) is
       Delta_0 s^3 + c kd s^2 + (c kp - a M g l) s + c ki,
   so its three gains place three poles exactly. They go at a pair with natural
   frequency `inner_speed` x the open-loop unstable pole p and damping ratio
   `inner_damping`, plus a slow integral pole `integral_ratio` x that frequency.
2. Outer velocity loop. With the inner loop holding the commanded lean, speed
   follows dv/dt = G theta_sp (PHYSICS.md 2.8), a first-order loop of bandwidth
   G kv_p. kv_p is set so that bandwidth is `outer_fraction` of the
   right-half-plane zero z (PHYSICS.md 3.3), the standard limit for a
   non-minimum-phase plant (Skogestad & Postlethwaite).

The defaults are the most common choices that leave the full loop stable with
the configured 10 ms latency: an inner loop critically damped at 2p, and an
outer bandwidth of z/3. Check any design with stability.stable_latency_range;
z/2, or an underdamped inner loop, is linearly unstable at 10 ms for this robot
(cascade bound, PHYSICS.md 6.3).
"""

import math
from typing import Tuple

from .params import RobotParams


def _plant_constants(p: RobotParams):
    """a, b0, d, Delta_0, c, M g l (PHYSICS.md 1) at upright."""
    r = p.wheel_radius
    a = p.body_mass + p.wheel_mass + (p.wheel_inertia + p.rotor_inertia) / r**2
    b0 = p.body_mass * p.body_com_height - p.rotor_inertia / r
    d = p.body_mass * p.body_com_height**2 + p.body_inertia + p.rotor_inertia
    mgl = p.body_mass * p.gravity * p.body_com_height
    return a, b0, d, a * d - b0**2, a + b0 / r, mgl


def model_based_pid_gains(p: RobotParams, inner_speed: float = 2.0, inner_damping: float = 1.0,
                          integral_ratio: float = 0.1,
                          outer_fraction: float = 1.0 / 3.0) -> Tuple[float, float, float, float]:
    """Cascaded PID gains (kp, ki, kd, kv_p) from the linear model; see the module docstring."""
    a, b0, d, det0, c, mgl = _plant_constants(p)
    pole = math.sqrt(a * mgl / det0)                    # open-loop unstable pole, PHYSICS.md 3.2
    zero = math.sqrt(mgl / (d + b0 * p.wheel_radius))   # RHP zero of X/T, PHYSICS.md 3.3
    lean_gain = mgl / (b0 + a * p.wheel_radius)         # G, m/s^2 per rad of steady lean, PHYSICS.md 2.8

    w = inner_speed * pole
    lam = integral_ratio * w
    # (s + lam)(s^2 + 2 zeta w s + w^2) = s^3 + s2 s^2 + s1 s + s0
    s2 = 2 * inner_damping * w + lam
    s1 = w * w + 2 * inner_damping * w * lam
    s0 = lam * w * w
    kd = det0 * s2 / c
    kp = (det0 * s1 + a * mgl) / c
    ki = det0 * s0 / c
    kv = outer_fraction * zero / lean_gain
    return kp, ki, kd, kv
