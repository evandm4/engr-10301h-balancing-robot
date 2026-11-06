# The physics in the simulation: derivations, justification, limitations

This document derives every equation in the simulator from first principles. For each one it shows where the equation lives in the code, checks it numerically, and lists what the model leaves out and what to do about it. It covers `simulation/balance_sim/`, the reference simulator. `simulation/batch_sim/` runs the same equations in the same order (Section 11), so everything here applies to it too.

All numbers use the robot defined in `config.py`. Two sets of gains appear throughout:

- **Tuned:** the 2026-11-06 CMA-ES result in `results/batch_tuning/20261106-154750/`: $k_p = 0.660$, $k_i = 0.498$, $k_d = 0.1115$, $k_v = 0.796$. It was tuned for the current Bryson-rule cost with friction and 5–20 ms latency randomized.
- **Hand-tuned:** `ControllerConfig.hand_tuned_gains`: $k_p = 4.0$, $k_i = 0.5$, $k_d = 0.15$, $k_v = 0.1$.

Sections 8.2 and 9.4 also show the earlier 2026-09-30 set ($k_p = 0.730$, $k_i = 0.230$, $k_d = 0.0572$, $k_v = 0.252$) for comparison.

Here $k_v$ is the code's `kv_p`. Every number quoted is printed by

```
python simulation/examples/physics_checks.py
```

whose output is labeled with this document's section numbers. It takes about 20 seconds.

## Contents

1. [Model and notation](#1-model-and-notation)
2. [Equations of motion](#2-equations-of-motion-planar2dpy) (`planar2d.py`)
3. [Linearization](#3-linearization)
4. [Actuator model](#4-actuator-model-actuatorspy) (`actuators.py`)
5. [Sensor model](#5-sensor-model-sensorspy) (`sensors.py`)
6. [Controller](#6-controller-controllerspy) (`controllers.py`)
7. [Time stepping, timing and delay](#7-time-stepping-timing-and-delay-simulatorpy) (`simulator.py`)
8. [Cost function and robust scoring](#8-cost-function-and-robust-scoring-metricspy-robustpy) (`metrics.py`, `robust.py`)
9. [Checking the assumptions against simulated runs](#9-checking-the-assumptions-against-simulated-runs)
10. [Limitations and future iterations](#10-limitations-and-future-iterations)
11. [The batched engine uses the same physics](#11-the-batched-engine-uses-the-same-physics)

## Main results

**What checks out:**

- **The equations of motion are correct.** Two independent derivations (Lagrange and Newton–Euler) agree, and both match `Planar2D.derivatives` term for term. The mass matrix is invertible at every pitch angle (proof in 2.5). Energy is conserved without torque and friction, and otherwise changes by exactly the work the motor and friction do (to $1.4\times10^{-8}$ J).
- **`critical_pitch_gain` is exact:** on the linearized robot, below it no choice of the other gains can hold the robot up (3.4). The integrator (RK4) shows its expected 4th-order convergence (7.1).

**What the derivations explain:**

- **The robot has an unstable pole at 13.1 rad/s and a right-half-plane zero at 7.46 rad/s** (3.2, 3.3). To speed up it must first roll backward, and the simulator does exactly that.
- **New closed-form result (6.3):** the cascaded controller's characteristic polynomial gives a ceiling on the outer-loop gain: $k_v < (c/e)\,(k_d/k_p)$. This explains why outer gains above about 0.25 oscillate with the hand-tuned inner loop. The tuned inner loop has $k_d/k_p = 0.169$ s, so its ceiling is 1.098, and the tuned $k_v = 0.796$ is below it.
- **An exact discrete-time analysis gives each controller's delay margin** (7.4). The tuned gains tolerate 36 ms of latency; the hand-tuned ones tolerate 11.5 ms. The nonlinear simulator confirms both to within 1 ms.

**What could break on hardware**, in order of severity (Section 9):

1. **The motor driver.** The model assumes the driver delivers the commanded current. If the real driver instead sets PWM duty in proportion to the torque command, the current tuned gains still stay up (0 of 30 fall), but their cost rises about 9 times (0.140 to 1.246). The 2026-09-30 gains fell in 30 of 30 runs, because the back-EMF drag was above their runaway threshold (9.3, 9.4).
2. **Friction.** Its size is unmeasured. With the earlier cost, viscous friction of 0.003 N m s/rad or more made the 2026-09-30 tuned gains fall: they have a small $k_pk_v$, and any load torque then pushes the speed up by $\tau_L/(k_pk_v)$ (9.4). Since 2026-11-06 robust scoring randomizes friction, and the retuned gains nearly triple $k_pk_v$ (9.4).
3. **The IMU model.** It is idealized: a raw accelerometer tilt estimate would be off by up to 41° (tuned) or 75° (hand-tuned) during these transients.
4. **Traction.** The no-slip assumption needs a friction coefficient of only 0.28 for the tuned gains but 1.18 for the hand-tuned ones.

---

## 1. Model and notation

The robot is a planar wheeled inverted pendulum: one rigid body pivoting on the axle of a pair of wheels that roll on flat ground. Both wheels are lumped into one (`params.py:7-8`).

```
                     G  body center of mass: (x + l sin θ, r + l cos θ)
                    /
                   /  l
                  /
                 /  θ: pitch from vertical, + = leaning forward
                /
           ----( ● )----   axle at (x, r); motors on the body drive the wheel
               (     )     wheel: radius r, angle φ; rolls without slip, x = r φ
   ===========================================================  +x (forward) →
```

| Symbol | Meaning | `config.py` field | Value |
|---|---|---|---|
| $M$ | body mass (everything above the wheels, motors included) | `body_mass` | 0.80 kg |
| $l$ | height of the body's center of mass above the axle | `body_com_height` | 0.10 m |
| $I_b$ | body pitch inertia about its own center of mass | `body_inertia` | 0.0033 kg m² |
| $m_w$ | mass of both wheels | `wheel_mass` | 0.10 kg |
| $r$ | wheel radius | `wheel_radius` | 0.035 m |
| $I_w$ | inertia of both wheels about the axle | `wheel_inertia` | 6.1e-5 kg m² |
| $b$ | viscous friction between wheels and body | `wheel_friction` | 0 |
| $g$ | gravity | `gravity` | 9.81 m/s² |
| $\tau_{\max}$ | torque limit, both motors together | `max_torque` | 0.40 N m |

The values are self-consistent. $I_w = \tfrac12 m_w r^2$ exactly, which is two solid disks. A uniform 0.8 kg block 0.20 m tall and 0.05 m deep, with its center of mass 0.10 m above the axle, has $I_b = 0.0028$ kg m², close to the 0.0033 used.

Four combinations recur everywhere:

```math
a = M + m_w + \frac{I_w}{r^2} = 0.9498\ \text{kg},\qquad
d = M l^2 + I_b = 0.0113\ \text{kg m}^2,
```
```math
\Delta_0 = a d - (Ml)^2 = 4.333\times10^{-3}\ \text{kg}^2\text{m}^2,\qquad
c = a + \frac{Ml}{r} = 3.2355\ \text{kg}.
```

- $a$ is the robot's effective translating mass. Wheel spin adds $I_w/r^2 = 0.0498$ kg to it.
- $d$ is the body's pitch inertia about the axle (parallel-axis theorem).
- $\Delta_0$ is the mass-matrix determinant when upright.
- $c$ converts motor torque into restoring moment (3.4).

**Sign conventions.**

- $x$ is positive forward.
- $\theta$ is measured from vertical, positive when leaning forward.
- $\varphi$ is the wheel angle, positive when rolling forward.

With forward drawn to the right, $\varphi$ and $\theta$ both increase clockwise. So the motor shaft angle (rotor relative to the stator, which is bolted to the body) is $\varphi-\theta$, and the motor speed is

```math
\omega = \dot\varphi - \dot\theta = \frac{\dot x}{r} - \dot\theta .
```

This is `relative_wheel_speed` (`planar2d.py:102`).

$\tau$ is the total torque the motors apply to the wheels. By Newton's third law the body receives $-\tau$. The state vector is $(x, \dot x, \theta, \dot\theta)$ (`dynamics.py:10`).

## 2. Equations of motion (`planar2d.py`)

### 2.1 Kinematics

Rolling without slipping means the contact point is momentarily at rest, so $\dot x = r\dot\varphi$. In planar motion this constraint integrates to $x = r\varphi$, so $\varphi$ is not an independent coordinate. Two generalized coordinates remain: $q = (x, \theta)$.

The axle is at $(x, r)$. The body's center of mass $G$ and its velocity are

```math
\mathbf p_G = \bigl(x + l\sin\theta,\; r + l\cos\theta\bigr),\qquad
\dot{\mathbf p}_G = \bigl(\dot x + l\dot\theta\cos\theta,\; -l\dot\theta\sin\theta\bigr),
```
```math
\lvert\dot{\mathbf p}_G\rvert^2 = \dot x^2 + 2l\,\dot x\dot\theta\cos\theta + l^2\dot\theta^2 .
```

### 2.2 Kinetic and potential energy

The kinetic energy has four parts: wheel translation, wheel spin, body translation and body rotation. Using $\dot\varphi = \dot x/r$:

```math
T = \tfrac12 m_w\dot x^2 + \tfrac12 I_w\dot\varphi^2 + \tfrac12 M\lvert\dot{\mathbf p}_G\rvert^2 + \tfrac12 I_b\dot\theta^2
  = \tfrac12 a\,\dot x^2 + Ml\cos\theta\,\dot x\dot\theta + \tfrac12 d\,\dot\theta^2 .
```

Only the body's height changes, so the potential energy (dropping constants) is

```math
V = Mgl\cos\theta .
```

In matrix form $T = \tfrac12\dot q^{\mathsf T}\mathbf M(\theta)\dot q$, with

```math
\mathbf M(\theta) = \begin{pmatrix} a & Ml\cos\theta \\ Ml\cos\theta & d \end{pmatrix}.
```

These are the $T$ and $V$ in the `planar2d.py` docstring and in `Planar2D.energy` (`planar2d.py:139`).

### 2.3 Generalized forces

Two torques act across the wheel–body joint:

- the motor torque: $+\tau$ on the wheels, $-\tau$ on the body;
- viscous friction: $-b\omega$ on the wheels, $+b\omega$ on the body.

Their virtual work is

```math
\delta W = (\tau - b\omega)(\delta\varphi - \delta\theta) = \tau_{\text{net}}\left(\frac{\delta x}{r} - \delta\theta\right),
\qquad \tau_{\text{net}} = \tau - b\,\omega,
```

so the generalized forces are

```math
Q_x = \frac{\tau_{\text{net}}}{r},\qquad Q_\theta = -\tau_{\text{net}} .
```

`planar2d.py:103-107` computes $\tau_{\text{net}}$ the same way: it gets the delivered torque from the actuator (Section 4), then subtracts friction. Friction modeled this way is equivalent to a Rayleigh dissipation function $\mathcal F = \tfrac12 b\omega^2$. Because it acts between two parts of the robot, it can only dissipate energy (2.7).

### 2.4 Euler–Lagrange equations

With $L = T - V$:

```math
\frac{\partial L}{\partial\dot x} = a\dot x + Ml\cos\theta\,\dot\theta,\qquad
\frac{d}{dt}\frac{\partial L}{\partial\dot x} = a\ddot x + Ml\cos\theta\,\ddot\theta - Ml\sin\theta\,\dot\theta^2,\qquad
\frac{\partial L}{\partial x} = 0,
```
```math
\frac{\partial L}{\partial\dot\theta} = Ml\cos\theta\,\dot x + d\dot\theta,\qquad
\frac{d}{dt}\frac{\partial L}{\partial\dot\theta} = Ml\cos\theta\,\ddot x - Ml\sin\theta\,\dot x\dot\theta + d\ddot\theta,
```
```math
\frac{\partial L}{\partial\theta} = -Ml\sin\theta\,\dot x\dot\theta + Mgl\sin\theta .
```

Setting $\frac{d}{dt}\frac{\partial L}{\partial\dot q_i} - \frac{\partial L}{\partial q_i} = Q_i$ gives:

```math
\text{(E1)}\qquad a\,\ddot x + Ml\cos\theta\,\ddot\theta = \frac{\tau_{\text{net}}}{r} + Ml\sin\theta\,\dot\theta^2
```
```math
\text{(E2)}\qquad Ml\cos\theta\,\ddot x + d\,\ddot\theta = -\tau_{\text{net}} + Mgl\sin\theta
```

In (E2) the $\dot x\dot\theta$ terms cancel. The two equations have a direct physical reading (2.6):

- (E1) is Newton's second law for the whole robot in the horizontal direction.
- (E2) is the body's moment balance about the axle.

### 2.5 Solving for the accelerations

(E1) and (E2) form the linear system $\mathbf M(\theta)\,(\ddot x, \ddot\theta)^{\mathsf T} = (\text{rhs}_x, \text{rhs}_\theta)^{\mathsf T}$. Cramer's rule solves it:

```math
\ddot x = \frac{d\,\text{rhs}_x - b(\theta)\,\text{rhs}_\theta}{\Delta(\theta)},\qquad
\ddot\theta = \frac{a\,\text{rhs}_\theta - b(\theta)\,\text{rhs}_x}{\Delta(\theta)},\qquad
\Delta(\theta) = ad - b(\theta)^2,\quad b(\theta) = Ml\cos\theta .
```

(Here $b(\theta)$ is the code's variable `b`, the off-diagonal mass-matrix entry, not the friction coefficient.) Each line of `Planar2D.derivatives` is one of these terms:

| Term | Code |
|---|---|
| $\omega = \dot x/r - \dot\theta$ | `planar2d.py:102` |
| $\tau_{\text{net}}$ = actuator output $- b\omega$ | `planar2d.py:103-107` |
| $a$, $b(\theta)$, $d$ | `planar2d.py:112-114` |
| $\text{rhs}_x = \tau_{\text{net}}/r + Ml\sin\theta\,\dot\theta^2$ | `planar2d.py:116` |
| $\text{rhs}_\theta = -\tau_{\text{net}} + Mgl\sin\theta$ | `planar2d.py:117` |
| $\Delta$, $\ddot x$, $\ddot\theta$ | `planar2d.py:119-121` |

**The solve never fails.** Because $\cos^2\theta \le 1$,

```math
\Delta(\theta) = ad - M^2l^2\cos^2\theta \;\ge\; ad - M^2l^2 = \Bigl(m_w + \frac{I_w}{r^2}\Bigr)d + M I_b \;>\; 0 .
```

The middle equality comes from expanding $a = M + m_w + I_w/r^2$. Every term on the right is positive, which `RobotParams.__post_init__` enforces. So $\mathbf M(\theta)$ is positive definite at every angle, including past the fall angle. For this robot $\Delta$ ranges from $4.33\times10^{-3}$ (upright) to $ad = 1.07\times10^{-2}$ (horizontal).

### 2.6 Independent check: Newton–Euler

Let $F$ be the horizontal ground (friction) force on the wheels and $N$ the normal force.

**Horizontal.** Momentum balance for the whole robot gives $F = m_w\ddot x + M\ddot x_G$. Differentiating $x_G$ from 2.1:

```math
F = (M + m_w)\,\ddot x + Ml\bigl(\cos\theta\,\ddot\theta - \sin\theta\,\dot\theta^2\bigr).
```

The wheels' spin about the axle gives $I_w\ddot\varphi = \tau_{\text{net}} - Fr$, so

```math
F = \frac{1}{r}\Bigl(\tau_{\text{net}} - \frac{I_w\ddot x}{r}\Bigr).
```

Equating the two expressions for $F$ gives exactly (E1).

**Body moments about the moving axle point $A$** (clockwise positive, like $\theta$). The balance is $\sum M_A = I_b\ddot\theta + (\mathbf p_G - \mathbf p_A)\times M\mathbf a_G$, where the moments are gravity's $Mgl\sin\theta$ and the joint's $-\tau_{\text{net}}$. The axle force passes through $A$, so it contributes no moment. Expanding the cross product gives $Ml^2\ddot\theta + Ml\cos\theta\,\ddot x$, which is exactly (E2).

**Vertical.** The same approach gives the normal force (used in 9.1):

```math
N = (M + m_w)\,g - Ml\bigl(\sin\theta\,\ddot\theta + \cos\theta\,\dot\theta^2\bigr).
```

*Numerical check (9.1–9.5 output):* along all 30 validation runs, the Lagrange and Newton–Euler expressions for $F$ agree to $2\times10^{-14}$ N.

### 2.7 Energy and power balance

Let $E = T + V$. For a Lagrangian whose kinetic energy is quadratic in $\dot q$ and has no explicit time dependence, the energy function $\sum_i \dot q_i\,\partial L/\partial\dot q_i - L$ equals $T + V$. Its rate of change is the power of the generalized forces. Here that gives

```math
\frac{dE}{dt} = Q_x\dot x + Q_\theta\dot\theta = \tau_{\text{net}}\Bigl(\frac{\dot x}{r} - \dot\theta\Bigr) = \tau\,\omega - b\,\omega^2 .
```

Three consequences:

1. With $\tau = 0$ and $b = 0$, energy is conserved. `test_energy_conserved_without_torque_or_friction` checks this.
2. Friction only removes energy, since $-b\omega^2 \le 0$. `test_friction_dissipates_energy` checks this.
3. The motor's mechanical power is $\tau\omega$, so the speed the actuator must be asked about is the relative speed $\omega$. That is why `Actuator.torque` takes `relative_wheel_speed`.

*Numerical check (2.7 output):* one 10 s closed-loop run with $b = 0.001$. The energy change of 0.091751 J matches the integrated power of 0.091751 J; the difference is $1.4\times10^{-8}$ J.

### 2.8 Equilibria, and lean versus acceleration

**Upright equilibrium.** With $\tau_{\text{net}} = 0$ and $\theta = \dot\theta = 0$, both accelerations vanish for any speed $\dot x$. So the robot can stand upright at any constant speed (`test_upright_equilibrium_holds`).

**Steady lean.** Hold $\theta = \theta_s$ constant ($\dot\theta = \ddot\theta = 0$). Then (E1) gives $\tau = a r\ddot x$. Substituting into (E2):

```math
\ddot x = \frac{Mgl\sin\theta_s}{Ml\cos\theta_s + a r},\qquad \tau = a r\,\ddot x .
```

Linearized, $\ddot x \approx G\,\theta_s$ with

```math
G = \frac{Mgl}{Ml + ar} = 6.93\ \text{m/s}^2\ \text{per rad}.
```

A point mass on a massless cart would give $G = g = 9.81$. This robot needs a larger lean for the same acceleration, because the torque that drives the wheels also pushes back on the body.

At the 25° lean cap (`ControllerConfig.max_lean_deg`): $\ddot x = 3.14$ m/s², which needs only $\tau = 0.104$ N m. So the cap limits acceleration well before the motors run out of torque at low speed. $G$ is the plant gain that the outer velocity loop works through (6.2).

## 3. Linearization

### 3.1 Linear model

Linearize about upright: $\sin\theta\approx\theta$, $\cos\theta\approx1$, drop $\dot\theta^2$. Also take $b = 0$ and an unsaturated actuator (delivered torque = command). Then

```math
\begin{pmatrix} a & Ml \\ Ml & d \end{pmatrix}\begin{pmatrix}\ddot x\\ \ddot\theta\end{pmatrix}
= \begin{pmatrix}\tau/r \\ Mgl\,\theta - \tau\end{pmatrix}
\quad\Longrightarrow\quad
\ddot x = \frac{-M^2l^2g\,\theta + (d/r + Ml)\,\tau}{\Delta_0},\qquad
\ddot\theta = \frac{aMgl\,\theta - c\,\tau}{\Delta_0}.
```

With state $s = (x, \dot x, \theta, \dot\theta)$, this is $\dot s = A s + B\tau$:

```math
A = \begin{pmatrix} 0&1&0&0\\ 0&0&-14.49&0\\ 0&0&0&1\\ 0&0&172.04&0 \end{pmatrix},\qquad
B = \begin{pmatrix} 0\\ 92.98\\ 0\\ -746.77 \end{pmatrix}.
```

The signs match the tests:

- $B_2 > 0$ and $B_4 < 0$: positive torque drives the wheels forward and tips the body back (`test_positive_torque_pushes_wheels_forward_and_body_back`).
- $A_{4,3} > 0$: leaning forward accelerates the fall (`test_leaning_forward_accelerates_pitch_forward_without_torque`).

### 3.2 Open-loop poles

The characteristic polynomial is $s^2\,(s^2 - aMgl/\Delta_0)$. Its roots are:

- $0, 0$: position and velocity, which nothing pulls back;
- $\pm p$, with $p = \sqrt{aMgl/\Delta_0} = 13.12$ rad/s.

The unstable pole $+p$ means a 76 ms time constant: a small tilt doubles every 53 ms.

- **Falling from rest:** the linear estimate of the time to fall from 0.05 rad to 60° is $\cosh^{-1}(1.047/0.05)/p = 0.285$ s. The simulator gives 0.311 s; the nonlinear terms account for the difference. `test_uncontrolled_robot_falls` allows 0.1–1 s.
- **Free versus locked wheels:** with the wheels locked the pole would be $\sqrt{Mgl/d} = 8.33$ rad/s. Free wheels fall faster, because the base rolls backward as the body tips forward.

### 3.3 Transfer functions and the right-half-plane zero

From the Laplace transform of 3.1:

```math
\frac{\Theta(s)}{T(s)} = \frac{-c}{\Delta_0 s^2 - aMgl},\qquad
\frac{X(s)}{T(s)} = \frac{(d/r + Ml)\,s^2 - Mgl/r}{s^2\,(\Delta_0 s^2 - aMgl)} .
```

$X/T$ has zeros at $\pm z$, where

```math
z = \sqrt{\frac{Mgl}{d + Mlr}} = 7.46\ \text{rad/s}.
```

The zero at $+z$ is in the right half plane, which makes position and velocity **non-minimum-phase**. To accelerate forward the robot must lean forward, and the only way to tip the body forward is to drive the wheels backward under it first. So a step up in goal velocity starts with motion the wrong way.

The simulator shows this (3 output, tuned gains, seed 100). The goal now blends smoothly over 2.5 s, so the effect is small:

- **Goal 0 → 1.0 m/s, starting at t = 2 s:** velocity goes from 0.009 to −0.031 m/s within 0.3 s before rising.
- **Goal 0 → −0.6 m/s, starting at t = 6 s:** velocity first rises from 0.018 to 0.034 m/s within 0.4 s.

The RHP zero is a fundamental limit on velocity control (6.2).

### 3.4 Critical pitch gain (`critical_pitch_gain`, `planar2d.py:41`)

Eliminate $\ddot x$ between the two linear equations. (E1) gives $\ddot x = (\tau/r - Ml\ddot\theta)/a$; substituting into (E2):

```math
\underbrace{\Bigl(d - \frac{M^2l^2}{a}\Bigr)}_{\Delta_0/a}\ddot\theta
= Mgl\,\theta - \underbrace{\Bigl(1 + \frac{Ml}{ar}\Bigr)}_{c/a}\tau .
```

The motor's restoring moment on the body is $(c/a)\,\tau$, which has two parts:

- the direct reaction torque $\tau$;
- an extra $(Ml/ar)\,\tau$ from accelerating the axle back under the center of mass.

For this robot $c/a = 3.41$, so the wheel acceleration contributes 2.4 times as much restoring moment as the reaction torque.

With proportional control $\tau = k_p\theta$ the equation becomes $(\Delta_0/a)\,\ddot\theta = -\bigl((c/a)k_p - Mgl\bigr)\theta$. This is a restoring spring if and only if

```math
k_p > k_{p,\text{crit}} = \frac{aMgl}{c} = 0.2304\ \text{N m/rad}.
```

That is `critical_pitch_gain`, line for line. It is 3.4 times smaller than the naive static answer $Mgl = 0.785$, which ignores the wheel-acceleration term.

The search space uses $1.1\,k_{p,\text{crit}}$ as the lower bound on $k_p$ (`search_space.py:87`). Gains below $k_{p,\text{crit}}$ fall in the linear model whatever the other gains are (`test_gain_threshold_matches_simulation`), so the tuner never searches them. Since 2026-11-06, $k_{p,\text{crit}}$ there is taken for the most demanding robot that robust scoring can draw (`ProjectConfig.kp_floor_robot`: mass, COM height and wheel inertia at the top of their spreads). It needs 0.273 N m/rad, 1.19 times the nominal robot, so the floor is 0.300 rather than 0.253.

### 3.5 Full pitch-loop stability (Routh–Hurwitz)

Add damping and integral action: $\tau = k_p\theta + k_d\dot\theta + k_i\!\int\!\theta$. Then the pitch loop's characteristic polynomial is

```math
\Delta_0\,s^3 + c\,k_d\,s^2 + (c\,k_p - aMgl)\,s + c\,k_i = 0 .
```

The Routh–Hurwitz conditions for stability are $k_d > 0$, $k_i > 0$, and

```math
k_p > k_{p,\text{crit}} + \frac{\Delta_0\,k_i}{c\,k_d}.
```

So $k_{p,\text{crit}}$ is necessary but not sufficient. The bound is 0.2364 for the tuned gains and 0.2348 for the hand-tuned ones.

The continuous-time pitch-loop poles, ignoring sampling and delay, are:

| Gains | Poles (rad/s) | $k_p / k_{p,\text{crit}}$ |
|---|---|---|
| Tuned | −79.3, −1.99 ± 0.85j | 2.9 |
| Hand-tuned | −74.1, −37.8, −0.13 | 17.4 |

These poles ignore delay, which is what actually limits the gains (7.4).

### 3.6 Why pitch-only control drifts

A pitch-only PID never feeds back $x$ or $\dot x$, so the two zero eigenvalues of 3.2 stay at zero. The pitch integrator does absorb a constant torque offset, such as the gyro bias's $k_db_g$ (5.3). But the transient while it does so leaves a velocity offset that nothing removes, so the robot drifts at a constant speed. `CONDITIONS.md` reports a pitch-only controller with gyro bias that drifted 0.39 m in 10 s and was still moving. Feeding velocity back is what the outer loop of the cascade does (6.2).

## 4. Actuator model (`actuators.py`)

### 4.1 One brushed DC motor

The armature circuit, and the motor's torque, are

```math
V = R\,i + L\frac{di}{dt} + k_e\,\omega_m,\qquad \tau_m = k_t\,i .
```

- **Inductance is neglected** (quasi-static current). This holds if the electrical time constant $L/R$ is much shorter than the 5 ms control period. That is typical of small gearmotors, but the datasheet should confirm it (10.3).
- **$k_e = k_t$ in SI units.** The electrical power converted, $k_e\omega_m i$, must equal the mechanical power $\tau_m\omega_m = k_t i\omega_m$. So one `torque_constant` serves for both (`actuators.py:49`).

### 4.2 Gearbox, two motors, and which speed

- **Gearbox:** a lossless ratio $N$ gives $\omega_m = N\omega$ and $\tau_{\text{out}} = N\tau_m$.
- **Two motors:** $n$ identical motors carrying equal current give a total torque $\tau = nNk_t\,i$.
- **Which speed:** the stator is bolted to the body, so the speed that generates back-EMF is the relative speed $\omega$ (2.7), not the wheel's absolute speed.

### 4.3 What `DCMotorActuator.torque` computes (`actuators.py:99-112`)

1. Clip the command: $\tau_c = \operatorname{clip}(\tau_{\text{cmd}}, \pm\tau_{\max})$.
2. Current needed for that torque: $i^* = \tau_c/(nNk_t)$.
3. Voltage needed to push $i^*$ against the back-EMF: $V^* = R\,i^* + k_tN\omega$.
4. Clip to the battery: $V = \operatorname{clip}(V^*, \pm V_s)$.
5. Resulting current, clipped to the driver limit: $i = \operatorname{clip}\bigl((V - k_tN\omega)/R,\ \pm i_{\max}\bigr)$.
6. Delivered torque: $\tau = nNk_t\,i$.

**When neither clip is active, $i = i^*$ and $\tau = \tau_c$ exactly.** The model is a current-controlled (torque-controlled) drive with perfect back-EMF compensation. The motor's physics only show up through its two limits. That assumption matters a great deal (9.3).

### 4.4 The torque–speed envelope

**Forward torque at forward motor speed $\omega \ge 0$:**

```math
\tau^+_{\max}(\omega) = \min\Bigl\{\tau_{\max},\; nNk_t\,i_{\max},\; nNk_t\,\frac{V_s - k_tN\omega}{R}\Bigr\}.
```

**Braking ($\tau < 0$ while $\omega > 0$):** setting $V = -V_s$ would drive the current to $-(V_s + k_tN\omega)/R$, past the current limit. So braking is current-limited at every speed. This is plugging; it assumes the driver and battery can carry that current.

With the configured motor (4 output):

| Quantity | Formula | Value |
|---|---|---|
| Stall torque | $nNk_t\min(i_{\max}, V_s/R)$ | 0.40 N m. Current-limited, since $V_s/R = 2.47$ A > 2 A. It equals `max_torque`. |
| No-load speed | $V_s/(k_tN)$ | 74 rad/s, which is 2.59 m/s with the body upright and still |
| Corner speed | $(V_s - Ri_{\max})/(k_tN)$ | 14 rad/s, which is 0.49 m/s |

Above the corner speed, forward torque falls linearly with speed:

| Speed | Back-EMF | Forward torque available |
|---|---|---|
| 0.8 m/s | 2.29 V | 0.341 N m |
| 1.0 m/s | 2.86 V | 0.303 N m |
| 1.4 m/s | 4.00 V | 0.227 N m (57% of stall) |
| 2.0 m/s | 5.71 V | 0.112 N m (28% of stall) |

The scenario's largest goal is 1.0 m/s, and the tuned runs peak at 0.86 m/s (hand-tuned 0.70 m/s), where 0.30 N m or more is available (9.1–9.5 output). So the motor model almost never binds in this scenario: the motor delivers less than the clipped command 0.00% of the time (tuned) and 0.01% (hand-tuned). (With the earlier step goals, tuned runs reached 2.0 m/s and it bound for 0.4% of the time.)

### 4.5 Ideal actuator and friction

`IdealActuator` returns $\operatorname{clip}(\tau_{\text{cmd}}, \pm\tau_{\max})$. It is the DC motor model in the limit $V_s, i_{\max}\to\infty$. Friction is applied after either actuator (`planar2d.py:104-107`), as derived in 2.3.

### 4.6 The cost's "effort" is copper loss

When unsaturated, each motor carries $i = \tau/(nNk_t)$, so the resistive heating in all motors is

```math
P_{\text{Cu}} = nR\,i^2 = \frac{R}{nN^2k_t^2}\,\tau^2 = 150\ \text{W per (N m)}^2 .
```

So the cost's effort term $\int\tau^2\,dt$ (Section 8) is proportional to the energy dissipated in the motor windings. This gives the effort weight a physical meaning.

## 5. Sensor model (`sensors.py`)

### 5.1 Model

Once per control tick, the controller sees

```math
\hat\theta = \theta + \sigma_\theta\xi_1,\qquad
\hat{\dot\theta} = \dot\theta + \sigma_{\dot\theta}\xi_2 + b_g,\qquad
\hat{\dot x} = \dot x + \sigma_v\xi_3,
```

where the $\xi$ are independent standard normal draws, and $b_g$ is a constant gyro bias (`sensors.py:37-44`). Position is also noised, but the controller never reads it. The project values are $\sigma_\theta = 0.005$ rad, $\sigma_{\dot\theta} = 0.02$ rad/s, $b_g = 0.01$ rad/s and $\sigma_v = 0.02$ m/s.

### 5.2 Noise propagated to torque

The controller (Section 6) gives torque sensitivities $\partial\tau/\partial\hat\theta = k_p$, $\partial\tau/\partial\hat{\dot\theta} = k_d$ and $\partial\tau/\partial\hat{\dot x} = k_pk_v$. The integral path adds only $k_iT\sigma$ per tick. With independent noise, the per-tick torque noise is

```math
\sigma_\tau = \sqrt{(k_p\sigma_\theta)^2 + (k_d\sigma_{\dot\theta})^2 + (k_pk_v\sigma_v)^2}.
```

This is 0.0112 N m (2.8% of $\tau_{\max}$) for the tuned gains, and 0.0217 N m (5.4%) for the hand-tuned ones (5 output). The tuned controller is about twice as quiet. Its lower $k_p$ removes most of the pitch noise (0.0033 against 0.0200 N m), but its velocity term is larger (0.0105 against 0.0080 N m) and now dominates.

### 5.3 Gyro bias in steady state

At constant velocity, (E1) and (E2) with $\ddot x = \ddot\theta = 0$ force $\tau_{\text{net}} = 0$ and $\theta = 0$. Two cases:

**PD inner loop ($k_i = 0$).** Setting the torque to zero gives $-k_p\theta_{\text{sp}} + k_db_g = 0$. Since $\theta_{\text{sp}} = k_v(v_{\text{goal}} - v)$, there is a steady velocity error

```math
v_{\text{goal}} - v = \frac{k_d\,b_g}{k_p\,k_v},
```

which is 2.1 mm/s for the tuned gains.

**With $k_i > 0$.** The integrator stops moving only when $\theta - \theta_{\text{sp}} = 0$, which forces $\theta_{\text{sp}} = 0$ and therefore $v = v_{\text{goal}}$. The integrator settles at $-k_db_g/k_i$, well inside its limit. Integral action rejects the bias completely.

Section 9.5 covers what this sensor model leaves out.

## 6. Controller (`controllers.py`)

### 6.1 Inner pitch loop (`controllers.py:187-198`)

At each tick $k$, with period $T = 5$ ms:

```math
e_k = \hat\theta_k - \theta_{\text{sp},k},\qquad
I_k = I_{k-1} + T\,e_k,\qquad
u_k = \operatorname{clip}\bigl(k_p e_k + k_i I_k + k_d\hat{\dot\theta}_k,\ \pm\tau_{\max}\bigr).
```

- **The integrator is backward Euler**, discrete transfer function $Tz/(z-1)$, with the anti-windup of 6.4.
- **The derivative acts on the measurement:** the term is $k_d\hat{\dot\theta}$ (the gyro), not $k_d\,\dot e$. When the goal velocity steps, $\theta_{\text{sp}}$ can jump by the full 25° in one tick, and $\dot e$ would spike ("derivative kick"). Using the gyro also avoids differencing noisy angles. The cost of this choice is that it treats $\dot\theta_{\text{sp}}$ as zero.
- **Sign:** $\theta > 0$ gives $\tau > 0$, which drives the wheels under the body (3.1).

### 6.2 Outer velocity loop (`controllers.py:279-289`)

```math
\theta_{\text{sp}} = \operatorname{clip}\bigl(k_v\,(v_{\text{goal}} - \hat{\dot x}) + k_{vi}\!\textstyle\int(\cdot),\ \pm 25^\circ\bigr),\qquad k_{vi} = 0 \text{ in this project}.
```

**Sign.** If the robot is too slow, $\theta_{\text{sp}} > 0$ and the inner loop leans it forward. By 3.3 it gets there by first rolling the wheels backward.

**Bandwidth estimate.** If the inner loop held $\theta = \theta_{\text{sp}}$ exactly, 2.8 would give $\dot v \approx G\theta_{\text{sp}} = Gk_v(v_{\text{goal}} - v)$. That is a first-order lag with bandwidth $\omega_v = Gk_v$:

| Gains | $\omega_v$ | Time constant |
|---|---|---|
| Tuned | 5.51 rad/s | 0.18 s |
| Hand-tuned | 0.69 rad/s | 1.44 s |

Two things cap $\omega_v$:

1. The outer loop must stay well below the inner loop's speed.
2. The RHP zero at $z = 7.46$ rad/s: a standard rule of thumb keeps bandwidth below $z/2 = 3.7$ rad/s (Skogestad & Postlethwaite, ch. 5).

The tuned estimate (5.51 rad/s) is above that rule of thumb, so the first-order picture is optimistic for the tuned gains. The exact poles in 6.3 show an oscillatory pair at $-2.79 \pm 9.76j$ rad/s and a slow pole at $-0.71$ rad/s, not a single lag.

The lean cap also bounds acceleration at 3.14 m/s² (2.8). There is no position loop, so position is free to drift.

### 6.3 Cascade stability in closed form: how much outer gain the inner loop allows

The estimate in 6.2 treats the two loops separately. Solving them together gives a sharper result. Use the linear plant of 3.1, written in terms of $V = sX$, and continuous control without delay. With goal 0, the error is $e = \theta + k_v v$, so

```math
T(s) = \Bigl(k_p + \frac{k_i}{s}\Bigr)\bigl(\Theta + k_vV\bigr) + k_d\,s\,\Theta .
```

Substitute into $asV + Mls^2\Theta = T/r$ and $MlsV + (ds^2 - Mgl)\Theta = -T$. Set the determinant to zero and multiply by $s$. The terms in $k_v\times(\text{pitch controller})$ cancel, which leaves

```math
\Delta_0 s^4 + (c\,k_d - e\,k_pk_v)\,s^3 + (c\,k_p - aMgl - e\,k_ik_v)\,s^2 + (c\,k_i + h\,k_pk_v)\,s + h\,k_ik_v = 0,
```

```math
e = \frac{d}{r} + Ml = 0.403\ \text{kg m},\qquad h = \frac{Mgl}{r} = 22.4\ \text{N}.
```

Two checks: with $k_v = 0$ this reduces to $s$ times the pitch-loop polynomial of 3.5, and its roots match the eigenvalues of the closed-loop state matrix to $10^{-13}$ (6.3 output).

**Every coefficient must be positive.** The $s^3$ coefficient gives a simple necessary condition:

```math
k_v < \frac{c}{e}\cdot\frac{k_d}{k_p}.
```

The velocity feedback reaches the torque as $k_pk_v v$ and acts through the non-minimum-phase path (3.3). There it subtracts from the damping of the fast pitch mode, which is what the $s^3$ coefficient represents. So **the outer gain the cascade can afford is proportional to the inner loop's ratio $k_d/k_p$.** The full Routh–Hurwitz conditions for a quartic give the exact limit (6.3 output):

| Inner loop | $k_d/k_p$ | Bound $(c/e)(k_d/k_p)$ | Exact Routh–Hurwitz limit on $k_v$ |
|---|---|---|---|
| Hand-tuned (4, 0.5, 0.15) | 0.0375 s | 0.301 | 0.295 |
| Tuned (0.660, 0.498, 0.1115) | 0.169 s | 1.357 | 1.098 |

The nonlinear simulator confirms the hand-tuned limit. With the hand-tuned inner loop and no latency:

- $k_v = 0.25$ settles to $2\times10^{-6}$ rad;
- $k_v = 0.30$ goes into a 0.37 rad limit cycle.

That is `CONDITIONS.md`'s empirical observation, "outer gains much above Kv_p of about 0.25 oscillate", now derived.

**What this says about the tuner's result.** Velocity tracking is about half of the current cost (8.2), so the tuner wants a large $k_v$. With the hand-tuned inner loop, $k_v$ is capped near 0.29. With the 10 ms latency even $k_v = 0.2$ is unstable (7.4). The tuner cut $k_p$ from 4 to 0.66 and $k_d$ only from 0.15 to 0.112, so $k_d/k_p$ went from 0.0375 to 0.169 s. That lifts the Routh–Hurwitz ceiling on $k_v$ to 1.098, and the tuned $k_v = 0.796$ sits below it. The ceiling is not what sets $k_v$, though. The delay margin peaks near $k_v = 0.8$ (7.4), which is the tighter limit. This explains the direction of the change; it does not prove the optimum is unique.

### 6.4 Anti-windup (`_limited_integral`, `controllers.py:14-51`)

**Why it is needed.** Without it, the integral term alone could ask for $k_i\times\text{integral limit} = 5\times1 = 5$ N m from an actuator that delivers 0.4 N m (`archive/CHANGES_FROM_ORIGINAL.md`, item 2).

**The algorithm.** Write the output as $u = P + kI$, where $P$ is the non-integral terms. Take the candidate step $\tilde I = \operatorname{clip}(I + eT, \pm I_{\lim})$.

- If $\lvert P + k\tilde I\rvert > u_{\max}$ and $e\cdot u > 0$, the step would push further into saturation. The integrator instead moves toward the value that puts the output exactly on the limit,
  ```math
  I^* = \frac{\operatorname{sign}(u)\,u_{\max} - P}{k},
  ```
  but no further than the candidate step, and never against the step's direction.
- Finally $\lvert kI\rvert \le u_{\max}$ is enforced.

**What it guarantees.**

1. The integrator never grows while growing could only add saturation.
2. It unwinds as soon as the error changes sign (then $e\cdot u \le 0$).
3. The integral term alone never asks for more than the actuator can deliver.

This is conditional integration (clamping), one of the standard anti-windup schemes (Åström & Hägglund). The outer loop uses the same function with the lean cap as its $u_{\max}$.

## 7. Time stepping, timing and delay (`simulator.py`)

### 7.1 RK4 integration (`simulator.py:26-31`)

Each step is

```math
s_{n+1} = s_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4),
```

with the torque command held constant over the step. The actuator and friction are re-evaluated inside every stage, so back-EMF follows the speed within a step. RK4's global error is $O(h^4)$, so halving $h$ should cut the error by a factor of 16.

*Convergence (7.1 output):* relative energy error in a 0.3 s free fall from 0.3 rad.

| Step $h$ | Relative energy error | Ratio to previous |
|---|---|---|
| 10 ms | 8.98e-5 | |
| 5 ms | 5.12e-6 | 17.6 |
| 2.5 ms (tuning step) | 3.03e-7 | 16.9 |
| 1.25 ms | 1.84e-8 | 16.4 |
| 0.625 ms | 1.13e-9 | 16.2 |

**Stability.** RK4 is stable for $\lvert\lambda h\rvert$ up to about 2.8 (Hairer, Nørsett & Wanner). The plant's fastest eigenvalue is 13.1 rad/s. With full back-EMF damping it is 27.6 rad/s (when the voltage saturates; 4.4, 9.3). So $\lvert\lambda h\rvert \le 0.07$. The 2.5 ms step is chosen for accuracy and to line up with the timing, not for stability.

**Closed loop.** The clips make the right-hand side non-smooth, which costs some order at the switching instants. Even so, the full scenario at 2.5 ms differs from 0.25 ms by at most $1.1\times10^{-9}$ rad in pitch and $3.5\times10^{-9}$ m/s in velocity for the tuned gains, and $2.7\times10^{-5}$ rad and $2.0\times10^{-4}$ m/s for the hand-tuned ones (7.5 output).

### 7.2 Multi-rate timing and latency (`simulator.py:72-96`)

**Rates.** The physics step is $h = 2.5$ ms, the control period is $T = 5$ ms (2 steps), and the latency is $L = 10$ ms (4 steps). A command computed from the measurement at $t_k$ takes effect at the first physics step at or after $t_k + L$. It is held until the next command arrives, which is a zero-order hold.

**Effective delay.** In general the delay is $\lceil L/h\rceil h$; here that is exact. The hold adds roughly another $T/2$, so the equivalent delay is about 12.5 ms. That is small next to the 76 ms unstable time constant ($p\cdot12.5\,\text{ms} = 0.16$), but not negligible (7.4).

**Unchecked condition.** The code does not check that $T$ and $L$ are multiples of $h$. If $T$ is not, the real control period becomes `round(T/h)·h` while the integrators still use $T$ (10.4).

### 7.3 Fall detection (`planar2d.py:129`)

A run ends when $\lvert\theta\rvert > 60°$. This threshold only decides when bookkeeping stops; it does not change the dynamics.

For comparison, setting $\dot\theta = \ddot\theta = 0$ in (E1)–(E2) gives the torque needed to hold an angle:

```math
\tau_{\text{hold}}(\theta) = \frac{aMgl\sin\theta}{a + Ml\cos\theta/r}.
```

The 0.40 N m limit can hold the body up to 69.6°. But holding requires the wheels to keep accelerating, and the motor envelope (4.4) collapses with speed. So the region the robot can actually recover from is far smaller, and it depends on $\dot\theta$ and $v$ as well.

### 7.4 Delay margins: an exact discrete-time analysis

Approximating the delay with Padé or a continuous-time phase margin is not exact here. Instead, the sampled, delayed linear loop can be written exactly.

**The discrete model.** Take the plant state $s = (v, \theta, \dot\theta)$; position is left out because nothing depends on it. Write the latency as $L = qT + \varepsilon$ with $0 \le \varepsilon < T$. During $[t_k, t_{k+1})$ the motor sees $u_{k-q-1}$ for the first $\varepsilon$ seconds and $u_{k-q}$ for the rest. Integrating the linear plant exactly over the period gives

```math
s_{k+1} = \Phi s_k + \Gamma_1 u_{k-q} + \Gamma_2 u_{k-q-1},\qquad
\Phi = e^{A_vT},\quad \Gamma_1 = \int_0^{T-\varepsilon}\! e^{A_v\sigma}d\sigma\,B_v,\quad
\Gamma_2 = e^{A_v(T-\varepsilon)}\!\int_0^{\varepsilon}\! e^{A_v\sigma}d\sigma\,B_v .
```

The controller of 6.1–6.2 (unsaturated) is

```math
u_k = (k_p + k_iT)\,e_k + k_iI_{k-1} + k_d\dot\theta_k,\qquad e_k = \theta_k + k_v v_k .
```

Stacking $z_k = (s_k, I_{k-1}, u_{k-1}, \dots, u_{k-q-1})$ gives $z_{k+1} = \mathcal A z_k$. The loop is stable exactly when the spectral radius of $\mathcal A$ is below 1. For $L = 10$ ms, $q = 2$ and $\varepsilon = 0$.

The analysis is exact for the linear model. It ignores saturation and noise. It also covers the DC motor below its limits, since there the motor delivers exactly the command (4.3). The implementation is `_discrete_loop` in `physics_checks.py`.

**Results (7.4 output):**

| Controller | Unstable at latency ≥ | Least-damped mode at the configured 10 ms | Nonlinear simulator (ideal actuator, 0.5 ms steps) |
|---|---|---|---|
| Hand-tuned, pitch-only (4, 0.5, 0.15) | 9.5 ms | 17.5 Hz, damping ratio −0.037 (unstable) | decays at 9.0 ms; 0.025 rad limit cycle at 9.5 ms; 0.2 rad at 20 ms; falls at 30 ms |
| Hand-tuned cascade | 11.5 ms | 13.5 Hz, damping ratio 0.071 | n/a |
| Tuned cascade | 36.0 ms | 1.7 Hz, damping ratio 0.293 | decays at 30 and 35 ms; 0.053 rad at 36 ms; falls at 37 ms |

What the table shows:

- **The analysis predicts the nonlinear simulator's stability boundary to within 1 ms.**
- **It explains `CONDITIONS.md`'s empirical latency findings** for the pitch-only hand-tuned PID: fine at 5 ms, oscillating at 10–20 ms, falling from 30 ms. The linear loop goes unstable at 9.5 ms. Torque saturation then bounds the growing oscillation into a limit cycle, until around 30 ms the limit cycle gets big enough to fall.
- **At the configured 10 ms, the hand-tuned cascade is at 87% of its delay margin** and rings at 13.5 Hz with a damping ratio of only 0.07. The tuned gains have a 3.6× margin and a damping ratio of 0.29 at 10 ms. The tuner is not given delay margins directly. It gets them because the simulation includes the delay, and since 2026-11-06 robust scoring also randomizes latency over 5–20 ms (8.4), well inside the 36 ms margin. This is direct evidence that modeling latency was worth it.

**Delay margin as a function of $k_v$** (same output), which links back to 6.3:

| Inner loop | $k_v$ = 0.1 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Hand-tuned | 11.5 ms | 8.25 ms | 3.25 ms | unstable | unstable | unstable | unstable | unstable | unstable | unstable |
| Tuned | 17.75 ms | 19.25 ms | 20.0 ms | 21.0 ms | 23.25 ms | 26.0 ms | 29.25 ms | 32.75 ms | 36.0 ms | 33.75 ms |

For the tuned inner loop the margin rises with $k_v$ up to 0.8 and falls beyond it. The tuned $k_v = 0.796$ sits at the peak of its curve (36 ms at 0.8), not on a flat part.

### 7.5 Accuracy of the cost integrals

The cost (Section 8) is accumulated as a left Riemann sum, $\sum_k f(t_k)\,h$ (`metrics.py:96-101`; `batch_sim/simulate.py:136-141`). On each smooth stretch $[t_a, t_b]$ of the integrand, the Euler–Maclaurin formula gives the error:

```math
\sum_k f(t_k)\,h - \int_{t_a}^{t_b} f\,dt = \frac{h}{2}\bigl(f(t_a) - f(t_b)\bigr) + O(h^2).
```

With the earlier step goals, the velocity term jumped at every goal step. When the robot had reached the previous goal, each jump started a stretch at $f(t_a) \approx \Delta v_{\text{goal}}^2$ that decayed toward 0. So the bias was about

```math
\frac{h}{2}\sum\Delta v_{\text{goal}}^2 = 0.00945 ,
```

and it was observed at +0.42% (tuned) and +0.18% (hand-tuned) at 2.5 ms. The smooth goal removes those jumps (`physics_checks.py` no longer prints this prediction), so the bias is now small (7.5 output, seed 100, 5.73° start):

| | Tuned | Hand-tuned |
|---|---|---|
| Left-sum cost at 2.5 ms against 0.25 ms | +0.04% | +0.03% |
| Trapezoid rule, same comparison | +0.03% | +0.03% |

The trapezoid variant takes the smooth goal at both ends of each step. Holding it over the step, which was right for step goals, adds a first-order error of its own: the earlier version of this check did that and reported −0.60% / −0.11%. The two rules now agree to 0.01%. The remaining +0.03–0.04% is common to both. The state trajectories agree to $10^{-9}$ (tuned) and $10^{-4}$ (hand-tuned), so it is not the integrator either; its source is not pinned down. Either way, with the smooth goal the quadrature bias is about 10 times smaller than with step goals, and the trapezoid rule no longer helps (10.4).

## 8. Cost function and robust scoring (`metrics.py`, `robust.py`)

### 8.1 The cost (`metrics.py:76-105`)

```math
J = \int_0^{t_{\text{end}}}\!\Bigl[(\theta - \theta_{\text{ref}})^2 + w_u\,\tau^2 + w_v\,(\dot x - v_{\text{goal}})^2\Bigr]dt
\;+\; \mathbb 1_{\text{fell}}\;P\Bigl(1 + \frac{t_{\text{dur}} - t_{\text{fall}}}{t_{\text{dur}}}\Bigr).
```

Since 2026-10-24 the parameters are $w_u = 9$, $w_v = 2.25$, $P = 500$, and $\theta_{\text{ref}} = 0$ (upright). The weights come from Bryson's rule (10.5) and are derived in `config.py` from three limits. Costs quoted in 8.2 to 9.4 use these values and the smooth goal, except the record table in 8.2, which used the earlier values: $w_u = 0.01$, $w_v = 1$, $P = 100$, $\theta_{\text{ref}}$ = the commanded lean, and a step goal.

- **Units.** The weights carry units, which makes each term comparable to rad² s: $w_u$ is in rad²/(N m)² and $w_v$ in rad²/(m/s)².
- **Why the reference changed (2026-10-24).** Measuring pitch against the commanded lean avoids penalizing the lean the task needs (2.8), but only a controller with a lean command has one. The LQR commands none, so it was scored from upright anyway, and the PID-vs-LQR comparison had to override the setting. Scoring everyone from upright gives one number for all controllers. The price is a floor: the start tilt must be recovered and the goal needs about 0.12 rad of lean, so $\int\theta^2\,dt \approx 0.027$ rad² s is unavoidable and the cost cannot reach zero. The limit $\theta_{\max} = 0.3$ rad is about 2.4 times the lean the goal needs, so it does not make that lean an error.
- **Effort.** $\tau$ here is the command after latency. While the motor is unsaturated, $\tau^2$ is proportional to winding loss (4.6).

### 8.2 What the cost actually measures

**Current cost (2026-11-06).** Mean over the 30 validation runs, nominal robot (8 output). "Tuned 2026-11-06" is `physics_checks.py`'s `TUNED_NOW`: the CMA-ES result in `results/batch_tuning/20261106-154750/` ($k_p = 0.660$, $k_i = 0.498$, $k_d = 0.112$, $k_v = 0.796$), tuned for this cost with friction and latency randomized. "Tuned 2026-09-30" is the set this document quotes elsewhere, which was tuned for the earlier cost.

| Gains | Pitch term | Effort term | Velocity term | Total |
|---|---|---|---|---|
| Tuned 2026-11-06 | 0.031 (22%) | 0.034 (25%) | 0.075 (53%) | 0.140 |
| Tuned 2026-09-30 | 0.023 (5%) | 0.016 (4%) | 0.391 (91%) | 0.430 |
| Hand-tuned | 0.009 (0.4%) | 0.312 (15%) | 1.809 (85%) | 2.130 |

So:

- **All three terms now matter for a controller tuned for this cost.** Velocity tracking is about half of it, and pitch and effort a quarter each. The pitch term includes the floor of about 0.027 that scoring from upright imposes (8.1), so the tuned gains are close to it.
- **Gains tuned for the earlier cost are poor at this one** (0.430, 3x the current tuned). Their outer loop is slow ($k_v = 0.25$, against 0.80), and the velocity term is what the new weights punish.
- **The hand-tuned gains pay most for effort and velocity.** Their fast inner loop spends 9 times the effort of the tuned gains, and their slow outer loop ($k_v = 0.1$) tracks poorly.

**Earlier cost and step-shaped goal (kept as a record).** $w_u = 0.01$, $w_v = 1$, pitch from the commanded lean. `physics_checks.py` no longer reproduces these numbers.

| Gains | Pitch term | Effort term | Velocity term | Total |
|---|---|---|---|---|
| Tuned 2026-09-30 | 0.087 (3.8%) | 0.0003 (0.01%) | 2.232 (96.2%) | 2.319 |
| Hand-tuned | 0.006 (0.1%) | 0.0007 (0.02%) | 4.505 (99.9%) | 4.512 |

Under that cost:

- **The cost is almost entirely velocity tracking.** The hand-tuned gains score twice as high mainly because their outer loop is 2.5 times slower (6.2), not because of their poor delay margin. The delay margin is a robustness problem, not the cost driver.
- **The effort weight is inactive.** At 0.01, the effort term is 0.01% of the cost and cannot influence the optimum (10.5).
- **Much of the velocity cost is set by physics and the scenario, not by the controller.** Suppose each goal step $\Delta v$ were met at the steady acceleration the lean cap allows, $a_{\text{cap}} = 3.14$ m/s² (2.8). The unavoidable tracking error would then be $\int_0^{\Delta v/a_{\text{cap}}}(\Delta v - a_{\text{cap}}t)^2\,dt = \Delta v^3/(3a_{\text{cap}})$. Summed over the four steps this is 1.36, against the tuned 2.23. The rest comes from lean-up time and the wrong-way start (3.3). This is an estimate, not a strict bound: transient leans can exceed the cap (in that step-goal scenario the tuned runs reached 36.5°; with the smooth goal they reach only 7.8°).

### 8.3 Fall penalty

A fall adds between $P$ and $2P$, more the earlier it happens ($P = 500$ since 2026-10-24, 100 before). That gives the optimizer a slope to climb out of falling gains. "A fall is always worse than any non-fall" holds whenever a non-fall run costs under $P$. It is not guaranteed by construction:

- the pitch term (from upright, 12 s runs) is bounded, by $(60°)^2 \times 12\,\text{s} = (1.05\ \text{rad})^2 \times 12\,\text{s} = 13.2$;
- the effort term is bounded, by $w_u\,\tau_{\max}^2 \times 12\,\text{s} = 9 \times 0.4^2 \times 12 = 17.3$;
- the velocity term has no a priori bound (the no-load speed is 2.59 m/s).

In practice, tuned controllers cost 0.15–0.4. But 45,000 random-search runs (1,500 random gain sets on 30 randomized robots) included a slow, wandering controller that never fell and cost up to 186 at the current weights (velocity term 174), where the same gains cost 80 at the old ones. With $P = 100$ that exceeds the cheapest fall (110), which is why $P$ was raised to 500. **If a cost limit or the scenario changes, re-check this headroom.**

### 8.4 Robust scoring (`robust.py:100-126`)

**Randomization.** Each scenario scales a parameter $p$ by $1 + s\,U(-1, 1)$, with spreads $s$ from `RobustnessSettings`. Three physical consequences:

- **Zero stays zero.** Multiplicative noise cannot perturb a parameter whose nominal value is zero. Friction is 0 nominally, so until 2026-11-06 robust tuning never saw it, and friction turns out to matter (9.4). Since then `RobustnessSettings.randomize_range` draws such parameters outright per scenario: viscous $b \in [0, 0.001]$ N m s/rad and Coulomb $\tau_c \in [0, 0.02]$ N m (guesses; see 9.4). `latency_range_s` draws each scenario's latency from the whole physics steps in 5–20 ms. The controllers are still designed for the nominal 10 ms.
- **Correlations are lost.** Masses and inertias are drawn independently, but for a fixed shape $I_b \propto M$. Independent draws explore some unphysical combinations (harmless at ±15–25%) and under-sample the correlated extremes.
- **The voltage spread reaches the voltage limit.** At the low-battery, high-resistance corner (6.66 V, 3.45 Ω), stall current falls to $V/R = 1.93$ A < 2 A. So the randomization does reach voltage-limited stall torque (0.386 N m), as intended.

**Chaos penalty.** This is $\lvert J(\theta_0 + \delta) - J(\theta_0)\rvert$ with $\delta = 10^{-4}$ rad. Where $J$ depends smoothly on $\theta_0$, it is a finite-difference sensitivity $\approx\lvert\partial J/\partial\theta_0\rvert\,\delta$. Large values flag gains whose runs depend sensitively on the initial condition, typically through repeated saturation switching.

## 9. Checking the assumptions against simulated runs

The derivations rest on assumptions. Each one below is checked against the 30 validation runs (10 noise seeds × 3 start tilts; 9.1–9.5 output).

### 9.1 Rolling without slip

The wheels roll without slipping only if $\lvert F\rvert \le \mu_s N$, with $F$ and $N$ from 2.6.

**Upright and at rest**, at any torque:

```math
\frac{\lvert F\rvert}{N} = \frac{\tau\,\bigl[1 - I_w(d/r + Ml)/(r\Delta_0)\bigr]}{r\,(M + m_w)\,g} = 2.71\ \text{per N m}.
```

So **traction, not the motor, sets the torque limit** whenever $\mu_s < 2.71\times0.40 = 1.08$. For $\mu_s = 0.8$ the usable torque is 0.295 N m.

**In the validation runs:**

| Gains | Max $\lvert F\rvert/N$ | Time above 0.6 | Time above 0.8 |
|---|---|---|---|
| Tuned | 0.28 | 0.00% | 0.00% |
| Hand-tuned | 1.18 | 0.49% | 0.22% |

Rubber on hard floors is typically around $\mu \approx 0.6$–1.0; it should be measured. The tuned gains stay well below that with the smooth goal. The hand-tuned gains would slip briefly. (With the earlier step goals the peaks were 0.91 and 1.43.) Peak axle accelerations reach 8.4 m/s² (tuned) and 36.9 m/s² (hand-tuned).

### 9.2 Wheels stay on the ground

The normal force never drops below 8.56 N (tuned) or 6.27 N (hand-tuned), against a static value of 8.83 N, so the wheels never lift. ✓

### 9.3 Motor driver: current mode versus voltage mode

Section 4.3 showed the model is a drive that delivers the commanded current at any speed. That requires either a current-sensing driver, or firmware that adds back-EMF compensation.

A common hobby setup instead sets the PWM duty in proportion to the torque command: $V = R\tau_c/(nNk_t)$, clipped to $\pm V_s$. In that **voltage mode**, below the limits,

```math
\tau = nNk_t\,\frac{V - k_tN\omega}{R} = \tau_c - c_m\,\omega,\qquad c_m = \frac{nN^2k_t^2}{R} = 0.0067\ \text{N m s/rad},
```

which is exactly extra viscous friction. Simulating it (`VoltageModeActuator` in `physics_checks.py`; 9.3 output):

| Gains | Current mode (as simulated) | Voltage mode |
|---|---|---|
| Tuned (2026-11-06) | 0 of 30 fall, cost 0.140 | 0 of 30 fall, cost 1.246 |
| Hand-tuned | 0 of 30 fall, cost 2.129 | 0 of 30 fall, cost 3.799 |

Section 9.4 derives the threshold. A drag of $c_m$ is above the 2026-09-30 gains' runaway threshold $rk_pk_v = 0.0064$ N m s/rad, so with those gains no steady speed exists. They fall in 30 of 30 runs, under the earlier cost and the current one alike. The current tuned gains' threshold is 0.018, 2.7 times $c_m$, and the hand-tuned threshold is 0.014, twice $c_m$. So both survive, but the extra drag costs the current tuned gains about 9 times as much (the hand-tuned gains 1.8 times).

**This is still the largest unmodeled effect on the tuned gains.** If the hardware uses a voltage-mode driver, the firmware should compensate back-EMF so that the hardware matches the simulation:

```math
V = \frac{R\,\tau_c}{nNk_t} + k_tN\hat\omega .
```

A motor-shaft encoder measures $\hat\omega$ directly, because it reads the rotor angle relative to the stator. The alternative is to simulate the real driver and retune (10.1).

### 9.4 Friction

Nominal friction is 0. Since 2026-11-06 robust scoring draws it per scenario (8.4): viscous $b \in [0, 0.001]$ N m s/rad and Coulomb $\tau_c \in [0, 0.02]$ N m. Sensitivity to viscous friction, for one run (seed 100, 5.73° start; 9.4 output), with the current cost and smooth goal:

| $b$ (N m s/rad) | 0 | 0.001 | 0.002 | 0.003 | 0.005 |
|---|---|---|---|---|---|
| Tuned 2026-11-06 (with friction in the scoring) | 0.15 | 0.15 | 0.20 | 0.29 | 0.67 |
| Tuned 2026-09-30 | 0.44 | 0.50 | 0.79 | 1.58 | 8.21 |
| Hand-tuned | 2.14 | 2.15 | 2.25 | 2.41 | 2.99 |

With the earlier cost and step goal, the 2026-09-30 gains fell at $b \ge 0.003$ (2.32, 2.63, 3.50, then falls at 4.56 s and 3.41 s). The smooth goal keeps them up here, but their cost still grows 19-fold by $b = 0.005$. The gains tuned with friction in the scoring grow 4.5-fold. They raised $k_pk_v$ from 0.18 to 0.53 N m per m/s and $k_i$ from 0.23 to 0.50, which is the trade this section predicts below.

**Why the tuned gains are the fragile ones: load torque shifts the speed.** Any steady load torque $\tau_L$ across the wheel–body joint has to be supplied by the motor. Friction, a slope and back-EMF drag are all examples. Until the slow integrator takes the load over, the proportional term must supply it, which needs a pitch error $\theta - \theta_{\text{sp}} = \tau_L/k_p$. At constant speed $\theta = 0$ (2.8), so the lean command must be $\theta_{\text{sp}} = -\tau_L/k_p$. The outer loop only commands that lean when

```math
v - v_{\text{goal}} = \frac{\tau_L}{k_p\,k_v}.
```

This is the same mechanism as the gyro-bias offset in 5.3. Counterintuitively, the robot runs faster than its goal: a backward lean command is the only way the proportional term can produce the forward torque that the load needs.

Viscous friction is a load that grows with speed, $\tau_L = b\,v/r$ (taking $\dot\theta \approx 0$). Substituting gives the quasi-steady speed

```math
v = \frac{v_{\text{goal}}}{1 - b/(r\,k_p k_v)},
```

and no steady speed at all once $b \ge r\,k_pk_v$. That threshold is 0.0064 N m s/rad for the 2026-09-30 tuned gains and 0.014 for the hand-tuned ones. The 2026-11-06 gains, tuned with friction randomized, raised it to 0.018. The tuner's small $k_p$ cut $k_pk_v$ from 0.40 to 0.18 N m per m/s, and the robot's resistance to load torque with it.

*Check (9.4 output):* with $b = 0.003$, $k_i = 0$, the ideal actuator and a constant 1.4 m/s goal, the formula predicts 1.673 m/s (tuned 2026-11-06), 2.618 m/s (tuned 2026-09-30) and 1.782 m/s (hand-tuned). The simulator settles at exactly those values.

In the real scenario the integrator does pull the speed back, but slowly. For the 2026-09-30 gains the pitch loop's integral pole was at 0.49 rad/s, about 2 s (3.5), and meanwhile the speed headed for 2.62 m/s. The 2026-11-06 gains' slowest pitch-loop poles are faster, at $-1.99 \pm 0.85j$ rad/s (3.5). With $b = 0.003$, the DC motor's forward headroom $\tau^+_{\max}(\omega) - b\omega$ (4.4) reaches zero at 51 rad/s (1.79 m/s); without friction that would only happen at the 2.59 m/s no-load speed. The 2026-09-30 robot passed 1.79 m/s in the earlier step-goal scenario (its run peaked at 1.87 m/s); the current tuned runs peak at 0.86 m/s. Beyond that the motor cannot drive the wheels forward to catch the forward lean, so the robot falls forward.

**Measuring friction.** Run the wheels unloaded at a few PWM levels and record the steady speed $\omega_0$ and the current $i_0$ per motor. The friction torque at the output is then $\tau_f(\omega_0) = nNk_t\,i_0$. Fitting $\tau_f = \tau_c + b\,\omega$ separates Coulomb from viscous friction.

### 9.5 The IMU

The model hands the controller pitch plus white noise. A real robot estimates pitch from a gyro and an accelerometer. An accelerometer measures specific force, so during horizontal acceleration its tilt reading is off by about $\arctan(\ddot x/g)$. At the 8.4 m/s² peaks of the tuned runs (9.1), that is up to 41° (75° for the hand-tuned gains, whose peaks reach 36.9 m/s²). The firmware must therefore rely on the gyro at high frequency, through a complementary or Kalman filter. Two consequences:

- **Pitch errors are low-frequency and correlated**, not white.
- **Gyro bias becomes a pitch-estimate offset** of about $b_g\tau_f$ ($\tau_f$ = filter time constant), not just the rate offset modeled in 5.3.

Mounting the IMU above the axle adds $l_{\text{imu}}\ddot\theta$ and $l_{\text{imu}}\dot\theta^2$ terms on top of this.

### 9.6 Assumptions that are always on

| Assumption | Holds when | What breaks if not |
|---|---|---|
| Rigid body, planar motion, flat level ground | chassis stiff; driving straight | yaw, slopes (add $Mg\sin\alpha$ terms) and bumps are absent |
| Both wheels share torque equally | identical motors and floor contact | any mismatch turns the robot |
| Quasi-static motor current ($L$ ignored) | $L/R \ll 5$ ms | current lags the command; extra phase lag in the loop |
| Lossless, stiff gearbox, no backlash | efficiency near 1, little play | backlash is a dead zone around zero torque, exactly where balancing operates |
| Rotor inertia ignored | $nN^2J_r$ small next to $I_w$ | see 10.3: it enters the mass matrix in three places |
| Coulomb friction smoothed, $\tau_c\tanh(\omega/\omega_\epsilon)$ with $\omega_\epsilon = 1$ rad/s; no rolling resistance | the wheels rarely sit near zero speed for long | real stiction is a dead zone, like backlash; the smoothing understates it near $\omega = 0$ |
| Sensing and actuation aligned to physics steps | latency and control period are multiples of $h$ | see 7.2 |

## 10. Limitations and future iterations

Grouped by how strong the evidence is that each one matters.

### 10.1 Evidence that the tuned gains fail: fix first

| Limitation | Evidence | Next iteration |
|---|---|---|
| Driver assumed to be current-controlled with exact back-EMF compensation | Voltage-mode driver: the 2026-09-30 gains fell 30 of 30; the current tuned gains survive at about 9 times the cost (9.3) | Decide the hardware driver. Either add the back-EMF feedforward of 9.3 to the firmware, or move the `VoltageModeActuator` from `physics_checks.py` into `actuators.py` and retune (or randomize between the two). |
| Friction is zero and cannot be randomized | Tuned gains fall at $b \ge 0.003$; the runaway threshold is $rk_pk_v$ (9.4) | *Implemented and retuned 2026-11-06* (`results/batch_tuning/20261106-154750/`; 8.2, 9.4). Both engines model $\tau_f = b\omega + \tau_c\tanh(\omega/\omega_\epsilon)$ (`RobotConfig.wheel_friction`, `coulomb_friction`, `coulomb_friction_speed`). `linearize` includes the slope $\tau_c/\omega_\epsilon$, and robust scoring draws $b$ and $\tau_c$ per scenario (8.4). With $\omega_\epsilon = 1$ rad/s, $\lvert\lambda h\rvert \approx 0.17$ at the largest $\tau_c$ (7.1). Still to do: measure friction (9.4) and replace the guessed ranges. |
| Latency is fixed at 10 ms | Delay margin is the key robustness number (7.4) | *Implemented and retuned 2026-11-06.* The retuned gains' nominal linear loop is stable from 0 to 35.75 ms (`results/lqr_comparison/20261106-154350/`). Robust scoring gives each scenario its own latency, uniform over the whole physics steps in 5–20 ms (`RobustnessSettings.latency_range_s`; `simulate_batch(delay_steps=...)`, `TorchSimulator(max_delay_steps=...)`). The alternative, adding the delay margin to the objective, is not done. |

### 10.2 Assumptions violated during transients

| Limitation | Evidence | Next iteration |
|---|---|---|
| No traction limit | $\lvert F\rvert/N$ up to 0.28 / 1.18 (9.1) | Simplest: add a penalty on time spent above a measured $\mu$. Full model: make $\varphi$ a fifth coordinate. While sticking, enforce $\dot x = r\dot\varphi$. When the required $\lvert F\rvert$ exceeds $\mu_sN$, switch to sliding with $F = -\mu_kN\operatorname{sign}(\dot x - r\dot\varphi)$. |
| Idealized IMU | Accelerometer tilt error up to 41° / 75° (9.5) | Simulate the accelerometer (from $\ddot x$, $\ddot\theta$ and the mount height) and the gyro, and run the same complementary filter the firmware will use. |
| Idealized encoders | Velocity is truth plus white noise | Quantize the wheel angle (counts per revolution) and difference it at the control rate, as the firmware will. |

### 10.3 Physical fidelity

- **Reflected rotor inertia.** The `DCMotorActuator` docstring suggests folding this into `wheel_inertia`, but that is only partly right. Each rotor spins at about $N\omega = N(\dot x/r - \dot\theta)$ relative to the body. So it adds $\tfrac12 J\,(\dot x/r - \dot\theta)^2$ to $T$, where $J = nN^2J_r$. The mass matrix becomes

  ```math
  \mathbf M(\theta) = \begin{pmatrix} a + J/r^2 & Ml\cos\theta - J/r \\ Ml\cos\theta - J/r & d + J \end{pmatrix}.
  ```

  Folding $J$ into $I_w$ gets the first entry right but misses the $-J/r$ coupling and the $+J$ on the body. The matrix stays positive definite, because $T$ is still a sum of squares. Only `planar2d.py:112-114` (and `batch_sim/physics.py:56-58`) would change.
- **Winding inductance.** If $L/R$ is not much less than 5 ms, add the current as a state: $L\,di/dt = V - Ri - k_eN\omega$.
- **Gearbox efficiency and backlash.** Multiply the output torque by an efficiency $\eta$. Model backlash as a dead band in $\varphi - \theta$.
- **Battery sag.** Use $V_s(t) = V_{oc} - R_{\text{batt}}\,i_{\text{total}}$ instead of a constant per run (`CONDITIONS.md` already lists this).
- **Noise as a density.** Specify sensor noise as a density (rad/s/√Hz, as datasheets do) and convert it with $\sigma = \text{ND}\sqrt{f_s/2}$. Today the per-sample $\sigma$ silently changes the physical noise level if the control rate changes.
- **Scope.** A 3D model with yaw (differential drive), slopes, and pushes on the body, as already planned in `ARCHITECTURE.md` and `CONDITIONS.md`.

### 10.4 Numerics and code

- **Cost quadrature.** With the earlier step goals the left-sum bias was 0.42%, and the trapezoid rule (holding the references over each step) cut it to 0.004%. With the smooth goal the left-sum bias is already 0.04% and the trapezoid variant checked in 7.5 shows no gain, so this is now low priority. Adding the running cost as a fifth state integrated by RK4 would still make it 4th-order accurate for free.
- **Timing validation.** Check that `control_dt` and `latency` are integer multiples of `dt` (7.2), and raise an error otherwise. Today a mismatch silently changes the control period or rounds up the latency.

### 10.5 Cost design

- **Choose the weights deliberately.** *Done 2026-10-24.* The effort term was 0.01% of the cost (8.2). Bryson's rule (Franklin, Powell & Emami-Naeini) sets each weight to $1/(\text{largest acceptable value})^2$; normalized to the pitch term, $w_v = (\theta_{\max}/v_{\max})^2$ and $w_u = (\theta_{\max}/\tau_{\max})^2$. The limits are $\theta_{\max} = 0.3$ rad (from upright), $v_{\max} = 0.2$ m/s and $\tau_{\max} = 0.1$ N m, giving $w_v = 2.25$ and $w_u = 9$. They are stored as `CostConfig`'s `pitch_limit_rad`, `velocity_limit` and `torque_limit`, and the weights are derived from them. $\tau_{\max}$ is a typical operating torque, not the 0.4 N m hardware limit, which would leave effort negligible. $\theta_{\max}$ is a tolerance on the lean error: it must sit above the lean the goal needs (about 0.12 rad here) and below the 25° lean cap. Result: pitch, effort and velocity are about 20%, 12% and 69% of the cost of the PID tuned on 2026-10-24, and 22%, 25% and 53% for the PID retuned with friction and latency randomized (8.2). The `kv_p` bound was 0.4 and every retune sat on it; it is now 0.9. A run with the bound at 2.0 settled at `kv_p` 0.82 with the same cost (0.1304 against 0.1305), and the re-run at 0.9 settled at 0.822, inside the bound.
- **Use feasible goal profiles.** *Done:* `goal_transition_s` blends each goal change smoothly, peaking at 0.9 m/s² (about 7° of lean). Step-shaped goals made much of the velocity cost unavoidable (8.2).
- **An LQR baseline.** The cost has LQR form, and 3.1 gives $A$ and $B$. Optimal full-state feedback on the linear model, with the same weights, would show how much performance the cascaded-PID structure costs. *Done: see `LQR.md`. The cost-matched LQR comes within 1.8% of the ML-tuned PID with no tuning; the PID still wins on every validation scenario.*
- **A physical fall criterion.** Replace the 60° threshold, or add to it, a check based on recoverability (7.3).

## 11. The batched engine uses the same physics

`batch_sim` implements the same equations in the same order of operations, as functions over arrays of robots:

| Physics | Reference (`balance_sim`) | Batched (`batch_sim`) |
|---|---|---|
| Equations of motion (Section 2) | `planar2d.py:97-123` | `physics.py:41-66` |
| DC motor and ideal actuator (Section 4) | `actuators.py:38, 99-112` | `physics.py:26-38` |
| RK4 (7.1) | `simulator.py:26-31` | `physics.py:73-80` |
| Anti-windup integrator (6.4) | `controllers.py:14-51` | `controller.py:29-52` |
| Cascaded controller (6.1–6.2) | `controllers.py:187-198, 279-289` | `controller.py:55-88` |
| Latency and hold (7.2) | `simulator.py:85-96` (deque) | `schedule.py` (precomputed ring buffer) |
| Cost, left Riemann sums (7.5, 8.1) | `metrics.py:96-105` | `simulate.py:136-141, 157-160` |

In float64 the NumPy backend reproduces `balance_sim` bit for bit (`simulation/tests/test_equivalence.py`; `BATCHED.md`). So every derivation, limitation and fix above applies to both engines. A physics change made in `planar2d.py` or `actuators.py` must be mirrored in `batch_sim/physics.py`, and the equivalence tests will catch a missed one.

## References

- F. Grasser, A. D'Arrigo, S. Colombi, A. C. Rufer, "JOE: A mobile, inverted pendulum," *IEEE Transactions on Industrial Electronics* 49(1), 107–114, 2002. The classic derivation of a two-wheeled inverted pendulum.
- D. T. Greenwood, *Principles of Dynamics*, 2nd ed., Prentice Hall, 1988. Lagrange's equations with generalized forces and dissipation; the energy theorem used in 2.7.
- K. J. Åström, T. Hägglund, *Advanced PID Control*, ISA, 2006. Derivative on measurement, anti-windup by conditional integration.
- S. Skogestad, I. Postlethwaite, *Multivariable Feedback Control: Analysis and Design*, 2nd ed., Wiley, 2005. Limits on bandwidth from RHP zeros, RHP poles and delay.
- G. F. Franklin, J. D. Powell, A. Emami-Naeini, *Feedback Control of Dynamic Systems*, 8th ed., Pearson, 2019. Routh–Hurwitz, root locus, Bryson's rule.
- E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed., Springer, 1993. Order and stability region of RK4.
