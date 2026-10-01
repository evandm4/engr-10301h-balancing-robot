# The physics in the simulation: derivations, justification, limitations

This document derives every equation in the simulator from first principles. For each one it shows where the equation lives in the code, checks it numerically, and lists what the model leaves out and what to do about it. It covers `simulation/balance_sim/`, the reference simulator. `simulation/batch_sim/` runs the same equations in the same order (Section 11), so everything here applies to it too.

All numbers use the robot defined in `config.py`. Two sets of gains appear throughout:

- **Tuned:** the CMA-ES result in `results/batch_tuning/20260930-103458/`: $k_p = 0.730$, $k_i = 0.230$, $k_d = 0.0572$, $k_v = 0.252$.
- **Hand-tuned:** `ControllerConfig.hand_tuned_gains`: $k_p = 4.0$, $k_i = 0.5$, $k_d = 0.15$, $k_v = 0.1$.

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

- **The equations of motion are correct.** Two independent derivations (Lagrange and Newton–Euler) agree, and both match `Planar2D.derivatives` term for term. The mass matrix is invertible at every pitch angle (proof in 2.5). Energy is conserved without torque and friction, and otherwise changes by exactly the work the motor and friction do (to $3\times10^{-7}$ J).
- **`critical_pitch_gain` is exact:** on the linearized robot, below it no choice of the other gains can hold the robot up (3.4). The integrator (RK4) shows its expected 4th-order convergence (7.1).

**What the derivations explain:**

- **The robot has an unstable pole at 13.1 rad/s and a right-half-plane zero at 7.46 rad/s** (3.2, 3.3). To speed up it must first roll backward, and the simulator does exactly that.
- **New closed-form result (6.3):** the cascaded controller's characteristic polynomial gives a ceiling on the outer-loop gain: $k_v < (c/e)\,(k_d/k_p)$. This explains why outer gains above about 0.25 oscillate with the hand-tuned inner loop. It also explains why the tuner lowered $k_p$: doing so raised the ceiling on $k_v$, and velocity tracking is 96% of the cost.
- **An exact discrete-time analysis gives each controller's delay margin** (7.4). The tuned gains tolerate 31 ms of latency; the hand-tuned ones tolerate 11.5 ms. The nonlinear simulator confirms both to within 0.5 ms.

**What could break on hardware**, in order of severity (Section 9):

1. **The motor driver.** The model assumes the driver delivers the commanded current. If the real driver instead sets PWM duty in proportion to the torque command, the tuned gains fall in 30 of 30 validation runs. Its back-EMF drag is above the gains' runaway threshold (9.3, 9.4).
2. **Friction.** It is zero in the model, and the robust tuner cannot vary it. With viscous friction of 0.003 N m s/rad or more, the tuned gains fall. The tuned gains have a small $k_pk_v$, and any load torque then pushes the speed up by $\tau_L/(k_pk_v)$ (9.4).
3. **Traction.** The no-slip assumption needs a friction coefficient up to 0.91 (tuned) or 1.43 (hand-tuned) during transients.
4. **The IMU model.** It is idealized: a raw accelerometer tilt estimate would be off by up to 72° during these transients.

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

This is `relative_wheel_speed` (`planar2d.py:69`).

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

These are the $T$ and $V$ in the `planar2d.py` docstring and in `Planar2D.energy` (`planar2d.py:99`).

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

`planar2d.py:70-71` computes $\tau_{\text{net}}$ the same way: it gets the delivered torque from the actuator (Section 4), then subtracts friction. Friction modeled this way is equivalent to a Rayleigh dissipation function $\mathcal F = \tfrac12 b\omega^2$. Because it acts between two parts of the robot, it can only dissipate energy (2.7).

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
| $\omega = \dot x/r - \dot\theta$ | `planar2d.py:69` |
| $\tau_{\text{net}}$ = actuator output $- b\omega$ | `planar2d.py:70-71` |
| $a$, $b(\theta)$, $d$ | `planar2d.py:76-78` |
| $\text{rhs}_x = \tau_{\text{net}}/r + Ml\sin\theta\,\dot\theta^2$ | `planar2d.py:80` |
| $\text{rhs}_\theta = -\tau_{\text{net}} + Mgl\sin\theta$ | `planar2d.py:81` |
| $\Delta$, $\ddot x$, $\ddot\theta$ | `planar2d.py:83-85` |

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

*Numerical check (2.7 output):* one 10 s closed-loop run with $b = 0.001$. The energy change of 0.480033 J matches the integrated power of 0.480032 J; the difference is $3\times10^{-7}$ J.

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

The simulator shows this (3 output, tuned gains, seed 100):

- **Goal 0 → 0.8 m/s at t = 2 s:** velocity goes from −0.006 to −0.443 m/s before rising.
- **Goal 1.4 → −0.6 m/s at t = 6 s:** velocity first rises from 1.36 to 1.99 m/s.

The RHP zero is a fundamental limit on velocity control (6.2).

### 3.4 Critical pitch gain (`critical_pitch_gain`, `planar2d.py:37`)

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

The search space uses $1.1\,k_{p,\text{crit}}$ as the lower bound on $k_p$ (`search_space.py:87`). Gains below $k_{p,\text{crit}}$ fall in the linear model whatever the other gains are (`test_gain_threshold_matches_simulation`), so the tuner never searches them.

### 3.5 Full pitch-loop stability (Routh–Hurwitz)

Add damping and integral action: $\tau = k_p\theta + k_d\dot\theta + k_i\!\int\!\theta$. Then the pitch loop's characteristic polynomial is

```math
\Delta_0\,s^3 + c\,k_d\,s^2 + (c\,k_p - aMgl)\,s + c\,k_i = 0 .
```

The Routh–Hurwitz conditions for stability are $k_d > 0$, $k_i > 0$, and

```math
k_p > k_{p,\text{crit}} + \frac{\Delta_0\,k_i}{c\,k_d}.
```

So $k_{p,\text{crit}}$ is necessary but not sufficient. The bound is 0.2358 for the tuned gains and 0.2348 for the hand-tuned ones.

The continuous-time pitch-loop poles, ignoring sampling and delay, are:

| Gains | Poles (rad/s) | $k_p / k_{p,\text{crit}}$ |
|---|---|---|
| Tuned | −30.8, −11.4, −0.49 | 3.2 |
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

The scenario asks for 1.4 m/s, and tuned runs briefly reach 2.0 m/s (9.1–9.5 output). So the motor model does bind in this scenario, but only for 0.4% of the time (tuned) and 0.65% (hand-tuned).

### 4.5 Ideal actuator and friction

`IdealActuator` returns $\operatorname{clip}(\tau_{\text{cmd}}, \pm\tau_{\max})$. It is the DC motor model in the limit $V_s, i_{\max}\to\infty$. Friction is applied after either actuator (`planar2d.py:71`), as derived in 2.3.

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

This is 0.0053 N m (1.3% of $\tau_{\max}$) for the tuned gains, and 0.0217 N m (5.4%) for the hand-tuned ones. Lower $k_p$ makes the tuned controller four times quieter.

### 5.3 Gyro bias in steady state

At constant velocity, (E1) and (E2) with $\ddot x = \ddot\theta = 0$ force $\tau_{\text{net}} = 0$ and $\theta = 0$. Two cases:

**PD inner loop ($k_i = 0$).** Setting the torque to zero gives $-k_p\theta_{\text{sp}} + k_db_g = 0$. Since $\theta_{\text{sp}} = k_v(v_{\text{goal}} - v)$, there is a steady velocity error

```math
v_{\text{goal}} - v = \frac{k_d\,b_g}{k_p\,k_v},
```

which is 3.1 mm/s for the tuned gains.

**With $k_i > 0$.** The integrator stops moving only when $\theta - \theta_{\text{sp}} = 0$, which forces $\theta_{\text{sp}} = 0$ and therefore $v = v_{\text{goal}}$. The integrator settles at $-k_db_g/k_i$, well inside its limit. Integral action rejects the bias completely.

Section 9.5 covers what this sensor model leaves out.

## 6. Controller (`controllers.py`)

### 6.1 Inner pitch loop (`controllers.py:155-166`)

At each tick $k$, with period $T = 5$ ms:

```math
e_k = \hat\theta_k - \theta_{\text{sp},k},\qquad
I_k = I_{k-1} + T\,e_k,\qquad
u_k = \operatorname{clip}\bigl(k_p e_k + k_i I_k + k_d\hat{\dot\theta}_k,\ \pm\tau_{\max}\bigr).
```

- **The integrator is backward Euler**, discrete transfer function $Tz/(z-1)$, with the anti-windup of 6.4.
- **The derivative acts on the measurement:** the term is $k_d\hat{\dot\theta}$ (the gyro), not $k_d\,\dot e$. When the goal velocity steps, $\theta_{\text{sp}}$ can jump by the full 25° in one tick, and $\dot e$ would spike ("derivative kick"). Using the gyro also avoids differencing noisy angles. The cost of this choice is that it treats $\dot\theta_{\text{sp}}$ as zero.
- **Sign:** $\theta > 0$ gives $\tau > 0$, which drives the wheels under the body (3.1).

### 6.2 Outer velocity loop (`controllers.py:247-257`)

```math
\theta_{\text{sp}} = \operatorname{clip}\bigl(k_v\,(v_{\text{goal}} - \hat{\dot x}) + k_{vi}\!\textstyle\int(\cdot),\ \pm 25^\circ\bigr),\qquad k_{vi} = 0 \text{ in this project}.
```

**Sign.** If the robot is too slow, $\theta_{\text{sp}} > 0$ and the inner loop leans it forward. By 3.3 it gets there by first rolling the wheels backward.

**Bandwidth estimate.** If the inner loop held $\theta = \theta_{\text{sp}}$ exactly, 2.8 would give $\dot v \approx G\theta_{\text{sp}} = Gk_v(v_{\text{goal}} - v)$. That is a first-order lag with bandwidth $\omega_v = Gk_v$:

| Gains | $\omega_v$ | Time constant |
|---|---|---|
| Tuned | 1.75 rad/s | 0.57 s |
| Hand-tuned | 0.69 rad/s | 1.44 s |

Two things cap $\omega_v$:

1. The outer loop must stay well below the inner loop's speed.
2. The RHP zero at $z = 7.46$ rad/s: a standard rule of thumb keeps bandwidth below $z/2 = 3.7$ rad/s (Skogestad & Postlethwaite, ch. 5).

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
| Tuned (0.730, 0.230, 0.0572) | 0.0784 s | 0.630 | 0.539 |

The nonlinear simulator confirms the hand-tuned limit. With the hand-tuned inner loop and no latency:

- $k_v = 0.25$ settles to $2\times10^{-6}$ rad;
- $k_v = 0.30$ goes into a 0.37 rad limit cycle.

That is `CONDITIONS.md`'s empirical observation, "outer gains much above Kv_p of about 0.25 oscillate", now derived.

**What this says about the tuner's result.** Velocity tracking is 96% of the cost (8.2), so the tuner wants a large $k_v$. With the hand-tuned inner loop, $k_v$ is capped near 0.29. With the 10 ms latency even $k_v = 0.2$ is unstable (7.4). The tuner cut $k_p$ from 4 to 0.73 and doubled $k_d/k_p$, which raised the $k_v$ ceiling and the delay margin at the same time. This explains the direction of the change; it does not prove the optimum is unique.

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

**Closed loop.** The clips make the right-hand side non-smooth, which costs some order at the switching instants. Even so, the full scenario at 2.5 ms differs from 0.25 ms by at most $4.4\times10^{-6}$ rad in pitch and $6.7\times10^{-5}$ m/s in velocity (7.5 output).

### 7.2 Multi-rate timing and latency (`simulator.py:62-84`)

**Rates.** The physics step is $h = 2.5$ ms, the control period is $T = 5$ ms (2 steps), and the latency is $L = 10$ ms (4 steps). A command computed from the measurement at $t_k$ takes effect at the first physics step at or after $t_k + L$. It is held until the next command arrives, which is a zero-order hold.

**Effective delay.** In general the delay is $\lceil L/h\rceil h$; here that is exact. The hold adds roughly another $T/2$, so the equivalent delay is about 12.5 ms. That is small next to the 76 ms unstable time constant ($p\cdot12.5\,\text{ms} = 0.16$), but not negligible (7.4).

**Unchecked condition.** The code does not check that $T$ and $L$ are multiples of $h$. If $T$ is not, the real control period becomes `round(T/h)·h` while the integrators still use $T$ (10.4).

### 7.3 Fall detection (`planar2d.py:92`)

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
| Tuned cascade | 31.0 ms | 3.3 Hz, damping ratio 0.548 | decays at 30 ms; 0.079 rad at 31 ms; falls at 32 ms |

What the table shows:

- **The analysis predicts the nonlinear simulator's stability boundary to within 0.5 ms.**
- **It explains `CONDITIONS.md`'s empirical latency findings** for the pitch-only hand-tuned PID: fine at 5 ms, oscillating at 10–20 ms, falling from 30 ms. The linear loop goes unstable at 9.5 ms. Torque saturation then bounds the growing oscillation into a limit cycle, until around 30 ms the limit cycle gets big enough to fall.
- **At the configured 10 ms, the hand-tuned cascade is at 87% of its delay margin** and rings at 13.5 Hz with a damping ratio of only 0.07. The tuned gains have a 3.1× margin and are well damped. The tuner was never told about delay margins; it found them because the simulation includes the delay. This is direct evidence that modeling latency was worth it.

**Delay margin as a function of $k_v$** (same output), which links back to 6.3:

| Inner loop | $k_v$ = 0.1 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| Hand-tuned | 11.5 ms | 8.25 ms | 3.25 ms | unstable | unstable | unstable |
| Tuned | 31.5 ms | 32.25 ms | 31.0 ms | 28.25 ms | 17.75 ms | 3.5 ms |

The tuned $k_v = 0.252$ sits on the flat part of its curve, where the margin is near its maximum.

### 7.5 Accuracy of the cost integrals

The cost (Section 8) is accumulated as a left Riemann sum, $\sum_k f(t_k)\,h$ (`metrics.py:96-101`; `batch_sim/simulate.py:109-114`). On each smooth stretch $[t_a, t_b]$ of the integrand, the Euler–Maclaurin formula gives the error:

```math
\sum_k f(t_k)\,h - \int_{t_a}^{t_b} f\,dt = \frac{h}{2}\bigl(f(t_a) - f(t_b)\bigr) + O(h^2).
```

The velocity term jumps at every goal step. When the robot has reached the previous goal, each jump starts a stretch at $f(t_a) \approx \Delta v_{\text{goal}}^2$ that decays toward 0. So the bias is about

```math
\frac{h}{2}\sum\Delta v_{\text{goal}}^2 = 0.00945 .
```

| | Tuned | Hand-tuned |
|---|---|---|
| Bias at 2.5 ms (observed) | 0.0096 (+0.42%) | 0.0081 (+0.18%) |
| With the trapezoid rule | +0.004% | +0.011% |

The hand-tuned bias is smaller because its slow loop has not reached the previous goal when the next step arrives.

This **first-order quadrature bias is about 100 times larger than the RK4 error**. It is nearly the same for every candidate, so it shifts costs more than it reorders them. It is still the cheapest accuracy improvement available (10.4).

## 8. Cost function and robust scoring (`metrics.py`, `robust.py`)

### 8.1 The cost (`metrics.py:76-105`)

```math
J = \int_0^{t_{\text{end}}}\!\Bigl[(\theta - \theta_{\text{ref}})^2 + w_u\,\tau^2 + w_v\,(\dot x - v_{\text{goal}})^2\Bigr]dt
\;+\; \mathbb 1_{\text{fell}}\;P\Bigl(1 + \frac{t_{\text{dur}} - t_{\text{fall}}}{t_{\text{dur}}}\Bigr).
```

The parameters are $w_u = 0.01$, $w_v = 1$, $P = 100$, and $\theta_{\text{ref}}$ = the commanded lean.

- **Units.** The weights carry units, which makes each term comparable to rad² s: $w_u$ is in rad²/(N m)² and $w_v$ in rad²/(m/s)².
- **Why measure pitch against the commanded lean.** Following a velocity goal requires leaning (2.8). Measuring pitch from upright would penalize the lean the task needs.
- **Effort.** $\tau$ here is the command after latency. While the motor is unsaturated, $\tau^2$ is proportional to winding loss (4.6).

### 8.2 What the cost actually measures

Mean over the 30 validation runs (8 output):

| Gains | Pitch term | Effort term | Velocity term | Total |
|---|---|---|---|---|
| Tuned | 0.087 (3.8%) | 0.0003 (0.01%) | 2.232 (96.2%) | 2.319 |
| Hand-tuned | 0.006 (0.1%) | 0.0007 (0.02%) | 4.505 (99.9%) | 4.512 |

So:

- **The cost is almost entirely velocity tracking.** The hand-tuned gains score twice as high mainly because their outer loop is 2.5 times slower (6.2), not because of their poor delay margin. The delay margin is a robustness problem, not the cost driver.
- **The effort weight is inactive.** At 0.01, the effort term is 0.01% of the cost and cannot influence the optimum (10.5).
- **Much of the velocity cost is set by physics and the scenario, not by the controller.** Suppose each goal step $\Delta v$ were met at the steady acceleration the lean cap allows, $a_{\text{cap}} = 3.14$ m/s² (2.8). The unavoidable tracking error would then be $\int_0^{\Delta v/a_{\text{cap}}}(\Delta v - a_{\text{cap}}t)^2\,dt = \Delta v^3/(3a_{\text{cap}})$. Summed over the four steps this is 1.36, against the tuned 2.23. The rest comes from lean-up time and the wrong-way start (3.3). This is an estimate, not a strict bound: transient leans can exceed the cap (the tuned runs reach 36.5°).

### 8.3 Fall penalty

A fall adds between 100 and 200, more the earlier it happens. That gives the optimizer a slope to climb out of falling gains. "A fall is always worse than any non-fall" holds whenever a non-fall run costs under 100. It is not guaranteed by construction:

- the pitch term is bounded, by $(60° + 25°)^2 \times 10\,\text{s} = (1.48\ \text{rad})^2 \times 10\,\text{s} = 22$;
- the effort term is bounded, by 0.016;
- the velocity term has no a priori bound.

In practice, non-fall costs in this project are 2–7.

### 8.4 Robust scoring (`robust.py:76-92`)

**Randomization.** Each scenario scales a parameter $p$ by $1 + s\,U(-1, 1)$, with spreads $s$ from `RobustnessSettings`. Three physical consequences:

- **Zero stays zero.** Multiplicative noise cannot perturb a parameter whose nominal value is zero. `wheel_friction` is 0 (and not in the list), so robust tuning never sees friction, and friction turns out to matter (9.4).
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
| Tuned | 0.91 | 0.47% | 0.15% |
| Hand-tuned | 1.43 | 3.28% | 1.40% |

The peaks come right after goal steps, and for the hand-tuned gains also right after the start. Rubber on hard floors is typically around $\mu \approx 0.6$–1.0; it should be measured. So the real robot would slip briefly in those transients, and the hand-tuned gains would slip more. Peak axle accelerations reach 29 m/s² (tuned) and 37 m/s² (hand-tuned).

### 9.2 Wheels stay on the ground

The normal force never drops below 5.06 N (static value 8.83 N), so the wheels never lift. ✓

### 9.3 Motor driver: current mode versus voltage mode

Section 4.3 showed the model is a drive that delivers the commanded current at any speed. That requires either a current-sensing driver, or firmware that adds back-EMF compensation.

A common hobby setup instead sets the PWM duty in proportion to the torque command: $V = R\tau_c/(nNk_t)$, clipped to $\pm V_s$. In that **voltage mode**, below the limits,

```math
\tau = nNk_t\,\frac{V - k_tN\omega}{R} = \tau_c - c_m\,\omega,\qquad c_m = \frac{nN^2k_t^2}{R} = 0.0067\ \text{N m s/rad},
```

which is exactly extra viscous friction. Simulating it (`VoltageModeActuator` in `physics_checks.py`; 9.3 output):

| Gains | Current mode (as simulated) | Voltage mode |
|---|---|---|
| Tuned | 0 of 30 fall, cost 2.32 | **30 of 30 fall** |
| Hand-tuned | 0 of 30 fall, cost 4.51 | 0 of 30 fall, cost 7.04 |

Section 9.4 derives why. A drag of $c_m$ is above the tuned gains' runaway threshold $rk_pk_v = 0.0064$ N m s/rad, so with the tuned gains no steady speed exists. The hand-tuned threshold is 0.014, twice $c_m$.

**This is the single most important assumption in the model.** If the hardware uses a voltage-mode driver, the firmware should compensate back-EMF so that the hardware matches the simulation:

```math
V = \frac{R\,\tau_c}{nNk_t} + k_tN\hat\omega .
```

A motor-shaft encoder measures $\hat\omega$ directly, because it reads the rotor angle relative to the stator. The alternative is to simulate the real driver and retune (10.1).

### 9.4 Friction

`wheel_friction` is 0, and robust scoring cannot change it (8.4). Sensitivity, for one run (seed 100, 5.73° start; 9.4 output):

| $b$ (N m s/rad) | 0 | 0.001 | 0.002 | 0.003 | 0.005 |
|---|---|---|---|---|---|
| Tuned | 2.32 | 2.63 | 3.50 | falls at 4.56 s | falls at 3.41 s |
| Hand-tuned | 4.52 | 4.72 | 4.97 | 5.32 | falls at 5.80 s |

**Why the tuned gains are the fragile ones: load torque shifts the speed.** Any steady load torque $\tau_L$ across the wheel–body joint has to be supplied by the motor. Friction, a slope and back-EMF drag are all examples. Until the slow integrator takes the load over, the proportional term must supply it, which needs a pitch error $\theta - \theta_{\text{sp}} = \tau_L/k_p$. At constant speed $\theta = 0$ (2.8), so the lean command must be $\theta_{\text{sp}} = -\tau_L/k_p$. The outer loop only commands that lean when

```math
v - v_{\text{goal}} = \frac{\tau_L}{k_p\,k_v}.
```

This is the same mechanism as the gyro-bias offset in 5.3. Counterintuitively, the robot runs faster than its goal: a backward lean command is the only way the proportional term can produce the forward torque that the load needs.

Viscous friction is a load that grows with speed, $\tau_L = b\,v/r$ (taking $\dot\theta \approx 0$). Substituting gives the quasi-steady speed

```math
v = \frac{v_{\text{goal}}}{1 - b/(r\,k_p k_v)},
```

and no steady speed at all once $b \ge r\,k_pk_v$. That threshold is 0.0064 N m s/rad for the tuned gains and 0.014 for the hand-tuned ones. The tuner's small $k_p$ cut $k_pk_v$ from 0.40 to 0.18 N m per m/s, and the robot's resistance to load torque with it.

*Check (9.4 output):* with $b = 0.003$, $k_i = 0$, the ideal actuator and a constant 1.4 m/s goal, the formula predicts 2.618 m/s (tuned) and 1.782 m/s (hand-tuned). The simulator settles at exactly those values.

In the real scenario the integrator does pull the speed back, but slowly: the tuned pitch loop's integral pole is at 0.49 rad/s, about 2 s (3.5). Meanwhile the speed heads for 2.62 m/s. With $b = 0.003$, the DC motor's forward headroom $\tau^+_{\max}(\omega) - b\omega$ (4.4) reaches zero at 51 rad/s (1.79 m/s); without friction that would only happen at the 2.59 m/s no-load speed. The tuned robot passes 1.79 m/s (its run peaks at 1.87 m/s). Beyond that the motor cannot drive the wheels forward to catch the forward lean, so the robot falls forward.

**Measuring friction.** Run the wheels unloaded at a few PWM levels and record the steady speed $\omega_0$ and the current $i_0$ per motor. The friction torque at the output is then $\tau_f(\omega_0) = nNk_t\,i_0$. Fitting $\tau_f = \tau_c + b\,\omega$ separates Coulomb from viscous friction.

### 9.5 The IMU

The model hands the controller pitch plus white noise. A real robot estimates pitch from a gyro and an accelerometer. An accelerometer measures specific force, so during horizontal acceleration its tilt reading is off by about $\arctan(\ddot x/g)$. At the 29 m/s² peaks of 9.1, that is up to 72°. The firmware must therefore rely on the gyro at high frequency, through a complementary or Kalman filter. Two consequences:

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
| No Coulomb friction or rolling resistance | rarely, on hardware | Coulomb friction is a dead zone, like backlash |
| Sensing and actuation aligned to physics steps | latency and control period are multiples of $h$ | see 7.2 |

## 10. Limitations and future iterations

Grouped by how strong the evidence is that each one matters.

### 10.1 Evidence that the tuned gains fail: fix first

| Limitation | Evidence | Next iteration |
|---|---|---|
| Driver assumed to be current-controlled with exact back-EMF compensation | Voltage-mode driver: 30 of 30 falls (9.3) | Decide the hardware driver. Either add the back-EMF feedforward of 9.3 to the firmware, or move the `VoltageModeActuator` from `physics_checks.py` into `actuators.py` and retune (or randomize between the two). |
| Friction is zero and cannot be randomized | Tuned gains fall at $b \ge 0.003$; the runaway threshold is $rk_pk_v$ (9.4) | Measure it (9.4). Add Coulomb plus viscous friction, $\tau_f = \tau_c\tanh(\omega/\omega_\epsilon) + b\omega$; the smooth tanh keeps RK4 accurate. Randomize it additively (e.g. $b \in [0, b_{\max}]$), since multiplicative spread cannot move a zero. Then the tuner must trade delay margin (small $k_p$) against load-torque rejection (large $k_pk_v$) explicitly. |
| Latency is fixed at 10 ms | Delay margin is the key robustness number (7.4) | Randomize latency per scenario (e.g. 5–20 ms), or add the delay margin from 7.4 to the objective. It is cheap: the eigenvalues of a 6×6 matrix at 10 ms latency. |

### 10.2 Assumptions violated during transients

| Limitation | Evidence | Next iteration |
|---|---|---|
| No traction limit | $\lvert F\rvert/N$ up to 0.91 / 1.43 (9.1) | Simplest: add a penalty on time spent above a measured $\mu$. Full model: make $\varphi$ a fifth coordinate. While sticking, enforce $\dot x = r\dot\varphi$. When the required $\lvert F\rvert$ exceeds $\mu_sN$, switch to sliding with $F = -\mu_kN\operatorname{sign}(\dot x - r\dot\varphi)$. |
| Idealized IMU | Accelerometer tilt error up to 72° (9.5) | Simulate the accelerometer (from $\ddot x$, $\ddot\theta$ and the mount height) and the gyro, and run the same complementary filter the firmware will use. |
| Idealized encoders | Velocity is truth plus white noise | Quantize the wheel angle (counts per revolution) and difference it at the control rate, as the firmware will. |

### 10.3 Physical fidelity

- **Reflected rotor inertia.** The `DCMotorActuator` docstring suggests folding this into `wheel_inertia`, but that is only partly right. Each rotor spins at about $N\omega = N(\dot x/r - \dot\theta)$ relative to the body. So it adds $\tfrac12 J\,(\dot x/r - \dot\theta)^2$ to $T$, where $J = nN^2J_r$. The mass matrix becomes

  ```math
  \mathbf M(\theta) = \begin{pmatrix} a + J/r^2 & Ml\cos\theta - J/r \\ Ml\cos\theta - J/r & d + J \end{pmatrix}.
  ```

  Folding $J$ into $I_w$ gets the first entry right but misses the $-J/r$ coupling and the $+J$ on the body. The matrix stays positive definite, because $T$ is still a sum of squares. Only `planar2d.py:76-78` (and `batch_sim/physics.py:49-51`) would change.
- **Winding inductance.** If $L/R$ is not much less than 5 ms, add the current as a state: $L\,di/dt = V - Ri - k_eN\omega$.
- **Gearbox efficiency and backlash.** Multiply the output torque by an efficiency $\eta$. Model backlash as a dead band in $\varphi - \theta$.
- **Battery sag.** Use $V_s(t) = V_{oc} - R_{\text{batt}}\,i_{\text{total}}$ instead of a constant per run (`CONDITIONS.md` already lists this).
- **Noise as a density.** Specify sensor noise as a density (rad/s/√Hz, as datasheets do) and convert it with $\sigma = \text{ND}\sqrt{f_s/2}$. Today the per-sample $\sigma$ silently changes the physical noise level if the control rate changes.
- **Scope.** A 3D model with yaw (differential drive), slopes, and pushes on the body, as already planned in `ARCHITECTURE.md` and `CONDITIONS.md`.

### 10.4 Numerics and code

- **Cost quadrature.** Integrate the cost with the trapezoid rule (holding the references over each step), which cuts the bias from 0.42% to 0.004% (7.5). Better still, add the running cost as a fifth state integrated by RK4, which makes it 4th-order accurate for free.
- **Timing validation.** Check that `control_dt` and `latency` are integer multiples of `dt` (7.2), and raise an error otherwise. Today a mismatch silently changes the control period or rounds up the latency.

### 10.5 Cost design

- **Choose the weights deliberately.** The effort term is 0.01% of the cost (8.2). Bryson's rule (Franklin, Powell & Emami-Naeini) sets each weight to $1/(\text{largest acceptable value})^2$. For example, 0.1 rad of pitch error, 0.4 N m of torque and 0.2 m/s of velocity error give $w_u \approx 0.06$ and $w_v \approx 0.25$ relative to pitch. That would make effort and pitch matter.
- **Use feasible goal profiles.** Step-shaped goals make much of the velocity cost unavoidable (8.2). Ramped goals at or below $a_{\text{cap}}$ would make the cost measure controller quality rather than physics limits.
- **An LQR baseline.** The cost has LQR form, and 3.1 gives $A$ and $B$. Optimal full-state feedback on the linear model, with the same weights, would show how much performance the cascaded-PID structure costs.
- **A physical fall criterion.** Replace the 60° threshold, or add to it, a check based on recoverability (7.3).

## 11. The batched engine uses the same physics

`batch_sim` implements the same equations in the same order of operations, as functions over arrays of robots:

| Physics | Reference (`balance_sim`) | Batched (`batch_sim`) |
|---|---|---|
| Equations of motion (Section 2) | `planar2d.py:64-87` | `physics.py:36-59` |
| DC motor and ideal actuator (Section 4) | `actuators.py:38, 99-112` | `physics.py:21-33` |
| RK4 (7.1) | `simulator.py:26-31` | `physics.py:66-73` |
| Anti-windup integrator (6.4) | `controllers.py:14-51` | `controller.py:26-49` |
| Cascaded controller (6.1–6.2) | `controllers.py:155-166, 247-257` | `controller.py:52-85` |
| Latency and hold (7.2) | `simulator.py:75-84` (deque) | `schedule.py` (precomputed ring buffer) |
| Cost, left Riemann sums (7.5, 8.1) | `metrics.py:96-105` | `simulate.py:109-114, 131-133` |

In float64 the NumPy backend reproduces `balance_sim` bit for bit (`simulation/tests/test_equivalence.py`; `BATCHED.md`). So every derivation, limitation and fix above applies to both engines. A physics change made in `planar2d.py` or `actuators.py` must be mirrored in `batch_sim/physics.py`, and the equivalence tests will catch a missed one.

## References

- F. Grasser, A. D'Arrigo, S. Colombi, A. C. Rufer, "JOE: A mobile, inverted pendulum," *IEEE Transactions on Industrial Electronics* 49(1), 107–114, 2002. The classic derivation of a two-wheeled inverted pendulum.
- D. T. Greenwood, *Principles of Dynamics*, 2nd ed., Prentice Hall, 1988. Lagrange's equations with generalized forces and dissipation; the energy theorem used in 2.7.
- K. J. Åström, T. Hägglund, *Advanced PID Control*, ISA, 2006. Derivative on measurement, anti-windup by conditional integration.
- S. Skogestad, I. Postlethwaite, *Multivariable Feedback Control: Analysis and Design*, 2nd ed., Wiley, 2005. Limits on bandwidth from RHP zeros, RHP poles and delay.
- G. F. Franklin, J. D. Powell, A. Emami-Naeini, *Feedback Control of Dynamic Systems*, 8th ed., Pearson, 2019. Routh–Hurwitz, root locus, Bryson's rule.
- E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed., Springer, 1993. Order and stability region of RK4.
