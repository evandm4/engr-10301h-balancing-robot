# The physics in the simulation: derivations, justification, limitations

This document derives every equation in the simulator from first principles. For each one it shows where the equation lives in the code, checks it numerically, and lists what the model leaves out and what to do about it. It covers `simulation/balance_sim/`, the reference simulator. `simulation/batch_sim/` runs the same equations in the same order (Section 11), so everything here applies to it too.

All numbers use the robot defined in `config.py`. Since 2026-11-08 that is the datasheet robot ([DATASHEET.md](DATASHEET.md)): real motor, wheel and battery constants, the motors' rotor inertia in the equations of motion (2.2), a gearbox efficiency (4.2) and a pitch-estimate offset (5.1). Two sets of gains appear throughout:

- **Tuned:** the 2026-11-08 CMA-ES result in `results/batch_tuning/20261108-153813/`: $k_p = 1.635$, $k_i = 1.766$, $k_d = 0.299$, $k_v = 0.620$. It was tuned for this robot, the Bryson-rule cost and the 20 s scenario with pushes (`ScenarioConfig`, whose docstring justifies each element), with friction, latency, rotor inertia and the other parameters randomized.
- **Hand-tuned:** `ControllerConfig.hand_tuned_gains`: $k_p = 4.0$, $k_i = 0.5$, $k_d = 0.15$, $k_v = 0.1$.

Several sections also show the 2026-11-06 set ($k_p = 0.660$, $k_i = 0.498$, $k_d = 0.1115$, $k_v = 0.796$), tuned for the robot as it was configured before the datasheet update, to show what the realistic constants change. Numbers for that earlier robot are in this document's git history (commit 944ec30). Numbers for the earlier 12 s scenario, and the gains tuned on it ($k_p = 1.567$, $k_i = 2.101$, $k_d = 0.285$, $k_v = 0.645$), are in commit c6b6ffc; 8.2 keeps its cost table as a record.

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
6. [Controller](#6-controller-controllerspy) (`controllers.py`, `pid_design.py`)
7. [Time stepping, timing and delay](#7-time-stepping-timing-and-delay-simulatorpy) (`simulator.py`)
8. [Cost function and robust scoring](#8-cost-function-and-robust-scoring-metricspy-robustpy) (`metrics.py`, `robust.py`)
9. [Checking the assumptions against simulated runs](#9-checking-the-assumptions-against-simulated-runs)
10. [Limitations and future iterations](#10-limitations-and-future-iterations)
11. [The batched engine uses the same physics](#11-the-batched-engine-uses-the-same-physics)

## Main results

**What checks out:**

- **The equations of motion are correct.** Two independent derivations (Lagrange and Newton–Euler) agree, and both match `Planar2D.derivatives` term for term, rotor inertia included. The mass matrix is invertible at every pitch angle (proof in 2.5). Energy is conserved without torque and friction, and otherwise changes by exactly the work the motor and friction do (to $1.1\times10^{-8}$ J).
- **`critical_pitch_gain` is exact:** on the linearized robot, below it no choice of the other gains can hold the robot up (3.4). The integrator (RK4) shows its expected 4th-order convergence (7.1).

**What the derivations explain:**

- **The robot has an unstable pole at 8.44 rad/s and a right-half-plane zero at 6.74 rad/s** (3.2, 3.3). To speed up it must first roll backward, and the simulator does exactly that.
- **The motors' rotors dominate the wheel side.** Seen through the 20.4:1 gearboxes they add $J/r^2 = 0.34$ kg to the robot's effective translating mass, against 0.018 kg for the wheels' own spin (1). That slows the unstable pole to 8.4 rad/s, against 10.6 with the same body and wheels but no rotors (13.1 for the robot before 2026-11-08), and it is why the gains tuned before 2026-11-08 are unstable on this robot at any latency (7.4).
- **New closed-form result (6.3):** the cascaded controller's characteristic polynomial gives a ceiling on the outer-loop gain: $k_v < (c/e)\,(k_d/k_p)$. This explains why outer gains above about 0.23 oscillate with the hand-tuned inner loop. The tuned inner loop has $k_d/k_p = 0.183$ s, so its ceiling is 0.954, and the tuned $k_v = 0.620$ is below it.
- **An exact discrete-time analysis gives each controller's delay margin** (7.4). The tuned gains tolerate 37.5 ms of latency; the hand-tuned ones tolerate 16 ms. The nonlinear simulator confirms both to within 1 ms.

**What could break on hardware**, in order of severity (Section 9):

1. **The motor driver.** The model assumes the driver delivers the commanded current. If the real driver instead sets PWM duty in proportion to the torque command, capped at the 0.34 N m torque limit, every gain set falls in 30 of 30 runs: with this 1.25 Ω motor the cap is only 2.5 V of the 7.6 V supply, and at 0.7 m/s back-EMF eats most of it (the scenario cruises at 1.3 m/s). Letting the duty reach the full supply keeps the tuned gains up at 33 times the cost (0.783 to 25.6), and the hand-tuned gains still fall in every run (9.3).
2. **Friction.** Its size is unmeasured; the robust-scoring ranges come from the motors' free-run current (DATASHEET.md 5.2). Any load torque pushes the speed up by $\tau_L/(k_pk_v)$ until the integrator takes it over (9.4). The tuned gains' $k_pk_v$ gives a runaway threshold of 0.035 N m s/rad, 23 times the largest viscous friction scored.
3. **The IMU model.** It is idealized: a raw accelerometer tilt estimate would be off by up to 50° (tuned) or 52° (hand-tuned) during these transients.
4. **Traction.** The no-slip assumption needs a friction coefficient of 0.54 for the tuned gains and 0.68 for the hand-tuned ones.

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
| $I_b$ | body pitch inertia about its own center of mass | `body_inertia` | 0.0065 kg m² |
| $m_w$ | mass of both wheels | `wheel_mass` | 0.034 kg |
| $r$ | wheel radius | `wheel_radius` | 0.035 m |
| $I_w$ | inertia of both wheels about the axle | `wheel_inertia` | 2.2e-5 kg m² |
| $J$ | motor rotors reflected to the axle, $nN^2J_r$ | `ActuatorConfig.rotor_inertia` ($J_r$ per motor) | $2\times20.4^2\times5\times10^{-7} = 4.16\times10^{-4}$ kg m² |
| $b$ | viscous friction between wheels and body | `wheel_friction` | 0 (randomized 0–0.0015 N m s/rad) |
| $g$ | gravity | `gravity` | 9.81 m/s² |
| $\tau_{\max}$ | torque limit, both motors together | `max_torque` | 0.34 N m |

Where the values come from (DATASHEET.md): $m_w$ is two 14 g Pololu 70 mm wheels with 3.2 g hubs. $I_w = 0.65\,m_wr^2$, between a disk and a ring because the tire sits at the rim. $I_b$ is what a battery-on-top tower tall enough to put the center of mass 0.10 m up has (DATASHEET.md 4). $J_r$ is estimated from a 370-size armature; Pololu doesn't publish it (DATASHEET.md 5.3).

Five combinations recur everywhere:

```math
a = M + m_w + \frac{I_w + J}{r^2} = 1.1917\ \text{kg},\qquad
b_0 = Ml - \frac{J}{r} = 0.06811\ \text{kg m},\qquad
d = M l^2 + I_b + J = 0.01492\ \text{kg m}^2,
```
```math
\Delta_0 = a d - b_0^2 = 1.3136\times10^{-2}\ \text{kg}^2\text{m}^2,\qquad
c = a + \frac{b_0}{r} = 3.1377\ \text{kg},\qquad
e = \frac{d}{r} + b_0 = 0.4943\ \text{kg m}.
```

- $a$ is the robot's effective translating mass. Wheel spin adds $I_w/r^2 = 0.018$ kg to it; the rotors add $J/r^2 = 0.340$ kg, 19 times as much and 40% of the body's mass.
- $b_0$ is the mass matrix's coupling when upright. The rotors take $J/r = 0.0119$ kg m off $Ml$ (2.2).
- $d$ is the body's pitch inertia about the axle (parallel-axis theorem), plus the rotors, which turn whenever the body pitches relative to the wheels.
- $\Delta_0$ is the mass-matrix determinant when upright.
- $c$ converts motor torque into restoring moment (3.4). $J$ cancels out of it: $c = M + m_w + I_w/r^2 + Ml/r$.
- $e$ converts motor torque into forward acceleration (3.1).

With $J = 0$, $b_0 = Ml$ and these reduce to the expressions used before 2026-11-08.

**Sign conventions.**

- $x$ is positive forward.
- $\theta$ is measured from vertical, positive when leaning forward.
- $\varphi$ is the wheel angle, positive when rolling forward.

With forward drawn to the right, $\varphi$ and $\theta$ both increase clockwise. So the motor shaft angle (rotor relative to the stator, which is bolted to the body) is $\varphi-\theta$, and the motor speed is

```math
\omega = \dot\varphi - \dot\theta = \frac{\dot x}{r} - \dot\theta .
```

This is `relative_wheel_speed` (`planar2d.py:110`). Each motor's rotor turns at $N\omega$ relative to the body.

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

The kinetic energy has five parts: wheel translation, wheel spin, body translation, body rotation, and the motor rotors' spin relative to the body. Each of the $n$ rotors (inertia $J_r$) turns at $N\omega$ relative to the body, so together they carry $\tfrac12 nJ_rN^2\omega^2 = \tfrac12 J\omega^2$. (Their absolute speed is $N\omega + \dot\theta$; the cross terms are smaller by a factor $N$ and are dropped.) Using $\dot\varphi = \dot x/r$ and $\omega = \dot x/r - \dot\theta$:

```math
T = \tfrac12 m_w\dot x^2 + \tfrac12 I_w\dot\varphi^2 + \tfrac12 M\lvert\dot{\mathbf p}_G\rvert^2 + \tfrac12 I_b\dot\theta^2 + \tfrac12 J\omega^2
  = \tfrac12 a\,\dot x^2 + b(\theta)\,\dot x\dot\theta + \tfrac12 d\,\dot\theta^2,\qquad b(\theta) = Ml\cos\theta - \frac{J}{r}.
```

The rotor term contributes $J/r^2$ to $a$, $J$ to $d$ and $-J/r$ to the coupling. Folding $J$ into $I_w$ instead (as the code's docstring once suggested) gets $a$ right but misses the other two.

Only the body's height changes, so the potential energy (dropping constants) is

```math
V = Mgl\cos\theta .
```

In matrix form $T = \tfrac12\dot q^{\mathsf T}\mathbf M(\theta)\dot q$, with

```math
\mathbf M(\theta) = \begin{pmatrix} a & b(\theta) \\ b(\theta) & d \end{pmatrix}.
```

These are the $T$ and $V$ in the `planar2d.py` docstring and in `Planar2D.energy` (`planar2d.py:147`).

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

`planar2d.py:110-115` computes $\tau_{\text{net}}$ the same way: it gets the delivered torque from the actuator (Section 4), then subtracts friction. Friction modeled this way is equivalent to a Rayleigh dissipation function $\mathcal F = \tfrac12 b\omega^2$. Because it acts between two parts of the robot, it can only dissipate energy (2.7).

### 2.4 Euler–Lagrange equations

With $L = T - V$:

```math
\frac{\partial L}{\partial\dot x} = a\dot x + b(\theta)\,\dot\theta,\qquad
\frac{d}{dt}\frac{\partial L}{\partial\dot x} = a\ddot x + b(\theta)\,\ddot\theta - Ml\sin\theta\,\dot\theta^2,\qquad
\frac{\partial L}{\partial x} = 0,
```
```math
\frac{\partial L}{\partial\dot\theta} = b(\theta)\,\dot x + d\dot\theta,\qquad
\frac{d}{dt}\frac{\partial L}{\partial\dot\theta} = b(\theta)\,\ddot x - Ml\sin\theta\,\dot x\dot\theta + d\ddot\theta,
```
```math
\frac{\partial L}{\partial\theta} = -Ml\sin\theta\,\dot x\dot\theta + Mgl\sin\theta .
```

Setting $\frac{d}{dt}\frac{\partial L}{\partial\dot q_i} - \frac{\partial L}{\partial q_i} = Q_i$ gives:

```math
\text{(E1)}\qquad a\,\ddot x + b(\theta)\,\ddot\theta = \frac{\tau_{\text{net}}}{r} + Ml\sin\theta\,\dot\theta^2
```
```math
\text{(E2)}\qquad b(\theta)\,\ddot x + d\,\ddot\theta = -\tau_{\text{net}} + Mgl\sin\theta
```

In (E2) the $\dot x\dot\theta$ terms cancel. $J$ is constant, so it adds no velocity terms. The two equations have a direct physical reading (2.6):

- (E1) is Newton's second law for the whole robot in the horizontal direction.
- (E2) is the body's moment balance about the axle.

### 2.5 Solving for the accelerations

(E1) and (E2) form the linear system $\mathbf M(\theta)\,(\ddot x, \ddot\theta)^{\mathsf T} = (\text{rhs}_x, \text{rhs}_\theta)^{\mathsf T}$. Cramer's rule solves it:

```math
\ddot x = \frac{d\,\text{rhs}_x - b(\theta)\,\text{rhs}_\theta}{\Delta(\theta)},\qquad
\ddot\theta = \frac{a\,\text{rhs}_\theta - b(\theta)\,\text{rhs}_x}{\Delta(\theta)},\qquad
\Delta(\theta) = ad - b(\theta)^2,\quad b(\theta) = Ml\cos\theta - J/r .
```

(Here $b(\theta)$ is the code's variable `b`, the off-diagonal mass-matrix entry, not the friction coefficient.) Each line of `Planar2D.derivatives` is one of these terms:

| Term | Code |
|---|---|
| $\omega = \dot x/r - \dot\theta$ | `planar2d.py:110` |
| $\tau_{\text{net}}$ = actuator output $- b\omega$ | `planar2d.py:111-115` |
| $a$, $b(\theta)$, $d$ | `planar2d.py:120-122` |
| $\text{rhs}_x = \tau_{\text{net}}/r + Ml\sin\theta\,\dot\theta^2$ | `planar2d.py:124` |
| $\text{rhs}_\theta = -\tau_{\text{net}} + Mgl\sin\theta$ | `planar2d.py:125` |
| $\Delta$, $\ddot x$, $\ddot\theta$ | `planar2d.py:127-129` |

**The solve never fails.** $T$ is a sum of squares, $\tfrac12(m_w + I_w/r^2)\dot x^2 + \tfrac12M\lvert\dot{\mathbf p}_G\rvert^2 + \tfrac12I_b\dot\theta^2 + \tfrac12J\omega^2$, and the first and third terms alone vanish only when $\dot x = \dot\theta = 0$. So $T > 0$ for every nonzero velocity, which makes $\mathbf M(\theta)$ positive definite at every angle, including past the fall angle. `RobotParams.__post_init__` enforces the positive masses and inertias this needs (and $J \ge 0$). With $J = 0$ the bound is explicit: $\Delta(\theta) \ge (m_w + I_w/r^2)d + MI_b > 0$. For this robot $\Delta = ad - (Ml\cos\theta - J/r)^2$ is smallest upside down, $9.33\times10^{-3}$ at $\theta = \pi$, and is $1.31\times10^{-2}$ upright and $1.76\times10^{-2}$ horizontal (1 output).

### 2.6 Independent check: Newton–Euler

Let $F$ be the horizontal ground (friction) force on the wheels and $N$ the normal force.

**Horizontal.** Momentum balance for the whole robot gives $F = m_w\ddot x + M\ddot x_G$. Differentiating $x_G$ from 2.1:

```math
F = (M + m_w)\,\ddot x + Ml\bigl(\cos\theta\,\ddot\theta - \sin\theta\,\dot\theta^2\bigr).
```

The motors' torque $\tau_{\text{net}}$ (referred to the wheels) first spins up the rotors: $J\dot\omega$ of it goes into them, and the gear trains pass $\tau_w = \tau_{\text{net}} - J\dot\omega$ to the wheels. The wheels' spin about the axle gives $I_w\ddot\varphi = \tau_w - Fr$, so

```math
F = \frac{1}{r}\Bigl(\tau_{\text{net}} - J\Bigl(\frac{\ddot x}{r} - \ddot\theta\Bigr) - \frac{I_w\ddot x}{r}\Bigr).
```

Equating the two expressions for $F$ gives exactly (E1).

**Body moments about the moving axle point $A$** (clockwise positive, like $\theta$). The balance is $\sum M_A = I_b\ddot\theta + (\mathbf p_G - \mathbf p_A)\times M\mathbf a_G$, where the moments are gravity's $Mgl\sin\theta$ and the reaction of the torque passed to the wheels, $-\tau_w = -\tau_{\text{net}} + J\dot\omega$. The axle force passes through $A$, so it contributes no moment. Expanding the cross product gives $Ml^2\ddot\theta + Ml\cos\theta\,\ddot x$; moving $J\dot\omega = J(\ddot x/r - \ddot\theta)$ to the left gives exactly (E2).

**Vertical.** The same approach gives the normal force (used in 9.1):

```math
N = (M + m_w)\,g - Ml\bigl(\sin\theta\,\ddot\theta + \cos\theta\,\dot\theta^2\bigr).
```

*Numerical check (9.1–9.5 output):* along all 30 validation runs, the Lagrange and Newton–Euler expressions for $F$ agree to $8\times10^{-15}$ N.

### 2.7 Energy and power balance

Let $E = T + V$. For a Lagrangian whose kinetic energy is quadratic in $\dot q$ and has no explicit time dependence, the energy function $\sum_i \dot q_i\,\partial L/\partial\dot q_i - L$ equals $T + V$. Its rate of change is the power of the generalized forces. Here that gives

```math
\frac{dE}{dt} = Q_x\dot x + Q_\theta\dot\theta = \tau_{\text{net}}\Bigl(\frac{\dot x}{r} - \dot\theta\Bigr) = \tau\,\omega - b\,\omega^2 .
```

Three consequences:

1. With $\tau = 0$ and $b = 0$, energy is conserved. `test_energy_conserved_without_torque_or_friction` checks this.
2. Friction only removes energy, since $-b\omega^2 \le 0$. `test_friction_dissipates_energy` checks this.
3. The motor's mechanical power is $\tau\omega$, so the speed the actuator must be asked about is the relative speed $\omega$. That is why `Actuator.torque` takes `relative_wheel_speed`.

*Numerical check (2.7 output):* one 10 s closed-loop run with $b = 0.001$. The energy change of 1.012850 J (rotor energy included) matches the integrated power of 1.012850 J; the difference is $1.1\times10^{-8}$ J.

### 2.8 Equilibria, and lean versus acceleration

**Upright equilibrium.** With $\tau_{\text{net}} = 0$ and $\theta = \dot\theta = 0$, both accelerations vanish for any speed $\dot x$. So the robot can stand upright at any constant speed (`test_upright_equilibrium_holds`).

**Steady lean.** Hold $\theta = \theta_s$ constant ($\dot\theta = \ddot\theta = 0$). Then (E1) gives $\tau = a r\ddot x$. Substituting into (E2):

```math
\ddot x = \frac{Mgl\sin\theta_s}{b(\theta_s) + a r},\qquad \tau = a r\,\ddot x .
```

Linearized, $\ddot x \approx G\,\theta_s$ with

```math
G = \frac{Mgl}{b_0 + ar} = 7.15\ \text{m/s}^2\ \text{per rad}.
```

A point mass on a massless cart would give $G = g = 9.81$. This robot needs a larger lean for the same acceleration, because the torque that drives the wheels also pushes back on the body.

At the 25° lean cap (`ControllerConfig.max_lean_deg`): $\ddot x = 3.24$ m/s², which needs $\tau = 0.135$ N m. So the cap limits acceleration well before the motors run out of torque at low speed. $G$ is the plant gain that the outer velocity loop works through (6.2).

## 3. Linearization

### 3.1 Linear model

Linearize about upright: $\sin\theta\approx\theta$, $\cos\theta\approx1$, drop $\dot\theta^2$. Also take $b = 0$ and an unsaturated actuator (delivered torque = command). Then

```math
\begin{pmatrix} a & b_0 \\ b_0 & d \end{pmatrix}\begin{pmatrix}\ddot x\\ \ddot\theta\end{pmatrix}
= \begin{pmatrix}\tau/r \\ Mgl\,\theta - \tau\end{pmatrix}
\quad\Longrightarrow\quad
\ddot x = \frac{-b_0Mgl\,\theta + e\,\tau}{\Delta_0},\qquad
\ddot\theta = \frac{aMgl\,\theta - c\,\tau}{\Delta_0}.
```

With state $s = (x, \dot x, \theta, \dot\theta)$, this is $\dot s = A s + B\tau$:

```math
A = \begin{pmatrix} 0&1&0&0\\ 0&0&-4.07&0\\ 0&0&0&1\\ 0&0&71.19&0 \end{pmatrix},\qquad
B = \begin{pmatrix} 0\\ 37.63\\ 0\\ -238.85 \end{pmatrix}.
```

Every entry is 2.4–3.6 times smaller than for the robot before 2026-11-08 ($A_{4,3} = 172$, $B_4 = -747$): the rotor inertia, and to a lesser degree the doubled body inertia, make the same torque or tilt produce much less acceleration. The signs match the tests:

- $B_2 > 0$ and $B_4 < 0$: positive torque drives the wheels forward and tips the body back (`test_positive_torque_pushes_wheels_forward_and_body_back`).
- $A_{4,3} > 0$: leaning forward accelerates the fall (`test_leaning_forward_accelerates_pitch_forward_without_torque`).

### 3.2 Open-loop poles

The characteristic polynomial is $s^2\,(s^2 - aMgl/\Delta_0)$. Its roots are:

- $0, 0$: position and velocity, which nothing pulls back;
- $\pm p$, with $p = \sqrt{aMgl/\Delta_0} = 8.44$ rad/s.

The unstable pole $+p$ means a 118.5 ms time constant: a small tilt doubles every 82 ms. Without the rotors ($J = 0$, same body and wheels) it would be 10.6 rad/s, and it was 13.1 rad/s for the robot before 2026-11-08.

- **Falling from rest:** the linear estimate of the time to fall from 0.05 rad to 60° is $\cosh^{-1}(1.047/0.05)/p = 0.443$ s. The simulator gives 0.457 s; the nonlinear terms account for the difference. `test_uncontrolled_robot_falls` allows 0.1–1 s.
- **Free versus locked wheels:** with the wheels locked the pole would be $\sqrt{Mgl/d} = 7.25$ rad/s ($d$ includes $J$: the rotors still turn as the body pitches against the locked wheels). Free wheels fall faster, because the base rolls backward as the body tips forward.

### 3.3 Transfer functions and the right-half-plane zero

From the Laplace transform of 3.1:

```math
\frac{\Theta(s)}{T(s)} = \frac{-c}{\Delta_0 s^2 - aMgl},\qquad
\frac{X(s)}{T(s)} = \frac{e\,s^2 - Mgl/r}{s^2\,(\Delta_0 s^2 - aMgl)} .
```

$X/T$ has zeros at $\pm z$, where

```math
z = \sqrt{\frac{Mgl}{d + b_0r}} = 6.74\ \text{rad/s}.
```

The zero at $+z$ is in the right half plane, which makes position and velocity **non-minimum-phase**. To accelerate forward the robot must lean forward, and the only way to tip the body forward is to drive the wheels backward under it first. So a step up in goal velocity starts with motion the wrong way.

The simulator shows this (3 output, tuned gains, seed 100). The goal blends smoothly (over 1.6 s and 3 s for these two changes), so the effect is small:

- **Goal 0 → 1.3 m/s, starting at t = 2 s:** velocity goes from −0.010 to −0.071 m/s within 0.4 s before rising.
- **Goal 1.3 → −0.5 m/s, starting at t = 9.5 s:** velocity first rises from 1.290 to 1.318 m/s within 0.4 s.

The RHP zero is a fundamental limit on velocity control (6.2).

### 3.4 Critical pitch gain (`critical_pitch_gain`, `planar2d.py:41`)

Eliminate $\ddot x$ between the two linear equations. (E1) gives $\ddot x = (\tau/r - b_0\ddot\theta)/a$; substituting into (E2):

```math
\underbrace{\Bigl(d - \frac{b_0^2}{a}\Bigr)}_{\Delta_0/a}\ddot\theta
= Mgl\,\theta - \underbrace{\Bigl(1 + \frac{b_0}{ar}\Bigr)}_{c/a}\tau .
```

The motor's restoring moment on the body is $(c/a)\,\tau$, which has two parts:

- the direct reaction torque $\tau$;
- an extra $(b_0/ar)\,\tau$ from accelerating the axle back under the center of mass.

For this robot $c/a = 2.63$, so the wheel acceleration contributes 1.6 times as much restoring moment as the reaction torque (2.7 times without the rotors, which make the axle harder to accelerate).

With proportional control $\tau = k_p\theta$ the equation becomes $(\Delta_0/a)\,\ddot\theta = -\bigl((c/a)k_p - Mgl\bigr)\theta$. This is a restoring spring if and only if

```math
k_p > k_{p,\text{crit}} = \frac{aMgl}{c} = 0.2981\ \text{N m/rad}.
```

That is `critical_pitch_gain`, line for line. It is 2.6 times smaller than the naive static answer $Mgl = 0.785$, which ignores the wheel-acceleration term. Since $c$ does not depend on $J$, the rotors raise $k_{p,\text{crit}}$ only through $a$.

The search space uses $1.1\,k_{p,\text{crit}}$ as the lower bound on $k_p$ (`search_space.py:87`). Gains below $k_{p,\text{crit}}$ fall in the linear model whatever the other gains are (`test_gain_threshold_matches_simulation`), so the tuner never searches them. Since 2026-11-06, $k_{p,\text{crit}}$ there is taken for the most demanding robot that robust scoring can draw (`ProjectConfig.kp_floor_robot`: mass, COM height, wheel and rotor inertia at the top of their spreads, body inertia at the bottom). It needs 0.386 N m/rad, 1.29 times the nominal robot, so the floor is 0.424 rather than 0.328.

### 3.5 Full pitch-loop stability (Routh–Hurwitz)

Add damping and integral action: $\tau = k_p\theta + k_d\dot\theta + k_i\!\int\!\theta$. Then the pitch loop's characteristic polynomial is

```math
\Delta_0\,s^3 + c\,k_d\,s^2 + (c\,k_p - aMgl)\,s + c\,k_i = 0 .
```

The Routh–Hurwitz conditions for stability are $k_d > 0$, $k_i > 0$, and

```math
k_p > k_{p,\text{crit}} + \frac{\Delta_0\,k_i}{c\,k_d}.
```

So $k_{p,\text{crit}}$ is necessary but not sufficient. The bound is 0.3228 for the tuned gains and 0.3120 for the hand-tuned ones.

The continuous-time pitch-loop poles, ignoring sampling and delay, are:

| Gains | Poles (rad/s) | $k_p / k_{p,\text{crit}}$ |
|---|---|---|
| Tuned | −66.6, −2.35 ± 0.90j | 5.5 |
| Hand-tuned | −17.9 ± 23.7j, −0.14 | 13.4 |
| Tuned 2026-11-06 (earlier robot's gains) | −23.1, −1.76 ± 1.43j | 2.2 |

These poles ignore delay, which is what actually limits the gains (7.4).

### 3.6 Why pitch-only control drifts

A pitch-only PID never feeds back $x$ or $\dot x$, so the two zero eigenvalues of 3.2 stay at zero. The pitch integrator does absorb a constant torque offset, such as the gyro bias's $k_db_g$ (5.3). But the transient while it does so leaves a velocity offset that nothing removes, so the robot drifts at a constant speed. `CONDITIONS.md` reports a pitch-only controller with gyro bias that drifted 0.39 m in 10 s and was still moving. Feeding velocity back is what the outer loop of the cascade does (6.2).

## 4. Actuator model (`actuators.py`)

### 4.1 One brushed DC motor

The armature circuit, and the motor's torque, are

```math
V = R\,i + L\frac{di}{dt} + k_e\,\omega_m,\qquad \tau_m = k_t\,i .
```

- **Inductance is neglected** (quasi-static current). This holds if the electrical time constant $L/R$ is much shorter than the 5 ms control period. For this 370-size motor it is about 0.2–0.5 ms (DATASHEET.md, an estimate).
- **$k_e = k_t$ in SI units.** The electrical power converted, $k_e\omega_m i$, must equal the mechanical power $\tau_m\omega_m = k_t i\omega_m$. So one `torque_constant` serves for both (`actuators.py:49`). Pololu's catalogue numbers give 5.3 mV s/rad from speed and 4.9 mN m/A from stall, which agree to 8%; config.py uses 0.0053.

### 4.2 Gearbox, two motors, and which speed

- **Gearbox:** a ratio $N$ with efficiency $\eta$ gives $\omega_m = N\omega$ and $\tau_{\text{out}} = \eta N\tau_m$ (`MotorParams.gear_efficiency`; 0.78 from Pololu's geared and bare stall torques). The back-EMF still uses the full $k_t$.
- **Two motors:** $n$ identical motors carrying equal current give a total torque $\tau = nN\eta k_t\,i$.
- **Which speed:** the stator is bolted to the body, so the speed that generates back-EMF is the relative speed $\omega$ (2.7), not the wheel's absolute speed.

### 4.3 What `DCMotorActuator.torque` computes (`actuators.py:108-121`)

1. Clip the command: $\tau_c = \operatorname{clip}(\tau_{\text{cmd}}, \pm\tau_{\max})$.
2. Current needed for that torque: $i^* = \tau_c/(nN\eta k_t)$. The firmware is taken to know the measured output torque per amp, losses included.
3. Voltage needed to push $i^*$ against the back-EMF: $V^* = R\,i^* + k_tN\omega$.
4. Clip to the battery: $V = \operatorname{clip}(V^*, \pm V_s)$.
5. Resulting current, clipped to the driver limit: $i = \operatorname{clip}\bigl((V - k_tN\omega)/R,\ \pm i_{\max}\bigr)$.
6. Delivered torque: $\tau = nN\eta k_t\,i$.

**When neither clip is active, $i = i^*$ and $\tau = \tau_c$ exactly.** The model is a current-controlled (torque-controlled) drive with perfect back-EMF compensation. The motor's physics only show up through its two limits. That assumption matters a great deal (9.3).

### 4.4 The torque–speed envelope

**Forward torque at forward motor speed $\omega \ge 0$:**

```math
\tau^+_{\max}(\omega) = \min\Bigl\{\tau_{\max},\; nN\eta k_t\,i_{\max},\; nN\eta k_t\,\frac{V_s - k_tN\omega}{R}\Bigr\}.
```

**Braking ($\tau < 0$ while $\omega > 0$):** setting $V = -V_s$ would drive the current to $-(V_s + k_tN\omega)/R$, past the current limit. So braking is current-limited at every speed. This is plugging; it assumes the driver and battery can carry that current.

With the configured motor (4 output):

| Quantity | Formula | Value |
|---|---|---|
| Stall torque | $nN\eta k_t\min(i_{\max}, V_s/R)$ | 0.337 N m. Current-limited, since $V_s/R = 6.08$ A > 2 A. Just under `max_torque` (0.34). |
| No-load speed | $V_s/(k_tN)$ | 70.3 rad/s, which is 2.46 m/s with the body upright and still |
| Corner speed | $(V_s - Ri_{\max})/(k_tN)$ | 47.2 rad/s, which is 1.65 m/s |

The low-resistance winding keeps the full current-limited torque up to 1.65 m/s (the guessed 3 Ω motor before 2026-11-08 lost torque above 0.49 m/s). Above the corner speed, forward torque falls linearly with speed:

| Speed | Back-EMF | Forward torque available |
|---|---|---|
| 0.8 m/s | 2.47 V | 0.337 N m |
| 1.0 m/s | 3.09 V | 0.337 N m |
| 1.4 m/s | 4.32 V | 0.337 N m |
| 2.0 m/s | 6.18 V | 0.192 N m (57% of stall) |

The scenario cruises at 1.3 m/s, 79% of the corner speed, chosen as the highest speed at which every randomized motor still has the torque that lean-cap acceleration needs (`ScenarioConfig`). The tuned runs peak at 1.69 m/s while recovering from the forward push (hand-tuned 1.39 m/s; 9.1–9.5 output), so on the nominal robot back-EMF binds only briefly. On the weakest randomized motors (corner 1.16 m/s) it binds through the whole cruise. The motor delivers less than the clipped command 0.08% of the time (tuned) and 0.81% (hand-tuned).

### 4.5 Ideal actuator and friction

`IdealActuator` returns $\operatorname{clip}(\tau_{\text{cmd}}, \pm\tau_{\max})$. It is the DC motor model in the limit $V_s, i_{\max}\to\infty$. Friction is applied after either actuator (`planar2d.py:112-115`), as derived in 2.3.

### 4.6 The cost's "effort" is copper loss

When unsaturated, each motor carries $i = \tau/(nN\eta k_t)$, so the resistive heating in all motors (driver and wiring resistance included in $R$) is

```math
P_{\text{Cu}} = nR\,i^2 = \frac{R}{nN^2\eta^2k_t^2}\,\tau^2 = 88\ \text{W per (N m)}^2 .
```

So the cost's effort term $\int\tau^2\,dt$ (Section 8) is proportional to the energy dissipated in the motor windings. This gives the effort weight a physical meaning.

## 5. Sensor model (`sensors.py`)

### 5.1 Model

Once per control tick, the controller sees

```math
\hat\theta = \theta + \sigma_\theta\xi_1 + b_\theta,\qquad
\hat{\dot\theta} = \dot\theta + \sigma_{\dot\theta}\xi_2 + b_g,\qquad
\hat{\dot x} = \dot x + \sigma_v\xi_3,
```

where the $\xi$ are independent standard normal draws, $b_\theta$ is a constant pitch-estimate offset (IMU mounting and balance-point error, `pitch_bias`) and $b_g$ is a constant gyro bias (`sensors.py:41-44`). Position is also noised, but the controller never reads it. The project values are $\sigma_\theta = 0.005$ rad, $b_\theta = 0.01$ rad, $\sigma_{\dot\theta} = 0.02$ rad/s, $b_g = 0.01$ rad/s and $\sigma_v = 0.02$ m/s. The white-noise levels are 10–30 times the IMU datasheets' and are kept conservative for chassis vibration until measured; the encoder velocity noise matches differencing 48 CPR counts at 200 Hz (DATASHEET.md 2.3).

### 5.2 Noise propagated to torque

The controller (Section 6) gives torque sensitivities $\partial\tau/\partial\hat\theta = k_p$, $\partial\tau/\partial\hat{\dot\theta} = k_d$ and $\partial\tau/\partial\hat{\dot x} = k_pk_v$. The integral path adds only $k_iT\sigma$ per tick. With independent noise, the per-tick torque noise is

```math
\sigma_\tau = \sqrt{(k_p\sigma_\theta)^2 + (k_d\sigma_{\dot\theta})^2 + (k_pk_v\sigma_v)^2}.
```

This is 0.0227 N m (6.7% of $\tau_{\max}$) for the tuned gains, and 0.0217 N m (6.4%) for the hand-tuned ones (5 output). The two are now about equally noisy. The tuned controller's lower $k_p$ removes most of the pitch noise (0.0082 against 0.0200 N m), but its larger $k_pk_v$ makes the velocity term dominate (0.0203 against 0.0080 N m).

### 5.3 Gyro bias in steady state

At constant velocity, (E1) and (E2) with $\ddot x = \ddot\theta = 0$ force $\tau_{\text{net}} = 0$ and $\theta = 0$. Two cases:

**PD inner loop ($k_i = 0$).** Setting the torque to zero gives $-k_p\theta_{\text{sp}} + k_db_g = 0$. Since $\theta_{\text{sp}} = k_v(v_{\text{goal}} - v)$, there is a steady velocity error

```math
v_{\text{goal}} - v = \frac{k_d\,b_g}{k_p\,k_v},
```

which is 2.9 mm/s for the tuned gains.

**With $k_i > 0$.** The integrator stops moving only when $\hat\theta - \theta_{\text{sp}} = 0$, which (with $b_\theta = 0$) forces $\theta_{\text{sp}} = 0$ and therefore $v = v_{\text{goal}}$. The integrator settles at $-k_db_g/k_i$, well inside its limit. Integral action rejects the gyro bias completely.

**A pitch offset is different: the pitch loop cannot reject it.** The controller sees $\hat\theta = \theta + b_\theta$, and at constant speed $\theta = 0$, so with or without $k_i$ the inner loop settles where $\theta_{\text{sp}} = b_\theta$. That needs $k_v(v_{\text{goal}} - v) = b_\theta$:

```math
v_{\text{goal}} - v = \frac{b_\theta}{k_v},
```

which is 16.1 mm/s for the tuned gains and 0.10 m/s for the hand-tuned ones (5 output). Only an integral on the velocity error (`kv_i`, 0 in this project) or calibrating the offset out would remove it. The torque it costs is small ($k_pb_\theta = 0.016$ N m, tuned), but the steady velocity error enters the cost directly.

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
| Tuned | 4.43 rad/s | 0.23 s |
| Hand-tuned | 0.71 rad/s | 1.40 s |

Two things cap $\omega_v$:

1. The outer loop must stay well below the inner loop's speed.
2. The RHP zero at $z = 6.74$ rad/s: a standard rule of thumb keeps bandwidth below $z/2 = 3.37$ rad/s (Skogestad & Postlethwaite, ch. 5).

The tuned estimate (4.43 rad/s) is above that rule of thumb, so the first-order picture is optimistic for the tuned gains. The exact poles in 6.3 show an oscillatory pair at $-3.37 \pm 7.98j$ rad/s and a slow pole at $-0.98$ rad/s (and a fast one at $-25.5$), not a single lag.

The lean cap also bounds acceleration at 3.24 m/s² (2.8). There is no position loop, so position is free to drift.

### 6.3 Cascade stability in closed form: how much outer gain the inner loop allows

The estimate in 6.2 treats the two loops separately. Solving them together gives a sharper result. Use the linear plant of 3.1, written in terms of $V = sX$, and continuous control without delay. With goal 0, the error is $e = \theta + k_v v$, so

```math
T(s) = \Bigl(k_p + \frac{k_i}{s}\Bigr)\bigl(\Theta + k_vV\bigr) + k_d\,s\,\Theta .
```

Substitute into $asV + b_0s^2\Theta = T/r$ and $b_0sV + (ds^2 - Mgl)\Theta = -T$. Set the determinant to zero and multiply by $s$. The terms in $k_v\times(\text{pitch controller})$ cancel (using $ea - cb_0 = \Delta_0/r$), which leaves

```math
\Delta_0 s^4 + (c\,k_d - e\,k_pk_v)\,s^3 + (c\,k_p - aMgl - e\,k_ik_v)\,s^2 + (c\,k_i + h\,k_pk_v)\,s + h\,k_ik_v = 0,
```

```math
e = \frac{d}{r} + b_0 = 0.494\ \text{kg m},\qquad h = \frac{Mgl}{r} = 22.4\ \text{N}.
```

Two checks: with $k_v = 0$ this reduces to $s$ times the pitch-loop polynomial of 3.5, and its roots match the eigenvalues of the closed-loop state matrix to $9\times10^{-14}$ (6.3 output).

**Every coefficient must be positive.** The $s^3$ coefficient gives a simple necessary condition:

```math
k_v < \frac{c}{e}\cdot\frac{k_d}{k_p}.
```

The velocity feedback reaches the torque as $k_pk_v v$ and acts through the non-minimum-phase path (3.3). There it subtracts from the damping of the fast pitch mode, which is what the $s^3$ coefficient represents. So **the outer gain the cascade can afford is proportional to the inner loop's ratio $k_d/k_p$.** The full Routh–Hurwitz conditions for a quartic give the exact limit (6.3 output):

| Inner loop | $k_d/k_p$ | Bound $(c/e)(k_d/k_p)$ | Exact Routh–Hurwitz limit on $k_v$ |
|---|---|---|---|
| Hand-tuned (4, 0.5, 0.15) | 0.0375 s | 0.238 | 0.226 |
| Tuned (1.635, 1.766, 0.299) | 0.183 s | 1.159 | 0.954 |

The nonlinear simulator confirms the hand-tuned limit. With the hand-tuned inner loop and no latency:

- $k_v = 0.20$ settles to $1.3\times10^{-5}$ rad;
- $k_v = 0.26$ falls at 1.55 s.

That is `CONDITIONS.md`'s empirical observation, "outer gains much above Kv_p of about 0.25 oscillate", now derived (on the earlier robot the limit was 0.295, and above it the robot went into a limit cycle instead of falling).

**What this says about the tuner's result.** Velocity tracking is half of the nominal cost (8.2), so the tuner wants a large $k_v$. With the hand-tuned inner loop, $k_v$ is capped near 0.23, and with the 10 ms latency even $k_v = 0.2$ is unstable (7.4). The tuner cut $k_p$ from 4 to 1.64 and raised $k_d$ from 0.15 to 0.299, so $k_d/k_p$ went from 0.0375 to 0.183 s. That lifts the Routh–Hurwitz ceiling on $k_v$ to 0.954, and the tuned $k_v = 0.620$ sits below it. The delay margin peaks near $k_v = 0.7$ (7.4), and the tuned value is close to that peak. This explains the direction of the change; it does not prove the optimum is unique.

### 6.4 Anti-windup (`_limited_integral`, `controllers.py:14-51`)

**Why it is needed.** Without it, the integral term alone could ask for $k_i\times\text{integral limit} = 5\times1 = 5$ N m from an actuator that delivers 0.34 N m (`archive/CHANGES_FROM_ORIGINAL.md`, item 2).

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

### 6.5 A model-based design: what the textbook method gives (`pid_design.py`)

The analysis above also gives a way to design the cascade without simulation, the model-based baseline (`ProjectConfig.model_based_gains`, `ControllerConfig.design_*`). Placing all four poles of the quartic is not possible: its coefficients are coupled through $k_pk_v$ and $k_ik_v$, and for typical targets no positive $k_v$ solves them. The textbook sequential method is used instead:

1. **Inner loop:** the pitch-loop cubic of 3.5 has three gains for three poles. They go at a critically damped pair at $2p = 16.9$ rad/s plus an integral pole at a tenth of that.
2. **Outer loop:** $k_v$ sets the first-order velocity bandwidth $Gk_v$ (6.2) to $z/3 = 2.2$ rad/s.

This gives $k_p = 1.729$, $k_i = 2.012$, $k_d = 0.148$, $k_v = 0.314$: $k_p$ and $k_i$ are close to the tuned values, but $k_d$ and $k_v$ are each about half. By 7.4's analysis (`stability.stable_latency_range`) the sampled, delayed loop is stable from 0 to 23.75 ms. The more aggressive standard choices fail, as 6.3 predicts: with $Gk_v = z/2$ the cascade is unstable even without delay, and an underdamped inner pair ($\zeta = 0.707$, at $z/3$) is unstable at 10 ms. `test_pid_design.py` checks the pole placement, the bandwidth and both stability claims. On the 300 held-out robots (LQR.md) its median cost per scenario is 1.36 times the tuned PID's, but it falls on 1 of them, and its halved $k_d$ and $k_v$ leave less delay margin (23.75 ms against 37.25 ms).

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
| 10 ms | 2.70e-7 | |
| 5 ms | 1.74e-8 | 15.6 |
| 2.5 ms (tuning step) | 1.10e-9 | 15.8 |
| 1.25 ms | 6.92e-11 | 15.9 |
| 0.625 ms | 4.34e-12 | 16.0 |

**Stability.** RK4 is stable for $\lvert\lambda h\rvert$ up to about 2.8 (Hairer, Nørsett & Wanner). The plant's fastest eigenvalue is 8.4 rad/s. With full back-EMF damping it is 20.9 rad/s (when the voltage saturates; 4.4, 9.3). So $\lvert\lambda h\rvert \le 0.052$. The 2.5 ms step is chosen for accuracy and to line up with the timing, not for stability.

**Closed loop.** The clips make the right-hand side non-smooth, which costs some order at the switching instants. Even so, the full scenario at 2.5 ms differs from 0.25 ms by at most $1.7\times10^{-10}$ rad in pitch and $7.8\times10^{-10}$ m/s in velocity for the tuned gains, and $4.3\times10^{-10}$ rad and $2.3\times10^{-9}$ m/s for the hand-tuned ones (7.5 output).

### 7.2 Multi-rate timing and latency (`simulator.py:72-96`)

**Rates.** The physics step is $h = 2.5$ ms, the control period is $T = 5$ ms (2 steps), and the latency is $L = 10$ ms (4 steps). A command computed from the measurement at $t_k$ takes effect at the first physics step at or after $t_k + L$. It is held until the next command arrives, which is a zero-order hold.

**Effective delay.** In general the delay is $\lceil L/h\rceil h$; here that is exact. The hold adds roughly another $T/2$, so the equivalent delay is about 12.5 ms. That is small next to the 118.5 ms unstable time constant ($p\cdot12.5\,\text{ms} = 0.11$), but not negligible (7.4).

**Checked condition.** `SimSpec` and `simulate` refuse a control period that is not a whole multiple of $h$ (since 2026-11-06); a latency between physics steps takes effect at the next step (10.4).

### 7.3 Fall detection (`planar2d.py:129`)

A run ends when $\lvert\theta\rvert > 60°$. This threshold only decides when bookkeeping stops; it does not change the dynamics.

For comparison, setting $\dot\theta = \ddot\theta = 0$ in (E1)–(E2) gives the torque needed to hold an angle:

```math
\tau_{\text{hold}}(\theta) = \frac{aMgl\sin\theta}{a + b(\theta)/r}.
```

The 0.34 N m limit can hold the body up to 53.5°. But holding requires the wheels to keep accelerating, and the motor envelope (4.4) collapses with speed. So the region the robot can actually recover from is far smaller, and it depends on $\dot\theta$ and $v$ as well.

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
| Hand-tuned, pitch-only (4, 0.5, 0.15) | 22.0 ms | 6.4 Hz, damping ratio 0.441 | decays at 11 and 21 ms; 0.051 rad limit cycle at 22 ms, 0.065 rad at 23 ms |
| Hand-tuned cascade | 16.0 ms | 5.2 Hz, damping ratio 0.128 | n/a |
| Tuned cascade | 37.5 ms | 1.4 Hz, damping ratio 0.416 | decays at 18.8 and 36.5 ms; 0.020 rad at 37.5 ms, 0.092 rad at 38.5 ms |
| Tuned 2026-11-06 (earlier robot's gains) | unstable at every latency | 1.7 Hz, damping ratio −0.236 | n/a |

What the table shows:

- **The analysis predicts the nonlinear simulator's stability boundary to within 1 ms.**
- **The rotors buy delay tolerance.** On the robot before 2026-11-08 the margins were 9.5 ms (pitch-only), 11.5 ms (hand-tuned cascade) and 36 ms (that robot's tuned gains). The slower unstable pole (3.2) more than doubles the pitch-only margin. `CONDITIONS.md`'s empirical latency findings (oscillating from 10 ms, falling from 30 ms) were made on that earlier robot.
- **The gains tuned for the earlier robot are unstable here at every latency, even zero.** Their low $k_p$ (2.2 times $k_{p,\text{crit}}$, 3.5) is not enough for the slower, heavier wheel side. That is why they cost about 55 on this robot (8.2).
- **At the configured 10 ms, the hand-tuned cascade is at 63% of its delay margin** and rings at 5.2 Hz with a damping ratio of 0.13. The tuned gains have a 3.75× margin and a damping ratio of 0.42 at 10 ms. The tuner is not given delay margins directly. It gets them because the simulation includes the delay, and robust scoring also randomizes latency over 5–20 ms (8.4), well inside the 37.5 ms margin.

**Delay margin as a function of $k_v$** (same output), which links back to 6.3:

| Inner loop | $k_v$ = 0.1 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Hand-tuned | 16.0 ms | 1.75 ms | unstable | unstable | unstable | unstable | unstable | unstable | unstable | unstable |
| Tuned | 20.75 ms | 23.0 ms | 24.25 ms | 25.5 ms | 28.75 ms | 32.5 ms | 36.75 ms | 40.0 ms | 34.25 ms | 11.0 ms |

For the tuned inner loop the margin rises with $k_v$ up to about 0.7 and falls steeply beyond it (the Routh–Hurwitz ceiling is 0.954, 6.3). The tuned $k_v = 0.620$ sits just below the peak (37.5 ms), not on a flat part.

### 7.5 Accuracy of the cost integrals

The cost (Section 8) is accumulated as a left Riemann sum, $\sum_k f(t_k)\,h$ (`metrics.py:96-101`; `batch_sim/simulate.py:136-141`). On each smooth stretch $[t_a, t_b]$ of the integrand, the Euler–Maclaurin formula gives the error:

```math
\sum_k f(t_k)\,h - \int_{t_a}^{t_b} f\,dt = \frac{h}{2}\bigl(f(t_a) - f(t_b)\bigr) + O(h^2).
```

With the earlier step goals, the velocity term jumped at every goal step. When the robot had reached the previous goal, each jump started a stretch at $f(t_a) \approx \Delta v_{\text{goal}}^2$ that decayed toward 0. So the bias was about

```math
\frac{h}{2}\sum\Delta v_{\text{goal}}^2 = 0.00945 ,
```

and it was observed at +0.42% (tuned) and +0.18% (hand-tuned) at 2.5 ms. The smooth goal removes those jumps (`physics_checks.py` no longer prints this prediction), so the bias is now small (7.5 output, seed 100, 2.86° start):

| | Tuned | Hand-tuned |
|---|---|---|
| Left-sum cost at 2.5 ms against 0.25 ms | +0.01% | +0.00% |
| Trapezoid rule, same comparison | +0.009% | +0.001% |

The trapezoid variant takes the smooth goal at both ends of each step. Holding it over the step, which was right for step goals, adds a first-order error of its own: an earlier version of this check did that and reported −0.60% / −0.11%. The state trajectories agree to $10^{-9}$ for both gain sets, so the integrator is not a source either. With the smooth goal and the slower robot the quadrature bias is about 40 times smaller than with step goals, and the trapezoid rule no longer helps (10.4).

## 8. Cost function and robust scoring (`metrics.py`, `robust.py`)

### 8.1 The cost (`metrics.py:76-105`)

```math
J = \int_0^{t_{\text{end}}}\!\Bigl[(\theta - \theta_{\text{ref}})^2 + w_u\,\tau^2 + w_v\,(\dot x - v_{\text{goal}})^2\Bigr]dt
\;+\; \mathbb 1_{\text{fell}}\;P\Bigl(1 + \frac{t_{\text{dur}} - t_{\text{fall}}}{t_{\text{dur}}}\Bigr).
```

Since 2026-10-24 the parameters are $w_u = 9$, $w_v = 2.25$ and $\theta_{\text{ref}} = 0$ (upright); $P = 1000$ since the 20 s scenario (2026-11-08; 500 before). The weights come from Bryson's rule (10.5) and are derived in `config.py` from three limits. Costs quoted in 8.2 to 9.4 use these values and the smooth goal, except the record table in 8.2, which used the earlier values: $w_u = 0.01$, $w_v = 1$, $P = 100$, $\theta_{\text{ref}}$ = the commanded lean, and a step goal.

- **Units.** The weights carry units, which makes each term comparable to rad² s: $w_u$ is in rad²/(N m)² and $w_v$ in rad²/(m/s)².
- **Why the reference changed (2026-10-24).** Measuring pitch against the commanded lean avoids penalizing the lean the task needs (2.8), but only a controller with a lean command has one. The LQR commands none, so it was scored from upright anyway, and the PID-vs-LQR comparison had to override the setting. Scoring everyone from upright gives one number for all controllers. The price is a floor: the start tilt must be recovered and every goal change needs lean (0.25 rad for the aggressive one), so the pitch term cannot reach zero (on the 12 s scenario the floor was about 0.027 rad² s). The limit $\theta_{\max} = 0.3$ rad sits above that 0.25 rad and equals the largest start tilt, so it does not make the task's own lean an error.
- **Effort.** $\tau$ here is the command after latency. While the motor is unsaturated, $\tau^2$ is proportional to winding loss (4.6).

### 8.2 What the cost actually measures

**Current robot, cost and 20 s scenario (2026-11-08).** Mean over the 30 validation runs, nominal robot, no friction (8 output). The fall penalty is left out of the terms.

| Gains | Pitch term | Effort term | Velocity term | Total |
|---|---|---|---|---|
| Tuned (2026-11-08) | 0.079 (10%) | 0.249 (32%) | 0.454 (58%) | 0.783 |
| Tuned 2026-11-06 (earlier robot's gains) | 0.634 (7%) | 1.564 (18%) | 6.298 (74%) | 8.50 before penalties; all 30 runs fall |
| Hand-tuned | 0.033 (0.5%) | 0.438 (7%) | 5.827 (92.5%) | 6.298 |

So:

- **The tuned gains beat the hand-tuned ones 8 to 1**, mostly on velocity tracking. Copper loss is 2.4 J per 20 s run (hand-tuned 4.3 J).
- **The pitch term includes a floor** that scoring from upright imposes (8.1): the 0.30 rad start tilt and the 0.25 rad lean of the aggressive change.
- **Gains tuned for the earlier robot fall in all 30 runs**, because their loop is unstable here (7.4).
- **The hand-tuned gains pay mostly for velocity.** Their slow outer loop ($k_v = 0.1$) tracks poorly, and the constant pitch offset costs them a 0.1 m/s steady error (5.3).

**12 s scenario, same robot and cost (kept as a record).** "Tuned" here is the set tuned on that scenario ($k_p = 1.567$, $k_i = 2.101$, $k_d = 0.285$, $k_v = 0.645$).

| Gains | Pitch term | Effort term | Velocity term | Total |
|---|---|---|---|---|
| Tuned (2026-11-08, 12 s) | 0.028 (11%) | 0.098 (39%) | 0.129 (50%) | 0.255 |
| Tuned 2026-11-06 (earlier robot's gains) | 2.268 (6%) | 4.904 (13%) | 29.62 (81%) | 36.80 (1 fall; 54.9 with its penalty) |
| Hand-tuned | 0.009 (0.4%) | 0.169 (8%) | 1.904 (91%) | 2.082 |

On that scenario the tuned cost was 1.8 times the earlier robot's (0.255 against 0.140): accelerating the rotors through the gearboxes takes torque. Its pitch term sat at the 0.027 floor.

**Cost of 2026-11-06 on the robot before the datasheet update (kept as a record).** "Tuned 2026-09-30" was tuned for the earlier cost.

| Gains | Pitch term | Effort term | Velocity term | Total |
|---|---|---|---|---|
| Tuned 2026-11-06 | 0.031 (22%) | 0.034 (25%) | 0.075 (53%) | 0.140 |
| Tuned 2026-09-30 | 0.023 (5%) | 0.016 (4%) | 0.391 (91%) | 0.430 |
| Hand-tuned | 0.009 (0.4%) | 0.312 (15%) | 1.809 (85%) | 2.130 |

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

A fall adds between $P$ and $2P$, more the earlier it happens ($P = 1000$ since the 20 s scenario of 2026-11-08; 500 from 2026-10-24; 100 before). That gives the optimizer a slope to climb out of falling gains. "A fall is always worse than any non-fall" holds whenever a non-fall run costs under $P$. Each term of a 20 s run that stays up is bounded:

- the pitch term (from upright), by $(60°)^2 \times 20\,\text{s} = (1.05\ \text{rad})^2 \times 20\,\text{s} = 21.9$;
- the effort term, by $w_u\,\tau_{\max}^2 \times 20\,\text{s} = 9 \times 0.34^2 \times 20 = 20.8$;
- the velocity term only through the motor: the speed error is at most the no-load speed of the fastest robot robust scoring can draw (3.02 m/s: high supply, low $k_t$) plus the largest goal (1.3 m/s), so $2.25 \times 4.32^2 \times 20 = 840$.

The total, 883, is under $P$ (`test_fall_penalty_exceeds_any_non_fall_cost`); for the 12 s scenario it was 462, under its 500. In practice, tuned controllers cost 0.8–2.3 on the 20 s scenario (robust scoring, with friction and pushes, gives the higher end). On the 12 s scenario, 45,000 random-search runs included a slow, wandering controller that never fell and cost up to 186, which is why $P$ was raised from 100 to 500. The 60 s validation runs are not covered by the bound (up to 3 × 883), so validation reports falls as a separate count. **If a cost limit or the scenario changes, re-check this headroom.**

### 8.4 Robust scoring (`robust.py:100-126`)

**Randomization.** Each scenario scales a parameter $p$ by $1 + s\,U(-1, 1)$, with spreads $s$ from `RobustnessSettings`. Three physical consequences:

- **Zero stays zero.** Multiplicative noise cannot perturb a parameter whose nominal value is zero. Friction is 0 nominally, so until 2026-11-06 robust tuning never saw it, and friction turns out to matter (9.4). Since then `RobustnessSettings.randomize_range` draws such parameters outright per scenario: since 2026-11-08 viscous $b \in [0, 0.0015]$ N m s/rad and Coulomb $\tau_c \in [0.02, 0.08]$ N m, from the motors' free-run current (DATASHEET.md 5.2; 0–0.001 and 0–0.02 before). Rotor inertia is scaled by up to ±50% and gear efficiency by ±10%. `latency_range_s` draws each scenario's latency from the whole physics steps in 5–20 ms. The controllers are still designed for the nominal 10 ms.
- **Correlations are lost.** Masses and inertias are drawn independently, but for a fixed shape $I_b \propto M$. Independent draws explore some unphysical combinations (harmless at ±15–25%) and under-sample the correlated extremes.
- **The voltage spread no longer reaches the voltage limit at stall.** At the low-battery, high-resistance corner (6.80 V, 1.44 Ω), $V/R = 4.73$ A, still above the 2 A limit. So stall torque is always current-limited; what varies it is $k_t$ (±10%) and $\eta$ (±10%), between about 0.27 and 0.34 N m (the cap). The voltage now matters only through the corner speed, which stays above about 1.15 m/s even at that corner with $k_t$ at +10%.

**Chaos penalty.** This is $\lvert J(\theta_0 + \delta) - J(\theta_0)\rvert$ with $\delta = 10^{-4}$ rad. Where $J$ depends smoothly on $\theta_0$, it is a finite-difference sensitivity $\approx\lvert\partial J/\partial\theta_0\rvert\,\delta$. Large values flag gains whose runs depend sensitively on the initial condition, typically through repeated saturation switching.

## 9. Checking the assumptions against simulated runs

The derivations rest on assumptions. Each one below is checked against the 30 validation runs (10 noise seeds × 3 start tilts; 9.1–9.5 output).

### 9.1 Rolling without slip

The wheels roll without slipping only if $\lvert F\rvert \le \mu_s N$, with $F$ and $N$ from 2.6.

**Upright and at rest**, at any torque:

```math
\frac{\lvert F\rvert}{N} = \frac{(M + m_w)\,\ddot x_\tau + Ml\,\ddot\theta_\tau}{(M + m_w)\,g}\,\tau = 1.50\ \text{per N m},
```

with $\ddot x_\tau = e/\Delta_0$ and $\ddot\theta_\tau = -c/\Delta_0$ the accelerations per unit torque (3.1). So traction sets the torque limit only when $\mu_s < 1.50\times0.34 = 0.51$. For $\mu_s = 0.8$ traction would allow 0.533 N m, more than the motors give. (On the robot before 2026-11-08 the ratio was 2.71 per N m and traction bound below $\mu_s = 1.08$: the rotors take up some of the torque that would otherwise push on the ground.)

**In the validation runs:**

| Gains | Max $\lvert F\rvert/N$ | Time above 0.6 | Time above 0.8 |
|---|---|---|---|
| Tuned | 0.54 | 0.00% | 0.00% |
| Hand-tuned | 0.68 | 0.04% | 0.00% |

Rubber on hard floors is typically around $\mu \approx 0.6$–1.0; it should be measured. The tuned gains stay below 0.6; the hand-tuned ones exceed it 0.04% of the time. Peak axle accelerations reach 11.6 m/s² (tuned) and 12.7 m/s² (hand-tuned; 36.9 on the earlier robot, whose lighter wheel side let the stiff hand-tuned loop jerk the wheels).

### 9.2 Wheels stay on the ground

The normal force never drops below 7.01 N (tuned) or 5.45 N (hand-tuned), against a static value of 8.18 N, so the wheels never lift. ✓

### 9.3 Motor driver: current mode versus voltage mode

Section 4.3 showed the model is a drive that delivers the commanded current at any speed. That requires either a current-sensing driver, or firmware that adds back-EMF compensation.

A common hobby setup instead sets the PWM duty in proportion to the torque command: $V = R\tau_c/(nNk_t)$, clipped to $\pm V_s$. In that **voltage mode**, below the limits,

```math
\tau = nN\eta k_t\,\frac{V - k_tN\omega}{R} = \tau_c - c_m\,\omega,\qquad c_m = \frac{\eta nN^2k_t^2}{R} = 0.0146\ \text{N m s/rad},
```

which is exactly extra viscous friction, 2.2 times as much as with the guessed 3 Ω motor before 2026-11-08. Simulating it (`VoltageModeActuator` in `physics_checks.py`; 9.3 output) shows a second, larger effect. With the duty proportional to the torque command and the command capped at $\tau_{\max} = 0.34$ N m, the largest voltage the driver ever applies is $R\tau_{\max}/(nN\eta k_t) = 2.52$ V, a third of the 7.6 V supply. At 0.7 m/s the back-EMF is already 2.2 V, so almost no forward torque is left (0.34 − 0.0146 × 20 ≈ 0.05 N m); at the scenario's 1.3 m/s cruise there is none. Letting the duty reach 100% means a command range of 1.025 N m:

| Gains | Current mode (as simulated) | Voltage mode, duty capped at 0.34 N m | Voltage mode, duty up to the full supply |
|---|---|---|---|
| Tuned | 0 of 30 fall, cost 0.783 | 30 of 30 fall | 0 of 30 fall, cost 25.6 |
| Tuned 2026-11-06 (earlier robot's gains) | 30 of 30 fall | 30 of 30 fall | 30 of 30 fall |
| Hand-tuned | 0 of 30 fall, cost 6.30 | 30 of 30 fall | 30 of 30 fall |

The drag alone is below the tuned gains' runaway threshold $rk_pk_v = 0.035$ N m s/rad (9.4), so they survive once the voltage can reach the supply, though their cost rises 33 times. It is above the hand-tuned gains' threshold (0.014 N m s/rad), and at the 1.3 m/s cruise they now fall in every run; on the 12 s scenario, which peaked at 0.71 m/s, they survived at 11 times the cost. On the earlier robot the capped duty reached 6 V of 7.4, and the tuned gains survived it at 9 times the cost.

**This is still the largest unmodeled effect on the tuned gains.** If the hardware uses a voltage-mode driver, the firmware should compensate back-EMF so that the hardware matches the simulation:

```math
V = \frac{R\,\tau_c}{nNk_t} + k_tN\hat\omega .
```

A motor-shaft encoder measures $\hat\omega$ directly, because it reads the rotor angle relative to the stator. The alternative is to simulate the real driver and retune (10.1).

### 9.4 Friction

Nominal friction is 0. Robust scoring draws it per scenario (8.4): since 2026-11-08 viscous $b \in [0, 0.0015]$ N m s/rad and Coulomb $\tau_c \in [0.02, 0.08]$ N m, from the motors' free-run current (DATASHEET.md 5.2). Sensitivity to viscous friction, for one run (seed 100, 2.86° start; 9.4 output):

| $b$ (N m s/rad) | 0 | 0.001 | 0.002 | 0.003 | 0.005 |
|---|---|---|---|---|---|
| Tuned (2026-11-08) | 0.67 | 0.74 | 1.02 | 1.52 | 3.18 |
| Tuned 2026-11-06 (earlier robot's gains) | falls at 3.80 s | falls at 3.59 s | falls at 3.45 s | falls at 3.37 s | falls at 3.45 s |
| Hand-tuned | 5.96 | 6.14 | 6.73 | 7.86 | 12.66 |

The tuned gains never fall, and their cost rises 4.7 times by $b = 0.005$, more than three times the largest viscous friction scored. Their $k_pk_v$ is 1.01 N m per m/s (0.53 for the 2026-11-06 gains, 0.18 for the 2026-09-30 ones) and $k_i$ is 1.77, which is the trade this section predicts below. The 2026-11-06 gains are unstable on this robot (7.4) and fall during the speed-up to 1.3 m/s at every friction level.

**Why the tuned gains are the fragile ones: load torque shifts the speed.** Any steady load torque $\tau_L$ across the wheel–body joint has to be supplied by the motor. Friction, a slope and back-EMF drag are all examples. Until the slow integrator takes the load over, the proportional term must supply it, which needs a pitch error $\theta - \theta_{\text{sp}} = \tau_L/k_p$. At constant speed $\theta = 0$ (2.8), so the lean command must be $\theta_{\text{sp}} = -\tau_L/k_p$. The outer loop only commands that lean when

```math
v - v_{\text{goal}} = \frac{\tau_L}{k_p\,k_v}.
```

This is the same mechanism as the gyro-bias offset in 5.3. Counterintuitively, the robot runs faster than its goal: a backward lean command is the only way the proportional term can produce the forward torque that the load needs.

Viscous friction is a load that grows with speed, $\tau_L = b\,v/r$ (taking $\dot\theta \approx 0$). Substituting gives the quasi-steady speed

```math
v = \frac{v_{\text{goal}}}{1 - b/(r\,k_p k_v)},
```

and no steady speed at all once $b \ge r\,k_pk_v$. That threshold is 0.035 N m s/rad for the tuned gains, 0.018 for the 2026-11-06 ones and 0.014 for the hand-tuned ones (0.0064 for the 2026-09-30 set).

*Check (9.4 output):* with $b = 0.003$, $k_i = 0$, the ideal actuator and a constant 1.4 m/s goal, the formula predicts 1.529 m/s (tuned) and 1.782 m/s (hand-tuned). The simulator settles at exactly those values. (For the 2026-11-06 gains it predicts 1.673 m/s, but on this robot their loop is unstable and the simulated speed is not steady.)

In the real scenario the integrator does pull the speed back. The tuned gains' slowest pitch-loop poles are at $-2.35 \pm 0.90j$ rad/s (3.5), about 0.4 s. With $b = 0.003$, the DC motor's forward headroom $\tau^+_{\max}(\omega) - b\omega$ (4.4) reaches zero at 58.3 rad/s (2.04 m/s); without friction that would only happen at the 2.46 m/s no-load speed. The tuned runs peak at 1.69 m/s (9.1), so at this friction the margin to that point is small. Beyond it the motor cannot drive the wheels forward to catch a forward lean, so the robot would fall forward.

**Measuring friction.** Run the wheels unloaded at a few PWM levels and record the steady speed $\omega_0$ and the current $i_0$ per motor. The friction torque at the output is then $\tau_f(\omega_0) = nNk_t\,i_0$ (no $\eta$: unloaded, the gearbox losses are part of what $i_0$ measures). Fitting $\tau_f = \tau_c + b\,\omega$ separates Coulomb from viscous friction.

### 9.5 The IMU

The model hands the controller pitch plus white noise. A real robot estimates pitch from a gyro and an accelerometer. An accelerometer measures specific force, so during horizontal acceleration its tilt reading is off by about $\arctan(\ddot x/g)$. At the 11.6 m/s² peaks of the tuned runs (9.1), that is up to 50° (52° for the hand-tuned gains, whose peaks reach 12.7 m/s²). The firmware must therefore rely on the gyro at high frequency, through a complementary or Kalman filter. Two consequences:

- **Pitch errors are low-frequency and correlated**, not white.
- **Gyro bias becomes a pitch-estimate offset** of about $b_g\tau_f$ ($\tau_f$ = filter time constant), on top of the rate offset modeled in 5.3. The constant `pitch_bias` (5.1, 5.3) stands in for that and for the IMU's mounting error; it is the same for every run, not drawn per robot.

Mounting the IMU above the axle adds $l_{\text{imu}}\ddot\theta$ and $l_{\text{imu}}\dot\theta^2$ terms on top of this.

### 9.6 Assumptions that are always on

| Assumption | Holds when | What breaks if not |
|---|---|---|
| Rigid body, planar motion, flat level ground | chassis stiff; driving straight | yaw, slopes (add $Mg\sin\alpha$ terms) and bumps are absent |
| Both wheels share torque equally | identical motors and floor contact | any mismatch turns the robot |
| Quasi-static motor current ($L$ ignored) | $L/R \ll 5$ ms | current lags the command; extra phase lag in the loop |
| Stiff gearbox, no backlash (efficiency is modelled) | little play | backlash is a dead zone around zero torque, exactly where balancing operates |
| Rotor inertia only as $\tfrac12J\omega^2$ (2.2) | $N \gg 1$ (20.4 here) | the dropped cross terms with $\dot\theta$ are $1/N$ of the kept ones |
| Coulomb friction smoothed, $\tau_c\tanh(\omega/\omega_\epsilon)$ with $\omega_\epsilon = 1$ rad/s; no rolling resistance | the wheels rarely sit near zero speed for long | real stiction is a dead zone, like backlash; the smoothing understates it near $\omega = 0$ |
| Sensing and actuation aligned to physics steps | latency and control period are multiples of $h$ | see 7.2 |

## 10. Limitations and future iterations

Grouped by how strong the evidence is that each one matters.

### 10.1 Evidence that the tuned gains fail: fix first

| Limitation | Evidence | Next iteration |
|---|---|---|
| Driver assumed to be current-controlled with exact back-EMF compensation | Voltage-mode driver with the duty capped at the torque limit: every gain set falls 30 of 30; with the full supply available the tuned gains survive at 33 times the cost and the hand-tuned ones fall (9.3) | Decide the hardware driver (DATASHEET.md suggests the DRV8874, whose current-sense output allows a firmware current loop). Either add the back-EMF feedforward of 9.3 to the firmware, or move the `VoltageModeActuator` from `physics_checks.py` into `actuators.py` and retune (or randomize between the two). |
| Friction is zero and cannot be randomized | Tuned gains fall at $b \ge 0.003$; the runaway threshold is $rk_pk_v$ (9.4) | *Implemented and retuned 2026-11-06* (`results/batch_tuning/20261106-154750/`; 8.2, 9.4). Both engines model $\tau_f = b\omega + \tau_c\tanh(\omega/\omega_\epsilon)$ (`RobotConfig.wheel_friction`, `coulomb_friction`, `coulomb_friction_speed`). `linearize` includes the slope $\tau_c/\omega_\epsilon$, and robust scoring draws $b$ and $\tau_c$ per scenario (8.4). With $\omega_\epsilon = 1$ rad/s, $\lvert\lambda h\rvert \approx 0.17$ at the largest $\tau_c$ (7.1). Since 2026-11-08 the ranges come from the motors' free-run current (DATASHEET.md 5.2). Still to do: measure friction (9.4). |
| Latency is fixed at 10 ms | Delay margin is the key robustness number (7.4) | *Implemented and retuned 2026-11-06, retuned again on the datasheet robot 2026-11-08.* The current tuned gains' nominal linear loop is stable from 0 to 37.25 ms (`results/lqr_comparison/20261108-225341/`; 7.4). Robust scoring gives each scenario its own latency, uniform over the whole physics steps in 5–20 ms (`RobustnessSettings.latency_range_s`; `simulate_batch(delay_steps=...)`, `TorchSimulator(max_delay_steps=...)`). The alternative, adding the delay margin to the objective, is not done. |

### 10.2 Assumptions violated during transients

| Limitation | Evidence | Next iteration |
|---|---|---|
| No traction limit | $\lvert F\rvert/N$ up to 0.54 / 0.68 (9.1) | Simplest: add a penalty on time spent above a measured $\mu$. Full model: make $\varphi$ a fifth coordinate. While sticking, enforce $\dot x = r\dot\varphi$. When the required $\lvert F\rvert$ exceeds $\mu_sN$, switch to sliding with $F = -\mu_kN\operatorname{sign}(\dot x - r\dot\varphi)$. |
| Idealized IMU | Accelerometer tilt error up to 50° / 52° (9.5) | Simulate the accelerometer (from $\ddot x$, $\ddot\theta$ and the mount height) and the gyro, and run the same complementary filter the firmware will use. |
| Idealized encoders | Velocity is truth plus white noise | Quantize the wheel angle (counts per revolution) and difference it at the control rate, as the firmware will. |

### 10.3 Physical fidelity

- **Reflected rotor inertia.** *Implemented 2026-11-08* (`RobotParams.rotor_inertia`, both engines and the pivot model; derived in 2.2, tested against that mass matrix and for energy conservation). It is the largest inertia on the wheel side (1), and it is estimated, not published: measure it with a spin-down test (DATASHEET.md 8).
- **Winding inductance.** Estimated at $L/R \approx 0.2$–0.5 ms for this motor (4.1). If a measurement shows it is not much less than 5 ms, add the current as a state: $L\,di/dt = V - Ri - k_eN\omega$.
- **Gearbox efficiency and backlash.** *Efficiency implemented 2026-11-08* (`MotorParams.gear_efficiency`, 0.78 from the Pololu stall data): the output torque per amp is $nN\eta k_t$, back-EMF still uses $k_t$, so the loss shows as less torque at the current limit. Backlash is still to do: a dead band in $\varphi - \theta$.
- **Battery sag.** Use $V_s(t) = V_{oc} - R_{\text{batt}}\,i_{\text{total}}$ instead of a constant per run (`CONDITIONS.md` already lists this). With this motor the voltage only matters above the 1.65 m/s corner speed (4.4), so this is low priority.
- **Pitch offset per robot.** `pitch_bias` is one constant for every run (5.1). Drawing it per scenario, with either sign, would keep the tuner from fitting one particular offset.
- **Noise as a density.** Specify sensor noise as a density (rad/s/√Hz, as datasheets do) and convert it with $\sigma = \text{ND}\sqrt{f_s/2}$. Today the per-sample $\sigma$ silently changes the physical noise level if the control rate changes.
- **Scope.** Pushes on the body are *implemented 2026-11-08* (a horizontal force at the center of mass, generalized forces $F$ on $x$ and $Fl\cos\theta$ on $\theta$; `ScenarioConfig.pushes`, both engines, `test_push.py`). Still to do: a 3D model with yaw (differential drive) and slopes, as planned in `ARCHITECTURE.md` and `CONDITIONS.md`.

### 10.4 Numerics and code

- **Cost quadrature.** With the earlier step goals the left-sum bias was 0.42%, and the trapezoid rule (holding the references over each step) cut it to 0.004%. With the smooth goal the left-sum bias is 0.01% and the trapezoid variant checked in 7.5 shows no gain, so this is now low priority. Adding the running cost as a fifth state integrated by RK4 would still make it 4th-order accurate for free.
- **Timing validation.** *Done 2026-11-06* for the control period (7.2). A latency between physics steps still rounds up to the next step, which robust scoring avoids by drawing whole steps.

### 10.5 Cost design

- **Choose the weights deliberately.** *Done 2026-10-24.* The effort term was 0.01% of the cost (8.2). Bryson's rule (Franklin, Powell & Emami-Naeini) sets each weight to $1/(\text{largest acceptable value})^2$; normalized to the pitch term, $w_v = (\theta_{\max}/v_{\max})^2$ and $w_u = (\theta_{\max}/\tau_{\max})^2$. The limits are $\theta_{\max} = 0.3$ rad (from upright), $v_{\max} = 0.2$ m/s and $\tau_{\max} = 0.1$ N m, giving $w_v = 2.25$ and $w_u = 9$. They are stored as `CostConfig`'s `pitch_limit_rad`, `velocity_limit` and `torque_limit`, and the weights are derived from them. $\tau_{\max}$ is a typical operating torque, not the 0.34 N m hardware limit, which would leave effort negligible. $\theta_{\max}$ is a tolerance on the lean error: it must sit above the lean the goal needs (0.25 rad for the aggressive change) and below the 25° lean cap. Result: pitch, effort and velocity are about 20%, 12% and 69% of the cost of the PID tuned on 2026-10-24, 22%, 25% and 53% for the PID retuned with friction and latency randomized, 11%, 39% and 50% for the PID retuned on the datasheet robot, and 10%, 32% and 58% on the 20 s scenario (8.2). The `kv_p` bound was 0.4 and every retune sat on it; it is now 0.9. A run with the bound at 2.0 settled at `kv_p` 0.82 with the same cost (0.1304 against 0.1305), and the re-run at 0.9 settled at 0.822, inside the bound.
- **Use feasible goal profiles.** *Done:* each goal change blends smoothly. The 20 s scenario's aggressive change peaks at 1.78 m/s² (0.25 rad of lean), and every element is sized from a limit of the robot (`ScenarioConfig` docstring). Step-shaped goals made much of the velocity cost unavoidable (8.2).
- **An LQR baseline.** The cost has LQR form, and 3.1 gives $A$ and $B$. Optimal full-state feedback on the linear model, with the same weights, would show how much performance the cascaded-PID structure costs. *Done: see `LQR.md`.* Under the first cost the cost-matched LQR came within 1.8% of the ML-tuned PID with no tuning. On the datasheet robot and the current cost it was 3.7 times worse on the 12 s scenario; on the 20 s scenario with pushes it is 4.2 times worse and falls on 3 of 300 held-out robots, the tuned LQR gains come within 34%, and the PID still wins.
- **A physical fall criterion.** Replace the 60° threshold, or add to it, a check based on recoverability (7.3).

## 11. The batched engine uses the same physics

`batch_sim` implements the same equations in the same order of operations, as functions over arrays of robots:

| Physics | Reference (`balance_sim`) | Batched (`batch_sim`) |
|---|---|---|
| Equations of motion (Section 2) | `planar2d.py:105-131` | `physics.py:41-66` |
| DC motor and ideal actuator (Section 4) | `actuators.py:38, 108-121` | `physics.py:26-38` |
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
