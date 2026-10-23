# Side project: a servo-driven pivot and a second body

The base robot is one rigid body on two wheels. This side project puts a servo-driven pivot on top of it and a second body piece on the pivot, which gives the robot a third degree of freedom (wheels, lower body, upper body) and a second actuator. It is a separate model next to the base one: nothing in `config.py`, `balance_sim`, `batch_sim`, `tuning` or `batch_tuning` changed. The new code reuses the base code (the DC motor model, sensor noise, the cascaded PID, the cost function, the search methods) by calling it, not by editing it.

```
pivot_config.py                    every new number: body split, servo, pivot sensor, pivot gains, extra cost terms, LQR weights, search bounds
simulation/pivot_sim/
  params.py                        PivotRobotParams (bodies, pivot), ServoParams; combined-COM pitch, rigid equivalent
  servo.py                         ServoActuator: angle command in, torque out
  dynamics.py                      PivotRobot: 3-DOF equations of motion, energy, linearization
  sensors.py                       PivotSensor: base IMU/encoder noise + pivot angle noise
  controllers.py                   PivotCascadedController (PID + pivot loop), PivotLQRController, lqr_gain
  simulator.py                     simulate_pivot: RK4, control rate, latency, servo frame rate
  metrics.py                       pivot_cost, pivot_metrics
  plotting.py                      run plots and an animated side view
simulation/examples/run_pivot_demo.py   held vs level vs LQR vs tuned, printed and plotted (or animated)
simulation/tests/test_pivot.py          physics, servo, timing, controller, config and tuner tests
optimization/pivot_tuning/              PivotEvaluator, for the existing search methods
optimization/run_pivot_tuning.py        tune the 7 gains, compare with the baselines and the LQR
results/pivot_tuning/                   its output
```

Everything shared with the base robot (wheels, wheel motors, IMU noise, latency, control rate, the goal-velocity scenario, start tilts, seeds, the base cost weights, the wheel-gain search bounds) is read from `config.py`'s `CONFIG`, so it can't drift apart. Only what is new is written in `pivot_config.py`.

## The robot

```
          o  upper body: mass m2, COM l2 above the pivot, inertia I2, pitch theta2 = theta1 + phi
          |
          X  pivot: servo, height h above the axle, angle phi (upper relative to lower)
          |
          o  lower body: mass m1, COM l1 above the axle, inertia I1, pitch theta1 (holds the IMU)
          |
         (+) wheels: as the base robot
```

Default numbers (`pivot_config.py`, rough guesses like the rest of the project): the base robot's 0.80 kg body split into a 0.55 kg lower body (battery, electronics, wheel motors, servo; COM 6 cm up) and a 0.25 kg upper body on a pivot 14 cm up (COM 6 cm above the pivot). With the pivot straight the combined COM is 10.4 cm above the axle, close to the base robot's 10 cm, so the two robots are directly comparable. The servo is a standard-size metal-gear hobby servo (MG996R class at 6 V): 1.0 N m stall, 0.15 s per 60°, ±60° range, 50 Hz command rate. Holding the upper body horizontal would take 0.15 N m.

## Equations of motion

Coordinates: axle position x, lower-body pitch θ1, upper-body pitch θ2 = θ1 + φ, both from vertical, forward positive. The wheels roll without slipping (wheel angle x/r).

Positions: lower COM (x + l1 sin θ1, l1 cos θ1); pivot (x + h sin θ1, h cos θ1); upper COM = pivot + l2 (sin θ2, cos θ2). Differentiating and collecting terms:

```
T = 1/2 a x'^2 + b1 cos(θ1) x' θ1' + b2 cos(θ2) x' θ2'
    + 1/2 d1 θ1'^2 + 1/2 d2 θ2'^2 + c cos(θ2 - θ1) θ1' θ2'
V = g (b1 cos θ1 + b2 cos θ2)

a  = m_w + I_w/r^2 + m1 + m2     b1 = m1 l1 + m2 h     b2 = m2 l2
d1 = m1 l1^2 + I1 + m2 h^2       d2 = m2 l2^2 + I2     c  = m2 h l2
```

Generalized forces. The wheel motors apply τw between the wheels and the lower body: Q_x = τw / r, and −τw on θ1 (as in the base robot). The servo applies τs between the lower and upper bodies; its virtual work is τs (δθ2 − δθ1), so +τs on θ2 and −τs on θ1. The Euler–Lagrange equations give M(q) q'' = f with

```
M = | a            b1 cos θ1    b2 cos θ2 |
    | b1 cos θ1    d1           c cos φ   |
    | b2 cos θ2    c cos φ      d2        |

f = | τw/r + b1 sin θ1 θ1'^2 + b2 sin θ2 θ2'^2 |
    | −τw − τs + c sin φ θ2'^2 + g b1 sin θ1   |
    | τs − c sin φ θ1'^2 + g b2 sin θ2         |
```

`PivotRobot.derivatives` solves this 3×3 system in closed form (cofactors) at every call and returns φ'' = θ2'' − θ1''. The state is `[x, x', θ1, θ1', φ, φ']`: the base robot's four states first, so the base sensor model and the base cascaded PID work on `state[:4]` unchanged.

How the derivation is checked (`simulation/tests/test_pivot.py`):

- **Power balance.** Along the dynamics, dE/dt must equal the power the actuators put in, τw (x'/r − θ1') + τs φ'. Checked at hundreds of random states and commands; it holds to about 1e-9. This exercises every entry of M and every Coriolis and gravity term.
- **Reduction to the base robot.** With no upper mass and no servo torque, x'' and θ1'' equal `balance_sim.Planar2D`'s to 1e-10.
- **Static balance.** With the pivot bent, the lower body at `balance_pitch(φ)` (combined COM over the axle) and the servo holding the upper body's weight, every acceleration is zero.
- **Locked pivot.** With the servo holding the pivot straight, the robot follows the rigid equivalent (`PivotRobotParams.locked()`) on the base simulator. Not exactly, because the servo is a stiff spring rather than a weld: the pivot flexes about a degree under hard acceleration, and the COM pitch stays within 0.25° of the rigid robot's.

## The servo

A hobby servo is told an angle, not a torque. Its driver applies a voltage proportional to the angle error, saturating at the supply; the motor's back-EMF makes torque fall linearly with speed (`ServoActuator`):

```
drive  = clip((command − φ) / full_drive_error, −1, 1)
τs     = clip(stall_torque × (drive − φ' / no_load_speed) − friction × φ',  ±stall_torque)
```

With the defaults that's a holding stiffness of 11.5 N m/rad (full torque at 5° of error) and back-EMF damping of 0.14 N m s/rad. The outer clip is the driver's current limit. Commands are clipped to the ±60° range. The servo accepts a new command only once per PWM frame (20 ms at 50 Hz; digital servos go to about 3 ms), on top of the 10 ms control latency both actuators share.

The pivot angle is assumed measurable (a servo with a feedback tap, or a smart servo), with 0.5° and 0.05 rad/s of noise. A plain hobby servo reports nothing. The IMU and encoder noise is the base robot's, drawn from the same stream, so the same seed gives the same noise on those channels as a base-robot run.

Not modeled: deadband, gear backlash, the servo's internal loop rate, reflected rotor inertia (fold it into `upper_inertia`), hard stops beyond the range, supply sag.

## Controlling two actuators

Balance is about the **combined** center of mass: with the upper body free to move, the lower body being upright no longer means the robot is balanced. `PivotRobotParams.com_pitch` computes the angle of the line from the axle to the combined COM, and its rate, from the lower body's IMU and the pivot angle.

### Cascaded PID + pivot loop (`PivotCascadedController`)

- **Wheels:** the base project's `CascadedPIDController`, unchanged, fed the combined-COM pitch instead of a body pitch. Same four gains `[kp, ki, kd, kv_p]`, same lean cap, same anti-windup.
- **Pivot:** a second, PD-style loop

  ```
  φ_cmd = −k_level θ1 + k_phi_p (θ_com − lean command) + k_phi_d θ_com'
  ```

  - `k_level` (0 to 1): how much of the lower body's tilt the upper body cancels. 1 keeps the upper body vertical (a level payload); 0 holds the pivot straight.
  - `k_phi_p`, `k_phi_d`: throw the upper body in response to the COM's pitch error and rate. Swinging the upper body backward moves the COM back toward the axle, but the servo's reaction torque pushes the lower body forward, so whether it helps depends on timing and sign. That's what tuning is for, so both signs are searched.

With the three pivot gains at 0 the pivot is held straight and the robot behaves like the base robot with the same wheel gains. The tuner searches all seven: `[kp, ki, kd, kv_p, k_level, k_phi_p, k_phi_d]`.

### LQR (`PivotLQRController`, `lqr_gain`)

The model-based answer to "what is the best thing to do with both actuators". The model is linearized about upright (`PivotRobot.linearize`, central differences on the real `derivatives`, so the servo's stiffness and damping are included), the axle position is dropped (nothing depends on it), and the integral of velocity error is added for integral action. It is discretized at the 5 ms control period, and the 10 ms latency is built in exactly by adding the two commands already in flight to the state. The quadratic cost mirrors the project's: velocity error 1, wheel torque 0.01, COM pitch 3 (at 1 it leans the robot up to 48° chasing the velocity steps), plus small weights on rates, the pivot angle and the servo command (`LQRConfig`).

`pivot_active=False` gives the same design with the servo input removed (its gain row comes out zero), so "LQR vs LQR (pivot held)" measures what the pivot is worth to an optimal controller.

What the LQR doesn't know about: saturation (wheel torque, back-EMF, servo torque), the servo's 20 ms frame, sensor noise, and nonlinearity at large angles. It is still stable on the full nonlinear simulation with all of them.

### Cost

`pivot_cost` is the base project's cost (`balance_sim.cost`: pitch error, 0.01 × wheel effort, velocity error, fall penalty) applied with the combined-COM pitch, plus `servo_effort_weight` (0.01) × ∫τs² and `upper_tilt_weight` (0 by default) × ∫θ2². The last term is the knob for a different notion of "optimal": a camera or tray on top that should stay level. A run falls when either body tilts past 60°.

One caveat when comparing: the base cost measures pitch error from the lean the controller commands. The LQR commands no lean, so its pitch error is from upright, which is a harsher score. `run_pivot_tuning.py` therefore also reports an "upright" column that scores every controller's validation runs the same way.

## Results so far

From `python optimization/run_pivot_tuning.py --workers 4 --compare-held` (default budget, about 880 evaluations per search, 6 minutes on 4 cores; `results/pivot_tuning/20261001-052500/`). Validation is 30 runs on unseen noise seeds, DC motors, 10 ms latency, 50 Hz servo; no controller fell.

| controller | kp | ki | kd | kv_p | k_level | k_phi_p | k_phi_d | validation | upright |
|---|---|---|---|---|---|---|---|---|---|
| hand-tuned, pivot held | 4.000 | 0.500 | 0.150 | 0.100 | 0 | 0 | 0 | 4.493 | 4.536 |
| hand-tuned, upper body level | 4.000 | 0.500 | 0.150 | 0.100 | 1 | 0 | 0 | 4.497 | 4.538 |
| LQR | | | | | | | | 2.707 | 2.707 |
| LQR, pivot held | | | | | | | | 2.745 | 2.745 |
| DE, pivot active (7 gains) | 0.779 | 0.666 | 0.077 | 0.283 | 0.312 | −0.422 | −0.083 | 2.494 | 2.606 |
| DE, pivot held (4 gains) | 0.761 | 0.274 | 0.059 | 0.243 | 0 | 0 | 0 | 2.475 | 2.574 |

What this says:

1. **The model behaves like the base robot when it should.** The hand-tuned gains with the pivot held score 4.49, the same as the base robot's 4.51 with those gains (`results/batch_tuning/20260930-103458`), and the tuned wheel gains (0.76, 0.27, 0.06, 0.24) land close to the base robot's tuned ones (0.73, 0.24, 0.06, 0.25).
2. **To an optimal controller, the pivot is worth a little: about 1.4%** on this scenario (LQR 2.707 vs 2.745 with the pivot held). The LQR swings the upper body up to about 25° during the velocity steps, which shaves the COM overshoot. The cost here is dominated by velocity tracking, which the wheels' torque and the lean cap limit, so there isn't much for the pivot to win.
3. **The tuned PID + pivot loop didn't find that gain.** At equal budget, the 7-gain search (2.494) came out slightly behind the 4-gain search with the pivot held (2.475): three extra dimensions cost more search effort than the pivot loop gives back. A lean feed-forward on the pivot (upper body thrown toward the commanded lean) was also tried by hand and made things worse (2.52 → 2.59 at a gain of 1, falls at 2).
4. **Keeping the upper body level is nearly free.** `k_level = 1` cuts the upper body's RMS tilt from 4.0° to 1.5° for a 0.1% change in cost.
5. **The tuned PID beats the LQR, even scored the LQR's way** (2.57 vs 2.71 from upright). The LQR doesn't know about saturation or the lean cap, leans the COM to about 43° at the big velocity reversal, and its higher gains turn sensor noise into a lot of wheel-torque chatter (the torque panel in `tuning_summary.png`).

## Commands

From the project folder:

```
python -m pytest simulation/tests/test_pivot.py                       # the pivot tests (a few seconds)
python simulation/examples/run_pivot_demo.py                          # held / level / LQR / tuned, plotted
python simulation/examples/run_pivot_demo.py --animate                # animated side view
python simulation/examples/run_pivot_demo.py --save pivot.gif
python simulation/examples/run_pivot_demo.py --gains 0.78 0.67 0.08 0.28 0.31 -0.42 -0.08
python optimization/run_pivot_tuning.py --workers -1 --compare-held   # tune the 7 gains, ~6 min on 4 cores
```

## Next steps

- **Give the tuner a structure that can use the pivot.** The LQR's servo row is a reasonable template: it uses lower-body tilt, velocity error and the pivot's own angle. Alternatively let the tuner search the LQR weights instead of PID gains (a few numbers, always stable designs).
- **A scenario where the pivot should matter more:** external pushes, a payload that must stay level (`upper_tilt_weight`), a heavier upper body, or a hard stop where the robot must shed speed fast.
- **Servo realism:** deadband and backlash (both make small corrections jittery), a plain servo with no position feedback (the controller only knows what it commanded), a digital servo's faster frame rate.
- **Batched version:** the equations are the same shape as `batch_sim`'s and would vectorize the same way, for the robust 50-seed, randomized-robot scoring.
