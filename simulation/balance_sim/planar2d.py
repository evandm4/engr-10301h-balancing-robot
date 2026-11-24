"""Planar (2D) wheeled inverted pendulum, derived with Lagrangian mechanics.

Generalized coordinates: x (axle position) and theta (body pitch). The wheels
roll without slipping, so wheel angle = x / r.

    Kinetic energy:
        T = 1/2 (m_w + I_w/r^2 + M) x'^2
            + M l cos(theta) x' theta'
            + 1/2 (M l^2 + I_b) theta'^2
            + 1/2 J (x'/r - theta')^2
    with J = n N^2 J_r the motor rotors reflected through the gearbox: they
    turn at N times the wheel speed relative to the body (PHYSICS.md 10.3).
    Potential energy:
        V = M g l cos(theta)

A motor torque tau acts between wheels and body, so the generalized forces are
Q_x = tau / r and Q_theta = -tau. The Euler-Lagrange equations give

    a x'' + b(theta) theta'' - M l sin(theta) theta'^2 = tau / r
    b(theta) x'' + d theta'' - M g l sin(theta)        = -tau

with a = m_w + I_w/r^2 + M + J/r^2, b = M l cos(theta) - J/r, d = M l^2 + I_b + J.
J is constant, so it adds no velocity terms. The 2x2 system is solved in closed
form for x'' and theta''.

Friction between wheels and body, tau_f = b w + tau_c tanh(w / w_eps) with w
the relative wheel speed, comes off the motor torque (viscous plus smoothed
Coulomb; both zero by default).

An external horizontal force F on the body at its center of mass (a push;
zero unless the simulator is given one) does virtual work F d(x + l sin theta),
so it adds Q_x = F and Q_theta = F l cos(theta) to the generalized forces.

Backlash (free play between motor and wheel; RobotParams.gearbox_backlash and
coupling_backlash, PHYSICS.md 10.3). With play, the motor side (rotor and gear
train: the inertia J, the motor torque and all the friction) gets its own angle,
and the state grows to

    [x, x_dot, theta, theta_dot, gap, motor_speed, shaft, contact]

gap is the motor side's angle minus the wheel's (both relative to the body),
motor_speed the motor side's speed relative to the body, shaft the angle of the
output shaft the encoder reads, and contact the side the play is closed on
(+1 or -1, the motor side pushing the wheel that way) or 0 while it is open.
  * Closed (|gap| = b): the rigid model above.
  * Open (|gap| < b): the wheel and body move as an unactuated pendulum on free
    wheels, with J taken out of the mass matrix, and the motor side alone:
    J motor_speed' = tau_net (motor torque less friction at motor_speed). The
    body feels no reaction, as in the rigid model (the rotor's J theta_dot
    coupling is dropped there too).
  * Closing (|gap| reaches b at the end of a step): an inelastic engagement,
    restitution 0. The generalized momentum of (x, theta) with the motor side
    included is conserved through the impact, and the two then move together.
  * Opening: at the start of each step, the closed contact's transmitted torque
    tau_net - J (x''/r - theta'') is checked; if it would pull instead of push,
    the play opens for that step.
Events are taken at step granularity, so the model keeps the tuning's 2.5 ms
step (a stiff tooth contact needs 0.1 ms). The shaft is massless, held by
friction where it is until a closed play pushes it (a play operator): it stays
within the coupling play of the wheel and the gearbox play of the motor side.

Assumptions: rolling without slip and a rigid body. The motors are handled by
a swappable Actuator: by default the commanded torque is delivered as-is (clipped
to the torque limit); pass a DCMotorActuator for back-EMF and current limits.
These are the first places to look when comparing against hardware.
"""

import math

import numpy as np

from .actuators import Actuator, IdealActuator
from .dynamics import Dynamics
from .params import RobotParams

# State indices with backlash, after the four every model shares (dynamics.py).
GAP, MOTOR_SPEED, SHAFT, CONTACT = 4, 5, 6, 7


def critical_pitch_gain(p: RobotParams) -> float:
    """Smallest proportional pitch gain (N m/rad) that can hold the robot up.

    From the linearized model with no friction or torque limit: below this,
    a pitch-only PID cannot overcome gravity no matter how the other gains
    are set. Useful as a lower bound when the tuner searches for gains.
    The rotor inertia J adds J/r^2 to a and takes J/r off b (= M l at upright),
    so the denominator a + b/r is unchanged and only the numerator grows.
    """
    ml = p.body_mass * p.body_com_height
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2 + p.rotor_inertia / p.wheel_radius**2
    b = ml - p.rotor_inertia / p.wheel_radius
    return a * ml * p.gravity / (a + b / p.wheel_radius)


def linearize(p: RobotParams):
    """The model linearized about upright: d/dt [x, x_dot, theta, theta_dot] = A s + B tau.

    PHYSICS.md 3.1, plus the wheel friction (zero by default), which takes
    b (x_dot / r - theta_dot) off the motor torque. The smoothed Coulomb term
    tau_c tanh(w / w_eps) enters through its slope at zero speed, tau_c / w_eps:
    a large damping, but only within about w_eps of zero. The actuator is taken
    as unsaturated: below its limits the DC motor model delivers the command
    exactly (PHYSICS.md 4.3). Returns A (4, 4) and B (4, 1).
    """
    ml = p.body_mass * p.body_com_height
    a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2 + p.rotor_inertia / p.wheel_radius**2
    b = ml - p.rotor_inertia / p.wheel_radius          # the mass matrix's off-diagonal at theta = 0
    d = p.body_mass * p.body_com_height**2 + p.body_inertia + p.rotor_inertia
    det = a * d - b * b
    # Accelerations per unit of net torque, from the 2x2 system in derivatives().
    x_per_torque = (d / p.wheel_radius + b) / det
    theta_per_torque = -(a + b / p.wheel_radius) / det
    A = np.zeros((4, 4))
    A[0, 1] = A[2, 3] = 1.0
    A[1, 2] = -b * ml * p.gravity / det
    A[3, 2] = a * ml * p.gravity / det
    damping = p.wheel_friction + p.coulomb_friction / p.coulomb_friction_speed
    friction = np.array([-damping / p.wheel_radius, damping])   # net torque per (x_dot, theta_dot)
    A[1, [1, 3]] += x_per_torque * friction
    A[3, [1, 3]] += theta_per_torque * friction
    B = np.array([[0.0], [x_per_torque], [0.0], [theta_per_torque]])
    return A, B


class Planar2D(Dynamics):
    """fall_angle (rad): past this pitch the robot counts as fallen; the
    project's value is TimingConfig.fall_angle_deg in config.py.
    actuator: None means the ideal actuator, clipped to params.max_torque."""

    def __init__(
        self,
        params: RobotParams,
        fall_angle: float,
        actuator: Actuator = None,
    ):
        self.p = params
        self.fall_angle = fall_angle
        self.actuator = actuator if actuator is not None else IdealActuator(params.max_torque)
        # With play the state has 8 entries and the simulator calls begin_step / end_step around each step.
        self.has_backlash = params.backlash > 0

    def drive(self, command: float, measured_state: np.ndarray) -> float:
        """The value the simulator delays and passes to derivatives() as `torque`:
        the actuator's drive signal for this command, from the measured state
        (Actuator.drive). The torque command itself for every actuator but the
        voltage-mode one."""
        if type(self.actuator).drive is Actuator.drive:
            return command
        measured_wheel_speed = float(measured_state[1]) / self.p.wheel_radius - float(measured_state[3])
        return self.actuator.drive(command, measured_wheel_speed)

    def derivatives(self, state: np.ndarray, torque: float, force: float = 0.0) -> np.ndarray:
        """force (N): horizontal push on the body at its center of mass, forward +.
        torque: what drive() returned (the torque command, or a voltage-mode
        actuator's drive voltage). With play, a 4-entry state (no play states) is taken as closed: the
        rigid model, which the closed play reproduces bit for bit."""
        if self.has_backlash and len(state) > 4:
            return self._backlash_terms(state, torque, force)[0]
        p = self.p
        # Plain floats are much faster than numpy scalars in this hot loop.
        x_dot, theta, theta_dot = float(state[1]), float(state[2]), float(state[3])

        relative_wheel_speed = x_dot / p.wheel_radius - theta_dot
        tau = self.actuator.torque(torque, relative_wheel_speed)
        tau -= p.wheel_friction * relative_wheel_speed
        if p.coulomb_friction:
            # np.tanh, not math.tanh: they differ in the last bit, and batch_sim uses NumPy's.
            tau -= p.coulomb_friction * float(np.tanh(relative_wheel_speed / p.coulomb_friction_speed))

        sin_t, cos_t = math.sin(theta), math.cos(theta)
        ml = p.body_mass * p.body_com_height

        a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2 + p.rotor_inertia / p.wheel_radius**2
        b = ml * cos_t - p.rotor_inertia / p.wheel_radius
        d = p.body_mass * p.body_com_height**2 + p.body_inertia + p.rotor_inertia

        rhs_x = tau / p.wheel_radius + ml * sin_t * theta_dot**2
        rhs_t = -tau + ml * p.gravity * sin_t
        if force:   # skipped without a push, so the arithmetic is unchanged (batch_sim does the same)
            rhs_x = rhs_x + force
            rhs_t = rhs_t + force * p.body_com_height * cos_t

        det = a * d - b * b
        x_ddot = (d * rhs_x - b * rhs_t) / det
        theta_ddot = (a * rhs_t - b * rhs_x) / det

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    # ---- backlash (see the module docstring). batch_sim.physics.backlash_* repeat this arithmetic. ----

    def _backlash_terms(self, state: np.ndarray, torque: float, force: float = 0.0):
        """(d(state)/dt, tau_net) with the play closed or open as state[CONTACT] says; tau_net is the
        motor torque less friction, at the motor side's speed (the wheel's while closed)."""
        p = self.p
        x_dot, theta, theta_dot = float(state[1]), float(state[2]), float(state[3])
        motor_speed, contact = float(state[MOTOR_SPEED]), float(state[CONTACT])
        c = abs(contact)            # 1 while the play is closed, 0 while it is open

        wheel_speed = x_dot / p.wheel_radius - theta_dot
        speed = wheel_speed if contact != 0.0 else motor_speed
        tau = self.actuator.torque(torque, speed)
        tau -= p.wheel_friction * speed
        if p.coulomb_friction:
            tau -= p.coulomb_friction * float(np.tanh(speed / p.coulomb_friction_speed))

        sin_t, cos_t = math.sin(theta), math.cos(theta)
        ml = p.body_mass * p.body_com_height
        coupled = c * p.rotor_inertia     # the motor side's inertia, carried by the wheel only while closed
        a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2 + coupled / p.wheel_radius**2
        b = ml * cos_t - coupled / p.wheel_radius
        d = p.body_mass * p.body_com_height**2 + p.body_inertia + coupled

        drive = c * tau                   # what reaches the wheel and body: nothing while open
        rhs_x = drive / p.wheel_radius + ml * sin_t * theta_dot**2
        rhs_t = -drive + ml * p.gravity * sin_t
        if force:
            rhs_x = rhs_x + force
            rhs_t = rhs_t + force * p.body_com_height * cos_t

        det = a * d - b * b
        x_ddot = (d * rhs_x - b * rhs_t) / det
        theta_ddot = (a * rhs_t - b * rhs_x) / det
        if contact != 0.0:
            gap_rate, motor_accel = 0.0, 0.0     # end_step puts the motor side back on the wheel
        else:
            gap_rate, motor_accel = motor_speed - wheel_speed, tau / p.rotor_inertia
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot, gap_rate, motor_accel, 0.0, 0.0]), tau

    def begin_step(self, state: np.ndarray, torque: float, force: float = 0.0) -> np.ndarray:
        """Open a closed play whose transmitted torque, tau_net - J (x''/r - theta''), has reversed.
        The simulator calls it before each step, with that step's drive and push."""
        contact = float(state[CONTACT])
        if contact == 0.0:
            return state
        derivs, tau = self._backlash_terms(state, torque, force)
        wheel_accel = derivs[1] / self.p.wheel_radius - derivs[3]
        transmitted = tau - self.p.rotor_inertia * wheel_accel
        if contact * transmitted < 0.0:
            state = state.copy()
            state[CONTACT] = 0.0
        return state

    def end_step(self, state: np.ndarray) -> np.ndarray:
        """After each step: an inelastic engagement where an open play has closed, the motor side
        held on the wheel while closed, and the encoder's shaft moved where the plays push it."""
        p = self.p
        r, inertia, play = p.wheel_radius, p.rotor_inertia, p.backlash
        x, x_dot, theta, theta_dot, gap, motor_speed, shaft, contact = (float(v) for v in state)
        if contact == 0.0 and abs(gap) >= play:
            contact = math.copysign(1.0, gap)
            # Momentum of (x, theta) with the motor side, before (open mass matrix) = after (closed).
            ml = p.body_mass * p.body_com_height
            a_open = p.body_mass + p.wheel_mass + p.wheel_inertia / r**2
            b_open = ml * math.cos(theta)
            d_open = p.body_mass * p.body_com_height**2 + p.body_inertia
            p_x = a_open * x_dot + b_open * theta_dot + (inertia / r) * motor_speed
            p_t = b_open * x_dot + d_open * theta_dot - inertia * motor_speed
            a = a_open + inertia / r**2
            b = b_open - inertia / r
            d = d_open + inertia
            det = a * d - b * b
            x_dot = (d * p_x - b * p_t) / det
            theta_dot = (a * p_t - b * p_x) / det
        if contact != 0.0:
            gap = contact * play
            motor_speed = x_dot / r - theta_dot
        wheel_angle = x / r - theta
        low = wheel_angle + max(gap - p.gearbox_backlash, -p.coupling_backlash)
        high = wheel_angle + min(gap + p.gearbox_backlash, p.coupling_backlash)
        shaft = min(max(shaft, low), high)
        return np.array([x, x_dot, theta, theta_dot, gap, motor_speed, shaft, contact])

    def step(self, state: np.ndarray, torque: float, dt: float, *push) -> np.ndarray:
        """One physics step with the play's events: begin_step, an RK4 step, end_step (what the simulator does;
        for loops that step the model themselves)."""
        from .simulator import _rk4_step
        state = self.begin_step(state, torque, *push)
        return self.end_step(_rk4_step(self, state, torque, dt, *push))

    def complete_state(self, state) -> np.ndarray:
        """A full state from the four shared entries: with play, the play open and centred, the motor
        side at the wheel's speed, and the shaft on the wheel. Unchanged without play or if complete."""
        state = np.array(state, dtype=float)
        if not self.has_backlash or len(state) > 4:
            return state
        x, x_dot, theta, theta_dot = (float(v) for v in state[:4])
        r = self.p.wheel_radius
        return np.array([x, x_dot, theta, theta_dot, 0.0, x_dot / r - theta_dot, x / r - theta, 0.0])

    def initial_state(self, pitch: float = 0.0) -> np.ndarray:
        return self.complete_state(np.array([0.0, 0.0, pitch, 0.0]))

    def is_fallen(self, state: np.ndarray) -> bool:
        return abs(state[2]) > self.fall_angle

    def critical_pitch_gain(self) -> float:
        """See the module-level critical_pitch_gain."""
        return critical_pitch_gain(self.p)

    def linearize(self):
        """See the module-level linearize."""
        return linearize(self.p)

    def energy(self, state: np.ndarray) -> float:
        """Total mechanical energy (kinetic + potential). Useful for tests."""
        p = self.p
        _, x_dot, theta, theta_dot = state[:4]
        ml = p.body_mass * p.body_com_height
        a = p.body_mass + p.wheel_mass + p.wheel_inertia / p.wheel_radius**2
        d = p.body_mass * p.body_com_height**2 + p.body_inertia
        relative_wheel_speed = x_dot / p.wheel_radius - theta_dot
        kinetic = (
            0.5 * a * x_dot**2
            + ml * np.cos(theta) * x_dot * theta_dot
            + 0.5 * d * theta_dot**2
            + 0.5 * p.rotor_inertia * relative_wheel_speed**2
        )
        potential = ml * p.gravity * np.cos(theta)
        return kinetic + potential
