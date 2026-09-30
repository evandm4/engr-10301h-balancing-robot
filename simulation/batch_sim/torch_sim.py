"""The batched simulation on a GPU with PyTorch and CUDA graphs.

Same loop as simulate.py, using the same physics and controller functions
(through the xp shim). What changes is how the loop is driven:

  * Every piece of state lives in a fixed GPU buffer that is updated in
    place, and the step and tick counters are GPU tensors too. So one physics
    step (and one "controller tick + physics step") does exactly the same
    thing on every call and can be recorded once as a CUDA graph.
  * Each graph holds a step's ~200 small kernels. Replaying it costs a single
    launch instead of ~200, which is what makes a Python loop of 2,400 steps
    fast. torch.compile would fuse the kernels as well, but on Windows it
    needs Triton, which isn't installed; CUDA graphs need nothing extra.
  * The per-step decisions the NumPy loop makes in Python (which command has
    reached the motor, the goal velocity, the fall time) become lookups into
    small tables on the GPU, indexed by the step counter.
  * The early stop once every robot has fallen is checked every 100 steps
    rather than every step, since each check waits for the GPU.

A TorchSimulator is built for one batch size and reused: capturing the graphs
takes a moment, and replays are cheap. On a CPU device it runs the same code
without graphs (useful for tests).
"""

from dataclasses import fields
from typing import Optional

import numpy as np

from .controller import cascaded_update, five_gain_columns
from .physics import rk4_step
from .robot import BatchRobot
from .schedule import Schedule, SimSpec, build_schedule
from .simulate import BatchResult
from .xp import torch_ops

EARLY_STOP_CHECK_STEPS = 100


class TorchSimulator:
    def __init__(
        self,
        spec: SimSpec,
        robot: BatchRobot,
        batch_size: int,
        noise: Optional[np.ndarray] = None,
        schedule: Optional[Schedule] = None,
        device="cuda",
        dtype=None,
        use_graphs: Optional[bool] = None,
        record: bool = False,
    ):
        """noise: (n_ticks, 4, n_seeds) table from schedule.noise_table, or None for a perfect sensor.
        dtype: torch.float32 (default; fast) or torch.float64 (for checking against NumPy).
        record: also keep the full state history on the GPU (n_steps x B x 4 values).
        """
        import torch

        self.torch = torch
        self.xp = torch_ops()
        self.spec = spec
        self.schedule = s = build_schedule(spec) if schedule is None else schedule
        self.B = B = batch_size
        self.device = device = torch.device(device)
        self.dtype = dtype = torch.float32 if dtype is None else dtype
        self.use_graphs = device.type == "cuda" if use_graphs is None else use_graphs
        self.record = record
        self.robot = robot.to_torch(device, dtype)
        tensor = lambda a, dt=dtype: torch.as_tensor(np.asarray(a), dtype=dt, device=device)

        # ---- lookup tables, indexed on the GPU by the step / tick counters ----
        R = s.ring_size
        # Ring slot holding the command that reaches the motor at each step.
        # Slot R is an extra row of zeros, for the steps before any command arrives.
        self.slot_per_step = tensor(np.where(s.active_command >= 0, s.active_command % R, R), torch.long)
        self.write_slot_per_tick = tensor(np.arange(s.n_ticks) % R, torch.long)
        self.goal_per_tick = tensor(s.goal_per_tick)
        self.goal_per_step = tensor(s.goal_per_step)
        self.step_end_time = tensor([(k + 1) * spec.dt for k in range(s.n_steps)])   # same floats as the CPU loop
        self.noise = None if noise is None else tensor(np.asarray(noise)[:, 1:, :])   # position noise is never read

        # ---- per-simulation inputs, copied in before each run ----
        zeros = lambda dt=dtype: torch.zeros(B, dtype=dt, device=device)
        self.gains = torch.zeros(5, B, dtype=dtype, device=device)
        self.start_pitch = zeros()
        self.seed_index = zeros(torch.long)

        # ---- state, updated in place every step ----
        self.state = tuple(zeros() for _ in range(4))
        self.pitch_integral, self.velocity_integral, self.setpoint = zeros(), zeros(), zeros()
        self.ring = torch.zeros(R + 1, B, dtype=dtype, device=device)
        self.alive = zeros(torch.bool)
        self.fall_time = zeros()
        self.pitch_error, self.effort, self.velocity_error = zeros(), zeros(), zeros()
        self.step = torch.zeros(1, dtype=torch.long, device=device)
        self.tick = torch.zeros(1, dtype=torch.long, device=device)
        self.history = (torch.zeros(s.n_steps + 1, B, 4, dtype=dtype, device=device) if record else None)

        self._graphs = None

    # ---- one step, written as in-place updates so it can be captured --------

    def _controller_tick(self):
        spec, xp = self.spec, self.xp
        _, x_dot, theta, theta_dot = self.state
        if self.noise is not None:
            n = self.noise[self.tick][0][:, self.seed_index]    # (3, B): velocity, pitch, pitch-rate noise
            x_dot = x_dot + spec.velocity_std * n[0]
            theta = theta + spec.pitch_std * n[1]
            theta_dot = theta_dot + (spec.pitch_rate_std * n[2] + spec.pitch_rate_bias)
        kp, ki, kd, kv_p, kv_i = self.gains
        command, setpoint, pitch_integral, velocity_integral = cascaded_update(
            x_dot, theta, theta_dot, self.goal_per_tick[self.tick], spec.control_dt,
            kp, ki, kd, kv_p, kv_i, self.pitch_integral, self.velocity_integral,
            self.robot.max_torque, spec.max_pitch_command,
            spec.pitch_integral_limit, spec.velocity_integral_limit, xp,
        )
        self.setpoint.copy_(setpoint)
        self.pitch_integral.copy_(pitch_integral)
        self.velocity_integral.copy_(velocity_integral)
        self.ring.index_copy_(0, self.write_slot_per_tick[self.tick], command.unsqueeze(0))
        self.tick.add_(1)

    def _physics_step(self):
        spec, xp, torch = self.spec, self.xp, self.torch
        dt, alive = spec.dt, self.alive
        torque = self.ring[self.slot_per_step[self.step]][0]

        # Cost terms at this sample, only for robots still up.
        x, x_dot, theta, theta_dot = self.state
        pitch_err = theta if spec.pitch_reference == "upright" else theta - self.setpoint
        self.pitch_error.add_(torch.where(alive, pitch_err**2 * dt, 0.0))
        self.effort.add_(torch.where(alive, torque**2 * dt, 0.0))
        self.velocity_error.add_(torch.where(alive, (x_dot - self.goal_per_step[self.step]) ** 2 * dt, 0.0))

        stepped = rk4_step(self.robot, self.state, torque, dt, xp)
        for buffer, new in zip(self.state, stepped):
            buffer.copy_(torch.where(alive, new, buffer))
        just_fell = alive & (torch.abs(self.state[2]) > spec.fall_angle)
        self.fall_time.copy_(torch.where(just_fell, self.step_end_time[self.step], self.fall_time))
        alive.logical_and_(~just_fell)
        if self.record:
            self.history.index_copy_(0, self.step + 1, torch.stack(self.state, dim=-1).unsqueeze(0))
        self.step.add_(1)

    def _control_step(self):
        self._controller_tick()
        self._physics_step()

    # ---- running ---------------------------------------------------------

    def _reset(self):
        for buffer in (*self.state, self.pitch_integral, self.velocity_integral, self.setpoint,
                       self.ring, self.pitch_error, self.effort, self.velocity_error, self.step, self.tick):
            buffer.zero_()
        self.state[2].copy_(self.start_pitch)
        self.alive.fill_(True)
        self.fall_time.fill_(float("nan"))
        if self.record:
            self.history.zero_()
            self.history[0].copy_(self.torch.stack(self.state, dim=-1))

    def _capture(self):
        torch = self.torch
        # Warm up on a side stream first (PyTorch's recommended pattern), then record.
        self._reset()
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                self._control_step()
                self._physics_step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        control, physics = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(control):
            self._control_step()
        with torch.cuda.graph(physics, pool=control.pool()):
            self._physics_step()
        self._graphs = (control, physics)

    def run(self, gains, start_pitch, seed_index=None, robot: Optional[BatchRobot] = None) -> BatchResult:
        """gains (B, 4) [kp, ki, kd, kv_p] or (B, 5) with kv_i; start_pitch scalar or (B,);
        seed_index (B,) column of the noise table.

        robot: new values for the per-simulation robot parameters (the fields
        that were (B,) arrays in the robot this simulator was built with).
        Other fields keep the values it was built with.
        """
        torch, s = self.torch, self.schedule
        gains = five_gain_columns(gains)
        if len(gains) != self.B:
            raise ValueError(f"gains must have shape ({self.B}, 4) or ({self.B}, 5), got {np.shape(gains)}")
        if self.noise is not None and seed_index is None:
            raise ValueError("seed_index is required when noise is given")
        if robot is not None:
            per_sim = [f.name for f in fields(self.robot) if isinstance(getattr(self.robot, f.name), torch.Tensor)]
            given = [f.name for f in fields(robot) if np.ndim(getattr(robot, f.name))]
            if sorted(per_sim) != sorted(given):
                raise ValueError(f"robot must give per-simulation values for exactly {per_sim}, got {given}")
            for name in per_sim:
                getattr(self.robot, name).copy_(torch.as_tensor(np.asarray(getattr(robot, name)), dtype=self.dtype))
        self.gains.copy_(torch.as_tensor(gains.T, dtype=self.dtype))
        self.start_pitch.copy_(torch.as_tensor(np.broadcast_to(start_pitch, (self.B,)).copy(), dtype=self.dtype))
        if seed_index is not None:
            self.seed_index.copy_(torch.as_tensor(np.asarray(seed_index), dtype=torch.long))

        if self.use_graphs and self._graphs is None:
            self._capture()
        self._reset()
        if self.use_graphs:
            control, physics = self._graphs
            step_fns = (control.replay, physics.replay)
        else:
            step_fns = (self._control_step, self._physics_step)
        last = s.n_steps
        for k in range(s.n_steps):
            step_fns[0 if k % s.steps_per_control == 0 else 1]()
            if (k + 1) % EARLY_STOP_CHECK_STEPS == 0 and not bool(self.alive.any()):
                last = k + 1
                break
        if self.record and last < s.n_steps:
            self.history[last + 1:].copy_(self.history[last].expand_as(self.history[last + 1:]))
        return self._result()

    def _result(self) -> BatchResult:
        spec = self.spec
        fell = ~self.alive
        total = self.pitch_error + spec.effort_weight * self.effort + spec.velocity_weight * self.velocity_error
        remaining = (spec.duration - self.fall_time) / spec.duration
        total = self.torch.where(fell, total + spec.fall_penalty * (1.0 + remaining), total)
        out = lambda t: t.cpu().numpy().astype(np.float64)
        return BatchResult(
            cost=out(total),
            pitch_error=out(self.pitch_error),
            effort=out(self.effort),
            velocity_error=out(self.velocity_error),
            fell=fell.cpu().numpy(),
            fall_time=out(self.fall_time),
            final_state=out(self.torch.stack(self.state, dim=-1)),
            states=out(self.history) if self.record else None,
        )
