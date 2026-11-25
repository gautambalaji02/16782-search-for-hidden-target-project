"""Plays a joint path on the simulated world (the only code that writes the arm's ctrl).

The arm actuators are PD position servos: force = kp*(ctrl - q) - kv*qdot. Sending
just the desired q_d makes the arm lag by about (kv/kp)*speed plus a gravity sag
(measured: up to 0.085 rad), so the executed path cuts corners. The executor adds
feedforward so the servo force equals what the motion needs when q = q_d:
    ctrl = q_d + (kv * qdot_d + qfrc_bias) / kp      (qfrc_bias = gravity + Coriolis)

Each physics step: set ctrl[arm] as above, mj_step, record any invalid
contact or bounds violation (as a safety monitor: if the planner is right these
stay empty), sync the viewer, and wait so sim time keeps pace with wall time.

on_frame(data) is called at frame_hz of sim time while moving (and once at the
end). The sensing/coverage code plugs in here so the camera observes during the
motion, not only at the end. If it returns True (e.g. target detected) the arm
stops and holds its current position.
"""
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from search import config as C
from search.robot import ARM_ACTUATORS
from search.trajectory import Trajectory


@dataclass
class ExecResult:
    completed: bool                      # reached the end (False: stopped by on_frame)
    stopped_by_callback: bool
    sim_time: float                      # s of sim time the motion took
    final_error: float                   # max |q - goal| (rad) at the end
    collisions: set = field(default_factory=set)        # geom-name pairs seen in contact
    out_of_bounds_steps: int = 0


class Executor:
    def __init__(self, model, data, robot, viewer=None, realtime=True, frame_hz=10.0):
        self.model, self.data, self.robot = model, data, robot
        self.viewer, self.realtime, self.frame_dt = viewer, realtime, 1.0 / frame_hz
        self.arm_act = np.array([model.actuator(n).id for n in ARM_ACTUATORS])
        self.kp = model.actuator_gainprm[self.arm_act, 0]
        self.kv = -model.actuator_biasprm[self.arm_act, 2]
        self.grip_act = model.actuator("g_fingers_actuator").id

    def _command(self, q_d, qdot_d):
        """Position command with velocity and gravity/Coriolis feedforward."""
        bias = self.data.qfrc_bias[self.robot.dadr]
        self.data.ctrl[self.arm_act] = q_d + (self.kv * qdot_d + bias) / self.kp

    def _step(self):
        if self.viewer is None:
            mujoco.mj_step(self.model, self.data)
        else:
            with self.viewer.lock():
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def run(self, path, on_frame=None):
        m, d, robot = self.model, self.data, self.robot
        traj = Trajectory(path)
        goal = np.asarray(path[-1], float)
        t0, wall0 = d.time, time.perf_counter()
        next_frame, stopped = 0.0, False
        collisions, oob = set(), 0

        while True:
            t = d.time - t0
            err = np.abs(robot.q_of(d) - goal).max()
            if t >= traj.duration and (err < C.SETTLE_TOL or t >= traj.duration + C.SETTLE_TIMEOUT):
                break
            self._command(traj(t), traj.velocity(t))
            self._step()

            bad = robot.bad_contacts(d)
            if bad:
                collisions.update(robot.contact_names(bad))
            if not robot.in_bounds(d):
                oob += 1

            if on_frame is not None and t >= next_frame:
                next_frame += self.frame_dt
                if on_frame(d):
                    self._command(robot.q_of(d), 0.0)      # hold where we are
                    stopped = True
                    break

            if self.realtime:                              # keep sim time ~ wall time
                lag = (d.time - t0) - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
            if self.viewer is not None and not self.viewer.is_running():
                break

        if on_frame is not None and not stopped:
            on_frame(d)                                    # observe at the final pose
        return self._result(stopped, t0, goal, collisions, oob)

    def gripper(self, value, seconds=C.GRIP_TIME, on_frame=None):
        """Command the gripper (0 open .. 255 closed) and wait `seconds`, holding the arm still."""
        d, robot = self.data, self.robot
        q_hold = robot.q_of(d)
        d.ctrl[self.grip_act] = value
        t0, wall0, next_frame = d.time, time.perf_counter(), 0.0
        collisions, oob = set(), 0
        while d.time - t0 < seconds:
            self._command(q_hold, 0.0)
            self._step()
            bad = robot.bad_contacts(d)
            if bad:
                collisions.update(robot.contact_names(bad))
            if on_frame is not None and d.time - t0 >= next_frame:
                next_frame += self.frame_dt
                on_frame(d)
            if self.realtime:
                lag = (d.time - t0) - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
            if self.viewer is not None and not self.viewer.is_running():
                break
        return self._result(False, t0, q_hold, collisions, oob)

    def _result(self, stopped, t0, goal, collisions, oob):
        d = self.data
        return ExecResult(completed=not stopped, stopped_by_callback=stopped, sim_time=d.time - t0,
                          final_error=float(np.abs(self.robot.q_of(d) - goal).max()),
                          collisions=collisions, out_of_bounds_steps=oob)

