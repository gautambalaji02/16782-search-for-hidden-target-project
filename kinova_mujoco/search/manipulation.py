"""Pick-and-place of a cover: grasp poses, place poses, and the executed sequence.

Grasp geometry (g_pinch frame: z points along the fingers, y is the closing axis):
  * top-down: z_pinch = world -z
  * the fingers close across the rod: y_pinch = +-(cover's local x), since the rod
    runs along the cover's local y. Both signs are tried (wrist rotated by pi).
  * g_pinch sits GRASP_DEPTH below the rod centre so the rod is between the pads.
Place: same, at a parking spot, with y_pinch = +-world x, so the cover ends up
axis-aligned (its footprint then fits the 10 cm strip outside the search region).

Sequence (execute):
  1 approach   RRT-Connect to the pre-grasp pose (PREGRASP_HEIGHT above the grasp)
  2 descend    straight down to the grasp pose      (contacts with this cover allowed)
  3 grasp      close the gripper, turn the weld on, robot.hold(cover) for planning
  4 lift       straight up to the pre-grasp pose    (table contacts allowed: the cover leaves it)
  5 carry      RRT-Connect to the pre-place pose    (held cover collision-checked)
  6 lower      straight down to the place pose      (table contacts allowed)
  7 release    weld off, robot.release(), open the gripper
  8 retreat    straight up to the pre-place pose    (contacts with this cover allowed)
The camera keeps observing throughout (on_frame), but detection does not interrupt
a pick-and-place half-way: the cover is always put down first.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from search import config as C
from search.motion_planner import path_time, plan_path


def _grasp_R(closing_dir):
    """g_pinch rotation for a top-down grasp whose closing axis is `closing_dir` (horizontal unit)."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.asarray(closing_dir, float)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


@dataclass
class PickPlace:
    cover: str
    spot: tuple
    q_pregrasp: np.ndarray
    q_grasp: np.ndarray
    q_preplace: np.ndarray
    q_place: np.ndarray
    est_time: float                     # s, straight-line estimate of the whole sequence


class Manipulator:
    def __init__(self, model, world, robot, executor, rng):
        self.model, self.world, self.robot, self.ex, self.rng = model, world, robot, executor, rng
        self.rest_rod_z = None          # rod-centre height of a cover resting on the table

    # ------------------------------------------------------------ poses

    def _rod(self, cover):
        p = self.world.site(f"{cover}_grasp").xpos.copy()
        if self.rest_rod_z is None:
            self.rest_rod_z = p[2]
        return p

    def pick_configs(self, cover, q_seed):
        """[(q_pregrasp, q_grasp)] for both wrist orientations that have valid IK."""
        r = self.robot
        rod = self._rod(cover)
        R_cover = self.world.body(cover).xmat.reshape(3, 3)
        out = []
        for sign in (1.0, -1.0):
            Rg = _grasp_R(sign * R_cover[:, 0] * np.array([1, 1, 0]) / np.linalg.norm(R_cover[:2, 0]))
            grasp = rod - [0, 0, C.GRASP_DEPTH]
            r.allow()
            q_pre = r.ik_pinch(grasp + [0, 0, C.PREGRASP_HEIGHT], Rg, q_seed, self.rng)
            if q_pre is None:
                continue
            r.allow([cover])
            q_g = r.ik_pinch(grasp, Rg, q_pre)           # seeded from the pre-grasp: a nearby solution
            r.allow()
            if q_g is not None:
                out.append((q_pre, q_g))
        return out

    def place_configs(self, spot, q_seed):
        """[(q_preplace, q_place)] for both wrist orientations (checks the held cover if one is held)."""
        r = self.robot
        z = (self.rest_rod_z or 0.102) - C.GRASP_DEPTH + C.PLACE_DROP
        place = np.array([spot[0], spot[1], z])
        out = []
        for sign in (1.0, -1.0):
            Rg = _grasp_R([sign, 0.0, 0.0])
            r.allow()
            q_pre = r.ik_pinch(place + [0, 0, C.PREGRASP_HEIGHT], Rg, q_seed, self.rng)
            if q_pre is None:
                continue
            r.allow(["table"])
            q_p = r.ik_pinch(place, Rg, q_pre)
            r.allow()
            if q_p is not None:
                out.append((q_pre, q_p))
        return out

    def best_pick_place(self, cover, spot, q_now):
        """Cheapest (by straight-line time) combination of pick and place configs, or None."""
        best = None
        for q_pre, q_g in self.pick_configs(cover, q_now):
            for q_pp, q_p in self.place_configs(spot, q_pre):
                t = (path_time([q_now, q_pre]) + 2 * path_time([q_pre, q_g])
                     + path_time([q_pre, q_pp]) + 2 * path_time([q_pp, q_p]) + C.TASK_FIXED_TIME)
                if best is None or t < best.est_time:
                    best = PickPlace(cover, tuple(spot), q_pre, q_g, q_pp, q_p, t)
        return best

    # ------------------------------------------------------------ weld

    def _attach(self, cover):
        m, d = self.model, self.world
        eq, g, o = m.equality(f"hold_{cover}").id, m.body("g_base").id, m.body(cover).id
        neg, pos, quat = np.zeros(4), np.zeros(3), np.zeros(4)
        mujoco.mju_negQuat(neg, d.xquat[g])
        mujoco.mju_rotVecQuat(pos, d.xpos[o] - d.xpos[g], neg)
        mujoco.mju_mulQuat(quat, neg, d.xquat[o])
        m.eq_data[eq, 3:6], m.eq_data[eq, 6:10] = pos, quat   # hold the current relative pose
        d.eq_active[eq] = 1

    def _detach(self, cover):
        self.world.eq_active[self.model.equality(f"hold_{cover}").id] = 0

    # ------------------------------------------------------------ execution

    def execute(self, pp, observe, log=print):
        """Run the pick-and-place. observe(data) is called while moving. Returns (ok, message)."""
        r, ex, w = self.robot, self.ex, self.world
        cover = pp.cover

        def move(path, allow=()):
            r.allow(allow)
            res = ex.run(path, on_frame=observe)
            if res.collisions:
                log(f"    contacts during motion: {sorted(res.collisions)}")
            return res

        def current():
            r.sync(w)
            return r.q_of(w)

        # 1 approach
        path = plan_path(r, current(), pp.q_pregrasp, self.rng)
        if path is None:
            return False, "no path to the pre-grasp pose"
        move(path)
        # 2 descend
        q = current()
        r.allow([cover])
        if not r.edge_valid(q, pp.q_grasp):
            r.allow()
            return False, "descent to the grasp pose is blocked"
        move([q, pp.q_grasp], allow=[cover])
        # 3 grasp
        ex.gripper(C.GRIP_CLOSED, on_frame=observe)
        self._attach(cover)
        r.sync(w)
        r.hold(cover, w)
        # 4 lift
        move([current(), pp.q_pregrasp], allow=["table"])
        # 5 carry (re-solve the place configs now that the held cover is collision-checked)
        q = current()
        places = self.place_configs(pp.spot, q)
        if not places:
            return self._abort(cover, "no valid place pose with the cover in hand")
        q_pp, q_p = min(places, key=lambda c: path_time([q, c[0]]))
        path = plan_path(r, q, q_pp, self.rng)
        if path is None:
            return self._abort(cover, "no path to the parking spot")
        move(path)
        # 6 lower
        move([current(), q_p], allow=["table"])
        # 7 release
        self._detach(cover)
        r.release()
        ex.gripper(C.GRIP_OPEN, on_frame=observe)
        # 8 retreat
        move([current(), q_pp], allow=[cover])
        r.allow()
        r.sync(w)
        return True, "ok"

    def _abort(self, cover, msg):
        """Put the cover back down where the arm is (straight down is not guaranteed; just release)."""
        self._detach(cover)
        self.robot.release()
        self.robot.allow()
        self.ex.gripper(C.GRIP_OPEN)
        return False, msg
