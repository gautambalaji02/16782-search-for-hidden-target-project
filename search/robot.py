"""Planner-side robot interface: FK, validity checks, IK. Never steps physics.

The Robot owns a *scratch* MjData. Planners set joint angles on it and call
kinematics/collision only, so planning never disturbs the simulated world.
Call sync(world_data) before planning so the scratch copy has the current
object poses (covers can move between plans).

Validity of a configuration q:
  * inside the joint limits (actuator ctrlrange, intersected with joint ranges)
  * no contact between a robot geom and anything else (table, covers, obstacles),
    and no robot self-contact. Contacts with ignored bodies (the target, which the
    planner does not know about) are skipped. Object-object contacts (a cover
    resting on the table) are irrelevant and skipped.
  * g_pinch and the camera are inside EE_BOUNDS.

Safety margins (the real arm lags and cuts corners a little while moving):
  * The Robot plans on its own copy of the model in which robot geoms have
    geom_margin = COLLISION_MARGIN, so contacts appear within that distance and
    the planner keeps that much clearance. The simulated world keeps margin 0.
  * Planning uses EE_BOUNDS shrunk by BOUNDS_MARGIN.
bad_contacts()/in_bounds() with margin 0 also serve as the executor's monitor on the world data.

Manipulation (used by manipulation.py):
  * allow(bodies): contacts with these bodies are not counted (e.g. the cover the
    gripper is descending onto, or the table while a held cover is lowered onto it).
  * hold(cover, world): from now on the cover moves rigidly with g_base in the
    scratch data (the same offset the weld holds in the world), and its contacts with
    anything but the robot count as collisions. For clearance its geoms are INFLATED by
    COLLISION_MARGIN in the planner's model (not given a geom_margin: MuJoCo's box-box
    collider ignores margins, and the cover walls and obstacles are all boxes).
"""
import copy

import mujoco
import numpy as np

from search import config as C

ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]
ARM_ACTUATORS = [f"act_joint_{i}" for i in range(1, 7)]


def look_axis(cam_pos, point):
    """Unit optical axis for a camera at cam_pos looking at point."""
    v = np.asarray(point, float) - cam_pos
    return v / np.linalg.norm(v)


class Robot:
    def __init__(self, model, camera="wrist_rgb", ignore_bodies=("target",)):
        m = self.model = copy.deepcopy(model)            # planner's copy (collision margins below)
        self.data = mujoco.MjData(m)                     # scratch, planner only

        joints = [m.joint(n) for n in ARM_JOINTS]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        self.lo = np.array([m.actuator(n).ctrlrange[0] for n in ARM_ACTUATORS])
        self.hi = np.array([m.actuator(n).ctrlrange[1] for n in ARM_ACTUATORS])
        for i, j in enumerate(joints):                   # joint range is tighter where it exists
            if m.jnt_limited[j.id]:
                self.lo[i] = max(self.lo[i], j.range[0])
                self.hi[i] = min(self.hi[i], j.range[1])

        self.cam = m.camera(camera).id
        self.cam_body = m.cam_bodyid[self.cam]
        self.pinch = m.site("g_pinch").id
        self.gbase = m.body("g_base").id

        robot_root = m.body_rootid[self.gbase]
        self.robot_geom = m.body_rootid[m.geom_bodyid] == robot_root     # bool per geom
        in_gripper = np.zeros(m.nbody, bool)                                 # g_base and its subtree
        for b in range(m.nbody):
            k = b
            while k > 0 and k != self.gbase:
                k = m.body_parentid[k]
            in_gripper[b] = k == self.gbase
        self.gripper_geom = in_gripper[m.geom_bodyid]
        self.ignored_geom = np.zeros(m.ngeom, bool)
        for name in ignore_bodies:
            self.ignored_geom |= m.geom_bodyid == m.body(name).id
        self.allowed_geom = np.zeros(m.ngeom, bool)
        self.held_geom = np.zeros(m.ngeom, bool)
        self.held = None                                 # (qposadr, rel_pos, rel_quat) of a held cover
        m.geom_margin[self.robot_geom] = C.COLLISION_MARGIN

        self.bmin = np.array([C.EE_BOUNDS[k][0] for k in "xyz"])
        self.bmax = np.array([C.EE_BOUNDS[k][1] for k in "xyz"])

    # ------------------------------------------------------------ state

    def sync(self, world):
        """Copy the world's full state (arm, gripper, object poses) into the scratch data."""
        self.data.qpos[:] = world.qpos
        self.data.eq_active[:] = world.eq_active
        self._kin()

    def q_of(self, data):
        return data.qpos[self.qadr].copy()

    def random_q(self, rng):
        return rng.uniform(self.lo, self.hi)

    def _set(self, q):
        self.data.qpos[self.qadr] = q

    def _kin(self):
        """Body, site and camera poses for the scratch qpos (cam_xpos is set by mj_camlight)."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        if self.held is not None:                        # carry the held cover with the gripper
            adr, rel_pos, rel_quat = self.held
            d.qpos[adr:adr + 3] = d.xpos[self.gbase] + d.xmat[self.gbase].reshape(3, 3) @ rel_pos
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, d.xquat[self.gbase], rel_quat)
            d.qpos[adr + 3:adr + 7] = q
            mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)

    # ------------------------------------------------------------ manipulation state

    def allow(self, bodies=()):
        """Ignore contacts with these bodies (replaces the previous set; allow() clears it)."""
        self.allowed_geom[:] = False
        for name in bodies:
            self.allowed_geom |= self.model.geom_bodyid == self.model.body(name).id

    def hold(self, body, world):
        """Attach `body` (a cover with a free joint) to g_base at its current offset in `world`."""
        b = self.model.body(body).id
        R = world.xmat[self.gbase].reshape(3, 3)
        rel_pos = R.T @ (world.xpos[b] - world.xpos[self.gbase])
        neg, rel_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(neg, world.xquat[self.gbase])
        mujoco.mju_mulQuat(rel_quat, neg, world.xquat[b])
        adr = self.model.jnt_qposadr[self.model.body_jntadr[b]]
        self.held = (adr, rel_pos, rel_quat)
        self.held_geom = self.model.geom_bodyid == b
        self.held_body = b
        self._inflate(self.held_geom, C.COLLISION_MARGIN)

    def release(self):
        if self.held is not None:
            self._inflate(self.held_geom, -C.COLLISION_MARGIN)
        self.held = None
        self.held_geom = np.zeros(self.model.ngeom, bool)

    def _inflate(self, mask, d):
        """Grow (d > 0) or shrink geoms by d in every direction, with their bounding volumes."""
        m = self.model
        for g in np.flatnonzero(mask):
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                m.geom_size[g, :3] += d
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER or t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                m.geom_size[g, :2] += d
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                m.geom_size[g, 0] += d
            else:
                continue
            m.geom_aabb[g, 3:6] += d                     # broad-phase bounds must grow too
            m.geom_rbound[g] += d * np.sqrt(3)
        for b in np.unique(m.geom_bodyid[mask]):         # and the body's bounding-volume tree
            a = m.body_bvhadr[b]                         # (MuJoCo culls body pairs with it first)
            if a >= 0:
                m.bvh_aabb[a:a + m.body_bvhnum[b], 3:6] += d

    # ------------------------------------------------------------ kinematics

    def fk(self, q):
        """(camera position, camera optical axis, g_pinch position) at q, world frame."""
        self._set(q)
        self._kin()
        return self.cam_pose(self.data) + (self.data.site_xpos[self.pinch].copy(),)

    def cam_frame(self, q):
        """Camera position and 3x3 rotation (columns: camera x right, y up, z backward) at q."""
        self._set(q)
        self._kin()
        return self.data.cam_xpos[self.cam].copy(), self.data.cam_xmat[self.cam].reshape(3, 3).copy()

    def pinch_frame(self, q):
        """g_pinch position and rotation (z: along the fingers, y: closing axis) at q."""
        self._set(q)
        self._kin()
        return self.data.site_xpos[self.pinch].copy(), self.data.site_xmat[self.pinch].reshape(3, 3).copy()

    def cam_pose(self, data):
        """Camera position and optical axis (MuJoCo cameras look along their -z)."""
        R = data.cam_xmat[self.cam].reshape(3, 3)
        return data.cam_xpos[self.cam].copy(), -R[:, 2].copy()

    # ------------------------------------------------------------ validity

    def bad_contacts(self, data):
        """Geom-id pairs of contacts that make a configuration invalid (works on any MjData).

        The arm and a held cover move together ("moving" geoms). A contact is bad if it involves
        a moving geom and is not with an ignored/allowed body, except:
          * robot vs held cover (the gripper is holding it) -> fine
          * robot vs robot -> only if they really penetrate (gripper parts sit within the margin)
          * gripper vs gripper -> fine: closed on a thin rod the fingertip pads meet below it
        """
        if data.ncon == 0:
            return []
        g = data.contact.geom[:data.ncon]                # (ncon, 2)
        dist = data.contact.dist[:data.ncon]
        r, h = self.robot_geom[g], self.held_geom[g]
        skip = (self.ignored_geom | self.allowed_geom)[g].any(axis=1)
        robot_held = (r[:, 0] & h[:, 1]) | (h[:, 0] & r[:, 1]) | h.all(axis=1)
        moving = (r | h).any(axis=1)
        finger_self = self.gripper_geom[g].all(axis=1)
        bad = moving & ~skip & ~robot_held & ~finger_self & (~r.all(axis=1) | (dist < 0))
        return [tuple(p) for p in g[bad]]

    def in_bounds(self, data, margin=0.0):
        pts = (data.site_xpos[self.pinch], data.cam_xpos[self.cam])
        return all(np.all(p >= self.bmin + margin) and np.all(p <= self.bmax - margin) for p in pts)

    def is_valid(self, q):
        if np.any(q < self.lo) or np.any(q > self.hi):
            return False
        self._set(q)
        self._kin()
        if not self.in_bounds(self.data, C.BOUNDS_MARGIN):   # cheap test first
            return False
        mujoco.mj_collision(self.model, self.data)
        return not self.bad_contacts(self.data)

    def edge_valid(self, q1, q2, step=None):
        """Straight joint-space edge, checked every `step` rad of the largest joint change (q1 not rechecked).

        Default step: EDGE_STEP, or the finer EDGE_STEP_HELD while holding a cover (its far corners
        sweep ~0.6 m from the base, so 0.03 rad steps could jump over a thin collision).
        """
        if step is None:
            step = C.EDGE_STEP_HELD if self.held is not None else C.EDGE_STEP
        dq = np.asarray(q2) - q1
        n = max(1, int(np.ceil(np.abs(dq).max() / step)))
        return all(self.is_valid(q1 + dq * k / n) for k in range(1, n + 1))

    def path_valid(self, path):
        return all(self.edge_valid(a, b) for a, b in zip(path[:-1], path[1:]))

    def contact_names(self, pairs):
        name = lambda g: self.model.geom(g).name or f"geom{g}({self.model.body(self.model.geom_bodyid[g]).name})"
        return [(name(a), name(b)) for a, b in pairs]

    # ------------------------------------------------------------ inverse kinematics

    def ik_camera(self, pos, axis, q_init, rng=None, restarts=5):
        """Joint angles putting the camera at `pos` looking along `axis` (roll left free).

        Damped least squares on a 5-D task: 3 for position, 2 for the axis direction
        (the angular rows are projected so rotation about the optical axis costs nothing).
        Tries q_init first, then `restarts` random seeds. Returns a *valid* q or None.
        """
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        seeds = [np.asarray(q_init, float)]
        if rng is not None:
            seeds += [self.random_q(rng) for _ in range(restarts)]
        for q0 in seeds:
            q = self._ik_from(q0, np.asarray(pos, float), axis)
            if q is not None and self.is_valid(q):
                return q
        return None

    def _ik_from(self, q, pos, axis):
        m, d = self.model, self.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lam2 = C.IK_DAMPING ** 2
        q = q.copy()
        for _ in range(C.IK_MAX_ITERS):
            self._set(q)
            self._kin()
            mujoco.mj_comPos(m, d)                       # needed by mj_jac
            p, a = self.cam_pose(d)
            ep, ea = pos - p, np.cross(a, axis)
            if np.linalg.norm(ep) < C.IK_POS_TOL and np.arccos(np.clip(a @ axis, -1, 1)) < C.IK_ANGLE_TOL:
                return q
            mujoco.mj_jac(m, d, jacp, jacr, p, self.cam_body)
            P = np.eye(3) - np.outer(a, a)               # drop roll about the optical axis
            J = np.vstack([jacp[:, self.dadr], P @ jacr[:, self.dadr]])
            e = np.concatenate([ep, ea])
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            dq *= min(1.0, C.IK_MAX_STEP / (np.abs(dq).max() + 1e-12))
            q = np.clip(q + dq, self.lo, self.hi)
        return None

    def ik_pinch(self, pos, R_goal, q_init, rng=None, restarts=5, check=True):
        """Joint angles putting g_pinch at `pos` with orientation `R_goal` (full 6-D).

        Same damped least squares, with the orientation error
        e_rot = 1/2 * sum_i (R[:, i] x R_goal[:, i])   (world frame, matches the rotational Jacobian).
        Returns a q (valid, if check=True) or None.
        """
        seeds = [np.asarray(q_init, float)]
        if rng is not None:
            seeds += [self.random_q(rng) for _ in range(restarts)]
        for q0 in seeds:
            q = self._ik_pinch_from(q0, np.asarray(pos, float), np.asarray(R_goal, float))
            if q is not None and (not check or self.is_valid(q)):
                return q
        return None

    def _ik_pinch_from(self, q, pos, Rg):
        m, d = self.model, self.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lam2 = C.IK_DAMPING ** 2
        q = q.copy()
        for _ in range(C.IK_MAX_ITERS):
            self._set(q)
            self._kin()
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.pinch]
            R = d.site_xmat[self.pinch].reshape(3, 3)
            ep = pos - p
            er = 0.5 * np.cross(R, Rg, axisa=0, axisb=0).sum(0)
            ang = np.arccos(np.clip((np.trace(R.T @ Rg) - 1) / 2, -1, 1))
            if np.linalg.norm(ep) < C.IK6_POS_TOL and ang < C.IK6_ANGLE_TOL:
                return q
            mujoco.mj_jacSite(m, d, jacp, jacr, self.pinch)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            dq *= min(1.0, C.IK_MAX_STEP / (np.abs(dq).max() + 1e-12))
            q = np.clip(q + dq, self.lo, self.hi)
        return None
