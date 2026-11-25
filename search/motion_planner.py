"""Joint-space RRT-Connect (Kuffner & LaValle, 2000) with shortcut smoothing.

Used for point-to-point moves during manipulation (go to a cover, carry it to a
parking spot), where the straight joint-space edge is often blocked.
All validity checks go through Robot.edge_valid, so whatever the Robot is set
up to allow or hold (robot.allow / robot.hold) is respected.

  1. If the straight edge start -> goal is valid, return it.
  2. Grow one tree from each end. Each iteration: sample q_rand, EXTEND tree A
     one step (<= RRTC_STEP rad in every joint) toward it, then CONNECT tree B
     toward the new node (repeat steps until blocked or reached). Swap trees.
  3. When the trees meet, join the two branches, then shortcut: repeatedly pick
     two waypoints and drop everything between them if the straight edge is valid.
Samples are drawn in the box spanned by start and goal, grown by RRTC_SAMPLE_PAD
(and clipped to the joint limits): the joints with +-2pi ranges otherwise waste
most samples on wrapped-around configurations.
"""
import numpy as np

from search import config as C


class _Tree:
    def __init__(self, root):
        self.q = [np.asarray(root, float)]
        self.parent = [-1]

    def nearest(self, q):
        return int(np.argmin(np.abs(np.array(self.q) - q).max(1)))

    def add(self, q, parent):
        self.q.append(q)
        self.parent.append(parent)
        return len(self.q) - 1

    def branch(self, k):
        out = []
        while k != -1:
            out.append(self.q[k])
            k = self.parent[k]
        return out                                       # node k ... root


def _steer(q_from, q_to, step):
    d = q_to - q_from
    m = np.abs(d).max()
    return q_to.copy() if m <= step else q_from + d * (step / m)


def _extend(robot, tree, q_target, step):
    """One step from the nearest node toward q_target. Returns (new index or None, reached)."""
    k = tree.nearest(q_target)
    q_new = _steer(tree.q[k], q_target, step)
    if not robot.edge_valid(tree.q[k], q_new):
        return None, False
    return tree.add(q_new, k), np.allclose(q_new, q_target)


def _connect(robot, tree, q_target, step):
    """Extend repeatedly toward q_target. Returns (last index or None, reached)."""
    last = None
    while True:
        k, reached = _extend(robot, tree, q_target, step)
        if k is None:
            return last, False
        last = k
        if reached:
            return k, True


def shortcut(robot, path, rng, iters=C.SHORTCUT_ITERS):
    path = list(path)
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i > 1 and robot.edge_valid(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def plan_path(robot, q_start, q_goal, rng, max_iters=C.RRTC_MAX_ITERS, step=C.RRTC_STEP):
    """Collision-free joint path [q_start, ..., q_goal], or None if none was found."""
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    if robot.edge_valid(q_start, q_goal):
        return [q_start, q_goal]
    lo = np.maximum(robot.lo, np.minimum(q_start, q_goal) - C.RRTC_SAMPLE_PAD)
    hi = np.minimum(robot.hi, np.maximum(q_start, q_goal) + C.RRTC_SAMPLE_PAD)

    a, b = _Tree(q_start), _Tree(q_goal)
    a_is_start = True
    for _ in range(max_iters):
        q_rand = rng.uniform(lo, hi)
        ka, _ = _extend(robot, a, q_rand, step)
        if ka is not None:
            kb, reached = _connect(robot, b, a.q[ka], step)
            if reached:
                from_a = a.branch(ka)[::-1]              # root_a ... meeting point
                from_b = b.branch(kb)[1:]                # (meeting point excluded) ... root_b
                path = from_a + from_b
                if not a_is_start:
                    path = path[::-1]
                return shortcut(robot, path, rng)
        a, b = b, a
        a_is_start = not a_is_start
    return None


def path_time(path):
    """Execution-time estimate: sum over edges of max joint change / JOINT_VMAX."""
    return float(sum(np.abs(np.asarray(q2) - q1).max() for q1, q2 in zip(path[:-1], path[1:])) / C.JOINT_VMAX)
