"""Information-gain RRT for choosing the next views (after Bircher et al., ICRA 2016).

One call to plan(q_root) grows a tree from the current configuration and returns
the path to the most valuable node; the caller executes it, observes, and calls
plan() again (receding horizon: the tree is rebuilt from the new state each time).

Growing the tree (sampling in task space, as discussed):
  1. Pick a random POSSIBLE cell and sample a camera view looking at it
     (viewpoints.random_view: position offset + IK, seeded from the root).
  2. Connect it to the cheapest of its IG_K_NEAR nearest nodes whose straight
     joint-space edge is valid (collision + bounds, Robot.edge_valid).
  3. Score it with the visibility model (gain.py):
        new(n)   = cells visible from n that no node on its branch already sees
        gain(n)  = |new(n)|
        value(n) = value(parent) + gain(n) * exp(-IG_LAMBDA * cost(n))
     cost(n) is the time to reach n along the branch (sum over edges of
     max_i |dq_i| / JOINT_VMAX), so far-away gains count less.
The best node maximises value. If the cells its whole branch would reveal are
fewer than IG_MIN_GAIN, the planner reports "stuck": no motion gives information
(what is left is under covers or permanently occluded) -> lift a cover.

Gain counts only views at nodes, not what the camera sees along the way; the
executor still observes during motion, so the real gain is usually higher.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from search import config as C
from search.coverage import POSSIBLE
from search.viewpoints import random_view


@dataclass
class Plan:
    path: list                          # joint configs, path[0] = current
    predicted_cells: np.ndarray         # flat cell indices the branch is expected to reveal
    stuck: bool = False
    tree_cam_xy: np.ndarray = None      # (N, 2) camera xy of every node (for drawing)
    tree_parent: np.ndarray = None      # (N,) parent index, -1 for the root
    branch: list = field(default_factory=list)   # node indices root -> best
    n_nodes: int = 0
    plan_time: float = 0.0


@dataclass
class _Node:
    q: np.ndarray
    parent: int
    cost: float                         # s to reach from the root
    seen: np.ndarray                    # bool over candidate cells, seen by this branch
    gain: int
    value: float
    cam_xy: np.ndarray


def edge_time(q1, q2):
    return float(np.abs(np.asarray(q2) - q1).max() / C.JOINT_VMAX)


class InfoGainRRT:
    def __init__(self, robot, grid, views, rng):
        self.robot, self.grid, self.views, self.rng = robot, grid, views, rng

    def plan(self, q_root):
        t0 = time.perf_counter()
        robot, grid = self.robot, self.grid
        cand = np.flatnonzero(grid.state.ravel() == POSSIBLE)       # cells that could still give gain
        cand_xy = grid.centers.reshape(-1, 2)[cand]

        pos, R = robot.cam_frame(q_root)
        root_seen = self.views.visible(pos, R, cand)                # already in view: no credit
        nodes = [_Node(np.asarray(q_root, float), -1, 0.0, root_seen, 0, 0.0, pos[:2])]

        for _ in range(C.IG_MAX_SAMPLES):
            if len(nodes) > C.IG_NODES or len(cand) == 0:
                break
            look = cand_xy[self.rng.integers(len(cand))]
            q = random_view(robot, q_root, self.rng, look_at=look)
            if q is None:
                continue
            parent = self._best_parent(nodes, q)
            if parent is None:
                continue
            p = nodes[parent]
            cost = p.cost + edge_time(p.q, q)
            pos, R = robot.cam_frame(q)
            vis = self.views.visible(pos, R, cand)
            new = vis & ~p.seen
            gain = int(new.sum())
            nodes.append(_Node(q, parent, cost, p.seen | vis, gain,
                               p.value + gain * np.exp(-C.IG_LAMBDA * cost), pos[:2]))

        best = int(np.argmax([n.value for n in nodes]))
        branch = []
        k = best
        while k != -1:
            branch.append(k)
            k = nodes[k].parent
        branch.reverse()
        revealed = nodes[best].seen & ~root_seen
        stuck = best == 0 or revealed.sum() < C.IG_MIN_GAIN

        return Plan(path=[nodes[k].q for k in branch], predicted_cells=cand[revealed], stuck=stuck,
                    tree_cam_xy=np.array([n.cam_xy for n in nodes]),
                    tree_parent=np.array([n.parent for n in nodes]), branch=branch,
                    n_nodes=len(nodes), plan_time=time.perf_counter() - t0)

    def _best_parent(self, nodes, q):
        """Cheapest-to-reach node among the k nearest (by edge time) with a valid straight edge."""
        Q = np.array([n.q for n in nodes])
        d = np.abs(Q - q).max(1) / C.JOINT_VMAX
        near = np.argsort(d)[:C.IG_K_NEAR]
        for k in sorted(near, key=lambda k: nodes[k].cost + d[k]):
            if self.robot.edge_valid(nodes[k].q, q):
                return int(k)
        return None
