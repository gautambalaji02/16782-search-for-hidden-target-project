"""Baseline view planners with the same interface as InfoGainRRT: plan(q_now) -> Plan.

RandomViewPlanner: look at a random POSSIBLE cell from a random nearby camera
pose (IK), move there on a straight edge. No prediction of what the view reveals.
It reports "stuck" only when it cannot find any reachable view.
"""
import time

import numpy as np

from search import config as C
from search.coverage import POSSIBLE
from search.ig_rrt import Plan
from search.viewpoints import random_view


class RandomViewPlanner:
    def __init__(self, robot, grid, rng, tries=C.IG_MAX_SAMPLES):
        self.robot, self.grid, self.rng, self.tries = robot, grid, rng, tries

    def plan(self, q_now):
        t0 = time.perf_counter()
        cand = np.flatnonzero(self.grid.state.ravel() == POSSIBLE)
        cand_xy = self.grid.centers.reshape(-1, 2)[cand]
        for _ in range(self.tries):
            if len(cand) == 0:
                break
            q = random_view(self.robot, q_now, self.rng, look_at=cand_xy[self.rng.integers(len(cand))])
            if q is not None and self.robot.edge_valid(q_now, q):
                return Plan(path=[np.asarray(q_now, float), q], predicted_cells=np.array([], int),
                            n_nodes=2, plan_time=time.perf_counter() - t0)
        return Plan(path=[np.asarray(q_now, float)], predicted_cells=np.array([], int), stuck=True,
                    plan_time=time.perf_counter() - t0)
