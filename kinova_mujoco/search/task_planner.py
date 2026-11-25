"""Task planner: which cover to lift, and where to put it (top level of the hierarchy).

Called when the info-gain RRT is stuck. Symbolic state: where each cover is, which
parking spots are taken. One action: Move(cover, spot). Its preconditions are that
a grasp and a place pose exist (IK, collision-free) and that the spot is free;
its effect is that the cells under the cover become observable.

Each feasible (cover, spot) option is scored by information per time:
    revealed = POSSIBLE cells whose centre is under the cover now
    time     = estimated pick-and-place time from the current arm configuration
               (straight-line joint times between the IK poses + fixed overheads,
               Manipulator.best_pick_place)
    score    = revealed / time            [cells per second]
and options are returned best first. The motion level (manipulation.py:
RRT-Connect + straight approach/retreat segments) then tries to execute the best
one; if it fails, the next option is tried.

Greedy, one cover at a time: after each move the info-gain RRT runs again on the
updated belief, so a revealed target is found before any further cover is moved.
"""
from dataclasses import dataclass

import numpy as np

from search import config as C
from search.coverage import POSSIBLE


@dataclass
class TaskOption:
    cover: str
    spot: tuple
    revealed: int
    pick_place: object                  # manipulation.PickPlace
    score: float

    def describe(self):
        return (f"lift {self.cover} -> park ({self.spot[0]:.2f}, {self.spot[1]:+.2f}): "
                f"{self.revealed} cells / {self.pick_place.est_time:.1f} s = {self.score:.2f} cells/s")


class CoverTaskPlanner:
    def __init__(self, grid, manipulator, info):
        self.grid, self.manip = grid, manipulator
        self.covers = [c.name for c in info.covers]
        self.moved = set()                              # covers already parked

    def revealed_cells(self, world, cover):
        """POSSIBLE cells whose centre lies under the cover's footprint (current pose)."""
        b = world.body(cover)
        R = b.xmat.reshape(3, 3)
        d = self.grid.centers - b.xpos[:2]
        lx = d[..., 0] * R[0, 0] + d[..., 1] * R[1, 0]          # into the cover's frame
        ly = d[..., 0] * R[0, 1] + d[..., 1] * R[1, 1]
        under = (np.abs(lx) < C.COVER_HALF) & (np.abs(ly) < C.COVER_HALF)
        return int(np.sum(under & (self.grid.state == POSSIBLE)))

    def free_spots(self, world):
        centres = np.array([world.body(c).xpos[:2] for c in self.covers])
        return [s for s in C.PARK_SPOTS if np.linalg.norm(centres - s, axis=1).min() >= C.PARK_CLEARANCE]

    def options(self, world, q_now):
        """Feasible Move(cover, spot) options, best score first."""
        spots = self.free_spots(world)
        out = []
        for cover in self.covers:
            if cover in self.moved:
                continue
            n = self.revealed_cells(world, cover)
            if n == 0:
                continue
            best = None
            for spot in spots:
                pp = self.manip.best_pick_place(cover, spot, q_now)
                if pp is not None and (best is None or pp.est_time < best.est_time):
                    best = pp
            if best is not None:
                out.append(TaskOption(cover, best.spot, n, best, n / best.est_time))
        return sorted(out, key=lambda o: -o.score)

    def mark_moved(self, cover):
        self.moved.add(cover)
