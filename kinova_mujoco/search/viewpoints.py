"""Camera viewpoint sampling (used by the demos now, and by the RRT / baselines later).

A view is chosen in task space and turned into joint angles by IK:
  look at a table point, from a camera position above it and slightly offset.
"""
import numpy as np

from search import config as C
from search.robot import look_axis

VIEW_OFFSET_X = (-0.12, 0.05)       # m, camera relative to the looked-at point
VIEW_OFFSET_Y = (-0.10, 0.10)
VIEW_HEIGHT = (0.30, 0.42)          # m above the table


def random_view(robot, q_now, rng, look_at=None):
    """Valid joint config with the camera looking at `look_at` (or a random table point), or None."""
    if look_at is None:
        look_at = [rng.uniform(*C.SEARCH_REGION["x"]), rng.uniform(*C.SEARCH_REGION["y"])]
    pt = np.array([look_at[0], look_at[1], 0.0])
    cam = pt + np.array([rng.uniform(*VIEW_OFFSET_X), rng.uniform(*VIEW_OFFSET_Y), rng.uniform(*VIEW_HEIGHT)])
    return robot.ik_camera(cam, look_axis(cam, pt), q_now, rng)
