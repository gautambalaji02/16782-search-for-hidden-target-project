"""Live top-down map of the search (OpenCV window). Display only: never feeds the planner.

Layout: robot base at the bottom, +x (away from the robot) up, +y to the left,
so the picture matches looking down at the table from behind the robot.

  amber cells   POSSIBLE (target could still be here)
  light cells   SEEN (ruled out by the camera)
  grey cells    SOLID (ruled out by known geometry)
  green boxes   static obstacles          blue outlines  covers (current pose, rod drawn)
  blue polygon  current RGB-camera footprint on the table (the area that rules cells out)
  purple line   camera trail; dot = camera now
  red square    true target (hidden by default; 't' toggles it, for debugging only)
Current plan (set_plan), for the info-gain RRT:
  thin grey     tree edges between node camera positions (xy)
  green line    chosen branch, ending in a ring at the goal view's camera position
  green outline cells the chosen branch is predicted to reveal
Cover lifting (set_task):
  grey squares  parking spots outside the search region
  orange        the cover being moved, with an arrow to its parking spot

Keys (map window focused): t toggle true target, q request quit.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")   # OpenCV's Qt build has no Wayland plugin

import cv2
import numpy as np

from search import config as C
from search.coverage import POSSIBLE, SEEN, SOLID

WINDOW = "coverage map  (t: true target, q: quit)"
CELL_BGR = {POSSIBLE: (60, 190, 250), SEEN: (238, 238, 238), SOLID: (150, 150, 150)}


def _rect_corners(center, half, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * half
    return center + pts @ R.T


def _yaw(xmat):
    R = np.asarray(xmat).reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


class MapView:
    def __init__(self, model, grid, info, show_truth=False, window=True):
        self.model, self.grid = model, grid
        self.covers = [c.name for c in info.covers]
        self.obstacles = [(o.name, np.array(o.half_xy)) for o in info.obstacles]
        self.target_xy = np.array(info.target_xy)        # display only, toggled with 't'
        self.show_truth, self.quit = show_truth, False

        m = C.MAP_MARGIN
        self.xmin, self.xmax = C.SEARCH_REGION["x"][0] - m, C.SEARCH_REGION["x"][1] + m
        self.ymin, self.ymax = C.SEARCH_REGION["y"][0] - m, C.SEARCH_REGION["y"][1] + m
        self.s = C.MAP_PX_PER_M
        self.h = int((self.xmax - self.xmin) * self.s)
        self.w = int((self.ymax - self.ymin) * self.s)
        self.trail = []
        self.plan = None
        self.task = None
        self.window = window                             # False: only build images (headless, saving)
        if window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def px(self, xy):
        """World (x, y) -> integer pixel (col, row)."""
        xy = np.atleast_2d(xy)
        col = (self.ymax - xy[:, 1]) * self.s
        row = (self.xmax - xy[:, 0]) * self.s
        return np.stack([col, row], -1).round().astype(np.int32)

    def set_plan(self, plan):
        self.plan = plan

    def set_task(self, option):
        """Current cover move (task_planner.TaskOption) or None."""
        self.task = option

    def _draw_parking(self, img, data):
        for spot in C.PARK_SPOTS:
            cv2.polylines(img, [self.px(_rect_corners(np.array(spot), np.full(2, C.COVER_HALF), 0.0))], True,
                          (185, 185, 185), 1)
        if self.task is not None:
            b = data.body(self.task.cover)
            corners = _rect_corners(b.xpos[:2], np.full(2, C.COVER_HALF), _yaw(b.xmat))
            cv2.polylines(img, [self.px(corners)], True, (0, 140, 255), 3)
            cv2.arrowedLine(img, tuple(self.px(b.xpos[:2])[0]), tuple(self.px(np.array(self.task.spot))[0]),
                            (0, 140, 255), 2, tipLength=0.08)

    def _draw_plan(self, img):
        p, g = self.plan, self.grid
        if p is None:
            return
        half = g.cell / 2
        for k in p.predicted_cells:                      # predicted reveal
            c = g.centers.reshape(-1, 2)[k]
            cv2.rectangle(img, tuple(self.px(c + half)[0]), tuple(self.px(c - half)[0]), (40, 160, 40), 2)
        if p.tree_cam_xy is not None:                    # tree (camera positions)
            pts = self.px(p.tree_cam_xy)
            for k, par in enumerate(p.tree_parent):
                if par >= 0:
                    cv2.line(img, tuple(pts[par]), tuple(pts[k]), (190, 190, 190), 1)
            for pt in pts:
                cv2.circle(img, tuple(pt), 2, (150, 150, 150), -1)
            if len(p.branch) > 1:
                cv2.polylines(img, [pts[p.branch]], False, (40, 160, 40), 2)
                cv2.circle(img, tuple(pts[p.branch[-1]]), 7, (40, 160, 40), 2)

    def draw(self, data, obs, lines=()):
        img = np.full((self.h, self.w, 3), 255, np.uint8)
        g = self.grid
        half = np.array([g.cell, g.cell]) / 2
        for i in range(g.nx):                            # cells
            for j in range(g.ny):
                c = g.centers[i, j]
                tl, br = self.px(c + [half[0], half[1]])[0], self.px(c - [half[0], half[1]])[0]
                cv2.rectangle(img, tuple(tl), tuple(br), CELL_BGR[g.state[i, j]], -1)
                cv2.rectangle(img, tuple(tl), tuple(br), (215, 215, 215), 1)

        for name, hxy in self.obstacles:                 # obstacles (static)
            b = data.body(name)
            cv2.fillPoly(img, [self.px(_rect_corners(b.xpos[:2], hxy, _yaw(b.xmat)))], (60, 120, 60))
        for name in self.covers:                         # covers (current pose: they can be lifted)
            b = data.body(name)
            yaw = _yaw(b.xmat)
            cv2.polylines(img, [self.px(_rect_corners(b.xpos[:2], np.full(2, C.COVER_HALF), yaw))], True,
                          (170, 90, 40), 2)
            rod = b.xpos[:2] + np.array([[-np.sin(yaw), np.cos(yaw)]]) * np.array([[-1], [1]]) * C.COVER_ROD_HALF_LEN
            cv2.line(img, tuple(self.px(rod[0])[0]), tuple(self.px(rod[1])[0]), (80, 80, 80), 2)

        self._draw_parking(img, data)
        self._draw_plan(img)

        if self.show_truth:
            cv2.polylines(img, [self.px(_rect_corners(self.target_xy, np.full(2, C.TARGET_HALF), 0.0))], True,
                          (40, 40, 230), 2)

        if obs is not None:                              # camera footprint, trail, position
            if not np.isnan(obs.footprint).any():
                cv2.polylines(img, [self.px(obs.footprint)], True, (200, 120, 0), 2)
            self.trail.append(obs.cam_pos[:2].copy())
            if len(self.trail) > 1:
                cv2.polylines(img, [self.px(np.array(self.trail))], False, (160, 60, 160), 1)
            cv2.circle(img, tuple(self.px(obs.cam_pos[:2])[0]), 5, (160, 60, 160), -1)

        cv2.rectangle(img, tuple(self.px([C.SEARCH_REGION["x"][1], C.SEARCH_REGION["y"][1]])[0]),
                      tuple(self.px([C.SEARCH_REGION["x"][0], C.SEARCH_REGION["y"][0]])[0]), (0, 0, 0), 1)

        text = [f"possible {g.n_possible}/{g.n_start}   ruled out {100 * g.fraction_ruled_out:.0f}%"]
        if obs is not None:
            text.append(f"target pixels {obs.target_pixels}" + ("   DETECTED" if obs.detected else ""))
        text += list(lines)
        panel = np.full((22 * len(text) + 10, self.w, 3), 255, np.uint8)
        for k, t in enumerate(text):
            color = (0, 0, 200) if "DETECTED" in t else (0, 0, 0)
            cv2.putText(panel, t, (8, 22 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        out = np.vstack([panel, img])

        if obs is not None and obs.rgb is not None:      # wrist image beside the map
            rgb = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)
            scale = min(1.0, out.shape[0] / rgb.shape[0])
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)))
            side = np.full((out.shape[0], rgb.shape[1], 3), 255, np.uint8)
            side[:rgb.shape[0]] = rgb
            out = np.hstack([out, side])

        if not self.window:
            return out
        cv2.imshow(WINDOW, out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("t"):
            self.show_truth = not self.show_truth
        elif k in (ord("q"), 27):
            self.quit = True
        return out

    def close(self):
        if self.window:
            cv2.destroyWindow(WINDOW)
