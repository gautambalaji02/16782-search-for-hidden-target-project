"""Predicted visibility of coverage cells from a camera pose (the info-gain model).

Mirrors what sensing.py would rule out, using only known geometry:
a cell counts as seen from a camera pose if ANY of 5 sample points in it (centre and
the 4 quarter points, just above the table) satisfies all of
  1. it projects inside the detection (RGB) camera's image,
  2. it is at least DEPTH_MIN in front of the camera (depth sensor min range), and
  3. the segment camera -> point crosses no occluder box.
"Any point" matches sensing, which rules a cell out if any seen table point lands in it.
Occluders are the static obstacles and the covers, each an oriented box from the
table up to its top (a cover is treated as solid: the cells under it are hidden).
The robot's own links are ignored (at viewing poses they are not in the image).

Occlusion is a vectorised slab test of K segments against B boxes in numpy:
in each box's frame, the segment o + t*(p - o), t in [0, 1], hits the box if the
intervals where it lies between the slabs of all three axes overlap for some t < 1.
"""
import numpy as np

from search import config as C

CELL_Z = 0.002                      # m, cell sample points just above the table surface
CELL_SAMPLES = np.array([[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]) / 4   # in cell widths


class ViewModel:
    def __init__(self, model, grid, info):
        self.grid = grid
        cam = model.camera(C.DETECT_CAMERA).id
        self.ty = np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
        self.tx = self.ty * C.DETECT_RES[0] / C.DETECT_RES[1]
        c = grid.centers.reshape(-1, 1, 2) + CELL_SAMPLES * grid.cell       # (nx*ny, S, 2)
        self.points = np.concatenate([c, np.full(c.shape[:2] + (1,), CELL_Z)], -1)   # (nx*ny, S, 3)
        self.obstacles = [(o.name, np.array([*o.half_xy, o.height / 2])) for o in info.obstacles]
        self.covers = [(cv.name, np.array([C.COVER_HALF, C.COVER_HALF, C.COVER_HEIGHT / 2])) for cv in info.covers]
        self.boxes = None

    def update_occluders(self, data):
        """Read current occluder poses from the world (covers move when lifted)."""
        centers, halves, yaws = [], [], []
        for name, half in self.obstacles + self.covers:
            b = data.body(name)
            R = b.xmat.reshape(3, 3)
            centers.append([b.xpos[0], b.xpos[1], half[2]])                 # box from the table up
            halves.append(half)
            yaws.append(np.arctan2(R[1, 0], R[0, 0]))
        self.boxes = (np.array(centers), np.array(halves), np.array(yaws))

    def visible(self, cam_pos, cam_R, cells):
        """Bool per entry of `cells` (flat cell indices): would the camera at this pose see it?"""
        S = self.points.shape[1]
        P = self.points[cells].reshape(-1, 3)                               # (K*S, 3)
        local = (P - cam_pos) @ cam_R                                       # camera frame, looks along -z
        z = -local[:, 2]
        vis = (z >= C.DEPTH_MIN) & (np.abs(local[:, 0]) <= self.tx * z) & (np.abs(local[:, 1]) <= self.ty * z)
        idx = np.flatnonzero(vis)
        if len(idx) and self.boxes is not None:
            vis[idx] = ~self._occluded(cam_pos, P[idx])
        return vis.reshape(-1, S).any(1)

    def _occluded(self, o, P):
        """(K,) True where segment o -> P[k] passes through any occluder box before reaching P[k]."""
        centers, halves, yaws = self.boxes                                  # (B,3), (B,3), (B,)
        c, s = np.cos(yaws), np.sin(yaws)

        def to_box(v):                                                      # world (..., 3) -> each box frame (B, ..., 3)
            d = v[None] - centers.reshape((-1,) + (1,) * (v.ndim - 1) + (3,))
            cc = c.reshape((-1,) + (1,) * (v.ndim - 1))
            ss = s.reshape((-1,) + (1,) * (v.ndim - 1))
            return np.stack([cc * d[..., 0] + ss * d[..., 1], -ss * d[..., 0] + cc * d[..., 1], d[..., 2]], -1)

        o_b = to_box(o)                                                     # (B, 3)
        p_b = to_box(P)                                                     # (B, K, 3)
        d = p_b - o_b[:, None, :]
        h = halves[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h - o_b[:, None, :]) / d
            t2 = (h - o_b[:, None, :]) / d
        # Axis with d == 0: inside the slab for all t (if the origin is) or never.
        inside = np.abs(o_b[:, None, :]) <= h
        lo = np.where(d == 0, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        hi = np.where(d == 0, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        t_in, t_out = lo.max(-1), hi.min(-1)                                # (B, K)
        hit = (t_in <= t_out) & (t_out > 0) & (t_in < 1.0 - 1e-6)
        return hit.any(0)
