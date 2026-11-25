"""Wrist-camera sensing: target detection and table points seen through free space.

Two outputs per observation:
  * target_pixels: wrist_rgb segmentation pixels belonging to the target geom.
    This is the ONLY way the search learns about the target (detected if
    >= TARGET_DETECT_PIXELS).
  * table_xy: world (x, y) of depth pixels whose ray hit the bare table top AND that
    lie inside the RGB (detection) camera's image. If the camera saw the table at a
    point, nothing (in particular no target) stands there, so the coverage grid can
    rule that cell out. The RGB filter matters: the depth camera is wider (64 vs 53 deg
    horizontal), and a cell must only be ruled out where the detector also looked;
    otherwise a target seen only by depth would stay POSSIBLE but never be detected,
    and the planner would keep predicting gain there. gain.py uses the same RGB view.

Which surface a depth pixel hit is read from a segmentation render of the depth
camera (same camera, same resolution, so pixels line up). This uses the known
scene geometry, as agreed (all poses are ground truth except the target's); it
replaces a height threshold on the depth, which would be noisy near the table.
Pixels that hit the gripper, covers, obstacles or nothing are simply not used.

Back-projection (MuJoCo cameras look along -z, x right, y up; image v grows down):
    p_cam = d * ((u - cx) / f, -(v - cy) / f, -1),   p_world = cam_pos + R_cam @ p_cam
where d is the depth along the optical axis.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from search import config as C


@dataclass
class Observation:
    target_pixels: int
    detected: bool
    table_xy: np.ndarray                 # (N, 2) table points seen, world frame
    cam_pos: np.ndarray
    footprint: np.ndarray                # (4, 2) where the RGB image corners meet the table (z=0)
    rgb: np.ndarray = None               # wrist image, only if requested


def _focal(model, cam_id, height):
    return 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)


class Sensor:
    def __init__(self, model, rgb_size=(320, 180)):
        self.model = model
        W, H = C.DEPTH_RES
        self.depth_cam = model.camera(C.DEPTH_CAMERA).id
        self.detect_cam = C.DETECT_CAMERA
        self.detect_id = model.camera(C.DETECT_CAMERA).id
        self.det_ty = np.tan(np.deg2rad(model.cam_fovy[self.detect_id]) / 2)
        self.det_tx = self.det_ty * C.DETECT_RES[0] / C.DETECT_RES[1]
        self.target_geom = model.geom("target_geom").id
        self.table_geom = model.geom("table_top").id

        # Keep every renderer alive for the Sensor's lifetime (freeing one EGL
        # context while others are in use breaks them).
        self.depth_r = mujoco.Renderer(model, H, W)
        self.depth_r.enable_depth_rendering()
        self.dseg_r = mujoco.Renderer(model, H, W)
        self.dseg_r.enable_segmentation_rendering()
        self.tseg_r = mujoco.Renderer(model, C.DETECT_RES[1], C.DETECT_RES[0])
        self.tseg_r.enable_segmentation_rendering()
        self.rgb_r = mujoco.Renderer(model, rgb_size[1], rgb_size[0])

        # Camera-frame ray directions (z = -1) for the sub-sampled pixel grid.
        s = C.DEPTH_STRIDE
        f = _focal(model, self.depth_cam, H)
        self.v, self.u = np.mgrid[s // 2:H:s, s // 2:W:s]
        self.rays = np.stack([(self.u - W / 2) / f, -(self.v - H / 2) / f, -np.ones_like(self.u, float)], -1)
        self.zfar = model.vis.map.zfar * model.stat.extent

    def target_pixels(self, data):
        self.tseg_r.update_scene(data, camera=self.detect_cam)
        seg = self.tseg_r.render()
        return int(np.sum((seg[..., 0] == self.target_geom) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)))

    def observe(self, data, with_rgb=False):
        cam_pos = data.cam_xpos[self.depth_cam].copy()
        R = data.cam_xmat[self.depth_cam].reshape(3, 3)

        self.depth_r.update_scene(data, camera=self.depth_cam)
        self.dseg_r.update_scene(data, camera=self.depth_cam)
        depth = self.depth_r.render()[self.v, self.u]
        seg = self.dseg_r.render()[self.v, self.u]

        hit_table = ((seg[..., 0] == self.table_geom) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
                     & (depth >= C.DEPTH_MIN) & (depth < 0.99 * self.zfar))
        pts = cam_pos + (self.rays[hit_table] * depth[hit_table][:, None]) @ R.T
        pts = pts[self._in_detect_view(data, pts)]

        n = self.target_pixels(data)
        rgb = None
        if with_rgb:
            self.rgb_r.update_scene(data, camera=self.detect_cam)
            rgb = self.rgb_r.render()
        return Observation(target_pixels=n, detected=n >= C.TARGET_DETECT_PIXELS, table_xy=pts[:, :2],
                           cam_pos=cam_pos, footprint=self._footprint(data), rgb=rgb)

    def _in_detect_view(self, data, pts):
        """Mask of world points that project inside the detection camera's image."""
        local = (pts - data.cam_xpos[self.detect_id]) @ data.cam_xmat[self.detect_id].reshape(3, 3)
        z = -local[:, 2]
        return (z > 0) & (np.abs(local[:, 0]) <= self.det_tx * z) & (np.abs(local[:, 1]) <= self.det_ty * z)

    def _footprint(self, data):
        """Corners of the detection (RGB) image projected onto z = 0 (for drawing; NaN if a corner ray misses)."""
        cam_pos = data.cam_xpos[self.detect_id]
        R = data.cam_xmat[self.detect_id].reshape(3, 3)
        tx, ty = self.det_tx, self.det_ty
        corners = np.array([[-tx, ty, -1], [tx, ty, -1], [tx, -ty, -1], [-tx, -ty, -1]])
        dirs = corners @ R.T
        out = np.full((4, 2), np.nan)
        down = dirs[:, 2] < -1e-6
        t = -cam_pos[2] / dirs[down, 2]
        out[down] = (cam_pos + t[:, None] * dirs[down])[:, :2]
        return out

    def close(self):
        for r in (self.depth_r, self.dseg_r, self.tseg_r, self.rgb_r):
            r.close()
