#!/usr/bin/env python3
"""Live wrist RGB-D view next to the interactive MuJoCo viewer.

Opens two windows:
  * the normal MuJoCo viewer (move the arm with the Control sliders), and
  * an OpenCV window with the wrist camera's colour image (left) and a
    colour-mapped depth image (right), updated live.

Depth window: hover the mouse to read the depth (m) under the cursor.
Keys (with the OpenCV window focused):
  s        save rgb_XXX.png, depth_XXX.npy (metres, float32) and depth_XXX.png
  q / Esc  quit

Usage:
  python camera_view.py                          # wrist_rgb, 640x360, starts at "table_view"
  python camera_view.py --key home               # start from the home pose instead
  python camera_view.py --camera wrist_depth --width 480 --height 270
  python camera_view.py --snapshot out.png       # render one frame, no GUI
"""
import argparse
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")   # OpenCV's Qt build has no Wayland plugin

import cv2
import mujoco
import mujoco.viewer
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE = os.path.join(ROOT, "scenes", "table_scene.xml")
MIN_DEPTH = 0.18   # Kinova vision module depth min-Z (User Guide r9.1, p.32)
WINDOW = "wrist RGB-D  (s: save, q: quit)"


def intrinsics(model, cam_id, width, height):
    """Pinhole K for a MuJoCo camera rendered at width x height (square pixels)."""
    f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])


class RGBDCamera:
    def __init__(self, model, camera, width, height):
        self.model, self.camera = model, camera
        self.cam_id = model.camera(camera).id
        self.rgb_r = mujoco.Renderer(model, height, width)
        self.depth_r = mujoco.Renderer(model, height, width)
        self.depth_r.enable_depth_rendering()
        self.K = intrinsics(model, self.cam_id, width, height)
        # Far clipping plane: pixels that hit it saw nothing (sky), like no return.
        self.zfar = model.vis.map.zfar * model.stat.extent

    def render(self, data):
        self.rgb_r.update_scene(data, camera=self.camera)
        self.depth_r.update_scene(data, camera=self.camera)
        rgb = self.rgb_r.render()
        depth = self.depth_r.render().astype(np.float32)   # metres along the optical axis
        depth[depth < MIN_DEPTH] = 0.0                      # mimic the real sensor: invalid
        depth[depth >= 0.99 * self.zfar] = 0.0              # nothing hit (sky): invalid
        return rgb, depth

    def close(self):
        self.rgb_r.close()
        self.depth_r.close()


def colorize_depth(depth, max_depth):
    norm = np.clip((depth - MIN_DEPTH) / (max_depth - MIN_DEPTH), 0, 1)
    img = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth == 0] = 0                                     # invalid pixels black
    return img


def compose(rgb, depth, max_depth, cursor=None):
    left = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    right = colorize_depth(depth, max_depth)
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    for img in (left, right):
        cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    label = f"centre: {depth[cy, cx]:.3f} m" if depth[cy, cx] > 0 else "centre: invalid"
    if cursor is not None and 0 <= cursor[0] < w and 0 <= cursor[1] < h:
        v = depth[cursor[1], cursor[0]]
        label += f"   cursor: {v:.3f} m" if v > 0 else "   cursor: invalid"
    cv2.putText(right, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([left, right])


def save(rgb, depth, max_depth, out_dir):
    i = 0
    while os.path.exists(os.path.join(out_dir, f"rgb_{i:03d}.png")):
        i += 1
    cv2.imwrite(os.path.join(out_dir, f"rgb_{i:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out_dir, f"depth_{i:03d}.npy"), depth)
    cv2.imwrite(os.path.join(out_dir, f"depth_{i:03d}.png"), colorize_depth(depth, max_depth))
    print(f"saved rgb_{i:03d}.png, depth_{i:03d}.npy, depth_{i:03d}.png in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--camera", default="wrist_rgb")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=20.0, help="camera window refresh rate")
    ap.add_argument("--max-depth", type=float, default=0.7, help="far end of the depth colour map (m)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "captures"))
    ap.add_argument("--key", default="table_view", help="keyframe to start from (home, table_view)")
    ap.add_argument("--snapshot", metavar="PNG", help="render one frame at --key and exit")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.key)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        print(f"keyframe '{args.key}' not found, starting from the default pose")
    mujoco.mj_forward(model, data)

    cam = RGBDCamera(model, args.camera, args.width, args.height)
    print(f"camera '{args.camera}' {args.width}x{args.height}, "
          f"fovy {model.cam_fovy[cam.cam_id]:.1f} deg\nK =\n{np.round(cam.K, 2)}")

    if args.snapshot:
        rgb, depth = cam.render(data)
        cv2.imwrite(args.snapshot, compose(rgb, depth, args.max_depth))
        print(f"wrote {args.snapshot}")
        cam.close()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    cursor = [None]
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        cursor[0] = (x - args.width, y) if x >= args.width else (x, y)   # either half

    cv2.setMouseCallback(WINDOW, on_mouse)

    def window_open():
        # OpenCV's Qt backend raises (instead of returning -1) once the window
        # has been closed with its X button.
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        period = 1.0 / args.fps
        next_frame = time.perf_counter()
        sim_start, wall_start = data.time, time.perf_counter()
        while viewer.is_running():
            # Advance physics in (roughly) real time.
            with viewer.lock():
                while data.time - sim_start < time.perf_counter() - wall_start:
                    mujoco.mj_step(model, data)
            viewer.sync()

            if time.perf_counter() >= next_frame:
                next_frame += period
                with viewer.lock():
                    rgb, depth = cam.render(data)
                cv2.imshow(WINDOW, compose(rgb, depth, args.max_depth, cursor[0]))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    print("quit key pressed")
                    break
                if k == ord("s"):
                    save(rgb, depth, args.max_depth, args.out_dir)
                if not window_open():
                    print("camera window closed")
                    break
            else:
                time.sleep(0.001)
        else:
            print("MuJoCo viewer closed")

    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    main()
