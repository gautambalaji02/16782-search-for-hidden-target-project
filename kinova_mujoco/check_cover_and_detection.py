#!/usr/bin/env python3
"""Sanity check for the hidden-target setup (put search_test_scene.xml in kinova_mujoco/scenes/).

1. Target detection by segmentation: count wrist-camera pixels that belong to the target geom.
2. Hollow cover: target invisible when covered, visible when the cover is moved.
3. Lifting by a weld constraint: activate it at the current gripper-cover offset, move the arm.
4. End-effector / camera positions, for setting workspace bounds.

Run from kinova_mujoco/:  MUJOCO_GL=egl python check_cover_and_detection.py   (MUJOCO_GL only needed headless)
"""
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("scenes/search_test_scene.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("table_view").id)
for _ in range(500):                      # let objects settle on the table
    mujoco.mj_step(m, d)

TARGET = m.geom("target_geom").id
MIN_PIXELS = 50                           # "detected" only if at least this many target pixels
seg_r = mujoco.Renderer(m, 360, 640)
seg_r.enable_segmentation_rendering()


def target_pixels():
    seg_r.update_scene(d, camera="wrist_rgb")
    seg = seg_r.render()                  # (H, W, 2): object id, object type
    return int(np.sum((seg[..., 0] == TARGET) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)))


print("target pixels, covered:  ", target_pixels())

adr = m.joint("cover1_joint").qposadr[0]  # free joint: 3 position + 4 quaternion values
d.qpos[adr:adr + 3] = [0.5, 0.25, 0.04]
mujoco.mj_forward(m, d)
px = target_pixels()
print("target pixels, uncovered:", px, "-> detected" if px >= MIN_PIXELS else "-> not detected")

print("g_pinch (EE) position:      ", d.site("g_pinch").xpos.round(3))
print("camera_color_frame position:", d.body("camera_color_frame").xpos.round(3))

# --- lift with a weld: freeze the current gripper->cover offset, then turn the weld on ---
d.qpos[adr:adr + 3] = [0.5, 0.0, 0.04]
mujoco.mj_forward(m, d)


def attach(eq_name, gripper_body, obj_body):
    eq, g, o = m.equality(eq_name).id, m.body(gripper_body).id, m.body(obj_body).id
    neg, pos, quat = np.zeros(4), np.zeros(3), np.zeros(4)
    mujoco.mju_negQuat(neg, d.xquat[g])
    mujoco.mju_rotVecQuat(pos, d.xpos[o] - d.xpos[g], neg)
    mujoco.mju_mulQuat(quat, neg, d.xquat[o])
    m.eq_data[eq, 3:6], m.eq_data[eq, 6:10] = pos, quat   # relative pose to hold
    d.eq_active[eq] = 1


def detach(eq_name):
    d.eq_active[m.equality(eq_name).id] = 0


attach("hold_cover1", "g_base", "cover1")
before = d.body("cover1").xpos.copy()
d.ctrl[1] -= 0.3                          # raise the arm
for _ in range(1500):
    mujoco.mj_step(m, d)
print("cover moved by", (d.body("cover1").xpos - before).round(3), "| target pixels now:", target_pixels())
detach("hold_cover1")