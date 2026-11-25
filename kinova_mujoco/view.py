#!/usr/bin/env python3
"""Open a MuJoCo model in the interactive viewer, starting from its "home" keyframe.

Usage:
    python view.py                              # table scene
    python view.py models/gen3_6dof_robotiq_2f85.xml
"""
import os
import sys

import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "scenes", "table_scene.xml")

model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)
key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key)
mujoco.viewer.launch(model, data)
