#!/usr/bin/env python3
"""Export a self-contained copy of the Gen3 6-DoF + Robotiq 2F-85 MuJoCo model
from the ros_kortex repo into this folder.

Only needed again if the robot model in ros_kortex changes. Steps it performs:
  1. copies the meshes the model uses into assets/gen3_6dof and assets/robotiq_2f85
  2. rewrites every mesh path to point at those copies (meshdir="../assets/")
  3. writes models/gen3_6dof_robotiq_2f85.xml and validates it loads

Usage:
    python tools/export_from_ros_kortex.py [--ros-kortex PATH]
"""
import argparse
import os
import re
import shutil

import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROS_KORTEX = os.path.join(os.path.dirname(ROOT), "ros_kortex")
SRC_MODEL = "kortex_description/arms/gen3/6dof/mjcf/gen3_6dof_vision_robotiq_2f85.xml"
GRIPPER_LICENSE = "kortex_description/grippers/robotiq_2f_85/mjcf/LICENSE"
OUT_MODEL = os.path.join(ROOT, "models", "gen3_6dof_robotiq_2f85.xml")
GRIPPER_PREFIX = "g_"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ros-kortex", default=DEFAULT_ROS_KORTEX)
    args = ap.parse_args()

    src = os.path.join(args.ros_kortex, SRC_MODEL)
    spec = mujoco.MjSpec.from_file(src)
    src_meshdir = os.path.normpath(os.path.join(os.path.dirname(src), spec.meshdir))

    for mesh in spec.meshes:
        sub = "robotiq_2f85" if mesh.name.startswith(GRIPPER_PREFIX) else "gen3_6dof"
        src_file = os.path.normpath(os.path.join(src_meshdir, mesh.file))
        dst_file = os.path.join(ROOT, "assets", sub, os.path.basename(src_file))
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)
        mesh.file = f"{sub}/{os.path.basename(src_file)}"
    spec.meshdir = "../assets/"
    spec.modelfiledir = os.path.dirname(OUT_MODEL) + os.sep   # resolve paths from models/

    shutil.copy2(os.path.join(args.ros_kortex, GRIPPER_LICENSE),
                 os.path.join(ROOT, "assets", "robotiq_2f85", "LICENSE"))

    # Drop the <size nkey=.../> hint so scenes that attach this model don't warn.
    xml = re.sub(r"\n\s*<size nkey=\"\d+\"/>\n", "\n", spec.to_xml())
    with open(OUT_MODEL, "w") as f:
        f.write(xml)
    mujoco.MjModel.from_xml_path(OUT_MODEL)   # validate
    print(f"Wrote {OUT_MODEL}")


if __name__ == "__main__":
    main()
