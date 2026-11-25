# kinova_mujoco

MuJoCo model of the Kinova Gen3 **6-DoF** arm (with vision module) and a
Robotiq 2F-85 gripper, mounted on a table. It does not depend on `ros_kortex`
at runtime; every mesh it needs is in `assets/`.

The hidden-target search planner that uses this model lives in `../search/`.
Setup and how to run it are in the [repository README](../README.md).

## 1. Setup

Use the virtual environment at the repository root (see the
[repository README](../README.md#1-setup-once)):

```bash
cd <repo root>
source .venv/bin/activate              # create it first if needed: python3 -m venv .venv
pip install -r kinova_mujoco/requirements.txt
cd kinova_mujoco                       # run the tools below from here
```

## 2. Tools

```bash
python view.py                                   # table scene at the "home" pose
python view.py models/gen3_6dof_robotiq_2f85.xml # robot alone
python -m mujoco.viewer --mjcf scenes/table_scene.xml   # plain MuJoCo viewer

python camera_view.py                            # viewer + live wrist RGB-D window
python camera_view.py --key home                 # start at "home" instead of "table_view"
python camera_view.py --snapshot out.png         # one RGB-D frame, no GUI

python check_cover_and_detection.py              # sanity check: cover hides target, weld lift
```
In the `camera_view.py` window, hover to read depth, `s` saves
`captures/rgb_XXX.png` and `depth_XXX.npy`, and `q` quits. Without a display, prefix with
`MUJOCO_GL=egl` (only the `--snapshot` and check scripts work headless).

## 3. Files

| File | What it does |
|---|---|
| `view.py` | Opens a model in the MuJoCo viewer at its `home` keyframe. |
| `camera_view.py` | Live wrist-camera colour + depth window next to the viewer; `RGBDCamera` class (RGB + metric depth). |
| `check_cover_and_detection.py` | One-off check on `scenes/search_test_scene.xml`: a cover hides the target from the camera, and a weld can lift the cover. |
| `models/gen3_6dof_robotiq_2f85.xml` | The robot: arm, gripper, wrist cameras `wrist_rgb` and `wrist_depth`. No world. |
| `scenes/table_scene.xml` | Floor, lights, table, robot. The search's scene generator (`../search/scene_gen.py`) starts from it. Keyframes `home`, `table_view`. |
| `scenes/search_test_scene.xml` | Hand-written one-cover test scene used by `check_cover_and_detection.py`. |
| `assets/` | Arm meshes (from ros_kortex) and gripper meshes + LICENSE (from MuJoCo Menagerie). |
| `tools/export_from_ros_kortex.py` | Regenerates `models/` and `assets/` from `ros_kortex` (only if the URDF changes). |
| `requirements.txt` | Python dependencies for this folder and `../search/` (MuJoCo, OpenCV). |

## 4. Robot interface and scene conventions

| Actuator | Joint | Control |
|---|---|---|
| `act_joint_1` … `act_joint_6` | `joint_1` … `joint_6` | target joint angle (rad) |
| `g_fingers_actuator` | gripper (tendon) | 0 = open, 255 = closed |

- **World origin = arm base** (bottom of `base_link`), +z up, arm faces +x.
  Every world-frame pose from MuJoCo is already in the robot base frame.
- Table surface at z = 0 (1.2 m × 0.8 m, centred at x = 0.45); floor at z = -0.75.
  Arm reach is about 0.735 m. To move the arm relative to the table, move the
  `table` body (and objects) instead of the robot.
- The robot is pulled in with `<include>` and must stay first, so `qpos[:6]` /
  `ctrl[:6]` are the arm joints, `qpos[6:14]` the gripper linkage and `ctrl[6]`
  the gripper. Object free joints follow; use `model.joint(name).qposadr`
  rather than fixed indices.
- Useful frames: body `end_effector_link` (flange); site `g_pinch` (between the
  fingertips, 0.145 m from the flange; its z axis points along the fingers and y
  is the closing axis); body `g_base` (gripper base).
- Cameras (from the Gen3 User Guide r9.1): `wrist_rgb` (fovy 31.6°, 1280×720)
  and `wrist_depth` (fovy 39.2°, 480×270). Depth is metric along the optical axis.
  Readings below 0.18 m (the sensor's minimum range) and background are invalid.
  The camera definitions were added by hand to the model file; re-exporting it
  overwrites them.
- Keyframes: `home` (arm `[0, 0.26, -2.27, 0, 0.96, 1.57]`), `table_view`
  (camera about 0.42 m above the table, looking down; every search starts here).

## 5. Regenerating the robot model from ros_kortex

If the URDF changes, in `ros_kortex/kortex_description/arms/gen3/6dof/mjcf/` run
`convert_urdf_to_mjcf.py` then `build_gen3_robotiq_2f85.py`, then here run
`python tools/export_from_ros_kortex.py`. Re-add the two wrist cameras afterwards.
