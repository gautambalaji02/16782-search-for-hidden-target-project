# kinova_mujoco

Standalone MuJoCo model of the Kinova Gen3 **6-DoF** arm (with vision module) and
a Robotiq 2F-85 gripper, mounted on a table. It does not depend on `ros_kortex`
at runtime; every mesh it needs is in `assets/`.

## Run

```bash
source .venv/bin/activate          # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python view.py                     # table scene, starts at the "home" pose
python view.py models/gen3_6dof_robotiq_2f85.xml   # robot alone
# or plain MuJoCo:  python -m mujoco.viewer --mjcf scenes/table_scene.xml
```

## Wrist camera (live RGB-D)

```bash
python camera_view.py              # viewer + live wrist_rgb colour/depth window
python camera_view.py --key home   # start at the home pose instead of "table_view"
python camera_view.py --snapshot out.png   # one frame, no GUI
```

Cameras (in `models/gen3_6dof_robotiq_2f85.xml`, added by hand; re-exporting
overwrites them): `wrist_rgb` (fovy 31.6°, 1280×720 colour) in
`camera_color_frame`, `wrist_depth` (fovy 39.2°, 480×270) in
`camera_depth_frame`. Values from the Kinova Gen3 User Guide r9.1.
Depth is metric (m) along the optical axis; < 0.18 m (sensor min-Z) and
background/sky are set to 0 (invalid). In the camera window, hover to read
depth, `s` saves `captures/rgb_XXX.png` + `depth_XXX.npy`, `q` quits.

## Layout

```
assets/gen3_6dof/        arm meshes (from ros_kortex kortex_description)
assets/robotiq_2f85/     gripper meshes + LICENSE (from MuJoCo Menagerie)
models/gen3_6dof_robotiq_2f85.xml   robot: arm + gripper, no world/scene
scenes/table_scene.xml   floor, lights, table, robot mounted on it, objects
tools/export_from_ros_kortex.py     regenerates models/ + assets/ from ros_kortex
view.py                  viewer launcher (loads the "home" keyframe)
```

## Robot interface

| Actuator | Joint | Control |
|---|---|---|
| `act_joint_1` … `act_joint_6` | `joint_1` … `joint_6` | target joint angle (rad) |
| `g_fingers_actuator` | gripper (tendon) | 0 = open, 255 = closed |

Keyframe `home`: arm `[0, 0.26, -2.27, 0, 0.96, 1.57]`, gripper open.
Useful frames: body `end_effector_link` (flange), site `g_pinch` (between the
fingertips, 0.145 m from the flange), bodies `camera_color_frame` /
`camera_depth_frame` (wrist camera, not yet a MuJoCo camera).

## Scene

- **World origin = arm base** (bottom of `base_link`), +Z up, arm faces +x.
  Any world-frame pose from MuJoCo is therefore already in the robot base frame.
- Table surface at z = 0; floor at z = -0.75.
- Table top 1.2 m (x) × 0.8 m (y), centred at x = +0.45, so the base sits
  0.15 m in from the back edge.
- Objects: `red_cube` (5 cm), `green_cylinder`, `blue_box`, all with free joints,
  0.5–0.6 m from the arm base (arm reach is 0.735 m).

The robot is pulled in with `<include>` and must stay first in the file, so
`qpos[:6]` / `ctrl[:6]` are the arm joints, `qpos[6:14]` the gripper linkage,
`ctrl[6]` the gripper, and object free joints follow. To move the arm relative
to the table, move the `table` body (and objects) instead of the robot.

## Regenerating from ros_kortex

If the URDF changes, in `ros_kortex/kortex_description/arms/gen3/6dof/mjcf/` run
`convert_urdf_to_mjcf.py` then `build_gen3_robotiq_2f85.py`, then here run
`python tools/export_from_ros_kortex.py`.
