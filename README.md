# 16782-search-for-hidden-target-project

CMU 16-782 (Planning & Decision Making for Robotics) final project.

A Kinova Gen3 6-DoF arm with a Robotiq 2F-85 gripper and a wrist RGB-D camera
must find a small target hidden on a table. Tall static obstacles block some
views, and the target may sit under one of several movable **covers** (hollow
boxes with a grasp rod). The arm can **look** (move the camera) or **lift a
cover** and park it elsewhere.

The search is hierarchical:
1. **Information-gain RRT** chooses camera viewpoints that rule out the most
   still-possible table cells per unit of motion time.
2. When no view gives new information, a **task planner** picks the cover with
   the most possible cells under it per second of pick-and-place time. It parks
   that cover outside the search region and hands control back to the RRT.

All object poses are known except the target's. Everything runs in MuJoCo
simulation.

## Repository layout

```
search/            the planner (Python package): scene generation, sensing, belief,
                   info-gain RRT, task planner, manipulation, live map
kinova_mujoco/     MuJoCo robot model, table scene, meshes, viewer/camera tools
                   (see kinova_mujoco/README.md)
rough_scripts/     throwaway scripts for the real Gen3 arm and depth camera
                   (see rough_scripts/README.md)
```

## 1. Setup (once)

Tested on Ubuntu 22.04, Python 3.10, MuJoCo 3.17.

```bash
cd <repo root>                         # the folder containing search/ and kinova_mujoco/
python3 -m venv .venv
source .venv/bin/activate
pip install -r kinova_mujoco/requirements.txt     # MuJoCo + OpenCV (NumPy comes with MuJoCo)
```

In each new terminal, run `source .venv/bin/activate` from the repo root.

**Run every `python -m search.…` command from the repository root.** `search`
is a Python package, and it is only found from there. It finds the robot model
in `kinova_mujoco/` on its own.

**No display** (SSH, a server): add `--no-view --no-map` and prefix the
command with `MUJOCO_GL=egl` so off-screen rendering works:
```bash
MUJOCO_GL=egl python -m search.run_search --seed 3 --no-view --no-map --save-map /tmp/map.png
```

## 2. Run the search, step by step

Each step tests one layer on top of the previous one. A scene is fully
determined by its `--seed`: the same seed always gives the same covers,
obstacles and target.

### Step 1: look at a generated scene
```bash
python -m search.scene_gen --seed 3
```
Prints where each cover, obstacle and the target are, then opens the viewer.
Grey boxes with a rod are covers, green boxes are static obstacles, and the red
cube is the target.

| Variation | Command |
|---|---|
| More/fewer objects | `--covers 4 --obstacles 2` |
| Force the target under a cover / in the open | `--p-covered 1.0` / `--p-covered 0.0` |
| Allow an open target to be visible from the start pose | `--show-open-target` |
| Save the scene as MJCF instead of viewing it | `--no-view --save /tmp/scene3.xml` |

### Step 2: check motion planning and execution
```bash
python -m search.demo_motion --seed 3
```
The arm moves between random camera views and random poses in real time. Each
move prints one line. The summary should report **no arm contacts and 0
out-of-bounds steps**.
- `--no-view --moves 30` runs faster than real time, without the viewer.

### Step 3: run the full search
```bash
python -m search.run_search --seed 3 --truth
```
This opens the MuJoCo viewer and a live map window. Close the viewer to exit;
in the map window, `t` toggles the true target and `q` quits.

Expected (seed 3): 4 view plans, then the info-gain RRT is stuck with only the
patches under the covers left. The task planner lifts `cover0` and the target is
detected while the cover is carried away, at about 13 s of sim time.

| Variation | Command |
|---|---|
| Random-view baseline (same cover lifting) | `--planner random` |
| Never lift covers (stop when views run out) | `--no-lift` |
| Several lifts before the target is found | `--seed 0` (3 lifts), `--seed 1` (2 lifts) |
| Target in the open (no lifting needed) | `--seed 5`, `--seed 2` |
| Draw the true target on the map (debugging only) | `--truth` |
| Headless, faster than real time, save the final map | `--no-view --no-map --save-map /tmp/map.png` |
| Limit the number of view plans | `--plans 20` (default 40) |

The terminal prints one line per view plan: nodes, planning time, predicted vs
actually ruled-out cells, and cells left. It prints one line per cover move:
which cover, which parking spot, and its score in cells per second. The final
summary gives when the target was found (sim time), the number of view plans
and lifts, and planning time.

**Reading the live map** (robot at the bottom, +x up, +y left):

| Colour | Meaning |
|---|---|
| Amber cells | still **possible** (the target could be here) |
| Light cells | **seen**: ruled out by the camera |
| Grey cells | **solid**: ruled out by known geometry (inside an obstacle or a cover wall) |
| Green boxes / blue outlines | static obstacles / covers at their current pose |
| Blue polygon | the camera's current field of view on the table |
| Purple line, dot | camera trail, current camera position |
| Thin grey tree, green line, green cells | info-gain RRT tree, chosen branch, cells it should reveal |
| Grey squares at the sides | parking spots for lifted covers |
| Orange cover + arrow | the cover being moved and its parking spot |
| Red square | the true target (only with `--truth` / `t`) |

The wrist RGB image is shown to the right of the map.

## 3. Files in `search/`

Entry points are `run_search.py`, `scene_gen.py` and `demo_motion.py`. Every
tunable number is in `config.py`.

| File | What it does |
|---|---|
| `config.py` | All constants (workspace bounds, object sizes, sensing, planner and manipulation parameters, parking spots) and the path to the MuJoCo scene in `kinova_mujoco/`. No logic. |
| `scene_gen.py` | Builds a random scene from a seed: covers (with a lift weld each), tall static obstacles, and the target under a cover or hidden from the start view. |
| `robot.py` | The planner's view of the arm on a scratch copy of the model: forward kinematics, collision/bounds checks, IK for the camera and the gripper, and carrying a held cover. |
| `trajectory.py` | Turns a joint path into a smooth timed trajectory (speed and acceleration limits). |
| `executor.py` | The only code that moves the simulated arm: plays trajectories in real time with feedforward, commands the gripper, and calls the sensor while moving. |
| `sensing.py` | Wrist-camera observation: target pixel count (detection) and the table points seen through free space. |
| `coverage.py` | The belief: a 2 cm grid over the search region with POSSIBLE / SEEN / SOLID cells. |
| `gain.py` | Predicts which possible cells a camera pose would see (field of view, sensor range, occlusion by boxes). This is the information gain. |
| `viewpoints.py` | Samples a camera view looking at a table point and solves IK for it. |
| `ig_rrt.py` | Information-gain RRT: a tree of views, scored by gain discounted by time-to-reach; returns the best branch or "stuck". |
| `baselines.py` | Random-view planner with the same interface, for comparison. |
| `motion_planner.py` | Joint-space RRT-Connect with shortcutting, for the long moves during pick-and-place. |
| `manipulation.py` | Grasp/place poses and the pick-and-place sequence for a cover (approach, grasp + weld, lift, carry, place, release, retreat). |
| `task_planner.py` | Chooses which cover to lift and where to park it: most possible cells revealed per second of pick-and-place. |
| `map_view.py` | The live 2D map window (display only; it never feeds the planner). |
| `run_search.py` | Main loop: view planning → execute and observe → update belief → replan, lifting a cover when stuck. |
| `demo_motion.py` | Motion-layer test: straight-line moves to random goals with a contact/bounds monitor. |

`.venv/` is local to your machine and is ignored by git.
