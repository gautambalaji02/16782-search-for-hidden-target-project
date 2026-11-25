"""Constants for the hidden-target search. No logic here.

All positions are in the world frame = robot base frame (metres), +z up,
arm faces +x, table surface at z = 0.
"""
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the folder containing search/
SIM_ROOT = os.path.join(REPO_ROOT, "kinova_mujoco")                         # robot model, scenes, assets
BASE_SCENE = os.path.join(SIM_ROOT, "scenes", "table_scene.xml")
BASE_SCENE_OBJECTS = ("red_cube", "green_cylinder", "blue_box")      # removed by scene_gen
START_KEY = "table_view"            # arm keyframe each trial starts from

# --- Workspace (planner validity check, not enforced in the XML) ---
EE_BOUNDS = {"x": (0.15, 0.75), "y": (-0.40, 0.40), "z": (0.06, 0.60)}   # g_pinch must stay inside
                                    # (z min 0.06: grasping a cover's rod puts g_pinch at ~0.09)
SEARCH_REGION = {"x": (0.35, 0.70), "y": (-0.30, 0.30)}                 # where objects/target can be
CELL_SIZE = 0.02                    # coverage grid resolution

# --- Reachability for grasping covers ---
# Arm reach is ~0.735 m from the base; a top-down grasp at rod height reaches less.
# Cover centres are kept within this horizontal distance of the base (tune with IK later).
COVER_MAX_REACH = 0.62

# --- Covers: hollow box (4 walls + top, open bottom) with a rod on two posts ---
COVER_HALF = 0.05                   # outer half-width (10 cm square footprint)
COVER_HEIGHT = 0.08
COVER_WALL = 0.0025                 # half-thickness of walls and top
COVER_ROD_RADIUS = 0.006
COVER_ROD_HALF_LEN = 0.034
COVER_POST_HALF = (0.004, 0.004, 0.01)
COVER_RGBA = (0.6, 0.6, 0.7, 1)
COVER_ROD_RGBA = (0.3, 0.3, 0.3, 1)

# --- Tall static obstacles: fixed boxes the arm cannot move; they occlude views ---
OBSTACLE_HALF_XY = (0.02, 0.035)    # range of half-widths (each side sampled separately)
OBSTACLE_HEIGHT = (0.15, 0.20)      # range of full heights
OBSTACLE_RGBA = (0.25, 0.45, 0.30, 1)

# --- Target ---
TARGET_HALF = 0.02                  # 4 cm cube
TARGET_RGBA = (1.0, 0.15, 0.15, 1)
TARGET_UNDER_COVER_OFFSET = 0.015   # max xy offset from the cover centre (interior half-width is 0.045)
TARGET_DETECT_PIXELS = 50           # "detected" if at least this many wrist-camera pixels

# --- Motion: validity checking and IK (robot.py) ---
COLLISION_MARGIN = 0.01             # m clearance the planner keeps between the arm and objects/table
BOUNDS_MARGIN = 0.01                # m the planner stays inside EE_BOUNDS
EDGE_STEP = 0.03                    # max joint change (rad) between collision checks along an edge
EDGE_STEP_HELD = 0.008              # same while carrying a cover (~5 mm motion of the cover per check)
IK_MAX_ITERS = 200
IK_DAMPING = 0.05                   # damped least squares lambda
IK_POS_TOL = 0.005                  # m
IK_ANGLE_TOL = np.deg2rad(2.0)      # camera optical axis error
IK_MAX_STEP = 0.2                   # max joint change (rad) per IK iteration
IK6_POS_TOL = 0.003                 # m, full-pose IK for g_pinch (grasping)
IK6_ANGLE_TOL = np.deg2rad(3.0)

# --- Execution: time scaling (executor.py) ---
JOINT_VMAX = 0.8                    # rad/s, all joints (Gen3 limits are higher; kept conservative)
JOINT_AMAX = 1.5                    # rad/s^2
SETTLE_TOL = 0.01                   # rad, max joint error to call a motion finished
SETTLE_TIMEOUT = 1.0                # s of extra sim time allowed to settle after a path

# --- Sensing (sensing.py) ---
DEPTH_CAMERA = "wrist_depth"
DEPTH_RES = (480, 270)              # width, height (Gen3 depth sensor)
DEPTH_STRIDE = 4                    # use every 4th depth pixel in each direction (~8k points/frame)
DEPTH_MIN = 0.18                    # m, sensor min range: closer pixels are invalid
DETECT_CAMERA = "wrist_rgb"
DETECT_RES = (640, 360)             # TARGET_DETECT_PIXELS was calibrated at this resolution
SENSE_HZ = 10.0                     # observations per second of sim time while moving

# --- Information-gain RRT (ig_rrt.py) ---
IG_NODES = 60                       # view nodes added to the tree per plan
IG_MAX_SAMPLES = 400                # view samples tried per plan (IK or connection can fail)
IG_K_NEAR = 6                       # parent candidates per new node (cheapest valid edge wins)
IG_LAMBDA = 0.5                     # 1/s, gain discount exp(-lambda * time-to-reach)
IG_MIN_GAIN = 3                     # cells: best branch below this -> "stuck" (hand over to cover lifting)

# --- Motion planner for manipulation (motion_planner.py) ---
RRTC_STEP = 0.25                    # rad, max joint change per RRT-Connect extension
RRTC_MAX_ITERS = 3000
RRTC_SAMPLE_PAD = 1.5               # rad, samples drawn in the start/goal joint box grown by this
SHORTCUT_ITERS = 80

# --- Cover manipulation (manipulation.py) and task planner (task_planner.py) ---
# Parking spots: on the table, just outside SEARCH_REGION (y = +-0.35, region ends at +-0.30),
# so a parked cover can never hide a cell. Covers are placed axis-aligned (footprint y in 0.30..0.40).
PARK_SPOTS = [(0.30, 0.35), (0.42, 0.35), (0.54, 0.35), (0.30, -0.35), (0.42, -0.35), (0.54, -0.35)]
PARK_CLEARANCE = 0.12               # m, a spot is taken if a cover centre is closer than this
PREGRASP_HEIGHT = 0.12              # m, g_pinch above the grasp/place point before descending
GRASP_DEPTH = 0.005                 # m, g_pinch below the rod centre (rod between the inner pads;
                                    # deeper and the closed fingertips press on the cover top)
PLACE_DROP = 0.005                  # m, cover released this far above the table
GRIP_OPEN, GRIP_CLOSED = 0.0, 255.0
GRIP_TIME = 0.8                     # s to close or open the gripper
# s added to every pick-and-place estimate: two gripper actions + speed-up/slow-down of its 6 arm moves
TASK_FIXED_TIME = 2 * GRIP_TIME + 6 * JOINT_VMAX / JOINT_AMAX

# --- Live map (map_view.py) ---
MAP_PX_PER_M = 900
MAP_MARGIN = 0.12                   # m drawn around SEARCH_REGION (includes the parking spots)

# --- Scene generation defaults ---
N_COVERS = 3
N_OBSTACLES = 3
P_TARGET_COVERED = 0.5              # chance the target is under a cover (else in the open)
MIN_GAP = 0.04                      # free space between object footprints (gripper clearance)
MAX_PLACEMENT_TRIES = 2000
HIDE_OPEN_TARGET = True             # an open target must not be visible from the start view
TARGET_HIDE_TRIES = 200             # target re-placements per layout before drawing a new layout
HIDE_SAMPLE_SCALE = 1.15            # hidden-check samples lie on a cube this much larger than the target
HIDE_FOV_SCALE = 1.05               # hidden-check image bounds this much wider than the real camera's
