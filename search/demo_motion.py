#!/usr/bin/env python3
"""Check robot.py + executor.py together: plan straight-line moves, play them in the sim.

Each move picks a goal, checks the straight joint-space edge from the current
configuration with Robot.edge_valid, and if it is valid plays it with the Executor.
Goals alternate between
  * camera views: camera 0.30-0.42 m above a random table point, looking at it (IK), and
  * random valid joint configurations.
The executor's monitor reports any arm contact or bounds violation in the real sim;
if the planner-side checks are right, both stay at zero.

Usage (from the repository root):
  python -m search.demo_motion --seed 3              # with the viewer, real time
  python -m search.demo_motion --seed 3 --no-view --moves 50
"""
import argparse

import mujoco
import mujoco.viewer
import numpy as np

from search.executor import Executor
from search.robot import Robot
from search.scene_gen import make_scene, reset
from search.viewpoints import random_view


def random_goal(robot, rng, tries=2000):
    for _ in range(tries):
        q = robot.random_q(rng)
        if robot.is_valid(q):
            return q
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--moves", type=int, default=10)
    ap.add_argument("--no-view", action="store_true")
    args = ap.parse_args()

    model, _, info = make_scene(args.seed)
    print(info.summary())
    world = mujoco.MjData(model)
    reset(model, world)
    robot = Robot(model)
    rng = np.random.default_rng(args.seed)

    viewer = None
    if not args.no_view:
        viewer = mujoco.viewer.launch_passive(model, world)

    ex = Executor(model, world, robot, viewer=viewer, realtime=viewer is not None)
    done = rejected = no_goal = 0
    total_collisions, total_oob = set(), 0
    while done < args.moves and (viewer is None or viewer.is_running()):
        robot.sync(world)
        q_now = robot.q_of(world)
        kind = "view" if done % 2 == 0 else "random"
        q_goal = random_view(robot, q_now, rng) if kind == "view" else random_goal(robot, rng)
        if q_goal is None:
            no_goal += 1
            continue
        if not robot.edge_valid(q_now, q_goal):
            rejected += 1                                   # an RRT would route around this
            continue
        res = ex.run([q_now, q_goal])
        done += 1
        total_collisions |= res.collisions
        total_oob += res.out_of_bounds_steps
        print(f"move {done:2d} ({kind:6s}) {res.sim_time:5.2f}s  final err {res.final_error:.4f} rad"
              f"  contacts {len(res.collisions)}  oob steps {res.out_of_bounds_steps}")

    print(f"\n{done} moves | {rejected} straight edges rejected | {no_goal} goals not found")
    print(f"arm contacts in sim: {sorted(total_collisions) or 'none'} | out-of-bounds steps: {total_oob}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
