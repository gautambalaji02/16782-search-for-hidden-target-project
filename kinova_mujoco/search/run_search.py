#!/usr/bin/env python3
"""Hierarchical search: view planning, plus lifting covers when views run out.

Loop:
  1. View planner -> execute (observing at SENSE_HZ) -> update the coverage map -> replan.
       --planner ig      information-gain RRT (ig_rrt.py)
       --planner random  random views of POSSIBLE cells (baselines.py)
  2. When the view planner is stuck (ig: best branch reveals < IG_MIN_GAIN cells;
     random: RANDOM_STUCK_AFTER views in a row ruled nothing out), the task planner
     (task_planner.py) picks the cover with the most POSSIBLE cells under it per
     second of pick-and-place time, and the manipulator (manipulation.py) parks it
     outside the search region. The belief needs no special update: the cells under
     the cover were POSSIBLE all along; the view model now reads the cover's new pose,
     so those cells become reachable by views. Then back to step 1.
Stops when the target is detected, no POSSIBLE cells remain, nothing is left to
lift (what remains is permanently occluded), or --plans is reached.

Usage (from kinova_mujoco/):
  python -m search.run_search --seed 3 --truth            # covered target: info-gain + lifting
  python -m search.run_search --seed 5 --planner random
  python -m search.run_search --seed 3 --no-lift          # stop when the view planner is stuck
  python -m search.run_search --seed 3 --no-view --no-map --save-map /tmp/map.png
"""
import argparse
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from search import config as C
from search.baselines import RandomViewPlanner
from search.coverage import CoverageGrid
from search.executor import Executor
from search.gain import ViewModel
from search.ig_rrt import InfoGainRRT
from search.manipulation import Manipulator
from search.map_view import MapView
from search.robot import Robot
from search.scene_gen import make_scene, reset
from search.sensing import Sensor
from search.task_planner import CoverTaskPlanner

RANDOM_STUCK_AFTER = 3                  # random baseline: views in a row with no new cells -> stuck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--planner", choices=["ig", "random"], default="ig")
    ap.add_argument("--plans", type=int, default=40, help="max view plan/execute iterations")
    ap.add_argument("--no-lift", action="store_true", help="never lift covers")
    ap.add_argument("--no-view", action="store_true", help="no MuJoCo viewer (runs faster than real time)")
    ap.add_argument("--no-map", action="store_true", help="no live map window")
    ap.add_argument("--truth", action="store_true", help="draw the true target on the map (debug)")
    ap.add_argument("--save-map", metavar="PNG", help="write the final map image")
    args = ap.parse_args()

    model, _, info = make_scene(args.seed)
    print(info.summary(), "\n")
    world = mujoco.MjData(model)
    reset(model, world)
    rng = np.random.default_rng(args.seed)

    robot = Robot(model)
    sensor = Sensor(model)
    grid = CoverageGrid(info.obstacles, info.covers)          # the target is never given to the planner
    views = ViewModel(model, grid, info)
    planner = InfoGainRRT(robot, grid, views, rng) if args.planner == "ig" else RandomViewPlanner(robot, grid, rng)

    show_map = not args.no_map
    mapv = MapView(model, grid, info, show_truth=args.truth, window=show_map) \
        if (show_map or args.save_map) else None
    viewer = None if args.no_view else mujoco.viewer.launch_passive(model, world)
    ex = Executor(model, world, robot, viewer=viewer, realtime=viewer is not None, frame_hz=C.SENSE_HZ)
    manip = Manipulator(model, world, robot, ex, rng)
    tasks = CoverTaskPlanner(grid, manip, info)

    st = {"plan": 0, "lifts": 0, "found": False, "found_at": None, "img": None, "status": "start"}

    def observe(d):
        obs = sensor.observe(d, with_rgb=show_map)
        grid.update(obs.table_xy)
        if obs.detected and not st["found"]:
            st["found"], st["found_at"] = True, d.time
        if mapv is not None:
            st["img"] = mapv.draw(d, obs, [f"{args.planner}  plan {st['plan']}  lifts {st['lifts']}"
                                           f"   sim time {d.time:.1f} s", st["status"]])
        return mapv is not None and mapv.quit

    def on_view_frame(d):                                     # view moves stop as soon as the target is seen
        return observe(d) or st["found"]

    observe(world)                                            # observe from the start pose
    print(f"start: {grid.n_start} possible cells, {grid.n_possible} after the first view")

    reason, plan_time, task_time, no_gain = "plan limit", 0.0, 0.0, 0
    while st["plan"] < args.plans:
        if st["found"]:
            reason = "target detected"
            break
        if grid.n_possible == 0:
            reason = "no possible cells left"
            break
        if (mapv is not None and mapv.quit) or (viewer is not None and not viewer.is_running()):
            reason = "quit"
            break

        robot.sync(world)
        views.update_occluders(world)
        plan = planner.plan(robot.q_of(world))
        plan_time += plan.plan_time
        stuck = plan.stuck or (args.planner == "random" and no_gain >= RANDOM_STUCK_AFTER)

        if stuck:                                             # ---- task level: lift a cover
            if args.no_lift:
                reason = f"stuck: {grid.n_possible} possible cells left (lifting disabled)"
                break
            t0 = time.perf_counter()
            options = tasks.options(world, robot.q_of(world))
            task_time += time.perf_counter() - t0
            if not options:
                reason = f"stuck: {grid.n_possible} possible cells left, no cover left to lift (occluded)"
                break
            moved = False
            for opt in options:
                st["status"] = opt.describe()
                print(f"task: {opt.describe()}   ({len(options)} option(s), {time.perf_counter() - t0:.2f}s)")
                if mapv is not None:
                    mapv.set_plan(None)
                    mapv.set_task(opt)
                before, t_sim = grid.n_possible, world.time
                ok, msg = manip.execute(opt.pick_place, observe=observe)
                if ok:
                    tasks.mark_moved(opt.cover)
                    st["lifts"] += 1
                    print(f"  done in {world.time - t_sim:.1f}s sim (estimated {opt.pick_place.est_time:.1f}s),"
                          f" ruled out {before - grid.n_possible} on the way" + ("  TARGET DETECTED" if st["found"] else ""))
                    moved = True
                    break
                print(f"  failed: {msg}")
            if mapv is not None:
                mapv.set_task(None)
            st["status"] = ""
            no_gain = 0
            if not moved:
                reason = "stuck: no cover could be moved"
                break
            continue

        st["plan"] += 1                                       # ---- view level
        st["status"] = f"view plan: predicted {len(plan.predicted_cells)} new cells"
        if mapv is not None:
            mapv.set_plan(plan)
        before = grid.n_possible
        res = ex.run(plan.path, on_frame=on_view_frame)
        gained = before - grid.n_possible
        no_gain = 0 if gained > 0 else no_gain + 1
        print(f"plan {st['plan']:2d}: {plan.n_nodes:3d} nodes {plan.plan_time:5.2f}s | branch {len(plan.path) - 1} edge(s)"
              f" {res.sim_time:4.1f}s | predicted {len(plan.predicted_cells):3d} cells, ruled out {gained:3d}"
              f" | possible {grid.n_possible:3d}" + ("  TARGET DETECTED" if st["found"] else ""))

    where = f"under {info.target_cover}" if info.target_cover else "in the open"
    found = f"FOUND at {st['found_at']:.1f} s sim" if st["found"] else "NOT FOUND"
    print(f"\n[{args.planner}] {found} ({reason})"
          f"\n  view plans {st['plan']} | covers lifted {st['lifts']} | sim time {world.time:.1f} s"
          f" | planning time: views {plan_time:.1f} s, tasks {task_time:.1f} s"
          f" | ruled out {100 * grid.fraction_ruled_out:.1f}% | target was {where}")

    if args.save_map and st["img"] is not None:
        cv2.imwrite(args.save_map, st["img"])
        print(f"wrote {args.save_map}")
    if viewer is not None:                                    # keep windows up until the viewer is closed
        while viewer.is_running() and not (mapv is not None and mapv.quit):
            viewer.sync()
            if mapv is not None and show_map:
                st["img"] = mapv.draw(world, None, [f"finished: {reason}", "close the viewer to exit"])
            cv2.waitKey(30)
        viewer.close()
    sensor.close()


if __name__ == "__main__":
    main()
