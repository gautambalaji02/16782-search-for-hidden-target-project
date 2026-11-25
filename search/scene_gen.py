#!/usr/bin/env python3
"""Random hidden-target scenes, reproducible from a seed.

Starts from scenes/table_scene.xml (robot + table), removes its demo objects and adds:
  * N movable covers   (hollow box + grasp rod, free joint, one inactive weld to g_base each)
  * M tall static obstacles (fixed boxes; they only block views and motion)
  * one target          (4 cm cube, free joint), under a random cover or in the open

How a layout is drawn:
  1. Covers, then obstacles, are placed by rejection sampling inside SEARCH_REGION.
     Each object is treated as a circle (its footprint's circumscribed circle) and
     circles must be MIN_GAP apart, so any yaw is safe. Cover centres must also be
     within COVER_MAX_REACH of the base so the arm can grasp the rod.
  2. The target goes under a chosen cover (small offset from its centre) or on free table.
  3. The model is compiled and the start pose (START_KEY) is checked for arm/object
     contact; if it collides, a new layout is drawn (same RNG, so still deterministic).
  4. If the target is in the open and hide_open_target is on, it must not be visible
     from the start view, otherwise every such trial ends before any planning. The
     target is re-placed (covers/obstacles kept) until no sample point on it can be
     seen by the detection camera at START_KEY; after TARGET_HIDE_TRIES failures a new
     layout is drawn. Visibility is checked with ray casts (no rendering):
       a point is visible if it projects inside the camera image and the ray from the
       camera reaches it without hitting anything drawn (geom groups 0-2, the same ones
       the renderer shows). The target itself is excluded from the rays, so points on
       its far side count as visible too: the test is conservative (hidden = really hidden).

The robot stays first in qpos (it comes from the <include> in the base scene). Object
free joints follow in the order: covers, then target. Use model.joint(name).qposadr.

Usage (from the repository root):
  python -m search.scene_gen --seed 3                 # print layout, open the viewer
  python -m search.scene_gen --seed 3 --no-view --save /tmp/scene3.xml
"""
import argparse
from dataclasses import dataclass, field

import mujoco
import mujoco.viewer
import numpy as np

from search import config as C


# ---------------------------------------------------------------- layout (pure geometry)

@dataclass
class Cover:
    name: str
    xy: np.ndarray
    yaw: float

    @property
    def weld(self):                     # equality constraint used to lift this cover
        return f"hold_{self.name}"

    @property
    def grasp_site(self):
        return f"{self.name}_grasp"


@dataclass
class Obstacle:
    name: str
    xy: np.ndarray
    yaw: float
    half_xy: tuple
    height: float


@dataclass
class SceneInfo:
    seed: int
    covers: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    target_xy: np.ndarray = None
    target_yaw: float = 0.0
    target_cover: str = None            # name of the cover hiding the target, or None

    def summary(self):
        lines = [f"seed {self.seed}"]
        for c in self.covers:
            lines.append(f"  {c.name:10s} xy=({c.xy[0]:.3f}, {c.xy[1]:+.3f})  yaw={np.degrees(c.yaw):6.1f} deg")
        for o in self.obstacles:
            lines.append(f"  {o.name:10s} xy=({o.xy[0]:.3f}, {o.xy[1]:+.3f})  yaw={np.degrees(o.yaw):6.1f} deg"
                         f"  half=({o.half_xy[0]:.3f}, {o.half_xy[1]:.3f})  h={o.height:.3f}")
        where = f"under {self.target_cover}" if self.target_cover else "in the open"
        lines.append(f"  target     xy=({self.target_xy[0]:.3f}, {self.target_xy[1]:+.3f})  {where}")
        return "\n".join(lines)


COVER_RADIUS = C.COVER_HALF * np.sqrt(2)
TARGET_RADIUS = C.TARGET_HALF * np.sqrt(2)


def _sample_point(rng, radius, placed, gap, max_reach=None):
    """Point in SEARCH_REGION whose circle (radius) keeps `gap` from every placed (xy, r)."""
    (x0, x1), (y0, y1) = C.SEARCH_REGION["x"], C.SEARCH_REGION["y"]
    for _ in range(C.MAX_PLACEMENT_TRIES):
        p = np.array([rng.uniform(x0 + radius, x1 - radius), rng.uniform(y0 + radius, y1 - radius)])
        if max_reach is not None and np.linalg.norm(p) > max_reach:
            continue
        if all(np.linalg.norm(p - q) >= radius + r + gap for q, r in placed):
            return p
    return None


def _sample_layout(rng, seed, n_covers, n_obstacles, p_covered):
    """One random layout, or None if the objects did not fit."""
    info, placed = SceneInfo(seed=seed), []          # placed: (xy, radius) of every footprint

    for i in range(n_covers):
        xy = _sample_point(rng, COVER_RADIUS, placed, C.MIN_GAP, C.COVER_MAX_REACH)
        if xy is None:
            return None
        info.covers.append(Cover(f"cover{i}", xy, rng.uniform(0, np.pi)))
        placed.append((xy, COVER_RADIUS))

    for i in range(n_obstacles):
        half = tuple(rng.uniform(*C.OBSTACLE_HALF_XY, size=2))
        r = float(np.hypot(*half))
        xy = _sample_point(rng, r, placed, C.MIN_GAP)
        if xy is None:
            return None
        info.obstacles.append(Obstacle(f"obstacle{i}", xy, rng.uniform(0, np.pi), half,
                                       rng.uniform(*C.OBSTACLE_HEIGHT)))
        placed.append((xy, r))

    info.target_yaw = rng.uniform(0, np.pi / 2)
    if info.covers and rng.random() < p_covered:
        cover = info.covers[rng.integers(len(info.covers))]
        info.target_cover = cover.name
        info.target_xy = cover.xy + rng.uniform(-C.TARGET_UNDER_COVER_OFFSET, C.TARGET_UNDER_COVER_OFFSET, 2)
    else:
        info.target_xy = _sample_point(rng, TARGET_RADIUS, placed, gap=0.01)
        if info.target_xy is None:
            return None
    return info


# ---------------------------------------------------------------- building the MjSpec

def _yaw_quat(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def _add_cover(spec, cover):
    """Four walls + top (open bottom), two posts, a rod along the cover's local y, a grasp site."""
    h, w, a = C.COVER_HEIGHT / 2, C.COVER_WALL, C.COVER_HALF
    body = spec.worldbody.add_body(name=cover.name, pos=[*cover.xy, h], quat=_yaw_quat(cover.yaw))
    body.add_freejoint(name=f"{cover.name}_joint")
    box = mujoco.mjtGeom.mjGEOM_BOX
    rgba = list(C.COVER_RGBA)
    body.add_geom(name=f"{cover.name}_top", type=box, size=[a, a, w], pos=[0, 0, h - w], rgba=rgba, mass=0.05)
    body.add_geom(name=f"{cover.name}_px", type=box, size=[w, a, h], pos=[a - w, 0, 0], rgba=rgba, mass=0.03)
    body.add_geom(name=f"{cover.name}_nx", type=box, size=[w, a, h], pos=[-(a - w), 0, 0], rgba=rgba, mass=0.03)
    body.add_geom(name=f"{cover.name}_py", type=box, size=[a - 2 * w, w, h], pos=[0, a - w, 0], rgba=rgba, mass=0.03)
    body.add_geom(name=f"{cover.name}_ny", type=box, size=[a - 2 * w, w, h], pos=[0, -(a - w), 0], rgba=rgba, mass=0.03)

    L, ph = C.COVER_ROD_HALF_LEN, C.COVER_POST_HALF
    post_z, rod_z = h + ph[2], h + 2 * ph[2] + 0.002
    rod_rgba = list(C.COVER_ROD_RGBA)
    for side, sy in (("p", 1), ("n", -1)):
        body.add_geom(name=f"{cover.name}_post_{side}", type=box, size=list(ph),
                      pos=[0, sy * (L - 0.004), post_z], rgba=rod_rgba, mass=0.005)
    rod = body.add_geom(name=f"{cover.name}_rod", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[C.COVER_ROD_RADIUS, 0, 0], rgba=rod_rgba, mass=0.01)
    rod.fromto = [0, -L, rod_z, 0, L, rod_z]
    body.add_site(name=cover.grasp_site, pos=[0, 0, rod_z], size=[0.005, 0, 0])

    # Lifting: inactive weld; attach() sets its relative pose and turns it on.
    eq = spec.add_equality(name=cover.weld, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1="g_base", name2=cover.name)
    eq.active = False
    # MjSpec's default equality data starts (0, 1, 0): for a weld that is the anchor point,
    # i.e. a point 1 m from the cover, which makes the weld fling the arm. The XML default is 0.
    data = np.array(eq.data, float)
    data[0:3] = 0.0
    eq.data = data


def _add_obstacle(spec, obs):
    body = spec.worldbody.add_body(name=obs.name, pos=[*obs.xy, obs.height / 2], quat=_yaw_quat(obs.yaw))
    body.add_geom(name=f"{obs.name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[*obs.half_xy, obs.height / 2], rgba=list(C.OBSTACLE_RGBA))


def _add_target(spec, info):
    body = spec.worldbody.add_body(name="target", pos=[*info.target_xy, C.TARGET_HALF],
                                   quat=_yaw_quat(info.target_yaw))
    body.add_freejoint(name="target_joint")
    body.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[C.TARGET_HALF] * 3,
                  rgba=list(C.TARGET_RGBA), mass=0.05)


def build_spec(info):
    spec = mujoco.MjSpec.from_file(C.BASE_SCENE)
    for name in C.BASE_SCENE_OBJECTS:
        spec.delete(spec.body(name))
    for cover in info.covers:            # covers before target: fixes the qpos order
        _add_cover(spec, cover)
    for obs in info.obstacles:
        _add_obstacle(spec, obs)
    _add_target(spec, info)
    return spec


# ---------------------------------------------------------------- public API

def reset(model, data, key=C.START_KEY):
    """Arm at `key`, objects at their generated poses."""
    mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
    mujoco.mj_forward(model, data)


def _start_pose_collides(model):
    """True if the arm touches any scene object (table excluded) at the start keyframe."""
    data = mujoco.MjData(model)
    reset(model, data)
    robot_root = model.body_rootid[model.body("g_base").id]
    table = model.body("table").id
    for con in data.contact[:data.ncon]:
        roots = model.body_rootid[model.geom_bodyid[[con.geom1, con.geom2]]]
        if robot_root in roots and not np.isin(roots, [table, 0]).any():
            return True
    return False


RENDERED_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)     # geom groups the renderer draws


def _target_sample_points(xy, yaw):
    """Points covering the target's surface: 5x5 on top, 5x3 on each side, on a cube HIDE_SAMPLE_SCALE larger
    than the target (so slivers peeking past an edge are still caught)."""
    h = C.HIDE_SAMPLE_SCALE * C.TARGET_HALF
    th = 2 * C.HIDE_SAMPLE_SCALE * C.TARGET_HALF
    g = np.linspace(-h, h, 5)
    z = np.linspace(0.2 * th, th, 3)
    local = [(a, b, th) for a in g for b in g]
    for a in g:
        for zz in z:
            local += [(h, a, zz), (-h, a, zz), (a, h, zz), (a, -h, zz)]
    local = np.array(local)
    c, s = np.cos(yaw), np.sin(yaw)
    out = local.copy()
    out[:, 0] = xy[0] + c * local[:, 0] - s * local[:, 1]
    out[:, 1] = xy[1] + s * local[:, 0] + c * local[:, 1]
    return out


def _target_visible(model, data, xy, yaw):
    """True if any sample point of a target at (xy, yaw) is visible from the detection camera."""
    cam = model.camera(C.DETECT_CAMERA).id
    cam_pos, R = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)
    W, H = model.cam_resolution[cam]
    ty = C.HIDE_FOV_SCALE * np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)   # slightly wider than the image
    tx = ty * W / H
    target_body = model.body("target").id
    # Rays start at the renderer's near clip plane: the camera sits ~1 mm from the wrist
    # mesh, which the renderer never draws (closer than znear) but a ray would hit.
    near = model.vis.map.znear * model.stat.extent
    geomid = np.zeros(1, np.int32)
    for p in _target_sample_points(xy, yaw):
        local = R.T @ (p - cam_pos)                       # camera frame, looking along -z
        if local[2] >= 0 or abs(local[0] / -local[2]) > tx or abs(local[1] / -local[2]) > ty:
            continue                                      # outside the image
        v = p - cam_pos
        dist = np.linalg.norm(v)
        u = v / dist
        hit = mujoco.mj_ray(model, data, cam_pos + near * u, u, RENDERED_GROUPS, 1, target_body, geomid)
        if hit < 0 or hit > dist - near - 1e-3:           # nothing in between
            return True
    return False


def _hide_target(model, info, rng):
    """Move an open target to a spot not visible from the start view. Returns True on success."""
    data = mujoco.MjData(model)
    reset(model, data)
    placed = [(c.xy, COVER_RADIUS) for c in info.covers]
    placed += [(o.xy, float(np.hypot(*o.half_xy))) for o in info.obstacles]
    xy = info.target_xy
    for _ in range(C.TARGET_HIDE_TRIES):
        if not _target_visible(model, data, xy, info.target_yaw):
            info.target_xy = xy
            return True
        xy = _sample_point(rng, TARGET_RADIUS, placed, gap=0.01)
        if xy is None:
            return False
    return False


def make_scene(seed, n_covers=C.N_COVERS, n_obstacles=C.N_OBSTACLES, p_covered=C.P_TARGET_COVERED,
               hide_open_target=C.HIDE_OPEN_TARGET, max_layouts=50):
    """Compile a random scene. Returns (model, spec, info). Same seed -> same scene."""
    rng = np.random.default_rng(seed)
    for _ in range(max_layouts):
        info = _sample_layout(rng, seed, n_covers, n_obstacles, p_covered)
        if info is None:
            continue
        spec = build_spec(info)
        model = spec.compile()
        if _start_pose_collides(model):
            continue
        if hide_open_target and info.target_cover is None:
            start_xy = info.target_xy.copy()
            if not _hide_target(model, info, rng):
                continue
            if not np.array_equal(start_xy, info.target_xy):   # target moved: rebuild with the new spot
                spec = build_spec(info)
                model = spec.compile()
        return model, spec, info
    raise RuntimeError(f"seed {seed}: no valid layout in {max_layouts} tries; "
                       "use fewer objects or a smaller MIN_GAP")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--covers", type=int, default=C.N_COVERS)
    ap.add_argument("--obstacles", type=int, default=C.N_OBSTACLES)
    ap.add_argument("--p-covered", type=float, default=C.P_TARGET_COVERED)
    ap.add_argument("--show-open-target", action="store_true", help="allow an open target visible from the start")
    ap.add_argument("--save", metavar="XML", help="write the generated scene as MJCF")
    ap.add_argument("--no-view", action="store_true")
    args = ap.parse_args()

    model, spec, info = make_scene(args.seed, args.covers, args.obstacles, args.p_covered,
                                   hide_open_target=not args.show_open_target)
    print(info.summary())
    if args.save:
        with open(args.save, "w") as f:
            f.write(spec.to_xml())
        print(f"wrote {args.save}")
    if not args.no_view:
        data = mujoco.MjData(model)
        reset(model, data)
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
