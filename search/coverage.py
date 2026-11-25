"""Coverage grid over SEARCH_REGION: the search's belief about where the target can be.

Each cell (CELL_SIZE square) is a hypothesis "the target's centre is in this cell".
With a uniform prior and a perfect detector the belief is just one state per cell:

  POSSIBLE  not ruled out yet
  SEEN      ruled out: the camera saw bare table in this cell (so nothing stands there)
  SOLID     ruled out by known geometry: the target cannot be centred here because it
            would intersect a static obstacle or a cover's walls. The target is a
            TARGET_HALF cube; it contains a disc of radius TARGET_HALF in any yaw, so
              - obstacle: centre inside the footprint grown by TARGET_HALF  -> impossible
              - cover:    centre inside the outer footprint grown by TARGET_HALF but
                          outside the interior shrunk by TARGET_HALF        -> impossible
            So under each cover only a small central patch stays POSSIBLE.

SOLID is computed once from the initial poses: the target never moves, so a cell that
was impossible at the start stays impossible after a cover is moved away.
Cells under a cover stay POSSIBLE until the cover is lifted and the table there is seen.
"""
import numpy as np

from search import config as C

SEEN, POSSIBLE, SOLID = 0, 1, 2


def _local(xy, center, yaw):
    """Points (.., 2) expressed in a frame at `center` rotated by `yaw`."""
    c, s = np.cos(yaw), np.sin(yaw)
    d = xy - center
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


class CoverageGrid:
    def __init__(self, obstacles, covers, cell=C.CELL_SIZE):
        """obstacles / covers: lists from SceneInfo (poses are known; the target is NOT passed in)."""
        (x0, x1), (y0, y1) = C.SEARCH_REGION["x"], C.SEARCH_REGION["y"]
        self.cell, self.x0, self.y0 = cell, x0, y0
        self.nx, self.ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
        xc = x0 + (np.arange(self.nx) + 0.5) * cell
        yc = y0 + (np.arange(self.ny) + 0.5) * cell
        self.centers = np.stack(np.meshgrid(xc, yc, indexing="ij"), -1)      # (nx, ny, 2)
        self.state = np.full((self.nx, self.ny), POSSIBLE, np.uint8)

        th = C.TARGET_HALF
        for o in obstacles:
            p = np.abs(_local(self.centers, o.xy, o.yaw))
            self.state[(p[..., 0] < o.half_xy[0] + th) & (p[..., 1] < o.half_xy[1] + th)] = SOLID
        outer, inner = C.COVER_HALF + th, (C.COVER_HALF - 2 * C.COVER_WALL) - th
        for cv in covers:
            p = np.abs(_local(self.centers, cv.xy, cv.yaw))
            in_outer = (p[..., 0] < outer) & (p[..., 1] < outer)
            in_inner = (p[..., 0] <= inner) & (p[..., 1] <= inner)
            self.state[in_outer & ~in_inner] = SOLID

        self.n_start = self.n_possible                   # possible cells before any observation

    # ------------------------------------------------------------ queries

    @property
    def n_possible(self):
        return int(np.sum(self.state == POSSIBLE))

    @property
    def fraction_ruled_out(self):
        return 1.0 - self.n_possible / max(self.n_start, 1)

    def possible_centers(self):
        return self.centers[self.state == POSSIBLE]       # (K, 2)

    def cell_index(self, xy):
        """(i, j) integer indices and a mask of points that fall inside the grid."""
        ij = np.floor((np.asarray(xy) - [self.x0, self.y0]) / self.cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        return ij, ok

    # ------------------------------------------------------------ update

    def update(self, table_xy):
        """Rule out every POSSIBLE cell that contains a seen table point. Returns how many were ruled out."""
        if len(table_xy) == 0:
            return 0
        ij, ok = self.cell_index(table_xy)
        ij = ij[ok]
        hit = np.zeros_like(self.state, bool)
        hit[ij[:, 0], ij[:, 1]] = True
        newly = hit & (self.state == POSSIBLE)
        self.state[newly] = SEEN
        return int(newly.sum())
