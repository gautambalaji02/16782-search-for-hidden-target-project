"""Time scaling of a joint-space path (pure math, no MuJoCo).

The path is the polyline through the waypoints. It is parametrised by u, the time
the path takes at full speed: a segment's length in u is max_i |dq_i| / VMAX, so the
slowest-to-finish joint moves at exactly VMAX. Then u(t) follows a trapezoidal
profile (accelerate, cruise at speed 1, decelerate) so the arm starts and stops
smoothly. Joint speeds stay <= VMAX and accelerations <= AMAX within a segment;
at waypoint corners the direction changes abruptly (fine for stiff position servos;
path shortcutting later will reduce the number of corners).
"""
import numpy as np

from search import config as C


class Trajectory:
    def __init__(self, path, vmax=C.JOINT_VMAX, amax=C.JOINT_AMAX):
        self.path = np.asarray(path, float)                       # (K, n)
        seg = np.abs(np.diff(self.path, axis=0)).max(axis=1) / vmax
        self.u = np.concatenate([[0.0], np.cumsum(seg)])          # waypoint positions in u
        self.length = self.u[-1]
        self.acc = amax / vmax                                    # u acceleration that keeps joints <= amax

        # Trapezoid with cruise speed 1, or a triangle if the path is too short to reach it.
        t_ramp = 1.0 / self.acc
        if self.length < self.acc * t_ramp ** 2:                  # can't reach full speed
            t_ramp = np.sqrt(self.length / self.acc)
        self.t_ramp = t_ramp
        self.v_peak = self.acc * t_ramp
        self.duration = 2 * t_ramp + (self.length - self.acc * t_ramp ** 2) / max(self.v_peak, 1e-12)

    def _u_at(self, t):
        t = np.clip(t, 0.0, self.duration)
        a, tr = self.acc, self.t_ramp
        if t < tr:                                                # accelerating
            return 0.5 * a * t ** 2
        if t > self.duration - tr:                                # decelerating
            return self.length - 0.5 * a * (self.duration - t) ** 2
        return 0.5 * a * tr ** 2 + self.v_peak * (t - tr)        # cruising

    def _du_dt(self, t):
        if t <= 0 or t >= self.duration:
            return 0.0
        if t < self.t_ramp:
            return self.acc * t
        if t > self.duration - self.t_ramp:
            return self.acc * (self.duration - t)
        return self.v_peak

    def velocity(self, t):
        """Joint velocities (rad/s) at time t: du/dt times the slope dq/du of the current segment."""
        if self.length == 0:
            return np.zeros(self.path.shape[1])
        u = self._u_at(t)
        k = min(np.searchsorted(self.u, u, side="right") - 1, len(self.u) - 2)
        while k > 0 and self.u[k + 1] == self.u[k]:       # skip zero-length segments
            k -= 1
        du = self.u[k + 1] - self.u[k]
        slope = (self.path[k + 1] - self.path[k]) / du if du > 0 else 0.0
        return self._du_dt(t) * slope

    def __call__(self, t):
        """Joint targets at time t (s) after the start; holds the end point afterwards."""
        if self.length == 0:
            return self.path[-1].copy()
        u = self._u_at(t)
        return np.array([np.interp(u, self.u, self.path[:, j]) for j in range(self.path.shape[1])])
