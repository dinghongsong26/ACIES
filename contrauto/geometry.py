"""
Geometric spatial-waypoint planner — L2 spatial decomposition (`spatial-waypoint`).

Given a start, a goal, and rectangular obstacles (shelves), produce an ordered
sequence of waypoints that routes *around* the non-convex obstacle field so that
each consecutive pair is joined by an obstacle-free straight corridor. Each such
segment is a convex reach-avoid task a CBF can execute without the boundary
deadlock (local minimum) that plagues a purely reactive controller in a maze.

Pure geometry (stdlib only): inflate obstacles by the robot radius, grid A*,
then line-of-sight ("string-pulling") simplification. Determinism + completeness
are exactly what the reactive CBF lacks; that is why region generation is a
geometric algorithm, not an LLM proposal (see docs/DECOMPOSITION_DESIGN.md).
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

Rect = Tuple[float, float, float, float]  # (cx, cy, half_w, half_h)
Bounds = Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
XY = Tuple[float, float]


def _inflate(rects: Sequence[Rect], r: float) -> List[Rect]:
    return [(cx, cy, hw + r, hh + r) for (cx, cy, hw, hh) in rects]


def _in_rect(x: float, y: float, rect: Rect) -> bool:
    cx, cy, hw, hh = rect
    return (cx - hw) <= x <= (cx + hw) and (cy - hh) <= y <= (cy + hh)


def _in_obb(x: float, y: float, obb) -> bool:
    """True if (x,y) is inside an oriented box (cx, cy, half_w, half_h, angle).

    An axis-aligned rect is the special case angle=0; this is what lets the same A*
    serve both the simple sims (axis-aligned shelves) and the two-layer sims (rotated
    walls / corridors).
    """
    cx, cy, hw, hh, ang = obb
    dx, dy = x - cx, y - cy
    c, s = math.cos(-ang), math.sin(-ang)
    return abs(dx * c - dy * s) <= hw and abs(dx * s + dy * c) <= hh


def _point_free(x: float, y: float, infl: Sequence[Rect], bounds: Bounds, r: float) -> bool:
    """True if the robot centre at (x,y) is collision-free (obstacles pre-inflated)."""
    xmin, xmax, ymin, ymax = bounds
    if not (xmin + r <= x <= xmax - r and ymin + r <= y <= ymax - r):
        return False
    for rc in infl:
        if _in_rect(x, y, rc):
            return False
    return True


def _seg_hits_rect(p: XY, q: XY, rect: Rect) -> bool:
    """Exact segment vs axis-aligned box intersection (Liang-Barsky). Touch counts."""
    cx, cy, hw, hh = rect
    xmin, xmax, ymin, ymax = cx - hw, cx + hw, cy - hh, cy + hh
    dx, dy = q[0] - p[0], q[1] - p[1]
    pp = (-dx, dx, -dy, dy)
    qq = (p[0] - xmin, xmax - p[0], p[1] - ymin, ymax - p[1])
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(pp, qq):
        if pi == 0.0:
            if qi < 0.0:
                return False  # parallel to a slab and outside it
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return u1 <= u2


def _seg_clear(p: XY, q: XY, infl: Sequence[Rect], bounds: Bounds, r: float, step: float = 0.0) -> bool:
    """True if the straight corridor p->q is collision-free (exact, no sampling).

    Endpoints are assumed in-bounds; the bound box is convex, so a straight
    segment between two in-bound points stays in-bound — only obstacles matter.
    """
    return not any(_seg_hits_rect(p, q, rc) for rc in infl)


def _astar(start_c, goal_c, free, nx, ny):
    """8-connected grid A* over a free(cell)->bool predicate. Returns cell list or None."""
    def h(a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)  # octile

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    openq = [(h(start_c, goal_c), 0.0, start_c)]
    came = {start_c: None}
    gscore = {start_c: 0.0}
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal_c:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        cx, cy = cur
        for dx, dy in nbrs:
            nc = (cx + dx, cy + dy)
            if not (0 <= nc[0] < nx and 0 <= nc[1] < ny) or not free(nc):
                continue
            if dx and dy:  # block diagonal corner-cutting
                if not free((cx + dx, cy)) or not free((cx, cy + dy)):
                    continue
            ng = g + (math.sqrt(2) if dx and dy else 1.0)
            if ng < gscore.get(nc, math.inf):
                gscore[nc] = ng
                came[nc] = cur
                heapq.heappush(openq, (ng + h(nc, goal_c), ng, nc))
    return None


def _nearest_free_cell(c, free, nx, ny, max_cells=8000):
    """BFS for the closest free cell (goal may sit inside a shelf). First popped
    free cell is the nearest in grid distance — picks the true stand-off."""
    if free(c):
        return c
    from collections import deque
    nbrs = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    seen = {c}
    dq = deque([c])
    cnt = 0
    while dq and cnt < max_cells:
        cur = dq.popleft()
        cnt += 1
        for dx, dy in nbrs:
            nc = (cur[0] + dx, cur[1] + dy)
            if 0 <= nc[0] < nx and 0 <= nc[1] < ny and nc not in seen:
                if free(nc):
                    return nc
                seen.add(nc)
                dq.append(nc)
    return None


def _simplify(pts: List[XY], infl, bounds, r, step) -> List[XY]:
    """String-pulling: drop intermediate points connectable by a clear corridor."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _seg_clear(pts[i], pts[j], infl, bounds, r, step):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def plan_waypoints(
    start: XY,
    goal: XY,
    rects: Sequence,
    bounds: Bounds,
    robot_radius: float,
    cell: float = 12.0,
    *,
    obb: bool = False,
    simplify: bool = True,
) -> Optional[List[XY]]:
    """
    Ordered waypoints from ``start`` to ``goal`` routing around ``rects``.

    ``rects`` are axis-aligned ``(cx, cy, hw, hh)`` by default, or oriented boxes
    ``(cx, cy, hw, hh, angle)`` when ``obb=True``. Returns ``[start, ..., goal]`` where
    each consecutive pair has an obstacle-free straight corridor (a convex reach-avoid
    segment). For axis-aligned obstacles the final point is snapped to the nearest free
    point when ``goal`` lies inside an inflated obstacle (the robot stops at the closest
    safe stand-off); for oriented boxes the raw ``goal`` is kept as the last waypoint.
    Returns ``None`` if no route exists. ``simplify`` applies line-of-sight
    string-pulling (axis-aligned obstacles only).
    """
    xmin, xmax, ymin, ymax = bounds
    nx = int((xmax - xmin) / cell) + 1
    ny = int((ymax - ymin) / cell) + 1

    def to_xy(c):
        return (xmin + c[0] * cell, ymin + c[1] * cell)

    def to_cell(p):
        return (int(round((p[0] - xmin) / cell)), int(round((p[1] - ymin) / cell)))

    if obb:
        infl = [(cx, cy, hw + robot_radius, hh + robot_radius, a)
                for (cx, cy, hw, hh, a) in rects]

        def free(c):
            x, y = to_xy(c)
            return (xmin <= x <= xmax and ymin <= y <= ymax
                    and not any(_in_obb(x, y, o) for o in infl))
    else:
        infl = _inflate(rects, robot_radius)

        def free(c):
            x, y = to_xy(c)
            return _point_free(x, y, infl, bounds, robot_radius)

    start_c = _nearest_free_cell(to_cell(start), free, nx, ny)
    goal_c = _nearest_free_cell(to_cell(goal), free, nx, ny)
    if start_c is None or goal_c is None:
        return None

    cells = _astar(start_c, goal_c, free, nx, ny)
    if cells is None:
        return None

    last = goal if obb else to_xy(goal_c)
    pts: List[XY] = [start] + [to_xy(c) for c in cells] + [last]
    return _simplify(pts, infl, bounds, robot_radius, cell * 0.5) if (simplify and not obb) else pts


class WaypointNav:
    """
    Stateful waypoint follower for runtime use in a simulator.

    Replans (geometric `plan_waypoints`) when the goal changes meaningfully or every
    `replan_period` frames, and advances to the next waypoint once the current one is
    reached. Returns the current sub-goal to feed the CBF, so the robot routes around
    obstacles instead of deadlocking at the boundary while aiming straight at the goal.

    One class serves both obstacle kinds. The defaults handle axis-aligned shelves;
    pass `obb=True` for oriented boxes (rotated walls / corridors), which also switches
    to the tuning the two-layer sims use (larger cell, periodic replan, no string-pull).
    """

    def __init__(self, *, obb: bool = False, cell: float | None = None,
                 simplify: bool | None = None, switch_tol: float = 18.0,
                 goal_eps: float | None = None, replan_period: int | None = None):
        self.obb = obb
        self.cell = cell if cell is not None else (16.0 if obb else 12.0)
        self.simplify = simplify if simplify is not None else (not obb)
        self.switch_tol = switch_tol
        self.goal_eps = goal_eps if goal_eps is not None else (22.0 if obb else 5.0)
        self.replan_period = replan_period if replan_period is not None else (24 if obb else 0)
        self._wps: List[XY] | None = None
        self._i = 0
        self._goal: XY | None = None
        self._t = 0

    def target(self, pos: XY, goal: XY, obstacles, bounds, robot_radius: float) -> XY:
        """Current sub-goal for the CBF: replan if the goal moved (or periodically), advance once reached."""
        self._t += 1
        moved = (self._goal is None or
                 math.hypot(goal[0] - self._goal[0], goal[1] - self._goal[1]) > self.goal_eps)
        periodic = self.replan_period and self._t % self.replan_period == 0
        if self._wps is None or moved or periodic:
            wps = plan_waypoints(pos, goal, obstacles, bounds, robot_radius,
                                 cell=self.cell, obb=self.obb, simplify=self.simplify)
            self._wps = wps if (wps and len(wps) >= 2) else [tuple(pos), tuple(goal)]
            self._goal = (goal[0], goal[1])
            self._i = 1 if len(self._wps) > 1 else 0
        while (self._i < len(self._wps) - 1 and
               math.hypot(pos[0] - self._wps[self._i][0],
                          pos[1] - self._wps[self._i][1]) < self.switch_tol):
            self._i += 1
        return self._wps[self._i]


# ── self-test: warehouse 4x3 shelves, dock -> a box resting on a shelf ────────
if __name__ == "__main__":
    SHELF_W, SHELF_H, TOP = 140.0, 20.0, 50.0
    rects: List[Rect] = []
    for row in range(4):
        sy = TOP + row * (SHELF_H + 110.0)
        for col in range(3):
            sx = 80.0 + col * (SHELF_W + 80.0)
            rects.append((sx + SHELF_W / 2, sy + SHELF_H / 2, SHELF_W / 2, SHELF_H / 2))
    bounds = (0.0, 900.0, 0.0, 800.0)
    R = 14.0

    cases = [
        ("dock -> S2 box (top, behind 3 shelf rows)", (160.0, 490.0), (370.0, 60.0)),
        ("dock -> S7 box (mid-left)", (160.0, 490.0), (150.0, 320.0)),
        ("corner -> corner", (40.0, 760.0), (860.0, 60.0)),
    ]
    infl = _inflate(rects, R)
    for name, s, g in cases:
        wp = plan_waypoints(s, g, rects, bounds, R)
        if wp is None:
            print(f"[{name}] NO ROUTE")
            continue
        # verify every corridor is clear
        ok = all(_seg_clear(wp[i], wp[i + 1], infl, bounds, R, 2.0) for i in range(len(wp) - 1))
        total = sum(math.hypot(wp[i + 1][0] - wp[i][0], wp[i + 1][1] - wp[i][1])
                    for i in range(len(wp) - 1))
        straight = math.hypot(g[0] - s[0], g[1] - s[1])
        print(f"[{name}]")
        print(f"   waypoints({len(wp)}): {[(round(x), round(y)) for x, y in wp]}")
        print(f"   all corridors clear = {ok} | path len={total:.0f} vs straight={straight:.0f} "
              f"(detour x{total / straight:.2f})")
