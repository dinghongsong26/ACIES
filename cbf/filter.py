"""Second-order (acceleration) CBF safety filter and its constraint rows.

Builds Lie-derivative constraint rows for circles, axis-aligned rectangles, world
bounds, and a time-varying finite-time goal, then solves a slack-relaxed CBF-QP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "CircleObstacle",
    "RectangleObstacle",
    "cbf_filter_acceleration",
]


@dataclass(frozen=True)
class CircleObstacle:
    """Disk obstacle (or another agent) in the plane."""

    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class RectangleObstacle:
    """Axis-aligned solid box obstacle (center ``x``, ``y`` and positive half-extents)."""

    x: float
    y: float
    half_w: float
    half_h: float


def rect_signed_distance_and_derivatives(
    px: float,
    py: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Signed distance to one closed axis-aligned rectangle, with derivatives.

    Positive outside, negative inside. Outside is the Euclidean distance to the
    set; inside is the negated min distance to the boundary (piecewise smooth,
    Hessian approximated).

    Returns:
        Tuple of (h, gradient (2,), Hessian (2, 2)).
    """

    # Outside deltas (distance along each axis if "past" the slab)
    if px < xmin:
        dx = xmin - px
        sgn_x = -1.0
    elif px > xmax:
        dx = px - xmax
        sgn_x = 1.0
    else:
        dx = 0.0
        sgn_x = 0.0

    if py < ymin:
        dy = ymin - py
        sgn_y = -1.0
    elif py > ymax:
        dy = py - ymax
        sgn_y = 1.0
    else:
        dy = 0.0
        sgn_y = 0.0

    if dx > 0.0 or dy > 0.0:
        # Outside (including corners)
        if dx == 0.0:
            h = dy
            g = np.array([0.0, sgn_y])
            H = np.zeros((2, 2))
        elif dy == 0.0:
            h = dx
            g = np.array([sgn_x, 0.0])
            H = np.zeros((2, 2))
        else:
            h = float(np.hypot(dx, dy))
            invh = 1.0 / h
            # h = sqrt(dx^2+dy^2), ddx/dpx = -sgn_x side... dx = xmin-px -> d dx /d px = -1 if px<xmin
            ddx_dpx = -1.0 if px < xmin else 1.0  # px > xmax branch
            ddy_dpy = -1.0 if py < ymin else 1.0
            gx = invh * dx * ddx_dpx
            gy = invh * dy * ddy_dpy
            g = np.array([gx, gy])
            # Hessian: h = sqrt(dx^2+dy^2), H = (1/h) I - (1/h^3) d d^T with d=(dx,dy)
            invh3 = invh**3
            H = invh * np.eye(2) - invh3 * np.array([[dx * dx, dx * dy], [dx * dy, dy * dy]])
        return h, g, H

    # Inside: h = -min margin to boundary
    m1 = px - xmin
    m2 = xmax - px
    m3 = py - ymin
    m4 = ymax - py
    margins = np.array([m1, m2, m3, m4])
    k = int(np.argmin(margins))
    dmin = float(margins[k])
    h = -dmin
    H = np.zeros((2, 2))
    if k == 0:
        g = np.array([-1.0, 0.0])
    elif k == 1:
        g = np.array([1.0, 0.0])
    elif k == 2:
        g = np.array([0.0, -1.0])
    else:
        g = np.array([0.0, 1.0])
    return h, g, H


def _cbf_accel_rows_rectangles(
    px: float,
    py: float,
    vx: float,
    vy: float,
    self_radius: float,
    obstacles: Sequence[RectangleObstacle],
    gamma1: float,
    gamma2: float,
    dist_floor: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Second-order CBF rows for static axis-aligned rectangles (Lie derivatives of ``h``)."""
    _ = dist_floor  # reserved for tuning / call-site compatibility
    obs_rows: list[np.ndarray] = []
    obs_rhs: list[float] = []
    v = np.array([vx, vy], dtype=float)
    # Match pole placement used in :func:`_cbf_accel_rows_circles`.
    alpha1 = gamma1 + gamma2
    alpha2 = gamma1 * gamma2
    for obs in obstacles:
        xmin = float(obs.x - obs.half_w)
        xmax = float(obs.x + obs.half_w)
        ymin = float(obs.y - obs.half_h)
        ymax = float(obs.y + obs.half_h)
        h, gh, Hh = rect_signed_distance_and_derivatives(px, py, xmin, xmax, ymin, ymax)
        h = float(h) - float(self_radius)
        gh = np.asarray(gh, dtype=float).reshape(2)
        Hh = np.asarray(Hh, dtype=float).reshape(2, 2)
        Lfh = float(gh @ v)
        Lf2h_plus = float(v @ Hh @ v)
        rhs_i = -Lf2h_plus - alpha1 * Lfh - alpha2 * h
        obs_rows.append(gh.reshape(1, 2))
        obs_rhs.append(rhs_i)

    if not obs_rows:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(obs_rows), np.asarray(obs_rhs, dtype=float)


def _unit2(x: float, y: float) -> Tuple[float, float]:
    """Normalize ``(x, y)``, returning ``(1, 0)`` for near-zero vectors."""
    n = float(np.hypot(x, y))
    if n < 1e-9:
        return 1.0, 0.0
    return x / n, y / n

def _cbf_accel_rows_circles(
    px: float,
    py: float,
    vx: float,
    vy: float,
    self_radius: float,
    obstacles: Sequence[CircleObstacle],
    margin: float,
    gamma1: float,
    gamma2: float,
    obstacle_velocities: Optional[Sequence[Tuple[float, float]]],
    dist_floor: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Second-order CBF rows for circular obstacles, optionally moving.

    Raises:
        ValueError: If ``obstacle_velocities`` length differs from ``obstacles``.
    """
    rows_a: list[list[float]] = []
    rows_b: list[float] = []
    n_obs = len(obstacles)
    if obstacle_velocities is not None and len(obstacle_velocities) != n_obs:
        raise ValueError(
            "obstacle_velocities must have the same length as obstacles"
        )

    for i, obs in enumerate(obstacles):
        dx = px - obs.x
        dy = py - obs.y
        dist = float(np.hypot(dx, dy))
        dist = max(dist, dist_floor)
        nx, ny = _unit2(dx, dy)
        r_sum = float(self_radius + obs.radius + margin)
        h = float(np.hypot(px - obs.x, py - obs.y)) - r_sum
        vx_o, vy_o = (0.0, 0.0)
        if obstacle_velocities is not None:
            vx_o, vy_o = obstacle_velocities[i]
        vrx = vx - vx_o
        vry = vy - vy_o
        hd = nx * vrx + ny * vry
        vn2 = vrx * vrx + vry * vry
        curved = (vn2 - hd * hd) / dist
        rhs = (
            -curved
            - (gamma1 + gamma2) * hd
            - gamma1 * gamma2 * h
        )
        rows_a.append([nx, ny])
        rows_b.append(rhs)

    if not rows_a:
        return np.zeros((0, 2)), np.zeros(0)
    return np.asarray(rows_a, dtype=float), np.asarray(rows_b, dtype=float)


def _cbf_accel_rows_bounds(
    px: float,
    py: float,
    vx: float,
    vy: float,
    radius: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    gamma1: float,
    gamma2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounds, second-order CBF in acceleration."""
    g12 = gamma1 + gamma2
    g1g2 = gamma1 * gamma2
    rows_a: list[list[float]] = []
    rows_b: list[float] = []

    h_left = (px - radius) - xmin
    rows_a.append([1.0, 0.0])
    rows_b.append(-g12 * vx - g1g2 * h_left)

    h_right = xmax - (px + radius)
    rows_a.append([-1.0, 0.0])
    rows_b.append(g12 * vx - g1g2 * h_right)

    h_bot = (py - radius) - ymin
    rows_a.append([0.0, 1.0])
    rows_b.append(-g12 * vy - g1g2 * h_bot)

    h_top = ymax - (py + radius)
    rows_a.append([0.0, -1.0])
    rows_b.append(g12 * vy - g1g2 * h_top)

    return np.asarray(rows_a, dtype=float), np.asarray(rows_b, dtype=float)


def _cbf_accel_rows_tv_goal(
    px: float,
    py: float,
    vx: float,
    vy: float,
    goal: Tuple[float, float],
    t: float,
    T: float,
    alpha: float,
    gamma: float,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Finite-time goal CBF row for ``h = T - t - alpha * dist(pos, goal)``.

    Returns:
        Tuple of (constraint row, rhs, h); empty rows when ``T`` or ``alpha`` <= 0.
    """
    if T <= 0.0 or alpha <= 0.0:
        return np.zeros((0, 2)), np.zeros(0), 0.0
    tx, ty = float(goal[0]), float(goal[1])
    eps = 1e-6
    dx = float(px - tx)
    dy = float(py - ty)
    dist = float(np.hypot(dx, dy))
    dist = max(dist, eps)
    nx = dx / dist
    ny = dy / dist
    h = float(T - t - alpha * dist)
    hd = float(nx * vx + ny * vy)
    finite_term = float(gamma * np.sign(h) * (abs(h) ** rho))
    rhs = float((-1.0 - alpha * hd + finite_term) / alpha)

    a_row = np.array([[nx, ny]], dtype=float)
    b_row = np.array([rhs], dtype=float)
    return a_row, b_row, h


def cbf_filter_acceleration(
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    a_nom: Union[Tuple[float, float], Sequence[float]],
    self_radius: float,
    obstacles: Sequence[CircleObstacle],
    *,
    rect_obstacles: Sequence[RectangleObstacle] = (),
    world_bounds: Optional[Tuple[float, float, float, float]] = None,
    margin: float = 2.0,
    gamma1: float = 5.0,
    gamma2: float = 5.0,
    a_max: Optional[float] = None,
    obstacle_velocities: Optional[Sequence[Tuple[float, float]]] = None,
    slack_weight: float = 1e4,
    tv_goal_position: Optional[Tuple[float, float]] = None,
    tv_goal_t: Optional[float] = None,
    tv_goal_T: Optional[float] = None,
    tv_goal_alpha: float = 1.0,
    tv_goal_gamma: float = 1.0,
    tv_goal_rho: float = 0.5,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Minimally modify ``a_nom`` to satisfy all CBF constraints via a QP.

    Assembles rows for circle and rectangle obstacles, optional world bounds, and
    an optional time-varying goal, then solves a slack-relaxed QP that stays closest
    to ``a_nom`` (CLARABEL when ``a_max`` caps the norm, else OSQP). With no
    constraints, returns ``a_nom`` clipped to ``a_max``.

    Returns:
        Tuple of (safe acceleration (2,), info dict with feasibility, solver status,
        objective, slack norm, and ``tv_goal_h`` when the goal CBF is active).

    Raises:
        ImportError: If cvxpy is not installed.
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cbf_filter_acceleration requires cvxpy (and a QP solver such as osqp). "
            "Install with: pip install cvxpy osqp"
        ) from e

    px, py = float(position[0]), float(position[1])
    vx, vy = float(velocity[0]), float(velocity[1])
    a_nom_v = np.asarray(a_nom, dtype=float).reshape(2)

    A1, b1 = _cbf_accel_rows_circles(
        px,
        py,
        vx,
        vy,
        self_radius,
        obstacles,
        margin,
        gamma1,
        gamma2,
        obstacle_velocities,
    )
    A1r, b1r = _cbf_accel_rows_rectangles(
        px,
        py,
        vx,
        vy,
        self_radius,
        rect_obstacles,
        gamma1,
        gamma2,
    )
    if len(A1) and len(A1r):
        A1 = np.vstack([A1, A1r])
        b1 = np.concatenate([b1, b1r])
    elif len(A1r):
        A1, b1 = A1r, b1r

    if world_bounds is not None:
        xmin, xmax, ymin, ymax = world_bounds
        A2, b2 = _cbf_accel_rows_bounds(
            px,
            py,
            vx,
            vy,
            self_radius,
            xmin,
            xmax,
            ymin,
            ymax,
            gamma1,
            gamma2,
        )
        A = np.vstack([A1, A2]) if len(A1) else A2
        b = np.concatenate([b1, b2]) if len(b1) else b2
    else:
        A, b = A1, b1

    # tv_h: Optional[float] = None
    # if (
    #     tv_goal_position is not None
    #     and tv_goal_t is not None
    #     and tv_goal_T is not None
    # ):
    #     A_tv, b_tv, h_tv = _cbf_accel_rows_tv_goal(
    #         px,
    #         py,
    #         vx,
    #         vy,
    #         tv_goal_position,
    #         float(tv_goal_t),
    #         float(tv_goal_T),
    #         float(tv_goal_alpha),
    #         float(tv_goal_gamma),
    #         float(tv_goal_rho),
    #     )
    #     if A_tv.shape[0] > 0:
    #         tv_h = h_tv
    #         if A.shape[0] > 0:
    #             A = np.vstack([A, A_tv])
    #             b = np.concatenate([b, b_tv])
    #         else:
    #             A, b = A_tv, b_tv

    n_con = A.shape[0]
    a = cp.Variable(2)
    if n_con == 0:
        print("No CBF constraints, returning a_nom_v")
        a_star = np.asarray(a_nom_v, dtype=float).copy()
        if a_max is not None and np.linalg.norm(a_star) > a_max + 1e-9:
            a_star = a_star * (a_max / np.linalg.norm(a_star))
        return a_star, {
            "feasible": True,
            "status": "no_cbf_constraints",
            "objective": float(np.sum((a_star - a_nom_v) ** 2)),
            "slack_norm": 0.0,
        }

    slack = cp.Variable(n_con, nonneg=True)
    constraints = [A @ a + slack >= b]
    if a_max is not None:
        constraints.append(cp.norm(a, 2) <= a_max)

    cost = cp.sum_squares(a - a_nom_v) + slack_weight * cp.sum_squares(slack)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    if a_max is not None:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    else:
        prob.solve(solver=cp.OSQP, verbose=False)

    status = prob.status
    feasible = status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    a_val = np.asarray(a.value, dtype=float).reshape(2)
    if not feasible or np.any(~np.isfinite(a_val)):
        a_val = np.zeros(2)
        feasible = False

    slack_norm = (
        float(np.linalg.norm(slack.value))
        if slack.value is not None
        else None
    )
    obj = float(prob.value) if prob.value is not None else None
    info: dict[str, Any] = {
        "feasible": feasible,
        "status": str(status),
        "objective": obj,
        "slack_norm": slack_norm,
    }
    # if tv_h is not None:
    #     info["tv_goal_h"] = float(tv_h)
    return a_val, info
