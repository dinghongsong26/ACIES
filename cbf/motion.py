"""Velocity-tracking double-integrator stepping with go-around steering.

Steers a desired velocity that orbits obstacles, tracks it through the CBF-QP, and
falls back to a clipped Euler step when the filter is unavailable.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from cbf.filter import (
    CircleObstacle,
    RectangleObstacle,
    rect_signed_distance_and_derivatives,
)


def _steer_desired_velocity(
    x: float,
    y: float,
    tx: float,
    ty: float,
    v_cap: float,
    robot_radius: float,
    obstacles: Sequence[CircleObstacle],
    obstacle_velocities: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    rect_obstacles: Sequence[RectangleObstacle] = (),
    influence: float = 60.0,
    repulse: float = 0.35,
) -> Tuple[float, float]:
    """
    Desired velocity toward ``(tx, ty)`` with a tangential "go-around" component
    near obstacles (circular peers/boxes *and* rectangular shelves).

    The pure go-to-goal direction is blended with a circulation (vortex) field so
    the robot orbits the unsafe region instead of pushing straight into it. Without
    this, the CBF-QP cancels the radial part of the nominal acceleration and the
    robot stalls at the boundary of the unsafe set -- e.g. wedged below a shelf
    plank that sits between it and the goods, or head-on against another robot.
    """
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return 0.0, 0.0
    gx, gy = dx / d, dy / d  # unit direction to goal

    # Accumulator seeded with the goal direction; tangential terms bend it.
    acc = [gx, gy]

    def _contribute(nx, ny, surf_dist, clearance, approach):
        """Add a go-around contribution for one obstacle (nx,ny = outward normal)."""
        if clearance >= influence:
            return
        # Only circle obstacles that lie *between* the robot and the goal. If the
        # obstacle surface is as far as (or farther than) the goal, the goal is not
        # hidden behind it -- e.g. goods resting on the very shelf we approach. In
        # that case circulating would push us off target; just drive in and let the
        # CBF-QP hold the safe standoff.
        if surf_dist >= d:
            return
        ahead = -(gx * nx + gy * ny)  # goal direction points into the obstacle?
        if ahead <= 0.0 and approach <= 0.0:
            return
        ahead = max(ahead, 0.0)
        w = max(0.0, min(1.0, (influence - clearance) / influence))
        w *= max(ahead, 0.3 if approach > 0.0 else 0.0)
        if w <= 0.0:
            return
        # Tangent perpendicular to n; pick the side that makes progress to the goal.
        p1x, p1y = -ny, nx
        dot1 = p1x * gx + p1y * gy
        if abs(dot1) < 1e-3:
            sgn = 1.0  # exact head-on: consistent rule so peers veer apart
        else:
            sgn = 1.0 if dot1 > 0.0 else -1.0
        acc[0] += w * (sgn * p1x + repulse * nx)
        acc[1] += w * (sgn * p1y + repulse * ny)

    for i, obs in enumerate(obstacles):
        rx, ry = x - obs.x, y - obs.y  # obstacle centre -> robot
        dist = math.hypot(rx, ry)
        if dist < 1e-6:
            continue
        nx, ny = rx / dist, ry / dist
        # Closing speed along the line of sight (positive => approaching) so a peer
        # moving away does not trigger a needless detour.
        approach = 0.0
        if obstacle_velocities is not None and i < len(obstacle_velocities):
            ovx, ovy = obstacle_velocities[i]
            approach = -((-nx) * ovx + (-ny) * ovy)
        _contribute(nx, ny, dist - obs.radius,
                    dist - (robot_radius + obs.radius), approach)

    for rect in rect_obstacles:
        xmin, xmax = rect.x - rect.half_w, rect.x + rect.half_w
        ymin, ymax = rect.y - rect.half_h, rect.y + rect.half_h
        h, g, _ = rect_signed_distance_and_derivatives(x, y, xmin, xmax, ymin, ymax)
        if h <= 1e-6:  # inside / on the plank: let the CBF push us out
            continue
        gnorm = math.hypot(float(g[0]), float(g[1]))
        if gnorm < 1e-9:
            continue
        nx, ny = float(g[0]) / gnorm, float(g[1]) / gnorm  # outward unit normal
        _contribute(nx, ny, h, h - robot_radius, 0.0)

    n = math.hypot(acc[0], acc[1])
    if n < 1e-9:
        return 0.0, 0.0
    return v_cap * acc[0] / n, v_cap * acc[1] / n


def step_double_integrator_toward_target_cbf(
    task: str,
    x: float,
    y: float,
    vx: float,
    vy: float,
    tx: float,
    ty: float,
    dt: float,
    speed_scale: float,
    base_speed: float,
    robot_radius: float,
    obstacles: Sequence[CircleObstacle],
    obstacle_velocities: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    rect_obstacles: Sequence[RectangleObstacle] = (),
    min_dist: float = 1.0,
    kp: float = 4.0,
) -> Tuple[float, float, float, float]:
    """
    One Euler step for ``x_dot = v``, ``v_dot = a`` with ``a`` from CBF-QP.
    """
    _ = kp  # retained for call-site compatibility; controller is velocity-tracking
    dx = tx - x
    dy = ty - y
    d = math.hypot(dx, dy)
    v_cap = base_speed * speed_scale

    if d <= min_dist:
        v_des_x, v_des_y = 0.0, 0.0
    else:
        v_des_x, v_des_y = _steer_desired_velocity(
            x, y, tx, ty, v_cap, robot_radius, obstacles, obstacle_velocities,
            rect_obstacles=rect_obstacles,
        )

    # Velocity-tracking nominal controller. A positional spring (kp * dx) is
    # deliberately omitted: it points straight at the goal through obstacles and
    # is exactly what the CBF cancels, producing the boundary deadlock. Tracking a
    # steered desired velocity lets the robot circle around instead.
    kd = 7.5 if d < 1.5 else 5.5
    a_nom_x = kd * (v_des_x - vx)
    a_nom_y = kd * (v_des_y - vy)

    def euler_fallback() -> Tuple[float, float, float, float]:
        try:
            from cbf.configs import get_cbf_config

            am = float(get_cbf_config(task).a_max)
        except Exception:
            am = 400.0
        ax, ay = a_nom_x, a_nom_y
        na = math.hypot(ax, ay)
        if na > am and na > 1e-9:
            ax, ay = ax * (am / na), ay * (am / na)
        vx_n = vx + ax * dt
        vy_n = vy + ay * dt
        return x + vx * dt, y + vy * dt, vx_n, vy_n

    try:
        from cbf.api import filter_acceleration_for_task
    except ImportError:
        return euler_fallback()

    try:
        a, _ = filter_acceleration_for_task(
            task,
            (x, y),
            (vx, vy),
            (a_nom_x, a_nom_y),
            obstacles,
            rect_obstacles=rect_obstacles,
            obstacle_velocities=obstacle_velocities,
            robot_radius=robot_radius,
            navigation_target=(tx, ty),
        )
    except ImportError:
        return euler_fallback()

    # update state
    ax, ay = float(a[0]), float(a[1])
    vx_n = vx + ax * dt
    vy_n = vy + ay * dt
    x_n = x + vx * dt
    y_n = y + vy * dt
    return x_n, y_n, vx_n, vy_n
