"""High-level entry points: pick parameters by task name, then run the CBF-QP."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from cbf.configs import CBFTaskConfig, get_cbf_config
from cbf.filter import CircleObstacle, RectangleObstacle, cbf_filter_acceleration

Array2 = np.ndarray


def filter_acceleration_for_task(
    task: str,
    position: Tuple[float, float],
    velocity: Tuple[float, float],
    a_nom: Union[Tuple[float, float], Sequence[float]],
    obstacles: Sequence[CircleObstacle] = (),
    *,
    rect_obstacles: Sequence[RectangleObstacle] = (),
    use_world_bounds: bool = True,
    obstacle_velocities: Optional[Sequence[Tuple[float, float]]] = None,
    config_overrides: Optional[CBFTaskConfig] = None,
    margin: Optional[float] = None,
    gamma1: Optional[float] = None,
    gamma2: Optional[float] = None,
    a_max: Optional[float] = None,
    robot_radius: Optional[float] = None,
    slack_weight: Optional[float] = None,
    navigation_target: Optional[Tuple[float, float]] = None,
) -> Tuple[Array2, dict[str, Any]]:
    """
    Run :func:`cbf.filter.cbf_filter_acceleration` with defaults from ``cbf.configs``.

    When ``sim_time`` and ``navigation_target`` are both set, a time-varying finite-time
    goal CBF row is added (using ``cfg.T_max`` as ``T``).
    """
    cfg = config_overrides if config_overrides is not None else get_cbf_config(task)
    bounds = cfg.world_bounds if use_world_bounds else None
    g1 = gamma1 if gamma1 is not None else cfg.gamma
    g2 = gamma2 if gamma2 is not None else (cfg.gamma2 if cfg.gamma2 is not None else cfg.gamma)
    tv_kw: dict[str, Any] = {}
    if navigation_target is not None:
        tv_kw = {
            "tv_goal_position": navigation_target,
            "tv_goal_T": float(cfg.T_max),
            "tv_goal_alpha": float(cfg.tv_goal_alpha),
            "tv_goal_gamma": float(cfg.tv_goal_gamma),
            "tv_goal_rho": float(cfg.tv_goal_rho),
        }
    return cbf_filter_acceleration(
        position,
        velocity,
        a_nom,
        robot_radius if robot_radius is not None else cfg.robot_radius,
        obstacles,
        rect_obstacles=rect_obstacles,
        world_bounds=bounds,
        margin=margin if margin is not None else cfg.margin,
        gamma1=g1,
        gamma2=g2,
        a_max=a_max if a_max is not None else cfg.a_max,
        obstacle_velocities=obstacle_velocities,
        slack_weight=slack_weight if slack_weight is not None else cfg.slack_weight,
        **tv_kw,
    )
