"""
Per-task CBF defaults aligned with pygame simulators in ``simulator/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Final, Optional, Tuple

Bounds = Tuple[float, float, float, float]


def _task_names() -> set:
    """Canonical task names = the contrauto/tasks/*.yaml mission specs."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "contrauto", "tasks")
    try:
        return {f[:-5] for f in os.listdir(d) if f.endswith(".yaml")}
    except OSError:
        return set()


@dataclass(frozen=True)
class CBFTaskConfig:
    """Geometry and tuning for one named task (keys match ``tasks.TASKS``)."""

    name: str
    world_bounds: Bounds
    robot_radius: float
    u_max: float
    gamma: float = 5.0
    margin: float = 0.0
    slack_weight: float = 1e3
    num_robots: int = 1
    T_max: float = 3.0  # horizon T for time-varying goal CBF (seconds of sim time)
    #: Max \|a\|_2 for double-integrator CBF-QP (pixels / s^2).
    a_max: float = None  # 400.0 is the default value
    #: Second class-K gain for relative-degree-2 CBF; defaults to ``gamma`` if None.
    gamma2: Optional[float] = None
    #: TV finite-time goal CBF: ``h = T - t - alpha * dist`` tuning (see ``cbf.filter``).
    tv_goal_alpha: float = 1.0
    tv_goal_gamma: float = 1.0
    tv_goal_rho: float = 0.5



CBF_TASK_CONFIGS: Final[Dict[str, CBFTaskConfig]] = {
    "warehouse": CBFTaskConfig(
        name="warehouse",
        world_bounds=(0.0, 900.0, 0.0, 800.0),
        robot_radius=14.0,
        u_max=130.0,
        num_robots=2,
        margin=2.0,  # buffer for two robots closing simultaneously (discrete step)
    ),
    "patrol": CBFTaskConfig(
        name="patrol",
        world_bounds=(0.0, 900.0, 0.0, 800.0),
        robot_radius=16.0,
        u_max=120.0,
    ),
    "static_2r_3b": CBFTaskConfig(
        name="static_2r_3b",
        world_bounds=(0.0, 880.0, 0.0, 800.0),
        robot_radius=15.0,
        u_max=140.0,
        num_robots=2,
        margin=2.0,  # buffer for two robots closing simultaneously (discrete step)
    ),
    "dynamic_2r_3b": CBFTaskConfig(
        name="dynamic_2r_3b",
        world_bounds=(0.0, 900.0, 0.0, 820.0),
        robot_radius=15.0,
        u_max=140.0,
        num_robots=2,
        margin=2.0,  # buffer for two robots closing simultaneously (discrete step)
    ),
    "two_layer_dock": CBFTaskConfig(
        name="two_layer_dock",
        world_bounds=(0.0, 1120.0, 0.0, 760.0),
        robot_radius=13.0,
        u_max=130.0,
        num_robots=2,
        margin=2.0,  # buffer for two robots closing simultaneously (discrete step)
    ),
    "two_layer_dock3": CBFTaskConfig(
        name="two_layer_dock3",
        world_bounds=(0.0, 1120.0, 0.0, 760.0),
        robot_radius=13.0,
        u_max=130.0,
        num_robots=3,
        margin=2.0,
    ),
    "two_layer_dock4": CBFTaskConfig(
        name="two_layer_dock4",
        world_bounds=(0.0, 1120.0, 0.0, 760.0),
        robot_radius=13.0,
        u_max=130.0,
        num_robots=4,
        margin=2.0,
    ),
    "tree_dock6": CBFTaskConfig(          # S2 tree-arbiter task (task-layer only; config
        name="tree_dock6",                # present so the import-time cross-check passes)
        world_bounds=(0.0, 1120.0, 0.0, 760.0),
        robot_radius=13.0,
        u_max=130.0,
        num_robots=6,
        margin=2.0,
    ),
    "charger": CBFTaskConfig(
        name="charger",
        world_bounds=(0.0, 1000.0, 0.0, 700.0),
        robot_radius=13.0,
        u_max=130.0,
        num_robots=3,
        margin=2.0,
    ),
    "corridor": CBFTaskConfig(
        name="corridor",
        world_bounds=(0.0, 1000.0, 0.0, 700.0),
        robot_radius=13.0,
        u_max=130.0,
        num_robots=3,
        margin=2.0,
    ),
    "warehouse5": CBFTaskConfig(
        name="warehouse5",
        world_bounds=(0.0, 900.0, 0.0, 800.0),
        robot_radius=14.0,
        u_max=130.0,
        num_robots=5,
        margin=2.0,  # buffer for five robots closing simultaneously (discrete step)
    ),
}


def _validate_task_keys() -> None:
    """Raise if any ``contrauto/tasks`` mission lacks a CBF config (run at import)."""
    tasks_keys = _task_names()
    if not tasks_keys:                       # tasks dir unavailable -> skip cross-check
        return
    cbf_keys = set(CBF_TASK_CONFIGS)
    missing = tasks_keys - cbf_keys          # a task with no CBF config is the real error
    if missing:
        raise RuntimeError(
            f"cbf.configs.CBF_TASK_CONFIGS missing config for contrauto tasks: {missing!r}"
        )


_validate_task_keys()


def get_cbf_config(task: str) -> CBFTaskConfig:
    """Return frozen parameters for ``task`` (must be a ``tasks.TASKS`` key)."""
    if task not in CBF_TASK_CONFIGS:
        raise KeyError(
            f"Unknown task {task!r}; expected one of {sorted(CBF_TASK_CONFIGS)}"
        )
    return CBF_TASK_CONFIGS[task]


def default_world_bounds(task: str) -> Bounds:
    """Convenience: world_bounds tuple for ``task``."""
    return get_cbf_config(task).world_bounds
