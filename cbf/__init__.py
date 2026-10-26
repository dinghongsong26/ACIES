"""
Control barrier functions for ContrAuto simulators.
"""

from cbf.api import filter_acceleration_for_task
from cbf.configs import CBF_TASK_CONFIGS, CBFTaskConfig, get_cbf_config
from cbf.filter import CircleObstacle, RectangleObstacle, cbf_filter_acceleration
from cbf.motion import step_double_integrator_toward_target_cbf
from cbf.obstacles import (
    boxes_as_disks,
    disk_obstacles_at,
    merge_obstacles,
    other_robots_as_obstacles,
    warehouse_shelf_plank_rects,
)

__all__ = [
    "CBF_TASK_CONFIGS",
    "CBFTaskConfig",
    "CircleObstacle",
    "RectangleObstacle",
    "boxes_as_disks",
    "cbf_filter_acceleration",
    "disk_obstacles_at",
    "filter_acceleration_for_task",
    "get_cbf_config",
    "merge_obstacles",
    "other_robots_as_obstacles",
    "step_double_integrator_toward_target_cbf",
    "warehouse_shelf_plank_rects",
]
