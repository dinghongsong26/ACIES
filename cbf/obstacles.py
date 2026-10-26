"""Build :class:`cbf.filter.CircleObstacle` / :class:`cbf.filter.RectangleObstacle` lists for sims."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from cbf.filter import CircleObstacle, RectangleObstacle

XY = Tuple[float, float]


def disk_obstacles_at(
    positions: Sequence[XY],
    radius: float,
) -> List[CircleObstacle]:
    """Each position becomes a disk of the given radius."""
    r = float(radius)
    return [CircleObstacle(float(x), float(y), r) for x, y in positions]


def other_robots_as_obstacles(
    positions: Sequence[XY],
    robot_radius: float,
    self_index: int,
) -> List[CircleObstacle]:
    """
    Treat every robot except ``self_index`` as a moving disk of radius
    ``robot_radius`` (same as the controlled agent).
    """
    out: List[CircleObstacle] = []
    rr = float(robot_radius)
    for i, (x, y) in enumerate(positions):
        if i == self_index:
            continue
        out.append(CircleObstacle(float(x), float(y), rr))
    return out


def merge_obstacles(
    *groups: Iterable[CircleObstacle],
) -> List[CircleObstacle]:
    """Concatenate obstacle lists (static scenery + peers + boxes, etc.)."""
    merged: List[CircleObstacle] = []
    for g in groups:
        merged.extend(g)
    return merged


def warehouse_shelf_plank_rects(
    top: float = 50.0,
    x0: float = 80.0,
    shelf_w: float = 140.0,
    shelf_h: float = 20.0,
    col_gap: float = 80.0,
    row_gap: float = 110.0,
    rows: int = 4,
    cols: int = 3,
) -> List[RectangleObstacle]:
    """
    Axis-aligned shelf planks matching ``simulator.warehouse``:
    ``pygame.Rect(sx, sy, shelf_w, shelf_h)`` with
    ``sx = x0 + col * (shelf_w + col_gap)``, ``sy = top + row * (shelf_h + row_gap)``.

    The defaults are the ORIGINAL simulator.warehouse layout constants; that module now
    passes its own current SHELF_TOP/SHELF_X0/etc. explicitly (see its `_cbf_obstacles`)
    so a purely-cosmetic relayout there can never silently desync the physical obstacle
    geometry from the drawn shelf positions -- these rectangles are real CBF obstacles,
    not decoration, so a robot's avoidance must always match wherever the shelf is drawn.
    """
    half_w, half_h = shelf_w / 2.0, shelf_h / 2.0
    obs: List[RectangleObstacle] = []
    for row in range(rows):
        sy = top + row * (shelf_h + row_gap)
        cy = sy + shelf_h / 2.0
        for col in range(cols):
            sx = x0 + col * (shelf_w + col_gap)
            cx = sx + shelf_w / 2.0
            obs.append(RectangleObstacle(cx, cy, half_w, half_h))
    return obs


def boxes_as_disks(
    box_centers: Sequence[XY],
    box_half_size: float,
) -> List[CircleObstacle]:
    """
    Approximate square boxes as circumscribing disks (conservative clearance).

    ``box_half_size`` is half the side length of the square (e.g. OBJ_SZ/2).
    """
    r = float(box_half_size) * (2**0.5)
    return disk_obstacles_at(box_centers, r)
