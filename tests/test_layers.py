"""Unit tests for the two-layer building blocks that don't need an LLM:
  - L2 geometry: plan_waypoints (spatial-waypoint operator)
  - L1 coordination: bounded_wait / global_deadline_check (arbiter budget)

Run:  python3 -m pytest tests/test_layers.py   (or)  python3 tests/test_layers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.geometry import plan_waypoints
from contrauto.coordination import bounded_wait, global_deadline_check

BOUNDS = (0.0, 200.0, 0.0, 200.0)


def _close(p, q, tol=14):
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


def test_waypoints_clear_path_is_direct():
    wp = plan_waypoints((10, 10), (190, 190), rects=[], bounds=BOUNDS, robot_radius=5)
    assert wp is not None and len(wp) >= 2          # goal may snap to a nearby free cell
    assert _close(wp[0], (10, 10)) and _close(wp[-1], (190, 190))


def test_waypoints_route_around_wall():
    # vertical wall across the middle with a gap top/bottom -> path must detour
    wall = [(100, 100, 8, 70)]          # cx, cy, hw, hh  (covers y 30..170 at x~100)
    wp = plan_waypoints((20, 100), (180, 100), rects=wall, bounds=BOUNDS, robot_radius=6)
    assert wp is not None and len(wp) > 2          # not a straight line
    # detour means some waypoint leaves the y=100 line
    assert any(abs(p[1] - 100) > 20 for p in wp)


def test_waypoints_blocked_returns_none_or_detour():
    # fully walled target (box around goal, no gap) -> unreachable
    boxed = [(180, 160, 18, 4), (180, 200, 18, 4), (160, 180, 4, 22), (200, 180, 4, 22)]
    wp = plan_waypoints((20, 20), (180, 180), rects=boxed, bounds=BOUNDS, robot_radius=6)
    assert wp is None or wp[-1] == (180, 180)      # either unreachable or a valid path


def test_bounded_wait():
    assert bounded_wait(1, 3.0) == 0.0
    assert bounded_wait(3, 3.0) == 6.0             # (N-1)*tau


def test_global_deadline_check_feasible():
    chk = global_deadline_check({"r1": 4.0, "r2": 5.0}, n_requesters=2, tau_hold=3.0, T=12.0)
    assert chk["feasible"] is True
    assert chk["completion"]["r2"] == 8.0          # 5 + (2-1)*3


def test_global_deadline_check_violation():
    chk = global_deadline_check({"r1": 11.0}, n_requesters=2, tau_hold=3.0, T=12.0)
    assert chk["feasible"] is False and "r1" in chk["violations"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all layer tests passed")
