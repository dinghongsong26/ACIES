"""Unit checks for shared-dock physical admission."""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dock_target_waits_outside_until_granted():
    from simulator.warehouse import DOCK_X, DOCK_Y, ROBOT_R, _dock_target

    class Robot:
        status = ""

        def __init__(self, granted):
            self.granted = granted

        def has_dock_grant(self):
            return self.granted

    waiting = _dock_target(Robot(False), None)
    admitted = _dock_target(Robot(True), None)

    at_dock_threshold = DOCK_Y - ROBOT_R - 6
    assert waiting == (DOCK_X, DOCK_Y - ROBOT_R - 32)
    assert waiting[1] < at_dock_threshold
    assert admitted == (DOCK_X, DOCK_Y - ROBOT_R - 4)
    assert admitted[1] >= at_dock_threshold
