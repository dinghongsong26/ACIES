"""Unit tests for the simulator interface: Nav/Stationary act() + run_fsm driver.

Uses a tiny mock backend (no pygame), so it exercises the SCENARIO/CORE logic only.
Run:  python3 -m pytest tests/test_sim_iface.py   (or)  python3 tests/test_sim_iface.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.sim_iface import (
    SimAgent, SimWorld, Nav, Stationary, FsmController, run_fsm, fsm_step, role_of,
    bind_by_role,
)


# ── a tiny 1-D mock backend ───────────────────────────────────────────────────
class MockAgent(SimAgent):
    name = "r"
    SENSORS = {"at_goal": lambda a, w: a.x >= 10.0}
    ACTIONS = {"go": Nav(lambda a, w: (10.0, 0.0)),
               "stop": Stationary(effect=lambda a, w: setattr(a, "stopped", True))}

    def __init__(self):
        self.x = 0.0
        self.target = (0.0, 0.0)
        self.stopped = False

    def pose(self):
        return (self.x, 0.0)

    def tick(self, dt, world):
        tx = self.target[0]
        self.x += max(-1.0, min(1.0, tx - self.x))   # move up to 1 unit/tick toward target


class MockWorld(SimWorld):
    def __init__(self):
        self._a = MockAgent()

    def agents(self):
        return [self._a]

    def step(self, dt):
        pass

    def metrics(self):
        return {"x": self._a.x, "stopped": self._a.stopped}


def test_act_nav_sets_target():
    a = MockAgent()
    a.act({"go": 1, "stop": 0}, None)
    assert a.target == (10.0, 0.0)


def test_act_stationary_holds_and_effects():
    a = MockAgent()
    a.x = 5.0
    a.act({"go": 0, "stop": 1}, None)
    assert a.target == (5.0, 0.0)      # held at current pose
    assert a.stopped is True           # effect ran


def test_act_stationary_not_shadowed_by_nav():
    """When the FSM asserts Nav and Stationary together (e.g. pickup & goto_dock), the
    Stationary effect must still run — first-true-only selection used to skip the grasp."""
    a = MockAgent()
    a.x = 5.0
    a.act({"go": 1, "stop": 1}, None)
    assert a.stopped is True
    assert a.target == (10.0, 0.0)     # Nav still wins motion after the effect


def test_run_fsm_reaches_goal_then_stops():
    # FSM: at_goal=0 -> go ; at_goal=1 -> stop
    fsm = {
        "variables": ["at_goal", "go", "stop"],
        "nodes": {
            "0": {"rank": 0, "state": [0, 1, 0], "trans": [0, 1]},
            "1": {"rank": 0, "state": [1, 0, 1], "trans": [0, 1]},
        },
    }
    world = MockWorld()
    ctrl = FsmController(fsm, in_vars=["at_goal"], out_vars=["go", "stop"])
    m = run_fsm(world, {"r": ctrl}, steps=40)
    assert m["x"] >= 10.0          # navigated to the goal
    assert m["stopped"] is True    # then switched to the stationary action


def test_action_holding_skips_fsm_until_done():
    """A `done_when` that is not yet satisfied makes the driver KEEP the action and not
    advance the FSM; once satisfied, the next step advances. Legacy actions (no done_when)
    never hold."""
    calls = {"n": 0}

    class FakeCtrl:
        def step(self, env):
            calls["n"] += 1
            return {"work": 1}

        def requests(self, names):
            return {}

    class HoldAgent(SimAgent):
        name = "h"

        def __init__(self):
            self.work = 0
            self._cur = "work"                      # pretend 'work' is already in progress
            self.ACTIONS = {"work": Stationary(done_when=lambda a, w: a.work >= 3)}

        def pose(self):
            return (0.0, 0.0)

        def tick(self, dt, world):
            if self.work < 3:                        # physical progress happens in tick
                self.work += 1

    class HoldWorld(SimWorld):
        def __init__(self):
            self._a = HoldAgent()

        def agents(self):
            return [self._a]

        def step(self, dt):
            pass

        def metrics(self):
            return {}

    w = HoldWorld()
    for _ in range(3):                               # held: FSM must NOT advance
        fsm_step(w, {"h": FakeCtrl()}, 0.1)
    assert calls["n"] == 0, f"FSM advanced while holding: {calls}"
    assert w._a.work == 3                            # tick kept progressing
    fsm_step(w, {"h": FakeCtrl()}, 0.1)              # done_when satisfied -> advance once
    assert calls["n"] == 1


def test_no_done_when_never_holds():
    a = MockAgent()
    a.act({"go": 1, "stop": 0}, None)               # picked 'go' (legacy Nav, no done_when)
    assert a.holding(None) is False


def test_role_of_prefixed_go_is_nav():
    """dock3/corridor FSMs name motion r1_go_pickup / r1_go_dock. A prefix-blind
    startswith('go') bound those to pickup/at_dock and the robots never moved."""
    assert role_of("r1_go_pickup") == "go_to_shelf"
    assert role_of("r1_go_dock") == "go_to_dock"
    assert role_of("r2_go_pickup") == "go_to_shelf"
    assert role_of("go_pickup") == "go_to_shelf"
    assert role_of("go_to_dock") == "go_to_dock"
    # location / grasp must stay distinct from nav
    assert role_of("r1_at_pickup") == "at_shelf"
    assert role_of("r1_at_dock") == "at_dock"
    assert role_of("r1_pickup") == "pickup"


def test_bind_by_role_dock3_motion_vars():
    preds = {"at_shelf": lambda a, w: False, "at_dock": lambda a, w: False,
             "has_item": lambda a, w: False, "task": lambda a, w: False}
    acts = {"go_to_shelf": object(), "go_to_dock": object(),
            "pickup": object(), "dropoff": object()}
    vars_ = ["r1_at_pickup", "r1_at_dock", "r1_has_item", "r1_task",
             "r1_grant_dock", "r1_go_pickup", "r1_go_dock",
             "r1_pickup", "r1_dropoff", "r1_request_dock"]
    sensors, action_map, _, _, req, grant = bind_by_role(vars_, None, None, preds, acts)
    assert "r1_go_pickup" in action_map and action_map["r1_go_pickup"] is acts["go_to_shelf"]
    assert "r1_go_dock" in action_map and action_map["r1_go_dock"] is acts["go_to_dock"]
    assert "r1_go_dock" not in sensors
    assert sensors["r1_at_dock"] is preds["at_dock"]
    assert action_map["r1_pickup"] is acts["pickup"]
    assert req == ["r1_request_dock"] and grant == ["r1_grant_dock"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all sim_iface tests passed")
