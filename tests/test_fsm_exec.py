"""Unit tests for the name-based FsmController (contrauto.sim_iface).

Run:  python3 -m pytest tests/test_fsm_exec.py     (or)  python3 tests/test_fsm_exec.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.sim_iface import FsmController


def _tiny_fsm():
    """Mealy strategy over input `req`, output `act`: act == req.
    node0 = (req=0, act=0); node1 = (req=1, act=1); each transitions by next req."""
    return {
        "variables": ["req", "act"],
        "nodes": {
            "0": {"rank": 0, "state": [0, 0], "trans": [0, 1]},
            "1": {"rank": 0, "state": [1, 1], "trans": [0, 1]},
        },
    }


def test_transition_and_output():
    c = FsmController(_tiny_fsm(), in_vars=["req"], out_vars=["act"])
    assert c.current == "0"
    # req=1 -> move to node1 -> act=1
    assert c.step({"req": 1}) == {"act": 1}
    assert c.current == "1"
    # req=0 -> back to node0 -> act=0
    assert c.step({"req": 0}) == {"act": 0}
    assert c.current == "0"


def test_hold_on_offmodel_input():
    """No successor AND no other node matches an impossible env -> hold, count stuck."""
    fsm = {
        "variables": ["req", "act"],
        "nodes": {"0": {"rank": 0, "state": [0, 0], "trans": []}},  # dead end
    }
    c = FsmController(fsm, in_vars=["req"], out_vars=["act"])
    out = c.step({"req": 1})
    assert c.current == "0" and c._stuck == 1 and out == {"act": 0}


def test_resync_when_successors_miss_but_another_node_matches():
    """Pushed off a location AP: no outgoing edge matches, but some other node does."""
    fsm = {
        "variables": ["at", "go"],
        "nodes": {
            "0": {"rank": 0, "state": [1, 0], "trans": [0]},   # thinks at=1
            "1": {"rank": 0, "state": [0, 1], "trans": [1]},   # at=0 -> go
        },
    }
    c = FsmController(fsm, in_vars=["at"], out_vars=["go"])
    assert c.step({"at": 0}) == {"go": 1}
    assert c.current == "1" and c._stuck == 0


def test_prefers_acting_successor_over_lazy_idle():
    """When two successors match the inputs, pick the one that asserts an output."""
    fsm = {
        "variables": ["req", "grant"],
        "nodes": {
            "0": {"rank": 0, "state": [0, 0], "trans": [1, 2]},
            "1": {"rank": 0, "state": [1, 0], "trans": [1, 2]},  # req, idle
            "2": {"rank": 0, "state": [1, 1], "trans": [1, 2]},  # req, grant
        },
    }
    c = FsmController(fsm, in_vars=["req"], out_vars=["grant"])
    assert c.step({"req": 1}) == {"grant": 1}
    assert c.current == "2"


def test_requests_reads_output_bits():
    fsm = {
        "variables": ["g", "req"],
        "nodes": {"0": {"rank": 0, "state": [0, 1], "trans": [0]}},
    }
    c = FsmController(fsm, in_vars=["g"], out_vars=["req"])
    assert c.requests(["req"]) == {"req": 1}


def test_reset():
    c = FsmController(_tiny_fsm(), in_vars=["req"], out_vars=["act"])
    c.step({"req": 1})
    c.reset()
    assert c.current == "0" and c._stuck == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all FsmController tests passed")


def test_bounded_progress_sentinel_fires_on_stall():
    # A world that never progresses must yield progress_violations=1 with channel-5 text.
    from contrauto.sim_iface import run_fsm

    class _IdleWorld:
        def arbiter(self): return None
        def agents(self): return []
        def metrics(self): return {"delivered": 0}
    import contrauto.sim_iface as si
    orig = si.fsm_step
    si.fsm_step = lambda *a, **k: None
    try:
        m = run_fsm(_IdleWorld(), {}, steps=2000, dt=1/60.0)
    finally:
        si.fsm_step = orig
    assert m.get("progress_violations") == 1
    assert "BOUNDED-PROGRESS" in m.get("progress_feedback", "")


def test_bounded_progress_sentinel_quiet_when_progressing():
    from contrauto.sim_iface import run_fsm

    class _BusyWorld:
        def __init__(self): self.n = 0
        def arbiter(self): return None
        def agents(self): return []
        def metrics(self):
            self.n += 1
            return {"delivered": self.n // 5}
    import contrauto.sim_iface as si
    orig = si.fsm_step
    si.fsm_step = lambda *a, **k: None
    try:
        m = run_fsm(_BusyWorld(), {}, steps=2000, dt=1/60.0)
    finally:
        si.fsm_step = orig
    assert "progress_violations" not in m


def test_offscript_sentinel_reports_stuck_controllers():
    from contrauto.sim_iface import run_fsm

    class _StuckCtrl:
        _stuck = 120
    class _W:
        def arbiter(self): return None
        def agents(self): return []
        def metrics(self): return {"delivered": 5}
    import contrauto.sim_iface as si
    orig = si.fsm_step
    si.fsm_step = lambda *a, **k: None
    try:
        m = run_fsm(_W(), {"Robot": _StuckCtrl()}, steps=200, dt=1/60.0)
    finally:
        si.fsm_step = orig
    assert m.get("offscript_violations") == 1
    assert "OFF-SCRIPT" in m.get("offscript_feedback", "")
