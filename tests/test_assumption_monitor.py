"""Unit tests for the run-time assumption-conformance monitor (channel 5)."""
from contrauto.assumption_monitor import (
    AssumptionMonitorSet, build_monitor, compile_bool, format_feedback,
    parse_chase_assumptions, _LivenessMon, _SafetyMon, _PersistMon,
)


# ── boolean evaluator ────────────────────────────────────────────────────────
def test_bool_eval():
    f = compile_bool("at_shelf | !go_to_shelf")
    assert f({"at_shelf": 1, "go_to_shelf": 1}) is True
    assert f({"at_shelf": 0, "go_to_shelf": 1}) is False   # navigating, not arrived
    assert f({"at_shelf": 0, "go_to_shelf": 0}) is True    # not navigating -> vacuously ok
    g = compile_bool("grant -> request")
    assert g({"grant": 1, "request": 0}) is False
    assert g({"grant": 1, "request": 1}) is True
    assert compile_bool("!(a & b)")({"a": 1, "b": 1}) is False


# ── classification ───────────────────────────────────────────────────────────
def test_classification():
    assert isinstance(build_monitor("G F(arrived | !act)"), _LivenessMon)
    assert isinstance(build_monitor("G(grant -> request)"), _SafetyMon)
    assert isinstance(build_monitor("has_item W at_dock"), _PersistMon)
    assert build_monitor("!act") is None                    # initial condition
    assert build_monitor("true") is None
    assert build_monitor("G(act -> X(arrived))") is None    # X is not monitorable
    assert build_monitor("G(arrived & !act -> X(arrived))") is None


# ── the durative-stuck failure: liveness SUSPECTED after patience ────────────
def test_durative_stuck_liveness():
    ms = AssumptionMonitorSet({"Robot1": ["G F(at_shelf | !go_to_shelf)"]}, patience=5)
    fired = []
    for step in range(20):                                  # navigating forever, never arrives
        fired += ms.update({"go_to_shelf": 1, "at_shelf": 0}, step)
    assert len(fired) == 1
    v = fired[0]
    assert v.contract == "Robot1" and v.kind == "liveness" and v.suspected is True
    assert "never" in v.evidence.lower()


def test_durative_arrives_in_time_no_violation():
    ms = AssumptionMonitorSet({"Robot1": ["G F(at_shelf | !go_to_shelf)"]}, patience=5)
    for step in range(20):
        obs = {"go_to_shelf": 1, "at_shelf": 1 if step % 3 == 0 else 0}   # arrives periodically
        ms.update(obs, step)
    assert not ms.any_violation()


# ── safety: sound, immediate reject ──────────────────────────────────────────
def test_safety_confirmed():
    ms = AssumptionMonitorSet({"Arb": ["G(grant -> request)"]}, patience=5)
    ms.update({"grant": 0, "request": 0}, 0)
    fired = ms.update({"grant": 1, "request": 0}, 1)        # granted a non-requester
    assert len(fired) == 1 and fired[0].suspected is False and fired[0].kind == "safety"


# ── weak-until persistence ───────────────────────────────────────────────────
def test_persistence_W():
    ms = AssumptionMonitorSet({"R": ["has_item W at_dock"]}, patience=5)
    ms.update({"has_item": 1, "at_dock": 0}, 0)
    fired = ms.update({"has_item": 0, "at_dock": 0}, 1)     # dropped item before docking
    assert len(fired) == 1 and fired[0].kind == "persistence"

    ok = AssumptionMonitorSet({"R": ["has_item W at_dock"]}, patience=5)
    ok.update({"has_item": 1, "at_dock": 0}, 0)
    ok.update({"has_item": 1, "at_dock": 1}, 1)             # docked -> discharged
    ok.update({"has_item": 0, "at_dock": 0}, 2)             # now free to drop
    assert not ok.any_violation()


# ── .ltl extraction ──────────────────────────────────────────────────────────
def test_parse_chase_assumptions():
    ltl = """NAME: X;

CONTRACT Robot1:
    input boolean variable at_shelf;
    output boolean variable go_to_shelf;
    Assumptions:
        !at_shelf;
        G F(at_shelf | !go_to_shelf);
    Guarantees:
        !go_to_shelf;
        G F(go_to_shelf);
"""
    got = parse_chase_assumptions(ltl)
    assert got["Robot1"] == ["!at_shelf", "G F(at_shelf | !go_to_shelf)"]


# ── feedback rendering ───────────────────────────────────────────────────────
def test_feedback_text():
    ms = AssumptionMonitorSet({"Robot1": ["G F(at_shelf | !go_to_shelf)"]}, patience=3)
    for step in range(10):
        ms.update({"go_to_shelf": 1, "at_shelf": 0}, step)
    fb = ms.feedback()
    assert "channel 5" in fb and "Contract: Robot1" in fb
    assert "G F(effect | !action)" in fb and "TASK-level" in fb


def test_unparseable_u_body_skipped_not_fatal():
    # A mission `(p U q)` folded into assumptions must not kill the whole monitor set.
    from contrauto.assumption_monitor import monitor_from_ltl
    ltl = """\
CONTRACT Robot:
    input boolean variable at_shelf;
    input boolean variable grant;
    output boolean variable req;
    Assumptions:
        G(at_shelf & !req -> (at_shelf U req));
        G F(grant | !req);
    Guarantees:
        G F(req);
"""
    m = monitor_from_ltl(ltl)
    assert m is not None                       # the justice assumption is still watched
    cm = m.contracts["Robot"]
    assert any("G F" in mon.formula for mon in cm.monitors)
    assert any(" U " in s for s in cm.skipped)
