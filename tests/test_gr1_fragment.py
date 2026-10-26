"""Static GR(1)-fragment check (contrauto.feedback.gr1_fragment).

Pure-Python, no external tools: classifies each sub-contract guarantee AST as init /
safety / justice / invalid, and asserts that non-GR(1) guarantees (bare F, response
G(p->F(q)), Until, nested temporal) are flagged with formula-level localization while
valid ones and System-contract / assumption formulas are not.

Run:  python3 -m pytest tests/test_gr1_fragment.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.feedback.ltl_parser import parse
from contrauto.feedback.gr1_fragment import (
    classify_guarantee, check_gr1_fragment, format_gr1_feedback,
)


# ── classify_guarantee: the three valid GR(1) shapes ──────────────────────────
def test_initial_boolean_is_valid():
    assert classify_guarantee(parse("!grant"))[:2] == ("init", True)
    assert classify_guarantee(parse("a & !b"))[:2] == ("init", True)


def test_safety_is_valid():
    assert classify_guarantee(parse("G(!(pickup & dropoff))"))[:2] == ("safety", True)
    # one-step transition with a single X (X-depth 1)
    assert classify_guarantee(parse("G(req -> X(grant | !req))"))[:2] == ("safety", True)


def test_justice_is_valid():
    assert classify_guarantee(parse("G F(done)"))[:2] == ("justice", True)
    assert classify_guarantee(parse("G F(grant | !request)"))[:2] == ("justice", True)


# ── classify_guarantee: the invalid (non-GR(1)) shapes ────────────────────────
def test_response_is_rejected():
    kind, ok, reason = classify_guarantee(parse("G(req -> F(grant))"))
    assert kind == "invalid" and not ok
    assert "response" in reason.lower()


def test_bare_F_is_rejected():
    kind, ok, reason = classify_guarantee(parse("F(done)"))
    assert kind == "invalid" and not ok
    assert "bare f" in reason.lower()


def test_until_is_rejected():
    _, ok, reason = classify_guarantee(parse("a U b"))
    assert not ok and "until" in reason.lower()


def test_nested_temporal_safety_rejected():
    # G with an Until inside the safety body
    _, ok, _ = classify_guarantee(parse("G(p -> (q U r))"))
    assert not ok


def test_double_x_rejected():
    # X-depth 2 is not a one-step transition
    _, ok, _ = classify_guarantee(parse("G(a -> X(X(b)))"))
    assert not ok


# ── check_gr1_fragment: contract-level, scope, localization ───────────────────
_LTL = """\
CONTRACT Robot:
    input boolean variable grant;
    output boolean variable req;
    Assumptions:
        G F(grant);
    Guarantees:
        G F(req);
        G(req -> F(grant));
CONTRACT System:
    input boolean variable grant;
    output boolean variable req;
    Assumptions:
        true;
    Guarantees:
        G(req -> F(grant));
"""


def test_flags_only_subcontract_guarantee_violation():
    vs = check_gr1_fragment(_LTL)
    # exactly one violation: Robot.guarantee[1] (the response). The IDENTICAL response in
    # the System contract is skipped (System is not synthesized).
    assert len(vs) == 1
    v = vs[0]
    assert v.formula.contract == "Robot"
    assert v.formula.section == "guarantee"
    assert v.formula.index == 1
    assert "response" in v.reason.lower()


def test_valid_spec_has_no_violations():
    clean = """\
CONTRACT Robot:
    output boolean variable req;
    Assumptions:
        G F(grant);
    Guarantees:
        G F(req);
        G(!(req & done));
CONTRACT System:
    output boolean variable req;
    Guarantees:
        G F(req);
"""
    assert check_gr1_fragment(clean) == []


def test_assumption_response_is_allowed():
    # G(p -> F(q)) in an ASSUMPTION is valid env liveness -> not flagged.
    ltl = """\
CONTRACT Robot:
    output boolean variable req;
    Assumptions:
        G(req -> F(grant));
    Guarantees:
        G F(req);
CONTRACT System:
    output boolean variable req;
    Guarantees:
        G F(req);
"""
    assert check_gr1_fragment(ltl) == []


def test_assumption_until_is_flagged():
    ltl = """\
CONTRACT Robot:
    output boolean variable req;
    Assumptions:
        a U b;
    Guarantees:
        G F(req);
CONTRACT System:
    output boolean variable req;
    Guarantees:
        G F(req);
"""
    vs = check_gr1_fragment(ltl)
    assert len(vs) == 1 and vs[0].formula.section == "assumption"


def test_feedback_mentions_each_violating_formula():
    vs = check_gr1_fragment(_LTL)
    msg = format_gr1_feedback(vs, _LTL)
    assert "Robot.guarantee[1]" in msg
    assert "GR(1) FRAGMENT VIOLATION" in msg
    assert "Return corrected .ltl only." in msg


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_assumption_may_not_prime_own_outputs():
    # slugs game semantics: env assumptions cannot constrain the system's next move.
    # Previously this crashed the backend ("variable report' is not allowed").
    from contrauto.feedback.gr1_fragment import check_gr1_fragment
    ltl = """\
CONTRACT PatrolRobot:
    input boolean variable alert_A;
    output boolean variable report;
    Assumptions:
        G(alert_A -> X(report));
        G(report -> X(alert_A));
    Guarantees:
        G F(report);
CONTRACT System:
    output boolean variable report;
    Assumptions:
        true;
    Guarantees:
        G F(report);
"""
    v = check_gr1_fragment(ltl)
    bad = [x for x in v if "OWN output" in x.reason]
    assert len(bad) == 1                      # only the X(report) assumption is flagged
    assert "report" in bad[0].formula.raw_text and "X(report)" in bad[0].formula.raw_text
