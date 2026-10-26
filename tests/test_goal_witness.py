"""Goal-witness check (contrauto.feedback.goal_witness) — anti cheap-exit.

Regression for the probe-verified done-mashing FSM: a justice goal over an output with no
input-referencing precondition is satisfiable by asserting the action vacuously.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.feedback.goal_witness import (
    unwitnessed_goal_outputs, format_witness_feedback,
)

_BASE = """\
CONTRACT Robot:
    input boolean variable at_shelf;
    input boolean variable delivered;
    output boolean variable go_to_shelf;
    output boolean variable done;
    Assumptions:
        true;
    Guarantees:
        G F(done);
{extra}
CONTRACT System:
    input boolean variable at_shelf;
    output boolean variable done;
    Assumptions:
        true;
    Guarantees:
        G F(done);
"""


def test_unguarded_goal_output_flagged():
    missing = unwitnessed_goal_outputs(_BASE.format(extra=""))
    assert ("Robot", "done") in missing
    fb = format_witness_feedback(missing)
    assert "done" in fb and "WITNESS" in fb.upper()


def test_witnessed_goal_output_passes():
    ok = _BASE.format(extra="        G(done -> delivered);\n")
    assert unwitnessed_goal_outputs(ok) == []


def test_output_only_guard_does_not_count():
    # guard referencing only outputs is not a witness
    bad = _BASE.format(extra="        G(done -> go_to_shelf);\n")
    assert ("Robot", "done") in unwitnessed_goal_outputs(bad)


def test_release_disjunct_not_treated_as_goal():
    # arbiter fairness G F(grant | !req): req appears only negated -> not an achievement
    arb = """\
CONTRACT Arbiter:
    input boolean variable req;
    output boolean variable grant;
    Assumptions:
        true;
    Guarantees:
        G F(grant | !req);
        G(grant -> req);
CONTRACT System:
    input boolean variable req;
    output boolean variable grant;
    Assumptions:
        true;
    Guarantees:
        G F(grant | !req);
"""
    assert unwitnessed_goal_outputs(arb) == []   # grant is guarded by G(grant -> req)
