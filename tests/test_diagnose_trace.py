"""NuXMV counterexample parsing in contrauto.feedback.diagnose.

Locks in three output facts the naive parse got wrong (each silently poisoned the
refinement diagnosis the LLM repairs from):
  1. per-LTLSPEC splitting — when BOTH refinement directions fail, the output holds
     two traces; parsing the whole dump spliced them into one bogus lasso.
  2. the '-- Loop starts here' marker (capital L) precedes the loop-entry state's
     header while that state is still being accumulated (+1), and the old check
     lowercased the line before searching for the capitalized needle, so it never
     fired and every lasso degenerated to a single-state loop.
  3. states after the first print only CHANGED variables — valuations accumulate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.feedback.diagnose import _parse_lasso_trace, _split_by_spec

FAKE_NUXMV = """\
*** This is nuXmv 2.1.0 ***
-- LTL specification (a -> b)  is false
-- as demonstrated by the following execution sequence
Trace Description: IC3 counterexample
Trace Type: Counterexample
  -> State: 1.1 <-
    p = TRUE
    q = FALSE
    r = FALSE
  -- Loop starts here
  -> State: 1.2 <-
    q = TRUE
  -> State: 1.3 <-
    q = FALSE
    r = TRUE
-- LTL specification (c -> d)  is false
-- as demonstrated by the following execution sequence
Trace Description: IC3 counterexample
Trace Type: Counterexample
  -> State: 2.1 <-
    p = FALSE
    q = FALSE
    r = TRUE
  -> State: 2.2 <-
  -- Loop starts here
  -> State: 2.3 <-
    p = TRUE
"""


def test_split_by_spec_two_failures():
    chunks = _split_by_spec(FAKE_NUXMV)
    assert len(chunks) == 2
    assert "State: 1.1" in chunks[0] and "State: 2.1" not in chunks[0]
    assert "State: 2.1" in chunks[1] and "State: 1.3" not in chunks[1]


def test_loop_marker_and_cumulative_valuations():
    prefix, loop = _parse_lasso_trace(_split_by_spec(FAKE_NUXMV)[0])
    # marker before State 1.2 -> prefix=[1.1], loop=[1.2, 1.3]
    assert len(prefix) == 1 and len(loop) == 2
    assert prefix[0] == {"p": True, "q": False, "r": False}
    # deltas inherit: 1.2 only prints q, p/r carry over from 1.1
    assert loop[0] == {"p": True, "q": True, "r": False}
    assert loop[1] == {"p": True, "q": False, "r": True}


def test_loop_marker_between_state_blocks():
    prefix, loop = _parse_lasso_trace(_split_by_spec(FAKE_NUXMV)[1])
    # marker after 2.2's (empty) block, before 2.3 -> loop=[2.3] only
    assert len(prefix) == 2 and len(loop) == 1
    assert loop[0] == {"p": True, "q": False, "r": True}


def test_missing_marker_falls_back_to_last_state():
    no_marker = "\n".join(l for l in FAKE_NUXMV.splitlines()
                          if "Loop starts" not in l)
    prefix, loop = _parse_lasso_trace(_split_by_spec(no_marker)[0])
    assert len(loop) == 1
    assert loop[0] == {"p": True, "q": False, "r": True}


FAKE_CONTRACT = """\
NAME: toy;

CONTRACT Robot:
    input boolean variable at_goal;
    output boolean variable req;
    Assumptions:
        G F(grant | !req);
    Guarantees:
        G F(req);

CONTRACT Arbiter:
    input boolean variable req;
    output boolean variable grant;
    Assumptions:
        G F(!req);
    Guarantees:
        G F(grant | !req);

CONTRACT System:
    input boolean variable at_goal;
    output boolean variable req;
    output boolean variable grant;
    Assumptions:
        true;
    Guarantees:
        G F(req);
"""


def test_output_owners_maps_internal_variables():
    from contrauto.feedback.diagnose import _output_owners
    owners = _output_owners(FAKE_CONTRACT)
    assert owners["req"] == ["Robot", "System"]
    assert owners["grant"] == ["Arbiter", "System"]


def test_assumption_feedback_names_sibling_guarantee_repair():
    from contrauto.feedback.diagnose import diagnose_refinement
    # LTLSPEC 1 (assumption direction) fails; the lasso keeps req high and grant low
    # forever, violating BOTH sides of the request/grant circularity.
    out = """\
-- LTL specification (x)  is false
Trace Description: IC3 counterexample
Trace Type: Counterexample
  -> State: 1.1 <-
    at_goal = FALSE
    req = TRUE
    grant = FALSE
  -- Loop starts here
  -> State: 1.2 <-
    at_goal = TRUE
-- LTL specification (y)  is false
Trace Description: IC3 counterexample
Trace Type: Counterexample
  -> State: 2.1 <-
    at_goal = FALSE
    req = TRUE
    grant = FALSE
  -- Loop starts here
  -> State: 2.2 <-
    at_goal = TRUE
"""
    d = diagnose_refinement(out, FAKE_CONTRACT, "toy")
    assert "Robot.assumption[0]" in d          # G F(grant | !req) flagged
    assert "Arbiter.assumption[0]" in d        # G F(!req) flagged
    # Robot and Arbiter assume over EACH OTHER's outputs -> the cycle is the root cause,
    # and the advice must NOT be "add matching guarantees" (that keeps the cycle).
    assert "CIRCULAR DISCHARGE" in d
    assert "part of the CIRCULAR DISCHARGE" in d
    # and no guarantee is mislabeled into the assumption-direction listing
    assert "Robot.guarantee" not in d.split("=== REPAIR OPTIONS ===")[0]


def test_assumption_feedback_sibling_hint_when_no_cycle():
    from contrauto.feedback.diagnose import diagnose_refinement
    # Same trace, but the Arbiter assumes nothing (true;) -> no cycle: the Robot's
    # grant assumption gets the plain sibling-guarantee repair hint.
    no_cycle = FAKE_CONTRACT.replace("        G F(!req);\n", "        true;\n", 1)
    assert no_cycle != FAKE_CONTRACT
    out = """\
-- LTL specification (x)  is false
Trace Description: IC3 counterexample
Trace Type: Counterexample
  -> State: 1.1 <-
    at_goal = FALSE
    req = TRUE
    grant = FALSE
  -- Loop starts here
  -> State: 1.2 <-
    at_goal = TRUE
"""
    d = diagnose_refinement(out, no_cycle, "toy")
    assert "CIRCULAR DISCHARGE" not in d
    assert "OUTPUT(s) of Arbiter" in d
