"""Counterstrategy-guided assumption mining (contrauto.feedback.gr1_assume).

Tool-free layers:
  * candidate_assumptions — held input -> justice template (G F(x) / G F(!x)).
  * add_assumption — inject a formula into one contract's Assumptions, byte-preserving the rest.
  * mine_assumptions — the loop, driven by a MOCK ToolPipeline (held inputs + a realizability
    oracle), asserts only assumptions that restore realizability are kept.
Plus a tool-gated integration probe that skips unless CHASE/SLUGS are configured.

Run:  python3 -m pytest tests/test_gr1_assume.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from contrauto.feedback.gr1_assume import (
    candidate_assumptions, add_assumption, mine_assumptions, format_assume_feedback,
)


# ── candidate templates ───────────────────────────────────────────────────────
def test_candidate_for_input_held_false():
    assert candidate_assumptions({"grant": "FALSE"}) == ["G F(grant)"]


def test_candidate_for_input_held_true():
    assert candidate_assumptions({"req": "TRUE"}) == ["G F(!req)"]


def test_candidates_order_and_multiple():
    held = {"a": "FALSE", "b": "TRUE"}
    assert candidate_assumptions(held) == ["G F(a)", "G F(!b)"]


# ── add_assumption injection ──────────────────────────────────────────────────
_LTL = """\
CONTRACT Robot:
    input boolean variable grant;
    output boolean variable req;
    Assumptions:
        !grant;
    Guarantees:
        !req;
        G F(req);
CONTRACT System:
    output boolean variable req;
    Assumptions:
        true;
    Guarantees:
        G F(req);
"""


def test_add_assumption_targets_right_contract():
    out = add_assumption(_LTL, "Robot", "G F(grant)")
    robot = out.split("CONTRACT System:")[0]
    sysb = out.split("CONTRACT System:")[1]
    # added to Robot's assumptions, before its Guarantees
    assert "G F(grant);" in robot
    assert robot.index("G F(grant);") < robot.index("Guarantees:")
    # System contract untouched
    assert "G F(grant);" not in sysb
    # existing lines preserved
    assert "!grant;" in robot and "G F(req);" in robot


def test_add_assumption_preserves_other_contracts_verbatim():
    out = add_assumption(_LTL, "Robot", "G F(grant)")
    # only one new line added overall
    assert out.count("G F(grant);") == 1
    assert "CONTRACT System:" in out


# ── mining loop with a mock oracle ────────────────────────────────────────────
class _MockTools:
    """Stand-in ToolPipeline: held inputs are fixed; realizable iff the variant contains the
    'good' assumption."""
    def __init__(self, held, good_assumption):
        self._held = held
        self._good = good_assumption
    def counterstrategy_held_inputs(self, ctrl, in_vars):
        return dict(self._held)
    def check_realizable(self, ctrl, content):
        return self._good in content      # True (realizable) once the good assumption is present


def test_mine_keeps_only_fixing_assumption():
    held = {"grant": "FALSE"}
    tools = _MockTools(held, good_assumption="G F(grant);")
    working, got_held = mine_assumptions(tools, _LTL, "Robot", {"grant"})
    assert working == ["G F(grant)"]
    assert got_held == held


def test_mine_returns_empty_when_no_template_fixes_it():
    held = {"grant": "FALSE"}
    tools = _MockTools(held, good_assumption="SOMETHING_ELSE")
    working, got_held = mine_assumptions(tools, _LTL, "Robot", {"grant"})
    assert working == []
    assert got_held == held          # held still reported (context for feedback)


def test_mine_empty_when_no_held_inputs():
    tools = _MockTools({}, good_assumption="G F(grant);")
    working, held = mine_assumptions(tools, _LTL, "Robot", {"grant"})
    assert working == [] and held == {}


def test_feedback_lists_working_assumptions():
    msg = format_assume_feedback("Robot", ["G F(grant)"], {"grant": "FALSE"})
    assert "G F(grant)" in msg and "grant=FALSE" in msg
    # the "no fix found" branch still reports the held inputs
    msg2 = format_assume_feedback("Robot", [], {"grant": "FALSE"})
    assert "grant=FALSE" in msg2


# ── live-toolchain integration (skips unless CHASE/SLUGS configured) ───────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WAREHOUSE = os.path.join(ROOT, "outputs/warehouse/workdir/specs.ltl")


def test_assumption_mining_restores_realizability(tmp_path):
    if not (os.environ.get("LOGICS_TOOL") and os.environ.get("SLUGS")
            and os.environ.get("SLUGS_COMP") and os.path.exists(_WAREHOUSE)):
        pytest.skip("needs configured CHASE/SLUGS toolchain")
    from contrauto.agents.tool_pipeline import ToolPipeline
    content = open(_WAREHOUSE).read()
    t = ToolPipeline(tmp_path, "wh", tmp_path / "s.ltl", tmp_path / "v.chase", tmp_path / "y.chase")
    if t.check_realizable("Robot", content) is not True:
        pytest.skip("toolchain probe failed")
    # Remove the env reaction that lets the robot FORCE at_shelf; now the environment can
    # withhold at_shelf and G F(done) becomes unrealizable -> mining should propose G F(at_shelf).
    broken = content.replace("        G(nav_to_shelf -> X(at_shelf));\n", "", 1)
    if t.check_realizable("Robot", broken) is not False:
        pytest.skip("constructed spec was not unrealizable on this toolchain")
    # seed the controller's .slugsin for the counterstrategy
    t._save(t.ltl_file, broken)
    ci = t.write_chase_command_files(broken)
    t.run_synthesis()
    t.synthesize_controller("Robot")
    from contrauto.feedback.gr1_assume import mine_assumptions
    working, held = mine_assumptions(t, broken, "Robot", {"at_shelf", "has_item", "at_dock",
                                                          "delivered", "task_assigned"})
    assert held, "expected the counterstrategy to hold some input fixed"
    assert any("at_shelf" in w for w in working), (working, held)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ── transition/persistence mining (the pulse-grant trap) ──────────────────────
from contrauto.feedback.gr1_assume import (
    contract_io, candidate_transitions, mine_transitions, format_transition_feedback,
)

_PULSE_LTL = """\
CONTRACT Robot:
    input boolean variable grant;
    input boolean variable at_dock;
    output boolean variable nav;
    output boolean variable dropoff;
    Assumptions:
        G F(grant | !nav);
    Guarantees:
        G(nav -> grant);
        G F(dropoff);
CONTRACT System:
    output boolean variable nav;
    Assumptions:
        true;
    Guarantees:
        G F(dropoff);
"""

_PULSE_CORE = ["G(nav -> grant)", "G F(dropoff)"]


def test_contract_io_reads_declarations():
    ins, outs = contract_io(_PULSE_LTL, "Robot")
    assert ins == ["grant", "at_dock"] and outs == ["nav", "dropoff"]


def test_transition_candidates_only_from_core_variables():
    cands = candidate_transitions(_PULSE_CORE, {"grant", "at_dock"}, {"nav", "dropoff"})
    # at_dock is NOT in the core -> no candidates over it; G/F operator letters are not atoms
    assert "G((grant & nav) -> X(grant))" in cands
    assert "G((grant & dropoff) -> X(grant))" in cands
    assert all("at_dock" not in c for c in cands)


def test_mine_transitions_keeps_only_verified_persistence():
    tools = _MockTools({}, good_assumption="G((grant & nav) -> X(grant));")
    singles, needed = mine_transitions(tools, _PULSE_LTL, "Robot", _PULSE_CORE)
    assert singles == ["G((grant & nav) -> X(grant))"] and needed == []


def test_mine_transitions_skips_already_present_assumption():
    with_assump = _PULSE_LTL.replace(
        "        G F(grant | !nav);\n",
        "        G F(grant | !nav);\n        G((grant & nav) -> X(grant));\n")
    tools = _MockTools({}, good_assumption="G((grant & nav) -> X(grant));")
    singles, needed = mine_transitions(tools, with_assump, "Robot", _PULSE_CORE)
    # the already-present candidate is not re-proposed (the mock oracle passes every variant
    # here since the good assumption is already in the content, so others may appear)
    assert "G((grant & nav) -> X(grant))" not in singles + needed


def test_transition_feedback_names_sibling_guarantee_crosscheck():
    fb = format_transition_feedback("Robot", ["G((grant & nav) -> X(grant))"])
    assert "verified by re-synthesis" in fb
    assert "CROSS-CHECK" in fb and "GUARANTEE" in fb


def test_transition_feedback_set_mode():
    fb = format_transition_feedback("Robot", [], ["G((a & b) -> X(a))", "G((c & d) -> X(c))"])
    assert "SET" in fb and "minimal" in fb


def test_mine_transitions_set_when_no_single_works():
    class _SetTools:
        # realizable only when BOTH persistence assumptions are present
        def check_realizable(self, ctrl, content):
            return "G((grant & nav) -> X(grant))" in content and \
                   "G((at_dock & nav) -> X(at_dock))" in content
    core2 = ["G(nav -> grant)", "G(dropoff -> at_dock)", "G F(dropoff)"]
    singles, needed = mine_transitions(_SetTools(), _PULSE_LTL, "Robot", core2)
    assert singles == []
    assert set(needed) == {"G((grant & nav) -> X(grant))", "G((at_dock & nav) -> X(at_dock))"}


def test_mine_transitions_iterative_two_clusters():
    # Symmetric resources: realizable only when BOTH grant-persistence assumptions are in.
    # Round 1's core only shows cluster 1; the recomputed core surfaces cluster 2.
    LTL2 = """\
CONTRACT Robot:
    input boolean variable g1;
    input boolean variable g2;
    output boolean variable r1;
    output boolean variable r2;
    Assumptions:
        G F(g1 | !r1);
        G F(g2 | !r2);
    Guarantees:
        G(r1 -> g1);
        G(r2 -> g2);
        G F(r2);
CONTRACT System:
    output boolean variable r1;
    Assumptions:
        true;
    Guarantees:
        G F(r2);
"""
    P1 = "G((g1 & r1) -> X(g1))"
    P2 = "G((g2 & r2) -> X(g2))"

    class _TwoClusterTools:
        def check_realizable(self, ctrl, content):
            return P1 in content and P2 in content

    from contrauto.feedback.gr1_assume import mine_transitions_iterative
    needed = mine_transitions_iterative(
        _TwoClusterTools(), LTL2, "Robot", ["G(r1 -> g1)"],
        recompute_core=lambda c: ["G(r2 -> g2)"])
    assert set(needed) == {P1, P2}
