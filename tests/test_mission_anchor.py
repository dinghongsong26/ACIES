"""Step 2.6 mission anchoring (contrauto.feedback.mission_anchor).

Regression for failure class 7 (spec erosion): flash dropped the translator's action
mutexes + carrying dynamics from its own System contract; every certificate passed and the
synthesized strategy asserted pickup&dropoff forever with zero deliveries.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.feedback.mission_anchor import (
    split_top_conjuncts, missing_mission_formulas, format_anchor_feedback,
)

_PAIRS = [
    {"nl": "goals", "ltl": "G F(r1_dropoff) & G F(r2_dropoff)"},
    {"nl": "mutex", "ltl": "G(!(r1_pickup & r1_dropoff))"},
    {"nl": "dyn",   "ltl": "!r1_carrying & G(r1_pickup -> X(r1_carrying))"},
]

_SYS_FULL = """\
CONTRACT Robot1:
    Guarantees:
        G F(r1_dropoff);
CONTRACT System:
    Assumptions:
        !r1_carrying;
        G(r1_pickup -> X(r1_carrying));
    Guarantees:
        G F(r1_dropoff);
        G F(r2_dropoff);
        G(!(r1_pickup & r1_dropoff));
"""


def test_split_respects_parentheses():
    assert split_top_conjuncts("G(a & b) & G F(c) & !d") == ["G(a & b)", "G F(c)", "!d"]
    assert split_top_conjuncts("G(!(a & b))") == ["G(!(a & b))"]


def test_full_system_passes():
    assert missing_mission_formulas(_PAIRS, _SYS_FULL) == []


def test_dropped_mutex_is_flagged():
    eroded = _SYS_FULL.replace("        G(!(r1_pickup & r1_dropoff));\n", "")
    missing = missing_mission_formulas(_PAIRS, eroded)
    assert missing == ["G(!(r1_pickup & r1_dropoff))"]
    fb = format_anchor_feedback(missing)
    assert "DROPPED" in fb and "G(!(r1_pickup & r1_dropoff))" in fb


def test_g_split_relaxation_counts_as_present():
    pairs = [{"nl": "x", "ltl": "G(a -> b & c -> d)"}]
    # System splits the one G into two — semantically legal, must count as present
    sysc = """\
CONTRACT System:
    Guarantees:
        G(a -> b);
        G(c -> d);
"""
    assert missing_mission_formulas(pairs, sysc) == []


def test_missing_system_block_reported():
    out = missing_mission_formulas(_PAIRS, "CONTRACT Robot1:\n    Guarantees:\n        !a;\n")
    assert out and "System" in out[0]


def test_placement_rule():
    from contrauto.feedback.mission_anchor import placement_of
    INS, OUTS = {"at_shelf", "at_dock", "task"}, {"pickup", "dropoff", "go_to_dock"}
    # env dynamics (X over input) -> assumption
    assert placement_of("G(at_shelf & !go_to_dock -> X(at_shelf))", INS, OUTS) == "assumption"
    # output persistence (X over output) -> guarantee
    assert placement_of("G(go_to_dock & !at_dock -> X(go_to_dock))", INS, OUTS) == "guarantee"
    # env fairness justice (mentions input) -> assumption
    assert placement_of("G F(at_shelf | !go_to_dock)", INS, OUTS) == "assumption"
    # pure-output justice -> guarantee
    assert placement_of("G F(pickup)", INS, OUTS) == "guarantee"
    # mutex over outputs -> guarantee
    assert placement_of("G(!(pickup & dropoff))", INS, OUTS) == "guarantee"
    # U holding behaviour with inputs -> assumption
    assert placement_of("G(at_shelf -> (at_shelf U go_to_dock))", INS, OUTS) == "assumption"


def test_anchor_complete_inserts_missing():
    from contrauto.feedback.mission_anchor import anchor_complete, missing_mission_formulas
    pairs = [{"nl": "m", "ltl": "G(!(r1_pickup & r1_dropoff)) & G F(r1_dropoff)"}]
    content = """\
CONTRACT Robot1:
    Guarantees:
        G F(r1_dropoff);
CONTRACT System:
    input boolean variable r1_at_pickup;
    output boolean variable r1_pickup;
    output boolean variable r1_dropoff;
    Assumptions:
        true;
    Guarantees:
        G F(r1_dropoff);
"""
    out, added = anchor_complete(pairs, content)
    assert ("guarantee", "G(!(r1_pickup & r1_dropoff))") in added
    assert missing_mission_formulas(pairs, out) == []
