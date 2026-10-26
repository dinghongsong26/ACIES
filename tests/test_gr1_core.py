"""Minimal unrealizable core (contrauto.feedback.gr1_core).

Two layers, both tool-free:
  * QuickXplain over a MOCK realizability oracle — asserts it returns a genuine minimal
    unrealizable subset (single culprit, joint pair, background-only, all-realizable), and
    that it stays minimal with far fewer than O(n) oracle calls.
  * project_guarantees / guarantee_candidates — the .ltl surgery that keeps a chosen subset
    of a controller's guarantees while leaving every other line byte-identical.

Run:  python3 -m pytest tests/test_gr1_core.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.feedback.gr1_core import (
    minimal_unrealizable_core, guarantee_candidates, project_guarantees,
)


# ── QuickXplain over a mock oracle ────────────────────────────────────────────
def _oracle(required):
    """Build an is_unrealizable predicate: unrealizable iff `required` ⊆ subset."""
    req = set(required)
    calls = {"n": 0}
    def is_unreal(subset):
        calls["n"] += 1
        return req.issubset(set(subset))
    return is_unreal, calls


def test_single_culprit():
    cand = ["a", "b", "c", "d", "e"]
    is_unreal, _ = _oracle({"c"})
    core = minimal_unrealizable_core(cand, is_unreal)
    assert core == ["c"]


def test_joint_pair_core():
    cand = ["a", "b", "c", "d", "e"]
    is_unreal, _ = _oracle({"b", "e"})
    core = minimal_unrealizable_core(cand, is_unreal)
    assert set(core) == {"b", "e"}
    # order is preserved (b before e in candidates)
    assert core == ["b", "e"]


def test_core_is_actually_unrealizable_and_minimal():
    cand = list("abcdefgh")
    is_unreal, _ = _oracle({"c", "f"})
    core = minimal_unrealizable_core(cand, is_unreal)
    # the returned core is unrealizable...
    assert is_unreal(core)
    # ...and dropping any one element makes it realizable (1-minimality)
    for x in core:
        assert not is_unreal([y for y in core if y != x])


def test_background_only_returns_empty():
    # unrealizable no matter what (even with no guarantees) -> [] (cause is in assumptions)
    is_unreal = lambda subset: True
    assert minimal_unrealizable_core(["a", "b"], is_unreal) == []


def test_all_realizable_returns_none():
    is_unreal = lambda subset: False
    assert minimal_unrealizable_core(["a", "b"], is_unreal) is None


def test_sublinear_call_count_for_small_core():
    # 64 candidates, core of size 1 -> QuickXplain should use far fewer than 64 checks.
    cand = [f"g{i}" for i in range(64)]
    is_unreal, calls = _oracle({"g40"})
    core = minimal_unrealizable_core(cand, is_unreal)
    assert core == ["g40"]
    assert calls["n"] < 32   # O(k log(n/k)) << O(n)


# ── .ltl guarantee projection ─────────────────────────────────────────────────
_LTL = """\
CONTRACT Robot:
    input boolean variable grant;
    output boolean variable req;
    Assumptions:
        !grant;
        G F(grant);
    Guarantees:
        !req;
        G F(req);
        G(req -> X(grant | !req));
        G(!(req & grant));
CONTRACT System:
    output boolean variable req;
    Guarantees:
        !req;
        G F(req);
"""


def test_guarantee_candidates_excludes_init_pins():
    cands = guarantee_candidates(_LTL, "Robot")
    assert cands == ["G F(req)", "G(req -> X(grant | !req))", "G(!(req & grant))"]
    # the `!req;` init pin is NOT a candidate
    assert "!req" not in cands


def test_project_keeps_subset_and_preserves_other_lines():
    keep = {"G(!(req & grant))"}
    out = project_guarantees(_LTL, "Robot", keep)
    # kept guarantee + init pin survive; dropped guarantees are gone
    assert "G(!(req & grant));" in out
    assert "!req;" in out                       # init pin always kept
    assert "G(req -> X(grant | !req));" not in out
    # Robot's justice guarantee was dropped from the projection...
    robot_block = out.split("CONTRACT System:")[0]
    assert "G F(req)" not in robot_block
    # ...but the System contract is untouched (still has its own G F(req))
    assert "CONTRACT System:" in out
    sys_block = out.split("CONTRACT System:")[1]
    assert "G F(req);" in sys_block
    # Assumptions are preserved verbatim
    assert "G F(grant);" in out


def test_project_empty_keeps_only_init_pins():
    out = project_guarantees(_LTL, "Robot", set())
    robot_block = out.split("CONTRACT System:")[0]
    # no time-tense guarantees remain in Robot...
    assert "G F(req)" not in robot_block
    assert "G(req ->" not in robot_block
    # ...only the init pin
    assert "!req;" in robot_block


# ── live-toolchain integration (skipped unless CHASE/SLUGS are configured) ────
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Frozen fixture, NOT the live outputs/ artifact: outputs/warehouse is rewritten by every
# run, so anchoring the injection below on its current text made the test silently no-op
# (and fail) whenever the LLM phrased the spec differently.
_WAREHOUSE = os.path.join(ROOT, "tests/fixtures/warehouse_specs.ltl")


def _toolchain_ready(tmp) -> bool:
    """True iff logics_tool + SLUGS actually run here (env vars set, dylibs resolvable)."""
    if not (os.environ.get("LOGICS_TOOL") and os.environ.get("SLUGS")
            and os.environ.get("SLUGS_COMP") and os.path.exists(_WAREHOUSE)):
        return False
    from contrauto.agents.tool_pipeline import ToolPipeline
    t = ToolPipeline(tmp, "probe", tmp / "s.ltl", tmp / "v.chase", tmp / "y.chase")
    return t.check_realizable("Robot", open(_WAREHOUSE).read()) is True


def test_core_localizes_injected_impossible_guarantee(tmp_path):
    if not _toolchain_ready(tmp_path):
        pytest.skip("needs configured CHASE/SLUGS toolchain (LOGICS_TOOL/SLUGS/SLUGS_COMP + dylibs)")
    from contrauto.agents.tool_pipeline import ToolPipeline
    from contrauto.feedback.gr1_core import compute_core

    content = open(_WAREHOUSE).read()
    # The justice assumptions only grant at_shelf / at_dock separately (and the robot's own
    # nav mutex forbids requesting both), so the env can forever avoid their conjunction ->
    # G F(at_shelf & at_dock) is unrealizable; the core must include this injected formula.
    broken = content.replace(
        "        G F(dropoff);\n",
        "        G F(dropoff);\n        G F(at_shelf & at_dock);\n", 1)
    assert broken != content, "injection anchor not found in fixture"
    t = ToolPipeline(tmp_path, "wh", tmp_path / "s.ltl", tmp_path / "v.chase", tmp_path / "y.chase")
    assert t.check_realizable("Robot", broken) is False
    core = compute_core(t, broken, "Robot")
    assert core is not None
    assert "G F(at_shelf & at_dock)" in core
    # minimality: every formula in the core is load-bearing
    from contrauto.feedback.gr1_core import project_guarantees
    for f in core:
        keep = set(core) - {f}
        assert t.check_realizable("Robot", project_guarantees(broken, "Robot", keep)) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
