"""Mechanical completion layer (contrauto.completion) — Step 2.7.

Mock-oracle unit tests for the three closure passes, plus a toolchain-gated golden test on
the frozen warehouse draft gemini failed to repair for nine rounds (the pulse trap):
completion must make all three controllers realizable and refinement pass, LLM-free.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from contrauto.completion import (
    CompletionResult, add_guarantee, complete, justice_cycle_gate,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PULSE = os.path.join(ROOT, "tests/fixtures/warehouse_pulse.ltl")
_CYCLIC = os.path.join(ROOT, "tests/fixtures/warehouse_cyclic.ltl")

_TOY = """\
CONTRACT Robot:
    input boolean variable grant;
    output boolean variable req;
    Assumptions:
        G F(grant | !req);
    Guarantees:
        G F(req);
CONTRACT Arbiter:
    input boolean variable req;
    output boolean variable grant;
    Assumptions:
        true;
    Guarantees:
        G(grant -> req);
CONTRACT System:
    input boolean variable dummy_env;
    output boolean variable req;
    output boolean variable grant;
    Assumptions:
        true;
    Guarantees:
        G F(req);
"""


class _Oracle:
    """check_realizable stub: per-contract verdicts, optionally content-dependent."""
    def __init__(self, verdicts):
        self.verdicts = verdicts
    def check_realizable(self, ctrl, content):
        v = self.verdicts.get(ctrl, True)
        return v(content) if callable(v) else v


def test_add_guarantee_targets_right_section():
    out = add_guarantee(_TOY, "Arbiter", "G F(grant | !req)")
    arb = out.split("CONTRACT Arbiter:")[1].split("CONTRACT System:")[0]
    assert "G F(grant | !req);" in arb
    assert out.count("G F(grant | !req);") == 2   # Robot assumption + Arbiter guarantee


def test_cycle_gate_detects_mutual_justice():
    diags = justice_cycle_gate(open(_CYCLIC).read())
    assert diags and "JUSTICE-CYCLE" in diags[0]
    # complete() no longer merely blocks: with a workable oracle it BREAKS the cycle
    # mechanically (see test_cycle_break_*); blocking is the fallback when no strip
    # survives realizability.


def test_discharge_prefers_sibling_guarantee_when_realizable():
    res = complete(_Oracle({"Robot": True, "Arbiter": True}), _TOY)
    kinds = [(k, c) for k, c, _ in res.applied]
    assert ("guarantee", "Arbiter") in kinds      # route (a): sibling underwrites
    assert not any(k.startswith("system") for k, _ in kinds)


def test_discharge_folds_into_system_when_sibling_breaks():
    # Arbiter would become unrealizable with the guarantee -> fold into System.Assumptions
    def arb_verdict(content):
        return "G F(grant | !req)" not in content.split("CONTRACT Arbiter:")[1] \
                                                 .split("CONTRACT System:")[0]
    res = complete(_Oracle({"Robot": True, "Arbiter": arb_verdict}), _TOY)
    kinds = [(k, c) for k, c, _ in res.applied]
    assert ("system-assumption (folded)", "System") in kinds
    sys_block = res.content.split("CONTRACT System:")[1]
    assert "G F(grant | !req);" in sys_block


def test_no_cycle_no_internal_assumptions_is_noop():
    toy = _TOY.replace("        G F(grant | !req);\n", "        true;\n")
    res = complete(_Oracle({}), toy)
    assert res.applied == [] and res.blocked == []


# ── toolchain-gated golden test (the nine-round gemini draft) ─────────────────
def test_completion_repairs_pulse_trap_end_to_end(tmp_path):
    if not (os.environ.get("LOGICS_TOOL") and os.environ.get("SLUGS")
            and os.environ.get("SLUGS_COMP") and os.environ.get("NUXMV")):
        pytest.skip("needs configured CHASE/SLUGS/NuXMV toolchain")
    from contrauto.agents.tool_pipeline import ToolPipeline
    import re
    content = open(_PULSE).read()
    t = ToolPipeline(tmp_path, "warehouse", tmp_path / "s.ltl",
                     tmp_path / "v.chase", tmp_path / "y.chase")
    res = complete(t, content)
    assert not res.blocked
    assert any(k == "assumption" for k, _, _ in res.applied)          # persistence added
    assert any(k.startswith("system") for k, _, _ in res.applied)     # folded discharge
    for ctrl in ("Robot1", "Robot2", "Arbiter"):
        assert t.check_realizable(ctrl, res.content) is True
    (tmp_path / "s.ltl").write_text(res.content)
    t.write_chase_command_files(res.content)
    ok, err = t.run_verification()
    assert ok, err[:300]
    ref_ok, out = t.run_nuxmv_refinement(tmp_path / "refinement_warehouse.smv")
    assert ref_ok, out[-500:]


def test_cycle_break_folds_both_sides_into_system():
    # Robot⇄Arbiter justice cycle -> BOTH sides' cycle assumptions are mirrored verbatim
    # into System.Assumptions (contracts untouched: stripping proved unsound — a robot
    # without grant fairness stays realizable but physically starves).
    src = open(_CYCLIC).read()
    res = complete(_Oracle({}), src)
    assert res.blocked == []
    folds = [(k, f) for k, c, f in res.applied if k == "system-assumption (cycle fold)"]
    assert folds, res.applied
    sys_asm = res.content.split("CONTRACT System:")[1].split("Guarantees:")[0]
    for _, f in folds:
        assert f.replace(" ", "") in sys_asm.replace(" ", "")
    # contracts keep their assumptions
    rob = res.content.split("CONTRACT Robot1:")[1].split("CONTRACT ")[0]
    assert "G F(" in rob.split("Guarantees:")[0]
