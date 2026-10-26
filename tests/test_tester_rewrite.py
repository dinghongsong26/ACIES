"""Unit tests for the W -> GR(1) tester rewrite (option 2: X-free contracts, SLUGS unchanged)."""
from contrauto.tester_rewrite import rewrite_formula, rewrite_ltl_text
from contrauto.assumption_monitor import build_monitor, _PersistMon


def test_rewrite_weak_until():
    assert rewrite_formula("has_item W at_dock") == "G((has_item) & !(at_dock) -> X((has_item)))"
    # arrival justice and plain invariants are already GR(1)-native -> untouched
    assert rewrite_formula("G F(at_shelf | !go_to_shelf)") == "G F(at_shelf | !go_to_shelf)"
    assert rewrite_formula("G(pickup -> at_shelf)") == "G(pickup -> at_shelf)"


def test_same_W_is_monitorable():
    # the SAME weak-until the rewrite feeds SLUGS is what the monitor reads (X-free)
    assert isinstance(build_monitor("has_item W at_dock"), _PersistMon)
    # ...and its GR(1) image carries X, so it is NOT (correctly) monitored
    assert build_monitor(rewrite_formula("has_item W at_dock")) is None


def test_rewrite_ltl_text_block():
    src = """NAME: X;

CONTRACT Robot1:
    input boolean variable at_dock;
    output boolean variable has_item;
    Assumptions:
        !at_dock;
        G F(at_dock | !has_item);
    Guarantees:
        !has_item;
        has_item W at_dock;
        G F(has_item);
"""
    out = rewrite_ltl_text(src)
    assert "has_item W at_dock;" not in out
    assert "G((has_item) & !(at_dock) -> X((has_item)));" in out
    # structure + non-W lines preserved
    assert "CONTRACT Robot1:" in out and "G F(has_item);" in out and "input boolean variable at_dock;" in out
    assert "!at_dock;" in out
