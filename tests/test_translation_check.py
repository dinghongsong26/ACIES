"""Mechanical translator checks: coverage, lint, and physics-derived completion."""

from contrauto.translation_check import complete, coverage, lint
from simulator.registry import grounding_vocab

VOCAB = grounding_vocab("warehouse")


def _ltls(pairs):
    return {p["ltl"] for p in pairs}


def test_coverage_reports_untranslated_requirements():
    pairs = [{"ltl": "G F(r1_dropoff)", "req": [1]}, {"ltl": "G(a)", "req": [2, 4]}]
    assert coverage(pairs, 4) == [3]


def test_complete_adds_missing_physics_formulas_per_robot():
    pairs, notes = complete([{"nl": "goal", "ltl": "G F(r1_dropoff)", "req": [1]}],
                            VOCAB, ["r1_", "r2_"])
    got = _ltls(pairs)
    for p in ("r1_", "r2_"):
        assert f"G(({p}pickup & X(!{p}has_item)) -> X({p}pickup))" in got
        assert f"G(({p}dropoff & X({p}has_item) & X({p}dock_grant)) -> X({p}dropoff))" in got
        assert f"G F({p}has_item | !{p}pickup)" in got
        assert f"G(({p}has_item & !{p}dropoff) -> X({p}has_item))" in got
    assert all(p["source"] == "grounding" for p in pairs[1:])
    assert notes


def test_complete_is_idempotent():
    once, _ = complete([], VOCAB, ["r1_"])
    twice, notes = complete(once, VOCAB, ["r1_"])
    assert _ltls(twice) == _ltls(once) and not notes


def test_complete_rewrites_known_bad_shapes_and_unprefixes_globals():
    bad = [{"nl": "", "ltl": "G((r1_pickup & !r1_has_item) -> X(r1_pickup))", "req": [7]},
           {"nl": "", "ltl": "G(r1_dropoff -> X(!r1_dock_req))", "req": [5]},
           {"nl": "", "ltl": "G F(r1_task)", "req": [9]}]
    pairs, _ = complete(bad, VOCAB, ["r1_"])
    got = _ltls(pairs)
    assert "G((r1_pickup & X(!r1_has_item)) -> X(r1_pickup))" in got
    assert "G((r1_dropoff & X(!r1_has_item)) -> X(!r1_dock_req))" in got
    assert "G F(task)" in got
    assert "G((r1_pickup & !r1_has_item) -> X(r1_pickup))" not in got


def test_lint_flags_unknown_variables_and_non_gr1_shapes():
    issues = lint([{"ltl": "G(r1_carrying -> r1_dock_req)"},
                   {"ltl": "X(G(r1_has_item -> r1_dock_req))"},
                   {"ltl": "G(r1_pickup -> F(r1_has_item))"},
                   {"ltl": "G F(r1_dropoff)"}], VOCAB, ["r1_"])
    assert any("r1_carrying" in i for i in issues)
    assert any("X(G" in i for i in issues)
    assert any("nested under `G`" in i for i in issues)
    assert not any("G F(r1_dropoff)" in i for i in issues)


def test_coverage_accepts_declared_not_applicable():
    pairs = [{"ltl": "G F(r1_dropoff)", "req": [1]}]
    assert coverage(pairs, 2, [{"req": 2, "reason": "single robot"}]) == []


def test_lint_flags_tautology():
    issues = lint([{"ltl": "G(r1_dock_grant -> r1_dock_grant)"}], VOCAB, ["r1_"])
    assert any("tautology" in i for i in issues)


def test_classify_splits_world_assumptions_from_system_guarantees():
    from contrauto.translation_check import classify
    kind = lambda f: [s for s, _ in classify(f, VOCAB, ["r1_"])]
    assert kind("G F(r1_dropoff)") == ["guarantee"]
    assert kind("G F(r1_dock_grant | !r1_dock_req)") == ["guarantee"]     # the arbiter's duty
    assert kind("G F(r1_at_dock | !r1_go_to_dock)") == ["assumption"]     # world: arrival
    assert kind("G((r1_has_item & !r1_dropoff) -> X(r1_has_item))") == ["assumption"]
    assert kind("G((r1_pickup & X(!r1_has_item)) -> X(r1_pickup))") == ["guarantee"]
    assert kind("!r1_at_dock & !r1_dock_req") == ["assumption", "guarantee"]


def test_realizability_spec_declares_sensors_as_inputs():
    from contrauto.translation_check import realizability_spec
    spec, owners = realizability_spec([{"ltl": "G(r1_dropoff -> r1_at_dock)"},
                                       {"ltl": "G F(r1_dropoff)"}], VOCAB, ["r1_"])
    assert "input boolean variable r1_at_dock;" in spec
    assert "output boolean variable r1_dropoff;" in spec
    assert "        !r1_at_dock;" in spec                  # robots start with sensors low
    assert [i for i, _ in owners] == [0, 1]
