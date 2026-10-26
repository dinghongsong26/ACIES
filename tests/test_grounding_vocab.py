"""Step 1.6 grounding-vocabulary check (DecomposerAgent._check_grounding).

Regression for the patrol 9/9 loop: role_of only classifies warehouse-family names
(shelf/dock/grant/item/...), so grounding via role_of ALONE rejected variables that are
literally in another domain's vocabulary (patrol's at_A / inspect / report) — an
unfixable failure the LLM can never rename its way out of. Exact vocabulary membership
must be checked first; role_of stays as the alias fallback (r1_nav_shelf et al).
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.agents.decomposer import DecomposerAgent

PATROL_VOCAB = {
    "sensors": ["ack", "alert_A", "alert_B", "alert_C", "at_A", "at_B", "at_C",
                "inspection_done", "report_ack"],
    "actions": ["go_A", "go_B", "go_C", "go_to_A", "go_to_B", "go_to_C",
                "inspect", "report"],
}

WAREHOUSE_VOCAB = {
    "sensors": ["at_shelf", "at_dock", "has_item"],
    "actions": ["nav_shelf", "nav_dock", "pickup", "dropoff"],
}


def _check(vocab, content, binder=None):
    fake = SimpleNamespace(ground_vocab=vocab, ground_binder=binder)
    return DecomposerAgent._check_grounding(fake, content)


def test_verbatim_vocab_names_ground():
    content = """CONTRACT PatrolRobot:
    input boolean variable at_A;
    input boolean variable alert_B;
    output boolean variable go_to_C;
    output boolean variable inspect;
    output boolean variable report;
"""
    assert _check(PATROL_VOCAB, content) is None


def test_off_vocab_alias_is_flagged():
    # translator wrote goto_A / ack_report; vocab has go_to_A / report_ack
    content = """CONTRACT PatrolRobot:
    input boolean variable ack_report;
    output boolean variable goto_A;
"""
    fb = _check(PATROL_VOCAB, content)
    assert fb is not None
    assert "goto_A" in fb and "ack_report" in fb
    assert "go_to_A" in fb          # allowed vocabulary is shown


def test_warehouse_role_aliases_still_ground():
    # not literal vocab names, but role_of understands the warehouse family
    content = """CONTRACT Robot1:
    input boolean variable r1_at_shelf;
    output boolean variable r1_nav_dock;
    output boolean variable r1_pickup;
"""
    assert _check(WAREHOUSE_VOCAB, content) is None


def test_no_vocab_is_noop():
    assert _check(None, "input boolean variable made_up_name;") is None


def test_box_delivery_binder_judges_grounding():
    # Regression for static_2r_3b delivered-0: `goto_delivery` is unbindable by the
    # box-delivery family's own classifier and must be flagged; `carrying` and the
    # canonical r1_-prefixed names bind. The binder — not role_of — is the judge.
    from simulator.box_delivery import grounds
    from simulator.registry import grounding_vocab, grounding_binder
    vocab = grounding_vocab("static_2r_3b")
    assert vocab is not None and "has_box" in vocab["sensors"]
    assert grounding_binder("static_2r_3b") is not None

    ok = """CONTRACT Robot1:
    input boolean variable r1_carrying;
    input boolean variable r1_at_pickup;
    output boolean variable r1_nav_delivery;
    output boolean variable r1_dropoff;
"""
    assert _check(vocab, ok, binder=grounds) is None

    bad = """CONTRACT Robot1:
    output boolean variable r1_goto_delivery;
"""
    fb = _check(vocab, bad, binder=grounds)
    assert fb is not None and "r1_goto_delivery" in fb
