"""Tests for the pre-NuXMV cold-start consistency gate."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.feedback.initial_consistency import initial_output_contradictions


def _contract(guard: str) -> str:
    return f"""\
CONTRACT Robot:
    input boolean variable at_pickup;
    input boolean variable task;
    output boolean variable go_to_pickup;
    output boolean variable has_item;
    Assumptions:
        !at_pickup;
        !task;
    Guarantees:
        !go_to_pickup;
        !has_item;
        {guard};
"""


def test_rejects_guard_that_forces_cold_start_action():
    conflicts = initial_output_contradictions(
        _contract("G(!has_item & !(at_pickup & task) -> go_to_pickup)")
    )
    assert [(item.contract, item.output) for item in conflicts] == [("Robot", "go_to_pickup")]


def test_allows_guard_with_positive_task_and_location_trigger():
    conflicts = initial_output_contradictions(
        _contract("G(task & !has_item & !at_pickup -> go_to_pickup)")
    )
    assert conflicts == []