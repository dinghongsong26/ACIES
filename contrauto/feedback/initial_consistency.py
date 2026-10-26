"""Detect immediate contradictions between canonical cold-start pins and safety guards."""

from __future__ import annotations

import re
from dataclasses import dataclass

from contrauto.feedback.contract_parser import extract_formulas
from contrauto.feedback.ltl_parser import And, Atom, G, Implies, Not, Or


@dataclass(frozen=True)
class InitialContradiction:
    contract: str
    output: str
    formula: str


def _declared_outputs(content: str) -> dict[str, set[str]]:
    outputs: dict[str, set[str]] = {}
    current = None
    for line in content.splitlines():
        match = re.match(r"\s*CONTRACT\s+(\w+)\s*:", line)
        if match:
            current = match.group(1)
            outputs.setdefault(current, set())
            continue
        match = re.match(r"\s*output\s+boolean\s+variable\s+(\w+)\s*;", line)
        if match and current:
            outputs[current].add(match.group(1))
    return outputs


def _initial_values(content: str, outputs: dict[str, set[str]]) -> dict[str, bool]:
    """Every declared output's canonical t=0 value, keyed by variable name GLOBALLY.

    A System-level formula routinely references a SIBLING contract's output as its own
    INPUT (e.g. System reads Robot1's `r1_shelf_req`). A per-contract-scoped map misses
    exactly that case, since it only records a contract's OWN outputs -- so a System
    guarantee gating on another contract's output silently escaped this check even when
    it forces that output true at cold start. A cold-start pin is canonical and global
    (`enforce_initial_conditions` re-injects `!x` for every declared output everywhere it
    appears), so one flat name -> bool map, merged across every contract, is correct for
    evaluating ANY contract's antecedent, whether it owns a variable as output or merely
    observes it as input.
    """
    values: dict[str, bool] = {name: False for names in outputs.values() for name in names}
    for labeled in extract_formulas(content):
        if labeled.formula is None:
            continue
        formula = labeled.formula
        if isinstance(formula, Atom):
            values[formula.name] = True
        elif isinstance(formula, Not) and isinstance(formula.child, Atom):
            values[formula.child.name] = False
    return values


def _evaluate_boolean(node, values: dict[str, bool]) -> bool | None:
    if isinstance(node, Atom):
        return values.get(node.name)
    if isinstance(node, Not):
        child = _evaluate_boolean(node.child, values)
        return None if child is None else not child
    if isinstance(node, And):
        left, right = _evaluate_boolean(node.left, values), _evaluate_boolean(node.right, values)
        if left is False or right is False:
            return False
        return True if left is True and right is True else None
    if isinstance(node, Or):
        left, right = _evaluate_boolean(node.left, values), _evaluate_boolean(node.right, values)
        if left is True or right is True:
            return True
        return False if left is False and right is False else None
    return None


def initial_output_contradictions(content: str) -> list[InitialContradiction]:
    """Find ``G(P -> action)`` rules that force a cold-start-false output at time zero.

    The check is intentionally narrow: it evaluates only propositional antecedents whose
    inputs and outputs have deterministic initial values. Temporal formulas and unknown
    inputs are left to the symbolic consistency check.
    """
    outputs = _declared_outputs(content)
    values = _initial_values(content, outputs)
    contradictions = []
    for labeled in extract_formulas(content):
        formula = labeled.formula
        if (labeled.section != "guarantee" or not isinstance(formula, G)
                or not isinstance(formula.child, Implies)
                or not isinstance(formula.child.right, Atom)):
            continue
        output = formula.child.right.name
        if output not in outputs.get(labeled.contract, set()):
            continue
        if values.get(output) is not False:
            continue
        if _evaluate_boolean(formula.child.left, values) is True:
            contradictions.append(InitialContradiction(labeled.contract, output, labeled.raw_text))
    return contradictions


def format_initial_consistency_feedback(contradictions: list[InitialContradiction], content: str) -> str:
    details = "\n".join(
        f"- {item.contract}.{item.output}: `{item.formula}` forces `{item.output}` true at t=0, "
        f"but cold-start initialization requires it false."
        for item in contradictions
    )
    return (
        "[INITIAL OUTPUT CONTRADICTION]\n"
        "A safety guard forces an output at time zero while the contract initializes that output "
        "to false. Add the missing positive trigger or state condition so its antecedent is false "
        "at cold start. For pickup navigation, use `G(task & !has_item & !at_pickup -> "
        "go_to_pickup)`, not `G(!has_item & !(at_pickup & task) -> go_to_pickup)`.\n"
        f"{details}\n\nCurrent .ltl:\n{content}\n\nReturn raw .ltl text only.\n"
    )