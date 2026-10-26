"""Extract individually labeled LTL formulas from a CHASE .ltl file.

Parses the contract structure to identify which formulas belong to
which contract's Assumptions vs Guarantees section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from contrauto.feedback.ltl_parser import LTLNode, parse


@dataclass
class LabeledFormula:
    """One contract formula tagged with its origin; `formula` is None if unparseable."""
    contract: str
    section: str
    index: int
    raw_text: str
    formula: Optional[LTLNode]


def extract_formulas(ltl_content: str) -> list[LabeledFormula]:
    """Split a .ltl file into per-contract, per-section labeled formulas."""
    contracts = re.split(r'(?=CONTRACT\s+\w+\s*:)', ltl_content)
    results = []

    for block in contracts:
        m = re.match(r'CONTRACT\s+(\w+)\s*:', block)
        if not m:
            continue
        contract_name = m.group(1)

        assumptions_match = re.search(
            r'Assumptions\s*:\s*(.*?)(?=Guarantees\s*:|$)',
            block, re.DOTALL
        )
        guarantees_match = re.search(
            r'Guarantees\s*:\s*(.*?)$',
            block, re.DOTALL
        )

        if assumptions_match:
            formulas = _split_formulas(assumptions_match.group(1))
            for i, raw in enumerate(formulas):
                ast = _safe_parse(raw)
                results.append(LabeledFormula(contract_name, "assumption", i, raw, ast))

        if guarantees_match:
            formulas = _split_formulas(guarantees_match.group(1))
            for i, raw in enumerate(formulas):
                ast = _safe_parse(raw)
                results.append(LabeledFormula(contract_name, "guarantee", i, raw, ast))

    return results


def _split_formulas(text: str) -> list[str]:
    """Split semicolon-delimited formulas, ignoring variable declarations."""
    raw_parts = text.split(';')
    formulas = []
    for part in raw_parts:
        # Drop full-line comments (# ...) and collapse whitespace/newlines to single spaces.
        # A comment line carries no ';', so without this it gets glued to the following formula,
        # yielding a multi-line "# comment\n FORMULA" chunk. Downstream exact-text matching
        # (gr1_core.project_guarantees) then fails to keep/drop that guarantee, silently breaking
        # the unrealizable-core computation on comment-laden (e.g. weak-model) contracts.
        no_comments = "\n".join(ln for ln in part.splitlines()
                                if not ln.strip().startswith('#'))
        cleaned = " ".join(no_comments.split())
        if not cleaned:
            continue
        if re.match(r'(input|output)\s+boolean\s+variable', cleaned):
            continue
        if re.match(r'CONTRACT\s+\w+', cleaned):
            break
        formulas.append(cleaned)
    return formulas


def _safe_parse(text: str) -> Optional[LTLNode]:
    try:
        return parse(text)
    except Exception:
        return None
