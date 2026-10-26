"""Static GR(1)-fragment check for synthesized (sub-contract) guarantees.

The GR(1) backend (SLUGS) synthesizes a strategy ONLY for a spec whose system side is
in the GR(1) fragment: a conjunction of

  * INITIAL    : a pure Boolean over current variables                 (theta_s)
  * SAFETY     : G(rho) with rho a Boolean over current AND next (X)   (G rho_s)
                 variables, X-depth <= 1
  * JUSTICE    : G F(J) with J a pure Boolean over current variables   (G F J_s)

Anything else in a Guarantee (a bare F, a response `G(p -> F(q))`, an Until, nested
temporal, or X-depth > 1) is NOT GR(1). The SLUGS printer flattens `G(p -> F(q))` into
the over-strong safety `G(p -> q)` (or emits an empty `[SYS_*]`), which surfaces only
LATER as an unrealizable/degenerate/empty FSM — hard to attribute. This module catches it
*statically*, BEFORE any tool call, with formula-level localization (`Robot.guarantee[2]`),
the same granularity the refinement diagnoser gives.

Scope:
  * Only SUB-contracts (controllers) are synthesized, so only THEIR guarantees are
    strict-checked. The System contract is used for refinement (nuXmv), not synthesis, so
    its guarantees may legitimately use response/Until and are skipped.
  * Assumptions become the GR(1) ENVIRONMENT spec; a response `G(p -> F(q))` and `G F` ARE
    accepted there (standard GR(1) env liveness), so assumptions get a lenient check that
    flags only genuinely unsupported operators (Until, a top-level bare temporal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contrauto.feedback.ltl_parser import (
    Atom, Not, And, Or, Implies, Iff, G, F, X, U_,
)
from contrauto.feedback.contract_parser import LabeledFormula, extract_formulas

_BIN = (And, Or, Implies, Iff)


@dataclass
class GR1Violation:
    """One guarantee/assumption formula that falls outside the GR(1) fragment."""
    formula: LabeledFormula
    reason: str


# ── structural predicates ─────────────────────────────────────────────────────
def _is_boolean(node) -> bool:
    """Pure Boolean: atoms combined with ! & | -> <->, NO temporal operator."""
    if isinstance(node, Atom):
        return True
    if isinstance(node, Not):
        return _is_boolean(node.child)
    if isinstance(node, _BIN):
        return _is_boolean(node.left) and _is_boolean(node.right)
    return False  # G / F / X / U


def _is_safety_body(node) -> bool:
    """A transition predicate rho: Booleans over current vars and `X(<pure Boolean>)`.

    Allows a single X layer (relate current to next state), so X-depth <= 1. Rejects any
    F / U / G, and rejects nested X (X(X ...)) since that is not a one-step transition.
    """
    if isinstance(node, Atom):
        return True
    if isinstance(node, Not):
        return _is_safety_body(node.child)
    if isinstance(node, _BIN):
        return _is_safety_body(node.left) and _is_safety_body(node.right)
    if isinstance(node, X):
        return _is_boolean(node.child)
    return False  # F, U, G


def _contains(node, cls) -> bool:
    """True iff an instance of `cls` appears anywhere in the AST."""
    if isinstance(node, cls):
        return True
    if isinstance(node, (Not, G, F, X)):
        return _contains(node.child, cls)
    if isinstance(node, (*_BIN, U_)):
        return _contains(node.left, cls) or _contains(node.right, cls)
    return False


# ── guarantee classification (strict GR(1)) ───────────────────────────────────
def classify_guarantee(node) -> tuple[str, bool, str]:
    """Classify a guarantee AST. Returns (kind, ok, reason).

    kind in {"init", "safety", "justice", "invalid"}; reason is non-empty only when
    ``ok`` is False.
    """
    if _is_boolean(node):
        return ("init", True, "")
    if isinstance(node, G):
        child = node.child
        if isinstance(child, F):
            if _is_boolean(child.child):
                return ("justice", True, "")
            return ("invalid", False,
                    "justice G F(...) must wrap a pure Boolean, but a temporal operator "
                    "appears inside the F.")
        if _is_safety_body(child):
            return ("safety", True, "")
        return ("invalid", False, _why_bad_safety(child))
    return ("invalid", False, _why_bad_top(node))


def _why_bad_safety(child) -> str:
    """Explain why the body of a top-level G(...) is not a valid safety predicate."""
    if _contains(child, U_):
        return ("the safety G(...) contains an Until (U); GR(1) safety must be a Boolean over "
                "current and next-step (X) variables only.")
    if _contains(child, F):
        return ("response/eventually inside G — e.g. G(p -> F(q)). The GR(1) backend flattens "
                "this into the over-strong safety G(p -> q). Express the liveness as a separate "
                "justice goal G F(...) instead.")
    if _contains(child, G):
        return ("nested G inside a G(...) safety; not GR(1).")
    return ("the G(...) body is not a one-step transition predicate (X-depth must be <= 1, "
            "no nested temporal).")


def _why_bad_top(node) -> str:
    """Explain why a top-level (non-G, non-Boolean) guarantee is not GR(1)."""
    if isinstance(node, F):
        return ("bare F at the top level; GR(1) liveness in a Guarantee must be justice "
                "G F(...), not a one-shot F.")
    if isinstance(node, U_) or _contains(node, U_):
        return ("uses Until (U); not expressible in the GR(1) Guarantee fragment.")
    if isinstance(node, X):
        return ("top-level X without an enclosing G; wrap transition constraints in G(...).")
    return ("top-level formula is neither a Boolean initial condition, a safety G(...), nor a "
            "justice G F(...). Wrap safety in G(...) and liveness in G F(...).")


# ── assumption check (lenient) ────────────────────────────────────────────────
def _why_bad_assumption(node) -> str:
    """Lenient GR(1) check for an ENVIRONMENT assumption.

    Response `G(p -> F(q))` and `G F` are valid env liveness, so they pass. Flags only what
    even the GR(1) environment spec cannot take: an Until anywhere, or a top-level bare
    temporal (F/X) not wrapped in G.
    """
    if _contains(node, U_):
        return ("uses Until (U); not expressible in the GR(1) environment fragment.")
    if isinstance(node, (F, X)):
        return ("top-level bare " + ("F" if isinstance(node, F) else "X") +
                "; wrap it in G(...) (e.g. justice G F(...) or transition G(... -> X ...)).")
    return ""


# ── public entry point ────────────────────────────────────────────────────────
def _system_name(ltl_content: str) -> str:
    """Mirror ToolPipeline's rule: a contract named 'System' (case-insensitive), else the
    last declared contract."""
    names = re.findall(r"CONTRACT\s+(\w+)\s*:", ltl_content)
    return next((n for n in names if n.lower() == "system"),
                names[-1] if names else "System")


def check_gr1_fragment(ltl_content: str) -> list[GR1Violation]:
    """Return every sub-contract formula that falls outside the synthesizable GR(1) fragment.

    Guarantees of sub-contracts are strict-checked; their assumptions are lenient-checked.
    System-contract formulas and unparseable formulas (handled by the syntax step) are
    skipped.
    """
    sys_name = _system_name(ltl_content)
    violations: list[GR1Violation] = []
    from contrauto.feedback.gr1_assume import contract_io
    io_cache: dict = {}
    for lf in extract_formulas(ltl_content):
        if lf.formula is None or lf.contract == sys_name:
            continue
        if lf.section == "guarantee":
            _, ok, reason = classify_guarantee(lf.formula)
            if not ok:
                violations.append(GR1Violation(lf, reason))
        else:  # assumption
            reason = _why_bad_assumption(lf.formula)
            if not reason:
                # Game-semantics rule slugs enforces at compile time (previously a raw
                # "variable x' is not allowed" backend crash): an ENV assumption may not
                # constrain the system's NEXT move — no X(...) over the contract's OWN
                # outputs in its Assumptions (env moves first; primed outputs are the
                # system player's choice).
                if lf.contract not in io_cache:
                    io_cache[lf.contract] = contract_io(ltl_content, lf.contract)
                outs = set(io_cache[lf.contract][1])
                xv = set(re.findall(r"X\(\s*!?\s*(\w+)", lf.raw_text))
                bad = sorted(xv & outs)
                if bad:
                    reason = (f"assumption applies X(...) to this contract's OWN output(s) "
                              f"{bad} — the environment cannot constrain the system's next "
                              f"move. Move it to Guarantees, or rephrase over inputs.")
            if reason:
                violations.append(GR1Violation(lf, reason))
    return violations


def format_gr1_feedback(violations: list[GR1Violation], contract_content: str) -> str:
    """Render GR(1)-fragment violations as formula-level repair feedback for the LLM."""
    lines = [
        "GR(1) FRAGMENT VIOLATION — some sub-contract formulas are NOT synthesizable.",
        "",
        "The GR(1) backend synthesizes ONLY: a Boolean initial condition, safety G(<Boolean",
        "over current and next-step X variables>), and justice G F(<Boolean>). The formulas",
        "below are outside that fragment and are rejected BEFORE synthesis.",
        "",
        "=== VIOLATING FORMULAS ===",
    ]
    for v in violations:
        lf = v.formula
        lines.append(f"  ✗ {lf.contract}.{lf.section}[{lf.index}]: {lf.raw_text}")
        lines.append(f"    → {v.reason}")
    lines.extend([
        "",
        "=== HOW TO FIX (target the ✗ formulas; keep it general) ===",
        "- Liveness in a Guarantee must be pure justice G F(p). Rewrite 'p -> eventually q' as",
        "  the justice goal G F(q | !p) (NOT a response G(p -> F(q))).",
        "- An Until used as a guard — '(!q) U p' (don't do q until p holds) — is re-expressed as",
        "  the safety guarantee G(!p -> !q). Convert Until to safety; do NOT just delete it (the",
        "  backend silently DROPS an Until guarantee, so the FSM would stop enforcing it).",
        "- A genuine response/Until property that must stay can live in the System contract",
        "  (checked by refinement, not synthesized) or, if it constrains the environment, in an",
        "  Assumption (where G(p -> F(q)) / G F are allowed).",
        "- Safety stays as G(<Boolean over vars and a single X>); remove nested temporal.",
        "",
        "=== CURRENT .ltl ===",
        contract_content,
        "",
        "Return corrected .ltl only.",
    ])
    return "\n".join(lines)
