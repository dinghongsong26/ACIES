"""Evaluate LTL formulas on lasso traces (prefix + loop → infinite word).

A lasso trace represents the infinite word: prefix · loop^ω
- prefix: [s₀, s₁, ..., s_{k-1}]  (may be empty)
- loop:   [sₖ, sₖ₊₁, ..., sₘ]     (must be non-empty)

Each state is dict[str, bool].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from contrauto.feedback.ltl_parser import (
    LTLNode, Atom, Not, And, Or, Implies, Iff, G, F, X, U_,
)


@dataclass
class CheckResult:
    satisfied: bool
    violation_step: Optional[int] = None
    explanation: str = ""


def check_lasso(formula: LTLNode, prefix: list[dict], loop: list[dict]) -> bool:
    """Return whether `formula` holds on the infinite word prefix · loop^ω."""
    full = prefix + loop + loop
    n_prefix = len(prefix)
    n_loop = len(loop)
    return _eval(formula, full, 0, n_prefix, n_loop)


def check_lasso_verbose(
    formula: LTLNode, prefix: list[dict], loop: list[dict]
) -> CheckResult:
    """Like check_lasso, but also report the first violating step as a label."""
    full = prefix + loop + loop
    n_prefix = len(prefix)
    n_loop = len(loop)
    sat = _eval(formula, full, 0, n_prefix, n_loop)
    if sat:
        return CheckResult(True, explanation="satisfied on lasso trace")

    step = _find_violation(formula, full, 0, n_prefix, n_loop)
    label = _step_label(step, n_prefix, n_loop) if step is not None else ""
    return CheckResult(False, step, f"violated at {label}" if label else "violated")


def _step_label(step: int, n_prefix: int, n_loop: int) -> str:
    if step < n_prefix:
        return f"prefix step {step}"
    loop_pos = (step - n_prefix) % n_loop
    return f"loop step {loop_pos}"


def _next_pos(pos: int, n_prefix: int, n_loop: int) -> int:
    """Successor position with lasso wrapping."""
    nxt = pos + 1
    if nxt >= n_prefix + n_loop:
        return n_prefix
    return nxt


def _eval(node: LTLNode, trace: list[dict], pos: int,
          n_prefix: int, n_loop: int) -> bool:
    """Evaluate `node` at position `pos` in the unrolled trace (prefix·loop·loop)."""
    if pos >= len(trace):
        return not isinstance(node, (F, U_))

    state = trace[pos]

    if isinstance(node, Atom):
        # Boolean literals parse as Atoms; a plain state.get would report `true` as
        # violated (absent from every state) and pollute the diagnosis with ✗ true.
        if node.name.lower() == "true":
            return True
        if node.name.lower() == "false":
            return False
        return bool(state.get(node.name, False))

    if isinstance(node, Not):
        return not _eval(node.child, trace, pos, n_prefix, n_loop)

    if isinstance(node, And):
        return (_eval(node.left, trace, pos, n_prefix, n_loop) and
                _eval(node.right, trace, pos, n_prefix, n_loop))

    if isinstance(node, Or):
        return (_eval(node.left, trace, pos, n_prefix, n_loop) or
                _eval(node.right, trace, pos, n_prefix, n_loop))

    if isinstance(node, Implies):
        return (not _eval(node.left, trace, pos, n_prefix, n_loop) or
                _eval(node.right, trace, pos, n_prefix, n_loop))

    if isinstance(node, Iff):
        l = _eval(node.left, trace, pos, n_prefix, n_loop)
        r = _eval(node.right, trace, pos, n_prefix, n_loop)
        return l == r

    if isinstance(node, G):
        limit = n_prefix + n_loop
        for i in range(pos, limit):
            if not _eval(node.child, trace, i, n_prefix, n_loop):
                return False
        return True

    if isinstance(node, F):
        limit = n_prefix + n_loop + n_loop
        for i in range(pos, min(limit, len(trace))):
            if _eval(node.child, trace, i, n_prefix, n_loop):
                return True
        return False

    if isinstance(node, X):
        nxt = _next_pos(pos, n_prefix, n_loop)
        return _eval(node.child, trace, nxt, n_prefix, n_loop)

    if isinstance(node, U_):
        limit = n_prefix + n_loop + n_loop
        for j in range(pos, min(limit, len(trace))):
            if _eval(node.right, trace, j, n_prefix, n_loop):
                return True
            if not _eval(node.left, trace, j, n_prefix, n_loop):
                return False
        return False

    raise TypeError(f"Unknown LTL node: {type(node)}")


def _find_violation(node: LTLNode, trace: list[dict], pos: int,
                    n_prefix: int, n_loop: int) -> Optional[int]:
    """Find the first step where the formula is violated."""
    if isinstance(node, G):
        limit = n_prefix + n_loop
        for i in range(pos, limit):
            if not _eval(node.child, trace, i, n_prefix, n_loop):
                return i
        return None

    if isinstance(node, Not):
        if _eval(node.child, trace, pos, n_prefix, n_loop):
            return pos
        return None

    if isinstance(node, And):
        if not _eval(node.left, trace, pos, n_prefix, n_loop):
            return _find_violation(node.left, trace, pos, n_prefix, n_loop)
        return _find_violation(node.right, trace, pos, n_prefix, n_loop)

    return pos
