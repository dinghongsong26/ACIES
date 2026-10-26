"""Minimal unrealizable core for a sub-contract's guarantees (QuickXplain).

When SLUGS reports a controller UNREALIZABLE, the existing counterstrategy hint
(`ToolPipeline.counterstrategy_diagnosis`) only flags inputs the environment holds fixed —
an intuitive pointer, not a precise localization. This module adds the refinement-quality
analogue: a SOLVER-based minimal unrealizable core over the controller's guarantee formulas,
i.e. a smallest subset of guarantees that is *still* unrealizable under the same assumptions.

It exploits monotonicity — more guarantees = stronger system obligation = harder to realize,
so "unrealizable" is upward-closed in the guarantee set — which is exactly what lets
QuickXplain (Junker, AAAI 2004) find a core in O(k·log(n/k)) realizability checks (k = core
size) instead of the O(n) of naive deletion. Each check is one logics_tool + SLUGS call; no
LLM, deterministic. Related: Cimatti–Roveri–Schuppan–Tchaltsev, *Diagnostic Information for
Realizability* (VMCAI 2008); Könighofer–Hofferek–Bloem, model-based diagnosis of unrealizable
specs (HVC 2010).

The core narrows the repair to a named handful of guarantees ("{Robot.g2, Robot.g5} are
jointly unrealizable"); whether the fix is to weaken one of them or to ADD a supporting
assumption (counter-strategy-guided assumption mining, Alur–Moarref–Topcu, FMCAD 2013) is the
complementary direction, left to the LLM with this localization in hand.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from contrauto.feedback.contract_parser import extract_formulas


# ══════════════════════════════════════════════════════════════════════════════
# QuickXplain (Junker 2004) — pure, oracle-driven, no tools.
# ══════════════════════════════════════════════════════════════════════════════
def minimal_unrealizable_core(
    candidates: list,
    is_unrealizable: Callable[[list], bool],
) -> Optional[list]:
    """Return a minimal subset of `candidates` that is still unrealizable.

    ``is_unrealizable(subset)`` must return True iff the fixed background plus ``subset``
    (as the controller's guarantees) is unrealizable. Monotonicity is assumed: a superset of
    an unrealizable subset is unrealizable.

    Returns:
        * ``[]``   — unrealizable even with NO guarantees (the cause is in the assumptions /
          background, not the guarantees).
        * ``None`` — realizable with ALL candidates (nothing to localize).
        * otherwise — an order-preserving minimal unrealizable subset.
    """
    if not is_unrealizable(list(candidates)):
        return None
    if is_unrealizable([]):
        return []
    return _qxp([], list(candidates), is_unrealizable, delta_nonempty=False)


def _qxp(background, cand, is_unreal, delta_nonempty):
    """QuickXplain' recursion. Invariant: is_unreal(background) is False on entry (unless
    delta added since the last check makes it True, handled below); is_unreal(background +
    cand) is True. Returns a minimal D ⊆ cand with is_unreal(background + D)."""
    if delta_nonempty and is_unreal(background):
        return []
    if len(cand) == 1:
        return list(cand)
    mid = len(cand) // 2
    c1, c2 = cand[:mid], cand[mid:]
    d2 = _qxp(background + c1, c2, is_unreal, len(c1) > 0)
    d1 = _qxp(background + d2, c1, is_unreal, len(d2) > 0)
    return d1 + d2


# ══════════════════════════════════════════════════════════════════════════════
# .ltl guarantee projection — keep a chosen subset of a controller's time-tense
# guarantees, leaving its var decls / Assumptions / init pins and every OTHER
# contract byte-identical. Relies on the normalized layout from
# DecomposerAgent._enforce_initial_conditions (one formula per line, `!out;` pins).
# ══════════════════════════════════════════════════════════════════════════════
def _is_initial(body: str) -> bool:
    """An initial-condition line: `true` or a single (optionally negated) atom, no operator."""
    t = body.strip().rstrip(";").strip()
    return t == "true" or bool(re.fullmatch(r"!?\s*[A-Za-z_]\w*", t))


def guarantee_candidates(contract_content: str, ctrl: str) -> list[str]:
    """The controller's time-tense guarantee formulas (raw text), excluding init pins."""
    out = []
    for lf in extract_formulas(contract_content):
        if lf.contract == ctrl and lf.section == "guarantee" and not _is_initial(lf.raw_text):
            out.append(lf.raw_text)
    return out


def project_guarantees(contract_content: str, ctrl: str, keep: set[str]) -> str:
    """Rebuild `contract_content` with ctrl's Guarantees = its init pins + formulas in `keep`.

    Every other line (other contracts, ctrl's decls/Assumptions, the init pins) is preserved
    verbatim; only ctrl's non-initial guarantee lines NOT in `keep` are dropped.
    """
    out = []
    cur = None
    section = None
    for ln in contract_content.splitlines():
        s = ln.strip()
        m = re.match(r"CONTRACT\s+(\w+)\s*:", s)
        if m:
            cur, section = m.group(1), None
            out.append(ln)
            continue
        if s == "Assumptions:":
            section = "A"; out.append(ln); continue
        if s == "Guarantees:":
            section = "G"; out.append(ln); continue
        if cur == ctrl and section == "G" and s and not s.startswith("#"):
            body = " ".join(s.rstrip(";").split())   # normalize whitespace to match candidates
            if _is_initial(body) or body in keep:
                out.append(ln)
            # else: drop this guarantee from the projection
            continue
        out.append(ln)
    return "\n".join(out).rstrip() + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration — wire the oracle to the toolchain and run the core.
# ══════════════════════════════════════════════════════════════════════════════
def compute_core(tools, contract_content: str, ctrl: str) -> Optional[list[str]]:
    """Compute the minimal unrealizable core of `ctrl`'s guarantees via the toolchain.

    `tools` is a :class:`contrauto.agents.tool_pipeline.ToolPipeline`. Returns the core (list
    of raw guarantee strings), ``[]`` (assumptions/background are the cause), or ``None`` if
    localization was inconclusive (a tool error on some subset, or realizable with all).
    """
    candidates = guarantee_candidates(contract_content, ctrl)
    if not candidates:
        return None

    inconclusive = {"hit": False}

    def is_unreal(subset: list[str]) -> bool:
        # A controller with NO time-tense guarantees has no obligation it can fail, so it is
        # realizable by definition. We short-circuit instead of calling the tool, because
        # logics_tool/SLUGS cannot synthesize a guarantee-less spec and returns no verdict --
        # which would otherwise mark the whole core computation inconclusive and abort it (the
        # empty subset is always probed first, to test whether the cause is in the assumptions).
        if not subset:
            return False
        variant = project_guarantees(contract_content, ctrl, set(subset))
        verdict = tools.check_realizable(ctrl, variant)   # True=realizable, False=unrealizable
        if verdict is None:
            inconclusive["hit"] = True
            return False     # be conservative: don't claim unrealizability we couldn't confirm
        return verdict is False

    core = minimal_unrealizable_core(candidates, is_unreal)
    if inconclusive["hit"]:
        return None
    return core


def compute_consistency_core(tools, contract_content: str, ctrl: str) -> Optional[list[str]]:
    """Minimal INCONSISTENT core of `ctrl`'s guarantees: a smallest subset that is jointly
    unsatisfiable (the Step-3 consistency analogue of the unrealizable core).

    Same QuickXplain machinery, with the consistency oracle (``tools.check_consistent``) instead
    of realizability. Monotone: a superset of an inconsistent set is inconsistent. Returns the
    core (list of raw guarantees), or ``None`` if inconclusive / all-consistent.
    """
    candidates = guarantee_candidates(contract_content, ctrl)
    if not candidates:
        return None
    inconclusive = {"hit": False}

    def is_inconsistent(subset: list[str]) -> bool:
        if not subset:
            return False    # no guarantees = trivially satisfiable (never inconsistent)
        variant = project_guarantees(contract_content, ctrl, set(subset))
        verdict = tools.check_consistent(ctrl, variant)   # True=consistent, False=inconsistent
        if verdict is None:
            inconclusive["hit"] = True
            return False
        return verdict is False

    core = minimal_unrealizable_core(candidates, is_inconsistent)
    if inconclusive["hit"]:
        return None
    return core


def format_consistency_core_feedback(ctrl: str, core: list[str]) -> str:
    """Render the inconsistent core as formula-level repair feedback."""
    if not core:
        return (
            f"INCONSISTENT for {ctrl}: the contract is unsatisfiable even with none of its "
            "time-tense guarantees, so the conflict is in the ASSUMPTIONS or initial conditions, "
            "not the guarantees."
        )
    lines = [f"INCONSISTENT CORE for {ctrl} (minimal set of guarantees that CONTRADICT each other):"]
    for f in core:
        lines.append(f"  ✗ {f}")
    lines += [
        "",
        "No behavior can satisfy all of these at once. Removing any one makes the rest "
        "satisfiable, so weaken or fix ONE of them. Do NOT touch guarantees outside this core.",
    ]
    return "\n".join(lines)


def format_core_feedback(ctrl: str, core: list[str]) -> str:
    """Render the unrealizable core as formula-level repair feedback."""
    if not core:
        return (
            f"UNREALIZABLE CORE for {ctrl}: even with NONE of its time-tense guarantees, the "
            "contract is unrealizable. The cause is NOT the guarantees but the ASSUMPTIONS / "
            "background (e.g. an input the environment can hold fixed with no assumption forcing "
            "it, or contradictory env constraints). Add or fix the relevant ASSUMPTION rather "
            "than touching the guarantees."
        )
    lines = [
        f"UNREALIZABLE CORE for {ctrl} (minimal set of guarantees that is JOINTLY unrealizable):",
    ]
    for f in core:
        lines.append(f"  ✗ {f}")
    lines += [
        "",
        "These guarantees CANNOT all be met together under the current assumptions. Removing "
        "any one of them makes the rest realizable, so the fix is targeted at THIS set:",
        "  (a) weaken/relax ONE of these guarantees, or",
        "  (b) ADD an assumption that supplies what they need (e.g. a justice G F(input) or a "
        "transition G(p -> X q) that forces the environment to cooperate).",
        "Do NOT touch guarantees outside this core — they are not the problem.",
        "CROSS-CHECK for (b): if the assumption you add mentions a variable that another",
        "sub-contract OWNS as an output (e.g. a grant), it must be discharged: EITHER the",
        "owning sibling guarantees it (ONLY if the sibling stays realizable — a mutex",
        "arbiter can guarantee fairness G F(grant|!req), but persistence-while-requested",
        "makes it UNREALIZABLE under never-releasing requesters), OR mirror the assumption",
        "verbatim into System.Assumptions (folded style — the right home for grant",
        "persistence). Do NOT give the arbiter obligations it cannot meet unconditionally.",
    ]
    return "\n".join(lines)
