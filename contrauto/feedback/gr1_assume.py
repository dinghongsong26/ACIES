"""Counterstrategy-guided assumption mining — the portable, template-based (B) direction.

When a controller is UNREALIZABLE, the fix is either to weaken a guarantee (the unrealizable
core, see `gr1_core`) or to ADD an environment assumption. This module does the latter: it
reads the environment counterstrategy (which inputs the environment holds FIXED to defeat the
system), enumerates candidate GR(1) environment-liveness assumptions (justice templates over
those inputs), and uses the realizability oracle to keep only the ones that actually make the
controller realizable — i.e. "add THIS assumption".

This is the template-based approximation of counterstrategy-guided assumption refinement
(Chatterjee–Henzinger–Jobstmann, *Environment Assumptions for Synthesis*, CONCUR 2008;
Alur–Moarref–Topcu, *Counter-Strategy Guided Refinement of GR(1)*, FMCAD 2013). The
state-of-the-art derives the assumption from the (counterstrategy, unrealizable-core) pair via
Craig interpolation (Cavezza–Alrajeh, *Interpolation-Based GR(1) Assumptions Refinement*,
TACAS 2017), which yields weaker assumptions but needs an interpolating solver — left for a
Spectra-backed path. At ContrAuto's contract scale, template enumeration over the few inputs
the counterstrategy actually holds fixed is sufficient and needs no new dependency (it reuses
SLUGS `--counterStrategy`, already invoked for the unrealizable diagnosis).
"""

from __future__ import annotations

import re
from typing import Optional


# ── candidate templates ───────────────────────────────────────────────────────
def candidate_assumptions(held: dict) -> list[str]:
    """GR(1) justice assumptions that force the environment off a value it holds fixed.

    `held` maps an input name to the constant value ("TRUE"/"FALSE") the environment keeps it
    at across its winning counterstrategy. If it pins x=FALSE forever, force it true infinitely
    often (`G F(x)`); if x=TRUE forever, force it false (`G F(!x)`).
    """
    out = []
    for var, val in held.items():
        lit = f"!{var}" if str(val).upper() == "TRUE" else var
        out.append(f"G F({lit})")
    return out


_NON_ATOMS = {"G", "F", "X", "U", "true", "false", "TRUE", "FALSE"}


def contract_io(contract_content: str, ctrl: str) -> tuple[list[str], list[str]]:
    """(inputs, outputs) declared by `ctrl`'s CONTRACT block."""
    ins, outs, cur = [], [], None
    for ln in contract_content.splitlines():
        m = re.match(r"\s*CONTRACT\s+(\w+)\s*:", ln)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"\s*(input|output)\s+boolean\s+variable\s+(\w+)\s*;", ln)
        if m and cur == ctrl:
            (ins if m.group(1) == "input" else outs).append(m.group(2))
    return ins, outs


def _atoms(formula: str) -> set:
    return {a for a in re.findall(r"[A-Za-z_]\w*", formula) if a not in _NON_ATOMS}


def candidate_transitions(core: list[str], in_vars: set, out_vars: set,
                          assumptions: list[str] = None) -> list[str]:
    """Persistence templates over the unrealizable core: for an input x mentioned in the core
    and an output c linked to x, propose `G((x & c) -> X(x))` — "x persists while c holds".
    Justice templates cannot express this: a 1-step pulse satisfies `G F(grant)` yet defeats
    a navigate-while-granted guarantee `G(nav -> grant)` (the robot can never travel), and a
    pulsing environment pins nothing, so held-input mining stays silent too. The condition c
    comes from the core's own outputs PLUS outputs co-occurring with x in the contract's
    assumptions (e.g. `G F(grant | !req)` links grant to req — the natural holding output —
    even though req appears in no core guarantee)."""
    core_atoms = set()
    for f in core or []:
        core_atoms |= _atoms(f)
    out = []
    for x in sorted(core_atoms & set(in_vars)):
        conds = core_atoms & set(out_vars)
        for a in assumptions or []:
            at = _atoms(a)
            if x in at:
                conds |= at & set(out_vars)
        out.extend(f"G(({x} & {c}) -> X({x}))" for c in sorted(conds) if c != x)
    return out


def mine_transitions(tools, contract_content: str, ctrl: str, core: list[str],
                     max_candidates: int = 8) -> tuple[list[str], list[str]]:
    """Solver-verified persistence assumptions for `ctrl`.

    Returns ``(singles, needed_set)``: `singles` = candidates that each INDIVIDUALLY restore
    realizability. When no single one does (typical with symmetric resource clusters — e.g.
    shelf and dock access are independently unrealizable, so fixing one leaves the other),
    try all candidates TOGETHER and greedily minimize; `needed_set` is that jointly-sufficient
    minimal set. Bounded: <= max_candidates + len(candidates) + 1 re-synthesis probes."""
    ins, outs = contract_io(contract_content, ctrl)
    try:
        from contrauto.feedback.contract_parser import extract_formulas
        asms = [lf.raw_text for lf in extract_formulas(contract_content)
                if lf.contract == ctrl and lf.section == "assumption"]
    except Exception:
        asms = []
    cands = candidate_transitions(core, set(ins), set(outs), asms)
    flat = contract_content.replace(" ", "")
    cands = [c for c in cands if c.replace(" ", "") not in flat][:max_candidates]

    def _with(cset):
        v = contract_content
        for c in cset:
            v = add_assumption(v, ctrl, c)
        return tools.check_realizable(ctrl, v) is True

    singles = [c for c in cands if _with([c])]
    if singles or len(cands) < 2:
        return singles, []
    if _with(cands):
        kept = list(cands)
        for c in list(kept):
            if len(kept) > 1 and _with([k for k in kept if k != c]):
                kept.remove(c)
        return [], kept
    return [], []


def mine_transitions_iterative(tools, contract_content: str, ctrl: str, core: list[str],
                               recompute_core, max_rounds: int = 3,
                               max_candidates: int = 8) -> list[str]:
    """Iterative persistence mining for MULTI-cluster unrealizability.

    A single QuickXplain core exposes only ONE unrealizable cluster (e.g. the dock chain);
    with symmetric resources the shelf chain is independently unrealizable too, so no
    assumption set derived from the first core alone can verify. Round r: tentatively add
    round-r candidates, recompute the core on the augmented spec (surfacing the next
    cluster), repeat; once the augmented spec IS realizable, greedily minimize the whole
    accumulated set. Returns the minimal jointly-sufficient set ([] if max_rounds exhausted
    without realizability). Bounded: <= max_rounds cores + ~(max_rounds*max_candidates)
    re-synthesis probes."""
    added: list[str] = []
    cur = contract_content
    cur_core = core

    def _with(cset):
        v = contract_content
        for c in cset:
            v = add_assumption(v, ctrl, c)
        return tools.check_realizable(ctrl, v) is True

    for _ in range(max_rounds):
        ins, outs = contract_io(cur, ctrl)
        try:
            from contrauto.feedback.contract_parser import extract_formulas
            asms = [lf.raw_text for lf in extract_formulas(cur)
                    if lf.contract == ctrl and lf.section == "assumption"]
        except Exception:
            asms = []
        cands = candidate_transitions(cur_core, set(ins), set(outs), asms)
        flat = cur.replace(" ", "")
        cands = [c for c in cands if c.replace(" ", "") not in flat][:max_candidates]
        if not cands:
            break
        for c in cands:
            cur = add_assumption(cur, ctrl, c)
        added += cands
        if tools.check_realizable(ctrl, cur) is True:
            kept = list(added)
            for c in list(kept):
                if len(kept) > 1 and _with([k for k in kept if k != c]):
                    kept.remove(c)
            return kept
        cur_core = recompute_core(cur) or cur_core
    return []


def format_transition_feedback(ctrl: str, singles: list[str],
                               needed_set: list[str] = None) -> str:
    """Render verified persistence assumptions as repair feedback for the LLM."""
    lines = [
        f"PERSISTENCE MINING for {ctrl}: the liveness goal is defeated not by an input that "
        "never comes but by one that does not LAST — the environment can satisfy a justice "
        "assumption with 1-step pulses while a guarantee needs the input to hold across steps.",
    ]
    if singles:
        lines.append(f"Adding ANY ONE of these persistence assumptions to {ctrl} makes it "
                     "realizable (verified by re-synthesis):")
        lines.extend(f"  + {w}" for w in singles)
    elif needed_set:
        lines.append(f"No single assumption suffices; adding this SET to {ctrl} makes it "
                     "realizable (verified by re-synthesis; the set is minimal — dropping any "
                     "one breaks it again):")
        lines.extend(f"  + {w}" for w in needed_set)
    lines += [
        "",
        "Use the form matching the intended environment behavior (e.g. a grant that stays "
        "while the robot keeps requesting/navigating). CROSS-CHECK: if a persisted variable "
        "is another sub-contract's OUTPUT (e.g. a grant), mirror the assumption VERBATIM "
        "into System.Assumptions (folded style) — do NOT make a mutex arbiter GUARANTEE "
        "persistence (unrealizable under never-releasing requesters). Fairness justice "
        "G F(grant|!req) is the only grant obligation the arbiter itself should carry.",
    ]
    return "\n".join(lines)


# ── inject an assumption into one contract ────────────────────────────────────
def add_assumption(contract_content: str, ctrl: str, formula: str) -> str:
    """Return `contract_content` with `formula` appended to ctrl's Assumptions section.

    Inserts just before ctrl's `Guarantees:` line (or the next CONTRACT / end of block),
    leaving every other line byte-identical.
    """
    lines = contract_content.splitlines()
    out = []
    cur = None
    in_ctrl_assumptions = False
    indent = "        "
    pending = f"{indent}{formula};"
    for ln in lines:
        s = ln.strip()
        m = re.match(r"CONTRACT\s+(\w+)\s*:", s)
        if m:
            if in_ctrl_assumptions:           # left ctrl block without seeing Guarantees
                out.append(pending)
                in_ctrl_assumptions = False
            cur = m.group(1)
            out.append(ln)
            continue
        if cur == ctrl and s == "Assumptions:":
            in_ctrl_assumptions = True
            out.append(ln)
            continue
        if in_ctrl_assumptions and s == "Guarantees:":
            out.append(pending)              # flush before the guarantees start
            in_ctrl_assumptions = False
            out.append(ln)
            continue
        out.append(ln)
    if in_ctrl_assumptions:                   # ctrl had no Guarantees line at all
        out.append(pending)
    return "\n".join(out).rstrip() + "\n"


# ── mining loop ───────────────────────────────────────────────────────────────
def mine_assumptions(tools, contract_content: str, ctrl: str, in_vars: set):
    """Find single assumptions whose addition makes `ctrl` realizable.

    `tools` is a :class:`contrauto.agents.tool_pipeline.ToolPipeline`. Returns
    ``(working, held)``: `working` = candidate assumption strings that each individually fix
    realizability; `held` = the inputs the environment was holding fixed (for context).
    """
    held = tools.counterstrategy_held_inputs(ctrl, in_vars)
    if not held:
        return [], {}
    working = []
    for cand in candidate_assumptions(held):
        variant = add_assumption(contract_content, ctrl, cand)
        if tools.check_realizable(ctrl, variant) is True:   # realizable now
            working.append(cand)
    return working, held


def held_hint(ctrl: str, held: dict) -> str:
    """DIRECTION-ONLY assumption feedback: name the inputs the environment holds fixed in its
    winning counterstrategy, WITHOUT prescribing (or re-synthesis-verifying) a specific assumption.
    The LLM decides what to add/relax. Contrast `mine_assumptions`/`format_assume_feedback`, which
    hand the LLM a verified `G F(x)` -- more help than a diagnosis should give, and expensive (a
    re-synthesis per candidate). This keeps the guarantee side PRECISE (the unrealizable core) and
    the assumption side DIRECTIONAL only."""
    held_str = ", ".join(f"{k}={v}" for k, v in held.items()) or "(none identified)"
    return (
        f"ASSUMPTION DIRECTION for {ctrl}: in its winning counterstrategy the environment keeps "
        f"these inputs pinned to defeat the system: {held_str}. If the fix is on the ASSUMPTION "
        f"side, add or relax an assumption about these input(s) so the environment must eventually "
        f"release them (you decide the exact form -- a justice G F(...), a transition G(p -> X q), "
        f"or have another contract's guarantee provide it). Otherwise fix a guarantee in the "
        f"unrealizable core above."
    )


def format_assume_feedback(ctrl: str, working: list[str], held: dict) -> str:
    """Render mined assumptions as repair feedback for the LLM."""
    held_str = ", ".join(f"{k}={v}" for k, v in held.items()) or "(none identified)"
    if not working:
        return (
            f"ASSUMPTION MINING for {ctrl}: the environment holds these inputs fixed in its "
            f"winning counterstrategy: {held_str}. No single justice assumption over them made "
            "the contract realizable, so either the fix is on the GUARANTEE side (see the "
            "unrealizable core) or a stronger/combined assumption is needed."
        )
    lines = [
        f"ASSUMPTION MINING for {ctrl}: the environment defeats the system by holding "
        f"{held_str} fixed forever. Adding ANY ONE of these environment-liveness assumptions "
        f"to {ctrl} makes it realizable (verified by re-synthesis):",
    ]
    for w in working:
        lines.append(f"  + {w}")
    lines += [
        "",
        "Pick the one that matches the intended environment behavior (e.g. a resource that is "
        "eventually granted, a location eventually reached). This forces the environment off "
        "the value it was exploiting, so the system's liveness goal becomes achievable.",
    ]
    return "\n".join(lines)
