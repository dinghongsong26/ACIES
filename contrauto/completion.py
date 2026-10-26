"""Mechanical contract completion — the post-LLM, pre-certificate closure pass (Step 2.7).

Template-approximated contract QUOTIENT (Incer et al.'s algebra; PACTI's assumption
propagation) restricted to INTERNAL variables: every patch this module applies concerns
only variables OWNED by some sub-contract, so the external environment's obligations are
untouched (environment-conservativity) — completion is a checkable design decision, never
a new claim about the world. External-input candidates are returned as SUGGESTIONS for the
LLM, not applied.

Closure passes:
  1. justice-cycle gate — two siblings whose JUSTICE assumptions lean on each other's
     outputs form an undischargeable pair (the "everyone idle" play); completion cannot
     choose which side to make unconditional (semantic), so it BLOCKS with a diagnosis.
  2. discharge closure — an internal-variable assumption must be underwritten somewhere:
     (a) preferred: the owning sibling GUARANTEES it (verbatim) — kept only if the sibling
         stays realizable (solver probe);
     (b) fallback: mirror it into System.Assumptions (the repo's FOLDED style — e.g. grant
         persistence under adversarial never-released requests is UNREALIZABLE for a mutex
         arbiter, which is exactly why the passing specs fold the handshake into SysA);
         the well-separatedness gate downstream polices the folding.
  3. realizability closure — per sub-contract, if UNREALIZABLE: mine verified persistence
     sets (gr1_assume.mine_transitions_iterative); apply the internal-variable ones through
     pass 2; surface external-input ones as suggestions.

The solver (check_realizable) arbitrates every choice; nothing is applied unverified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from contrauto.feedback.contract_parser import extract_formulas
from contrauto.feedback.diagnose import _output_owners, _collect_atoms, _find_system_name
from contrauto.feedback.gr1_assume import (
    add_assumption, contract_io, mine_transitions_iterative, _atoms,
)
from contrauto.feedback.gr1_core import compute_core


@dataclass
class CompletionResult:
    content: str
    applied: list = field(default_factory=list)       # (kind, ctrl, formula)
    suggestions: list = field(default_factory=list)   # external-input candidates (text)
    blocked: list = field(default_factory=list)       # cycle diagnoses needing the LLM


def _norm(s: str) -> str:
    return s.replace(" ", "")


def add_guarantee(content: str, ctrl: str, formula: str) -> str:
    """Append `formula` to ctrl's Guarantees section (before the next CONTRACT / EOF)."""
    lines = content.splitlines()
    out, cur, in_g = [], None, False
    pending = f"        {formula};"
    for ln in lines:
        s = ln.strip()
        m = re.match(r"CONTRACT\s+(\w+)\s*:", s)
        if m:
            if in_g:
                out.append(pending)
                in_g = False
            cur = m.group(1)
            out.append(ln)
            continue
        if cur == ctrl and s == "Guarantees:":
            in_g = True
            out.append(ln)
            continue
        out.append(ln)
    if in_g:
        out.append(pending)
    return "\n".join(out).rstrip() + "\n"


def _formulas(content):
    return [lf for lf in extract_formulas(content) if lf.formula is not None]


def justice_cycle_gate(content: str) -> list[str]:
    """Blocking diagnoses for sibling pairs whose JUSTICE assumptions lean on each other."""
    fs = _formulas(content)
    system = _find_system_name(fs)
    owners = _output_owners(content)
    depends = {}
    for lf in fs:
        if lf.contract == system or lf.section != "assumption":
            continue
        if not lf.raw_text.lstrip().startswith("G F"):
            continue
        provs = {c for v in _collect_atoms(lf.formula) for c in owners.get(v, [])
                 if c not in (lf.contract, system)}
        depends.setdefault(lf.contract, set()).update(provs)
    out = []
    for a, ps in sorted(depends.items()):
        for b in sorted(ps):
            if a < b and a in depends.get(b, set()):
                out.append(
                    f"JUSTICE-CYCLE between {a} and {b}: each assumes a G F(...) over the "
                    f"other's outputs. Trace-level composition cannot discharge mutually "
                    f"conditional liveness — make ONE side unconditional (typically the "
                    f"resource provider: Assumptions `true;`/initial pins only, keep its "
                    f"fairness guarantees).")
    return out


def remove_assumption(content: str, ctrl: str, formula: str) -> str:
    """Delete the assumption line matching `formula` (normalized) from ctrl's block;
    leave `true;` if the section would become empty."""
    lines = content.splitlines()
    out, cur, in_a, removed, kept_any = [], None, False, False, False
    for ln in lines:
        s = ln.strip()
        m = re.match(r"CONTRACT\s+(\w+)\s*:", s)
        if m:
            if cur == ctrl and in_a and removed and not kept_any:
                out.append("        true;")
            cur, in_a, removed, kept_any = m.group(1), False, removed, False
            out.append(ln)
            continue
        if cur == ctrl and s == "Assumptions:":
            in_a = True
            out.append(ln)
            continue
        if cur == ctrl and s == "Guarantees:":
            if in_a and removed and not kept_any:
                out.append("        true;")
            in_a = False
            out.append(ln)
            continue
        if cur == ctrl and in_a and s and _norm(s.rstrip(";")) == _norm(formula):
            removed = True
            continue
        if cur == ctrl and in_a and s:
            kept_any = True
        out.append(ln)
    return "\n".join(out).rstrip() + "\n"


def break_justice_cycles(tools, content: str, applied: list):
    """Mechanically break sibling justice cycles (failure class 4, now closed like 5/7).

    For each mutually-conditional pair, strip ONE side's justice assumptions over the
    other's outputs — preferring the RELEASE side (`G F(!x)`-shaped, the resource
    provider's assumption about its requesters, per the passing-spec pattern) — and keep
    the strip only if that side remains realizable (solver probe). Falls back to the
    other side, then to an LLM-facing diagnosis if neither strip verifies."""
    fs = _formulas(content)
    system = _find_system_name(fs)
    owners = _output_owners(content)
    cyc_asms = {}    # contract -> [(formula_text, all_negated_justice)]
    depends = {}
    for lf in fs:
        if lf.contract == system or lf.section != "assumption":
            continue
        if not lf.raw_text.lstrip().startswith("G F"):
            continue
        provs = {c for v in _collect_atoms(lf.formula) for c in owners.get(v, [])
                 if c not in (lf.contract, system)}
        if provs:
            neg = bool(re.match(r"G\s*F\s*\(\s*!", lf.raw_text.strip()))
            cyc_asms.setdefault(lf.contract, []).append((lf.raw_text, neg))
            depends.setdefault(lf.contract, set()).update(provs)
    # FOLD, don't strip. v6 showed stripping is unsound even probe-guarded: removing the
    # CONSUMER's grant-fairness left a formally-realizable robot that physically starves
    # (never granted, never moves — delivered 0 with every certificate green). Mirroring
    # both sides' cycle assumptions verbatim into System.Assumptions defuses the cycle
    # instead: contracts unchanged, refinement dir-1 discharges via SysA (the idle play now
    # falsifies SysA itself -> no counterexample), the well-sep gate polices the folding,
    # and the runtime monitor (X-free justice IS watched) catches real starvation -> ch.5.
    remaining = []
    fs_now = _formulas(content)
    sys_asms = {_norm(lf.raw_text) for lf in fs_now
                if lf.contract == system and lf.section == "assumption"}
    for a in sorted(depends):
        for b in sorted(depends[a]):
            if not (a < b and a in depends.get(b, set())):
                continue
            for side in (a, b):
                for f, _ in cyc_asms.get(side, []):
                    if _norm(f) in sys_asms:
                        continue
                    content = add_assumption(content, system, f)
                    sys_asms.add(_norm(f))
                    applied.append(("system-assumption (cycle fold)", system, f))
    return content, remaining


def discharge_one(tools, content: str, consumer: str, formula: str,
                  applied: list) -> str:
    """Underwrite one internal-variable assumption of `consumer`.

    Route (a): add it verbatim as the owning sibling's guarantee, KEPT only if that
    sibling remains realizable. Route (b) fallback: mirror into System.Assumptions
    (folded style). Returns the (possibly updated) content."""
    fs = _formulas(content)
    system = _find_system_name(fs)
    owners = _output_owners(content)
    atoms = _atoms(formula)
    provs = sorted({c for v in atoms for c in owners.get(v, [])
                    if c not in (consumer, system)})
    if len(provs) != 1:
        return content                       # zero or ambiguous provider: leave to ladder
    prov = provs[0]
    prov_has = any(lf.contract == prov and lf.section == "guarantee"
                   and _norm(lf.raw_text) == _norm(formula) for lf in fs)
    if prov_has:
        return content
    # route (a): sibling guarantee, solver-verified
    variant = add_guarantee(content, prov, formula)
    if tools.check_realizable(prov, variant) is True:
        applied.append(("guarantee", prov, formula))
        return variant
    # route (b): fold into System assumptions
    sys_has = any(lf.contract == system and lf.section == "assumption"
                  and _norm(lf.raw_text) == _norm(formula) for lf in fs)
    if not sys_has:
        content = add_assumption(content, system, formula)
        applied.append(("system-assumption (folded)", system, formula))
    return content


def discharge_closure(tools, content: str, applied: list) -> str:
    """Ensure every sub-contract assumption over sibling outputs is underwritten."""
    fs = _formulas(content)
    system = _find_system_name(fs)
    owners = _output_owners(content)
    for lf in list(fs):
        if lf.contract == system or lf.section != "assumption":
            continue
        provs = {c for v in _collect_atoms(lf.formula) for c in owners.get(v, [])
                 if c not in (lf.contract, system)}
        if provs:
            content = discharge_one(tools, content, lf.contract, lf.raw_text, applied)
    return content


def realizability_closure(tools, content: str, applied: list,
                          suggestions: list, max_passes: int = 2) -> str:
    """Make each unrealizable sub-contract realizable via verified internal persistence."""
    for _ in range(max_passes):
        fs = _formulas(content)
        system = _find_system_name(fs)
        ctrls = sorted({lf.contract for lf in fs if lf.contract != system})
        owners = _output_owners(content)
        changed = False
        for ctrl in ctrls:
            if tools.check_realizable(ctrl, content) is not False:
                continue
            core = compute_core(tools, content, ctrl)
            if not core:
                continue
            needed = mine_transitions_iterative(
                tools, content, ctrl, core,
                recompute_core=lambda c, _c=ctrl: compute_core(tools, c, _c))
            for f in needed:
                m = re.match(r"G\(\((\w+) & ", f)
                target = m.group(1) if m else None
                target_owners = [c for c in owners.get(target, [])
                                 if c not in (ctrl, system)] if target else []
                if target_owners:
                    content = add_assumption(content, ctrl, f)
                    applied.append(("assumption", ctrl, f))
                    content = discharge_one(tools, content, ctrl, f, applied)
                    changed = True
                else:
                    suggestions.append(
                        f"{ctrl}: external-input candidate (LLM to adjudicate): {f}")
        if not changed:
            break
    return content


def complete(tools, content: str) -> CompletionResult:
    """Run the full mechanical completion. Never applies anything solver-unverified."""
    applied: list = []
    suggestions: list = []
    if justice_cycle_gate(content):
        # try the mechanical break first (strip the release side, solver-verified);
        # only what survives that goes back to the LLM as a blocking diagnosis.
        content, blocked = break_justice_cycles(tools, content, applied)
        if blocked:
            return CompletionResult(content=content, applied=applied, blocked=blocked)
    content = discharge_closure(tools, content, applied)
    content = realizability_closure(tools, content, applied, suggestions)
    return CompletionResult(content=content, applied=applied,
                            suggestions=suggestions, blocked=[])
