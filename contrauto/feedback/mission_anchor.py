"""Step 2.6 — mission anchoring: the System contract must CARRY the translator's formulas.

Refinement proves composition ⊑ System, but System itself is LLM-written: nothing stopped
the model from quietly WEAKENING it (dropping the mission's action mutexes and carrying
dynamics) until "assert every action forever" became a certified strategy with zero
physical effect (spec erosion — failure class 7). This check anchors the ladder: every
top-level conjunct of every translator LTL pair must appear verbatim (whitespace-
normalized) in the System contract — under Assumptions or Guarantees, placement is the
designer's choice; PRESENCE is not.

Purely textual by design: with the translator vocabulary-pinned (same grounding names end
to end), verbatim inclusion is well-defined, mechanical, and yields formula-level feedback
("you dropped these N formulas"), which is exactly the localization the repair loop needs.
One relaxation: a dropped `G(A & B & ...)` also counts as present when every inner
conjunct appears somewhere in System (the LLM may legally split one G into several).
"""

from __future__ import annotations


def _norm(s: str) -> str:
    return "".join(s.split())


def split_top_conjuncts(formula: str) -> list[str]:
    """Split on top-level '&' only (parenthesis-aware)."""
    out, depth, cur = [], 0, []
    i, n = 0, len(formula)
    while i < n:
        ch = formula[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "&" and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return [c for c in out if c]


def _system_block(contract_content: str) -> str:
    """Text of the CONTRACT System block (to end or next CONTRACT)."""
    marker = None
    for cand in ("CONTRACT System:", "CONTRACT system:"):
        if cand in contract_content:
            marker = cand
            break
    if marker is None:
        return ""
    tail = contract_content.split(marker, 1)[1]
    nxt = tail.find("CONTRACT ")
    return tail if nxt < 0 else tail[:nxt]


def _inner_split_matches(conjunct: str, sys_norm: str) -> bool:
    """Relaxation: `G(A & B)` counts as present if G(A)-style pieces all appear."""
    c = conjunct.strip()
    if not (c.startswith("G(") and c.endswith(")")):
        return False
    inner = c[2:-1]
    parts = split_top_conjuncts(inner)
    if len(parts) < 2:
        return False
    return all(_norm(p) in sys_norm for p in parts)


def missing_mission_formulas(ltl_pairs: list[dict], contract_content: str) -> list[str]:
    """Translator conjuncts absent from the System contract (normalized match)."""
    sys_norm = _norm(_system_block(contract_content))
    if not sys_norm:
        return ["<no CONTRACT System block found>"]
    missing = []
    for p in ltl_pairs:
        for c in split_top_conjuncts(p.get("ltl", "")):
            if _norm(c) in sys_norm:
                continue
            if _inner_split_matches(c, sys_norm):
                continue
            missing.append(c)
    return missing


def placement_of(formula: str, sys_inputs: set, sys_outputs: set) -> str:
    """'assumption' | 'guarantee' for a mission conjunct, by a mechanical rule.

    The temporal HEAD decides: an X(...) consequent over inputs = environment dynamics
    (folded) -> assumption; over any output -> system obligation. A justice G F(...)
    mentioning an input = env fairness (arrival/grant) -> assumption; pure-output justice
    (G F(pickup)) -> guarantee. U-formulas or pure-input formulas involving inputs ->
    assumption (env holding behaviour). Everything else -> guarantee. Misplacements are
    not fatal — consistency/refinement catch them with localized feedback downstream."""
    import re as _re
    atoms = {a for a in _re.findall(r"[A-Za-z_]\w*", formula)
             if a not in ("G", "F", "X", "U", "true", "false")}
    xvars = _re.findall(r"X\(\s*!?\s*(\w+)", formula)
    if xvars:
        return "assumption" if all(v in sys_inputs for v in xvars) else "guarantee"
    stripped = formula.strip()
    if stripped.startswith("G F") or stripped.startswith("GF"):
        return "assumption" if atoms & sys_inputs else "guarantee"
    if " U " in formula and atoms & sys_inputs:
        return "assumption"
    if atoms and atoms <= sys_inputs:
        return "assumption"
    return "guarantee"


def anchor_complete(ltl_pairs: list[dict], contract_content: str):
    """Mechanically insert missing mission conjuncts into the System contract.

    Returns (new_content, added) where added = [(section, formula)]. Spec-erosion repair
    is mechanical: the formulas are KNOWN (translator output) and the placement rule is
    syntactic — burning LLM rounds on transcription wasted a third of the attempt budget."""
    from contrauto.feedback.gr1_assume import add_assumption, contract_io
    from contrauto.completion import add_guarantee
    missing = missing_mission_formulas(ltl_pairs, contract_content)
    if not missing or missing[0].startswith("<no CONTRACT System"):
        return contract_content, []
    # find the System contract's declared I/O for the placement rule
    sys_name = "System"
    ins, outs = contract_io(contract_content, sys_name)
    added = []
    for f in missing:
        sec = placement_of(f, set(ins), set(outs))
        if sec == "assumption":
            contract_content = add_assumption(contract_content, sys_name, f)
        else:
            contract_content = add_guarantee(contract_content, sys_name, f)
        added.append((sec, f))
    return contract_content, added


def format_anchor_feedback(missing: list[str]) -> str:
    lines = [
        "MISSION ANCHORING FAILED — the CONTRACT System block DROPPED mission formulas.",
        "The System contract is the specification the whole certificate chain proves",
        "against; weakening it makes every downstream certificate vacuous. Reinstate each",
        "of these VERBATIM in the System contract (environment-behaviour formulas under",
        "Assumptions, system obligations under Guarantees):",
    ]
    lines += [f"  ✗ {m}" for m in missing]
    lines += [
        "",
        "Do not paraphrase, rename variables, or weaken operators. Sub-contracts must then",
        "jointly imply the System guarantees as usual.",
    ]
    return "\n".join(lines)
