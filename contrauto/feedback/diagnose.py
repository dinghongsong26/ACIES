"""Main refinement diagnosis engine.

Parses NuXmv counterexample output, checks each LTL formula from the
CHASE contracts against the trace, and produces a structured two-option
feedback message for the LLM repair agent.
"""

from __future__ import annotations

import re
from typing import Optional

from contrauto.feedback.contract_parser import LabeledFormula, extract_formulas
from contrauto.feedback.trace_checker import check_lasso, check_lasso_verbose


def diagnose_refinement(
    nuxmv_output: str,
    contract_content: str,
    mission: str,
) -> str:
    """Turn a failed NuXmv refinement check into two-option LLM repair feedback.

    Parses the counterexample lasso, classifies every contract formula as
    satisfied/violated on it, and routes to assumption- or guarantee-direction
    feedback. Falls back to a spec-level message when no trace is parseable.
    """
    spec_results = _parse_spec_results(nuxmv_output)
    assumption_failed = _assumption_direction_failed(spec_results)
    # When BOTH LTLSPECs fail the output holds TWO counterexample traces; parsing the whole
    # output would splice them into one bogus lasso (the 2nd trace's loop marker wins). Cut
    # the output at the '-- specification' boundaries and diagnose on the trace that belongs
    # to the direction we report: LTLSPEC 1 (assumption) is chunk 0, LTLSPEC 2 chunk 1.
    chunks = _split_by_spec(nuxmv_output)
    if chunks:
        target = chunks[0] if assumption_failed or len(chunks) == 1 else chunks[1]
    else:
        target = nuxmv_output
    prefix, loop = _parse_lasso_trace(target)

    if not loop:
        return _fallback_message(spec_results)

    formulas = extract_formulas(contract_content)
    parseable = [f for f in formulas if f.formula is not None]

    if not parseable:
        return _fallback_message(spec_results)

    system_name = _find_system_name(formulas)
    sub_names = sorted({f.contract for f in formulas if f.contract != system_name})

    violated = []
    satisfied = []
    for lf in parseable:
        result = check_lasso_verbose(lf.formula, prefix, loop)
        if result.satisfied:
            satisfied.append((lf, result))
        else:
            violated.append((lf, result))

    if assumption_failed:
        return _format_assumption_feedback(
            violated, satisfied, system_name, sub_names,
            prefix, loop, contract_content)
    else:
        return _format_guarantee_feedback(
            violated, satisfied, system_name, sub_names,
            prefix, loop, contract_content)


def _parse_spec_results(nuxmv_output: str) -> list[dict]:
    results = []
    for line in nuxmv_output.splitlines():
        if "-- specification" in line.lower() or "is true" in line or "is false" in line:
            is_true = "is true" in line
            is_false = "is false" in line
            if is_true or is_false:
                results.append({"line": line.strip(), "passed": is_true})
    return results


def _assumption_direction_failed(spec_results: list[dict]) -> bool:
    if not spec_results:
        return False
    return not spec_results[0]["passed"]


def _split_by_spec(nuxmv_output: str) -> list[str]:
    """Cut the NuXMV output into per-LTLSPEC chunks, each starting at its
    '-- specification' verdict line and containing that spec's counterexample trace."""
    starts = [m.start() for m in
              re.finditer(r"^-- (LTL )?specification", nuxmv_output, re.M)]
    return [nuxmv_output[s:e]
            for s, e in zip(starts, starts[1:] + [len(nuxmv_output)])]


def _parse_lasso_trace(
    nuxmv_output: str,
) -> tuple[list[dict[str, bool]], list[dict[str, bool]]]:
    """Parse ONE NuXMV counterexample into (prefix, loop) with FULL valuations.

    Two NuXMV output facts the naive parse got wrong (poisoning every downstream
    formula check): (1) states after the first print only the variables that
    CHANGED, so valuations must be accumulated, not taken per-block; (2) the
    '-- Loop starts here' marker precedes the loop-entry state's header while
    that state is still being filled, so it needs a +1. Pass a single-trace
    chunk (see _split_by_spec) — traces are not separated here."""
    states: list[dict[str, bool]] = []
    current: dict[str, bool] = {}
    loop_start: Optional[int] = None
    started = False

    for line in nuxmv_output.splitlines():
        stripped = line.strip()

        if "-- loop starts here" in stripped.lower():
            loop_start = len(states) + (1 if started else 0)

        if "-> State:" in stripped:
            if started:
                states.append(dict(current))
            started = True

        elif "=" in stripped and started:
            m = re.match(r'(\w+)\s*=\s*(TRUE|FALSE)', stripped)
            if m:
                current[m.group(1)] = m.group(2) == "TRUE"

    if started:
        states.append(dict(current))

    if not states:
        return [], []

    if loop_start is None:
        if len(states) >= 2:
            loop_start = len(states) - 1
        else:
            loop_start = 0

    prefix = states[:loop_start]
    loop = states[loop_start:]

    if not loop:
        loop = [states[-1]]

    return prefix, loop


def _find_system_name(formulas: list[LabeledFormula]) -> str:
    names = {f.contract for f in formulas}
    for n in names:
        if n.lower() == "system":
            return n
    return sorted(names)[-1] if names else "System"


def _output_owners(contract_content: str) -> dict[str, list[str]]:
    """Map variable -> contracts that declare it as an OUTPUT (System usually re-declares
    every output, so callers filter to sibling SUB-contracts when hinting repairs)."""
    owners: dict[str, list[str]] = {}
    current = None
    for line in contract_content.splitlines():
        m = re.match(r'\s*CONTRACT\s+(\w+)\s*:', line)
        if m:
            current = m.group(1)
            continue
        m = re.match(r'\s*output\s+boolean\s+variable\s+(\w+)\s*;', line)
        if m and current:
            owners.setdefault(m.group(1), []).append(current)
    return owners


def _format_trace(prefix: list[dict], loop: list[dict], max_vars: int = 15) -> str:
    lines = []
    all_vars = set()
    for s in prefix + loop:
        all_vars.update(s.keys())
    show_vars = sorted(all_vars)[:max_vars]

    for i, s in enumerate(prefix):
        vals = ", ".join(f"{v}={'1' if s.get(v) else '0'}" for v in show_vars)
        lines.append(f"  Step {i} (prefix): {{{vals}}}")

    lines.append(f"  --- Loop starts (step {len(prefix)}) ---")

    for i, s in enumerate(loop):
        vals = ", ".join(f"{v}={'1' if s.get(v) else '0'}" for v in show_vars)
        lines.append(f"  Step {len(prefix)+i} (loop):   {{{vals}}}")

    return "\n".join(lines)


def _fmt_formula(lf: LabeledFormula) -> str:
    return f"{lf.contract}.{lf.section}[{lf.index}]: {lf.raw_text}"


def _format_assumption_feedback(
    violated, satisfied, system_name, sub_names,
    prefix, loop, contract_content,
) -> str:
    # LTLSPEC 1 (assumption direction) failed: the culprit is an ASSUMPTION that is too strong /
    # inconsistent across the composition. Localize to the violated ASSUMPTIONS only -- a single bad
    # counterexample play also violates dozens of GUARANTEES (they are not the root cause here and
    # listing them mislabels guarantees as assumptions and buries the fix). Section filter, not just
    # contract filter, is what turns a 60-formula dump into the few offending assumptions.
    violated = [(lf, r) for lf, r in violated if lf.section == "assumption"]
    violated_comp = [(lf, r) for lf, r in violated if lf.contract != system_name]
    violated_sys = [(lf, r) for lf, r in violated if lf.contract == system_name]

    lines = [
        "REFINEMENT FAILED — Assumption direction (LTLSPEC 1)",
        "",
        "The system assumptions are satisfied but the composed assumptions",
        "are violated. Some sub-contract assumes too much from its environment.",
        "",
        "=== FORMULA-LEVEL DIAGNOSIS ===",
    ]

    if violated_comp:
        lines.append("\nViolated sub-contract assumption formulas:")
        for lf, r in violated_comp:
            lines.append(f"  ✗ {_fmt_formula(lf)}")
            lines.append(f"    → {r.explanation}")

    if violated_sys:
        lines.append("\nViolated system formulas:")
        for lf, r in violated_sys:
            lines.append(f"  ✗ {_fmt_formula(lf)}")
            lines.append(f"    → {r.explanation}")

    sat_comp_a = [(lf, r) for lf, r in satisfied
                  if lf.contract != system_name and lf.section == "assumption"]
    if sat_comp_a:
        lines.append(f"\nSatisfied sub-contract assumptions ({len(sat_comp_a)} formulas): all OK")

    # An assumption over a sibling's OUTPUT is not discharged by the environment at all —
    # the sanctioned fix is a matching sibling GUARANTEE (e.g. arbiter grant-justice), not
    # deleting the assumption. Only env-input assumptions get the plain "weaken" advice.
    # EXCEPT when two siblings assume over EACH OTHER's outputs: trace-level (saturated)
    # composition admits the play where both assumptions fail at once and every saturated
    # guarantee holds vacuously, so matching guarantees can never discharge the pair — one
    # side must drop its assumption instead (solver-verified on warehouse: turning the
    # arbiter's G F(!request) assumptions into `true;` flips refinement 0/2 -> 2/2).
    owners = _output_owners(contract_content)
    needs = []                      # aligned with violated_comp (LabeledFormula is unhashable)
    for lf, r in violated_comp:
        atoms = _collect_atoms(lf.formula) if lf.formula else set()
        needs.append(sorted({c for v in atoms for c in owners.get(v, [])
                             if c != system_name and c != lf.contract}))
    depends = {}
    for (lf, _), sibs in zip(violated_comp, needs):
        depends.setdefault(lf.contract, set()).update(sibs)
    cycles = sorted({tuple(sorted((a, b))) for a, ps in depends.items()
                     for b in ps if a in depends.get(b, set())})
    cyclic_contracts = {c for pair in cycles for c in pair}

    if cycles:
        lines.append("")
        lines.append("=== CIRCULAR DISCHARGE (root cause) ===")
        for a, b in cycles:
            lines.append(f"  {a} assumes over {b}'s outputs AND {b} assumes over {a}'s outputs.")
        lines.extend([
            "  Mutually-conditional assumptions admit a play where both sides' assumptions fail",
            "  at once and every saturated guarantee holds vacuously — adding matching guarantees",
            "  on both sides can NEVER fix this. BREAK the cycle: make ONE side unconditional.",
            "  Typically the resource provider (arbiter): replace its Assumptions block with",
            "  `true;` (or initial pins only) and KEEP its fairness guarantees; leave the",
            "  consumer's assumption in place — the unconditional guarantees then discharge it.",
        ])

    lines.extend([
        "",
        "=== REPAIR OPTIONS ===",
        "",
        "OPTION A (preferred) — For EACH flagged assumption, either make the composition",
        "discharge it or weaken it:",
    ])
    for (lf, r), sibs in zip(violated_comp, needs):
        lines.append(f"    - {_fmt_formula(lf)}")
        if sibs and lf.contract in cyclic_contracts and any(s in cyclic_contracts for s in sibs):
            lines.append("      ↳ part of the CIRCULAR DISCHARGE above — do NOT add a matching "
                         "guarantee; if this contract is the provider side, DELETE this "
                         "assumption (make the block `true;`), otherwise keep it.")
        elif sibs:
            lines.append(f"      ↳ mentions OUTPUT(s) of {', '.join(sibs)} — the environment can "
                         f"never discharge this. Either ADD a {', '.join(sibs)} GUARANTEE that "
                         "provides it (justice like G F(grant|!req) is fine; do NOT make a mutex "
                         "arbiter guarantee persistence — unrealizable), or mirror the assumption "
                         "verbatim into System.Assumptions (folded style), or weaken it.")
        else:
            lines.append("      ↳ over environment inputs — weaken/remove it here, or use "
                         "OPTION B if the real environment truly provides it.")

    lines.extend([
        "",
        "OPTION B — Strengthen system assumptions:",
        f"  Add new assumptions to the {system_name} contract that would",
        "  prevent this counterexample scenario.",
        "  WARNING: This changes the problem specification.",
    ])

    lines.extend([
        "",
        "=== COUNTEREXAMPLE TRACE ===",
        _format_trace(prefix, loop),
        "",
        "=== CURRENT .ltl ===",
        contract_content,
        "",
        "Choose one repair option and return corrected .ltl only.",
    ])

    return "\n".join(lines)


def _format_guarantee_feedback(
    violated, satisfied, system_name, sub_names,
    prefix, loop, contract_content,
) -> str:
    violated_sys_g = [(lf, r) for lf, r in violated
                      if lf.contract == system_name and lf.section == "guarantee"]
    satisfied_comp_g = [(lf, r) for lf, r in satisfied
                        if lf.contract != system_name and lf.section == "guarantee"]
    # Guarantee direction: only sub-contract GUARANTEES are relevant (a violated sub-contract
    # assumption on this trace is not what makes the composition promise too little). Section filter
    # so this does not dump the sub-contract's assumptions alongside the guarantee-side diagnosis.
    violated_comp = [(lf, r) for lf, r in violated
                     if lf.contract != system_name and lf.section == "guarantee"]

    var_owners = {}
    for lf, _ in violated_sys_g:
        if lf.formula:
            vars_in = _collect_atoms(lf.formula)
            for v in vars_in:
                var_owners.setdefault(v, set())

    lines = [
        "REFINEMENT FAILED — Guarantee direction (LTLSPEC 2)",
        "",
        "The composed contract is satisfied, but the system guarantee is",
        "violated. The sub-contracts do not promise enough to cover the",
        "system-level requirements.",
        "",
        "=== FORMULA-LEVEL DIAGNOSIS ===",
    ]

    if violated_sys_g:
        lines.append("\nViolated system guarantee formulas:")
        for lf, r in violated_sys_g:
            lines.append(f"  ✗ {_fmt_formula(lf)}")
            lines.append(f"    → {r.explanation}")

    if violated_comp:
        lines.append("\nViolated sub-contract formulas:")
        for lf, r in violated_comp:
            lines.append(f"  ✗ {_fmt_formula(lf)}")
            lines.append(f"    → {r.explanation}")

    if satisfied_comp_g:
        lines.append(f"\nSatisfied sub-contract guarantees ({len(satisfied_comp_g)} formulas):")
        shown = 0
        for lf, r in satisfied_comp_g:
            if shown < 5:
                lines.append(f"  ✓ {_fmt_formula(lf)}")
                shown += 1
        if len(satisfied_comp_g) > 5:
            lines.append(f"  ... and {len(satisfied_comp_g)-5} more (all satisfied)")

    target_contracts = set()
    for lf, _ in violated_sys_g:
        for sub in sub_names:
            target_contracts.add(sub)
    target_str = ", ".join(sorted(target_contracts)) or "sub-contracts"

    lines.extend([
        "",
        "=== REPAIR OPTIONS ===",
        "",
        f"OPTION A (preferred) — Strengthen sub-contract guarantees:",
        f"  Add new guarantees to {target_str} that ensure:",
    ])
    for lf, r in violated_sys_g:
        lines.append(f"    - {lf.raw_text}  can be derived from the composition")
    lines.extend([
        "  The current sub-contract guarantees are all satisfied but too",
        "  weak to imply these system-level goals.",
    ])

    lines.extend([
        "",
        f"OPTION B — Weaken system guarantees:",
        f"  Relax or remove these {system_name} guarantees:",
    ])
    for lf, r in violated_sys_g:
        lines.append(f"    - {_fmt_formula(lf)}")
    lines.append("  WARNING: This changes the system specification.")

    lines.extend([
        "",
        "=== COUNTEREXAMPLE TRACE ===",
        _format_trace(prefix, loop),
        "",
        "=== CURRENT .ltl ===",
        contract_content,
        "",
        "Choose one repair option and return corrected .ltl only.",
    ])

    return "\n".join(lines)


def _collect_atoms(node) -> set[str]:
    from contrauto.feedback.ltl_parser import Atom, Not, And, Or, Implies, Iff, G, F, X, U_
    if isinstance(node, Atom):
        return {node.name}
    if isinstance(node, (Not, G, F, X)):
        return _collect_atoms(node.child)
    if isinstance(node, (And, Or, Implies, Iff, U_)):
        return _collect_atoms(node.left) | _collect_atoms(node.right)
    return set()


def _fallback_message(spec_results: list[dict]) -> str:
    lines = ["REFINEMENT FAILED (formula-level diagnosis unavailable)",
             ""]
    for r in spec_results:
        lines.append(f"  {r['line']}")
    lines.append("")
    lines.append("No counterexample trace could be parsed.")
    lines.append("Check Mirror Principle: every System guarantee must appear "
                 "verbatim in the owning subsystem.")
    return "\n".join(lines)


# ── degeneracy (non-well-separated synthesis) diagnosis ───────────────────────
def diagnose_degeneracy(path_states, deadend_state, contract_content, contract_name=None) -> str:
    """Traceable diagnosis for a DEGENERATE (not well-separated) synthesized strategy.

    The 'counterexample' is the path init -> dead-end (analogous to the refinement lasso).
    We check every formula of the offending sub-contract along that path (looping on the stuck
    dead-end) and report which constraints the dead-end violates, so the repair targets a
    SPECIFIC formula instead of generic advice. `contract_name` restricts the check to that
    sub-contract (its trace only has its variables), avoiding cross-contract false positives.
    """
    formulas = extract_formulas(contract_content)
    parseable = [f for f in formulas
                 if f.formula is not None and (contract_name is None or f.contract == contract_name)]
    prefix = path_states[:-1] if path_states and len(path_states) > 1 else []
    loop = [deadend_state]
    if not parseable:
        return _degeneracy_fallback(deadend_state)

    violated = []
    for lf in parseable:
        try:
            r = check_lasso_verbose(lf.formula, prefix, loop)
        except Exception:
            continue
        if not r.satisfied:
            violated.append((lf, r))
    # The well-separation culprit is always a SAFETY/exclusion violation at the dead-end;
    # pure-liveness (G F ...) shows spuriously violated on this short finite trace — drop it.
    safety = [(lf, r) for lf, r in violated if "F(" not in lf.raw_text and "F " not in lf.raw_text]
    return _format_degeneracy_feedback(safety or violated, prefix, loop, deadend_state)


def _format_degeneracy_feedback(violated, prefix, loop, deadend_state) -> str:
    on = sorted(k for k, v in deadend_state.items() if v)
    lines = [
        "SYNTHESIS PRODUCED A DEGENERATE STRATEGY — the specification is NOT well-separated.",
        "The system can drive the play into a DEAD-END (a state with no legal move): it forces",
        "the ENVIRONMENT to break its own assumptions and then 'wins' vacuously. The trace below",
        "is the counterexample (initial -> dead-end); at the dead-end the listed constraints",
        "are violated.",
        "",
        "=== COUNTEREXAMPLE TRACE (initial -> dead-end) ===",
        _format_trace(prefix, loop),
        "",
        f"Dead-end state (true variables): {on}",
        "",
        "=== FORMULA-LEVEL DIAGNOSIS ===",
    ]
    guars = [(lf, r) for lf, r in violated if lf.section == "guarantee"]
    asms = [(lf, r) for lf, r in violated if lf.section == "assumption"]
    if guars:
        lines.append("VIOLATED GUARANTEES — the system broke its OWN rule here; FIX THESE FIRST:")
        for lf, r in guars:
            lines.append(f"  ✗ {_fmt_formula(lf)}")
            if r.explanation:
                lines.append(f"    → {r.explanation}")
    if asms:
        lines.append("Violated assumptions at the dead-end (context — the env state the system drove into):")
        for lf, r in asms:
            lines.append(f"  ✗ {_fmt_formula(lf)}")
    if not violated:
        lines.append("  (no parseable formula flagged; inspect the dead-end state above)")
    repair = [
        "",
        "=== HOW TO REPAIR (target the ✗ formulas above; keep it general) ===",
    ]
    if len(prefix) <= 1:                      # dead-end at step 0/1 -> outputs never pinned
        on_g = sorted(k for k, v in deadend_state.items() if v)
        repair.append(
            "- INITIAL DEGENERACY: the dead-end is at the very FIRST step, so the offending "
            f"outputs ({', '.join(on_g)}) were never pinned. PIN every output's initial value "
            "`!output;` at the start of Guarantees, and ensure the pairwise exclusions "
            "`G(!(a & b))` among them are ALL present. This is the most common fix.")
    lines += repair + [
        "- If a violated ENV assumption is an exclusion `G(!(x & y))` and the system forces",
        "  both x and y via reactions `G(act_x -> X(x))`, `G(act_y -> X(y))`, ADD the matching",
        "  SYSTEM exclusion `G(!(act_x & act_y))` so the system cannot force both at once.",
        "- Or weaken / drop the violated assumption so the environment always has a legal move.",
        "- Keep objectives as recurring justice `G F(progress)`, never a flag that latches once.",
        "Fix the specific ✗ formula(s); do NOT hard-code the scenario.",
    ]
    return "\n".join(lines)


def _degeneracy_fallback(deadend_state) -> str:
    on = sorted(k for k, v in deadend_state.items() if v)
    return ("SYNTHESIS DEGENERATE (not well-separated): the strategy reaches a dead-end with "
            f"true variables {on}. Make the environment non-blocking — do not let the system "
            "force contradictory environment states; add the matching system-side exclusions, "
            "or weaken the offending assumption.")


def violates_safety_guarantee(state, contract_content, contract_name) -> bool:
    """True iff `state` violates a SAFETY GUARANTEE of `contract_name` (the system's OWN
    obligation). A reachable dead-end is a real degeneracy ONLY in this case (the system
    broke its own rule / gave up). A dead-end that only violates an ASSUMPTION means the
    ENVIRONMENT broke its promise — that is a legitimate system win, NOT degeneracy."""
    for lf in extract_formulas(contract_content):
        if lf.contract != contract_name or lf.section != "guarantee" or lf.formula is None:
            continue
        if "F(" in lf.raw_text or "F " in lf.raw_text:      # skip liveness (G F ...)
            continue
        try:
            if not check_lasso(lf.formula, [], [state]):
                return True
        except Exception:
            continue
    return False
