"""Goal-witness check — every justice-goal OUTPUT needs a sensor-witness precondition.

Closes the cheap-exit starvation variant at design time (probe-verified on the pro
warehouse run: the FSM looped between two nodes asserting `done` forever — its justice
goal was satisfiable by pressing the button because `G(done -> delivered)` was never
written). Rule: for each sub-contract, every output variable occurring in a `G F(...)`
GUARANTEE must also appear as the antecedent of a precondition guarantee
`G(out -> φ)` where φ references at least one of the contract's INPUTS. Which input is
the right witness is semantics — the LLM's call — so this check emits localized feedback
rather than auto-patching (unlike anchoring/completion, there is no mechanically-safe fix).
"""

from __future__ import annotations

import re

from contrauto.feedback.contract_parser import extract_formulas
from contrauto.feedback.diagnose import _find_system_name
from contrauto.feedback.gr1_assume import contract_io, _atoms


def unwitnessed_goal_outputs(contract_content: str) -> list[tuple[str, str]]:
    """[(contract, output)] for justice-goal outputs lacking an input-referencing
    precondition `G(out -> ...input...)` in the same contract."""
    fs = [lf for lf in extract_formulas(contract_content) if lf.formula is not None]
    system = _find_system_name(fs)
    out = []
    io_cache: dict = {}
    for ctrl in sorted({lf.contract for lf in fs if lf.contract != system}):
        ins, outs = io_cache.setdefault(ctrl, contract_io(contract_content, ctrl))
        goals = [lf for lf in fs if lf.contract == ctrl and lf.section == "guarantee"
                 and lf.raw_text.lstrip().startswith("G F")]
        guards = [lf.raw_text for lf in fs
                  if lf.contract == ctrl and lf.section == "guarantee"
                  and not lf.raw_text.lstrip().startswith("G F")]
        for g in goals:
            # positively-occurring outputs in the goal (a negated output like !req in
            # G F(grant | !req) is a release disjunct, not an achievement to witness)
            body = g.raw_text
            for v in sorted(_atoms(body) & set(outs)):
                body_wo_neg = re.sub(rf"!\s*{re.escape(v)}\b", "", body)
                if not re.search(rf"\b{re.escape(v)}\b", body_wo_neg):
                    continue    # only negated occurrences: a release disjunct, not a goal
                witnessed = False
                for gd in guards:
                    m = re.match(rf"\s*G\s*\(\s*{re.escape(v)}\s*->(.*)\)\s*$", gd)
                    if m and (_atoms(m.group(1)) & set(ins)):
                        witnessed = True
                        break
                if not witnessed and (ctrl, v) not in out:
                    out.append((ctrl, v))
    return out


def format_witness_feedback(missing: list[tuple[str, str]]) -> str:
    lines = [
        "GOAL-WITNESS CHECK FAILED — justice goals over UNGUARDED outputs.",
        "A GR(1) strategy may satisfy G F(action) by asserting the action vacuously,",
        "forever, without any physical effect (certified starvation). Every goal output",
        "needs a precondition tying it to the SENSOR that witnesses its effect:",
    ]
    for c, v in missing:
        lines.append(f"  ✗ {c}: goal output `{v}` has no `G({v} -> ...sensor...)` guard "
                     f"(e.g. G({v} -> <the input that becomes true when {v} physically "
                     f"succeeds>))")
    lines += [
        "",
        "Add the missing precondition guarantee(s); choose the witness input that the",
        "simulator actually raises when the action takes physical effect.",
    ]
    return "\n".join(lines)
