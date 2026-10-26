"""Tester rewrite: compile X-free contract operators into the GR(1) fragment SLUGS accepts.

Contracts are AUTHORED X-free (stutter-invariant, monitorable): persistence/frame use
weak-until `p W q`, liveness uses `G F(...)`, everything else is a state invariant. SLUGS,
however, is a next-state (GR(1)) synthesiser. This module is the source-to-source compilation
step between the two -- a temporal tester (Pnueli-Zaks, "On the Merits of Temporal Testers",
2008; the GR(1) target follows Maoz-Ringert, "GR(1) Synthesis for LTL Specification Patterns",
FSE 2015). For the shapes we use it collapses to the standard persistence idiom, so the SLUGS
input is exactly the GR(1) form the pipeline already synthesises today:

    p W q   -->   G((p) & !(q) -> X(p))          (pure safety; no auxiliary variable)

The SAME `p W q`, unmodified, is what the run-time monitor consumes as a 2-valued DFA
(assumption_monitor._PersistMon). One operator, two faces: synthesis gets the X form, the
monitor gets the X-free form. The next-operator therefore never appears in the contract the
author writes or the monitor reads -- only in this generated GR(1) encoding.
"""
from __future__ import annotations

import re
from typing import List


def _split_top(s: str, op: str) -> List[str]:
    """Split `s` on a top-level (paren-depth 0) whitespace-delimited operator token `op`."""
    parts, depth, i, last = [], 0, 0, 0
    pat = re.compile(r"\s" + re.escape(op) + r"\s")
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0:
            m = pat.match(s, i)
            if m:
                parts.append(s[last:i])
                last = m.end()
                i = m.end()
                continue
        i += 1
    parts.append(s[last:])
    return parts


def rewrite_formula(formula: str) -> str:
    """Rewrite one contract formula into GR(1)-native form. Currently: weak-until `p W q`
    at the top level -> the persistence-safety idiom. All other formulas pass through."""
    f = formula.strip()
    parts = _split_top(f, "W")
    if len(parts) == 2:
        p, q = parts[0].strip(), parts[1].strip()
        return f"G(({p}) & !({q}) -> X(({p})))"
    return f


def rewrite_ltl_text(text: str) -> str:
    """Apply `rewrite_formula` to every `;`-terminated formula inside Assumptions:/Guarantees:
    blocks of a CHASE .ltl file, leaving headers, variable declarations and structure intact."""
    out = []
    in_block = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if re.match(r"(Assumptions|Guarantees)\s*:", stripped):
            in_block = True
            out.append(line)
            continue
        if re.match(r"(CONTRACT\b|input\b|output\b|NAME\s*:)", stripped):
            in_block = in_block and not re.match(r"CONTRACT\b", stripped)
            out.append(line)
            continue
        if in_block and stripped.endswith(";") and " W " in f" {stripped} ":
            indent = line[:len(line) - len(line.lstrip())]
            body = stripped[:-1]
            out.append(f"{indent}{rewrite_formula(body)};")
        else:
            out.append(line)
    return out and "\n".join(out) or text
