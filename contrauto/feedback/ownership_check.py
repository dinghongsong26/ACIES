"""Detect a sub-contract referencing another node's PRIVATE point-to-point signal.

A point-to-point edge signal (e.g. `grant_r4`, declared once on the Robot4<->Arbiter
edge) belongs exclusively to that edge's two endpoints. For an N-robot symmetric
instance the LLM occasionally writes the wrong index -- e.g. Robot4's own charging
guarantee references `grant_r1` (Robot1's grant) instead of its own `grant_r4` --
because every robot's contract looks almost identical and only the index differs.
NuXMV/SLUGS cannot catch this: `grant_r1` is a perfectly valid variable, just not one
Robot4 should be reading. Left uncaught, it makes Robot4 UNREALIZABLE (or, worse,
realizable but semantically wrong) many attempts deep, long after the mistake was
made, wasting the repair budget on symptoms instead of the cause.

This check is purely mechanical: it derives signal ownership from the TaskSpec's
edges (the same derivation `TaskSpec._expand_edge_interfaces` uses to name variables)
and flags any contract that references a private signal it does not own. A plain node
proposition (e.g. `charging`, `battery_low`) is NOT a point-to-point signal -- siblings
may legitimately read another robot's `charging` output for mutual exclusion -- so
only edge-declared signals are checked.
"""

from __future__ import annotations

from dataclasses import dataclass

from contrauto.feedback.contract_parser import extract_formulas


@dataclass(frozen=True)
class OwnershipLeak:
    contract: str
    variable: str
    owners: tuple[str, ...]
    formula: str


def _collect_atoms(node) -> set[str]:
    from contrauto.feedback.ltl_parser import Atom, Not, And, Or, Implies, Iff, G, F, X, U_
    if isinstance(node, Atom):
        return {node.name}
    if isinstance(node, (Not, G, F, X)):
        return _collect_atoms(node.child)
    if isinstance(node, (And, Or, Implies, Iff, U_)):
        return _collect_atoms(node.left) | _collect_atoms(node.right)
    return set()


def edge_signal_owners(task) -> dict[str, frozenset[str]]:
    """Map each point-to-point edge-signal variable to its two owning node names.

    Mirrors ``TaskSpec._expand_edge_interfaces``'s naming so the derived variable
    matches what actually ends up in each node's declared inputs/outputs. A
    "world"-direction signal is environment-owned and observable by everyone, so
    it is not private and is excluded here.
    """
    from contrauto.schema import _node_prefix

    owners: dict[str, frozenset[str]] = {}
    for edge in getattr(task, "edges", []) or []:
        if not edge.signals:
            continue
        fid = _node_prefix(edge.from_node).rstrip("_")
        for signal in edge.signals:
            if signal.get("dir") == "world":
                continue
            base = signal.get("var") or signal.get("base") or signal.get("name")
            if not base:
                continue
            var = signal["var"] if signal.get("var") else f"{base}_{fid}"
            owners[var] = frozenset({edge.from_node, edge.to_node})
    return owners


def ownership_leaks(content: str, task) -> list[OwnershipLeak]:
    """Find contracts that reference a private edge signal they do not own."""
    owners = edge_signal_owners(task)
    if not owners:
        return []
    node_names = {n.name for n in getattr(task, "nodes", [])}
    leaks: list[OwnershipLeak] = []
    for labeled in extract_formulas(content):
        if labeled.contract not in node_names or labeled.formula is None:
            continue
        for var in _collect_atoms(labeled.formula):
            allowed = owners.get(var)
            if allowed is not None and labeled.contract not in allowed:
                leaks.append(OwnershipLeak(labeled.contract, var, tuple(sorted(allowed)),
                                           labeled.raw_text))
    return leaks


def format_ownership_feedback(leaks: list[OwnershipLeak], content: str) -> str:
    details = "\n".join(
        f"- {leak.contract}: `{leak.formula}` references `{leak.variable}`, a PRIVATE "
        f"point-to-point signal owned only by {{{', '.join(leak.owners)}}}."
        for leak in leaks
    )
    return (
        "[CROSS-CONTRACT OWNERSHIP LEAK]\n"
        "A sub-contract references ANOTHER node's private edge signal instead of its own "
        "(e.g. Robot4 wrote `grant_r1` -- Robot1's own grant -- instead of its own "
        "`grant_r4`). A req/grant signal declared on an edge belongs EXCLUSIVELY to that "
        "edge's two endpoints; every other contract must use ITS OWN copy of that signal, "
        "with ITS OWN index. (A plain state output like `chargingN`/`battery_lowN` is NOT "
        "an edge signal -- siblings MAY read another robot's `chargingN` for mutual "
        "exclusion; only req_rN/grant_rN-style point-to-point signals are private.)\n"
        f"{details}\n\nCurrent .ltl:\n{content}\n\nReturn raw .ltl text only.\n"
    )
