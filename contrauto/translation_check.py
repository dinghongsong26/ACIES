"""Mechanical checks and completion for the Translator's output (no LLM calls).

The translator LLM reads the natural language; this module makes its output complete and
well-formed without trusting it to be exhaustive:

  coverage()  -- every numbered requirement is translated by at least one formula
                 (each pair carries ``"req": [k, ...]``);
  lint()      -- variable names bind to the simulator vocabulary, no known-bad shapes;
  complete()  -- the formula families that follow from the simulator's durative-action
                 physics (vocab ``action_effects`` / ``frame``) are added per robot when
                 missing, and the two known-bad shapes are rewritten to their correct form.

Only what lint() reports and complete() cannot fix goes back to the LLM for a repair.
"""

from __future__ import annotations

import re

_TEMPORAL = {"G", "F", "X", "U", "W", "R", "true", "false", "TRUE", "FALSE"}


def _norm(f: str) -> str:
    return re.sub(r"\s+", "", f)


def _lit(prefix: str, literal: str) -> str:
    """`!has_item` -> `!r1_has_item`."""
    neg = literal.startswith("!")
    return ("!" if neg else "") + prefix + literal.lstrip("!")


def _neg(literal: str) -> str:
    return literal[1:] if literal.startswith("!") else "!" + literal


def coverage(pairs: list[dict], n_requirements: int,
             not_applicable: list[dict] | None = None) -> list[int]:
    """1-based requirement numbers that no pair claims via its ``req`` field and that are
    not declared vacuous for this instance (``not_applicable``)."""
    covered = set()
    for na in not_applicable or []:
        try:
            covered.add(int(na.get("req")))
        except (TypeError, ValueError, AttributeError):
            pass
    for p in pairs:
        reqs = p.get("req", [])
        for k in (reqs if isinstance(reqs, list) else [reqs]):
            try:
                covered.add(int(k))
            except (TypeError, ValueError):
                pass
    return [k for k in range(1, n_requirements + 1) if k not in covered]


def _templates(vocab: dict, prefix: str) -> list[tuple[str, str, str]]:
    """(presence key, formula, nl) for one robot's physics-derived formula families."""
    out = []
    for action, (effect, grant) in (vocab.get("action_effects") or {}).items():
        a, e = prefix + action, _lit(prefix, effect)
        out.append((f"GF({e}|!{a})", f"G F({e} | !{a})",
                    f"world: {action} eventually produces {effect}"))
        guard = f" & X({prefix}{grant})" if grant else ""
        out.append((f"->X({a}))", f"G(({a} & X({_neg(e)}){guard}) -> X({a}))",
                    f"{action} is held until {effect} appears"
                    + (f" while {grant} is held" if grant else "")))
    for state, action in (vocab.get("frame") or {}).items():
        s, a = _lit(prefix, state), prefix + action
        out.append((f"->X({s}))", f"G(({s} & !{a}) -> X({s}))",
                    f"world: {state} persists until {action}"))
    return out


def _rewrites(vocab: dict, prefix: str) -> dict[str, str]:
    """Known-bad normalized formula -> its correct form (see the translator prompt)."""
    fixes = {}
    requests = (vocab.get("signals") or {}).get("requests", [])
    for action, (effect, grant) in (vocab.get("action_effects") or {}).items():
        a, e = prefix + action, _lit(prefix, effect)
        g = f"&X({prefix}{grant})" if grant else ""
        guard = f" & X({prefix}{grant})" if grant else ""
        # persistence guarded on the CURRENT effect value (unrealizable with the action's guard)
        fixes[f"G(({a}&{_neg(e)}{g})->X({a}))"] = f"G(({a} & X({_neg(e)}){guard}) -> X({a}))"
        fixes[f"G({a}&{_neg(e)}{g}->X({a}))"] = f"G(({a} & X({_neg(e)}){guard}) -> X({a}))"
        # releasing a request on the action alone instead of on its completed effect
        for req in requests:
            fixes[f"G({a}->X(!{prefix}{req}))"] = f"G(({a} & X({e})) -> X(!{prefix}{req}))"
    return fixes


def complete(pairs: list[dict], vocab: dict | None, prefixes: list[str]):
    """Return (pairs, notes): globals un-prefixed, known-bad shapes rewritten, and missing
    physics-derived formulas appended (tagged ``"source": "grounding"``)."""
    if not vocab:
        return pairs, []
    notes = []
    globals_ = vocab.get("globals") or []
    rewrites = {}
    for prefix in prefixes:
        rewrites.update(_rewrites(vocab, prefix))
    out = []
    for p in pairs:
        p = dict(p)
        ltl = p["ltl"]
        for g in globals_:
            ltl = re.sub(rf"\br\d+_{g}\b", g, ltl)
        fixed = rewrites.get(_norm(ltl))
        if fixed:
            notes.append(f"rewrote {ltl}  →  {fixed}")
            ltl = fixed
        elif ltl != p["ltl"]:
            notes.append(f"un-prefixed shared signal: {p['ltl']}  →  {ltl}")
        p["ltl"] = ltl
        out.append(p)
    present = "\n".join(_norm(p["ltl"]) for p in out)
    for prefix in prefixes:
        for key, formula, nl in _templates(vocab, prefix):
            if key in present:
                continue
            out.append({"nl": f"{prefix.rstrip('_').upper()}: {nl}", "ltl": formula,
                        "req": [], "source": "grounding"})
            present += "\n" + _norm(formula)
            notes.append(f"added {formula}")
    return out, notes


def lint(pairs: list[dict], vocab: dict | None, prefixes: list[str],
         extra_known=()) -> list[str]:
    """Issues the LLM must fix: unknown variables and non-GR(1) shapes (one line each).

    A variable is known if it is in the simulator vocabulary (optionally r<i>_-prefixed), in
    `extra_known` (e.g. the task's edge-generated signals such as ``req_r1``), or binds by role
    (sim_iface.role_of) -- the same acceptance rule as the decomposer's grounding check.
    """
    from contrauto.sim_iface import role_of

    issues = []
    known = set(extra_known)
    if vocab:
        known = set(vocab.get("sensors", [])) | set(vocab.get("actions", []))
        for names in (vocab.get("signals") or {}).values():
            known |= set(names)
    prefix_re = re.compile(r"^(?:" + "|".join(map(re.escape, prefixes)) + r")") if prefixes else None
    for i, p in enumerate(pairs, 1):
        ltl = p["ltl"]
        if known:
            for tok in set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", ltl)) - _TEMPORAL:
                base = prefix_re.sub("", tok) if prefix_re else tok
                if base not in known and tok not in known and role_of(tok) is None:
                    issues.append(f"[{i}] {ltl}: unknown variable `{tok}`")
        if re.search(r"\bX\s*\(\s*G\b", ltl):
            issues.append(f"[{i}] {ltl}: `X(G(...))` is not GR(1) -- write `G(...)` (the initial "
                          "values already make it hold at step 0)")
        m = re.fullmatch(r"G\(\(?(!?\w+)\)?->\(?(!?\w+)\)?\)", _norm(ltl))
        if m and m.group(1) == m.group(2):
            issues.append(f"[{i}] {ltl}: tautology -- if the requirement is vacuous here, "
                          "declare it under not_applicable instead")
        if re.search(r"\bG\s*\(.*\bF\s*\(", ltl) and not re.match(r"\s*G\s*F\s*\(", ltl):
            issues.append(f"[{i}] {ltl}: `F` nested under `G` is not GR(1) -- use a justice "
                          "`G F(...)` instead")
    return issues


# ── Translation-level realizability (one GR(1) game for the whole translation) ──────────────
# The translation is the System specification: the environment owns the simulator's sensors,
# the system owns every action, request and grant. If that game is already unrealizable, no
# decomposition can succeed, so the contradiction is localized here (seconds, no LLM) instead
# of surfacing at Decomposer Step 12 several LLM rounds later.

def _vars(f: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", f)) - _TEMPORAL


def _is_sensor(var: str, vocab: dict, prefixes: list[str]) -> bool:
    sensors = set(vocab.get("sensors", []))
    base = re.sub(r"^(?:" + "|".join(map(re.escape, prefixes)) + r")", "", var) if prefixes else var
    return base in sensors


def classify(ltl: str, vocab: dict, prefixes: list[str]) -> list[tuple[str, str]]:
    """[(section, formula)] -- 'assumption' for world behaviour over sensors, else
    'guarantee'. A temporal-free initial conjunction is split literal by literal."""
    f = ltl.strip()
    sensor = lambda v: _is_sensor(v, vocab, prefixes)
    if not re.search(r"\b[GFXU]\s*\(", f):                               # initial values
        body = f.strip("()").strip()
        if "|" in body:
            return [("guarantee", f)]
        lits = [x.strip().strip("()").strip() for x in body.split("&") if x.strip()]
        return [("assumption" if sensor(lit.lstrip("!")) else "guarantee", lit) for lit in lits]
    if re.match(r"G\s*F\s*\(", f):                                      # justice
        return [("assumption" if any(sensor(v) for v in _vars(f)) else "guarantee", f)]
    m = re.search(r"->\s*X\s*\((.*)\)\s*\)\s*$", f)                    # transition
    if m:
        return [("assumption" if all(sensor(v) for v in _vars(m.group(1))) else "guarantee", f)]
    return [("assumption" if all(sensor(v) for v in _vars(f)) else "guarantee", f)]


def realizability_spec(pairs: list[dict], vocab: dict, prefixes: list[str],
                       name: str = "Spec") -> tuple[str, list[tuple[int, str]]]:
    """CHASE .ltl text of the whole translation as one contract, plus [(pair index, guarantee)]
    so an unrealizable core can be mapped back to the translator's formulas."""
    asms, gars, owners = [], [], []
    for i, p in enumerate(pairs):
        for section, f in classify(p["ltl"], vocab, prefixes):
            (asms if section == "assumption" else gars).append(f)
            if section == "guarantee":
                owners.append((i, f))
    used = set().union(*(_vars(p["ltl"]) for p in pairs)) if pairs else set()
    inputs = sorted(v for v in used if _is_sensor(v, vocab, prefixes))
    outputs = sorted(used - set(inputs))
    pinned = {re.sub(r"^!", "", f) for f in asms if not re.search(r"\b[GFXU]\s*\(", f)}
    asms = [f"!{v}" for v in inputs if v not in pinned] + asms   # robots start with all sensors low
    lines = [f"NAME: translation_check;", "", f"CONTRACT {name}:"]
    lines += [f"    input boolean variable {v};" for v in inputs]
    lines += [f"    output boolean variable {v};" for v in outputs]
    lines += ["    Assumptions:"] + [f"        {f};" for f in asms]
    lines += ["    Guarantees:"] + [f"        {f};" for f in gars]
    return "\n".join(lines) + "\n", owners


def check_realizable(tools, pairs: list[dict], vocab: dict, prefixes: list[str]):
    """(verdict, core_pair_indices): verdict True/False/None (None = tools unavailable or no
    verdict). On False, the indices of the translator pairs forming the minimal unrealizable
    core ([] when the cause lies in the assumptions)."""
    from contrauto.feedback.gr1_core import compute_core

    spec, owners = realizability_spec(pairs, vocab, prefixes)
    verdict = tools.check_realizable("Spec", spec)
    if verdict is not False:
        return verdict, []
    core = compute_core(tools, spec, "Spec") or []
    core_norm = {_norm(c) for c in core}
    return False, sorted({i for i, f in owners if _norm(f) in core_norm})
