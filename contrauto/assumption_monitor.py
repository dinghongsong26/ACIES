"""Run-time assumption-conformance monitors: the 5th feedback channel.

The certificate ladder (consistency / refinement / realizability / non-vacuity) certifies
each contract at DESIGN time. A synthesized strategy only holds while the ENVIRONMENT keeps
the contract's ASSUMPTIONS. This module checks that at RUN time: each contract's assumptions
are compiled to small monitors that watch the simulator's observation stream, and a violation
becomes a localized witness + structured feedback for the decomposer -- the same repair loop
the design-time certificates already use.

Monitorability (Bauer-Leucker-Schallhart, LTL3): a SAFETY assumption gives a sound 2-valued
reject; a LIVENESS assumption (`G F ...`) can only be flagged SUSPECTED after a patience bound
(a finite trace can never disprove an eventuality). Assumptions must be X-FREE
(stutter-invariant, Peled-Wilke) to be monitorable against a durative, tick-sampled trace --
the next-operator lives only in the synthesis-internal GR(1) encoding, never here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ── a tiny boolean evaluator over an observation dict {ap: 0/1} ───────────────
# Grammar (lowest to highest precedence): <-> , -> , | , & , ! , atom.
# atom: ( expr ) | ident | true | false. Idents are AP names read from the obs dict
# (missing AP == 0). This is intentionally NOT a full LTL parser: temporal structure is
# classified away first (see build_monitor); only the propositional body reaches here.
_TOKEN = re.compile(r"\s*(<->|->|[()!&|]|[A-Za-z_][A-Za-z0-9_]*)")


def _tokenize(s: str) -> List[str]:
    toks, i = [], 0
    while i < len(s):
        m = _TOKEN.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise ValueError(f"bad token in boolean expr near: {s[i:]!r}")
        toks.append(m.group(1))
        i = m.end()
    return toks


def compile_bool(expr: str) -> Callable[[Dict[str, int]], bool]:
    """Compile a propositional formula into `env -> bool`. Raises ValueError on syntax."""
    toks = _tokenize(expr)
    pos = [0]

    def peek() -> Optional[str]:
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat(t: str) -> None:
        if peek() != t:
            raise ValueError(f"expected {t!r} in {expr!r}, got {peek()!r}")
        pos[0] += 1

    def p_iff():
        f = p_imp()
        while peek() == "<->":
            eat("<->"); g = p_imp(); a, b = f, g
            f = (lambda a, b: (lambda e: a(e) == b(e)))(a, b)
        return f

    def p_imp():
        f = p_or()
        if peek() == "->":
            eat("->"); g = p_imp(); a, b = f, g       # right-assoc
            return (lambda a, b: (lambda e: (not a(e)) or b(e)))(a, b)
        return f

    def p_or():
        f = p_and()
        while peek() == "|":
            eat("|"); g = p_and(); a, b = f, g
            f = (lambda a, b: (lambda e: a(e) or b(e)))(a, b)
        return f

    def p_and():
        f = p_not()
        while peek() == "&":
            eat("&"); g = p_not(); a, b = f, g
            f = (lambda a, b: (lambda e: a(e) and b(e)))(a, b)
        return f

    def p_not():
        if peek() == "!":
            eat("!"); g = p_not()
            return (lambda g: (lambda e: not g(e)))(g)
        return p_atom()

    def p_atom():
        t = peek()
        if t == "(":
            eat("("); f = p_iff(); eat(")")
            return f
        if t in ("true", "false"):
            pos[0] += 1
            v = t == "true"
            return lambda e, v=v: v
        if t is None or t in (")", "&", "|", "!", "->", "<->"):
            raise ValueError(f"unexpected {t!r} in {expr!r}")
        pos[0] += 1                                    # identifier
        return lambda e, name=t: bool(e.get(name, 0))

    f = p_iff()
    if pos[0] != len(toks):
        raise ValueError(f"trailing tokens in {expr!r}: {toks[pos[0]:]}")
    return f


# ── witness ───────────────────────────────────────────────────────────────
@dataclass
class Violation:
    """One assumption-conformance witness, scoped to a single contract."""
    contract: str
    formula: str                       # the assumption LTL, as written in the contract
    kind: str                          # "safety" | "liveness" | "persistence"
    suspected: bool                    # liveness can only be SUSPECTED (patience bound)
    step: int
    evidence: str                      # human-readable trace evidence
    observed: Dict[str, object] = field(default_factory=dict)


# ── per-assumption monitors ─────────────────────────────────────────────────
class _SafetyMon:
    """`G(body)` -- reject the first step where `body` is false. Sound, 2-valued."""
    kind = "safety"

    def __init__(self, formula: str, body: str):
        self.formula, self._body_src = formula, body
        self._body = compile_bool(body)

    def update(self, obs: Dict[str, int], step: int) -> Optional[Violation]:
        if not self._body(obs):
            return Violation(contract="", formula=self.formula, kind=self.kind,
                             suspected=False, step=step,
                             evidence=f"at step {step} the safety body `{self._body_src}` "
                                      f"was false under obs { {k: obs.get(k, 0) for k in _vars(self._body_src)} }")
        return None


class _LivenessMon:
    """`G F(body)` -- SUSPECTED violation once `body` has been false for `patience`
    consecutive steps. A finite trace can never disprove an eventuality, so this is a
    bounded-patience heuristic, never a proof."""
    kind = "liveness"

    def __init__(self, formula: str, body: str, patience: int = 200):
        self.formula, self._body_src, self.patience = formula, body, patience
        self._body = compile_bool(body)
        self._false_run = 0
        self._fired = False

    def update(self, obs: Dict[str, int], step: int) -> Optional[Violation]:
        if self._body(obs):
            self._false_run = 0
            return None
        self._false_run += 1
        if self._false_run > self.patience and not self._fired:
            self._fired = True
            vs = {k: obs.get(k, 0) for k in _vars(self._body_src)}
            return Violation(contract="", formula=self.formula, kind=self.kind,
                             suspected=True, step=step,
                             evidence=f"`{self._body_src}` stayed false for {self._false_run} "
                                      f"consecutive steps (> patience {self.patience}); "
                                      f"the environment never discharged this eventuality",
                             observed=vs)
        return None


class _PersistMon:
    """`p W q` (weak-until) -- SAFETY. Once the obligation is active (q not yet seen), a step
    with `!p & !q` is a sound violation; `q` discharges it forever."""
    kind = "persistence"

    def __init__(self, formula: str, p: str, q: str):
        self.formula, self._p_src, self._q_src = formula, p, q
        self._p, self._q = compile_bool(p), compile_bool(q)
        self._discharged = False

    def update(self, obs: Dict[str, int], step: int) -> Optional[Violation]:
        if self._discharged or self._q(obs):
            self._discharged = True
            return None
        if not self._p(obs):                           # p dropped before q -> W violated
            return Violation(contract="", formula=self.formula, kind=self.kind,
                             suspected=False, step=step,
                             evidence=f"at step {step} `{self._p_src}` dropped before "
                                      f"`{self._q_src}` held (weak-until `{self.formula}` broken)")
        return None


def _vars(expr: str) -> List[str]:
    return sorted({t for t in _tokenize(expr)
                   if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t) and t not in ("true", "false")})


# ── classify an assumption LTL string into a monitor (or None = design-time only) ──
_GF = re.compile(r"^\s*G\s*F\s*\((?P<b>.*)\)\s*$", re.S)
_G = re.compile(r"^\s*G\s*\((?P<b>.*)\)\s*$", re.S)
_W = re.compile(r"^\s*(?P<p>.+?)\s+W\s+(?P<q>.+?)\s*$", re.S)


def build_monitor(formula: str):
    """Return a per-assumption monitor, or None if the assumption is NOT run-time monitorable
    (an initial condition like `!p`, or a formula still carrying the X operator -- X is not
    monitorable against a durative trace and must not appear in an assumption)."""
    f = formula.strip().rstrip(";").strip()
    if not f or f.lower() == "true":
        return None
    if "X(" in f or re.search(r"\bX\b", f):
        return None                                    # X in an assumption is not monitorable
    m = _GF.match(f)
    if m and _balanced(m.group("b")):
        return _LivenessMon(f, m.group("b").strip())
    m = _W.match(f)
    if m and _balanced(m.group("p")) and _balanced(m.group("q")) and " U " not in f:
        return _PersistMon(f, m.group("p").strip(), m.group("q").strip())
    m = _G.match(f)
    if m and _balanced(m.group("b")):
        try:
            return _SafetyMon(f, m.group("b").strip())
        except Exception:
            # Body outside the propositional fragment (e.g. a mission `(p U q)` folded into
            # the assumptions by anchoring). One such formula used to raise through
            # ContractMonitor and — via the registry's blanket except — silently kill the
            # ENTIRE monitor set, so runtime starvation went unreported. Skip just it.
            return None
    if not re.search(r"\b[GFU]\b|W", f):               # bare propositional -> initial condition
        return None
    return None                                        # unrecognized temporal shape -> skip


def _balanced(s: str) -> bool:
    d = 0
    for c in s:
        d += (c == "(") - (c == ")")
        if d < 0:
            return False
    return d == 0


# ── per-contract bundle + the full set ───────────────────────────────────────
class ContractMonitor:
    def __init__(self, contract: str, assumptions: List[str]):
        self.contract = contract
        self.monitors = [mm for a in assumptions if (mm := build_monitor(a)) is not None]
        self.skipped = [a for a in assumptions if build_monitor(a) is None]

    def update(self, obs: Dict[str, int], step: int) -> List[Violation]:
        out = []
        for mon in self.monitors:
            v = mon.update(obs, step)
            if v is not None:
                v.contract = self.contract
                out.append(v)
        return out


class AssumptionMonitorSet:
    """Watch every contract's assumptions against the running simulator.

    `by_contract`: {contract_name: [assumption LTL strings]} (see parse_chase_assumptions).
    Call `update(obs, step)` each simulation step with the observed AP dict; collect
    `violations` and render `feedback()` for the decomposer.
    """

    def __init__(self, by_contract: Dict[str, List[str]], patience: int = 200):
        self.contracts = {c: ContractMonitor(c, a) for c, a in by_contract.items()}
        self.violations: List[Violation] = []
        for cm in self.contracts.values():
            for mon in cm.monitors:
                if isinstance(mon, _LivenessMon):
                    mon.patience = patience

    def update(self, obs: Dict[str, int], step: int) -> List[Violation]:
        new = []
        for cm in self.contracts.values():
            new += cm.update(obs, step)
        self.violations += new
        return new

    def any_violation(self) -> bool:
        return bool(self.violations)

    def feedback(self) -> str:
        """Structured NL for the decomposer's repair loop -- channel 5, post-synthesis."""
        return format_feedback(self.violations)


# ── decomposer feedback (channel 5) ─────────────────────────────────────────
def format_feedback(violations: List[Violation]) -> str:
    if not violations:
        return ""
    lines = ["[RUNTIME ASSUMPTION-CONFORMANCE VIOLATION]  (channel 5, post-synthesis)",
             "A synthesized strategy only holds while the environment keeps the contract's",
             "assumptions. Physical execution showed some assumption is NOT dischargeable here:"]
    for v in violations:
        tag = "SUSPECTED" if v.suspected else "confirmed"
        lines += [
            "",
            f"Contract: {v.contract}",
            f"Assumption: {v.formula}    kind: {v.kind} ({tag})",
            f"Observed: {v.evidence}",
        ]
    lines += [
        "",
        "Legal repairs (pick what fits the witness; never hand-code the scenario):",
        "  1. If the assumption demanded tighter timing than the durative world can meet,",
        "     reformulate it in eventual, X-free form:  G F(effect | !action).",
        "  2. If the strategy relied on the effect before it was available, add the missing",
        "     ordering guarantee:  (!next_action) U effect.",
        "  3. If the effect is physically unreachable in this environment, this is a TASK-level",
        "     infeasibility, not a spec repair -- report it rather than reshaping the contract.",
        "  4. Otherwise weaken the guarantee that depended on the violated assumption.",
        "Each witness is scoped to one contract's assumptions; fix that contract only.",
    ]
    return "\n".join(lines)


# ── extract Assumptions blocks from CHASE .ltl text ─────────────────────────
def parse_chase_assumptions(ltl_text: str) -> Dict[str, List[str]]:
    """Map {contract_name: [assumption formulas]} from a CHASE .ltl file. A contract block is
    `CONTRACT Name:` ... `Assumptions:` <formulas ;> ... up to `Guarantees:` / next CONTRACT."""
    out: Dict[str, List[str]] = {}
    cur = None
    in_asm = False
    for raw in ltl_text.splitlines():
        line = raw.strip()
        mc = re.match(r"CONTRACT\s+(\w+)\s*:?", line)
        if mc:
            cur, in_asm = mc.group(1), False
            out.setdefault(cur, [])
            continue
        if re.match(r"Assumptions\s*:", line):
            in_asm = True
            continue
        if re.match(r"(Guarantees|input|output)\b", line) or re.match(r"CONTRACT\b", line):
            in_asm = in_asm and False
            continue
        if in_asm and cur is not None and line and not line.startswith(("input", "output")):
            for f in line.split(";"):
                f = f.strip()
                if f:
                    out[cur].append(f)
    return out


def monitor_from_ltl(ltl_text: str, patience: int = 1200):
    """Build an AssumptionMonitorSet watching each contract's X-FREE temporal assumptions.

    Only stutter-invariant assumptions (no next-step X) are monitored: the durative execution
    stutters relative to the discrete step model, so an X-formula cannot be soundly checked at run
    time (see the X-free / stutter-invariance rationale). Initial-value literals (`!at_shelf`) and
    `X(...)` formulas are skipped. Returns None if nothing is monitorable.

    `patience` is the bounded-patience horizon for a LIVENESS (`G F body`) SUSPECTED violation. It
    must exceed the DURATIVE cost of one action (a navigation is `at_shelf=0 & nav_shelf=1` for many
    continuous ticks, during which `G F(at_shelf | !nav_shelf)`'s body is legitimately false) so a
    normally-progressing robot is NOT flagged; only a robot that stays body-false FOREVER (e.g. a
    starved requester whose grant never comes) trips it. 1200 clears the 2D sims' per-action nav time
    (~hundreds of ticks over 3000--8000-step runs) while still catching genuine forever-stuck."""
    raw = parse_chase_assumptions(ltl_text)
    by_contract: Dict[str, List[str]] = {}
    for c, formulas in raw.items():
        keep = []
        for f in formulas:
            if "X(" in f or "X (" in f:                 # not stutter-invariant -> do not monitor
                continue
            if not re.search(r"\b[GFUW]\b", f):          # skip bare initial-value literals / true
                continue
            keep.append(f)
        if keep:
            by_contract[c] = keep
    return AssumptionMonitorSet(by_contract, patience=patience) if by_contract else None
