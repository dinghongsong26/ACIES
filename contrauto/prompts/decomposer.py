"""System prompt for the Decomposer agent: few-shot, syntax rules, and CHASE contract-design instructions."""
CHASE_FEW_SHOT = """You MUST design one System contract PLUS one or more sub-contracts.

══════════════════════════════════════════════════════════════
EXAMPLE — shared resource with an Arbiter (ABSTRACT, generic names, GR(1)-correct).
Two workers each need EXCLUSIVE access to one shared resource; an Arbiter coordinates.
KEY: "a request is eventually granted" is a JUSTICE goal  G F(grant | !request) ,
NOT a response  G(request -> F(grant))  (the GR(1) backend flattens response into a
same-step safety and the spec degenerates). This shape is verified realizable.
══════════════════════════════════════════════════════════════
NAME: Shared_Resource;

CONTRACT Worker1:
    input boolean variable w1_grant;
    output boolean variable w1_request;
    output boolean variable w1_use;
    Assumptions:
        !w1_grant;
        G F(w1_grant | !w1_request);
    Guarantees:
        !w1_request;
        !w1_use;
        G(w1_use -> w1_grant);
        G(w1_request & !w1_grant -> X(w1_request));
      G F(!w1_request | !w1_grant);
        G F(w1_use);

CONTRACT Worker2:
    input boolean variable w2_grant;
    output boolean variable w2_request;
    output boolean variable w2_use;
    Assumptions:
        !w2_grant;
        G F(w2_grant | !w2_request);
    Guarantees:
        !w2_request;
        !w2_use;
        G(w2_use -> w2_grant);
        G(w2_request & !w2_grant -> X(w2_request));
      G F(!w2_request | !w2_grant);
        G F(w2_use);

CONTRACT Arbiter:
    input boolean variable w1_request;
    input boolean variable w2_request;
    output boolean variable w1_grant;
    output boolean variable w2_grant;
    Assumptions:
        !w1_request;
        !w2_request;
      G F(!w1_request | !w1_grant);
      G F(!w2_request | !w2_grant);
    Guarantees:
        !w1_grant;
        !w2_grant;
        G(!(w1_grant & w2_grant));
        G(w1_grant -> w1_request);
        G(w2_grant -> w2_request);
      G((w1_grant & w1_request & X(w1_request)) -> X(w1_grant));
      G((w2_grant & w2_request & X(w2_request)) -> X(w2_grant));
        G F(w1_grant | !w1_request);
        G F(w2_grant | !w2_request);

CONTRACT System:
    output boolean variable w1_request;
    output boolean variable w1_use;
    output boolean variable w2_request;
    output boolean variable w2_use;
    output boolean variable w1_grant;
    output boolean variable w2_grant;
    Guarantees:
        !w1_use;
        !w2_use;
        G(!(w1_grant & w2_grant));
        G F(w1_use);
        G F(w2_use);

STRUCTURAL LESSONS (general):
1. One sub-contract per agent; add an Arbiter ONLY when a resource is genuinely shared.
2. A sub-contract's OUTPUTS include its request(s); its INPUTS include the grant(s) it gets.
3. The Arbiter has the requests as INPUTS and the grants as OUTPUTS; it guarantees mutual
   exclusion  G(!(grant_i & grant_j)) , grants only on request  G(grant_i -> request_i) ,
   FAIRNESS as a JUSTICE goal  G F(grant_i | !request_i) , and, for a durative entry/service,
   grant persistence while the request continues:
   G((grant_i & request_i & X(request_i)) -> X(grant_i)) . This NEXT-request guard is
   essential: it permits release but forbids a one-tick grant pulse while a worker navigates.
4. A worker ASSUMES the arbiter fairness  G F(grant | !request)  and PERSISTS its request
   until granted  G(request & !grant -> X(request)) .
4b. A revocable durable lease needs one bounded-release assumption on the Arbiter:
   `G F(!request | !grant)`. Its matching Worker/System guarantee says a granted request
   is eventually withdrawn after service. This makes grant persistence compatible with
   mutual exclusion and fairness. Do not use the stronger `G F(!request)`, which constrains
   an ungranted requester; otherwise an Arbiter has no worker assumptions (`true;` or
   initial pins only). The matching release formula must be discharged through System.
5. System INPUTS = pure environment state only; System OUTPUTS = all controllable variables;
   System Guarantees = end-to-end only (e.g. mutual exclusion + each objective  G F(...) ).
6. NEVER use response  G(p -> F(q)) , bare F, or nested F in any Guarantees (see GR(1)
   rules below). Environment Assumptions MAY use them.

DECOMPOSITION RULES:
1. Always produce sub-contract(s) + exactly one CONTRACT System.
2. Single agent/role  -> one sub-contract.
3. Multiple agents    -> one sub-contract per agent + an Arbiter if a resource is shared.
4. System contract: end-to-end properties only, no per-agent detail.
5. synthesis.chase: ONLY sub-contracts, never System.
6. verification.chase compose order: sub-contracts first, then System.
7. Replace <mission> in filenames with the actual mission name.
"""

CHASE_SYNTAX_RULES = """
CHASE .ltl FILE STRICT SYNTAX RULES -- violations cause parse errors:

1. FILE HEADER:   NAME: SomeName;          <- ends with semicolon
2. CONTRACT BLOCK:
   CONTRACT Name:                          <- colon, NO semicolon after the name
       input boolean variable p;
       output boolean variable a;
       Assumptions:                        <- colon, NO semicolon
           !p;                             <- initial condition ends with ;
           G F(p);                         <- LTL formula ends with ;
       Guarantees:                         <- colon, NO semicolon
           !a;
           G F(a);
3. FORBIDDEN inside CONTRACT blocks: saturate / compose / refinement / synthesize commands.
   Those belong ONLY in .chase files, never in the .ltl.
4. LTL OPERATORS: G(phi) F(phi) X(phi)  phi U psi  G F(phi)  G(p -> q)  G(!(p & q))
   - always wrap implication: G(p -> X(q));   - G F needs parens: G F(a);
5. COMMON MISTAKES: missing colon after CONTRACT/Assumptions/Guarantees; `G F a` (needs
   `G F(a)`); `G(p -> F q)` (needs inner parens `G(p -> F(q))`).
6. NEVER leave an Assumptions: or Guarantees: block empty -- the parser fails with
   "mismatched input 'Guarantees'". A component with no real environment assumptions (e.g. a
   pure coordinator/arbiter whose only inputs are requests) MUST still write at least one
   line under Assumptions:, namely `true;`.

COMPLETE VALID EXAMPLE (abstract, GR(1)-correct: justice + safety, no response in Guarantees):
NAME: Example;

CONTRACT Agent:
    input boolean variable trigger;
    input boolean variable arrived;
    output boolean variable act;
    output boolean variable goal;
    Assumptions:
        !trigger;
        !arrived;
        G F(arrived | !act);
        G(arrived & !act -> X(arrived));
        G F(trigger);
    Guarantees:
        !act;
        !goal;
        G(act & !arrived -> X(act));
        G(goal <-> arrived);
        G F(goal);

CONTRACT System:
    input boolean variable trigger;
    input boolean variable arrived;
    output boolean variable act;
    output boolean variable goal;
    Assumptions:
        !trigger;
        !arrived;
        G F(arrived | !act);
        G F(trigger);
    Guarantees:
        !act;
        !goal;
        G(goal <-> arrived);
        G F(goal);
"""

CONSTRAINT_RULES = """
══════════════════════════════════════════════════════════════
CONSTRAINT CATEGORIES (task-independent; backend is GR(1), which does NOT saturate). The natural
language gives the OBJECTIVE; the CONSTRAINTS come from the ACTION MODEL (each action's PRECONDITION
and EFFECT) and the ARCHITECTURE (shared resources / arbiter). These are recurrent specification
patterns (Dwyer et al.; robotic-mission patterns, Menghi et al.). Emit each category with the task's
own predicate/action names, and self-check against the action model at the end.
══════════════════════════════════════════════════════════════

1. OBJECTIVE (liveness) = pure justice `G F(goal)`, normally ONE per agent. NEVER put a response
   `G(p -> F(q))`, a bare `F`, or a nested `F` in Guarantees — the backend flattens response to the
   over-strong same-step safety `G(p -> q)` and the spec degenerates. Do NOT add a justice per
   action; ONE objective is enough, the synthesizer derives the steps. "request eventually granted"
   is the justice `G F(served | !pending)`. (The ONLY `U` allowed in Guarantees is the causal
   ordering in category 4.)

2. PRECONDITION GUARD — for EVERY action A with precondition P:  `G(A -> P)`
   (pickup needs the shelf: `G(pickup -> at_shelf)`; dropoff: `G(dropoff -> (at_dock & has_item))`).

3. ACTION MUTEX — for EVERY pair of actions that cannot physically co-occur (two motions, or a
   motion and a manipulation):  `G(!(A & B))`  — ALL pairs, none dropped, a single dropped pair
   degenerates the spec. (Shared resource across N agents: the arbiter needs all O(N^2) grant
   exclusions `G(!(grant_i & grant_j))`, every i<j.)
   THIS ALSO APPLIES INSIDE EACH WORKER'S OWN CONTRACT for a shared-resource STATE it exposes
   (e.g. `charging`): worker K must mutex against EVERY OTHER worker J != K, not just worker 1 —
   `G(!(charging_k & charging_j))` for ALL J != K, one clause per sibling. Do NOT single out worker
   1 as a stand-in "representative" for the whole group and skip the rest: for N=5, worker 4 needs
   3 such clauses (vs. 1, 2, 3, 5 — whichever indices are not its own), not one. Every one of these
   clauses, and every guarantee that gates worker K's own action on ITS OWN grant/req (e.g.
   `G(charging_k -> grant_k)`), MUST use worker K's OWN index on every occurrence — never copy
   worker 1's `grant_1`/`req_1` into another worker's contract by habit. A point-to-point signal
   like `grant_k`/`req_k` belongs EXCLUSIVELY to worker k and the arbiter; no sibling contract may
   reference it.

4. CAUSAL PRECEDENCE — a causal "resource R must hold before action A" is ALREADY enforced by A's
   PRECONDITION GUARD `G(A -> R)` (cat 2): the guard forbids A whenever R is false, and the liveness
   objective `G F(goal)` forces the strategy to obtain R first. Do NOT emit any separate precedence /
   ordering clause (no `<fact> before <action>` block, no `(!action) U resource`). For a SHARED
   resource the precondition is the grant: guard every action that ENTERS, OCCUPIES, or USES the
   capacity-limited resource with `G(action -> grant)`. Distinguish a service point from an entry
   zone: a shelf pickup needs `G(pickup -> shelf_grant)`, but navigation toward an individually
   reserved shelf bin need not require a grant. If `go_to_dock` enters the single shared dock,
   require `G(go_to_dock -> dock_grant)`, not merely `G(dropoff -> dock_grant)`.
   The worker must request before that guarded navigation and persist its request until granted;
   rely on the arbiter's FAIRNESS `G F(grant | !request)` to make the grant arrive (see the arbiter
   rules above). Never put physical occupancy mutexes such as `G(!(r1_at_dock & r2_at_dock))` in
   System Assumptions: occupancy is induced by these controlled grant/navigation outputs.

5. ARRIVAL / EFFECT ASSUMPTION (environment, X-FREE) — for each action A whose effect E the durative
   world realizes over MANY steps, assume EVENTUAL, not next-step:  `G F(E | !A)`.
   NEVER `G(A -> X(E))` (a one-step reaction the durative world cannot honour) and NEVER
   `G(A -> F(E))` (CHASE flattens bare F to the SAME step). Add at least one fairness assumption
   `G F(trigger)`. Every Assumption checked against the simulator MUST be X-free (stutter-invariant).

6. PERSISTENCE (the ONLY X — synthesis-internal, never monitored):
   - hold an action until its effect appears:  `G((action & X(!effect)) -> X(action))`  (a Guarantee;
     guard on the effect's NEXT value -- the current-value form `G(action & !effect -> X(action))`
     forces the action in the very step its effect arrives and is UNREALIZABLE whenever the
     action's own precondition forbids that effect, e.g. `G(pickup -> !has_item)`)
   - a world state persists until something changes it:  `G(state & !changing_action -> X(state))`
     (an Assumption).

6b. SHARED-RESOURCE HANDSHAKE (charger, dock, corridor): an environment-owned location and a
   sibling-owned grant may otherwise alternate on different steps, making a guarded service action
   impossible. A worker must request while it is outside an exclusive entry zone and persist
   that request until granted; it may navigate into that zone only after the grant. For a shared
   service point such as a shelf, it may instead navigate to its reserved bin before requesting
   service; only the pickup is grant-guarded. For a worker that waits at location `at_resource`
   under `request`, add the frame Assumption
   `G(at_resource & request -> X(at_resource))`: after arrival, the world cannot displace the
   waiting worker before its request is released. The worker must keep its request and travel
   action asserted while the triggering condition persists and service has not occurred; use
   Guarantees such as `G(request & trigger & !service -> X(request))` and
   `G(travel & !at_resource -> X(travel))`. Guard entry/travel and service with the grant, e.g.
   `G(travel -> grant)` and `G(service -> (at_resource & grant))`. The Arbiter retains mutual exclusion and fairness
   `G F(grant | !request)`, and must preserve an active grant while its request continues:
   `G((grant & request & X(request)) -> X(grant))`; do not require the Arbiter to grant an
   unbounded request forever.

7. INITIAL VALUES (the #1 cause of DEGENERATE strategies) — pin EVERY output `!x;` at the top of its
   Guarantees (one per action and per flag, no exceptions) and every input's initial in Assumptions.
   An un-pinned output may be asserted at step 0 and dead-end (e.g. `pickup & dropoff` at once).

8. WELL-SEPARATED ENVIRONMENT — the system must never force the environment into a state with no
   legal move. The X-free arrival `G F(E | !A)` already avoids the classic trap; still, match mutually
   exclusive environment states with mutually exclusive system outputs `G(!(a & b))` + persistence,
   and make each reaction a TOTAL function of the system's action.

ABSTRACT shape (GENERIC names — substitute the task's own predicates/actions):
    Assumptions:  !state; !goal;
                  G F(state | !action);                  ; durative arrival, X-free (cat 5)
                  G(state & !other_action -> X(state));  ; frame axiom (cat 6)
                  G F(trigger);
    Guarantees:   !action; !goal;                        ; initial values (cat 7)
                  G(action -> precond);                  ; guard (cat 2)
                  G(!(action & other_action));           ; mutex (cat 3)
                  G((action & X(!state)) -> X(action));  ; persistence, internal X (cat 6)
                  G(goal <-> state);
                  G F(goal);                             ; objective (cat 1)

SELF-CHECK before finishing (against the action model + architecture):
  - every action has a precondition guard (cat 2)?  every exclusive pair a mutex (cat 3)?
    (a causal "R before A" precedence is covered by the guard `G(A -> R)` — no separate ordering.)
  - ARBITER FAIRNESS (the #1 cause of an intermittent RUN-TIME deadlock): does EVERY Arbiter
    guarantee, for EVERY requester, BOTH mutual exclusion `G(!(grant_i & grant_j))` AND fairness as a
    justice goal `G F(grant_i | !request_i)`?  And does each worker ASSUME that same fairness
    `G F(grant | !request)`?  Mutual exclusion WITHOUT fairness is REALIZABLE but DEADLOCKS at run
    time by STARVING a robot (a standing request that is never granted): the arbiter legally grants
    one robot forever. A shared resource with only a mutex and no `G F(grant | !request)` on BOTH the
    arbiter (guarantee) and every worker (assumption) is WRONG — add the fairness justice.
  - every durative effect an X-free `G F(E | !A)` (cat 5)?  every output pinned `!x;` (cat 7)?
   - for every shared-resource service, does the worker keep its request while pending AND assume
      `G(at_resource & request -> X(at_resource))`? Without this location-while-waiting frame axiom,
      an environment can alternate `at_resource` and `grant` forever and defeat the service objective.
If a sub-contract is UNREALIZABLE or degenerate, do NOT hand-patch with scenario knowledge; the
pipeline diagnoses it generically (counter-strategy for unrealizable; dead-end for non-well-
separated) and repairs by adding the missing fairness / exclusion / persistence.
"""

DECOMPOSER_INSTRUCTIONS = (
    "You are a CHASE contract-based design expert.\n"
    "Generate a complete CHASE .ltl file from LTL formulas.\n\n"
    + CHASE_FEW_SHOT
    + CONSTRAINT_RULES
    + CHASE_SYNTAX_RULES
    + "\nOUTPUT: the raw .ltl only (starting at `NAME:`). No markdown, no other prose.\n"
)
