"""
L1 coordination — `resource-split` (star / arbiter) + global-deadline budget.

For the centralized-coordination case (an arbiter node connected to all agents,
agents not connected to each other), the cross-agent timing reduces to two
localized facts (see docs/DECOMPOSITION_DESIGN.md §6):

  - each agent guarantees its own *active* execution time (from Step A:
    stl_contract.build_nav_contract -> total_time);
  - the arbiter guarantees **bounded waiting**: round-robin with bounded resource
    holding time τ_hold over N requesters ⇒ wait ≤ (N-1)·τ_hold.

Global deadline T is then a per-agent budget check:
    ∀ r:  active_r + waiting_r^max ≤ T
which is a simple inequality (response-time / makespan analysis), NOT a cross
product model check. The only quantitative timing lives in the arbiter contract
(bounded response F[0,τ_wait]); agents only assume the arbiter.
"""
from __future__ import annotations

from typing import Dict, Sequence


def bounded_wait(n_requesters: int, tau_hold: float) -> float:
    """Round-robin worst-case wait for one resource: (N-1)·τ_hold."""
    return max(0, n_requesters - 1) * tau_hold


def global_deadline_check(
    active: Dict[str, float],
    n_requesters: int,
    tau_hold: float,
    T: float,
) -> Dict:
    """Check ∀r: active_r + (N-1)·τ_hold ≤ T (method A: constrain whole system)."""
    w = bounded_wait(n_requesters, tau_hold)
    completion = {r: a + w for r, a in active.items()}
    return {
        "wait_bound": w,
        "completion": completion,
        "T": T,
        "feasible": all(c <= T for c in completion.values()),
        "violations": {r: c for r, c in completion.items() if c > T},
    }


def llm_pick_arbiter(feedback: str, robots, model: str = "gemini") -> Dict:
    """Coordination redesign on a runtime DEADLOCK at the shared dock: the agent
    introduces a DOCK ARBITER (the operational form of build_arbiter_contract —
    mutual exclusion + bounded waiting) and CHOOSES its capacity in [1, N-1].
    Returns {"capacity": int, "rationale": str}. Robust to LLM errors (-> serialize)."""
    try:
        from llm import chat, parse_json_response
    except ModuleNotFoundError:
        from contrauto.llm import chat, parse_json_response
    n = len(robots)
    hi = max(1, n - 1)
    system = (
        "You redesign COORDINATION for a multi-robot system that hit a runtime "
        "DEADLOCK at a shared resource (the dock). Introduce a DOCK ARBITER with "
        "mutual exclusion + bounded waiting: at most `capacity` robots may occupy the "
        "dock zone at once; the rest wait outside and are admitted as slots free. Pick "
        "the capacity that removes contention with the LEAST serialization (larger = "
        "more parallel but more contention; 1 = fully serialized). "
        "Reply ONLY JSON {\"capacity\": int, \"rationale\": str}.")
    user = feedback + f"\nN = {n} robots. Choose capacity in [1,{hi}]."
    try:
        txt, _ = chat(model, system, user)
        j = parse_json_response(txt)
        cap = max(1, min(int(j.get("capacity", 1)), hi))
        return {"capacity": cap, "rationale": str(j.get("rationale", ""))[:140]}
    except Exception as e:
        return {"capacity": 1, "rationale": f"(LLM {type(e).__name__}; defaulting to serialize)"}


def llm_pick_coordination(feedback: str, robots, tried=None, model: str = "gemini") -> Dict:
    """Coordination redesign on a runtime DEADLOCK at the dock (currently ONE shared
    drop point). The agent chooses a mechanism + parameter:
      - "slots":   SPLIT the dock into K spatially separated unloading points
                   (resource split; removes spatial contention). K in [2, N].
      - "arbiter": mutual exclusion admitting at most K robots at the single point
                   (temporal; coordination.build_arbiter_contract). K in [1, N-1].
    Returns {"strategy","param","rationale"}. Robust to LLM errors (-> slots=N)."""
    try:
        from llm import chat, parse_json_response
    except ModuleNotFoundError:
        from contrauto.llm import chat, parse_json_response
    n = len(robots)
    hi = max(1, n - 1)
    system = (
        "You redesign COORDINATION for a multi-robot delivery system that hit a runtime "
        "DEADLOCK at the DOCK, which is currently ONE shared drop point. Choose a "
        "mechanism:\n"
        " - \"slots\": SPLIT the dock into K spatially separated unloading points so "
        "robots stop contending in space (resource split). param=K in [2,N].\n"
        " - \"arbiter\": mutual exclusion admitting at most K robots at the single point "
        "at once (temporal). param=K in [1,N-1].\n"
        "For a single congested POINT, spatial separation (slots) removes the contention "
        "most directly. Reply ONLY JSON "
        "{\"strategy\":\"slots\"|\"arbiter\",\"param\":int,\"rationale\":str}.")
    user = feedback + f"\nN = {n} robots."
    if tried:
        user += "\nAlready tried (did not resolve): " + "; ".join(tried)
    try:
        txt, _ = chat(model, system, user)
        j = parse_json_response(txt)
        strat = "arbiter" if str(j.get("strategy", "slots")).startswith("arb") else "slots"
        param = int(j.get("param", n))
        param = max(2, min(param, n)) if strat == "slots" else max(1, min(param, hi))
        return {"strategy": strat, "param": param, "rationale": str(j.get("rationale", ""))[:140]}
    except Exception as e:
        return {"strategy": "slots", "param": n, "rationale": f"(LLM {type(e).__name__}; default slots=N)"}


def build_arbiter_contract(
    robots: Sequence[str],
    tau_wait: float,
    *,
    name: str = "Arbiter",
    resource: str = "",
) -> str:
    """
    STL/MTL arbiter contract text (logics_tool syntax):
      mutual exclusion  G(!(grant_i & grant_j))
      grant only on request  G(grant_i -> req_i)
      bounded response  G(req_i -> F[0,τ_wait](grant_i))
    """
    sfx = f"_{resource}" if resource else ""
    req = {r: f"req_{r}{sfx}" for r in robots}
    grant = {r: f"grant_{r}{sfx}" for r in robots}
    tau = round(tau_wait)

    L = [f"NAME: {name};", "", f"CONTRACT {name}:"]
    for r in robots:
        L.append(f"    input boolean variable {req[r]};")
    for r in robots:
        L.append(f"    output boolean variable {grant[r]};")
    L.append("")
    L.append("    Assumptions:")
    for r in robots:
        L.append(f"        !{req[r]};")
    L.append("")
    L.append("    Guarantees:")
    for r in robots:
        L.append(f"        !{grant[r]};")
    # mutual exclusion over every pair
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            L.append(f"        G(!({grant[robots[i]]} & {grant[robots[j]]}));")
    for r in robots:
        L.append(f"        G({grant[r]} -> {req[r]});")
        L.append(f"        G({req[r]} -> F[0,{tau}]({grant[r]}));")
    return "\n".join(L) + "\n"

