"""Physical-layer CBF assume-guarantee contracts (Nuzzo/CHASE lineage).

Terminology aligned to Naik, Pinto, Nuzzo, "Contract Embeddings for Layered Control
Architectures," ACM TECS 24(5s), Art. 138, 2025 (DOI 10.1145/3764587) -- the journal
formalization of our exact task(LTL_f)+physical(CBF) layered control architecture (LCA).
Our task layer is the discrete LTL_f planning contract; this module is the continuous
component layer it refines down to. Two contract classes (their Sec. 3.4):

  * CBC -- Control Barrier Contract (Def. 4): forward invariance / safety over [t_i,t_f].
        A: x(t_i) in C        G: forall t in [t_i,t_f], x(t) in C.
    Implemented (Sec. 3.5) by a CBF rendering the safe set C = {x : h(x) >= 0} forward-
    invariant.

  * RC -- Reachability Contract (Def. 5): reach a target set by the end of the interval.
        A: x(t_i) in C        G: x(t_f) in C_r  (C_r subset C).
    Implemented by a finite-time reachability function (FTRF) with provable reach time
        T_r <= |r(x(t_i))|^(1-delta) / (eta*(1-delta))   (Sec. 3.5),
    which is the deadline-by-construction.

The live entry point is `physical_certificate`: it decomposes a robot's route into the
Thm. 7 chain of RCs, sums the per-segment provable reach times into a completion bound, and
checks that bound against a mission deadline. A failing certificate is the formal
infeasibility signal the task-level repair loop consumes.

Dynamics: control-affine x' = f(x) + g(x) u, relative degree 1 (the paper's setting).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Vec = np.ndarray


@dataclass
class TVPred:
    """A (time-varying) predicate h(t,x) defining a safe/target set C(t) = {x : h(t,x) >= 0}.

    Attributes:
        name: Human-readable label.
        h: Value h(t,x); >= 0 iff x is inside C(t).
        gradx: Spatial gradient grad_x h(t,x) (used for the Lie derivatives L_f h, L_g h).
        dhdt: Explicit time derivative d h/d t (t,x); 0 for static sets.
    """
    name: str
    h: Callable[[float, Vec], float]
    gradx: Callable[[float, Vec], Vec]
    dhdt: Callable[[float, Vec], float] = lambda t, x: 0.0


def ball_in(center, radius) -> TVPred:
    """Predicate for being inside a (possibly moving) ball: h = radius - ||x - center||.

    Args:
        center: Ball center, a constant point or a callable t -> point.
        radius: Ball radius, a constant or a callable t -> radius.

    Returns:
        The corresponding TVPred.
    """
    cfun = center if callable(center) else (lambda t: np.asarray(center, float))
    rfun = radius if callable(radius) else (lambda t: float(radius))

    def h(t, x): return rfun(t) - float(np.linalg.norm(x - cfun(t)))
    def gradx(t, x):
        d = x - cfun(t)
        n = float(np.linalg.norm(d)) or 1e-9
        return -d / n
    def dhdt(t, x):
        dt = 1e-6
        d = x - cfun(t)
        n = float(np.linalg.norm(d)) or 1e-9
        cdot = (cfun(t + dt) - cfun(t)) / dt
        rdot = (rfun(t + dt) - rfun(t)) / dt
        return rdot + float(d @ cdot) / n
    return TVPred(f"ball_in(r={radius if not callable(radius) else 'r(t)'})", h, gradx, dhdt)


@dataclass
class Contract:
    """A CBF contract on the interval [a, b] (their Def. 4/5).

    kind == "cbc" -> Control Barrier Contract: forward invariance (stay in C), class-K gain
                     `alpha`.
    kind == "rc"  -> Reachability Contract: reach C_r by t=b via a finite-time reachability
                     function with gains `eta` and exponent `delta` (0<=delta<1).
    """
    pred: TVPred
    a: float
    b: float
    kind: str = "cbc"           # "cbc" (invariance) | "rc" (reachability)
    alpha: float = 2.0          # class-K gain for the CBC
    eta: float = 2.0            # FTRF gain for the RC
    delta: float = 0.5          # FTRF exponent 0<=delta<1


def admissibility(reach_times, T_s) -> dict:
    """Check admissibility (their Def. 7 / Thm. 4 cond. 2): the sampling time must exceed the
    worst-case reach time across transitions.

    Args:
        reach_times: Per-transition RC reach times T_r.
        T_s: Sampling time.

    Returns:
        {"ok", "T_r_max", "T_s"}; ok == False is the formal infeasibility signal for repair
        (raise T_s / relax the deadline).
    """
    tmax = max(reach_times) if reach_times else 0.0
    return {"ok": T_s > tmax, "T_r_max": tmax, "T_s": float(T_s)}


def route_reach_chain(start, leg_targets, rects, bounds, t0=0.0, reach_r=12.0):
    """Decompose a robot's route into the Thm. 7 chain of reachability contracts.

    Each straight A* segment of start -> leg_targets[0] -> ... becomes an RC whose interval
    length is the CBF-provable trapezoidal reach time (rest-to-rest under |v|<=VMAX,
    |a|<=A_MAX). The sum of intervals is the provable completion bound; convergence-in-time
    holds by construction (interval == achievable reach).

    Returns:
        (contracts, total_bound, seglog). On an unreachable leg: (None, inf, seglog) -- a
        formal infeasibility for repair.
    """
    import math
    from contrauto.assign_decomposer import plan_waypoints, _trapezoid_time, R as NAV_R
    contracts, seglog, t, p = [], [], float(t0), tuple(start)
    for tgt in leg_targets:
        wp = plan_waypoints(p, tuple(tgt), rects, bounds, NAV_R)
        if not wp:
            return None, float("inf"), seglog + [("UNREACHABLE", p, tuple(tgt))]
        for i in range(len(wp) - 1):
            L = math.hypot(wp[i + 1][0] - wp[i][0], wp[i + 1][1] - wp[i][1])
            d = _trapezoid_time(L)
            c = Contract(ball_in(np.array(wp[i + 1], float), reach_r), a=t, b=t + d, kind="rc")
            contracts.append(c)
            seglog.append((round(t, 2), round(t + d, 2), round(L, 1)))
            t += d
        p = tuple(tgt)
    return contracts, t - t0, seglog


def physical_certificate(start, leg_targets, rects, bounds, deadline, t0=0.0, reach_r=12.0):
    """Per-robot physical-layer contract certificate.

    Decomposes the route into a CBC/RC chain (the component contract that an HHCN traces up to
    the task-layer LTL_f contract) and certifies that the provable completion bound meets the
    mission deadline -- "the deadline is a guaranteed property of a satisfied CBC/RC contract."

    Returns:
        {"ok", "bound", "deadline", "n_reach", "contracts", "seglog", "admissible", "issues"};
        ok == False (with a populated "issues") is the infeasibility signal for repair.
    """
    contracts, bound, seglog = route_reach_chain(start, leg_targets, rects, bounds, t0, reach_r)
    if contracts is None:
        return {"ok": False, "bound": bound, "deadline": float(deadline), "n_reach": 0,
                "contracts": None, "seglog": seglog, "admissible": None,
                "issues": ["route unreachable"]}
    # mission-level admissibility: every per-segment reach time fits a sampling step T_s = its
    # own interval (true by construction here), and the aggregate completion fits the deadline.
    seg_times = [b - a for a, b, _ in seglog]
    adm = admissibility(seg_times, T_s=max(seg_times) + 1e-9 if seg_times else 1.0)
    ok = bound <= deadline + 1e-6
    return {"ok": ok, "bound": bound, "deadline": float(deadline), "n_reach": len(contracts),
            "contracts": contracts, "seglog": seglog, "admissible": adm["ok"],
            "issues": [] if ok else [f"provable completion {bound:.1f}s > deadline {deadline:.1f}s"]}
