"""
STL -> time-varying CBF constructor — SECOND ORDER (double integrator).

This is the relative-degree-2 HOCBF EXECUTION layer that realizes the physical-layer CBF
contracts of contrauto/cbf_contracts.py (the CBC / RC contracts of the Nuzzo "Contract
Embeddings for Layered Control Architectures" lineage). cbf_contracts.py defines the contracts
and the provable bounds (relative degree 1); THIS module is the per-step QP that actuates them
under the simulator's double-integrator dynamics. (The single-agent reach/avoid HOCBF rules
below are in the spirit of Lindemann & Dimarogonas, L-CSS 2019 — see References.)

Matches the simulator's dynamics: state = (p, v), control u = acceleration
(x'=v, v'=u). A position predicate mu(p) has relative degree 2, so each leaf
emits a High-Order CBF (HOCBF / ECBF) constraint:

  b(p,t) >= 0 ,  with poles -gamma1,-gamma2  (alpha1=g1+g2, alpha2=g1*g2):
     b_ddot + alpha1*b_dot + alpha2*b >= 0
     b_dot  = grad·v + slack'(t)
     b_ddot = vᵀHμv + grad·u           (slack linear -> slack''=0)
  => grad·u  >=  -vᵀHμv - alpha1*(grad·v + slack') - alpha2*(mu + slack)

Same recursive rules as before (predicate / Always / Eventually / And) — only the
constraint is now relative-degree-2. NOT a fixed pattern; generality is in the
per-operator rules. Predicate vocabulary carries mu, grad, and curv = vᵀHμv.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

Vec = np.ndarray


# ── predicate vocabulary:  mu(p) >= 0  with gradient and curvature term vᵀHμv ──
@dataclass
class Pred:
    """A position predicate mu(p) >= 0 with its gradient and curvature term vᵀHμv."""

    name: str
    mu: Callable[[Vec], float]
    grad: Callable[[Vec], Vec]
    curv: Callable[[Vec, Vec], float]              # vᵀ ∇²mu(p) v
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    kind: str = ""                                  # "in" | "out"


def in_circle(cx, cy, r) -> Pred:
    """Predicate for staying INSIDE the disc of radius r at (cx, cy)."""
    c = np.array([cx, cy], float)
    def mu(p): return r - float(np.linalg.norm(p - c))
    def grad(p):
        d = p - c
        n = float(np.linalg.norm(d)) or 1e-9
        return -d / n
    def curv(p, v):
        d = p - c
        dist = float(np.linalg.norm(d)) or 1e-9
        n = d / dist
        return -(float(v @ v) - float(v @ n) ** 2) / dist     # H = -(I-nnᵀ)/dist
    return Pred(f"in_circle({cx},{cy},{r})", mu, grad, curv, (cx, cy), r, "in")


def out_circle(cx, cy, r) -> Pred:
    """Predicate for staying OUTSIDE the disc of radius r at (cx, cy)."""
    c = np.array([cx, cy], float)
    def mu(p): return float(np.linalg.norm(p - c)) - r
    def grad(p):
        d = p - c
        n = float(np.linalg.norm(d)) or 1e-9
        return d / n
    def curv(p, v):
        d = p - c
        dist = float(np.linalg.norm(d)) or 1e-9
        n = d / dist
        return (float(v @ v) - float(v @ n) ** 2) / dist       # H = (I-nnᵀ)/dist
    return Pred(f"out_circle({cx},{cy},{r})", mu, grad, curv, (cx, cy), r, "out")


def out_obb(cx, cy, hw, hh, ang, margin=0.0) -> Pred:
    """Outside an ORIENTED (rotated) rectangle by `margin`: mu = signed_dist(p, OBB) - margin
    (>= 0  <=>  the robot centre stays `margin` clear of the shelf). Reuses the simulator's
    rect_signed_distance_and_derivatives in the shelf's local frame, then rotates the gradient
    and Hessian back to world coordinates (h is rotation-invariant). This is the SHELF / wall
    avoidance barrier that two_layer_dock's CBF was missing (it only avoided peers)."""
    from cbf.filter import rect_signed_distance_and_derivatives
    c = np.array([cx, cy], float)
    ca, sa = float(np.cos(ang)), float(np.sin(ang))
    R = np.array([[ca, -sa], [sa, ca]])                 # local -> world
    Rt = R.T                                            # world -> local

    def _sd(p):
        pl = Rt @ (np.asarray(p, float) - c)
        return rect_signed_distance_and_derivatives(pl[0], pl[1], -hw, hw, -hh, hh)
    def mu(p): return _sd(p)[0] - margin
    def grad(p): return R @ _sd(p)[1]
    def curv(p, v):
        H = _sd(p)[2]
        vl = Rt @ np.asarray(v, float)                  # vᵀ(R Hₗ Rᵀ)v = (Rᵀv)ᵀ Hₗ (Rᵀv)
        return float(vl @ H @ vl)
    return Pred(f"out_obb({cx:.0f},{cy:.0f},{hw:.0f},{hh:.0f},{ang:.2f})", mu, grad, curv,
                (cx, cy), None, "out")


def _hocbf_row(pred, p, v, mu_off, dmu_off, g1, g2):
    """One HOCBF constraint grad·u >= rhs for barrier b = mu + mu_off(t)."""
    a1, a2 = g1 + g2, g1 * g2
    grad = pred.grad(p)
    b = pred.mu(p) + mu_off
    bdot = float(grad @ v) + dmu_off
    rhs = -pred.curv(p, v) - a1 * bdot - a2 * b
    return grad, rhs


# ── STL AST.  constraints(p,v,t,g1,g2) -> list of (a_vec, b_scalar):  a·u >= b ─
@dataclass
class Always:
    """G[a,b] pred: enforce the predicate at every t in [a, b]."""

    a: float
    b: float
    pred: Pred

    def constraints(self, p, v, t, g1, g2):
        if not (self.a <= t <= self.b):
            return []
        return [_hocbf_row(self.pred, p, v, 0.0, 0.0, g1, g2)]


@dataclass
class Eventually:
    """F[a,b] pred: reach the predicate by deadline b via a slack that shrinks to 0."""

    a: float
    b: float
    pred: Pred
    _mu0: Optional[float] = field(default=None, repr=False)

    def constraints(self, p, v, t, g1, g2):
        if t > self.b:
            return []
        if self._mu0 is None:
            self._mu0 = self.pred.mu(p)
        deficit = max(0.0, -self._mu0)
        d = max(self.b, 1e-6)
        slack = deficit * max(0.0, (d - t) / d)        # shrinks to 0 by deadline
        dslack = -deficit / d if t < d else 0.0
        return [_hocbf_row(self.pred, p, v, slack, dslack, g1, g2)]


@dataclass
class And:
    """Conjunction: the union of all children's constraints."""

    children: list

    def constraints(self, p, v, t, g1, g2):
        out = []
        for c in self.children:
            out.extend(c.constraints(p, v, t, g1, g2))
        return out


# ── CBF-QP step (double integrator: x'=v, v'=u=accel) ────────────────────────
def stl_cbf_step(formula, p, v, t, u_nom, dt, a_max=400.0, v_max=None, g1=3.0, g2=3.0, slack_w=1e4):
    """One CBF-QP step. Returns (p_new, v_new, u, slack) where `slack` is the max constraint
    relaxation used this step: slack==0 means the STL constraints were met by construction;
    slack>0 means they could NOT be (the spec — e.g. a reach deadline — is infeasible here),
    which is the signal the physical layer feeds back to repair.

    `a_max` caps acceleration; `v_max` (if given) caps speed so reach TIMES are physically
    real — without it the QP can saturate accel and let speed grow unbounded, "reaching"
    impossibly fast."""
    import cvxpy as cp
    rows = formula.constraints(p, v, t, g1, g2)
    u = cp.Variable(2)
    s = cp.Variable(len(rows), nonneg=True) if rows else None
    cons = [cp.norm(u, 2) <= a_max]
    if v_max is not None:
        cons.append(cp.norm(v + u * dt, 2) <= v_max)        # speed limit -> real reach times
    for i, (a, b) in enumerate(rows):
        cons.append(a @ u + s[i] >= b)
    cost = cp.sum_squares(u - np.asarray(u_nom, float))
    if rows:
        cost = cost + slack_w * cp.sum_squares(s)
    cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.CLARABEL, verbose=False)
    uval = np.asarray(u.value, float).reshape(2) if u.value is not None else np.zeros(2)
    slack = float(np.max(s.value)) if (rows and s is not None and s.value is not None) else 0.0
    v_new = v + uval * dt
    p_new = p + v_new * dt                              # semi-implicit Euler (stable)
    return p_new, v_new, uval, slack


def active_target(formula, p, t):
    """u_nom driver: centre of the earliest still-unmet Eventually region."""
    evs = formula.children if isinstance(formula, And) else [formula]
    best = None
    for c in evs:
        if isinstance(c, Eventually) and c.pred.center and t <= c.b and c.pred.mu(p) < 0:
            if best is None or c.b < best.b:
                best = c
    return np.array(best.pred.center, float) if best else None


# ── self-test: SECOND-ORDER constructor on stay and reach-avoid(+waypoint) ────
if __name__ == "__main__":
    def run(formula, p0, T, goal=None, obs=None, stay=None, dt=1 / 30.0, vmax=130.0):
        p = np.array(p0, float)
        v = np.zeros(2)
        t = 0.0
        reached, min_clear, stay_ok = None, 1e9, True
        gc = np.array(goal[:2], float) if goal else None
        while t <= T + 1e-9:
            tgt = active_target(formula, p, t)
            if tgt is None and gc is not None:
                tgt = gc
            # acceleration nominal: PD toward target (capped to a cruise speed)
            if tgt is not None:
                dirn = tgt - p
                vdes = dirn / (np.linalg.norm(dirn) or 1) * vmax
                u_nom = 6.0 * (vdes - v)
            else:
                u_nom = -6.0 * v
            p, v, _, _ = stl_cbf_step(formula, p, v, t, u_nom, dt)
            if goal and reached is None and np.linalg.norm(p - gc) <= goal[2]:
                reached = round(t, 2)
            if obs:
                min_clear = min(min_clear, np.linalg.norm(p - np.array(obs[:2])) - obs[2])
            if stay and stay[3] <= t <= stay[4] and np.linalg.norm(p - np.array(stay[:2])) > stay[2]:
                stay_ok = False
            t += dt
        return reached, min_clear, stay_ok, p

    print("=== (1) reach-then-stay (2nd order): F[0,3](in A) ∧ G[3,7](in A) ===")
    f1 = And([Eventually(0, 3, in_circle(500, 400, 40)),
              Always(3, 7, in_circle(500, 400, 40))])
    rb, _, ok, p = run(f1, (100, 400), 7.0, goal=(500, 400, 40), stay=(500, 400, 40, 3, 7))
    print(f"   reached_by={rb}s  stayed_in_A[3,7]={ok}  final={p.round(0)}")

    print("=== (2) reach-avoid + waypoint (2nd order): F(wp)∧F(goal)∧G(out obs) ===")
    f2 = And([Eventually(0, 3, in_circle(400, 180, 35)),
              Eventually(3, 7, in_circle(700, 300, 30)),
              Always(0, 7, out_circle(400, 300, 60))])
    rb, mc, _, p = run(f2, (100, 300), 7.0, goal=(700, 300, 30), obs=(400, 300, 60))
    print(f"   reached_by={rb}s  min_obs_clear={mc:.1f}px (>=0 safe)  final={p.round(0)}")
