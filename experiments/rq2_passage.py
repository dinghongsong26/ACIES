"""RQ2 (passage form): the synthesized passage arbiter holds the lock through a whole
crossing; a terminal (dock-style) arbiter preempts it -- the corridor bug, quantified.

A passage resource (single-lane corridor, shared region) requires the grant to be HELD
for the entire crossing. We drive each arbiter with adversarial *crossing workloads*: a
robot, once it starts requesting, keeps req=1 for a crossing of adversarially-chosen
length C (>=1), then releases (drops req) -- this respects the arbiter's env assumption
GF!req_i (crossings are finite). The adversary chooses who starts when and every C.

We check the three passage guarantees at each step:
  mutex          -- at most one grant active
  non-preemption -- a standing lock is never revoked: grant_i & req_i  =>  next grant_i
                    (a preemption is exactly the 'crossing=1, grant=0' corridor failure)
  starvation     -- a robot requesting until served is eventually granted

Compared policies:
  * passage (ours)      -- the synthesized hold-until-release arbiter (outputs/corridor_direct)
  * terminal (dock)     -- the synthesized single-resource dock arbiter (outputs/sweep gen_Nr_1res):
                           correct for terminal resources, but MEMORYLESS, so it may reassign the
                           grant mid-crossing -> preemption on a passage workload
  * greedy / tdma       -- non-reactive plan baselines (as in rq2_adversarial.py)

Run:  python experiments/rq2_passage.py [--n 2 3 4]
"""
import os
import sys
import json
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrauto.sim_iface import FsmController, load_fsm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASSAGE = {n: f"outputs/corridor_direct/passage_{n}r.json" for n in range(2, 6)}
TERMINAL = {  # synthesized dock (terminal) arbiter, same req/grant interface
    2: "outputs/sweep/gen_2r_1res_arbiter_justice__seed0/workdir/arbiter.json",
    3: "outputs/sweep/gen_3r_1res_arbiter_justice__seed0/workdir/arbiter.json",
    4: "outputs/sweep/gen_4r_1res_arbiter_justice__seed0/workdir/arbiter.json",
    5: "outputs/sweep/gen_5r_1res_arbiter_justice__seed0/workdir/arbiter.json",
}


class FsmArb:
    """Drive a synthesized arbiter FSM (passage or terminal)."""
    def __init__(self, path, n, name):
        self.n, self.name = n, name
        req = [f"req_r{i+1}" for i in range(n)]
        grant = [f"grant_r{i+1}" for i in range(n)]
        self.req, self.grant = req, grant
        self.ctrl = FsmController(load_fsm(os.path.join(ROOT, path)), in_vars=req, out_vars=grant)
        self.offmodel = 0

    def reset(self):
        self.ctrl.reset(); self.offmodel = 0

    def step(self, reqvec, t):
        env = {self.req[i]: reqvec[i] for i in range(self.n)}
        before = self.ctrl.current
        out = self.ctrl.step(env)
        if self.ctrl.current == before and self.ctrl._stuck:
            self.offmodel += 1
        return tuple(out.get(self.grant[i], 0) for i in range(self.n))


class Greedy:
    def __init__(self, n): self.n = n
    name = "greedy"
    def reset(self): pass
    def step(self, reqvec, t): return tuple(reqvec)


class StaticTDMA:
    def __init__(self, n): self.n = n
    name = "static_tdma"
    def reset(self): pass
    def step(self, reqvec, t):
        g = [0] * self.n; g[t % self.n] = 1; return tuple(g)


def crossing_stream(n, rng, H, cmax):
    """A legal crossing workload: each robot idles/crosses; a crossing holds req=1 for
    C in [1,cmax] steps then releases for >=1 step (GF!req_i respected)."""
    req = [[0] * H for _ in range(n)]
    for i in range(n):
        t = rng.randint(0, 2)
        while t < H:
            if rng.random() < 0.6:                     # start a crossing
                C = rng.randint(1, cmax)
                for k in range(t, min(H, t + C)):
                    req[i][k] = 1
                t = t + C + rng.randint(1, 2)           # release gap (idle) -> GF!req
            else:
                t += 1
    return [tuple(req[i][k] for i in range(n)) for k in range(H)]


def eval_passage(policy, stream, n):
    policy.reset()
    H = len(stream)
    mutex = preempt = 0
    run = [0] * n
    starved = [False] * n
    prev_g = tuple([0] * n)
    prev_req = tuple([0] * n)
    for t, reqvec in enumerate(stream):
        g = policy.step(reqvec, t)
        if sum(g) > 1:
            mutex += 1
        # non-preemption: a lock held last step under a standing request must persist
        for i in range(n):
            if prev_g[i] and prev_req[i] and reqvec[i] and not g[i]:
                preempt += 1                            # revoked a standing lock -> corridor bug
        for i in range(n):
            if reqvec[i] and not g[i]:
                run[i] += 1
                if run[i] >= 2 * n + 1 and t == H - 1:
                    starved[i] = True
            else:
                run[i] = 0
        prev_g, prev_req = g, reqvec
    return {"mutex": mutex > 0, "preempt": preempt > 0, "starved": any(starved)}


def run_n(n, sample, seed):
    rng = random.Random(seed)
    H, cmax = 6 * n, n + 2
    streams = [crossing_stream(n, rng, H, cmax) for _ in range(sample)]
    policies = []
    if os.path.exists(os.path.join(ROOT, PASSAGE[n])):
        policies.append(FsmArb(PASSAGE[n], n, "passage (ours)"))
    if os.path.exists(os.path.join(ROOT, TERMINAL[n])):
        policies.append(FsmArb(TERMINAL[n], n, "terminal (dock)"))
    policies += [Greedy(n), StaticTDMA(n)]
    out = {}
    for p in policies:
        agg = {"mutex": 0, "preempt": 0, "starved": 0}
        for s in streams:
            v = eval_passage(p, s, n)
            for k in agg:
                agg[k] += int(v[k])
        rec = {k: agg[k] / len(streams) for k in agg}
        if isinstance(p, FsmArb):
            rec["offmodel"] = p.offmodel
        out[p.name] = rec
    return {"n": n, "H": H, "n_streams": len(streams), "policies": out}


def fmt(all_res):
    L = []
    for r in all_res:
        L.append(f"\n### passage N={r['n']}  ({r['n_streams']} adversarial crossing streams, H={r['H']})")
        L.append("| policy | mutex-viol | preemption | starvation |")
        L.append("|---|---|---|---|")
        for name, p in r["policies"].items():
            b = name.startswith("passage")
            s, e = ("**", "**") if b else ("", "")
            L.append(f"|{s}{name}{e} | {p['mutex']:.3f} | {p['preempt']:.3f} | {p['starved']:.3f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/experiments/rq2_passage_results.json")
    a = ap.parse_args()
    all_res = [run_n(n, a.sample, a.seed) for n in a.n]
    tbl = fmt(all_res)
    print(tbl)
    with open(a.out, "w") as f:
        json.dump(all_res, f, indent=2)
    with open(a.out.replace(".json", ".md"), "w") as f:
        f.write("# RQ2 passage: hold-until-release vs preemption\n" + tbl + "\n")
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
