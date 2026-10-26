"""Behavioral end-to-end check for the HIERARCHICAL arbiter tree (N=8).

Wires the three synthesized tree FSMs (GroupArb1, GroupArb2, RootArb from
outputs/hierarchy_replay/) into one composite arbiter and drives it with the same
adversarial request streams as RQ2, checking the three guarantees the flat arbiter
is certified for: mutual exclusion, no spurious grant, no starvation.

Execution order per tick (faithful to the certified contracts):
  1. RootArb steps on the groups' PREVIOUS greq outputs (matches the one-step
     DELAYED demand aggregator in the group contracts) -> ggrant(t).
  2. Each GroupArb steps on {req(t), ggrant(t)} -> grants(t), and its greq output
     is buffered for the root's next tick.

Run:  python experiments/hierarchy_behavioral.py [--sample 50000]
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rq2_adversarial as term
from contrauto.sim_iface import FsmController, load_fsm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, "outputs/hierarchy_replay")
GROUPS = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}


class TreeArb:
    """Composite two-level arbiter: (req vector over 8 robots) -> (grant vector)."""
    name = "hierarchical tree (2x GroupArb + RootArb)"

    def __init__(self):
        self.g = {}
        for k, rs in GROUPS.items():
            self.g[k] = FsmController(
                load_fsm(f"{WD}/grouparb{k}.json"),
                in_vars=[f"req_r{i}" for i in rs] + [f"ggrant_g{k}"],
                out_vars=[f"grant_r{i}" for i in rs] + [f"greq_g{k}"])
        self.root = FsmController(
            load_fsm(f"{WD}/rootarb.json"),
            in_vars=["greq_g1", "greq_g2"],
            out_vars=["ggrant_g1", "ggrant_g2"])
        self.reset()

    def reset(self):
        for c in self.g.values():
            c.reset()
        self.root.reset()
        self.prev_greq = {1: 0, 2: 0}

    def step(self, reqvec, t):
        rout = self.root.step({"greq_g1": self.prev_greq[1],
                               "greq_g2": self.prev_greq[2]})
        grants = [0] * 8
        for k, rs in GROUPS.items():
            env = {f"req_r{i}": reqvec[i - 1] for i in rs}
            env[f"ggrant_g{k}"] = rout.get(f"ggrant_g{k}", 0)
            out = self.g[k].step(env)
            for i in rs:
                grants[i - 1] = out.get(f"grant_r{i}", 0)
            self.prev_greq[k] = out.get(f"greq_g{k}", 0)
        return tuple(grants)


def main():
    n = 8
    sample = 50000
    if "--sample" in sys.argv:
        sample = int(sys.argv[sys.argv.index("--sample") + 1])
    rng = random.Random(0)
    H = 4 * n
    starve_suffix = 2 * n + 1
    streams = list(term.streams_random(n, H, sample, rng)) + list(term.streams_targeted(n, H))
    p = TreeArb()
    agg = {"mutex": 0, "spurious": 0, "starved": 0}
    for s in streams:
        r = term.eval_stream(p, s, n, starve_suffix)
        for k in agg:
            agg[k] += bool(r.get(k) or r.get(k + "_viol"))
    tot = len(streams)
    print(f"TREE ARBITER, N={n}, {tot} adversarial request streams (H={H}):")
    for k, v in agg.items():
        print(f"  {k:10s} violation rate = {v/tot:.4f}")


if __name__ == "__main__":
    main()
