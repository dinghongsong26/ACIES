"""RQ2 extra row: a REAL LLM-authored arbitration rule, tested in both harnesses.

Asked gemini-3.1-pro (2026-07-09, transcript: outputs/case_studies/p1_llm_schedule.txt):
  "Three robots share one loading dock ... Produce a fixed schedule or priority rule
   that decides, at every timestep, which requesting robot gets the dock."
Answer: "Assign the dock to the requesting robot that has waited the longest since its
   last service; resolve any initial ties using a fixed ID hierarchy (Robot 1 > 2 > 3)."

This is NOT one of the three fixed-commitment reference policies -- it is a reactive,
memoryful fairness rule (longest-wait-first). The interesting question is therefore not
"which policy does it reduce to" but "is an UNCERTIFIED plausible rule actually correct":
  * terminal harness (rq2_adversarial): expected to hold mutex/starvation/spurious;
  * passage workload (rq2_passage): expected to PREEMPT a standing lock mid-crossing
    (the rule reassigns as soon as a waiter out-waits the holder), i.e. the corridor bug.

Run:  python experiments/rq2_llm_row.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rq2_adversarial as term
import rq2_passage as passage


class LlmLongestWait:
    """gemini-3.1-pro's rule: grant the requesting robot with the longest wait since
    its last service; ties broken by fixed ID order (lowest index first)."""
    name = "llm_longest_wait (gemini)"

    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.wait = [0] * self.n          # steps since last service (or since t=0)

    def step(self, reqvec, t):
        g = [0] * self.n
        requesters = [i for i in range(self.n) if reqvec[i]]
        if requesters:
            # max wait; tie -> lowest ID (Robot 1 > 2 > 3)
            best = max(requesters, key=lambda i: (self.wait[i], -i))
            g[best] = 1
        for i in range(self.n):
            self.wait[i] = 0 if g[i] else self.wait[i] + 1
        return tuple(g)


def terminal_row(n=3, sample=50000, seed=0):
    rng = random.Random(seed)
    H = 4 * n
    starve_suffix = 2 * n + 1
    streams = list(term.streams_random(n, H, sample, rng)) + list(term.streams_targeted(n, H))
    p = LlmLongestWait(n)
    agg = {"mutex": 0, "spurious": 0, "starved": 0}
    for s in streams:
        r = term.eval_stream(p, s, n, starve_suffix)
        for k in agg:
            agg[k] += bool(r.get(k) or r.get(k + "_viol"))
    tot = len(streams)
    print(f"TERMINAL (N={n}, {tot} adversarial request streams):")
    for k, v in agg.items():
        print(f"  {k:10s} violation rate = {v/tot:.3f}")
    return agg


def passage_row(n=3, sample=50000, seed=0):
    rng = random.Random(seed)
    H = 6 * n
    p = LlmLongestWait(n)
    agg = {"mutex": 0, "preempt": 0, "starved": 0}
    tot = 0
    for _ in range(sample):
        s = passage.crossing_stream(n, rng, H, cmax=4)
        r = passage.eval_passage(p, s, n)
        for k in agg:
            agg[k] += bool(r[k])
        tot += 1
    print(f"PASSAGE (N={n}, {tot} adversarial crossing streams):")
    for k, v in agg.items():
        print(f"  {k:10s} violation rate = {v/tot:.3f}")
    return agg


if __name__ == "__main__":
    terminal_row()
    passage_row()
