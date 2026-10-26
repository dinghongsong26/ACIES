"""RQ2 -- Correct coordination under an ADVERSARIAL scheduler (discrete harness).

The paper's central claim is that a synthesized reactive strategy is correct against
*every* environment interleaving, whereas a plan/schedule that commits to a fixed
ordering is correct only on the traces it anticipated. That claim lives at the
DISCRETE reaction layer (the arbiter + worker FSMs), not in the CBF physical layer,
so we test it there directly and rigorously.

Ground truth (from the synthesized arbiter contract, e.g.
gen_2r_1res_arbiter_justice): the arbiter's environment is UNCONSTRAINED
(`[ENV_INIT]` and `[ENV_TRANS]` are empty). Every request interleaving is therefore
admissible, and the synthesized arbiter guarantees, for ALL of them:
  (G1) mutual exclusion         !(grant_i & grant_j)          i != j
  (G2) no spurious grant        grant_i -> req_i
  (G3) no starvation (justice)  a persistent requester is eventually granted

We compare the synthesized arbiter FSM (ours) against non-reactive "plan" policies,
each an abstraction of what a plan/DAG/schedule-based LLM planner commits to:
  * greedy        -- grant every requester now (no coordination): breaks G1
  * fixed_priority-- always serve the lowest-index requester: breaks G3
  * static_tdma   -- commit to a fixed cyclic schedule, ignore live requests: breaks G2
Ours is the only policy that holds all three under an adversary that picks the worst
legal interleaving for the policy under test.

For N=2 we ENUMERATE every request stream over the horizon (2^(N*H) of them), so
"ours never violates" is machine-checked over the whole horizon; for N>=3 we combine
a large random sample with three targeted worst-case adversaries.

Run:  python experiments/rq2_adversarial.py            # default N=2..5
      python experiments/rq2_adversarial.py --n 2 --exhaustive
"""
import os
import sys
import json
import argparse
import itertools
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrauto.sim_iface import FsmController, load_fsm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# synthesized single-resource arbiters produced by the overnight sweep (one per N)
ARBITERS = {
    2: "outputs/sweep/gen_2r_1res_arbiter_justice__seed0/workdir/arbiter.json",
    3: "outputs/sweep/gen_3r_1res_arbiter_justice__seed0/workdir/arbiter.json",
    4: "outputs/sweep/gen_4r_1res_arbiter_justice__seed0/workdir/arbiter.json",
    5: "outputs/sweep/gen_5r_1res_arbiter_justice__seed0/workdir/arbiter.json",
}


# ── policies: (request-vector over N robots, step index) -> grant-vector ──────────
class Synth:
    """OURS: drive the synthesized arbiter FSM (Moore). Correct by construction."""
    name = "ContrAuto (synth)"

    def __init__(self, arbiter_path, n):
        req = [f"req_r{i+1}" for i in range(n)]
        grant = [f"grant_r{i+1}" for i in range(n)]
        self.n, self.req, self.grant = n, req, grant
        self.ctrl = FsmController(load_fsm(arbiter_path), in_vars=req, out_vars=grant)
        self.offmodel = 0

    def reset(self):
        self.ctrl.reset()
        self.offmodel = 0

    def step(self, reqvec, t):
        env = {self.req[i]: reqvec[i] for i in range(self.n)}
        before = self.ctrl.current
        out = self.ctrl.step(env)
        if self.ctrl.current == before and self.ctrl._stuck:
            self.offmodel += 1          # should stay 0: env is unconstrained
        return tuple(out.get(self.grant[i], 0) for i in range(self.n))


class Greedy:
    """PLAN: every requester takes the resource immediately (no arbitration)."""
    name = "greedy"

    def __init__(self, n): self.n = n
    def reset(self): pass
    def step(self, reqvec, t): return tuple(reqvec)          # breaks mutex


class FixedPriority:
    """PLAN: always serve the lowest-index requester (a fixed priority order)."""
    name = "fixed_priority"

    def __init__(self, n): self.n = n
    def reset(self): pass

    def step(self, reqvec, t):
        g = [0] * self.n
        for i in range(self.n):
            if reqvec[i]:
                g[i] = 1                                    # first requester only
                break
        return tuple(g)                                     # starves higher indices


class StaticTDMA:
    """PLAN: commit to a fixed cyclic schedule; grant slot (t mod N), ignore requests."""
    name = "static_tdma"

    def __init__(self, n): self.n = n
    def reset(self): pass

    def step(self, reqvec, t):
        g = [0] * self.n
        g[t % self.n] = 1                                   # grants even a non-requester
        return tuple(g)


# ── evaluation of one policy over one request stream ─────────────────────────────
def eval_stream(policy, stream, n, starve_suffix):
    """Return dict of guarantee-violation flags over a single request stream.

    stream: list of request-vectors (tuples of 0/1), length H.
    starve_suffix: a robot counts as STARVED if there is a suffix of >= this many
      steps in which it requests continuously yet is never granted (bounded-latency
      service, as ours provides, never triggers it; a policy that never serves the
      victim does).
    """
    policy.reset()
    H = len(stream)
    mutex_viol = spurious = 0
    # per-robot: length of the current run of (req & !grant), and whether it ever
    # reached the tail of the horizon (a genuine never-served suffix).
    run = [0] * n
    starved = [False] * n
    for t, reqvec in enumerate(stream):
        g = policy.step(reqvec, t)
        active = sum(g)
        if active > 1:
            mutex_viol += 1
        for i in range(n):
            if g[i] and not reqvec[i]:
                spurious += 1
            if reqvec[i] and not g[i]:
                run[i] += 1
                if run[i] >= starve_suffix and t == H - 1:
                    starved[i] = True           # requested & unserved through horizon end
            else:
                run[i] = 0
    return {
        "mutex": mutex_viol > 0,
        "spurious": spurious > 0,
        "starved": any(starved),
    }


# ── adversaries: sets of legal request streams ───────────────────────────────────
def streams_exhaustive(n, H):
    """ALL request streams over the horizon: {0,1}^(n*H). Only feasible for small n*H."""
    perbit = list(itertools.product([0, 1], repeat=n))      # all req-vectors
    for combo in itertools.product(perbit, repeat=H):
        yield list(combo)


def streams_random(n, H, count, rng):
    for _ in range(count):
        yield [tuple(rng.randint(0, 1) for _ in range(n)) for _ in range(H)]


def streams_targeted(n, H):
    """Three deliberate worst-case legal interleavings (named), one per plan failure."""
    # (a) saturating: everyone requests every step -> breaks greedy's mutex
    yield "saturate", [tuple([1] * n) for _ in range(H)]
    # (b) persistent low-index contention: r1 (and r2) always request -> starves the rest
    persist = [1, 1] + [0] * (n - 2) if n >= 2 else [1] * n
    yield "persist_low", [tuple(persist) for _ in range(H)]
    # (c) bursty: each robot requests only on the step AFTER its tdma slot -> tdma
    #     grants the wrong (idle) robot: maximal spurious grants / service latency
    burst = []
    for t in range(H):
        v = [0] * n
        v[(t - 1) % n] = 1
        burst.append(tuple(v))
    yield "bursty_offslot", burst


def run_n(n, arbiter_path, exhaustive, sample, seed):
    rng = random.Random(seed)
    H = 4 * n                              # horizon: generous vs justice latency (~2N)
    starve_suffix = 2 * n + 1              # > bounded service latency; catches never-served
    policies = [
        Synth(arbiter_path, n),
        Greedy(n), FixedPriority(n), StaticTDMA(n),
    ]

    # collect the stream set
    if exhaustive and n * H <= 20:
        streams = list(streams_exhaustive(n, H))
        mode = f"exhaustive 2^{n*H}={len(streams)} streams, H={H}"
    else:
        streams = list(streams_random(n, H, sample, rng))
        mode = f"random {len(streams)} streams, H={H}"
    named = list(streams_targeted(n, H))

    results = {}
    for p in policies:
        agg = {"mutex": 0, "spurious": 0, "starved": 0, "offmodel": 0}
        for s in streams:
            v = eval_stream(p, s, n, starve_suffix)
            for k in ("mutex", "spurious", "starved"):
                agg[k] += int(v[k])
        # targeted worst-cases (report which named adversary breaks the policy)
        broken_by = {"mutex": [], "spurious": [], "starved": []}
        for label, s in named:
            v = eval_stream(p, s, n, starve_suffix)
            for k in ("mutex", "spurious", "starved"):
                if v[k]:
                    broken_by[k].append(label)
        if isinstance(p, Synth):
            agg["offmodel"] = p.offmodel
        tot = len(streams)
        results[p.name] = {
            "mutex_rate": agg["mutex"] / tot,
            "spurious_rate": agg["spurious"] / tot,
            "starved_rate": agg["starved"] / tot,
            "broken_by": {k: v for k, v in broken_by.items() if v},
            "offmodel_steps": agg["offmodel"],
        }
    return {"n": n, "mode": mode, "H": H, "n_streams": len(streams),
            "arbiter": arbiter_path, "policies": results}


def fmt_table(all_res):
    lines = []
    for r in all_res:
        lines.append(f"\n### N={r['n']}  ({r['mode']})")
        lines.append("| policy | mutex-viol | starvation | spurious-grant | worst-case broken by |")
        lines.append("|---|---|---|---|---|")
        for name, p in r["policies"].items():
            wb = "; ".join(f"{k}:{','.join(v)}" for k, v in p["broken_by"].items()) or "--"
            star = " **" if name.startswith("ContrAuto") else ""
            end = "**" if name.startswith("ContrAuto") else ""
            lines.append(
                f"|{star}{name}{end} | {p['mutex_rate']:.3f} | {p['starved_rate']:.3f} "
                f"| {p['spurious_rate']:.3f} | {wb} |")
        off = r["policies"]["ContrAuto (synth)"]["offmodel_steps"]
        lines.append(f"\n_ours off-model steps (should be 0, env is unconstrained): {off}_")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="single N (default: all in ARBITERS)")
    ap.add_argument("--exhaustive", action="store_true", help="enumerate all streams if n*H<=20")
    ap.add_argument("--sample", type=int, default=200000, help="random streams when not exhaustive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/experiments/rq2_results.json")
    a = ap.parse_args()

    ns = [a.n] if a.n else sorted(ARBITERS)
    all_res = []
    for n in ns:
        path = os.path.join(ROOT, ARBITERS[n])
        if not os.path.exists(path):
            print(f"[skip] N={n}: arbiter not found at {path}")
            continue
        # exhaustive is only feasible for N=2 (2^16); auto-enable there
        ex = a.exhaustive or n == 2
        print(f"[run] N={n} ({'exhaustive' if ex and n*4*n<=20 else 'sampled'}) ...", flush=True)
        all_res.append(run_n(n, path, ex, a.sample, a.seed))

    table = fmt_table(all_res)
    print(table)
    with open(a.out, "w") as f:
        json.dump(all_res, f, indent=2)
    with open(a.out.replace(".json", ".md"), "w") as f:
        f.write("# RQ2: adversarial-scheduler coordination correctness\n" + table + "\n")
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
