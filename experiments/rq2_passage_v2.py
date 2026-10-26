"""tab:passage REDO with a SERVICE-COUPLED workload (replaces rq2_passage's numbers).

The original harness withdrew requests OPEN-LOOP (after a fixed crossing length,
regardless of being served). That satisfies the world-release justice GF(!req) for
free, so an arbiter that NEVER GRANTS still shows zero mutex/preempt/starvation --
which is exactly how the liveness-vacuous (unprimed-hold) arbiters passed. Here the
demand is CLOSED-LOOP: a robot's request stands until it has been granted for its
full crossing length (service complete), then drops for an idle gap. Under this
workload a never-granting arbiter starves every stream, a terminal arbiter preempts
crossings, and the primed passage arbiter should hold all guarantees.

Policies (N=3):
  passage-primed    outputs/corridor_direct/passage_3r.json      (hold with X(req))
  passage-unprimed  re-synthesized to scratch (the OLD encoding)  (never grants)
  terminal (dock)   outputs/sweep/gen_3r_1res_arbiter_justice__seed0/workdir/arbiter.json

Run:  python experiments/rq2_passage_v2.py [--sample 10000]
"""
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import passage_arbiter as pa
from rq2_passage import FsmArb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 3
H = 60
CMAX = 4          # max crossing length (ticks of held grant needed per service)
GMAX = 6          # max idle gap between episodes
STARVE = 40       # demand standing this long without completion => starved


def synth_unprimed(outdir):
    """Re-synthesize the OLD (unprimed-hold) passage arbiter for the contrast row."""
    os.makedirs(outdir, exist_ok=True)
    spec = pa.passage_arbiter_spec(N)
    for i in range(1, N + 1):
        spec = spec.replace(f"(grant_r{i} & req_r{i} & req_r{i}') -> grant_r{i}'",
                            f"(grant_r{i} & req_r{i}) -> grant_r{i}'")
    ss = os.path.join(outdir, "passage_3r_unprimed.structuredSlugs")
    si = os.path.join(outdir, "passage_3r_unprimed.slugsin")
    jo = os.path.join(outdir, "passage_3r_unprimed.json")
    open(ss, "w").write(spec)
    r1 = subprocess.run(["python3", pa.SLUGS_COMP, ss], capture_output=True, text=True)
    open(si, "w").write(r1.stdout)
    r2 = subprocess.run([pa.SLUGS, "--explicitStrategy", "--jsonOutput", si],
                        capture_output=True, text=True)
    open(jo, "w").write(r2.stdout)
    return jo


def eval_stream(policy, rng):
    """One service-coupled episode stream; returns violation flags for this stream."""
    policy.reset()
    # per-robot episode schedule: (arrival_gap, crossing_length) pairs, adversarial
    next_arrival = [rng.randint(0, GMAX) for _ in range(N)]
    need = [0] * N              # remaining held-grant ticks to complete current service
    demanding = [False] * N
    stand = [0] * N             # how long current demand has been standing
    served_any = [False] * N
    mutex = preempt = starved = 0
    prev_grant = (0,) * N
    grants_total = 0
    for t in range(H):
        for i in range(N):
            if not demanding[i] and t >= next_arrival[i]:
                demanding[i] = True
                need[i] = rng.randint(1, CMAX)
                stand[i] = 0
        reqvec = tuple(1 if demanding[i] else 0 for i in range(N))
        grant = policy.step(reqvec, t)
        grants_total += sum(grant)
        if sum(grant) > 1:
            mutex = 1
        for i in range(N):
            if demanding[i]:
                stand[i] += 1
                if prev_grant[i] and not grant[i]:
                    preempt = 1          # revoked mid-service (demand still standing)
                    need[i] = rng.randint(1, CMAX)   # crossing aborted, must restart
                if grant[i]:
                    need[i] -= 1
                    if need[i] <= 0:     # service complete -> release, schedule next
                        demanding[i] = False
                        served_any[i] = True
                        next_arrival[i] = t + 1 + rng.randint(1, GMAX)
                if stand[i] >= STARVE:
                    starved = 1
        prev_grant = grant
    return {"mutex": mutex, "preempt": preempt, "starved": starved,
            "grants": grants_total}


def main():
    sample = 10000
    if "--sample" in sys.argv:
        sample = int(sys.argv[sys.argv.index("--sample") + 1])
    unprimed = synth_unprimed(os.path.join(ROOT, "outputs/corridor_direct/unprimed_contrast"))
    policies = [
        # each row demonstrates one mechanism from the vacuity/mismatch taxonomy:
        ("primed+biased (ours)", "outputs/corridor_direct/passage_3r.json"),
        ("primed, lazy extraction", "outputs/corridor_direct/unbiased_contrast/passage_3r_primed_unbiased.json"),
        ("unprimed (contract trap)", os.path.relpath(unprimed, ROOT)),
        ("terminal (form mismatch)", "outputs/sweep/gen_3r_1res_arbiter_justice__seed0/workdir/arbiter.json"),
    ]
    print(f"SERVICE-COUPLED passage workload, N={N}, {sample} streams, H={H}:")
    print(f"{'policy':26s} {'mutex':>7s} {'preempt':>8s} {'starved':>8s} {'mean grants/stream':>19s}")
    for name, path in policies:
        p = FsmArb(path, N, name)
        agg = {"mutex": 0, "preempt": 0, "starved": 0, "grants": 0}
        for s in range(sample):
            r = eval_stream(p, random.Random(s))
            for k in ("mutex", "preempt", "starved"):
                agg[k] += r[k]
            agg["grants"] += r["grants"]
        print(f"{name:26s} {agg['mutex']/sample:7.3f} {agg['preempt']/sample:8.3f} "
              f"{agg['starved']/sample:8.3f} {agg['grants']/sample:19.1f}")


if __name__ == "__main__":
    main()
