"""Passage-type (corridor / shared-region) arbiter: synthesize it DIRECTLY.

The dock/charger are TERMINAL resources: the grant gates one operation, and a
memoryless mutex+justice arbiter suffices (its env is unconstrained). A PASSAGE
(a single-lane corridor, a shared region) is different: the grant must be HELD for
the whole crossing. The naive contract `G(crossing -> grant)` + "keep crossing until
you exit" is unrealizable, because the arbiter may revoke the grant mid-crossing.
The counterstrategy from the LLM run showed exactly this: a state with
`crossing = TRUE, grant = FALSE`.

The missing guarantee is a LOCK with HOLD-UNTIL-RELEASE: once the arbiter grants a
requester, it keeps that grant while the request stands, and only reassigns after the
holder releases (drops its request on exit). This is expressible in GR(1) as safety +
justice, no `U`/response:

  SYS (arbiter guarantees):
    mutex               !(grant_i & grant_j)                  i != j
    grant only requester grant_i -> req_i
    HOLD-UNTIL-RELEASE  (grant_i & req_i) -> X grant_i        <-- the key piece
    no starvation       GF(!req_i | grant_i)
  ENV (assumed of the workers/world):
    finite crossing     GF !(req_i & grant_i)                 the holder eventually exits & releases

We generate this arbiter for N robots, compile it, and run SLUGS. A realizable verdict
+ an FSM is the generality result: the SAME synthesis method that handles the terminal
dock/charger also produces a correct passage arbiter, once the contract carries the
hold-until-release guarantee.

Run:  python experiments/passage_arbiter.py [--n 2 3 4] [--outdir outputs/corridor_direct]
"""
import os
import sys
import subprocess
import argparse
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUGS = os.environ.get("SLUGS", "slugs")
SLUGS_COMP = os.environ.get("SLUGS_COMP", "compiler.py")  # StructuredSlugsParser/compiler.py from the slugs repo


def passage_arbiter_spec(n):
    """Structured-SLUGS text for the N-robot passage (hold-until-release) arbiter."""
    req = [f"req_r{i+1}" for i in range(n)]
    grant = [f"grant_r{i+1}" for i in range(n)]
    L = []
    L += ["[INPUT]"] + req + [""]
    L += ["[OUTPUT]"] + grant + [""]
    L += ["[ENV_INIT]"] + [f"!{r}" for r in req] + [""]
    L += ["[SYS_INIT]"] + [f"!{g}" for g in grant] + [""]
    L += ["[ENV_TRANS]", ""]                                   # requests are unconstrained in time
    sys_trans = []
    # mutual exclusion: at most one grant at a time (all distinct pairs)
    for i in range(n):
        for j in range(i + 1, n):
            sys_trans.append(f"!({grant[i]} & {grant[j]})")
    for i in range(n):
        sys_trans.append(f"{grant[i]} -> {req[i]}")           # never grant a non-requester
    for i in range(n):
        # HOLD-UNTIL-RELEASE, with the request PRIMED: hold only while the demand
        # persists into the next step. The unprimed form ((grant & req) -> grant')
        # forces the grant one step past the demand's withdrawal, which together with
        # the no-spurious rule (grant -> req) makes every granting state LOSING -- the
        # synthesized strategy then never grants and wins its fairness goal vacuously
        # through the world's release assumption (discovered 2026-07-10 by the
        # physical end-to-end run; see outputs/corridor_e2e and simulator/corridor.py).
        sys_trans.append(f"({grant[i]} & {req[i]} & {req[i]}') -> {grant[i]}'")
    L += ["[SYS_TRANS]"] + sys_trans + [""]
    # env liveness: every holder eventually releases its request (crossings are finite), so a
    # held lock rotates. GF !req_i is the clean assumption that makes hold-until-release + no
    # starvation jointly realizable.
    L += ["[ENV_LIVENESS]"] + [f"!{req[i]}" for i in range(n)] + [""]
    # sys liveness: no starvation -- a persistent requester is eventually granted
    L += ["[SYS_LIVENESS]"] + [f"!{req[i]} | {grant[i]}" for i in range(n)] + [""]
    return "\n".join(L)


def synth(n, outdir):
    os.makedirs(outdir, exist_ok=True)
    ss = os.path.join(outdir, f"passage_{n}r.structuredSlugs")
    si = os.path.join(outdir, f"passage_{n}r.slugsin")
    jo = os.path.join(outdir, f"passage_{n}r.json")
    with open(ss, "w") as f:
        f.write(passage_arbiter_spec(n))

    r1 = subprocess.run(["python3", SLUGS_COMP, ss], capture_output=True, text=True, timeout=120)
    if r1.returncode != 0 or not r1.stdout.strip():
        return {"n": n, "status": "compile_error", "detail": (r1.stderr or r1.stdout)[:400]}
    with open(si, "w") as f:
        f.write(r1.stdout)

    # synthesize (same invocation as the production pipeline): --explicitStrategy --jsonOutput.
    # SLUGS prints the realizability verdict on stderr and the strategy JSON on stdout.
    rr = subprocess.run([SLUGS, "--explicitStrategy", "--biasForAction", "--jsonOutput", si],
                        capture_output=True, text=True, timeout=300)
    combined = (rr.stdout + rr.stderr).lower()
    verdict_line = next((ln for ln in (rr.stderr + rr.stdout).splitlines()
                         if "realizable" in ln.lower()), "").strip()
    if "unrealizable" in combined:
        return {"n": n, "status": "unrealizable", "spec": ss, "slugs_verdict": verdict_line}
    if "error" in rr.stderr.lower():
        return {"n": n, "status": "error", "spec": ss, "detail": rr.stderr[:400]}

    result = {"n": n, "status": "realizable", "spec": ss, "slugs_verdict": verdict_line}
    out = rr.stdout
    if "{" in out:
        data = json.loads(out[out.index("{"):])
        with open(jo, "w") as f:
            json.dump(data, f)
        result["fsm"] = jo
        result["fsm_states"] = len(data.get("nodes", {}))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs/corridor_direct"))
    a = ap.parse_args()
    results = []
    for n in a.n:
        print(f"[synth] passage arbiter N={n} ...", flush=True)
        res = synth(n, a.outdir)
        print(f"   -> {res['status']}"
              + (f", FSM {res.get('fsm_states','?')} states" if res.get("fsm") else ""), flush=True)
        results.append(res)
    with open(os.path.join(a.outdir, "passage_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== summary ===")
    for r in results:
        print(f"  N={r['n']}: {r['status']}"
              + (f"  |M|={r.get('fsm_states')}" if r.get("fsm") else ""))
    print(f"[saved] {a.outdir}/passage_results.json")


if __name__ == "__main__":
    main()
