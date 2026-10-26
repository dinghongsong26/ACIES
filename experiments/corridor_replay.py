"""Offline replay for the corridor refinement failure -- no LLM calls.

Takes the failed run's final contracts (outputs/corridor/workdir/_tmp.ltl), applies a
named variant (a block-scoped text transformation), and re-runs the local certificate
chain: logics_tool (compose + refinement smv) -> nuXmv IC3. Reports true/false counts
per LTLSPEC (2x true = refinement PASS).

Diagnosis being tested: the baseline fails on a CIRCULAR LIVENESS discharge --
  Robot assumes  GF(!req | grant)   (fairness, to guarantee eventual release GF(!req))
  Arbiter assumes GF(!req)          (eventual release, to guarantee fairness)
Non-circular refinement rightly rejects the cycle. V1 breaks it by making release
SAFETY-discharged: arbiter assumes G(grant -> X !req) (one-step service), robots
guarantee use-on-grant G(grant -> crossing | at_b) so physics (crossing -> X at_b)
plus G(at_b -> !req) closes the chain without any liveness assumption.

Run:  python experiments/corridor_replay.py [variant ...]
"""
import os
import re
import shutil
import sys
from pathlib import Path

_ANTLR = os.environ.get("ANTLR4_LIB", "")
if _ANTLR:
    os.environ["DYLD_LIBRARY_PATH"] = _ANTLR + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contrauto.agents.tool_pipeline import ToolPipeline  # noqa: E402

BASE = (ROOT / "outputs/corridor/workdir/_tmp.ltl").read_text()


def _blocks(text):
    """Split the .ltl into (name, block_text) preserving order."""
    parts = re.split(r"(?=^CONTRACT \w+:)", text, flags=re.M)
    out = []
    for p in parts:
        m = re.match(r"CONTRACT (\w+):", p)
        out.append((m.group(1) if m else "_header", p))
    return out


def _rejoin(blocks):
    return "".join(b for _, b in blocks)


def edit_block(text, contract, fn):
    """Apply fn to the named contract's block only."""
    bs = _blocks(text)
    return _rejoin([(n, fn(b) if n == contract else b) for n, b in bs])


# ── variants ──────────────────────────────────────────────────────────────────
def v_baseline(t):
    return t


def v1_safety_release(t):
    # Arbiter: liveness release assumption -> one-step safety release
    def arb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        G F(!req_r{i});\n",
                          f"        G(grant_r{i} -> X(!req_r{i}));\n")
        return b
    t = edit_block(t, "Arbiter", arb)
    # Robots: guarantee use-on-grant; drop the over-strong "never crossing while at_a"
    for i in (1, 2, 3):
        def rob(b, i=i):
            b = b.replace(f"        G(!(r{i}_at_a & r{i}_crossing));\n", "")
            b = b.replace(f"        G(r{i}_crossing -> grant_r{i});\n",
                          f"        G(r{i}_crossing -> grant_r{i});\n"
                          f"        G(grant_r{i} -> r{i}_crossing | r{i}_at_b);\n")
            return b
        t = edit_block(t, f"Robot{i}", rob)
    # System: drop the matching over-strong guarantee
    def sysb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        G(!(r{i}_at_a & r{i}_crossing));\n", "")
        return b
    t = edit_block(t, "System", sysb)
    return t


def v2_min_system(t):
    """Diagnostic: shrink System Guarantees to ONLY init pins + grant mutex.
    If even this fails, the blocker is the discharge structure, not any single
    over-strong system guarantee."""
    def sysb(b):
        head, _, _ = b.partition("    Guarantees:")
        keep = [
            "    Guarantees:",
            "        !grant_r1;", "        !grant_r2;", "        !grant_r3;",
            "        G(!(grant_r1 & grant_r2));",
            "        G(!(grant_r1 & grant_r3));",
            "        G(!(grant_r2 & grant_r3));",
        ]
        return head + "\n".join(keep) + "\n"
    return edit_block(t, "System", sysb)


def v5_arbiter_observes_atb(t):
    """ARCHITECTURE fix: the arbiter observes each robot's completion signal at_b.
    All cross-contract 'eventually' promises become per-step safety:
      arbiter A:  G(grant_i & !at_b_i -> X at_b_i)   (granted robot progresses -- discharged
                  by robot's use-on-grant G + the physics crossing -> X at_b)
      arbiter G:  hold keyed on at_b (matches the System contract's own hold rule),
                  no-spurious dropped (System never demanded it), fairness kept.
      robot A:    drop both grant-persistence assumptions (no longer needed: crossing
                  lasts one step and needs grant only at that step).
      robot G:    add use-on-grant; drop the dubious 'never crossing while at_a'
                  (crossing naturally STARTS at side A) -- also dropped from System."""
    def arb(b):
        b = b.replace("    input boolean variable req_r1;",
                      "    input boolean variable req_r1;\n"
                      "    input boolean variable r1_at_b;\n"
                      "    input boolean variable r2_at_b;\n"
                      "    input boolean variable r3_at_b;")
        for i in (1, 2, 3):
            b = b.replace(f"        G F(!req_r{i});\n",
                          f"        G(grant_r{i} & !r{i}_at_b -> X(r{i}_at_b));\n")
            b = b.replace(f"        G(grant_r{i} -> req_r{i});\n", "")
            b = b.replace(f"        G(grant_r{i} & req_r{i} -> X(grant_r{i}));\n",
                          f"        G(grant_r{i} & !r{i}_at_b -> X(grant_r{i}));\n")
        return b
    t = edit_block(t, "Arbiter", arb)
    for i in (1, 2, 3):
        def rob(b, i=i):
            b = b.replace(f"        G(grant_r{i} & req_r{i} -> X(grant_r{i}));\n", "")
            b = b.replace(f"        G(grant_r{i} -> req_r{i});\n", "")
            b = b.replace(f"        G(!(r{i}_at_a & r{i}_crossing));\n", "")
            b = b.replace(f"        G(r{i}_crossing -> grant_r{i});\n",
                          f"        G(r{i}_crossing -> grant_r{i});\n"
                          f"        G(grant_r{i} & !r{i}_at_b -> r{i}_crossing);\n")
            return b
        t = edit_block(t, f"Robot{i}", rob)
    def sysb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        G(!(r{i}_at_a & r{i}_crossing));\n", "")
        return b
    return edit_block(t, "System", sysb)


def _drop_robot_internal_assumptions(t):
    for i in (1, 2, 3):
        def rob(b, i=i):
            b = b.replace(f"        G(grant_r{i} & req_r{i} -> X(grant_r{i}));\n", "")
            b = b.replace(f"        G(grant_r{i} -> req_r{i});\n", "")
            return b
        t = edit_block(t, f"Robot{i}", rob)
    return t


def _drop_system_hold(t):
    def sysb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        G(grant_r{i} & !r{i}_at_b -> X(grant_r{i}));\n", "")
        return b
    return edit_block(t, "System", sysb)


def _drop_arbiter_hold(t):
    def arb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        G(grant_r{i} & req_r{i} -> X(grant_r{i}));\n", "")
        return b
    return edit_block(t, "Arbiter", arb)


def _drop_arbiter_assumptions(t):
    def arb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        G F(!req_r{i});\n", "")
        return b.replace("    Assumptions:\n    Guarantees:",
                         "    Assumptions:\n        true;\n    Guarantees:")
    return edit_block(t, "Arbiter", arb)


# Ladder from dock-like structure (L0, expected PASS) back to baseline (fails):
def l0_dockified(t):
    """No hold anywhere, assumption-free arbiter, no robot internal assumptions."""
    return _drop_system_hold(_drop_arbiter_hold(_drop_arbiter_assumptions(
        _drop_robot_internal_assumptions(t))))


def l1_plus_arbiter_hold(t):
    """L0 + arbiter hold rules back (still assumption-free)."""
    return _drop_system_hold(_drop_arbiter_assumptions(
        _drop_robot_internal_assumptions(t)))


def l2_plus_arbiter_assumption(t):
    """L1 + arbiter liveness assumptions GF(!req) back."""
    return _drop_system_hold(_drop_robot_internal_assumptions(t))


def l3_plus_system_hold(t):
    """L2 + System hold back = baseline minus robot internal assumptions."""
    return _drop_robot_internal_assumptions(t)


def l1t_tautology_assumption(t):
    """L1 + a TAUTOLOGICAL arbiter assumption. Discriminates 'any nonempty
    Assumptions section breaks the encoding' (tool quirk) from 'undischargeable
    assumption content' (semantic limitation)."""
    t = l1_plus_arbiter_hold(t)
    def arb(b):
        return b.replace("    Assumptions:\n        true;\n",
                         "    Assumptions:\n        G(grant_r1 | !grant_r1);\n")
    return edit_block(t, "Arbiter", arb)


def v7_fold_arbiter(t):
    """Semantics-preserving fold of the ARBITER contract: [A, G] -> [true, A -> G].
    The implementation sets of the two forms are identical (saturation), but the
    refinement encoding no longer exports the arbiter's assumption into the
    composed assumption, so the discharge chain terminates at the arbiter exactly
    as it does for the (assumption-free) terminal arbiter of the gen family."""
    def arb(b):
        head, _, rest = b.partition("    Assumptions:")
        a_part, _, g_part = rest.partition("    Guarantees:")
        a_lines = [l.strip().rstrip(";") for l in a_part.splitlines() if l.strip().endswith(";")]
        g_lines = [l.strip().rstrip(";") for l in g_part.splitlines() if l.strip().endswith(";")]
        a_expr = " & ".join(f"({a})" for a in a_lines) or "true"
        g_expr = " & ".join(f"({g})" for g in g_lines)
        return (head + "    Assumptions:\n        true;\n    Guarantees:\n"
                f"        ({a_expr}) -> ({g_expr});\n")
    return edit_block(t, "Arbiter", arb)


def v9_req_owned_by_world(t):
    """ARCHITECTURE fix at the OWNERSHIP level: req_i becomes a WORLD signal (the
    task demands the crossing; the robot does not choose to request). Release is
    then world behaviour in A_sys -- exactly what corridor.yaml's 'World behaviour'
    clause intended -- and every discharge chain terminates at A_sys (acyclic):
      robot fairness <- arbiter G;  arbiter's GF(!req) <- A_sys directly."""
    for i in (1, 2, 3):
        def rob(b, i=i):
            b = b.replace(f"    output boolean variable req_r{i};",
                          f"    input boolean variable req_r{i};")
            # req-management guarantees move out of the robot entirely
            for line in (f"        !req_r{i};\n",
                         f"        G(req_r{i} & !grant_r{i} & !r{i}_at_b -> X(req_r{i}));\n",
                         f"        G(grant_r{i} & !r{i}_at_b -> req_r{i});\n",
                         f"        G(r{i}_at_b -> !req_r{i});\n",
                         f"        G F(!req_r{i});\n"):
                b = b.replace(line, "")
            # world provides: initial no-request + eventual release
            b = b.replace("    Assumptions:\n",
                          f"    Assumptions:\n        !req_r{i};\n        G F(!req_r{i});\n")
            return b
        t = edit_block(t, f"Robot{i}", rob)
    def sysb(b):
        for i in (1, 2, 3):
            b = b.replace(f"    output boolean variable req_r{i};",
                          f"    input boolean variable req_r{i};")
            b = b.replace(f"        !req_r{i};\n        !req_r{i};\n", f"        !req_r{i};\n")
        # move req init pins + release to System ASSUMPTIONS; drop from guarantees
        for i in (1, 2, 3):
            b = b.replace("    Assumptions:\n",
                          f"    Assumptions:\n        !req_r{i};\n        G F(!req_r{i});\n")
        head, _, rest = b.partition("    Guarantees:")
        for i in (1, 2, 3):
            rest = rest.replace(f"        !req_r{i};\n", "")
        return head + "    Guarantees:" + rest
    return edit_block(t, "System", sysb)


def v10_safety_workers(t):
    """v9 + demand-side liveness removed from the workers (and System): with the
    world owning req (blanket release GF(!req)), an UNCONDITIONAL 'cross infinitely
    often' goal is unrealizable (a world that never demands is legal). The certifiable
    shape is exactly the hand-authored design: safety-only workers, all liveness
    carried by the arbiter's fairness under the world's release assumption."""
    t = v9_req_owned_by_world(t)
    for i in (1, 2, 3):
        def rob(b, i=i):
            b = b.replace(f"        G F(r{i}_at_b);\n", "")
            b = b.replace(f"        G F(r{i}_crossing);\n", "")
            return b
        t = edit_block(t, f"Robot{i}", rob)
    def sysb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        G F(r{i}_at_b);\n", "")
            b = b.replace(f"        G F(r{i}_crossing);\n", "")
        return b
    return edit_block(t, "System", sysb)


def v11_drop_crossing_atb_clash(t):
    """v10 + drop G(!(crossing & at_b)) from robots and System: physics forces at_b
    one step after crossing, so the prohibition clashes with Moore timing and drives
    the strategy into a degenerate dead-end (caught by the well-sep gate on v10)."""
    t = v10_safety_workers(t)
    for i in (1, 2, 3):
        def rob(b, i=i):
            return b.replace(f"        G(!(r{i}_crossing & r{i}_at_b));\n", "")
        t = edit_block(t, f"Robot{i}", rob)
    def sysb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        G(!(r{i}_crossing & r{i}_at_b));\n", "")
        return b
    return edit_block(t, "System", sysb)


def v12_drop_until(t):
    """v11 + drop the U formulas: their safety content is already covered by
    G(crossing -> grant); the residual eventuality ('grant eventually comes') is the
    arbiter's fairness job, not the robot's -- and U is not GR(1) anyway (the task
    yaml itself says 'no until'). Also fixes the gate false-positive: the gate's
    liveness filter skips 'G F' but not the eventuality inside U."""
    t = v11_drop_crossing_atb_clash(t)
    for i in (1, 2, 3):
        def rob(b, i=i):
            return b.replace(f"        ((!r{i}_crossing) U grant_r{i});\n", "")
        t = edit_block(t, f"Robot{i}", rob)
    def sysb(b):
        for i in (1, 2, 3):
            b = b.replace(f"        ((!r{i}_crossing) U grant_r{i});\n", "")
        return b
    return edit_block(t, "System", sysb)


VARIANTS = {"baseline": v_baseline, "v1": v1_safety_release, "v2min": v2_min_system,
            "v5": v5_arbiter_observes_atb,
            "l0": l0_dockified, "l1": l1_plus_arbiter_hold,
            "l2": l2_plus_arbiter_assumption, "l3": l3_plus_system_hold,
            "l1t": l1t_tautology_assumption, "v7": v7_fold_arbiter,
            "v9": v9_req_owned_by_world, "v10": v10_safety_workers,
            "v11": v11_drop_crossing_atb_clash, "v12": v12_drop_until}


def run_variant(name):
    fn = VARIANTS[name]
    wd = ROOT / "outputs" / "corridor_replay" / name
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    content = fn(BASE)
    (wd / "variant.ltl").write_text(content)

    tp = ToolPipeline(workdir=wd, mission=f"replay_{name}",
                      ltl_file=wd / "variant.ltl",
                      ver_file=wd / "verification.chase",
                      syn_file=wd / "synthesis.chase")
    ok, out = tp.parse_ltl(content)
    if not ok:
        print(f"[{name}] SYNTAX FAIL:\n{out[-600:]}")
        return
    # parse_ltl overwrote _tmp.ltl with content; point logics_tool at the variant
    tp.write_chase_command_files(content)
    ok, out = tp.run_verification()
    if not ok:
        print(f"[{name}] logics_tool FAIL:\n{out[-600:]}")
        return
    smv = wd / f"refinement_replay_{name}.smv"
    passed, out = tp.run_nuxmv_refinement(smv)
    (wd / "nuxmv.log").write_text(out)
    t = len(re.findall(r"\bis true\b", out))
    f = len(re.findall(r"\bis false\b", out))
    print(f"[{name}] refinement: {t}x true / {f}x false  -> {'PASS' if passed else 'FAIL'}"
          f"   (full output: {wd / 'nuxmv.log'})")
    if passed and "--synth" in sys.argv:
        ok, err = tp.run_synthesis()
        if not ok:
            print(f"[{name}] synthesis chase FAIL:\n{err[-400:]}")
            return
        import json as _json
        for ctrl in re.findall(r"CONTRACT (\w+):", content):
            if ctrl.lower() == "system":
                continue
            res = tp.synthesize_controller(ctrl)
            n_states = ""
            if res.get("status") == "ok":
                try:
                    d = _json.load(open(res["json_path"]))
                    nodes = d.get("nodes") or d.get("states") or d
                    n_states = f"  |M|={len(nodes)}"
                except Exception:
                    pass
            print(f"[{name}]   synth {ctrl}: {res['status']}{n_states}"
                  + (f"  ({res.get('detail','')[:120]})" if res.get("status") == "error" else ""))


if __name__ == "__main__":
    names = sys.argv[1:] or ["baseline", "v1"]
    for n in names:
        run_variant(n)
