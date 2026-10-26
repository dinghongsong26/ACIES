"""Hierarchical-arbiter demo: the RQ4 2^N wall is a FLAT-architecture artifact.

Takes the certified FLAT gen_8r_1res decomposition (single arbiter, |M|=3072) and
replaces the arbiter with a TWO-LEVEL TREE:

    RootArb                 -- arbitrates between two GROUPS (the N=2 justice arbiter)
    GroupArb1 (robots 1-4)  -- arbitrates within its group, gated by the root's grant
    GroupArb2 (robots 5-8)

New internal signals (the coordination interface INVENTED at this level):
    greq_gk   (GroupArbk output) -- "some robot in group k demands the resource"
    ggrant_gk (RootArb output)   -- "group k currently holds the resource"

Discharge stays ACYCLIC exactly as in the flat/terminal family: RootArb is
assumption-free (revocable justice arbiter), GroupArb's only assumption is the root's
fairness (discharged by RootArb's unconditional guarantee), robots' fairness is
discharged by their group's guarantee. Global mutex composes: grant_ri -> ggrant_gk
plus the root's mutex on ggrant.

Certificates: refinement (tree composition vs the ORIGINAL flat System contract,
unchanged) + GR(1) synthesis of every component + well-separatedness gate.

Run:  python experiments/hierarchy_replay.py [--synth]
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

FLAT = ROOT / "outputs/sweep/gen_8r_1res_arbiter_justice__seed0/workdir/_tmp.ltl"
GROUPS = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}


def group_arb(g, robots):
    rs = robots
    lines = [f"CONTRACT GroupArb{g}:"]
    lines += [f"    input boolean variable req_r{i};" for i in rs]
    lines += [f"    input boolean variable ggrant_g{g};"]
    lines += [f"    output boolean variable grant_r{i};" for i in rs]
    lines += [f"    output boolean variable greq_g{g};"]
    lines += ["    Assumptions:",
              f"        G F(!greq_g{g} | ggrant_g{g});"]
    lines += ["    Guarantees:"]
    lines += [f"        !grant_r{i};" for i in rs]
    lines += [f"        !greq_g{g};"]
    # group demand aggregates the robots' demands (the invented upward interface).
    # One-step DELAYED aggregator: an instantaneous iff clashes with the canonical
    # initial pin !greq at t=0 (the environment may already demand at t=0), making
    # the group's game unrealizable.
    ors = " | ".join(f"req_r{i}" for i in rs)
    lines += [f"        G(({ors}) -> X(greq_g{g}));",
              f"        G(!({ors}) -> X(!greq_g{g}));"]
    # a robot grant needs the group to hold the resource (the downward gate)
    lines += [f"        G(grant_r{i} -> ggrant_g{g});" for i in rs]
    # mutex within the group
    for i in rs:
        others = " & ".join(f"!grant_r{j}" for j in rs if j != i)
        lines += [f"        G(grant_r{i} -> ({others}));"]
    lines += [f"        G(grant_r{i} -> req_r{i});" for i in rs]
    lines += [f"        G F(!req_r{i} | grant_r{i});" for i in rs]
    return "\n".join(lines) + "\n\n"


def root_arb():
    lines = ["CONTRACT RootArb:",
             "    input boolean variable greq_g1;",
             "    input boolean variable greq_g2;",
             "    output boolean variable ggrant_g1;",
             "    output boolean variable ggrant_g2;",
             "    Assumptions:",
             "        true;",
             "    Guarantees:",
             "        !ggrant_g1;",
             "        !ggrant_g2;",
             "        G(!(ggrant_g1 & ggrant_g2));",
             "        G(ggrant_g1 -> greq_g1);",
             "        G(ggrant_g2 -> greq_g2);",
             "        G F(!greq_g1 | ggrant_g1);",
             "        G F(!greq_g2 | ggrant_g2);"]
    return "\n".join(lines) + "\n\n"


def build_tree(flat_text):
    parts = re.split(r"(?=^CONTRACT \w+:)", flat_text, flags=re.M)
    out = []
    for p in parts:
        m = re.match(r"CONTRACT (\w+):", p)
        if m and m.group(1) == "Arbiter":
            out.append(group_arb(1, GROUPS[1]))
            out.append(group_arb(2, GROUPS[2]))
            out.append(root_arb())
        else:
            out.append(p)
    return "".join(out)


def main():
    wd = ROOT / "outputs" / "hierarchy_replay"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    content = build_tree(FLAT.read_text())
    (wd / "tree.ltl").write_text(content)

    tp = ToolPipeline(workdir=wd, mission="tree8", ltl_file=wd / "tree.ltl",
                      ver_file=wd / "verification.chase", syn_file=wd / "synthesis.chase")
    ok, out = tp.parse_ltl(content)
    if not ok:
        print("SYNTAX FAIL:\n", out[-600:]); return
    tp.write_chase_command_files(content)
    ok, out = tp.run_verification()
    if not ok:
        print("logics_tool FAIL:\n", out[-600:]); return
    passed, out = tp.run_nuxmv_refinement(wd / "refinement_tree8.smv")
    (wd / "nuxmv.log").write_text(out)
    t = len(re.findall(r"\bis true\b", out))
    f = len(re.findall(r"\bis false\b", out))
    timed = "timeout" in out
    print(f"[tree8] refinement: {t}x true / {f}x false"
          f" -> {'PASS' if passed else 'FAIL'}{' (TIMEOUT — inconclusive!)' if timed else ''}")
    if passed and "--synth" in sys.argv:
        ok, err = tp.run_synthesis()
        if not ok:
            print("synthesis chase FAIL:\n", err[-400:]); return
        import json
        # tree_dock.py loads all eight worker controllers, not only the first one.
        controllers = ["GroupArb1", "GroupArb2", "RootArb"] + [
            f"Robot{i}" for i in range(1, 9)
        ]
        for ctrl in controllers:
            res = tp.synthesize_controller(ctrl)
            n = ""
            if res.get("status") == "ok":
                d = json.load(open(res["json_path"]))
                n = f"  |M|={len(d.get('nodes', d))}"
            print(f"[tree8]   synth {ctrl}: {res['status']}{n}")
        from contrauto.agents.decomposer import DecomposerAgent
        dec = DecomposerAgent(model="gemini", workdir=str(wd), mission="tree8")
        for ctrl in ["GroupArb1", "GroupArb2", "RootArb"]:
            did, _ = dec._degeneracy_witness(wd / f"{ctrl.lower()}.json", content, ctrl)
            print(f"[tree8]   gate {ctrl}: " + ("WELL-SEPARATED" if did is None
                                                else f"DEGENERATE at {did}"))


if __name__ == "__main__":
    main()
