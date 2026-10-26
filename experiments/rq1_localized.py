"""RQ1 -- localized vs generic feedback, a CONTROLLED ablation.

The earlier benchmark's ``--no-feedback`` was not a clean baseline: it only dropped the
QuickXplain core and assumption-mining blocks while STILL sending the environment
counterstrategy, the well-separatedness dead-end trace, the refinement counterexample,
and the current .ltl to the LLM. So the "no feedback" arm still received most of the
localization, and WITH vs WITHOUT showed no difference.

This harness uses the new ``localized_feedback`` master switch (contrauto/agents/
decomposer.py). With it OFF, every failure channel returns ONLY a generic hint that
names the failing stage ("synthesis is unrealizable, revise") and withholds the
counterexample / core / mined assumption / violated formula. That is the true
no-localization baseline.

To isolate the decomposer (the RQ1 variable), we call ``DecomposerAgent.run`` directly
with a FIXED translator output per task, so the only thing that varies between arms is
the feedback string. We run each (task, arm) several times (LLM decoding is stochastic)
and report repair rounds to a certified decomposition and the success rate. Each run's
full step_logs + per-round contract text are saved so a localized repair can be shown
as a transcript (which formula the hint named, and what the LLM changed next round).

Run:  python experiments/rq1_localized.py --tasks patrol warehouse --runs 3
"""
import os
import sys
import json
import time
import argparse

# macOS SIP strips DYLD_* from the env of SIP-protected launchers (nohup, /bin/sh), so
# logics_tool (CHASE) cannot find libantlr4-runtime when launched through them. Setting it
# IN-PROCESS here guarantees our python has it, and subprocess.run(logics_tool) — a user
# binary, not stripped — then loads the dylib. Must precede any contrauto import.
_ANTLR = os.environ.get("ANTLR4_LIB", "")
if _ANTLR:
    os.environ["DYLD_LIBRARY_PATH"] = _ANTLR + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrauto.schema import TaskSpec                          # noqa: E402
from contrauto.agents.decomposer import DecomposerAgent        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fixed translator output per task (any prior run's cache -- it is identical across arms) and the
# task-yaml path. Small hand-written tasks have 1-2 contracts (localization has little to
# disambiguate); the gen_* family is LARGE (3-4 robots + arbiter, 12+ formulas, many contracts),
# where a generic "it failed" hint should leave the LLM guessing while localization names the culprit.
TRANS = {
    "patrol": ("benchmark_outputs/patrol_gemini_WITH_FEEDBACK_run1/translator_output.json",
               "contrauto/tasks/patrol.yaml"),
    # "warehouse" was renamed from "multi_robot" (contrauto/tasks/multi_robot.yaml ->
    # warehouse.yaml); this benchmark_outputs path is the multi_robot-labeled cache,
    # which matches this task's actual (multi-robot) architecture -- the old single-robot
    # "warehouse" cache this key used to point to does not.
    "warehouse": ("benchmark_outputs/multi_robot_gemini_WITH_FEEDBACK_run1/translator_output.json",
                  "contrauto/tasks/warehouse.yaml"),
    "patrol": ("benchmark_outputs/patrol_gemini_NO_FEEDBACK_run1/translator_output.json",
               "contrauto/tasks/patrol.yaml"),
    "static_2r_3b": ("benchmark_outputs/static_2r_3b_gemini_NO_FEEDBACK_run1/translator_output.json",
                     "contrauto/tasks/static_2r_3b.yaml"),
    "dynamic_2r_3b": ("benchmark_outputs/dynamic_2r_3b_gemini_NO_FEEDBACK_run1/translator_output.json",
                      "contrauto/tasks/dynamic_2r_3b.yaml"),
    "gen_3r_1res": ("outputs/sweep/gen_3r_1res_arbiter_justice__seed1/translator_output.json",
                    "contrauto/tasks/generated/gen_3r_1res_arbiter_justice.yaml"),
    "gen_4r_1res": ("outputs/sweep/gen_4r_1res_arbiter_justice__seed2/translator_output.json",
                    "contrauto/tasks/generated/gen_4r_1res_arbiter_justice.yaml"),
    "gen_3r_2res": ("outputs/sweep/gen_3r_2res_arbiter_justice__seed0/translator_output.json",
                    "contrauto/tasks/generated/gen_3r_2res_arbiter_justice.yaml"),
    "gen_5r_1res": ("outputs/sweep/gen_5r_1res_arbiter_justice__seed1/translator_output.json",
                    "contrauto/tasks/generated/gen_5r_1res_arbiter_justice.yaml"),
    "gen_4r_2res": ("outputs/sweep/gen_4r_2res_arbiter_justice__seed0/translator_output.json",
                    "contrauto/tasks/generated/gen_4r_2res_arbiter_justice.yaml"),
    "gen_6r_1res": ("outputs/sweep/gen_6r_1res_arbiter_justice__seed0/translator_output.json",
                    "contrauto/tasks/generated/gen_6r_1res_arbiter_justice.yaml"),
}


# each arm is a per-stage feedback_modes config. Three feedback modes per stage: generic (no
# diagnosis) / crisp (named core/formula pointer, no trace) / verbose (pointer + full trace).
STAGES = ['refinement', 'consistency', 'unrealizable', 'wellsep']


def _all(mode):
    return {s: mode for s in STAGES}


def _one(stage, mode):
    d = _all('verbose'); d[stage] = mode; return d


# global arms (all stages same) + per-stage ablation arms (one stage varied, others held verbose)
ARMS = {"generic": _all('generic'), "crisp": _all('crisp'), "verbose": _all('verbose')}
for _s in STAGES:
    ARMS[f"{_s}:generic"] = _one(_s, 'generic')
    ARMS[f"{_s}:crisp"] = _one(_s, 'crisp')
# composed best-per-stage config (from the per-stage ablation): refinement wants the terse hint
# (its verbose counterexample trace overwhelms a weak model), unrealizable wants the full trace
# (its core + counterstrategy focuses the fix), well-sep wants the crisp named guarantee.
ARMS["best"] = {"refinement": "generic", "unrealizable": "verbose",
                "wellsep": "crisp", "consistency": "crisp"}


def run_arm(task, tro, arm, model, workdir, enforce):
    dec = DecomposerAgent(model=model, workdir=workdir, mission=task.name,
                          enforce_initial_conditions=enforce, feedback_modes=ARMS[arm])
    t0 = time.time()
    try:
        out = dec.run(tro, task)
        err = None
    except Exception as e:
        out, err = {}, f"{type(e).__name__}: {e}"
    return {
        "success": bool(out.get("fsm_files")),
        "rounds": len(out.get("round_contracts", [])),      # one LLM (re)generation per round
        "elapsed_sec": round(time.time() - t0, 1),
        "error": err,
        "step_logs": out.get("step_logs", []),
        "round_contracts": out.get("round_contracts", []),
    }


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["gen_3r_1res", "gen_3r_2res"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=["generic", "crisp", "verbose"],
                    choices=list(ARMS))
    ap.add_argument("--model", default="gemini")
    ap.add_argument("--enforce", dest="enforce", action="store_true", default=False,
                    help="keep deterministic initial-condition enforce ON (default OFF, so tasks "
                         "enter the repair loop and feedback granularity can be measured)")
    ap.add_argument("--out", default=f"{ROOT}/experiments/rq1_localized_results.json")
    a = ap.parse_args()

    results = {}
    for tname in a.tasks:
        tro_rel, yaml_rel = TRANS[tname]
        tro = json.load(open(os.path.join(ROOT, tro_rel)))
        task = TaskSpec.from_yaml(os.path.join(ROOT, yaml_rel))
        results[tname] = {arm: [] for arm in a.arms}
        for arm in a.arms:
            for r in range(a.runs):
                wd = os.path.join(ROOT, "outputs/rq1", f"{tname}_{arm}_run{r+1}", "workdir")
                print(f"\n===== {tname} | {arm} | run {r+1}/{a.runs} "
                      f"(enforce={a.enforce}) =====", flush=True)
                res = run_arm(task, tro, arm, a.model, wd, a.enforce)
                results[tname][arm].append(res)
                print(f"  -> success={res['success']} rounds={res['rounds']} "
                      f"({res['elapsed_sec']}s){' ERR '+res['error'] if res['error'] else ''}",
                      flush=True)
                # persist incrementally (LLM runs are expensive -- do not lose progress)
                json.dump(results, open(a.out, "w"), indent=2)

    # summary
    print("\n\n================ RQ1 summary ================")
    print(f"{'task':16s} {'arm':10s} {'succ':>6s} {'med.rounds':>11s} {'rounds':>16s}")
    for tname, arms in results.items():
        for arm in a.arms:
            runs = arms[arm]
            succ = sum(1 for x in runs if x["success"])
            rounds = [x["rounds"] for x in runs if x["success"]]
            allr = [x["rounds"] for x in runs]
            print(f"{tname:16s} {arm:10s} {succ:>3d}/{len(runs):<2d} "
                  f"{str(median(rounds)):>11s} {str(allr):>16s}")
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
