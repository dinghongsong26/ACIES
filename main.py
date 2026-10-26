"""ContrAuto entry point — the two-layer orchestrator / coordinator.

    python main.py --task <name> [--no-sim] [--resynth] [--model gemini]
                   [--no-feedback] [--skip-translator] [--save-attempts]

Decides, per task, which layers to run (a coordinator):
  LAYER 1 · TASK (discrete):    reuse synthesized FSMs if present, else synthesize them
            (translator → decomposer → CHASE checks → SLUGS FSM → well-sep gate).
  LAYER 2 · PHYSICAL (continuous): execute in the matching simulator (deadline / mutex /
            delivery), reusing the task FSMs via naming-agnostic binding.

`--no-sim` runs the task (synthesis) layer only; the synthesis-debug flags
(`--no-feedback`, `--skip-translator`, `--save-attempts`) tune that layer.

The two layers stay decoupled — the physical layer only consumes the FSM json files, and can
reuse one synthesized FSM under different maps / instance counts. This script is the thin
coordinator that runs them together for a task and prints one coherent two-layer report.
"""
import argparse
import glob
import os
import shutil

from contrauto import report
from contrauto.pipeline import ContrAuto
from contrauto.schema import TaskSpec
from simulator.registry import SIMULATORS, fsm_files, has_fsm


from contrauto.llm import MODEL_ALIASES
MODELS = sorted(MODEL_ALIASES)


def _known_tasks():
    """Scan contrauto/tasks/*.yaml (excluding generated/) for known task names."""
    root = os.path.dirname(os.path.abspath(__file__))
    yamls = glob.glob(os.path.join(root, "contrauto", "tasks", "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(y))[0] for y in yamls)


TASKS = _known_tasks()  # currently: charger | corridor | warehouse (scan contrauto/tasks/*.yaml for the live set)

# python main.py --task charger --n 3 --model gpt
# python main.py --task corridor --n 3 --model gpt
# python main.py --task warehouse --n 2 --model gpt


def run(task, model="gemini", simulate=True, resynth=False, outdir=None,
        use_dfa_feedback=True, skip_translator=False, save_attempts=False,
    enable_completion=True, visualize=True, n_robots=None, feedback_modes=None,
    retranslate=False):
    """Run both layers for one task: synthesize/reuse FSMs, then execute in the simulator.

    Channel 5 (run-time assumption conformance): if the simulator's assumption monitor reports the
    environment could NOT discharge a contract assumption during execution, that feedback is fed
    back to the decomposer for ONE task-level re-decomposition (physical_feedback)."""
    name = task.name if isinstance(task, TaskSpec) else os.path.splitext(os.path.basename(task))[0]
    outdir = outdir or f"outputs/{name}" + (f"_n{n_robots}" if n_robots is not None else "")
    fsm_dir = f"{outdir}/workdir"

    physical_feedback = ""
    prev_snapshot = None
    for task_attempt in range(2):          # +1 re-decompose on a channel-5 assumption violation
        # ── LAYER 1 · TASK (discrete logic → FSM) ─────────────────────────────
        report.layer(1, f"TASK · {name} · Contracts → FSM")
        force = resynth or task_attempt > 0        # a re-decompose always re-synthesizes
        if (not force) and has_fsm(name, fsm_dir, n_robots):
            report.step(True, "FSM reuse", f"synthesized FSMs found in {fsm_dir} — skipping synthesis")
        else:
            report.info("synthesize", "NL → LTL → Contracts → consistency & compatibility → refinement "
                        "→ FSM synthesis")
            # Intermediate synthesis rounds can leave JSON artifacts behind even when the
            # retry budget is exhausted. They are not certified controllers and must not be
            # mistaken for a fresh successful synthesis below.
            for filename in fsm_files(name, n_robots):
                path = os.path.join(fsm_dir, filename)
                if os.path.exists(path):
                    os.remove(path)
            # --no-feedback is the FULL harness-off baseline: it must also drop the master
            # localization switch, else Step 5's core/mined-assumption feedback still reaches
            # the LLM and the ablation only measures the Step-2.5/4 diagnosers.
            result = ContrAuto(task=task, model=model, outdir=outdir, use_dfa_feedback=use_dfa_feedback,
                               localized_feedback=use_dfa_feedback,
                               enable_completion=enable_completion,
                               skip_translator=skip_translator, retranslate=retranslate,
                               save_attempts=save_attempts,
                               physical_feedback=physical_feedback,
                               feedback_modes=feedback_modes).run()
            synthesized = result.get("success", False) and has_fsm(name, fsm_dir, n_robots)
            report.step(synthesized, "synthesis",
                        "FSM produced" if synthesized else "no certified FSM produced")
            if not synthesized:
                # FSM JSON left by a failed round (e.g. a degenerate strategy) is not a
                # certified controller -- never hand it to the physical layer.
                report.info("physical layer skipped", "(synthesis failed)")
                return

        # ── LAYER 2 · PHYSICAL (continuous execution) ─────────────────────────
        if not simulate:
            report.info("physical layer skipped", "(--no-sim)")
            return
        if name not in SIMULATORS:
            report.info(f"no simulator registered for '{name}'", "— task layer only")
            return
        if not has_fsm(name, fsm_dir, n_robots):
            report.step(False, "physical layer", "no FSM to execute")
            return

        report.layer(2, f"TASK · {name} · PHYSICAL · execute FSM under CBF + deadline")
        adapter = SIMULATORS[name][0]
        adapter_kwargs = {"visualize": visualize}
        if n_robots is not None:
            adapter_kwargs["n_robots"] = n_robots
        if visualize:
            report.info("physical visualization", f"exporting GIF to {outdir}/physical_simulation.gif")
        m = adapter(fsm_dir, **adapter_kwargs)
        progress = m.get("delivered", 0) + m.get("cleared", 0)   # delivery- or patrol-style success
        ok = progress > 0 and m.get("violations", 0) == 0
        report.summary(ok, "PHYSICAL EXECUTION", m)

        # keep-better: a channel-5 re-decomposition is not guaranteed to improve (observed:
        # 6 delivered -> 0). If this is the re-decomposed run and it is WORSE, restore the
        # pre-ch5 FSMs and report the kept controller's (measured) metrics as final.
        if task_attempt == 1 and prev_snapshot is not None:
            prev_m, prev_dir = prev_snapshot
            prev_progress = prev_m.get("delivered", 0) + prev_m.get("cleared", 0)
            if progress < prev_progress:
                for f in os.listdir(prev_dir):
                    shutil.copy2(os.path.join(prev_dir, f), os.path.join(fsm_dir, f))
                report.info("keep-better", f"re-decomposed controller regressed "
                            f"({progress} < {prev_progress}) — restored pre-ch5 FSMs")
                ok = prev_progress > 0 and prev_m.get("violations", 0) == 0
                report.summary(ok, "PHYSICAL EXECUTION (kept pre-ch5 controller)", prev_m)
        
        # # Runtime Conformance
        # # ── Channel 5: run-time violation (assumption OR bounded-progress) → re-decompose
        # # once. Gated on use_dfa_feedback: channel 5 is part of the feedback harness, so
        # # the --no-feedback baseline must not get the repair loop either.
        # afb = "\n\n".join(x for x in (m.get("assumption_feedback"),
        #                               m.get("progress_feedback"),
        #                               m.get("offscript_feedback")) if x)
        # n_viol = (m.get("assumption_violations", 0) + m.get("progress_violations", 0)
        #           + m.get("offscript_violations", 0))
        # if n_viol > 0 and afb and task_attempt == 0 and use_dfa_feedback:
        #     report.info("runtime violation (channel 5)",
        #                 "assumption not discharged / no bounded progress — re-decomposing")
        #     # snapshot the current FSMs + metrics so keep-better can restore them
        #     snap_dir = os.path.join(fsm_dir, "_prev_fsm")
        #     os.makedirs(snap_dir, exist_ok=True)
        #     for f in os.listdir(fsm_dir):
        #         if f.endswith(".json"):
        #             shutil.copy2(os.path.join(fsm_dir, f), os.path.join(snap_dir, f))
        #     prev_snapshot = (m, snap_dir)
        #     physical_feedback = afb
        #     continue
        return


def main():
    """Parse CLI args and dispatch to run()."""
    def _task_type(v):
        if v in TASKS or os.path.isfile(v):
            return v
        raise argparse.ArgumentTypeError(
            f"unknown task {v!r}. Known tasks: {', '.join(TASKS)}")

    p = argparse.ArgumentParser(description="ContrAuto — two-layer (task + physical) runner")
    p.add_argument("--task", required=True, type=_task_type,
                   metavar="{" + "|" .join(TASKS) + "}",
                   help="task name or path to a custom YAML")
    p.add_argument("--model", default="gemini", choices=MODELS)
    p.add_argument("--no-sim", action="store_true", help="run the task layer only")
    p.add_argument("--visualize", dest="visualize", action="store_true", default=True,
                   help="export physical validation as outputs/<task>/physical_simulation.gif (default)")
    p.add_argument("--no-visualize", dest="visualize", action="store_false",
                   help="run physical validation without exporting a GIF")
    p.add_argument("--resynth", action="store_true", help="force re-synthesis even if FSMs exist")
    p.add_argument("--outdir", default=None)
    p.add_argument("--n", type=int, default=1, metavar="ROBOTS",
                   help="worker count for scalable tasks (charger, corridor, warehouse, "
                        "two_layer_dock); required in spirit -- defaults to 1 (a single worker, "
                        "not each task's historical robot count) when omitted")
    # synthesis-debug flags (task layer)
    p.add_argument("--no-feedback", action="store_true", help="disable DFA/refinement feedback in synthesis")
    p.add_argument("--no-completion", action="store_true",
                   help="disable the Step 9 mechanical completion layer (C2 ablation)")
    p.add_argument("--skip-translator", action="store_true", help="reuse cached translator output")
    p.add_argument("--retranslate", action="store_true",
                   help="redo the translator LLM call even when its inputs are unchanged "
                        "(by default an identical prompt + model reuses translator_output.json)")
    p.add_argument("--save-attempts", action="store_true", help="save each synthesis attempt's spec")
    p.add_argument("--no-quickxplain", action="store_true",
                   help="skip the QuickXplain minimal-core search on Step 10 (consistency) and "
                        "Step 12 (unrealizable) failures -- cheaper generic feedback only, trades "
                        "repair precision for fewer NuXMV/SLUGS calls per failed attempt")
    a = p.parse_args()
    report.logo()
    if a.n < 1:
        p.error("--n must be at least 1")
    task_name = os.path.splitext(os.path.basename(a.task))[0]
    scalable = {"charger", "corridor", "warehouse", "two_layer_dock"}
    n_robots = None
    if task_name in scalable:
        task_path = a.task if os.path.isfile(a.task) else os.path.join("contrauto", "tasks", f"{task_name}.yaml")
        task = TaskSpec.from_yaml(task_path).with_robot_count(a.n)
        n_robots = a.n
    else:
        if a.n != 1:
            p.error(f"--n is not supported for fixed-size task {task_name!r}")
        task = a.task
    feedback_modes = ({"consistency": "generic", "unrealizable": "generic"}
                      if a.no_quickxplain else None)
    run(task, model=a.model, simulate=not a.no_sim, resynth=a.resynth, outdir=a.outdir,
        use_dfa_feedback=not a.no_feedback, skip_translator=a.skip_translator,
        save_attempts=a.save_attempts,
        enable_completion=not a.no_completion, visualize=a.visualize, n_robots=n_robots,
        feedback_modes=feedback_modes, retranslate=a.retranslate)


if __name__ == "__main__":
    main()
