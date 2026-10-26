"""Contract decomposition agent: LLM-driven LTL -> CHASE contracts -> verified FSM.

Houses :class:`DecomposerAgent`, which runs the 5-step symbolic synthesis pipeline
(LLM contract generation, syntax check, consistency/compatibility, refinement, FSM
synthesis) with an episodic-memory retry loop. The low-level external-tool
plumbing (logics_tool / nuXmv / SLUGS invocations, chase-command-file writing, and
NuXMV-output parsing) lives in :mod:`contrauto.agents.tool_pipeline`; this module
keeps the LLM loop, step orchestration, feedback-prompt construction, and the
well-separatedness degeneracy gate.
"""

import os
import re
import json
import time
from pathlib import Path

from contrauto import report
from contrauto.llm import chat, MAX_RETRIES
from contrauto.prompts import DECOMPOSER_INSTRUCTIONS
from contrauto.memory import EpisodicMemory
from contrauto.agents.tool_pipeline import ToolPipeline, SLUGS_SYNTH_TIMEOUT
from contrauto.ordering_derivation import strip_flow

# Overall wall-clock budget for one DecomposerAgent.run() call (Translator's caller,
# ContrAuto, invokes this fresh per channel-5 re-decomposition round too, so this bounds
# EACH round independently). Without it a run can wander for an hour+: 9 attempts each
# potentially stacking multi-minute NuXMV/SLUGS work plus LLM retry backoff (observed: a
# single round taking 2139s). Checked between attempts (see run()) and passed to chat()
# as a deadline so one slow/rate-limited call can't alone burn the whole budget.
DECOMPOSER_TIME_BUDGET_SEC = int(os.environ.get("DECOMPOSER_TIME_BUDGET_SEC", "1200"))


# ══════════════════════════════════════════════════════════════════════════════
# DecomposerAgent
#
# Pipeline:  LTL  →  CHASE contracts  →  verify  →  Refinement → synthesize FSM
#
#   Step 1 : LLM generates CHASE contracts from LTL
#   Step 2 : Syntax check                (logics_tool show)
#   Step 3 : Consistency & Compatibility (Verification) (logics_tool verify_<contract_name>.chase
#                                          -> <contract_name>.smv -> NuXMV, expect 2xfalse)
#   Step 4 : Refinement                  (logics_tool verification.chase
#                                          -> refinement.smv -> NuXMV, expect 2xtrue)
#   Step 5 : FSM synthesis               (logics_tool synthesis.chase -> SLUGS -> JSON)
# ══════════════════════════════════════════════════════════════════════════════


# Display numbering for the per-attempt ladder, in execution order, with the tools each step
# runs (the step_logs keys such as "Step2.5:GR1Fragment" keep their historical names).
_STEPS = {
    "generate":    (1,  "LLM"),
    "grounding":   (2,  "Python · simulator vocabulary"),
    "syntax":      (3,  "CHASE logics_tool"),
    "gr1":         (4,  "Python · GR(1) fragment"),
    "initial":     (5,  "Python · initial values"),
    "ownership":   (6,  "Python · variable ownership"),
    "anchor":      (7,  "Python · mission anchoring"),
    "witness":     (8,  "Python · goal witness"),
    "completion":  (9,  "Python + SLUGS realizability check"),
    "consistency": (10, "CHASE logics_tool + nuXmv"),
    "refinement":  (11, "CHASE logics_tool + nuXmv IC3"),
    "synthesis":   (12, "CHASE logics_tool + SLUGS"),
}


def _step(key: str) -> str:
    num, tools = _STEPS[key]
    return f"Step {num:<2} " + report.dim(f"[{tools}]")


class DecomposerAgent:
    """Decompose verified LTL into refined CHASE contracts and synthesize FSMs.

    Drives the 5-step pipeline with an LLM-in-the-loop repair cycle backed by
    episodic (within-problem) and optional semantic (cross-task) memory. External
    tools are run through a :class:`ToolPipeline`; this class owns the LLM calls,
    step sequencing, feedback-prompt construction, and the degeneracy gate.
    """

    def __init__(self, model: str, workdir: str, mission: str,
                 use_dfa_feedback: bool = True, save_attempts: bool = False,
                 physical_feedback: str = "",
                 localized_feedback: bool = True, enforce_initial_conditions: bool = True,
                 crisp_feedback: bool = False, feedback_modes: dict = None,
                 ground_vocab: dict = None, ground_binder=None,
                 enable_completion: bool = True):
        """Configure an agent for one mission.

        Args:
            model: LLM backend name passed to ``contrauto.llm.chat``.
            workdir: Directory for all .ltl/.chase/.smv/.json artifacts.
            mission: Mission name (used to name refinement_<mission>.smv).
            use_dfa_feedback: Use the structured DFA refinement diagnoser when
                available; falls back to a generic hint on failure.
            save_attempts: Persist each LLM contract attempt as specs_attempt_N.ltl.
            physical_feedback: Structural feedback from the physical layer's
                task-level repair (escalation ladder step 4); empty if none.
        """
        self.model = model
        self.workdir = Path(workdir)
        self.mission = mission
        # GROUNDING VOCABULARY of the target simulator: {'sensors': [...], 'actions': [...]} -- the
        # atomic propositions the contract must be written over. Every declared variable must map
        # (via role_of) to a grounding, else the synthesized FSM cannot bind to the physical layer
        # and silently deadlocks in execution (e.g. an LLM-invented `carrying` that role_of cannot
        # place). None (unknown simulator) disables the check -> naming-agnostic fallback.
        self.ground_vocab = ground_vocab
        # Optional per-task binder (simulator family's own classifier) — when present it is
        # the grounding judge; vocab is then only the allowed-names listing in the feedback.
        self.ground_binder = ground_binder
        # Step 2.7 toggle for the completion-layer ablation (C2).
        self.enable_completion = enable_completion
        self.ltl_file = self.workdir / "specs.ltl"
        self.ver_file = self.workdir / "verification.chase"
        self.syn_file = self.workdir / "synthesis.chase"
        self._tra_out = None
        self.use_dfa_feedback = use_dfa_feedback
        # MASTER localization switch for the RQ1 ablation. When False, EVERY failure channel
        # (refinement, unrealizability, well-separatedness) returns only a GENERIC hint that names
        # the failing stage but withholds the counterexample / core / mined assumption / violated
        # formula. This is the true no-localization baseline: unlike use_dfa_feedback (which only
        # gated the DFA/core/assume blocks while the counterstrategy and vacuity traces still went
        # to the LLM), this isolates the localization variable cleanly.
        self.localized_feedback = localized_feedback
        # Deterministic initial-condition ownership (contribution C1). Default ON. Turned OFF for the
        # RQ1 feedback ablation so the LLM's raw drafts fail and enter the repair loop -- otherwise
        # enforce makes most tasks certify in one round and there is nothing for feedback to repair.
        self.enforce_initial_conditions = enforce_initial_conditions
        # CRISP localization mode (needs localized_feedback=True). The default localized feedback
        # dumps the full environment counterstrategy TRACE plus the named core; crisp sends ONLY the
        # one-line named cause (unrealizable core / violated formula), dropping the trace. This
        # isolates whether the value is in the precise POINTER or drowned by verbose text (which may
        # even overload a weak model). Three modes: generic (off) / crisp / verbose (default on).
        self.crisp_feedback = crisp_feedback
        # PER-STAGE feedback mode, one of 'generic' | 'crisp' | 'verbose' for each failure channel.
        # Lets the per-stage ablation pick the best form for each stage independently. Defaults are
        # derived from the global localized_feedback/crisp_feedback for backward compatibility;
        # pass `feedback_modes={'refinement': 'crisp', ...}` to override individual stages.
        _default = 'generic' if not localized_feedback else ('crisp' if crisp_feedback else 'verbose')
        self.feedback_modes = {s: _default for s in
                               ('refinement', 'consistency', 'unrealizable', 'wellsep')}
        if feedback_modes:
            self.feedback_modes.update(feedback_modes)
        self.save_attempts = save_attempts
        # Per-round contract text (one entry per LLM (re)generation), so a repair transcript can
        # show exactly which formula the LLM changed after a localized hint. Emitted in the output.
        self._round_contracts = []
        # TASK-LEVEL REPAIR channel: structural feedback from the PHYSICAL layer (escalation
        # ladder step 4). When physical repair (reassign / priority / deadline relax) cannot
        # satisfy the deadline because the failure is STRUCTURAL (deadlock / starvation /
        # infeasible decomposition), the outer loop re-synthesizes the FSM with this note so the
        # LLM fixes the contracts/arbiter/decomposition — not just the quantitative knobs.
        self.physical_feedback = physical_feedback or ""
        self._attempt_counter = 0
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._step_logs = []

        # All external-tool plumbing (logics_tool / nuXmv / SLUGS, chase-command-file
        # writing, NuXMV-output parsing) is delegated to this helper.
        self._tools = ToolPipeline(self.workdir, self.mission, self.ltl_file,
                                   self.ver_file, self.syn_file)

    def _log(self, step: str, passed: bool, detail: str = "") -> None:
        """Record the outcome of each pipeline step."""
        self._step_logs.append({
            "step": step,
            "passed": passed,
            "detail": detail,
        })

    def _static_ok(self, key: str, detail: str = "", body=()):
        """Buffer a passing cheap static check (Steps 4-9); _flush_static prints them as one line."""
        self._static.append((_STEPS[key][0], detail, list(body)))

    def _flush_static(self):
        if not getattr(self, "_static", None):
            return
        nums = [n for n, _, _ in self._static]
        span = f"{min(nums)}–{max(nums)}" if len(nums) > 1 else str(nums[0])
        details = "; ".join(d for _, d, _ in self._static if d)
        tools = ("Python + SLUGS realizability check" if _STEPS["completion"][0] in nums
                 else "Python")
        report.ok(f"Step {span} " + report.dim(f"[{tools}]") + "  static checks passed", details)
        for _, _, body in self._static:
            for line in body:
                report.body(line)
        self._static = []

    def _fbmode(self, stage: str) -> str:
        """Feedback mode ('generic' | 'crisp' | 'verbose') for a failure stage."""
        return self.feedback_modes.get(stage, 'verbose')

    def run(self, translator_output: dict, task) -> dict:
        report.agent("Decomposer", "LTL → Contracts → FSM")

        self._deadline = time.monotonic() + DECOMPOSER_TIME_BUDGET_SEC
        self._tra_out = translator_output
        pairs = translator_output["ltl_pairs"]
        variables = translator_output["variables"]
        base_msg = self._build_base_prompt(task, variables, pairs)
        base_msg += (
            "\n\n## HOW TO REPAIR (read every round before revising)\n"
            "This is a MULTI-CONTRACT decomposition. When a check fails we point you at a LOCAL "
            "cause (a named guarantee, or the input the environment exploits), but the fix is almost "
            "never local: a shared variable is one contract's OUTPUT and another's INPUT (e.g. a "
            "`grant` is the Arbiter's output and a Robot's input). So a change in one contract must "
            "be mirrored in the contracts it interfaces with:\n"
            "  - If a Robot needs to ASSUME something about an input (e.g. a grant eventually comes), "
            "the contract that OUTPUTS that variable (the Arbiter) must GUARANTEE it.\n"
            "  - For that producer to guarantee it, IT may need an assumption about what it receives "
            "(e.g. the Arbiter may assume a holder eventually RELEASES the resource), which the Robot "
            "must then guarantee.\n"
            "Do NOT patch only the named contract: trace each changed variable to its producer and "
            "consumers and revise ALL of them together, so you do not fix one certificate and break "
            "another. Think about the WHOLE protocol, not the single reported line.\n"
        )
        if self.physical_feedback:
            base_msg += ("\n\n## STRUCTURAL FEEDBACK FROM PHYSICAL EXECUTION (task-level repair)\n"
                         "A previous synthesis produced an FSM that the physical layer could NOT "
                         "satisfy even after exhausting physical repair (reassignment, priority "
                         "yielding, deadline relaxation). The failure is STRUCTURAL — fix it in "
                         "the contracts / arbiter policy / decomposition, e.g. ensure fairness so "
                         "no robot starves, break circular waits, or re-split the tasks:\n"
                         + self.physical_feedback)

        contract_content = None
        ci = None
        fsm_files = None
        succeeded = False
        user_msg = base_msg
        # EPISODIC memory (within THIS design problem): every failure is distilled -- via one
        # SummarizerAgent.run() merge call, capped at lesson_cap characters -- into a single
        # persisted lesson at workdir/episodic_lessons.md, so a later step never repeats an
        # earlier mistake and a rerun of this SAME task (e.g. --resynth) picks up where the last
        # run's lesson left off. It is a distillation, not a growing transcript: the cap forces
        # old/resolved points to be dropped so the prompt does not balloon over many retries.
        episodic = EpisodicMemory(model=self.model, lessons_path=self.workdir / "episodic_lessons.md",
                                  mission=self.mission)

        for attempt in range(MAX_RETRIES * 3):
            if time.monotonic() >= self._deadline:
                elapsed = DECOMPOSER_TIME_BUDGET_SEC + (time.monotonic() - self._deadline)
                report.fail(f"Timeout  exceeded the {DECOMPOSER_TIME_BUDGET_SEC}s budget",
                            f"{elapsed:.0f}s elapsed after {attempt} attempt(s) — stopping")
                self._log("Timeout", False,
                          f"exceeded {DECOMPOSER_TIME_BUDGET_SEC}s budget after {attempt} attempt(s)")
                break
            report.log(report.dim(f"\n  attempt {attempt + 1}/{MAX_RETRIES * 3}"))
            self._static = []
            report.branch()

            # ── Step 1: LLM contract generation ───────────────────────────────
            if contract_content is None:
                contract_content, error = self._step_generate_contracts(user_msg)
                if error:
                    episodic.remember(error, "")
                    user_msg = episodic.render(base_msg)
                    contract_content = None
                    continue
                # Defensive: strip any stray `CAUSAL FLOW:` / `<x> before <y>` line (the prompt no
                # longer asks for one, but a leftover would not be valid CHASE .ltl).
                contract_content = strip_flow(contract_content)
                # ── Step 1.6: grounding-vocabulary completeness (every AP must bind) ──
                ground_fb = self._check_grounding(contract_content)
                if ground_fb:
                    self._log("Step 1.6 Grounding vocabulary", False, ground_fb)
                    report.fail(f"{_step('grounding')}  ungrounded variable(s), won't bind to simulator → re-decompose")
                    if len(ground_fb.splitlines()) > 1:
                        report.body(ground_fb.splitlines()[1])
                    episodic.remember(ground_fb, base_msg)
                    user_msg = episodic.render(base_msg)
                    contract_content = None
                    continue
                report.ok(f"{_step('grounding')}  every variable binds to the simulator")
                if self.enforce_initial_conditions:
                    contract_content = self._enforce_initial_conditions(contract_content)
                self._round_contracts.append(contract_content)
                self._attempt_counter += 1
                if self.save_attempts:
                    path = self.workdir / f"specs_attempt_{self._attempt_counter}.ltl"
                    self._save(path, contract_content)

            # ── Step 2: Syntax check ───────────────────────────────────────────
            ok, user_msg = self._step_syntax_check(contract_content, base_msg)
            if not ok:
                episodic.remember(user_msg, base_msg)
                user_msg = episodic.render(base_msg)
                contract_content = None
                continue

            # ── Step 2.5: Static GR(1)-fragment check ─────────────────────────
            # Catch non-GR(1) sub-contract guarantees (response / bare F / Until / nested
            # temporal) HERE, with formula-level localization, instead of letting them
            # surface downstream as an unrealizable/degenerate/empty FSM. Pure-Python, no
            # tool call. Disabled together with the other structured diagnosers under
            # --no-feedback so a baseline run can attribute failures to the symbolic loop.
            if self.use_dfa_feedback and self.localized_feedback:
                ok, user_msg = self._step_gr1_fragment(contract_content, base_msg)
                if not ok:
                    episodic.remember(user_msg, base_msg)
                    user_msg = episodic.render(base_msg)
                    contract_content = None
                    continue

                # A cold-start output pin and a current-state safety guard can contradict
                # before NuXMV runs. Catch the deterministic, propositional cases locally.
                from contrauto.feedback.initial_consistency import (
                    format_initial_consistency_feedback, initial_output_contradictions,
                )
                initial_conflicts = initial_output_contradictions(contract_content)
                if initial_conflicts:
                    if len(initial_conflicts) > 3:
                        shown = initial_conflicts[:3]
                        feedback = format_initial_consistency_feedback(shown, contract_content)
                        feedback = feedback.replace(
                            "\n\nCurrent .ltl:",
                            f"\n- ... and {len(initial_conflicts) - len(shown)} more equivalent conflict(s)."
                            "\n\nCurrent .ltl:",
                        )
                    else:
                        feedback = format_initial_consistency_feedback(initial_conflicts, contract_content)
                    short = "; ".join(f"{item.contract}.{item.output}" for item in initial_conflicts)
                    self._flush_static()
                    report.fail(f"{_step('initial')}  initial output contradiction(s)", short)
                    self._log("Step2.55:InitialConsistency", False, short)
                    episodic.remember(feedback, base_msg)
                    user_msg = episodic.render(base_msg)
                    contract_content = None
                    continue
                self._static_ok("initial")

                # A symmetric N-robot instance names each robot's point-to-point signals
                # after its own index (grant_r4, req_r4, ...); the LLM occasionally copies
                # a sibling's index by mistake (Robot4 reading `grant_r1`). NuXMV cannot
                # catch this -- `grant_r1` is a valid variable, just not Robot4's -- and it
                # otherwise only surfaces many attempts later as an UNREALIZABLE core.
                from contrauto.feedback.ownership_check import (
                    format_ownership_feedback, ownership_leaks,
                )
                leaks = ownership_leaks(contract_content, task)
                if leaks:
                    feedback = format_ownership_feedback(leaks, contract_content)
                    short = "; ".join(f"{leak.contract}->{leak.variable}" for leak in leaks)
                    self._flush_static()
                    report.fail(f"{_step('ownership')}  cross-contract ownership leak(s)", short)
                    self._log("Step2.56:OwnershipLeak", False, short)
                    episodic.remember(feedback, base_msg)
                    user_msg = episodic.render(base_msg)
                    contract_content = None
                    continue
                self._static_ok("ownership")

            # ── Step 2.6: mission anchoring (System must carry the translator's formulas) ──
            # Anchors the ladder against SPEC EROSION (failure class 7): refinement proves
            # composition ⊑ System, but System is LLM-written — dropping the mission's
            # mutexes/dynamics lets a physically useless strategy certify. Well-formedness
            # precondition like Steps 2/1.6 -> active in BOTH ablation arms.
            # Erosion repair is MECHANICAL (formulas known, placement syntactic): auto-insert
            # instead of burning an LLM round on transcription — v2-run1 spent 3/9 attempts
            # on anchor round-trips and 3 more fixing the LLM's mis-sorted sections.
            try:
                from contrauto.feedback.mission_anchor import (
                    missing_mission_formulas, anchor_complete, format_anchor_feedback)
                _missing = missing_mission_formulas(self._tra_out.get("ltl_pairs", []),
                                                    contract_content)
            except Exception:
                _missing = []
            if _missing and _missing[0].startswith("<no CONTRACT System"):
                afb = format_anchor_feedback(_missing)
                self._log("Step2.6:MissionAnchor", False, afb)
                self._flush_static()
                report.fail(f"{_step('anchor')}  no CONTRACT System block → re-decompose")
                episodic.remember(afb, base_msg)
                user_msg = episodic.render(base_msg)
                contract_content = None
                continue
            if _missing:
                contract_content, _added = anchor_complete(
                    self._tra_out.get("ltl_pairs", []), contract_content)
                by_sec = {}
                for sec, f in _added:
                    by_sec[sec] = by_sec.get(sec, 0) + 1
                self._static_ok("anchor", f"{len(_added)} mission formula(s) added to System")
                self._log("Step2.6:MissionAnchor", True,
                          "; ".join(f"[{s}] {f}" for s, f in _added))
            else:
                self._static_ok("anchor")

            # ── Step 2.65: goal-witness check (anti-cheap-exit) ───────────────
            # A justice goal over an unguarded OUTPUT is satisfiable by asserting the
            # action vacuously forever (probe-verified: an FSM looping on `done` with
            # zero deliveries). Which input witnesses the action is SEMANTIC, so this
            # emits localized feedback, never an auto-patch. Well-formedness gate like
            # 2.5/2.6 -> both ablation arms.
            try:
                from contrauto.feedback.goal_witness import (
                    unwitnessed_goal_outputs, format_witness_feedback)
                _unwit = unwitnessed_goal_outputs(contract_content)
            except Exception:
                _unwit = []
            if _unwit:
                wfb = format_witness_feedback(_unwit)
                self._log("Step2.65:GoalWitness", False, wfb)
                self._flush_static()
                report.fail(f"{_step('witness')}  unguarded goal output(s) → re-decompose",
                            ", ".join(f"{c}.{v}" for c, v in _unwit))
                episodic.remember(wfb, base_msg)
                user_msg = episodic.render(base_msg)
                contract_content = None
                continue
            self._static_ok("witness")

            # ── Step 2.7: mechanical completion (internal-variable closure) ───
            # Template-approximated quotient: discharge internal assumptions (sibling
            # guarantee if the sibling stays realizable, else fold into System.Assumptions)
            # + verified persistence closure. Solver-arbitrated, never touches external-input
            # obligations; justice-cycles block with a diagnosis (breaking one is semantic).
            # Part of the harness -> disabled with the localization ablation switches.
            if self.use_dfa_feedback and self.localized_feedback and self.enable_completion:
                try:
                    from contrauto.completion import complete
                    cres = complete(self._tools, contract_content)
                    if cres.blocked:
                        user_msg = (base_msg + "\n\n" + "\n".join(cres.blocked)
                                    + f"\n\nCurrent .ltl:\n{contract_content}\n\n"
                                    "Break the cycle as instructed and return raw .ltl only.\n")
                        self._log("Step2.7:Completion", False, "\n".join(cres.blocked))
                        self._flush_static()
                        report.fail(f"{_step('completion')}  justice cycle → re-decompose")
                        episodic.remember(user_msg, base_msg)
                        user_msg = episodic.render(base_msg)
                        contract_content = None
                        continue
                    if cres.applied:
                        contract_content = cres.content
                        summary = "; ".join(f"[{k}] {c}: {f}" for k, c, f in cres.applied)
                        self._static_ok("completion",
                                        f"{len(cres.applied)} completion patch(es)",
                                        [f"+ {c}: {f}   [{k}]" for k, c, f in cres.applied])
                        self._log("Step2.7:Completion", True, summary)
                    if not cres.applied:
                        self._static_ok("completion")
                    for s in cres.suggestions:
                        self._flush_static()
                        report.note(f"{_step('completion')}  suggestion (external input)", s)
                except Exception as e:
                    self._flush_static()
                    report.note(f"{_step('completion')}  completion skipped", f"{type(e).__name__}: {e}")

            # ── Prepare all chase command files ───────────────────────────────
            self._save(self.ltl_file, contract_content)
            ci = self._tools.write_chase_command_files(contract_content)

            # ── Step 3: verify Consistency & Compatibility  (NuXMV, expect 2xfalse) ──
            ok, user_msg = self._step_consistency_compatibility(ci, contract_content, base_msg)
            if not ok:
                episodic.remember(user_msg, base_msg)
                user_msg = episodic.render(base_msg)
                contract_content = None
                continue

            # ── Step 4: Refinement  (NuXMV, expect 2xtrue) ────────────────────
            ok, user_msg = self._step_refinement(contract_content, base_msg)
            if not ok:
                episodic.remember(user_msg, base_msg)
                user_msg = episodic.render(base_msg)
                contract_content = None
                continue

            # ── Step 5: FSM synthesis ──────────────────────────────────────────
            ok, fsm_files, user_msg = self._step_synthesize(ci, contract_content, base_msg)
            if not ok:
                episodic.remember(user_msg, base_msg)
                user_msg = episodic.render(base_msg)
                contract_content = None
                continue

            report.ok(report.bold("All checks passed"))
            succeeded = True
            break

        else:
            report.fail("max retries reached")

        # ci is None if every attempt failed before Step 3 (e.g. a persistent tool error), so
        # guard the subscripts — a failed run must still return a well-formed record.
        return {
            "success": succeeded,
            "ltl_content": contract_content,
            "contracts": (ci["controllers"] + [ci["system"]]) if ci else [],
            "controllers": ci["controllers"] if ci else [],
            "system_name": ci["system"] if ci else None,
            "fsm_files": fsm_files or {},
            "usage": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
            },
            "step_logs": self._step_logs,
            "round_contracts": self._round_contracts,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Accumulated (monotonic) feedback — the fix for the non-convergence we observed on
    # N>=3 shared-resource cases: each round only saw the LATEST diagnosis, so the LLM kept
    # re-introducing failures it had already been told about (e.g. degenerate node 28 / a
    # dropped mutex). We now carry EVERY distinct prior diagnosis forward.
    # ══════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════
    # Step 1 -- LLM contract generation
    # ══════════════════════════════════════════════════════════════════════════
    def _step_generate_contracts(self, user_msg: str):
        """
        Ask the LLM to decompose LTL into CHASE contracts.
        Returns (contract_content | None, error_msg | None).
        """
        try:
            with report.spinner("Decomposing"):
                text, usage = chat(self.model, DECOMPOSER_INSTRUCTIONS, user_msg,
                                   deadline=self._deadline)
        except Exception as e:
            # Any LLM-call failure (network/billing/rate-limit exhaustion, or the deadline
            # cutting a retry short) folds back into the normal per-attempt error path
            # instead of crashing the whole run with a raw traceback -- the next loop
            # iteration's deadline check (see run()) is what actually decides whether to
            # stop cleanly or try again.
            msg = f"LLM call failed: {type(e).__name__}: {e}"
            report.fail(f"{_step('generate')}  LLM call failed", f"{type(e).__name__}: {e}")
            self._log("Step1:GenerateContracts", False, msg)
            return None, msg
        contract_content = self._strip(text)

        self.total_input_tokens += usage["input_tokens"]
        self.total_output_tokens += usage["output_tokens"]

        found = re.findall(r'CONTRACT\s+(\w+)\s*:', contract_content)
        sys_names = [n for n in found if n.lower() == "system"]
        sub_names = [n for n in found if n.lower() != "system"]

        if not found:
            report.fail(f"{_step('generate')}  no CONTRACT blocks found")
            self._log("Step1:GenerateContracts", False, "No CONTRACT blocks found")
            return None, "No CONTRACT blocks found."
        if not sys_names:
            report.fail(f"{_step('generate')}  missing CONTRACT System")
            self._log("Step1:GenerateContracts", False, "Missing CONTRACT System.")
            return None, "Missing CONTRACT System."
        if not sub_names:
            report.fail(f"{_step('generate')}  no sub-contracts found")
            self._log("Step1:GenerateContracts", False, "No sub-contracts found.")
            return None, "No sub-contracts found."

        report.ok(f"{_step('generate')}  contracts generated", ", ".join(found))
        self._log("Step1:GenerateContracts", True, f"Contracts: {found}")
        return contract_content, None

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2 -- Syntax check
    # ══════════════════════════════════════════════════════════════════════════
    def _step_syntax_check(self, contract_content: str, base_msg: str):
        """
        Syntax-check via logics_tool (minimal 'show' command).
        Returns (passed: bool, next_user_msg: str).
        """
        ok, err = self._tools.parse_ltl(contract_content)
        if not ok:
            err_short = "\n".join(err.splitlines()[:5])
            self._log("Step2:SyntaxCheck", False, f"Syntax error:\n{err_short}")
            report.fail(f"{_step('syntax')}  syntax error")
            report.body(err_short)
            return False, (
                base_msg + contract_content + "\n\n"
                + f"Syntax errors:\n{err_short}\n\n"
                  "Fix syntax only. Return raw .ltl text only.\n\n"
            )

        report.ok(f"{_step('syntax')}  syntax check")
        self._log("Step2:SyntaxCheck", True)
        return True, base_msg

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2.5 -- Static GR(1)-fragment check (sub-contract guarantees)
    # ══════════════════════════════════════════════════════════════════════════
    def _step_gr1_fragment(self, contract_content: str, base_msg: str):
        """Reject sub-contract guarantees outside the synthesizable GR(1) fragment.

        Only sub-contracts (controllers) are synthesized, so only their guarantees must be
        GR(1) (init Boolean / safety G(rho) / justice G F(J)); the System contract and the
        assumptions get the lenient treatment inside :mod:`contrauto.feedback.gr1_fragment`.
        Returns (passed: bool, next_user_msg: str).
        """
        from contrauto.feedback.gr1_fragment import (
            check_gr1_fragment, format_gr1_feedback,
        )
        violations = check_gr1_fragment(contract_content)
        if violations:
            short = "; ".join(f"{v.formula.contract}.{v.formula.section}[{v.formula.index}]"
                              for v in violations)
            self._flush_static()
            report.fail(f"{_step('gr1')}  non-GR(1) guarantees", short)
            self._log("Step2.5:GR1Fragment", False, short)
            return False, base_msg + "\n\n" + format_gr1_feedback(violations, contract_content)

        self._static_ok("gr1")
        self._log("Step2.5:GR1Fragment", True)
        return True, base_msg

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3 -- Consistency & Compatibility
    # ══════════════════════════════════════════════════════════════════════════
    def _step_consistency_compatibility(self, ci: dict, contract_content: str, base_msg: str):
        """
        For each contract:
          logics_tool -i specs.ltl -c verify_<name>.chase -V  ->  <name>.smv
          NuXMV <name>.smv                                    ->  expect 2xfalse
        Returns (passed: bool, next_user_msg: str).
        """
        self._flush_static()
        report.status("Checking consistency & compatibility")
        ok, errors = self._tools.check_consistency_compatibility(ci)

        if not ok:
            err_short = "\n".join(errors.splitlines()[:10])
            report.fail(f"{_step('consistency')}  consistency / compatibility")
            report.body(err_short)
            self._log("Step3:ConsistencyCompatibility", False, err_short)
            mode = self._fbmode('consistency')
            if mode == 'generic':
                return False, (
                    base_msg + contract_content + "\n\n"
                    "CONSISTENCY/COMPATIBILITY FAILED: some contract is not satisfiable. "
                    "Revise the contracts and try again. Return raw .ltl text only.\n\n"
                )
            # localized: minimal INCONSISTENT core (which guarantees contradict) per failing contract.
            core_blocks = []
            try:
                from contrauto.feedback.gr1_core import (
                    compute_consistency_core, format_consistency_core_feedback)
                for ctrl in ci["controllers"] + [ci["system"]]:
                    if self._tools.check_consistent(ctrl, contract_content) is False:
                        core = compute_consistency_core(self._tools, contract_content, ctrl)
                        if core:
                            core_blocks.append(format_consistency_core_feedback(ctrl, core))
            except Exception:
                core_blocks = []
            crisp = "\n\n".join(core_blocks)
            trace_block = "" if mode == 'crisp' else f"\n\nNuXMV output:\n{err_short}"
            return False, (
                base_msg + "\n\n"
                "CONSISTENCY/COMPATIBILITY FAILED (a contract is not satisfiable).\n\n"
                + (f"LOCALIZED CAUSE:\n{crisp}\n\n" if crisp
                   else "(could not isolate a minimal contradicting core.)\n\n")
                + (trace_block + "\n\n" if trace_block else "")
                + f"Current .ltl:\n{contract_content}\n\n"
                "Weaken ONE guarantee in the inconsistent core, or fix the assumptions. "
                "Return raw .ltl text only.\n\n"
            )

        report.ok(f"{_step('consistency')}  consistent & compatible", ", ".join(ci["verify_files"]))
        self._log("Step3:ConsistencyCompatibility", True)
        return True, base_msg

    def _diagnose_counterexample(self, ce_raw: str) -> str:
        """Generic, case-agnostic refinement-failure hint — a fallback for when the structured
        `diagnose_refinement` is disabled (--no-feedback) or errors out. (The previous version
        hard-coded warehouse/2-robot variable names and ISSUE 1-6 templates; removed for
        generality — the structured diagnoser parses the actual counterexample instead.)"""
        ce = self.format_counterexample(ce_raw) if ce_raw else ""
        return (
            "Refinement failed: the composition of the sub-contracts does not refine the "
            "System contract.\n"
            + (f"Counterexample (allowed by the composition, forbidden by System):\n{ce}\n" if ce else "")
            + "Fix generically: every System guarantee must be IMPLIED by the sub-contract "
              "guarantees, and every sub-contract assumption must be discharged by another "
              "sub-contract's guarantee or a System assumption. Do not hard-code scenario specifics."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4 -- Refinement
    # ══════════════════════════════════════════════════════════════════════════
    def _step_refinement(self, contract_content: str, base_msg: str):
        """
        Step 4.1: logics_tool generates refinement_<mission>.smv
        Step 4.2: NuXMV IC3 checks refinement -- expect 2x true
        Returns (passed: bool, next_user_msg: str).
        """
        report.status("Checking refinement")

        ok, err = self._tools.run_verification()
        if not ok:
            err_short = "\n".join(err.splitlines()[:6])
            report.fail(f"{_step('refinement')}  logics_tool error")
            report.body(err_short)
            self._log("Step4:Refinement", False, f"logics_tool error:\n{err_short}")
            return False, (
                f"logics_tool error:\n{err_short}\n\n"
                "Fix syntax only. Do NOT change contract structure.\n\n"
                + base_msg
            )

        smv_file = self.workdir / f"refinement_{self.mission}.smv"
        ref_ok, nux_out = self._tools.run_nuxmv_refinement(smv_file)
        true_cnt = len(re.findall(r"\bis true\b", nux_out))

        if not ref_ok:
            if self._fbmode('refinement') == 'generic':
                # Generic baseline (RQ1): report only that refinement failed, withhold the
                # counterexample trace and the per-formula diagnosis.
                self._log("Step4:Refinement", False, f"true_cnt={true_cnt}/2 (generic)")
                return False, (
                    base_msg + "\n\n"
                    "REFINEMENT FAILED: the composition of the sub-contracts does not refine the "
                    "system requirement. Revise the contract decomposition and try again.\n\n"
                    f"Current .ltl:\n{contract_content}\n\n"
                    "Return raw .ltl text only.\n\n"
                )
            ce_raw = self.extract_counterexample(nux_out)
            if not ce_raw:
                ce_raw = nux_out
            ce_lines = self.format_counterexample(ce_raw)

            if self.use_dfa_feedback:
                try:
                    from contrauto.feedback.diagnose import diagnose_refinement
                    diagnosis = diagnose_refinement(nux_out, contract_content, self.mission)
                except Exception:
                    diagnosis = self._diagnose_counterexample(ce_raw)
            else:
                diagnosis = self._diagnose_counterexample(ce_raw)

            report.fail(f"{_step('refinement')}  refinement failed ({true_cnt}/2 true)")
            report.body(f"Counterexample:\n{ce_lines}")
            self._log("Step4:Refinement", False,
                      f"true_cnt={true_cnt}/2\n"
                      f"Counterexample:\n{ce_lines}\n"
                      f"Current (failing) .ltl:\n{contract_content}\n\n"
                      "Redesign CONTRACT decomposition")

            # crisp: the per-formula pointer only; verbose: also append the raw counterexample
            # lasso trace (the analogue of the counterstrategy dump in the unrealizable channel).
            trace_block = ("" if self._fbmode('refinement') == 'crisp'
                           else f"\n\nCounterexample trace:\n{ce_lines}")
            return False, (
                base_msg + "\n\n"
                + diagnosis + trace_block + "\n\n"
                # NOTE: this block previously opened with "Keep ALL LTL formulas unchanged",
                # which contradicts the diagnosis's repair options outright — the LLM obeyed
                # it and only reshuffled formulas, never touching the flagged ones.
                "Rules:\n"
                "- The CONTRACT System block is the FIXED problem specification: do NOT weaken,\n"
                "  drop or rewrite ITS formulas to make refinement pass (only OPTION B may add\n"
                "  a System assumption, and only if the real environment provides it).\n"
                "- Sub-contract formulas ARE yours to redesign: change the flagged ✗ formulas\n"
                "  specifically — reordering or re-parenthesising fixes nothing.\n"
                "- Sub-contract Guarantees must jointly IMPLY the System Guarantees.\n"
                "- Every sub-contract assumption on an INTERNAL (sibling-output) variable must\n"
                "  be DISCHARGED by a sibling guarantee in the composition.\n"
            )

        report.ok(f"{_step('refinement')}  refinement")
        self._log("Step4:Refinement", True)
        return True, base_msg

    # ══════════════════════════════════════════════════════════════════════════
    # Well-separatedness gate: a synthesized GR(1) strategy is DEGENERATE if, from the
    # initial node, it can reach a DEAD-END (a node with no successors). For a recurring
    # (G F) objective a correct controller always has a move, so a reachable dead-end means
    # the spec is NOT well-separated: the system won vacuously by forcing the environment
    # into a state with no legal ENV_TRANS move, and SLUGS extracted a useless strategy
    # (it even violates the system's own SYS_TRANS there). Pure graph reachability on the
    # FSM JSON — no extra tools. Validated on /tmp/slugsexp sym (degenerate) vs clean2 (ok).
    # ══════════════════════════════════════════════════════════════════════════
    def _degeneracy_witness(self, json_path, contract_content, ctrl):
        """Find a REAL degeneracy: a reachable DEAD-END whose state violates a SAFETY
        GUARANTEE of `ctrl` (the system broke its OWN rule / gave up). A dead-end that only
        violates an ASSUMPTION means the ENVIRONMENT broke its promise there — a legitimate
        win, NOT degeneracy — and is ignored (it is unreachable under a faithful environment).
        Returns (deadend_id, path_states) for the 'counterexample' trace, or (None, None)."""
        try:
            d = json.loads(Path(json_path).read_text())
        except Exception:
            return None, None
        nodes = d.get("nodes", {})
        variables = d.get("variables", [])
        if "0" not in nodes:
            return None, None
        from collections import deque
        from contrauto.feedback.diagnose import violates_safety_guarantee
        parent = {"0": None}
        q = deque(["0"])
        deadends = []
        while q:
            x = q.popleft()
            tr = nodes[x].get("trans", [])
            if not tr:
                deadends.append(x)
            for t in tr:
                t = str(t)
                if t in nodes and t not in parent:
                    parent[t] = x
                    q.append(t)

        def to_state(nid):
            st = nodes[nid].get("state", [])
            return {variables[i]: bool(st[i]) for i in range(min(len(variables), len(st)))}

        for de in deadends:
            if not violates_safety_guarantee(to_state(de), contract_content, ctrl):
                continue                                  # env-illegal sink, not a system fault
            chain, cur = [], de
            while cur is not None:
                chain.append(cur)
                cur = parent[cur]
            chain.reverse()
            return de, [to_state(n) for n in chain]
        return None, None

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5 -- FSM Synthesis
    # ══════════════════════════════════════════════════════════════════════════
    def _step_synthesize(self, ci: dict, contract_content: str, base_msg: str):
        """
        logics_tool synthesis.chase  ->  <ctrl>.structuredSlugs
        SLUGS_COMP                   ->  <ctrl>.slugsin
        SLUGS                        ->  <ctrl>.json
        Returns (passed: bool, fsm_files: dict, next_user_msg: str).
        """
        report.status("Synthesizing FSMs")
        fsm_files = {}
        unrealizable = []
        degenerate = []
        inactive = []          # liveness vacuity: controlled liveness outputs never asserted
        slugs_errors = []
        timed_out = []         # SLUGS exceeded its synthesis time budget (too big, not wrong)

        ok_syn, syn_err = self._tools.run_synthesis()
        if not ok_syn:
            err_short = syn_err[:400]
            report.fail(f"{_step('synthesis')}  logics_tool synthesis error")
            report.body(err_short)
            self._log("Step5:Synthesis", False, f"logics_tool synthesis error:\n{err_short}")
            return False, {}, (
                base_msg + contract_content + "\n\n"
                f"logics_tool synthesis error:\n{err_short}\n\n"
                "Fix contracts so logics_tool can generate .structuredSlugs. "
                "Return raw .ltl text only.\n\n"
            )

        for ctrl in ci["controllers"]:
            report.status(f"Synthesizing {ctrl} (SLUGS)")
            res = self._tools.synthesize_controller(ctrl)
            status = res["status"]

            if status == "missing":
                slugs_errors.append(f"{ctrl}: .structuredSlugs not found")
                continue
            if status == "timeout":
                report.fail(f"{_step('synthesis')}  {ctrl}: SLUGS timeout", f"> {SLUGS_SYNTH_TIMEOUT}s")
                timed_out.append(ctrl)
                continue
            if status == "error":
                err_short = res["detail"]
                report.fail(f"{_step('synthesis')}  {ctrl}: SLUGS error")
                report.body(err_short)
                slugs_errors.append(f"{ctrl}: {err_short}")
                continue
            if status == "unrealizable":
                report.fail(f"{_step('synthesis')}  {ctrl}: unrealizable")
                unrealizable.append(ctrl)
                continue
            if status == "empty":
                slugs_errors.append(f"{ctrl}: empty FSM output")
                continue

            jo = res["json_path"]
            did, path = self._degeneracy_witness(jo, contract_content, ctrl)
            if did is not None:
                report.fail(f"{_step('synthesis')}  {ctrl}: degenerate (not well-separated)",
                            f"reaches dead-end node {did}")
                degenerate.append((ctrl, did, path))
                continue
            dead_outputs = sorted(
                set(self._activity_witness(jo, contract_content, ctrl))
                | set(self._activity_witness(jo, contract_content, ctrl,
                                             liveness_from="System")))
            if dead_outputs:
                report.fail(f"{_step('synthesis')}  {ctrl}: vacuous liveness",
                            f"never asserts {', '.join(dead_outputs)} anywhere in its strategy")
                inactive.append((ctrl, dead_outputs))
                continue
            report.ok(f"{_step('synthesis')}  FSM {jo.name}", "well-separated")
            fsm_files[ctrl] = str(jo)

        if unrealizable:
            if self._fbmode('unrealizable') == 'generic':
                # Generic baseline (RQ1): name only the failing stage, withhold the counterstrategy,
                # core, and mined assumption.
                self._log("Step5:Synthesis", False, f"Unrealizable (generic): {unrealizable}")
                return False, {}, (
                    base_msg + "\n\n"
                    "SYNTHESIS FAILED: the decomposition is not realizable by GR(1) synthesis. "
                    "Revise the contracts and try again.\n\n"
                    f"Current .ltl:\n{contract_content}\n\n"
                    "Return raw .ltl text only.\n\n"
                )
            pairs = self._tra_out["ltl_pairs"]
            variables = self._tra_out["variables"]
            ltl_list = "\n".join(f"  [{i+1}] {p['nl']}\n       => {p['ltl']}"
                                for i, p in enumerate(pairs))
            in_vars = set(self._tra_out.get("variables", {}).get("inputs", []))
            cstrat = self._tools.counterstrategy_diagnosis(unrealizable, in_vars)  # verbose trace

            # SOLVER-based localization (QuickXplain minimal unrealizable core): pin down which
            # GUARANTEES are jointly unrealizable, the realizability analogue of the refinement
            # diagnoser's per-formula classification. Best-effort. These blocks are the CRISP
            # pointer (named guarantees / mined assumption), separate from the verbose trace above.
            core_blocks, assume_blocks = [], []
            cores = {}
            if self._fbmode('unrealizable') != 'generic':
                try:
                    from contrauto.feedback.gr1_core import compute_core, format_core_feedback
                    for ctrl in unrealizable:
                        core = compute_core(self._tools, contract_content, ctrl)
                        if core is not None:
                            cores[ctrl] = core
                            core_blocks.append(format_core_feedback(ctrl, core))
                except Exception:
                    core_blocks = []
                # (B) assumption side = DIRECTION ONLY. We name the inputs the environment holds
                # fixed in its counterstrategy, but do NOT prescribe a specific assumption: the
                # guarantee side is localized precisely (the core), the assumption side is left for
                # the LLM to author. This also drops the per-candidate re-synthesis that made the
                # unrealizable diagnosis expensive (and ate the repair budget on multi-contract runs).
                try:
                    from contrauto.feedback.gr1_assume import held_hint
                    for ctrl in unrealizable:
                        held = self._tools.counterstrategy_held_inputs(ctrl, in_vars)
                        if held:
                            assume_blocks.append(held_hint(ctrl, held))
                except Exception:
                    assume_blocks = []
                # (B2) transition/persistence mining — SOLVER-VERIFIED, bounded. Justice cannot
                # express "the input must LAST long enough to be used": a 1-step grant pulse
                # satisfies G F(grant|!req) yet defeats G(nav -> grant) (the warehouse trap) —
                # and a pulsing environment pins nothing, so held_hint stays silent exactly then.
                # Candidates come only from the core's own variables; <=8 re-synthesis probes.
                try:
                    from contrauto.feedback.gr1_assume import (
                        mine_transitions_iterative, format_transition_feedback)
                    from contrauto.feedback.gr1_core import compute_core as _cc
                    for ctrl in unrealizable:
                        if cores.get(ctrl):
                            needed = mine_transitions_iterative(
                                self._tools, contract_content, ctrl, cores[ctrl],
                                recompute_core=lambda c, _ctrl=ctrl: _cc(self._tools, c, _ctrl))
                            if needed:
                                assume_blocks.append(format_transition_feedback(
                                    ctrl, needed if len(needed) == 1 else [],
                                    needed if len(needed) > 1 else None))
                except Exception:
                    pass

            crisp = "\n\n".join(core_blocks + assume_blocks)
            # VISIBILITY: surface the localized cause to stdout. It otherwise lives only in the
            # next-round prompt, making unrealizable repair a black box -- which hid why the
            # multi-contract (arbiter) coordination fails to converge. Now we can see, per round,
            # exactly which guarantees the QuickXplain core blamed and which assumption was mined.
            report.note(f"{_step('synthesis')}  unrealizable cause localized",
                        f"{len(core_blocks)} core block(s), {len(assume_blocks)} mined assumption(s)"
                        + ("" if crisp else " — nothing isolated"))
            if crisp:
                report.body(crisp)
            if self._fbmode('unrealizable') == 'crisp':
                # CRISP localized: the named core / assumption only, WITHOUT the counterstrategy trace.
                self._log("Step5:Synthesis", False, f"Unrealizable (crisp): {unrealizable}\n{crisp}")
                return False, {}, (
                    base_msg + "\n\n"
                    f"SYNTHESIS FAILED: contracts {unrealizable} are UNREALIZABLE (GR(1) synthesis).\n\n"
                    "LOCALIZED CAUSE (fix exactly this):\n"
                    f"{crisp or '(no minimal core isolated; weaken one guarantee or add an assumption)'}\n\n"
                    f"Current .ltl:\n{contract_content}\n\n"
                    "Weaken ONE named guarantee in the core, or add the named assumption. "
                    "Keep guarantees GR(1). Return raw .ltl text only.\n\n"
                )

            diagnosis = cstrat + (("\n\n" + crisp) if crisp else "")  # for the log
            self._log("Step5:Synthesis", False, f"Unrealizable: {unrealizable}\n{diagnosis}")
            # MESSAGE ORDER MATTERS: when a verified repair exists it must LEAD, the trace is
            # context, and the closing instructions must point back at it. With the patch buried
            # mid-message after the counterstrategy and the closing bullet saying "Relax
            # sub-contract Guarantees", gemini obeyed the tail and ignored a solver-proven
            # assumption set for nine consecutive rounds (guarantee-fiddling every time).
            lead = (f"LOCALIZED CAUSE & VERIFIED REPAIR (act on THIS first):\n{crisp}\n\n"
                    if crisp else "")
            closing = (
                "Fix the decomposition:\n"
                + ("- FIRST: apply the VERIFIED assumption set above VERBATIM, plus the matching "
                   "sibling guarantees per its CROSS-CHECK — it is solver-proven to restore "
                   "realizability.\n"
                   "- Only if that patch contradicts the intended environment behavior: weaken "
                   "ONE guarantee named in the core instead; touch nothing outside the core.\n"
                   if crisp else
                   "- Relax sub-contract Guarantees that conflict\n"
                   "- Move overly strict constraints to System contract only\n"
                   "- Ensure Until formulas do not block liveness goals\n")
                + "Return raw .ltl text only.\n\n")
            return False, {}, (
                base_msg + "\n\n"
                f"SYNTHESIS FAILED: contracts {unrealizable} are UNREALIZABLE (GR(1) synthesis).\n\n"
                + lead +
                "WHY (environment counterstrategy — how the environment forces the system to "
                "lose):\n"
                f"{cstrat}\n\n"
                "How to read it: the environment picks INPUT values each step; the trace shows a "
                "play the system cannot win. If the env exploits an input, add an ASSUMPTION that "
                "constrains it — justice G F(x) (it keeps coming), persistence G((x & c) -> X(x)) "
                "(it lasts while needed), or effect G(a -> X(e)) (actions take hold) — or weaken "
                "a core guarantee. Keep guarantees GR(1) (G F justice + safety; no G(p->F q) / "
                "bare F / U in Guarantees).\n\n"
                f"ORIGINAL LTL FORMULAS (do NOT remove any):\n{ltl_list}\n\n"
                f"VARIABLES:\n"
                f"  inputs : {variables.get('inputs', [])}\n"
                f"  outputs: {variables.get('outputs', [])}\n\n"
                f"Current .ltl:\n{contract_content}\n\n"
                + closing)

        if degenerate:
            who = ", ".join(c for (c, _, _) in degenerate)
            if self._fbmode('wellsep') == 'generic':
                # Generic baseline (RQ1): report only that the strategy is vacuous, withhold the
                # dead-end trace and the violated guarantee.
                self._log("Step5:Synthesis", False, f"Degenerate (generic): {who}")
                return False, {}, (
                    base_msg + "\n\n"
                    "SYNTHESIS FAILED: the strategy is vacuous (not well-separated). "
                    "Revise the contracts and try again.\n\n"
                    f"Current .ltl:\n{contract_content}\n\n"
                    "Return raw .ltl text only.\n\n"
                )
            ctrl0, did0, path0 = degenerate[0]
            try:
                from contrauto.feedback.diagnose import diagnose_degeneracy
                diag = diagnose_degeneracy(path0, path0[-1] if path0 else {}, contract_content, ctrl0)
            except Exception as e:
                on = {k: v for k, v in (path0[-1] if path0 else {}).items() if v}
                diag = (f"DEGENERATE (not well-separated): {ctrl0} reaches a dead-end at {on}. "
                        "Make the environment non-blocking; add the matching system-side "
                        f"exclusions or weaken the offending assumption. ({e})")
            # crisp: the named violated guarantee only; verbose: also append the dead-end path trace.
            if self._fbmode('wellsep') == 'crisp' or not path0:
                trace_block = ""
            else:
                steps = "\n".join(f"  step {i}: " + ", ".join(k for k, v in st.items() if v)
                                  for i, st in enumerate(path0))
                trace_block = f"\n\nDead-end path (initial -> dead-end):\n{steps}"
            self._log("Step5:Synthesis", False,
                      f"Degenerate (not well-separated): {who}\n{diag}")
            return False, {}, (
                base_msg + "\n\n"
                f"SYNTHESIS FAILED — degenerate strategy for: {who}\n\n"
                + diag + trace_block + "\n\n"
                f"Current (failing) .ltl:\n{contract_content}\n\n"
                "Return raw .ltl text only.\n\n"
            )

        if inactive:
            who = ", ".join(c for (c, _) in inactive)
            detail = "; ".join(f"{c}: never asserts {', '.join(d)}" for (c, d) in inactive)
            if self._fbmode('wellsep') == 'generic':
                self._log("Step5:Synthesis", False, f"Vacuous liveness (generic): {who}")
                return False, {}, (
                    base_msg + "\n\n"
                    "SYNTHESIS FAILED: the strategy is vacuous. Revise the contracts and "
                    "try again.\n\n"
                    f"Current .ltl:\n{contract_content}\n\n"
                    "Return raw .ltl text only.\n\n"
                )
            self._log("Step5:Synthesis", False, f"Vacuous liveness: {detail}")
            return False, {}, (
                base_msg + "\n\n"
                f"SYNTHESIS FAILED — VACUOUS LIVENESS for: {who}\n\n"
                f"{detail}.\n\n"
                "These outputs appear positively in G F(...) liveness guarantees (the "
                "component's own, or the System contract's), yet the synthesized winning "
                "strategy never asserts them in any reachable state: the strategy wins its "
                "liveness goals vacuously, through the environment's assumptions, by never "
                "acting. Two usual causes: (1) a safety rule makes asserting the output "
                "LOSING — most often an UNPRIMED persistence condition (G(out & cond -> "
                "X(out)) instead of G((out & cond & X(cond)) -> X(out))), whose forced hold "
                "collides with a no-spurious rule one step after the condition is withdrawn; "
                "revise the safety rules so acting is safe. (2) this component's own progress "
                "guarantee was DROPPED in a revision while the System contract still requires "
                "the behavior — refinement tolerates that only because the System's world "
                "assumptions discharge the liveness vacuously; RESTORE the component-level "
                "liveness guarantee (e.g. G F(!demand | action)).\n\n"
                f"Current (failing) .ltl:\n{contract_content}\n\n"
                "Return raw .ltl text only.\n\n"
            )

        if timed_out:
            who = ", ".join(timed_out)
            self._log("Step5:Synthesis", False, f"SLUGS timeout (> {SLUGS_SYNTH_TIMEOUT}s): {who}")
            return False, {}, (
                base_msg + contract_content + "\n\n"
                f"SLUGS synthesis for {who} did not finish within {SLUGS_SYNTH_TIMEOUT}s. This is "
                "a SOLVER SIZE problem, not necessarily a logic error: the contract's state space "
                "(inputs x outputs) is too large to synthesize in the time budget -- most often "
                "because a coordinator/arbiter contract carries too many near-duplicate mutex/grant "
                "clauses across many peers. Reduce the contract's variable/formula count where "
                "possible (e.g. drop any clause that is already implied by another, or split an "
                "over-broad coordinator into independent per-resource arbiters if the resources "
                "are never contended together) without dropping any required guarantee. "
                "Return raw .ltl text only.\n\n"
            )

        if slugs_errors:
            err_summary = "\n".join(slugs_errors)
            self._log("Step5:Synthesis", False, f"SLUGS synthesis errors:\n{err_summary}\n\n")
            return False, {}, (
                base_msg + contract_content + "\n\n"
                f"SLUGS synthesis errors:\n{err_summary}\n\n"
                "Fix the contract:\n"
                "- Assumptions must ONLY contain INPUT variables\n"
                "- OUTPUT variables must ONLY appear in Guarantees\n"
                "Return raw .ltl text only.\n\n"
            )

        self._log("Step5:Synthesis", True)
        return True, fsm_files, base_msg

    # ══════════════════════════════════════════════════════════════════════════
    # Counterexample helpers (thin delegates to the tool pipeline; kept on the
    # public surface because other modules / tests reference them here)
    # ══════════════════════════════════════════════════════════════════════════
    def extract_counterexample(self, nuxmv_output: str) -> list:
        """Parse a NuXMV trace into a list of (state_header, [var=val ...])."""
        return self._tools.extract_counterexample(nuxmv_output)

    def format_counterexample(self, counterexample: list) -> str:
        """Render a parsed counterexample as indented text."""
        return self._tools.format_counterexample(counterexample)

    # ══════════════════════════════════════════════════════════════════════════
    # Prompt builder
    # ══════════════════════════════════════════════════════════════════════════
    def _build_base_prompt(self, task, variables: dict, pairs: list) -> str:
        ltl_list = "\n".join(
            f"  [{i+1}] {p['nl']}\n       => {p['ltl']}"
            for i, p in enumerate(pairs)
        )

        skeleton = self._generate_contract_skeleton(task)

        return (
            f"SCENARIO:\n{task.format_for_prompt(include_decomposer_notes=True)}\n\n"
            f"LTL FORMULAS (already verified, use ALL of them):\n{ltl_list}\n\n"
            f"CONTRACT SKELETON (use this structure, fill in LTL formulas):\n"
            f"{skeleton}\n\n"
            "INSTRUCTIONS:\n"
            "- Use the contract skeleton above as your starting structure.\n"
            "- The input/output variables listed are FIXED (simulator-grounded and architecture-derived).\n"
            "- You MAY add additional internal variables if needed for decomposition.\n"
            "- Fill in Assumptions and Guarantees using the LTL formulas above.\n"
            "- INITIAL VALUES ARE MANAGED BY THE SYSTEM: do NOT write any initial-value line (a\n"
            "  bare `!x;` or `x;`). Write ONLY time-tense formulas (G, X, G F, G(p->X(q)) ...).\n"
            "  Any initial-value line you write is discarded and re-injected canonically, and the\n"
            "  initial value of a grant/communication INPUT is owned by its producer — never\n"
            "  assume it on the consumer side. (This overrides any 'pin every output' rule above.)\n"
            "- ACTION TRIGGERS MUST BE POSITIVE AND COLD-START-SAFE: never force an action merely\n"
            "  because a task/state is absent. For pickup navigation use\n"
            "  `G(task & !has_item & !at_pickup -> go_to_pickup)`, so the antecedent is false\n"
            "  at the all-false cold start; do NOT negate a conjunction such as\n"
            "  `G(!has_item & !(at_pickup & task) -> go_to_pickup)`.\n"
            "- STAY IN THE GR(1) FRAGMENT: safety (G over state formulas, G(p -> X(q))) and\n"
            "  justice (G F ...) only. NEVER use the U (until) operator: the safety half of an\n"
            "  until is expressible as G(...) rules, and its eventuality half belongs in a G F\n"
            "  fairness of whichever component can actually force it.\n"
            "- WORLD-OWNED SIGNALS: a variable that appears as an INPUT on every contract (it has\n"
            "  no producer) is driven by the world. Constrain it ONLY in Assumptions (e.g. its\n"
            "  release behaviour G F(!x)); NEVER write a guarantee about it.\n"
            "- PLACE LIVENESS WHERE IT IS FORCEABLE: a component may only promise G F(p) if it can\n"
            "  force p against every environment satisfying its assumptions. If a component's\n"
            "  progress depends on another component's service (e.g. being granted a shared\n"
            "  resource), write that dependency as an assumption (a G F fairness) and leave the\n"
            "  service liveness to the component that can deliver it (e.g. the coordinator).\n"
            "  Workers whose progress is entirely service-dependent should carry safety\n"
            "  guarantees only.\n"
            "- PRIME PERSISTENCE CONDITIONS: a guarantee that maintains an output while an\n"
            "  input condition stands must condition on the input's NEXT value, e.g.\n"
            "  G((grant & req & X(req)) -> X(grant)), NOT G(grant & req -> X(grant)). The\n"
            "  unprimed form forces the output one step past the condition's withdrawal;\n"
            "  combined with a no-spurious rule G(grant -> req) it makes every granting\n"
            "  state losing, and the synthesized strategy then wins vacuously by never\n"
            "  acting.\n"
            "- PARENTHESIZE ANTECEDENTS CONTAINING X: whenever an implication's antecedent\n"
            "  contains an X(...) term, wrap the WHOLE antecedent in parentheses:\n"
            "  G((a & X(b)) -> X(c)). Without them the parser attaches the implication to\n"
            "  the X term (reading X(b -> X c)), silently changing the formula's meaning.\n"
            "- System contract should contain end-to-end properties only.\n"
            "- Sub-contract guarantees must collectively imply System guarantees.\n"
            "Return raw .ltl text only."
        )

    def _check_grounding(self, content: str):
        """Verify every declared variable maps to the target simulator's grounding vocabulary.

        The contract's atomic propositions must be the simulator's own predicates/actions, else
        bind_by_role/role_of cannot connect a variable to a physical grounding and the FSM silently
        deadlocks in execution (the failure mode behind the realizable-yet-delivered=0 runs). A
        variable GROUNDS iff it is literally in the task's vocabulary OR role_of knows it —
        exact membership must come FIRST: role_of only classifies the warehouse-family names
        (shelf/dock/grant/item/...), so checking role_of alone rejected verbatim vocabulary
        names of every other domain (patrol's at_A/inspect/report), an unfixable 9/9 loop.
        Returns a feedback message (naming the offending variables and the allowed vocabulary)
        or None if all variables ground. No-op when the simulator vocabulary is unknown
        (naming-agnostic fallback)."""
        if not self.ground_vocab:
            return None
        declared = []
        for ln in content.splitlines():
            m = re.search(r"\bvariable\s+(\w+)", ln)
            if m and m.group(1) not in declared:
                declared.append(m.group(1))
        if self.ground_binder is not None:
            # judge with the simulator family's OWN runtime binder — check == binding
            ungrounded = [v for v in declared if not self.ground_binder(v)]
        else:
            from contrauto.sim_iface import role_of
            vocab = set(self.ground_vocab.get("sensors", [])) \
                | set(self.ground_vocab.get("actions", []))
            ungrounded = [v for v in declared if v not in vocab and role_of(v) is None]
        if not ungrounded:
            return None
        sensors = ", ".join(self.ground_vocab.get("sensors", []))
        actions = ", ".join(self.ground_vocab.get("actions", []))
        lines = [
            "[UNGROUNDED VARIABLE — will not bind to the simulator]",
            f"These declared variables map to NO physical grounding: {', '.join(ungrounded)}.",
            "Every atomic proposition must be one of the simulator's grounding names (or an obvious",
            "synonym of one). Rewrite the contract using ONLY this vocabulary:",
            f"    sensor inputs : {sensors}",
            f"    action outputs: {actions}",
            "For example use `has_item` (not `carrying`), `task` (not `task_present`). Keep the "
            "logic identical; only rename variables to the grounding vocabulary.",
        ]
        return "\n".join(lines)

    def _enforce_initial_conditions(self, content: str) -> str:
        """Deterministically OWN the initial conditions so the System and the sub-contracts can
        never disagree on them — the failure mode behind the N>=5 refinement non-convergence.

        The LLM writes only the TIME-TENSE logic (G / F / X / U ...). Here we DROP whatever
        initial-proposition lines it wrote and re-inject the canonical ones:
          * every OUTPUT of a contract       -> `!out;`  in that contract's Guarantees
          * every real ENVIRONMENT input     -> `!in;`   in that contract's Assumptions
          * an INTERNAL signal (produced by one contract AND consumed by another — e.g. a
            grant) is NOT pinned on the CONSUMER side: its initial value is guaranteed by the
            PRODUCER, so the consumer must not assume it. (That unsatisfiable consumer
            assumption — Robot assuming `!grant` while System cannot, since grant is its
            output — is exactly what made the assumption-direction refinement fail.)

        Initial value defaults to false (cold start) — domain-independent for reactive control.
        A spec could later override specific variables to start true; this only sets the default.
        """
        decl_in = re.compile(r"\s*input\s+boolean\s+variable\s+(\w+)\s*;")
        decl_out = re.compile(r"\s*output\s+boolean\s+variable\s+(\w+)\s*;")

        all_in, all_out = set(), set()
        for ln in content.splitlines():
            mi, mo = decl_in.match(ln), decl_out.match(ln)
            if mi:
                all_in.add(mi.group(1))
            if mo:
                all_out.add(mo.group(1))
        internal = all_in & all_out      # produced AND consumed -> internal communication signal

        def is_initial(s: str) -> bool:
            """An initial-condition line: a single (optionally negated) atom or `true`, no
            temporal operator / parens — as opposed to a G(...)/F(...)/... time-tense formula."""
            t = s.strip().rstrip(";").strip()
            return t == "true" or bool(re.fullmatch(r"!?\s*[A-Za-z_]\w*", t))

        out: list[str] = []
        decls: list[str] = []
        cur_in: list[str] = []
        cur_out: list[str] = []
        keep_a: list[str] = []
        keep_g: list[str] = []
        section = None
        in_contract = False

        def flush() -> None:
            if not in_contract:
                return
            out.extend(decls)
            env_pins = [f"        !{v};" for v in cur_in if v not in internal]
            a_body = env_pins + keep_a or ["        true;"]   # never leave Assumptions empty
            out.append("    Assumptions:")
            out.extend(a_body)
            out.append("    Guarantees:")
            out.extend([f"        !{v};" for v in cur_out] + keep_g)
            out.append("")

        for ln in content.splitlines():
            stripped = ln.strip()
            if stripped.startswith("CONTRACT ") and stripped.endswith(":"):
                flush()
                in_contract, section = True, None
                decls = [ln]
                cur_in, cur_out, keep_a, keep_g = [], [], [], []
                continue
            if not in_contract:
                out.append(ln)
                continue
            mi, mo = decl_in.match(ln), decl_out.match(ln)
            if mi:
                cur_in.append(mi.group(1)); decls.append(ln); continue
            if mo:
                cur_out.append(mo.group(1)); decls.append(ln); continue
            if stripped == "Assumptions:":
                section = "A"; continue
            if stripped == "Guarantees:":
                section = "G"; continue
            if not stripped or stripped.startswith("#") or is_initial(stripped):
                continue                                     # drop blanks/comments/initial lines
            keep_line = ln if ln.startswith("    ") else "        " + stripped
            if section == "A":
                keep_a.append(keep_line)
            elif section == "G":
                keep_g.append(keep_line)
        flush()
        return "\n".join(out).rstrip() + "\n"

    def _activity_witness(self, json_path, contract_content, ctrl, liveness_from=None):
        """LIVENESS-VACUITY (activity) check: every output of `ctrl` that appears
        POSITIVELY inside a `G F(...)` guarantee must be asserted in at least one
        reachable strategy state. A strategy that wins its liveness goals without ever
        asserting its own contribution is winning vacuously through environment
        assumptions (e.g. an arbiter that never grants, satisfying GF(!req | grant) via
        the world's GF(!req)) — a vacuity flavor the dead-end check cannot see, because
        nothing dead-ends in a strategy that never acts.

        `liveness_from` selects WHOSE liveness guarantees are scanned (default: ctrl's
        own contract). Passing "System" closes the second hole: a repair round may DROP
        a component's progress obligation while the System contract still states it —
        refinement tolerates this (the System's world assumptions discharge the service
        liveness vacuously), so the System's liveness outputs must additionally be
        checked against the strategy of the component that OWNS each output.

        Returns the never-asserted liveness outputs."""
        try:
            d = json.loads(Path(json_path).read_text())
        except Exception:
            return []
        nodes = d.get("nodes", {})
        variables = d.get("variables", [])
        block = re.search(rf"CONTRACT {re.escape(ctrl)}:(.*?)(?=CONTRACT |\Z)",
                          contract_content, re.S)
        if not block:
            return []
        outputs = re.findall(r"output boolean variable (\w+);", block.group(1))
        src = liveness_from or ctrl
        src_block = re.search(rf"CONTRACT {re.escape(src)}:(.*?)(?=CONTRACT |\Z)",
                              contract_content, re.S)
        if not src_block:
            return []
        _, _, g_part = src_block.group(1).partition("Guarantees:")
        live_pos = set()
        for m in re.finditer(r"G\s*F\s*\(([^;]*)\)\s*;", g_part):
            expr = m.group(1)
            for v in outputs:
                for occ in re.finditer(rf"(!?)\s*\b{re.escape(v)}\b", expr):
                    if occ.group(1) != "!":
                        live_pos.add(v)
                        break
        if not live_pos:
            return []
        asserted = set()
        for node in nodes.values():
            st = node.get("state", [])
            for i, v in enumerate(variables):
                if v in live_pos and i < len(st) and st[i]:
                    asserted.add(v)
        return sorted(live_pos - asserted)

    def _generate_contract_skeleton(self, task) -> str:
        """Auto-generate a CHASE contract skeleton from the architecture graph.

        Variables come directly from task.nodes (already prefixed by schema.py).
        Source 1: simulator-grounded propositions (from physics)
        Source 2: architecture-derived propositions (coordinator overrides)
        Source 3: LLM-proposed (added by LLM as needed, not generated here)
        """
        lines = [f"NAME: {task.name};", ""]

        for node in task.nodes:
            lines.append(f"CONTRACT {node.name}:")
            for var in node.prop_inputs:
                lines.append(f"    input boolean variable {var};")
            for var in node.prop_outputs:
                lines.append(f"    output boolean variable {var};")
            lines.append("    # You may add more variables if needed (LLM-proposed)")
            lines.append("    Assumptions:")
            lines.append("        # Fill in assumptions using LTL formulas")
            lines.append("    Guarantees:")
            lines.append("        # Fill in guarantees using LTL formulas")
            lines.append("")

        lines.append("CONTRACT System:")
        all_inputs = set()
        all_outputs = set()
        for node in task.nodes:
            all_inputs.update(node.prop_inputs)
            all_outputs.update(node.prop_outputs)
        env_only = all_inputs - all_outputs
        for var in sorted(env_only):
            lines.append(f"    input boolean variable {var};")
        for var in sorted(all_outputs):
            lines.append(f"    output boolean variable {var};")
        lines.append("    Assumptions:")
        lines.append("        # System-level environment assumptions")
        lines.append("    Guarantees:")
        lines.append("        # System-level end-to-end guarantees")
        lines.append("")

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════════
    # Low-level utilities
    # ══════════════════════════════════════════════════════════════════════════
    def _save(self, path, content: str):
        """Write `content` to `path` (overwriting)."""
        Path(path).write_text(content)

    def _strip(self, text: str) -> str:
        """Remove markdown code fences if LLM wraps output in them."""
        return re.sub(r'```[a-z]*\n?|```', '', text).strip()
