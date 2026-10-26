"""Translator Agent: architecture + NL requirements → variables + LTL formulas."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from contrauto import report
from contrauto.llm import chat, model_name, parse_json_response, MAX_RETRIES
from contrauto.prompts import TRANSLATOR_INSTRUCTIONS
from contrauto.schema import TaskSpec, _node_prefix
from contrauto.translation_check import check_realizable, complete, coverage, lint

# Targeted LLM repair rounds for untranslated requirements / lint issues.
REPAIR_ROUNDS = 2
# Reasoning effort for the translator call. The physics-derived formula families that a
# lower effort tends to drop are completed mechanically (translation_check.complete), so a
# faster setting keeps full coverage; TRANSLATOR_EFFORT overrides ("" = provider default).
TRANSLATOR_EFFORT = {"gemini": "medium"}


class TranslatorAgent:
    """LLM agent turning a TaskSpec's architecture + NL requirements into LTL pairs."""

    def __init__(self, model: str, ground_vocab: dict = None):
        self.model = model
        # Simulator grounding vocabulary (sensors/actions). When the task YAML declares no
        # propositions, the translator used to INVENT names (r1_to_pickup) that drift from
        # the simulator binder (nav_pickup) — the decomposer then renames under Step 1.6
        # pressure and the mission formulas can no longer be anchored textually. Pinning
        # the vocabulary HERE keeps names stable end-to-end (translator -> System contract
        # -> sub-contracts -> FSM -> physical binding).
        self.ground_vocab = ground_vocab
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._workdir = None           # scratch dir for the translation realizability check

    def run(self, task: TaskSpec, cache_path: str | Path | None = None,
            refresh: bool = False) -> dict:
        """Translate `task` to LTL, retrying on malformed JSON.

        If `cache_path` holds a previous translation made from the exact same prompt and
        model (matching ``cache_key``), it is returned without an LLM call; `refresh`
        forces a fresh translation.

        Returns:
            The parsed LLM response (an ``ltl_pairs`` list plus a ``usage`` token tally).

        Raises:
            RuntimeError: If valid JSON is not produced within MAX_RETRIES attempts.
        """
        self._workdir = (Path(cache_path).parent / "workdir" / "_translation_check"
                         if cache_path else None)

        # YAML-declared propositions; empty for vocabulary-grounded tasks (the usual case).
        all_props = task.all_propositions()

        vocab_block = ""
        if all_props:
            vocab_block = (f"\n\nATOMIC PROPOSITIONS (use ONLY these variable names):\n"
                           f"  {all_props}\n")
        elif self.ground_vocab:
            sensor_list = self.ground_vocab.get("sensors", [])
            action_list = self.ground_vocab.get("actions", [])
            sensors = ", ".join(sensor_list)
            actions = ", ".join(action_list)
            signals = self.ground_vocab.get("signals") or {}
            signal_lines = ""
            if signals.get("requests"):
                signal_lines += (f"  requests (robot outputs -> Arbiter inputs): "
                                 f"{', '.join(signals['requests'])}\n")
            if signals.get("grants"):
                signal_lines += (f"  grants (Arbiter outputs -> robot inputs)  : "
                                 f"{', '.join(signals['grants'])}\n")
            # Prefix examples drawn from the real vocabulary, so the example never names a
            # variable that does not bind.
            example = ", ".join(f"r{i}_{name}" for i, name in
                                enumerate((action_list[:1] + sensor_list[:1]) or ["x"], 1))
            vocab_block = (
                "\n\nVARIABLE NAMING (MANDATORY — these are the physical simulator's own "
                "grounding names; any other name will not bind):\n"
                f"  sensor inputs : {sensors}\n"
                f"  action outputs: {actions}\n"
                + signal_lines +
                "For per-agent instances prefix with the agent id (r1_, r2_, ...): e.g. "
                f"{example}. Shared/global signals take no prefix"
                + (f" ({', '.join(self.ground_vocab['globals'])})"
                   if self.ground_vocab.get("globals") else "") + ". Do "
                "NOT invent synonyms (no goto_/to_/move_ variants).\n"
                "Every completion/goal ACTION must be gated on a sensor that witnesses its "
                "physical effect, and the witness must be a PERSISTENT sensor (position/"
                "carrying state), NOT a one-step pulse: e.g. G(dropoff -> at_dock & has_item). "
                "Put the liveness GOAL on the action that actually completes the mission "
                "(G F(dropoff)), not on a bookkeeping flag such as `done`; one-step pulses "
                "(delivered, acks) are things to REACT to "
                "in the step they appear, never goal gates — gating a recurring goal on a "
                "pulse stalls the controller after its first success.\n"
                + (f"TIMING SEMANTICS of these groundings (write formulas that match the "
                   f"physics):\n  {self.ground_vocab['notes']}\n"
                   if self.ground_vocab.get("notes") else ""))
        user_msg = (
            task.format_for_prompt()
            + vocab_block
            + "\nReturn JSON only."
        )

        # Any change to the task YAML, grounding vocabulary, prompt or model changes the key.
        effort = os.environ.get("TRANSLATOR_EFFORT", TRANSLATOR_EFFORT.get(self.model, ""))
        cache_key = hashlib.sha256("\0".join(
            (self.model, effort, TRANSLATOR_INSTRUCTIONS, user_msg)).encode()).hexdigest()
        cached = None
        if cache_path:
            try:
                cached = json.loads(Path(cache_path).read_text())
            except (OSError, ValueError):
                cached = None
        hit = bool(cached and not refresh and cached.get("cache_key") == cache_key
                   and cached.get("ltl_pairs"))
        # --retranslate never reads the cache, so it is not listed among the inputs.
        self._show_inputs(task, cache_path if cached and not refresh else None,
                          "cache hit — inputs unchanged" if hit else
                          "stale — inputs changed, translating again")
        report.agent("Translator", "NL requirements → LTL")
        if cache_path and not refresh:
            if hit:
                self._show(cached["ltl_pairs"], "no LLM call (--retranslate to redo)",
                           source="cache")
                cached["usage"] = {"input_tokens": 0, "output_tokens": 0}
                return cached

        t0 = time.monotonic()
        for attempt in range(MAX_RETRIES):
            with report.spinner("Translating"):
                raw = self._chat(user_msg)
            try:
                parsed = parse_json_response(raw)
                parsed.setdefault("not_applicable", [])
                parsed["ltl_pairs"], notes, summary = self._verify(task, parsed, user_msg)
                report.took(time.monotonic() - t0)       # total, including any repair round
                self._show(parsed["ltl_pairs"])
                summary()
                parsed["completion"] = notes
                parsed["cache_key"] = cache_key
                parsed["usage"] = {
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                }
                return parsed
            except Exception as e:
                report.fail(f"attempt {attempt + 1}: invalid JSON", str(e))
                user_msg = user_msg + f"\nJSON error: {e}. Return valid JSON only."

        raise RuntimeError("[Translator] Failed after max retries")

    def _show_inputs(self, task: TaskSpec, cache_path, cache_state: str):
        """`⏺ Read(...)` for everything the translator prompt is built from."""
        def rel(p):
            r = os.path.relpath(p)
            return p if r.startswith("..") else r
        plural = lambda n, w: f"{n} {w}" + ("" if n == 1 else "s")
        robots = sum(1 for n in task.nodes if re.fullmatch(r"Robot\d+", n.name))
        for i, src in enumerate(task.sources):
            if i == 0:
                report.read(rel(src), " · ".join((plural(len(task.requirements), "requirement"),
                                                  plural(len(task.nodes), "node"),
                                                  plural(len(task.edges), "edge"),
                                                  plural(robots, "robot"))))
            else:
                report.read(rel(src), "referenced by the task YAML")
        v = self.ground_vocab or {}
        if v.get("source"):
            n_sig = sum(len(x) for x in (v.get("signals") or {}).values())
            report.read(rel(v["source"]),
                        f"vocabulary: {len(v.get('sensors', []))} sensors · "
                        f"{len(v.get('actions', []))} actions · {n_sig} signals · "
                        f"{len(v.get('action_effects') or {})} action effects")
        import contrauto.prompts.translator as prompt_mod
        report.read(rel(prompt_mod.__file__),
                    f"system prompt · {len(TRANSLATOR_INSTRUCTIONS.split()):,} words")
        if cache_path:
            report.read(rel(cache_path), cache_state)

    def _chat(self, user_msg: str) -> str:
        effort = os.environ.get("TRANSLATOR_EFFORT", TRANSLATOR_EFFORT.get(self.model, ""))
        raw, usage = chat(self.model, TRANSLATOR_INSTRUCTIONS, user_msg,
                          reasoning_effort=effort or None)
        self.total_input_tokens += usage["input_tokens"]
        self.total_output_tokens += usage["output_tokens"]
        return raw

    def _verify(self, task: TaskSpec, parsed: dict, user_msg: str):
        """Complete, then check coverage + lint + realizability; send only the gaps back to
        the LLM for a targeted repair (at most REPAIR_ROUNDS)."""
        pairs, na = parsed["ltl_pairs"], parsed["not_applicable"]
        prefixes = [_node_prefix(n.name) for n in task.nodes if re.fullmatch(r"Robot\d+", n.name)]
        pairs, notes = complete(pairs, self.ground_vocab, prefixes)
        n_req = len(task.requirements)
        realizable, core = None, []
        for _ in range(REPAIR_ROUNDS + 1):
            missing = coverage(pairs, n_req, na)
            issues = lint(pairs, self.ground_vocab, prefixes, task.all_propositions())
            realizable, core = None, []
            if not missing and not issues:
                realizable, core = self._realizable(pairs, prefixes)
                if realizable is False:
                    issues = ([f"[{i + 1}] {pairs[i]['ltl']}: in the minimal UNREALIZABLE core"
                               for i in core] or
                              ["the world assumptions make every controller lose (core lies in "
                               "the assumptions)"])
            if not missing and not issues:
                break
            if _ == REPAIR_ROUNDS:
                break
            report.note(f"repairing translation " + report.dim("[LLM]"),
                        f"{len(missing)} untranslated requirement(s), {len(issues)} issue(s)")
            for k in missing:
                report.body(f"untranslated: requirement {k}")
            for issue in issues:
                report.body(issue)
            listing = "\n".join(f"  [{i}] {p['ltl']}   # req {p.get('req', [])}: {p['nl']}"
                                 for i, p in enumerate(pairs, 1))
            problems = [f"- requirement {k} has NO formula: {task.requirements[k - 1]}"
                        for k in missing] + [f"- {issue}" for issue in issues]
            if realizable is False:
                problems.append(
                    "- The formulas marked as the UNREALIZABLE core cannot all hold together: "
                    "SLUGS proved no controller satisfies them under the world assumptions. At "
                    "least one of them is mistranslated (too strong, or missing a condition the "
                    "requirement states, e.g. 'not yet at the dock'). Re-read the requirement "
                    "text and correct the wrong formula(s); do not drop a requirement.")
            repair_msg = (user_msg + "\n\nYOUR TRANSLATION SO FAR:\n" + listing
                          + "\n\nPROBLEMS TO FIX:\n" + "\n".join(problems)
                          + '\n\nReturn JSON only: {"replace": {"<index>": {"nl": ..., "ltl": ..., '
                          '"req": [k]}}, "add": [{"nl": ..., "ltl": ..., "req": [k]}]} -- replace '
                          "every flagged formula (by its [index]; an empty ltl deletes it) and add "
                          "formulas for every untranslated requirement (or, if it is vacuous for "
                          'this instance, list it as "not_applicable": [{"req": k, "reason": ...}]). '
                          "Do not touch the rest.")
            with report.spinner("Repairing translation"):
                raw = self._chat(repair_msg)
            try:
                fix = parse_json_response(raw)
            except Exception as e:
                report.fail("repair returned invalid JSON", str(e))
                continue
            for idx, new in (fix.get("replace") or {}).items():
                try:
                    pairs[int(idx) - 1] = new
                except (ValueError, IndexError):
                    pass
            pairs = [p for p in pairs if p.get("ltl", "").strip()] + list(fix.get("add") or [])
            na = na + list(fix.get("not_applicable") or [])
            parsed["not_applicable"] = na
            pairs, more = complete(pairs, self.ground_vocab, prefixes)
            notes += more

        def summary():
            if missing or (issues and realizable is not False):
                report.fail("translation still incomplete after repair " + report.dim("[Python]"),
                            f"untranslated requirement(s) {missing}; {len(issues)} issue(s)")
                for issue in issues:
                    report.body(issue)
            else:
                report.ok(f"all {n_req} requirements translated " + report.dim("[Python]"),
                          "coverage, vocabulary, GR(1) shape")
            for item in na:
                report.note(f"requirement {item.get('req')} not applicable here",
                            str(item.get("reason", "")))
            if notes:
                report.note(f"{len(notes)} mechanical completion(s) "
                            + report.dim("[Python · simulator physics]"), "marked [auto] above")
            tag = report.dim("[CHASE logics_tool + SLUGS]")
            if realizable is True:
                report.ok(f"translation realizable {tag}", "one GR(1) game over all formulas")
            elif realizable is False:
                report.fail(f"translation unrealizable {tag}",
                            "minimal core below; the Decomposer cannot succeed until it is fixed")
                for i in core:
                    report.body(pairs[i]["ltl"])
            else:
                report.note(f"realizability check skipped {tag}",
                            "tools unavailable or no verdict")
        return pairs, notes, summary

    def _realizable(self, pairs: list, prefixes: list):
        """SLUGS verdict on the whole translation as one GR(1) game (see translation_check)."""
        if not self.ground_vocab or self._workdir is None:
            return None, []
        from contrauto.agents.tool_pipeline import ToolPipeline
        self._workdir.mkdir(parents=True, exist_ok=True)
        tools = ToolPipeline(self._workdir, "translation", self._workdir / "spec.ltl",
                             self._workdir / "verification.chase", self._workdir / "synthesis.chase")
        with report.spinner("Checking translation realizability (SLUGS)"):
            try:
                return check_realizable(tools, pairs, self.ground_vocab, prefixes)
            except Exception:
                return None, []

    def _show(self, pairs: list, detail: str = "", source: str = ""):
        effort = os.environ.get("TRANSLATOR_EFFORT", TRANSLATOR_EFFORT.get(self.model, ""))
        tag = source or ("LLM · " + model_name(self.model) + (f" · {effort}" if effort else ""))
        report.ok(f"Translated {report.dim('[' + tag + ']')}  {len(pairs)} LTL formulas", detail)
        width = min(48, max((len(p["ltl"]) for p in pairs), default=0))
        for p in pairs:
            tag = ("req " + ", ".join(f"#{k}" for k in p["req"])) if p.get("req") else "auto"
            report.log(f"       {p['ltl']:<{width}}  " + report.dim(f"# [{tag}] {p['nl']}"))
