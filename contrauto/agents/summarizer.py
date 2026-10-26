"""Summarizer Agent: step logs → reusable lessons in memory.md."""

from __future__ import annotations

import re
from pathlib import Path

from contrauto import report
from contrauto.llm import chat
from contrauto.prompts import SUMMARIZER_INSTRUCTIONS


class SummarizerAgent:
    """LLM agent distilling a run's step logs into one synthesized lesson in memory.md."""

    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def run(self, step_logs: list, mission: str, memory_file: str = "memory.md",
            cap: int | None = None) -> None:
        """Merge the current run with any existing lesson for `mission` and rewrite memory_file.

        `cap`: optional hard character budget for the upgraded lesson (told to the model AND
        enforced by truncation on the returned text) -- used by EpisodicMemory, which calls this
        once per FAILURE (not once per finished run) and needs the lesson to stay short across
        many merges rather than grow without bound."""
        memory_path = Path(memory_file)
        existing = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""

        old_lesson = self._extract_lesson(existing, mission)
        other_lessons = self._remove_lesson(existing, mission)

        log_text = "\n".join(
            f"[{'PASS' if l['passed'] else 'FAIL'}] {l['step']}"
            + (f"\n  {l['detail']}" if l['detail'] else "")
            for l in step_logs
        )
        final = step_logs[-1] if step_logs else {}

        prompt = (
            f"## Task: {mission}\n\n"
            f"## Pipeline Step Logs (current run)\n{log_text}\n\n"
            f"## Overall Result: {'SUCCESS' if final.get('passed') else 'FAILED'}\n\n"
            + (
                f"## Existing Lesson for This Task\n{old_lesson}\n\n"
                "You must synthesize the existing lesson and the current run experience "
                "into ONE improved lesson. Requirements:\n"
                "- Do NOT simply append or concatenate the two lessons\n"
                "- Identify common failure patterns across both runs and generalize them\n"
                "- Merge overlapping fix rules into broader, more robust design principles\n"
                "- Remove outdated or contradicted points from the old lesson\n"
                "- The result should be MORE insightful than either lesson alone\n\n"
                if old_lesson else
                "No existing lesson for this task. Summarize the current run.\n\n"
            ) +
            "Output a single synthesized lesson in this format:\n"
            f"### Lesson: {mission}\n"
            "- **Failure pattern**: what went wrong (if anything)\n"
            "- **Root cause**: why it failed\n"
            "- **Fix strategy**: concrete rules to prevent recurrence\n"
            + (f"\nHard limit: the lesson body MUST be under {cap} characters. If keeping "
               "everything would exceed that, drop resolved/least-important points first.\n"
               if cap else "")
        )

        with report.spinner("Summarizing the failure into a lesson"):
            text, usage = chat(self.model, SUMMARIZER_INSTRUCTIONS, prompt)
        self.total_input_tokens += usage["input_tokens"]
        self.total_output_tokens += usage["output_tokens"]

        lesson_text = text.strip()
        if cap:
            lesson_text = lesson_text[:cap]

        lesson_block = (
            f"\n---\n\n<!-- mission: {mission} -->\n"
            + lesson_text
            + "\n"
        )
        updated = other_lessons.rstrip() + "\n" + lesson_block
        memory_path.write_text(updated, encoding="utf-8")
        report.note("lesson updated", f"{memory_file}  (in={usage['input_tokens']} "
                                      f"out={usage['output_tokens']})")

    @staticmethod
    def _extract_lesson(content: str, mission: str) -> str:
        """Return the existing lesson block for `mission`, or "" if none. Static: pure string
        parsing, callable without an LLM-backed instance (see EpisodicMemory)."""
        pattern = rf"(<!-- mission: {re.escape(mission)} -->.*?)(?=<!-- mission:|\Z)"
        m = re.search(pattern, content, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _remove_lesson(content: str, mission: str) -> str:
        """Return `content` with the lesson block for `mission` stripped out."""
        pattern = rf"\n---\n\n<!-- mission: {re.escape(mission)} -->.*?(?=\n---\n|<!-- mission:|\Z)"
        return re.sub(pattern, "", content, flags=re.DOTALL)
