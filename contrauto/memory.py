"""Agent memory: EpisodicMemory, a per-design-problem LLM-distilled lesson.

Maintains a single persisted lesson: after every failure it calls SummarizerAgent
(contrauto/agents/summarizer.py) to FOLD the new diagnosis into the lesson so far — merge and
escalate if the failure recurs, drop whatever is no longer the most important thing to say, and
stay under a hard character budget. It is a distillation, not a log: nothing is kept just
because it happened, only what is still worth telling the next attempt. Written through to
`lessons_path` (via SummarizerAgent.run) so a later rerun of the SAME task (e.g. --resynth)
starts from where the last run left off.

There used to be a second, cross-task SemanticMemory role here (a global memory.md of lessons
transferring to OTHER tasks, opt-in via --semantic-memory). That has been retired: the
Summarizer's merge machinery now serves EpisodicMemory exclusively, at per-problem scope.
"""
from __future__ import annotations

from pathlib import Path


class EpisodicMemory:
    """Per-design-problem memory: one persisted, LLM-distilled lesson, capped in size.

    Was DecomposerAgent._remember/_accumulate (which kept a growing list of raw diagnoses).
    That is gone: every failure is folded into ONE lesson by calling SummarizerAgent.run(),
    capped at `lesson_cap` characters, so the prompt sent on the next attempt carries only what
    is still the most important thing to fix — not an ever-growing transcript. The merge is
    SummarizerAgent's own file-based operation (it reads/writes `lessons_path` itself, keyed by
    `mission`), so BOTH `model` and `lessons_path` are required for the lesson mechanism to run
    at all -- omitting either one (e.g. in tests) makes `remember()` a no-op and `render()`
    return `base` unchanged.
    """

    def __init__(self, cap: int = 4000, model: str | None = None,
                 lessons_path: str | Path | None = None, lesson_cap: int = 1500,
                 mission: str = "episodic"):
        self.cap = cap                # bound on the raw diagnosis text fed INTO the merge call
        self.lesson_cap = lesson_cap  # hard character budget for the distilled lesson itself
        self.model = model            # LLM used to merge/upgrade the lesson; None -> disabled
        self.mission = mission        # SummarizerAgent's <!-- mission: --> key in lessons_path
        self.lessons_path = Path(lessons_path) if lessons_path else None
        self._lesson = ""
        if self.lessons_path and self.lessons_path.exists():
            from contrauto.agents.summarizer import SummarizerAgent
            existing = self.lessons_path.read_text(encoding="utf-8")
            self._lesson = SummarizerAgent._extract_lesson(existing, self.mission)
        self._n_failures = 0

    def remember(self, msg: str, base: str = "") -> None:
        """Strip the static base prompt off this round's diagnosis, bound it, and fold it into
        the persisted lesson (one LLM call, best-effort)."""
        diag = (msg.replace(base, "") if base else msg).strip()[: self.cap]
        if not diag:
            return
        self._n_failures += 1
        self._update_lesson(diag)

    def _update_lesson(self, diag: str) -> None:
        """Merge `diag` into the persisted lesson via SummarizerAgent.run() (one LLM call);
        write-through to disk happens inside that call. Best-effort: a summarization failure
        (e.g. a network hiccup) must not crash the decomposer's retry loop, so it just leaves
        the lesson unchanged on error."""
        if not self.model or not self.lessons_path:
            return
        from contrauto.agents.summarizer import SummarizerAgent
        summarizer = SummarizerAgent(model=self.model)
        step_logs = [{"step": "Decomposer failure", "passed": False, "detail": diag}]
        try:
            summarizer.run(step_logs, mission=self.mission,
                           memory_file=str(self.lessons_path), cap=self.lesson_cap)
        except Exception:
            return
        content = self.lessons_path.read_text(encoding="utf-8")
        self._lesson = SummarizerAgent._extract_lesson(content, self.mission)

    def render(self, base: str) -> str:
        """base prompt + the distilled lesson (diagnosis + fix, escalated on repeats), if one
        has been built. No raw failure transcript — the lesson IS the memory."""
        if not self._lesson:
            return base
        return (base + "\n\n=== LESSON FROM THIS PROBLEM'S PRIOR FAILURES (distilled, updated "
                "after every failure — a point marked recurring means your last fix did not "
                "work; read it instead of repeating the same change) ===\n" + self._lesson)

    def __len__(self) -> int:
        return self._n_failures
