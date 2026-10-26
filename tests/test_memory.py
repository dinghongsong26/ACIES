"""Tests for EpisodicMemory (contrauto/memory.py).

Within-problem, LLM-distilled lesson capped in size (was decomposer's raw-notes
_remember/_accumulate; now a single merged lesson, built by calling SummarizerAgent.run() after
every failure -- not a growing transcript). Calls SummarizerAgent's merge primitive (in
contrauto/agents/summarizer.py), so these tests mock `contrauto.agents.summarizer.chat` (the
name bound inside that module), not `contrauto.llm.chat` directly.

Run:  python3 -m pytest tests/test_memory.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.memory import EpisodicMemory


def test_episodic_no_model_is_noop():
    """Without a model, remember() must not call the LLM and render() returns base unchanged."""
    m = EpisodicMemory()
    m.remember("BASEdiag-A", base="BASE")
    assert len(m) == 1                    # failure count still tracked
    assert m.render("BASE") == "BASE"     # but no lesson was built -> nothing added


def test_episodic_render_empty_is_base():
    assert EpisodicMemory().render("BASE") == "BASE"


def test_episodic_merges_not_appends(tmp_path):
    """Each failure triggers one SummarizerAgent.run() call that replaces (not appends to) the
    lesson, and is given the prior lesson as context to merge into."""
    calls = []

    def fake_chat(model, system, user, temperature=0.0):
        calls.append(user)
        return (f"### Lesson so far\n- distilled after {len(calls)} failure(s)",
                {"input_tokens": 1, "output_tokens": 1})

    with mock.patch("contrauto.agents.summarizer.chat", fake_chat):
        m = EpisodicMemory(model="claude", lessons_path=tmp_path / "episodic_lessons.md")
        m.remember("diag A")
        first = m._lesson
        m.remember("diag B")
        second = m._lesson

    assert len(calls) == 2
    assert "diag A" in calls[0] and "No existing lesson for this task" in calls[0]
    assert "diag B" in calls[1] and "distilled after 1 failure(s)" in calls[1]  # merges the 1st
    assert second != first
    assert "distilled after 1" not in second  # replaced, not appended


def test_episodic_lesson_is_capped(tmp_path):
    """The lesson is hard-truncated to lesson_cap even if the LLM ignores the budget, and the
    cap is communicated to the merge call."""
    def fake_chat(model, system, user, temperature=0.0):
        assert "under 200 characters" in user
        return ("x" * 5000, {"input_tokens": 1, "output_tokens": 1})

    with mock.patch("contrauto.agents.summarizer.chat", fake_chat):
        m = EpisodicMemory(model="claude", lessons_path=tmp_path / "l.md", lesson_cap=200)
        m.remember("diag A")
        # _lesson includes the small, fixed "<!-- mission: ... -->\n" marker line ahead of the
        # capped LLM text -- that prefix doesn't grow across retries, only the body is capped.
        body = m._lesson.split("\n", 1)[1]
        assert len(body) == 200


def test_episodic_persists_and_reloads(tmp_path):
    """A later EpisodicMemory for the same lessons_path picks up the persisted lesson."""
    path = tmp_path / "episodic_lessons.md"

    def fake_chat(model, system, user, temperature=0.0):
        return ("### Lesson so far\n- the one true lesson", {"input_tokens": 1, "output_tokens": 1})

    with mock.patch("contrauto.agents.summarizer.chat", fake_chat):
        m1 = EpisodicMemory(model="claude", lessons_path=path)
        m1.remember("diag A")

    assert path.exists()
    m2 = EpisodicMemory(model="claude", lessons_path=path)
    assert m2._lesson == m1._lesson
    assert "the one true lesson" in m2._lesson
    assert "LESSON FROM THIS PROBLEM'S PRIOR FAILURES" in m2.render("BASE")


def test_episodic_distinct_missions_dont_collide(tmp_path):
    """Two EpisodicMemory instances sharing a lessons_path but different missions keep
    separate lessons (SummarizerAgent keys blocks by <!-- mission: X --> in the same file)."""
    path = tmp_path / "episodic_lessons.md"

    def fake_chat(model, system, user, temperature=0.0):
        return ("### Lesson\n- lesson body", {"input_tokens": 1, "output_tokens": 1})

    with mock.patch("contrauto.agents.summarizer.chat", fake_chat):
        a = EpisodicMemory(model="claude", lessons_path=path, mission="corridor")
        a.remember("diag A")
        b = EpisodicMemory(model="claude", lessons_path=path, mission="charger")
        b.remember("diag B")

    content = path.read_text(encoding="utf-8")
    assert "<!-- mission: corridor -->" in content
    assert "<!-- mission: charger -->" in content


def test_episodic_survives_llm_failure(tmp_path):
    """A summarization failure (e.g. network hiccup) must not crash remember()."""
    def boom(*a, **k):
        raise RuntimeError("network dead")

    with mock.patch("contrauto.agents.summarizer.chat", boom):
        m = EpisodicMemory(model="claude", lessons_path=tmp_path / "l.md")
        m.remember("diag A")   # should not raise
        assert len(m) == 1
        assert m._lesson == ""


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
