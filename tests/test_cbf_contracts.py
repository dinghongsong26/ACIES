"""Tests for the physical-layer CBF contracts and the task-level repair loop (no LLM needed).

Covers:
  - cbf_contracts: admissibility and the per-robot physical_certificate.
  - task-level repair: classify_structural_failure (the 4 cases) and the run_with_task_repair
    outer loop (mocked synthesis + physical), plus the physical_feedback channel wiring.

Run:  SDL_VIDEODRIVER=dummy python3 -m pytest tests/test_cbf_contracts.py
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.cbf_contracts import admissibility, physical_certificate

BOUNDS = (0.0, 800.0, 0.0, 600.0)


# ── cbf_contracts: admissibility (Def. 7 / Thm. 4 cond. 2) ────────────────────────
def test_admissibility():
    assert admissibility([0.4, 0.7, 0.6], T_s=1.0)["ok"] is True
    bad = admissibility([0.4, 1.3, 0.6], T_s=1.0)
    assert bad["ok"] is False and bad["T_r_max"] == 1.3


def test_physical_certificate_feasible_and_over_deadline():
    start, legs = (100.0, 100.0), [(400.0, 300.0), (100.0, 100.0)]   # reachable, clear space
    ok = physical_certificate(start, legs, rects=[], bounds=BOUNDS, deadline=1e6)
    assert ok["ok"] and ok["n_reach"] >= 2 and 0 < ok["bound"] < 1e6
    # an impossibly tight deadline -> certificate fails (infeasibility signal for repair)
    tight = physical_certificate(start, legs, rects=[], bounds=BOUNDS, deadline=0.01)
    assert tight["ok"] is False and tight["issues"]


# ── task-level repair: structural-failure classifier ──────────────────────────────
def _hist(rounds):  # rounds = [(deadline, rows, delivered)], rows = [(name, finish_t, slack)]
    return [{"round": i, "deadline": d, "rows": r, "delivered": dl}
            for i, (d, r, dl) in enumerate(rounds)]


def test_classify_starvation():
    from simulator.two_layer_dock import classify_structural_failure
    h = _hist([(15, [("r1", 12, 0), ("r2", 13, 0), ("r3", None, 0)], 2),
               (15, [("r1", 12, 0), ("r2", 13, 0), ("r3", None, 0)], 2)])
    d = classify_structural_failure(h, ["r1", "r2", "r3"])
    assert d["structural"] and d["kind"] == "starvation" and "r3" in d["message"]


def test_classify_no_progress():
    from simulator.two_layer_dock import classify_structural_failure
    h = _hist([(15, [("r1", 12, 0), ("r2", None, 0), ("r3", 14, 0)], 1),
               (15, [("r1", None, 0), ("r2", 13, 0), ("r3", 14, 0)], 1)])
    d = classify_structural_failure(h, ["r1", "r2", "r3"])
    assert d["structural"] and d["kind"] == "no_progress"


def test_classify_diverging():
    from simulator.two_layer_dock import classify_structural_failure
    h = _hist([(15, [("r1", 12, 0), ("r2", None, 0), ("r3", 14, 0)], 1),
               (21, [("r1", 12, 0), ("r2", 20, 0), ("r3", None, 0)], 2)])
    d = classify_structural_failure(h, ["r1", "r2", "r3"])
    assert d["structural"] and d["kind"] == "diverging"


def test_classify_quantitative_not_structural():
    from simulator.two_layer_dock import classify_structural_failure
    h = _hist([(15, [("r1", 12, 0), ("r2", 13, 0), ("r3", 14, 0)], 3)])
    d = classify_structural_failure(h, ["r1", "r2", "r3"])
    assert d["structural"] is False


# ── task-level repair: the outer loop (mocked synthesis + physical) ───────────────
def test_run_with_task_repair_escalates_then_converges(monkeypatch):
    import simulator.two_layer_dock as M
    import contrauto.pipeline as P
    calls = {"synth": 0, "phys": 0, "fb": ""}

    class FakeContrAuto:
        def __init__(self, **kw): calls["fb"] = kw.get("physical_feedback", "")
        def run(self): calls["synth"] += 1; return {}

    def fake_phys(cfg, fsm_dir=None, **kw):
        calls["phys"] += 1
        if calls["phys"] == 1:   # structural failure (r3 starved)
            return {"converged": False, "world": None, "delivered": 2, "mutex_viol": 0,
                    "rounds_used": 6, "final_deadline": 15,
                    "history": _hist([(15, [("r1", 12, 0), ("r2", 13, 0), ("r3", None, 0)], 2)])}
        return {"converged": True, "world": None, "delivered": 3, "mutex_viol": 0,
                "rounds_used": 1, "final_deadline": 15, "history": []}

    monkeypatch.setattr(P, "ContrAuto", FakeContrAuto)
    monkeypatch.setattr(M, "run_closed_loop", fake_phys)
    res = M.run_with_task_repair(task="two_layer_dock3", cfg=object(), fsm_dir="x",
                                 outdir="out", max_task_rounds=3)
    assert res["converged"] and res["task_rounds"] == 2
    assert calls["phys"] == 2 and calls["synth"] == 1          # re-synthesized exactly once
    assert "starv" in calls["fb"].lower()                       # structural feedback delivered


def test_physical_feedback_channel_wired():
    from contrauto.agents.decomposer import DecomposerAgent
    a = DecomposerAgent(model="gemini", workdir="/tmp/_t_dec", mission="m",
                        physical_feedback="ROBOT r2 STARVED")
    assert a.physical_feedback == "ROBOT r2 STARVED"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
