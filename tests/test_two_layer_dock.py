"""End to end test for the two layer dock simulator (no LLM).

Runs the physical layer on the synthesized two_layer_dock3 FSMs using a geometric greedy
assignment, so the test does not call the LLM. Asserts that the assignment is satisfied, that
goods are delivered, and that the capacity 1 dock mutex is never violated. Skips if the FSMs have
not been synthesized yet.

Run:  SDL_VIDEODRIVER=dummy python3 -m pytest tests/test_two_layer_dock.py
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSM_DIR = os.path.join(ROOT, "outputs", "two_layer_dock_n3", "workdir")

needs_fsm = pytest.mark.skipif(
    not os.path.exists(os.path.join(FSM_DIR, "robot1.json")),
    reason="needs synthesized FSMs (run: python main.py --task two_layer_dock --n 3 --no-sim)")


def _greedy(robots, norm, rects, bounds, deadline, model="gemini", rounds=4,
            seed_assignment=None, dock=None):
    """Geometric greedy assignment (nearest reachable, deadline checked), so the test runs the
    physical layer without the LLM. Same shape as assign_decomposer.decompose."""
    import contrauto.assign_decomposer as ad
    inst_pos, inst_group, groups = norm
    asg = {r: [] for r in robots}
    pos = dict(robots)
    t = {r: 0.0 for r in robots}
    for g, names in groups.items():
        best = None
        for r in robots:
            for n in names:
                d = ad.path_time(pos[r], inst_pos[n], rects, bounds)
                if d == ad.math.inf:
                    continue
                add, p2 = d, inst_pos[n]
                if dock is not None:
                    back = ad.path_time(p2, dock, rects, bounds)
                    if back == ad.math.inf:
                        continue
                    add, p2 = add + back, dock
                cand = (t[r] + add, r, n, p2)
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is None:
            return None, []
        tt, r, n, p2 = best
        asg[r].append(n)
        t[r] = tt
        pos[r] = p2
    issues, _ = ad.validate(asg, robots, norm, rects, bounds, deadline, dock=dock)
    return (asg, []) if not issues else (None, [])


@needs_fsm
def test_two_layer_dock_runs_and_is_safe(monkeypatch):
    import contrauto.assign_decomposer as ad
    monkeypatch.setattr(ad, "decompose", _greedy)
    from simulator.two_layer_dock import run, Config

    w = run(Config("two_layer_dock3", n_robots=3, n_demands=4, equiv=2, seed=3,
                   deadline=23.0, dock_hw=28.0, dock_hh=20.0),
            fsm_dir=FSM_DIR, max_steps=2600)

    assert w.report["assignment"], "assignment should be satisfiable"
    assert w.delivered >= 3, f"most goods should be delivered, got {w.delivered}"
    assert w.violations == 0, "the capacity-1 dock mutex must never be violated"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
