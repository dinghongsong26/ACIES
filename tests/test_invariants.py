"""Physical-layer auto-checks (sim_iface invariants): run the FSM-driven sims with
runtime invariants enabled and assert ZERO violations — instead of trusting only the
delivered/cleared metrics. Catches: a stalled/degenerate FSM (no_stall_invariant) and
robots overlapping (separation_invariant / CBF safety breach).

Run:  SDL_VIDEODRIVER=dummy python3 tests/test_invariants.py
"""
import os
import sys

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.sim_iface import run_fsm, separation_invariant, no_stall_invariant

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_warehouse_separation_and_no_stall():
    from simulator.warehouse import WarehouseGame, ROBOT_R
    W = os.path.join(ROOT, "outputs/warehouse_n2/workdir/")
    if not os.path.exists(W + "robot1.json"):
        pytest.skip("needs synthesized warehouse FSMs (run: python main.py --task warehouse --n 2 --no-sim)")
    defs = [(1, "Robot1", 200, 490, "red", W + "robot1.json"),
            (2, "Robot2", 600, 490, "blue", W + "robot2.json")]
    g = WarehouseGame(defs, W + "arbiter.json")
    inv = [separation_invariant(ROBOT_R), no_stall_invariant(g.controllers)]
    m = run_fsm(g, g.controllers, 6000, invariants=inv)
    assert m["n_violations"] == 0, m["invariant_violations"]
    assert m["violations"] == 0                      # arbiter mutual exclusion


def test_patrol_no_stall():
    from simulator.patrol import PatrolGame, HOME_XY
    fsm = os.path.join(ROOT, "outputs/patrol_regen/workdir/patrolrobot.json")
    if not os.path.exists(fsm):
        pytest.skip("needs synthesized patrol FSMs (run: python main.py --task patrol --no-sim)")
    g = PatrolGame([(1, "P1", HOME_XY[0], HOME_XY[1], "blue", fsm)])
    m = run_fsm(g, g.controllers, 4000, invariants=[no_stall_invariant(g.controllers)])
    assert m["n_violations"] == 0, m["invariant_violations"]


def test_dynamic_no_stall():
    # box_delivery robots don't run peer-CBF (single pickup/delivery points), so only the
    # FSM no-stall invariant is meaningful here, not inter-robot separation.
    from simulator.box_delivery import DeliveryGame
    from simulator.dynamic_2r_3b import _defs, W, CBF_TASK
    if not os.path.exists(W + "robot1.json"):
        pytest.skip("needs synthesized dynamic_2r_3b FSMs (run: python main.py --task dynamic_2r_3b --no-sim)")
    g = DeliveryGame(_defs(W + "robot1.json", W + "robot2.json"), CBF_TASK, W + "dispatcher.json")
    m = run_fsm(g, g.controllers, 6000, invariants=[no_stall_invariant(g.controllers)])
    assert m["n_violations"] == 0, m["invariant_violations"]
    assert m["delivered"] > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all invariant tests passed")
