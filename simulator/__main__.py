"""Unified simulator entry point (live pygame window):

    python -m simulator --task warehouse
    python -m simulator --task patrol
    python -m simulator --task static_2r_3b
    python -m simulator --task dynamic_2r_3b

Each task uses its module's default regenerated FSMs unless overridden with --fsm /
--robot1 / --robot2 / --arbiter / --dispatcher. All four are driven by the synthesized
GR(1) FSMs through contrauto/sim_iface.py.
"""
import argparse


def main():
    """Parse --task (and optional FSM overrides) and launch the matching simulator window."""
    p = argparse.ArgumentParser(description="ContrAuto simulator (FSM-driven)")
    p.add_argument("--task", required=True,
                   choices=["warehouse", "patrol", "static_2r_3b", "dynamic_2r_3b"])
    p.add_argument("--fsm", default=None, help="FSM JSON (patrol)")
    p.add_argument("--robot1", default=None)
    p.add_argument("--robot2", default=None)
    p.add_argument("--arbiter", default=None, help="arbiter FSM (warehouse)")
    p.add_argument("--dispatcher", default=None, help="dispatcher FSM (dynamic_2r_3b)")
    a = p.parse_args()

    if a.task == "warehouse":
        from simulator.warehouse import WarehouseGame
        W = "outputs/warehouse_n2/workdir/"
        defs = [(1, "Robot1", 200, 490, "red", a.robot1 or W + "robot1.json"),
                (2, "Robot2", 600, 490, "blue", a.robot2 or W + "robot2.json")]
        WarehouseGame(defs, a.arbiter or W + "arbiter.json").run()

    elif a.task == "patrol":
        from simulator.patrol import PatrolGame, HOME_XY
        fsm = a.fsm or "outputs/patrol_regen/workdir/patrolrobot.json"
        PatrolGame([(1, "P1", HOME_XY[0], HOME_XY[1], "blue", fsm)]).run()

    elif a.task == "static_2r_3b":
        from simulator.static_2r_3b import StaticDeliveryGame, _defs, W, CBF_TASK
        StaticDeliveryGame(_defs(a.robot1 or W + "robot1.json", a.robot2 or W + "robot2.json"),
                           CBF_TASK).run()

    elif a.task == "dynamic_2r_3b":
        from simulator.box_delivery import DeliveryGame
        from simulator.dynamic_2r_3b import _defs, W, CBF_TASK
        DeliveryGame(_defs(a.robot1 or W + "robot1.json", a.robot2 or W + "robot2.json"),
                     CBF_TASK, a.dispatcher or W + "dispatcher.json").run()


if __name__ == "__main__":
    main()
