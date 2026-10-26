"""dynamic_2r_3b — two robots + a synthesized Dispatcher, on the common interface.
The Dispatcher FSM (inputs r*_has_task -> outputs assign_r*) is driven exactly like
warehouse's arbiter: it reads each robot's has_task state and grants new tasks.
See simulator/box_delivery.py.
"""
import argparse
from simulator.box_delivery import DeliveryGame, run_headless
from simulator.box_delivery import PREDICATES, ACTIONS, grounds, VOCAB_NOTES   # registry vocab/binder hooks

CBF_TASK = "dynamic_2r_3b"
W = "outputs/dynamic_2r_3b_regen/workdir/"


def _defs(r1, r2):
    return [(1, 'Robot1', 200, 480, 'red', r1), (2, 'Robot2', 560, 200, 'blue', r2)]


def headless(r1=W + "robot1.json", r2=W + "robot2.json", disp=W + "dispatcher.json", steps=8000,
             assumption_monitor=None):
    """Run the two robots + Dispatcher FSMs without a display; return metrics."""
    return run_headless(_defs(r1, r2), CBF_TASK, disp, steps=steps,
                        assumption_monitor=assumption_monitor)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--robot1', default=W + 'robot1.json')
    p.add_argument('--robot2', default=W + 'robot2.json')
    p.add_argument('--dispatcher', default=W + 'dispatcher.json')
    p.add_argument('--headless', type=int, default=0)
    a = p.parse_args()
    if a.headless:
        print(headless(a.robot1, a.robot2, a.dispatcher, steps=a.headless))
    else:
        DeliveryGame(_defs(a.robot1, a.robot2), CBF_TASK, a.dispatcher).run()
