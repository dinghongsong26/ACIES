"""static_2r_3b — two robots with a TRUE STATIC ASSIGNMENT, on the common interface.

Design-time assignment (asymmetric):  Robot1 owns box1 + box3 ;  Robot2 owns box2.

Each robot's synthesized FSM is warehouse-isomorphic (nav_pickup -> pickup ->
nav_delivery -> dropoff -> done) and runs independently — no coordinator. The static
ownership lives in the SIMULATION layer (which the GR(1) contracts cannot encode, since
each robot's boolean channel is symmetric by construction):

  * boxes carry an `owner` field (set at spawn: box1->R1, box2->R2, box3->R1, repeating);
  * a robot only ever receives a task (`has_task`) when a box IT OWNS is available, so it
    only navigates to pickup when its own box is there (keeps the `pickup -> X has_box`
    environment assumption honest);
  * a robot only ever picks up a box it owns;
  * per-robot delivery counts are tracked, and we additionally verify each robot only ever
    delivered boxes assigned to it (`owned_delivered` / `foreign_delivered`).

This module SUBCLASSES simulator/box_delivery.py without modifying it.
"""
import argparse

from simulator.box_delivery import (
    Box, DeliveryRobot, DeliveryGame, run_fsm, run_headless as _bd_run_headless,
)
from simulator.box_delivery import PREDICATES, ACTIONS, grounds, VOCAB_NOTES   # registry vocab/binder hooks
from simulator.box_delivery import _near, PICKUP_XY
from contrauto.sim_iface import Stationary

CBF_TASK = "static_2r_3b"
W = "outputs/static_2r_3b_regen/workdir/"

# Static assignment: which robot id owns which box "slot". box1 & box3 -> Robot1, box2 -> Robot2.
OWNER_BY_SLOT = {1: 1, 2: 2, 3: 1}   # slot (1-based, cycles) -> robot id


def _defs(r1, r2):
    return [(1, 'Robot1', 200, 480, 'red', r1), (2, 'Robot2', 560, 200, 'blue', r2)]


class OwnedBox(Box):
    """A box that belongs to exactly one robot (static assignment)."""
    def __init__(self, owner, slot):
        super().__init__()
        self.owner = owner   # robot id that is statically allowed to handle this box
        self.slot = slot     # 1-based logical box index (1->box1, 2->box2, 3->box3, ...)


def _owned_pickup(a, world):
    """Pickup effect that only grabs a box OWNED by this robot. Replaces box_delivery's
    owner-blind `_do_pickup` (which takes any free box)."""
    if a.carry is None and _near(a, PICKUP_XY):
        for b in world.boxes:
            if (b.carrier is None and not b.delivered
                    and getattr(b, "owner", None) == a.id):
                b.carrier, a.carry = a.id, b
                break


class StaticDeliveryRobot(DeliveryRobot):
    """DeliveryRobot whose pickup action is ownership-filtered, and which keeps per-robot
    owned/foreign delivery tallies."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owned_delivered = 0     # deliveries of boxes this robot owns (should == done_count)
        self.foreign_delivered = 0   # deliveries of boxes it does NOT own (must stay 0)
        # Re-bind every pickup action to the ownership-aware effect.
        for v, action in list(self.ACTIONS.items()):
            if isinstance(action, Stationary) and action.effect is not None \
                    and action.effect.__name__ == "_do_pickup":
                self.ACTIONS[v] = Stationary(_owned_pickup)

    def reset(self):
        super().reset()
        self.owned_delivered = 0
        self.foreign_delivered = 0


class StaticDeliveryGame(DeliveryGame):
    """DeliveryGame that spawns OWNED boxes and only assigns a task to a robot when a box
    it owns is available — the true static-assignment behaviour."""
    def __init__(self, robot_defs, cbf_task, dispatcher_path=None):
        # Bypass DeliveryGame.__init__'s robot construction by replicating it with our
        # subclassed robot; everything else (controllers, boxes, buttons) is identical.
        from simulator.core import BaseGame
        from contrauto.sim_iface import SimWorld  # noqa: F401  (mixin already in MRO)
        BaseGame.__init__(self)
        for rid, name, x, y, col, path in robot_defs:
            self.robots.append(StaticDeliveryRobot(rid, name, x, y, col, path, cbf_task))
        self.controllers = {r.name: r.ctrl for r in self.robots}
        self.selected = self.robots[0]
        self.boxes, self.delivered = [], 0
        self._frame, self.spawn_frames = 0, 90
        self._slot = 0
        self._disp = None  # static = no dispatcher
        self._init_control_btns()
        self._init_robot_btns(robot_defs)

    def _spawn_owned_box(self):
        self._slot += 1
        n = max(OWNER_BY_SLOT)                      # cycle length (3 box slots)
        slot = ((self._slot - 1) % n) + 1
        self.boxes.append(OwnedBox(OWNER_BY_SLOT[slot], slot))

    def _owned_box_available(self, rid):
        return any(b.carrier is None and not b.delivered and getattr(b, "owner", None) == rid
                   for b in self.boxes)

    def step(self, dt):
        self._frame += 1
        # Ensure each robot always has SOME upcoming owned box: spawn on the timer, and
        # also spawn proactively if any robot currently has no available owned box.
        need_spawn = (self._frame % self.spawn_frames == 0)
        for r in self.robots:
            if not self._owned_box_available(r.id) and r.carry is None:
                need_spawn = True
        if need_spawn or not self.boxes:
            self._spawn_owned_box()
            # Spawn enough so that, after the first box, every robot can eventually get one.
            for _ in range(len(OWNER_BY_SLOT)):
                if all(self._owned_box_available(r.id) or r.carry is not None for r in self.robots):
                    break
                self._spawn_owned_box()
        # Static assignment of tasks: a robot gets a task ONLY when a box it owns is free
        # (or it is mid-delivery carrying its own box).
        for r in self.robots:
            if not r.has_task and (self._owned_box_available(r.id) or r.carry is not None):
                r.has_task = True
        self.delivered = sum(b.delivered for b in self.boxes)
        # keep the box list bounded; retain undelivered + a tail of delivered
        self.boxes = ([b for b in self.boxes if not b.delivered][:30]
                      + [b for b in self.boxes if b.delivered][-3:])

    def metrics(self):
        # Re-tally owned/foreign from box history would require keeping all boxes; instead we
        # track at delivery time. Here we surface the running per-robot tallies.
        return {
            "delivered": sum(r.done_count for r in self.robots),
            "per_robot": {r.name: r.done_count for r in self.robots},
            "owned_per_robot": {r.name: r.owned_delivered for r in self.robots},
            "foreign_per_robot": {r.name: r.foreign_delivered for r in self.robots},
        }


# Patch dropoff accounting so owned/foreign tallies are kept. box_delivery._do_dropoff runs
# as the bound effect and increments done_count; we wrap it to also classify ownership.
from simulator import box_delivery as _bd

_orig_do_dropoff = _bd._do_dropoff


def _do_dropoff_owned(a, world):
    """Wrap box_delivery._do_dropoff to also tally the delivery as owned vs foreign."""
    box = a.carry
    before = a.done_count
    _orig_do_dropoff(a, world)
    if a.done_count > before and box is not None and hasattr(a, "owned_delivered"):
        if getattr(box, "owner", None) == a.id:
            a.owned_delivered += 1
        else:
            a.foreign_delivered += 1


def headless(r1=W + "robot1.json", r2=W + "robot2.json", steps=8000, assumption_monitor=None):
    """Run the static-assignment FSMs without a display; return metrics (owned/foreign tallies)."""
    # Temporarily route dropoff through the ownership-tallying wrapper.
    _bd._do_dropoff = _do_dropoff_owned
    _bd.SUFFIX_ACT["dropoff"] = Stationary(_do_dropoff_owned)
    try:
        g = StaticDeliveryGame(_defs(r1, r2), CBF_TASK)
        return run_fsm(g, g.controllers, steps, 1 / 60.0,
                       assumption_monitor=assumption_monitor)
    finally:
        _bd._do_dropoff = _orig_do_dropoff
        _bd.SUFFIX_ACT["dropoff"] = Stationary(_orig_do_dropoff)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--robot1', default=W + 'robot1.json')
    p.add_argument('--robot2', default=W + 'robot2.json')
    p.add_argument('--headless', type=int, default=0)
    a = p.parse_args()
    if a.headless:
        print(headless(a.robot1, a.robot2, steps=a.headless))
    else:
        _bd._do_dropoff = _do_dropoff_owned
        _bd.SUFFIX_ACT["dropoff"] = Stationary(_do_dropoff_owned)
        StaticDeliveryGame(_defs(a.robot1, a.robot2), CBF_TASK).run()
