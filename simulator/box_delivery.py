"""Shared box-delivery backend for static_2r_3b and dynamic_2r_3b, on the common
simulator interface (contrauto.sim_iface).

Two robots carry boxes from a single pickup point to a single delivery point. The
scenario only DECLARES per-robot predicates/actions (bound to each FSM's own r1_/r2_
prefixed variables by suffix); FsmController/run_fsm are shared. static = two
independent robots (arbiter None); dynamic = two robots + a synthesized Dispatcher FSM
that assigns tasks (driven exactly like warehouse's arbiter).
"""
import math, random, re
import pygame
from simulator.core import load_slugs, COLORS, BaseRobot, BaseGame
from contrauto.sim_iface import (
    SimAgent, SimWorld, Nav, Stationary, FsmController, Coordinator, fsm_step, run_fsm)
from contrauto.grounding import Grounding, bind_groundings
from cbf.motion import step_double_integrator_toward_target_cbf

PICKUP_XY, DELIVERY_XY = (160, 160), (640, 520)
ROBOT_R, SPEED, ARRIVE_R = 15, 130, 40
_PREFIX = re.compile(r"^r\d+_")


def _suffix(v):
    return _PREFIX.sub("", v)


def _classify(v):
    """Map an FSM variable (possibly r1_/r2_ prefixed, naming variants tolerated) to a
    canonical predicate/action key. Returns ('pred'|'act', key) or None."""
    s = _suffix(v).lower()
    # actions (check nav_* before at_*; exact pickup/drop/done)
    if "nav" in s and "pick" in s:                       return ("act", "nav_pickup")
    if "nav" in s and ("deliv" in s or "drop" in s):     return ("act", "nav_delivery")
    if "drop" in s:                                       return ("act", "dropoff")
    if "done" in s:                                       return ("act", "done")
    if "pick" in s and "at" not in s and "nav" not in s:  return ("act", "pickup")
    # sensors
    if "at" in s and "pick" in s:                         return ("pred", "at_pickup")
    if "at" in s and "deliv" in s:                        return ("pred", "at_delivery")
    if "box" in s or "carry" in s or "has_item" in s:     return ("pred", "has_box")
    if "credit" in s or "delivered" in s:                 return ("pred", "delivered")
    if "task" in s:                                       return ("pred", "task")
    return None


def _near(a, xy):
    return math.hypot(a.x - xy[0], a.y - xy[1]) < ARRIVE_R


# canonical predicates/actions keyed by the variable SUFFIX (after the r1_/r2_ prefix).
# Warehouse-isomorphic: at_pickup/has_box/at_delivery/delivered/task ->
# nav_pickup/pickup/nav_delivery/dropoff/done. `delivered` is a one-step latch the env
# raises after dropoff; `done` is an output mirroring delivered (no physical effect).
# The ONE place that reads raw box-delivery state, keyed by capability token.
PROVIDERS = {
    "agent.pos":            lambda a, w: (a.x, a.y),
    "agent.carry":          lambda a, w: a.carry,
    "agent.just_delivered": lambda a, w: a._just_delivered,
    "agent.has_task":       lambda a, w: a.has_task,
    "geom.point:pickup":    lambda a, w: PICKUP_XY,
    "geom.point:delivery":  lambda a, w: DELIVERY_XY,
}


class AtPickup(Grounding):
    def evaluate(self, cap): return cap.dist(cap.get("agent.pos"), cap.get("geom.point:pickup")) < ARRIVE_R


class AtDelivery(Grounding):
    def evaluate(self, cap): return cap.dist(cap.get("agent.pos"), cap.get("geom.point:delivery")) < ARRIVE_R


class Carrying(Grounding):
    def evaluate(self, cap): return cap.get("agent.carry") is not None


class Delivered(Grounding):
    def evaluate(self, cap): return cap.get("agent.just_delivered")


class HasTask(Grounding):
    def evaluate(self, cap): return cap.get("agent.has_task")


SUFFIX_PRED = bind_groundings({
    "at_pickup":   AtPickup(),
    "at_delivery": AtDelivery(),
    "carrying":    Carrying(),
    "has_box":     Carrying(),
    "delivered":   Delivered(),
    "task":        HasTask(),
    "has_task":    HasTask(),
}, PROVIDERS)


def _do_pickup(a, world):
    if a.carry is None:
        for b in world.boxes:
            if b.carrier is None and not b.delivered and _near(a, PICKUP_XY):
                b.carrier, a.carry = a.id, b
                break


def _do_dropoff(a, world):
    # The contract keeps `task` active until delivery is CREDITED (the next-step `done`),
    # so do NOT clear has_task here — clearing it now races the FSM's done transition
    # (which needs task=1 & delivered=1 together) and dead-ends the strategy.
    if a.carry is not None and _near(a, DELIVERY_XY):
        a.carry.carrier = None
        a.carry.delivered = True
        a.carry = None
        a._just_delivered = True
        a.done_count += 1


def _do_done(a, world):
    a.has_task = False           # delivery fully credited -> task complete; dispatcher reassigns


SUFFIX_ACT = {
    "nav_pickup":   Nav(lambda a, w: PICKUP_XY),
    "nav_delivery": Nav(lambda a, w: DELIVERY_XY),
    "pickup":       Stationary(_do_pickup),
    "dropoff":      Stationary(_do_dropoff),
    "done":         Stationary(_do_done),   # output mirroring delivered; clears the task
}

# Registry hooks (simulator.registry.grounding_vocab / grounding_binder). Without these the
# box-delivery tasks had NO vocabulary, Step 1.6 was a silent no-op, and an LLM `goto_delivery`
# slipped through unbound -> realizable FSM, delivered 0 (nothing ever navigates to delivery).
PREDICATES = SUFFIX_PRED
ACTIONS = SUFFIX_ACT


def grounds(v: str) -> bool:
    """True iff DeliveryRobot's OWN binder (_classify) can bind FSM variable `v` — the
    design-time grounding check judges with the exact function the runtime uses."""
    return _classify(v) is not None


class Box:
    """One deliverable box at the shared pickup point."""
    _n = 0
    def __init__(self):
        self.id = Box._n; Box._n += 1
        self.carrier, self.delivered = None, False


class DeliveryRobot(BaseRobot, SimAgent):
    """Box carrier whose FSM variables are bound to canonical roles by suffix (via _classify)."""
    def __init__(self, rid, name, x, y, color_key, fsm_path, cbf_task):
        super().__init__(rid, name, x, y, color_key, cbf_task, ROBOT_R, SPEED)
        self.name = name
        data = load_slugs(fsm_path)
        V = data["variables"]
        self.SENSORS, self.ACTIONS = {}, {}
        for v in V:
            c = _classify(v)
            if c is None:
                continue
            if c[0] == "pred":
                self.SENSORS[v] = SUFFIX_PRED[c[1]]
            else:
                self.ACTIONS[v] = SUFFIX_ACT[c[1]]
        in_vars = list(self.SENSORS.keys())
        out_vars = list(self.ACTIONS.keys())
        self.ctrl = FsmController(data, in_vars, out_vars)
        self.carry, self.has_task, self.action = None, False, {}
        self._just_delivered = False
        self.done_count = 0
        self.target = (float(x), float(y))

    def pose(self):
        return (self.x, self.y)

    def act(self, action, world):
        self._just_delivered = False
        self.action = action
        SimAgent.act(self, action, world)

    def tick(self, dt, world):
        if not any(self.action.values()):
            self.target = (self.hx, self.hy)
        self.tx, self.ty = self.target
        self.x, self.y, self.vx, self.vy = step_double_integrator_toward_target_cbf(
            self._cbf_task, self.x, self.y, self.vx, self.vy,
            self.tx, self.ty, dt, 1.0, self._base_speed, self._robot_radius, (), min_dist=1.5)
        if self.carry is not None:
            pass

    def reset(self):
        BaseRobot.reset(self)
        self.carry, self.has_task, self.done_count = None, False, 0
        self._just_delivered = False
        self.ctrl.reset()
        self.target = (self.x, self.y)

    def draw(self, surf, font):
        ix, iy = int(self.x), int(self.y)
        pygame.draw.circle(surf, self.color, (ix, iy), ROBOT_R)
        pygame.draw.circle(surf, COLORS["white"], (ix, iy), ROBOT_R, 2)
        if self.carry is not None:
            pygame.draw.rect(surf, COLORS["yellow"], (ix - 6, iy - ROBOT_R - 12, 12, 12))
        surf.blit(font.render(self.name[0], True, COLORS["white"]),
                  font.render(self.name[0], True, COLORS["white"]).get_rect(center=(ix, iy)))


class DeliveryGame(BaseGame, SimWorld):
    """Box-delivery SimWorld; with a dispatcher_path it adds a task-assigning Coordinator (dynamic)."""
    def __init__(self, robot_defs, cbf_task, dispatcher_path=None):
        super().__init__()
        for rid, name, x, y, col, path in robot_defs:
            self.robots.append(DeliveryRobot(rid, name, x, y, col, path, cbf_task))
        self.controllers = {r.name: r.ctrl for r in self.robots}
        self.selected = self.robots[0]
        self.boxes, self.delivered = [], 0
        self._frame, self.spawn_frames = 0, 90
        # optional Dispatcher (dynamic): a generic Coordinator whose inputs are the robots'
        # done signals and whose outputs assign tasks. Built ONCE so its FSM keeps state.
        self._coord = None
        if dispatcher_path:
            dd = load_slugs(dispatcher_path)
            do = [v for v in dd["variables"] if "assign" in v]            # outputs: assign_task_*
            di = [v for v in dd["variables"] if "assign" not in v]        # inputs: done_*
            self._coord = Coordinator(dd, di, do, source=self._disp_source, apply=self._disp_apply)
        self._init_control_btns()
        self._init_robot_btns(robot_defs)

    # ── SimWorld interface ───────────────────────────────────────────────────
    def agents(self):
        return self.robots

    def arbiter(self):
        return getattr(self, "_coord", None)        # generic Coordinator (getattr: safe for subclasses)

    def _robot_for(self, var):
        """The robot whose id matches the trailing digit of an FSM var (done_1 -> robot 1)."""
        m = re.search(r"(\d+)$", var)
        if not m:
            return None
        return next((r for r in self.robots if r.id == int(m.group(1))), None)

    # Coordinator wiring (dispatcher): SOURCE the FSM's inputs from robot done signals,
    # APPLY its assignment outputs by granting tasks. No bespoke adapter class.
    def _disp_source(self, _reqs, _world):
        return {v: int(self._robot_for(v)._just_delivered) if self._robot_for(v) else 0
                for v in self._coord.ctrl.in_vars}

    def _disp_apply(self, out, _world):
        avail = any(b.carrier is None and not b.delivered for b in self.boxes)
        for v in self._coord.ctrl.out_vars:
            r = self._robot_for(v)
            if r and out.get(v) and not r.has_task and (avail or r.carry is not None):
                r.has_task = True
        return {}                                   # robots read has_task via their own SENSORS

    def step(self, dt):
        self._frame += 1
        if self._frame % self.spawn_frames == 0 or not [b for b in self.boxes if b.carrier is None and not b.delivered]:
            self.boxes.append(Box())
        # no coordinator -> assign a task to any idle robot while boxes remain (the env
        # liveness `tasks assigned infinitely often`). With a Dispatcher, _disp_apply does this.
        if getattr(self, "_coord", None) is None:
            avail = any(b.carrier is None and not b.delivered for b in self.boxes)
            for r in self.robots:
                if not r.has_task and (avail or r.carry is not None):
                    r.has_task = True
        self.delivered = sum(b.delivered for b in self.boxes)
        self.boxes = [b for b in self.boxes if not b.delivered][:20] + [b for b in self.boxes if b.delivered][-1:]

    def metrics(self):
        return {"delivered": sum(r.done_count for r in self.robots),
                "per_robot": {r.name: r.done_count for r in self.robots}}

    def update(self, dt):
        if self.paused: return
        fsm_step(self, self.controllers, dt, self.arbiter())

    def _draw_arena(self):
        self._draw_grid()
        pygame.draw.circle(self.screen, COLORS["shelf"], PICKUP_XY, ARRIVE_R, 0)
        pygame.draw.circle(self.screen, COLORS["dock"], DELIVERY_XY, ARRIVE_R, 0)
        self.screen.blit(self.fS.render("PICKUP", True, COLORS["white"]), (PICKUP_XY[0] - 24, PICKUP_XY[1] - 6))
        self.screen.blit(self.fS.render("DELIVERY", True, COLORS["white"]), (DELIVERY_XY[0] - 30, DELIVERY_XY[1] - 6))
        pygame.draw.line(self.screen, COLORS["border"], (self.ARENA_W, 0), (self.ARENA_W, self.H), 3)

    def _draw_objects(self):
        pass

    def _draw_panel(self):
        self._draw_panel_header()
        self._draw_panel_stats(extra_stats=[f"Delivered {self.metrics()['delivered']}"] +
                               [f"{r.name} {r.done_count}{' [T]' if r.has_task else ''}" for r in self.robots])
        self._draw_panel_controls()

    def _reset(self):
        super()._reset()
        self.boxes.clear(); self.delivered = 0; self._frame = 0
        if getattr(self, "_coord", None) is not None:
            self._coord.ctrl.reset()


def run_headless(robot_defs, cbf_task, dispatcher_path=None, steps=8000, dt=1 / 60.0,
                 assumption_monitor=None):
    """Run the box-delivery FSMs without a display; return metrics (per-robot deliveries)."""
    g = DeliveryGame(robot_defs, cbf_task, dispatcher_path)
    return run_fsm(g, g.controllers, steps, dt, assumption_monitor=assumption_monitor)

VOCAB_NOTES = (
    "at_pickup/at_delivery: position sensors; persist while the robot stays (movement is "
    "DURATIVE — many steps; use justice G F(at_x | !nav_x), never next-step arrival). "
    "has_box/carrying: rises the step after a successful pickup, falls the step after "
    "dropoff. delivered: ONE-STEP pulse the step after a successful dropoff; on that step "
    "has_box/carrying is ALREADY false (both flip together) — never assume delivered "
    "implies carrying, react immediately. done: output completing the "
    "task cycle (clears task). task: task-available sensor from the dispatcher."
)
