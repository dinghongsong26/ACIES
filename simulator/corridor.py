"""B (angle-3, GENERALITY): SPATIAL mutual exclusion — a single-lane CORRIDOR that only ONE
robot may occupy at a time.

Same capacity-1-resource CONTRACT as the dock, so the SAME synthesized FSM (two_layer_dock3 /
dock4) drives it unchanged — only the PHYSICAL resource changes: a passage in space, not a
device. The arbiter's grant gates corridor entry exactly as it gated dock use, so "one robot at
a time in the corridor" is enforced by the SAME synthesized mutual exclusion.

Layout is RANDOM per seed: a wall splits the map into side A (pickup) and side B (delivery),
with a gap (the corridor) at a random height; the pickup/delivery points are random too.

Role mapping (bind_fsm via role_of, reusing the dock FSM's variable names):
  at_shelf  -> at the pickup point on side A     go_to_shelf -> navigate to side-A pickup
  at_dock   -> at the delivery point on side B   go_to_dock  -> CROSS the corridor to side B
  pickup / dropoff / has_item / task / request / grant : identical to the dock
The corridor (the capacity-1 resource) == the "dock" role: crossing to side B needs the grant.

Run:  python simulator/corridor.py [--seed N]
"""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import sys
import math
import random
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
from PIL import Image

from contrauto.sim_iface import (SimAgent, SimWorld, Nav, Stationary, Coordinator, fsm_step,
                                 load_fsm, FsmController)
from contrauto.grounding import Grounding, bind_groundings
from contrauto.stl_cbf import in_circle, out_circle, out_obb, Always, Eventually, And, stl_cbf_step
from contrauto.geometry import WaypointNav

RR, SPEED = 13, 155
# DRAW_RR is display-only (the circle radius drawn on screen/in the GIF) -- it does NOT
# feed the CBF avoidance/queue-spacing math, which stays tuned to the physical RR. Bumping
# RR itself would tighten the (2*RR+6)-vs-40px queue-spacing margin we already made thin
# once (see the queue_pt note above); this makes the robots easier to see without touching
# any of that.
DRAW_RR = 20
# The world's patience for an unserviced standing demand (E2ECorridorWorld.step): after this
# many seconds without the robot even ENTERING the corridor (crossing_active still False), the
# world withdraws the demand itself and starts a fresh cooldown. Without this, `req`/
# `task_arrived` only ever cleared/repulsed via the ROBOT completing a crossing -- so a
# strategy that (legally, per an assumption it never saw violated) never asserts `crossing`
# leaves the contract's own `G F(task_arrived)` / `G F(!req)` assumptions permanently
# unsatisfied by the real world, making its guarantees non-binding forever. Bounding the
# demand's patience makes the world satisfy those assumptions unconditionally, exactly as the
# requirements describe it ("raised and withdrawn by the world, not chosen by the robots").
DEMAND_TIMEOUT = 15.0
_tobytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSM_DIR = ROOT + "/outputs/two_layer_dock_n3/workdir"


@dataclass
class Config:
    name: str = "corridor"
    n_robots: int = 3
    n_tasks: int = 2            # crossings (deliveries) per robot
    W: int = 1000
    H: int = 700
    seed: int = 3
    gap_h: float = 96.0         # corridor (gap) height
    wall_thick: float = 24.0


class Layout:
    """A vertical wall at x=W/2 with a single GAP (the corridor) at a random height; random
    pickup (side A, left) and delivery (side B, right) points."""

    # How far "A home" (pickup_a) must stay from the wall/corridor mouth, so the home
    # cluster and the per-robot queue line (up to 40px * n_robots apart, see E2ERobot.
    # queue_pt) don't visually overlap -- was a bare `random.uniform(m, wx - m)`, which
    # could (and at seed=3 did) land pickup_a right next to the wall, making robots
    # queuing on their own line look like they'd never left home.
    MIN_HOME_WALL_GAP = 220.0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bounds = (0.0, float(cfg.W), 0.0, float(cfg.H))
        self.wx = cfg.W / 2.0
        wt, m = cfg.wall_thick, 90.0
        self.gap_y = random.uniform(m + cfg.gap_h, cfg.H - m - cfg.gap_h)   # RANDOM corridor height
        gy0, gy1 = self.gap_y - cfg.gap_h / 2, self.gap_y + cfg.gap_h / 2
        # two wall segments (oriented boxes cx,cy,hw,hh,ang) above and below the gap
        self.rects = [
            (self.wx, gy0 / 2.0,            wt / 2, gy0 / 2.0,             0.0),
            (self.wx, (cfg.H + gy1) / 2.0,  wt / 2, (cfg.H - gy1) / 2.0,   0.0),
        ]
        # rectangle (xmin,ymin,xmax,ymax) used to test "inside the corridor"
        self.corridor = (self.wx - wt, gy0, self.wx + wt, gy1)
        home_x_max = max(m, self.wx - m - self.MIN_HOME_WALL_GAP)
        self.pickup_a = (random.uniform(m, home_x_max), random.uniform(m, cfg.H - m))
        self.deliver_b = (random.uniform(self.wx + m, cfg.W - m), random.uniform(m, cfg.H - m))


def _at(a, pt, tol=RR + 14):
    return (a.x - pt[0]) ** 2 + (a.y - pt[1]) ** 2 <= tol * tol


def _in_corridor(a, w) -> bool:
    x0, y0, x1, y1 = w.lay.corridor
    return x0 <= a.x <= x1 and y0 <= a.y <= y1


def _do_pick(a, w):
    if _at(a, w.lay.pickup_a) and a.carry is None and a.qi < a.n_tasks:
        a.carry = True


def _do_drop(a, w):
    if a.carry is not None and _at(a, w.lay.deliver_b):
        a.carry = None
        a.qi += 1
        a._jd = True


# predicates/actions keyed by the SAME roles the dock FSM binds to; physical meaning is corridor.
# The ONE place that reads raw corridor state, keyed by capability token.
PROVIDERS = {
    "agent.pos":              lambda a, w: (a.x, a.y),
    "agent.carry":            lambda a, w: a.carry,
    "agent.qi":               lambda a, w: a.qi,
    "agent.n_tasks":          lambda a, w: a.n_tasks,
    "geom.point:pickup_a":    lambda a, w: w.lay.pickup_a,
    "geom.point:deliver_b":   lambda a, w: w.lay.deliver_b,
}


class AtShelf(Grounding):   # at side-A pickup
    def evaluate(self, cap): return cap.dist(cap.get("agent.pos"), cap.get("geom.point:pickup_a")) <= RR + 14


class AtDock(Grounding):    # at side-B delivery (across the corridor)
    def evaluate(self, cap): return cap.dist(cap.get("agent.pos"), cap.get("geom.point:deliver_b")) <= RR + 14


class HasItem(Grounding):
    def evaluate(self, cap): return cap.get("agent.carry") is not None


class HasTask(Grounding):
    def evaluate(self, cap): return (cap.get("agent.qi") < cap.get("agent.n_tasks")) or (cap.get("agent.carry") is not None)


PRED = bind_groundings({
    "at_shelf":  AtShelf(),
    "has_item":  HasItem(),
    "at_dock":   AtDock(),
    "task":      HasTask(),
}, PROVIDERS)
ACT = {
    "go_to_shelf": Nav(lambda a, w: w.lay.pickup_a,  done_when=lambda a, w: _at(a, w.lay.pickup_a)),
    "pickup":      Stationary(_do_pick),
    "go_to_dock":  Nav(lambda a, w: w.lay.deliver_b, done_when=lambda a, w: _at(a, w.lay.deliver_b)),
    "dropoff":     Stationary(_do_drop),
    "done":        Stationary(None),
}

ROBOT_COLORS = [(225, 70, 70), (70, 130, 225), (80, 200, 100), (225, 165, 50), (165, 100, 220)]


class CorridorRobot(SimAgent):
    """Worker driven by the dock FSM, where the granted 'dock' role gates crossing the corridor."""
    def __init__(self, rid, fsm_path, lay, n_tasks, color):
        self.id, self.name, self.color = rid, f"r{rid}", color
        self.bind_fsm(fsm_path, PRED, ACT)
        ang = rid * 1.7
        self.x = lay.pickup_a[0] + 55 * math.cos(ang)
        self.y = lay.pickup_a[1] + 55 * math.sin(ang)
        self.carry = None
        self._jd = False
        self.qi = 0
        self.n_tasks = n_tasks
        self.target = (self.x, self.y)
        self._nav = WaypointNav(obb=True)
        self.vel = np.zeros(2)
        self.priority = rid
        self._pass = False   # latched corridor authorization (see tick)

    def pose(self):
        return (self.x, self.y)

    def act(self, action, world):
        self._jd = False
        SimAgent.act(self, action, world)

    def tick(self, dt, world):
        lay = world.lay
        # GRANT physicalized as corridor gating: the arbiter's grant decides who may cross. If
        # this robot is heading to side B (must cross) but is NOT granted, hold it at the corridor
        # ENTRANCE on side A — so only the granted robot is ever inside the passage. (The dock FSM
        # gates "use the resource"; for a passage the physical layer enforces "enter the resource".)
        granted = world.last_grants.get(self.GRANTS[0], 0) if self.GRANTS else 1
        # The reused dock arbiter PULSES grant for one discrete step (a dock visit is
        # one FSM tick). A corridor crossing is hundreds of frames at 60 Hz, so a
        # per-tick "ungranted → yank back to the mouth" gate livelocks everyone at
        # (wx-62, gap_y): ~2.6 px in on the pulse, then pulled back. Latch the pass
        # on a grant while the corridor is free; hold it until this robot clears
        # the east edge so one pulse is enough to actually cross, mutex still 1.
        if granted and not any(o._pass for o in world.robots if o is not self):
            self._pass = True
        if self._pass and self.x > lay.corridor[2] + 4:
            self._pass = False
        if (not self._pass) and self.target[0] > lay.wx and self.x < lay.wx - 2:
            self.target = (lay.wx - 62, lay.gap_y)            # wait just left of the gap
        wp = self._nav.target((self.x, self.y), self.target, lay.rects, lay.bounds, RR)
        p = np.array([self.x, self.y], float)
        seg_dl = world.t + float(np.linalg.norm(np.array(wp, float) - p)) / SPEED * 1.3 + dt
        walls = [Always(0.0, 1e9, out_obb(cx, cy, hw, hh, ang, margin=RR + 4))
                 for (cx, cy, hw, hh, ang) in lay.rects]
        formula = And([Eventually(world.t, seg_dl, in_circle(wp[0], wp[1], RR))]
                      + [Always(0.0, 1e9, out_circle(o.x, o.y, 2 * RR + 6))
                         for o in world.robots if o is not self and o.priority > self.priority]
                      + walls)
        dirn = np.array(wp, float) - p
        nrm = float(np.linalg.norm(dirn)) or 1.0
        u_nom = 6.0 * (dirn / nrm * SPEED - self.vel)
        p_new, self.vel, _, _ = stl_cbf_step(formula, p, self.vel, world.t, u_nom, dt, v_max=SPEED)
        self.x = min(lay.cfg.W - RR, max(RR, float(p_new[0])))
        self.y = min(lay.cfg.H - RR, max(RR, float(p_new[1])))


class CorridorWorld(SimWorld):
    """SimWorld reusing the dock arbiter FSM to enforce one-robot-at-a-time spatial mutex."""
    def __init__(self, cfg, fsm_dir=FSM_DIR):
        random.seed(cfg.seed)
        self.cfg = cfg
        self.lay = Layout(cfg)
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.W, cfg.H))
        self.font = pygame.font.SysFont(None, 26)
        self.robots = [CorridorRobot(i + 1, f"{fsm_dir}/robot{i+1}.json", self.lay,
                                     cfg.n_tasks, ROBOT_COLORS[i % len(ROBOT_COLORS)])
                       for i in range(cfg.n_robots)]
        self.controllers = {r.name: r.ctrl for r in self.robots}
        ad = load_fsm(f"{fsm_dir}/arbiter.json")
        rq = [v for v in ad["variables"] if "req" in v.lower()]
        gr = [v for v in ad["variables"] if "grant" in v.lower()]
        self.last_grants = {}
        self._coord = Coordinator(ad, rq, gr, apply=self._grant_apply)
        self.violations = 0
        self.max_in_corridor = 0
        self.t = 0.0

    def agents(self):
        return self.robots

    def arbiter(self):
        return self._coord

    def _grant_apply(self, out, world):
        self.last_grants = dict(out)        # remember grants so tick() can gate corridor entry
        return out

    def step(self, dt):
        self.t += dt
        n_in = sum(1 for r in self.robots if _in_corridor(r, self))   # SPATIAL mutex monitor
        self.max_in_corridor = max(self.max_in_corridor, n_in)
        if n_in > 1:
            self.violations += 1

    def all_done(self):
        return all(r.qi >= r.n_tasks and r.carry is None for r in self.robots)

    def metrics(self):
        return {"delivered": sum(r.qi for r in self.robots),
                "corridor_violations": self.violations, "max_in_corridor": self.max_in_corridor}

    def draw(self):
        s, lay = self.screen, self.lay
        s.fill((24, 26, 32))
        # walls
        for (cx, cy, hw, hh, _ang) in lay.rects:
            pygame.draw.rect(s, (90, 95, 105), (cx - hw, cy - hh, 2 * hw, 2 * hh))
        # corridor gap (highlight) — green when free, red when occupied
        x0, y0, x1, y1 = lay.corridor
        n_in = sum(1 for r in self.robots if _in_corridor(r, self))
        col = (60, 170, 90) if n_in <= 1 else (210, 70, 70)
        gap = pygame.Surface((x1 - x0, y1 - y0), pygame.SRCALPHA)
        gap.fill((*col, 70))
        s.blit(gap, (x0, y0))
        pygame.draw.rect(s, col, (x0, y0, x1 - x0, y1 - y0), 2)
        # pickup (A) and delivery (B)
        pygame.draw.circle(s, (90, 150, 240), (int(lay.pickup_a[0]), int(lay.pickup_a[1])), 16, 3)
        pygame.draw.circle(s, (240, 165, 70), (int(lay.deliver_b[0]), int(lay.deliver_b[1])), 16, 3)
        s.blit(self.font.render("A pickup", True, (90, 150, 240)),
               (lay.pickup_a[0] - 28, lay.pickup_a[1] + 18))
        s.blit(self.font.render("B deliver", True, (240, 165, 70)),
               (lay.deliver_b[0] - 30, lay.deliver_b[1] + 18))
        # robots
        for r in self.robots:
            pygame.draw.circle(s, r.color, (int(r.x), int(r.y)), RR)
            if r.carry is not None:
                pygame.draw.rect(s, (240, 220, 90), (int(r.x) - 7, int(r.y) - RR - 14, 14, 12))
        hud = f"t={self.t:5.1f}s   corridor occupants now: {n_in}   mutex violations: {self.violations}"
        s.blit(self.font.render(hud, True, (235, 235, 240)), (16, 12))


def run(cfg, fsm_dir=FSM_DIR, gif=None, max_steps=16000, dt=1 / 60.0, every=18, scale=0.6):
    """Run the corridor world to completion (optionally capture a GIF); return the world.

    When `gif` is None, this runs an interactive pygame loop so the simulator window stays
    visible and updates on every frame. When `gif` is given, the same simulation is also
    sampled into frames for off-screen export.
    """
    w = CorridorWorld(cfg, fsm_dir)
    arb = w.arbiter()
    frames = []
    clock = pygame.time.Clock()
    if gif is None:
        for _ in range(max_steps):
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return w
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return w
            fsm_step(w, w.controllers, dt, arb)
            w.draw()
            pygame.display.flip()
            clock.tick(60)
            if w.all_done():
                break
    else:
        for stp in range(max_steps):
            fsm_step(w, w.controllers, dt, arb)
            if stp % every == 0:
                w.draw()
                img = Image.frombytes("RGB", w.screen.get_size(), _tobytes(w.screen, "RGB"))
                frames.append(img.resize((int(img.width * scale), int(img.height * scale))))
            if w.all_done():
                break
    if gif and frames:
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=int(every * dt * 1000), loop=0, optimize=True)
    if gif is None:
        pygame.quit()
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# E2E route: the corridor_e2e CERTIFIED FSMs with a WORLD-driven request signal.
#
# outputs/corridor_e2e/workdir/{robot1..3,arbiter}.json come from the full LLM-free-at-
# runtime end-to-end synthesis (see experiments/instances/corridor/run.sh phase (a)).
# Unlike the dock-style FSMs above, req_r{i} is NOT a robot output: it is an ENVIRONMENT
# signal (dir: world in contrauto/tasks/corridor.yaml) that BOTH the robot FSM and the
# arbiter FSM consume as an input.  The world raises req_r{i} when robot i has an active
# crossing task, withdraws it when the robot reaches side B (task done), and re-raises it
# only when a NEW task arrives -- with real gaps between tasks so GF(!req_i) holds.
#
# Two timing semantics of the certified contracts matter here (decoded from specs.ltl and
# the strategy graphs -- do NOT "fix" them in the sim, they are properties of the spec):
#
#  1. at_a / at_b are ABSORBING end-of-service markers, not per-trip position pulses.
#     The env assumption G(at_x & !crossing -> X at_x) together with the system guarantee
#     G(!(at_x & crossing)) implies that once at_a or at_b is raised it stays raised
#     forever and crossing is thereafter permanently impossible (robot FSM nodes 8-15 /
#     16-23 are closed classes with crossing=0).  In the only assumption-satisfying traces
#     with infinitely many crossings, at_a = at_b = false at every step -- so the live
#     world binds them to constant false and counts completed crossings physically.
#
#  2. The certified arbiter.json NEVER grants (all 24 nodes have grant bits 0!).  This is
#     not extraction laziness: in arbiter.slugsin the pair G(grant_i -> req_i) and
#     G(grant_i & req_i -> X grant_i) (UNPRIMED req in the antecedent) makes every
#     granting state losing -- the world may withdraw req_i one step ahead of the forced
#     grant persistence, a safety contradiction -- so ALL winning strategies satisfy the
#     no-starvation justice GF(!req_i | grant_i) vacuously through the world justice
#     GF(!req_i).  Against THIS world, where a demand is withdrawn only when it is served,
#     that strategy deadlocks (robots queue forever; zero crossings).
#     arbiter_fixed.json (same workdir) is re-synthesized from the one-token-corrected
#     encoding grant_i & req_i & req_i' -> grant_i' with `slugs --biasForAction`; pass it
#     via arbiter_json=... / --arbiter to obtain the live closed loop.
#
# The robot FSM emits crossing as a Moore justice signal that TOGGLES 1,0,1,0 while
# req & grant are held (node 3 <-> node 32); physically the crossing MISSION is therefore
# level-latched: latched on the first crossing=1 (entering the corridor is authorized;
# the arbiter holds the grant until release), released when the world withdraws req at
# side B.  A robot is inside the certified corridor only while latched, and latched
# implies granted, so the certified grant mutex IS the corridor mutex.
# ═══════════════════════════════════════════════════════════════════════════════

E2E_FSM_DIR = ROOT + "/outputs/corridor_e2e/workdir"


class E2ELayout(Layout):
    """Layout + a one-way RETURN lane: a second gap in the wall (default near the bottom)
    through which robots travel B->A after delivering.  The CERTIFIED resource is still the
    single-lane corridor gap (self.corridor, unchanged); the return lane is an uncontended
    one-way path, so completed robots never re-enter the certified corridor."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        wt = cfg.wall_thick
        gy0, gy1 = self.gap_y - cfg.gap_h / 2, self.gap_y + cfg.gap_h / 2
        ry1 = cfg.H - 26.0                      # return gap at the bottom of the wall...
        ry0 = ry1 - cfg.gap_h
        if gy1 > ry0 - 12:                      # ...unless the corridor is already there
            ry0 = 26.0
            ry1 = ry0 + cfg.gap_h
        gaps = sorted([(gy0, gy1), (ry0, ry1)])
        ys = [0.0, gaps[0][0], gaps[0][1], gaps[1][0], gaps[1][1], float(cfg.H)]
        self.rects = [(self.wx, (a + b) / 2.0, wt / 2, (b - a) / 2.0, 0.0)
                      for a, b in ((ys[0], ys[1]), (ys[2], ys[3]), (ys[4], ys[5]))
                      if b - a > 2.0]
        self.return_gap = (self.wx - wt, ry0, self.wx + wt, ry1)
        self.ret_y = (ry0 + ry1) / 2.0


class E2ERobot(SimAgent):
    """Worker driven by the certified corridor_e2e robot FSM (world-driven req).

    The FSM is bound by its OWN variable names (no role guessing -- role_of does not know
    at_a/at_b/crossing): inputs r{i}_at_a, r{i}_at_b, r{i}_task_arrived (one-tick pulse),
    req_r{i} (the WORLD's demand, read from world state) and grant_r{i} (arbiter);
    output r{i}_crossing.

    at_b driving is CONTRACT-SEMANTICS-FIRST, decided by which world model was certified.
    TWO independent properties are read off the certified strategy table:

      forced_atb -- one-step crossing PHYSICS G(crossing -> X at_b) is assumed: every
        successor of every crossing node has at_b=1, so the world MUST raise at_b the tick
        after a crossing output (feeding at_b=0 there is an illegal env move: it has no
        successor and stalls the FSM -- the e2e6 ~5900-stalls bug).
      one_shot -- at_b is ABSORBING: no crossing node is reachable from the crossing
        nodes' successors (strategy-graph recurrence check).  With e2e6's persistence
        assumption G(at_b & !crossing -> X at_b) + guarantee G(!(at_b & crossing)) the
        at_b=1 class is closed and crossing=0 there: each robot crosses AT MOST ONCE.

    The three certified world models map to:
      * shuttle, markers never raised (corridor_e2e / e2e4): no physics assumption
        (forced_atb False); at_a/at_b are absorbing end-of-service markers, the live world
        keeps them false and the robot shuttles A->B->A forever (see the module note).
      * one-shot (corridor_e2e6): forced_atb & not recurrent -> raise at_b after the
        crossing and hold it forever; the robot parks on side B (returning would drop at_b
        while !crossing, illegal under the persistence assumption).
      * pulse shuttle (corridor_e2e7): forced_atb & recurrent -- the contract swaps the
        at_b persistence for GF(!at_b) ("being at side B is temporary"), so at_b must
        FALL: raise it the tick after the crossing, drop it when the robot physically
        leaves side B on the return lane, then serve the next task (crossing nodes are
        re-entered; holding at_b forever would violate GF(!at_b) -- the mirror image of
        the e2e6 bug)."""

    def __init__(self, rid, fsm_path, lay, color, n_robots):
        self.id, self.name, self.color = rid, f"r{rid}", color
        i = rid
        data = load_fsm(fsm_path)
        variables = data["variables"]

        def variable(role):
            for candidate in (f"r{i}_{role}", f"{role}_r{i}", role):
                if candidate in variables:
                    return candidate
            raise ValueError(f"{fsm_path} has no variable for {role!r}; variables: {variables}")

        self._xvar = variable("crossing")
        at_b = variable("at_b")
        task_arrived = variable("task_arrived")
        request = variable("req")
        grant = variable("grant")
        # No `at_a` sensor: corridor.yaml no longer declares it (it grounded to a constant
        # `False` -- a robot's presence at side A is never gated/interesting -- so declaring
        # it only let the decomposer write an unsatisfiable `G F(at_a)` assumption; see
        # corridor.yaml's own note). An OLDER already-synthesized FSM that still happens to
        # declare `r{i}_at_a` is unaffected: FsmController only tracks variables it is told
        # about here, and that bit was always fed as constant 0 anyway.
        self.SENSORS = {
            at_b: lambda a, w: a.atb,                     # raised per the certified dynamics (below)
            task_arrived: lambda a, w: a.task_pulse,
            request: lambda a, w: a.req,                  # WORLD-driven request (env signal)
        }
        self.GRANTS = [grant]
        self.REQUESTS = []                              # req is NOT a robot output any more
        self.ACTIONS = {}                               # motion is driven by the mission latch
        self.ctrl = FsmController(data, list(self.SENSORS) + self.GRANTS, [self._xvar])
        # read the certified world model off the strategy table (see class docstring)
        ib = variables.index(at_b)
        nodes = data["nodes"]
        xids = {nid for nid, n in nodes.items() if n["state"][-1]}
        self.forced_atb = bool(xids) and all(nodes[str(s)]["state"][ib]
                                             for nid in xids for s in nodes[nid]["trans"])
        recurrent = False
        if self.forced_atb:             # can the strategy cross AGAIN after a crossing?
            seen, frontier = set(), {str(s) for nid in xids for s in nodes[nid]["trans"]}
            while frontier and not recurrent:
                nid = frontier.pop()
                if nid in seen:
                    continue
                seen.add(nid)
                recurrent = nid in xids
                frontier |= {str(s) for s in nodes[nid]["trans"]}
        self.one_shot = self.forced_atb and not recurrent
        self.atb = False                # the world's at_b report (forced_atb modes only)
        ang = rid * 1.7
        self.home = (lay.pickup_a[0] + 55 * math.cos(ang), lay.pickup_a[1] + 55 * math.sin(ang))
        # Single-file queue directly on the entry line (gap_y): staggered only in x, one robot
        # behind the next, spaced by more than the peer-avoidance diameter (2*RR+6) so CBF
        # doesn't jam them. No y-offset -- earlier this queued at gap_y - 66 - 46*(rid-1), which
        # pulled every waiting robot up and to the left of the mouth for no reason (only the
        # lateral spread matters; the corridor is not vertical here).
        self.queue_pt = (lay.wx - lay.cfg.wall_thick - 18 - 40 * rid, lay.gap_y)
        self.x, self.y = self.home
        self.target = self.home
        self.stage = "idle"
        self.req = False                # the world's standing crossing demand for this robot
        self.task_pulse = False         # one-tick task_arrived pulse
        self.crossing_active = False    # mission latch (see module note on the Moore toggle)
        self.cool = 0.0                 # world task-arrival cooldown (set by the world)
        self._demand_age = 0.0          # how long the current standing req has waited unserviced
        self.n_done = 0                 # completed crossings (physical side-B arrivals)
        self.n_tasks = 0                # tasks the world has issued
        self._nav = WaypointNav(obb=True)
        self.vel = np.zeros(2)
        self.priority = rid

    def pose(self):
        return (self.x, self.y)

    def _near(self, pt, tol):
        return (self.x - pt[0]) ** 2 + (self.y - pt[1]) ** 2 <= tol * tol

    def act(self, action, world):
        lay, wt = world.lay, world.lay.cfg.wall_thick
        if self.req and action.get(self._xvar, 0):
            self.crossing_active = True             # corridor entry authorized (grant is held)
            if self.forced_atb:
                self.atb = True                     # world honors G(crossing -> X at_b): the
                                                    # NEXT observe reports at_b (held forever
                                                    # in one-shot mode; dropped when the robot
                                                    # leaves side B in pulse-shuttle mode)
        if self.stage == "idle":
            self.target = self.home
            if self.req and self._near(self.home, RR + 14):
                self.stage = "queue"
        if self.stage == "queue":                   # wait on side A; never enter ungranted
            self.target = self.queue_pt
            if self.crossing_active:
                self.stage = "enter"
        if self.stage == "enter":                   # west mouth of the CERTIFIED corridor
            self.target = (lay.wx - wt - 18, lay.gap_y)
            if self._near(self.target, 26):
                self.stage = "thru"
        if self.stage == "thru":                    # straight through the single-lane gap
            self.target = (lay.wx + wt + 34, lay.gap_y)
            if self.x > lay.wx + wt + 20:
                # Physically CLEAR of the passage (already outside _in_corridor's bounds) --
                # release the demand/access HERE, not after the separate walk to deliver_b.
                # The corridor is the contended resource; holding req/grant through an
                # unrelated walk on the open floor starves every other robot's turn for no
                # mutex reason (the certified guarantee only needs grant -> crossing, which
                # this robot no longer needs once it has left the passage).
                self.req = False
                self.crossing_active = False
                self.stage = "deliver"
        if self.stage == "deliver":
            self.target = lay.deliver_b
            if _at(self, lay.deliver_b):            # actual delivery: count it, then head back
                self.n_done += 1
                # one-shot contract (at_b absorbing): PARK on side B forever -- returning
                # would drop at_b while !crossing, an illegal env move.  Shuttle contract:
                # return home via the one-way lane for the next task.
                self.stage = "park" if self.one_shot else "ret_e"
        if self.stage == "park":                    # stationed on side B (at_b persists)
            ang = 2.1 * self.id + 0.7
            self.target = (min(lay.cfg.W - 40.0, max(lay.wx + wt + 40.0,
                               lay.deliver_b[0] + 60 * math.cos(ang))),
                           min(lay.cfg.H - 40.0, max(40.0, lay.deliver_b[1] + 60 * math.sin(ang))))
        if self.stage == "ret_e":                   # east mouth of the one-way return lane
            self.target = (lay.wx + wt + 18, lay.ret_y)
            if self._near(self.target, 26):
                self.stage = "ret_w"
        if self.stage == "ret_w":
            self.target = (lay.wx - wt - 34, lay.ret_y)
            if self.x < lay.wx - wt - 20:
                self.stage = "go_home"
                self.atb = False        # physically left side B -> at_b falls (pulse-shuttle
                                        # mode honors GF(!at_b); no persistence assumption)
        if self.stage == "go_home":
            self.target = self.home
            if self._near(self.home, RR + 14):
                self.stage = "idle"

    def tick(self, dt, world):
        lay = world.lay
        wp = self._nav.target((self.x, self.y), self.target, lay.rects, lay.bounds, RR)
        p = np.array([self.x, self.y], float)
        seg_dl = world.t + float(np.linalg.norm(np.array(wp, float) - p)) / SPEED * 1.3 + dt
        walls = [Always(0.0, 1e9, out_obb(cx, cy, hw, hh, ang, margin=RR + 4))
                 for (cx, cy, hw, hh, ang) in lay.rects]
        formula = And([Eventually(world.t, seg_dl, in_circle(wp[0], wp[1], RR))]
                      + [Always(0.0, 1e9, out_circle(o.x, o.y, 2 * RR + 6))
                         for o in world.robots if o is not self and o.priority > self.priority]
                      + walls)
        dirn = np.array(wp, float) - p
        nrm = float(np.linalg.norm(dirn)) or 1.0
        u_nom = 6.0 * (dirn / nrm * SPEED - self.vel)
        p_new, self.vel, _, _ = stl_cbf_step(formula, p, self.vel, world.t, u_nom, dt, v_max=SPEED)
        self.x = min(lay.cfg.W - RR, max(RR, float(p_new[0])))
        self.y = min(lay.cfg.H - RR, max(RR, float(p_new[1])))


class E2ECorridorWorld(SimWorld):
    """SimWorld for the corridor_e2e FSMs: world-driven req injected into BOTH the robot
    envs (as a sensor) and the arbiter (via Coordinator.source); continuous task service."""

    def __init__(self, cfg: Config, fsm_dir=E2E_FSM_DIR, arbiter_json=None, robot_suffix=""):
        random.seed(cfg.seed)
        self.cfg = cfg
        self.lay = E2ELayout(cfg)
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.W, cfg.H))
        # Rendered at native (cfg.W, cfg.H) resolution, then the WHOLE frame is downscaled
        # (run_e2e's `scale`, typically 0.6) before it lands in the GIF -- so the on-screen
        # font must be sized well above what looks right in this live window, or every label
        # and the HUD line come out too small to read once shrunk. 26 -> 40 keeps text
        # legible post-scale (was rendering at an effective ~16px).
        self.font = pygame.font.SysFont(None, 40)
        self.robots = [E2ERobot(i + 1, f"{fsm_dir}/robot{i+1}{robot_suffix}.json", self.lay,
                                ROBOT_COLORS[i % len(ROBOT_COLORS)], cfg.n_robots)
                       for i in range(cfg.n_robots)]
        for r in self.robots:                       # stagger the first task arrivals
            r.cool = 0.6 + 0.5 * r.id
        self.controllers = {r.name: r.ctrl for r in self.robots}
        ad = load_fsm(arbiter_json or f"{fsm_dir}/arbiter.json")
        rq = [v for v in ad["variables"] if v.startswith("req")]
        gr = [v for v in ad["variables"] if v.startswith("grant")]
        self.last_grants = {}
        self._coord = Coordinator(ad, rq, gr, source=self._world_reqs, apply=self._grant_apply)
        self.violations = 0
        self.max_in_corridor = 0
        self.in_wo_grant = 0            # ticks with a robot inside the corridor but ungranted
        self.grant_edges = 0            # grant rising edges (how often access was awarded)
        self.max_wait = 0.0             # longest standing demand without a grant (seconds)
        self._waits = [0.0] * cfg.n_robots
        self.completions = []           # timestamps of completed crossings (liveness trail)
        self._prev_done = 0
        self._prev_grants = {}
        self.t = 0.0

    def agents(self):
        return self.robots

    def arbiter(self):
        return self._coord

    def _world_reqs(self, reqs, world):
        """The arbiter's env: the WORLD's request signals (not collected robot outputs)."""
        return {f"req_r{r.id}": int(r.req) for r in self.robots}

    def _grant_apply(self, out, world):
        for g, v in out.items():
            if v and not self._prev_grants.get(g, 0):
                self.grant_edges += 1
        self._prev_grants = dict(out)
        self.last_grants = dict(out)
        return out

    def step(self, dt):
        self.t += dt
        # certified-resource monitors
        n_in = sum(1 for r in self.robots if _in_corridor(r, self))
        self.max_in_corridor = max(self.max_in_corridor, n_in)
        if n_in > 1:
            self.violations += 1
        for r in self.robots:
            if _in_corridor(r, self) and not self.last_grants.get(f"grant_r{r.id}", 0):
                self.in_wo_grant += 1
            if r.req and not self.last_grants.get(f"grant_r{r.id}", 0):
                self._waits[r.id - 1] += dt
                self.max_wait = max(self.max_wait, self._waits[r.id - 1])
            else:
                self._waits[r.id - 1] = 0.0
        # liveness trail
        done = sum(r.n_done for r in self.robots)
        if done > self._prev_done:
            self.completions += [self.t] * (done - self._prev_done)
            self._prev_done = done
        # task process: consume pulses; new tasks arrive only when the robot is back home,
        # after a random cooldown -- so req always DROPS between tasks (GF !req_i holds)
        for r in self.robots:
            if r.task_pulse:
                r.task_pulse = False
            elif not r.req and r.stage == "idle":
                r.cool -= dt
                if r.cool <= 0:
                    r.req = True                    # the WORLD raises the demand...
                    r.task_pulse = True             # ...and pulses task_arrived for one step
                    r.n_tasks += 1
                    r.cool = random.uniform(0.8, 2.2)
                    r._demand_age = 0.0
            elif r.req and not r.crossing_active:
                # WORLD-side patience: give up on a standing demand the robot never even
                # entered the corridor for, so GF(!req)/GF(task_arrived) hold unconditionally
                # (see DEMAND_TIMEOUT) instead of only via the robot completing a crossing.
                r._demand_age += dt
                if r._demand_age >= DEMAND_TIMEOUT:
                    r.req = False
                    r.stage = "idle"
                    r.target = r.home
                    r.cool = random.uniform(0.8, 2.2)
                    r._demand_age = 0.0

    def metrics(self):
        q = max(self.t, 1e-9) / 4.0
        crossings = sum(r.n_done for r in self.robots)
        return {"delivered": crossings,
            "violations": self.violations,
            "crossings_per_robot": [r.n_done for r in self.robots],
            "crossings_total": crossings,
                "tasks_issued": sum(r.n_tasks for r in self.robots),
                "corridor_violations": self.violations,
                "max_in_corridor": self.max_in_corridor,
                "in_corridor_without_grant_ticks": self.in_wo_grant,
                "grant_edges": self.grant_edges,
                "max_wait_s": round(self.max_wait, 2),
                "completions_by_quarter": [sum(1 for c in self.completions if i * q <= c < (i + 1) * q)
                                           for i in range(4)],
                "fsm_stalls": {**{r.name: r.ctrl._stuck for r in self.robots},
                               "arbiter": self._coord.ctrl._stuck}}

    def draw(self):
        s, lay = self.screen, self.lay
        s.fill((24, 26, 32))
        for (cx, cy, hw, hh, _ang) in lay.rects:
            pygame.draw.rect(s, (90, 95, 105), (cx - hw, cy - hh, 2 * hw, 2 * hh))
        # certified corridor: green when free, red when >1 inside
        x0, y0, x1, y1 = lay.corridor
        n_in = sum(1 for r in self.robots if _in_corridor(r, self))
        col = (60, 170, 90) if n_in <= 1 else (210, 70, 70)
        gap = pygame.Surface((x1 - x0, y1 - y0), pygame.SRCALPHA)
        gap.fill((*col, 70))
        s.blit(gap, (x0, y0))
        pygame.draw.rect(s, col, (x0, y0, x1 - x0, y1 - y0), 2)
        s.blit(self.font.render("corridor A->B", True, col), (x0 - 42, y0 - 24))
        # one-way return lane
        rx0, ry0, rx1, ry1 = lay.return_gap
        ret = pygame.Surface((rx1 - rx0, ry1 - ry0), pygame.SRCALPHA)
        ret.fill((95, 125, 190, 55))
        s.blit(ret, (rx0, ry0))
        pygame.draw.rect(s, (95, 125, 190), (rx0, ry0, rx1 - rx0, ry1 - ry0), 1)
        s.blit(self.font.render("return B->A", True, (120, 145, 200)), (rx0 - 40, ry0 - 24))
        pygame.draw.circle(s, (90, 150, 240), (int(lay.pickup_a[0]), int(lay.pickup_a[1])), DRAW_RR, 3)
        pygame.draw.circle(s, (240, 165, 70), (int(lay.deliver_b[0]), int(lay.deliver_b[1])), DRAW_RR, 3)
        s.blit(self.font.render("A depart", True, (90, 150, 240)),
               (lay.pickup_a[0] - 26, lay.pickup_a[1] + 18))
        s.blit(self.font.render("B deliver", True, (240, 165, 70)),
               (lay.deliver_b[0] - 30, lay.deliver_b[1] + 18))
        for r in self.robots:
            pygame.draw.circle(s, r.color, (int(r.x), int(r.y)), DRAW_RR)
            if self.last_grants.get(f"grant_r{r.id}", 0):        # granted: yellow ring
                pygame.draw.circle(s, (240, 220, 90), (int(r.x), int(r.y)), DRAW_RR + 5, 2)
            if r.req:                                            # standing world demand
                pygame.draw.rect(s, (240, 220, 90), (int(r.x) - 7, int(r.y) - DRAW_RR - 16, 14, 12))
        grants = "".join(str(self.last_grants.get(f"grant_r{i+1}", 0)) for i in range(self.cfg.n_robots))
        reqs = "".join(str(int(r.req)) for r in self.robots)
        hud = (f"t={self.t:5.1f}s  req={reqs}  grant={grants}  in corridor: {n_in}  "
               f"violations: {self.violations}  crossings: {sum(r.n_done for r in self.robots)}")
        s.blit(self.font.render(hud, True, (235, 235, 240)), (16, 12))


def run_e2e(cfg=None, fsm_dir=E2E_FSM_DIR, arbiter_json=None, steps=6000, gif=None,
            dt=1 / 60.0, every=18, scale=0.6, robot_suffix=""):
    """Run the corridor world on the certified corridor_e2e FSMs with world-driven req for
    a FIXED number of steps (continuous task service; optionally capture a GIF); return the
    world.  arbiter_json overrides {fsm_dir}/arbiter.json (e.g. the arbiter_fixed.json
    re-synthesis -- see the module note on why arbiter.json never grants).  robot_suffix
    selects alternate robot strategies: "_fixed" loads robot{i}_fixed.json -- for the
    corridor_e2e4 workdir these restore the ONE justice guarantee GF(!req_i | crossing_i)
    that the repair rounds dropped from the robot contracts (the certified e2e4
    robot{i}.json have ZERO crossing=1 nodes, so they can never serve any demand)."""
    w = E2ECorridorWorld(cfg or Config(), fsm_dir, arbiter_json, robot_suffix)
    arb = w.arbiter()
    frames = []
    clock = pygame.time.Clock()
    if gif is None:
        for _ in range(steps):
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return w
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return w
            fsm_step(w, w.controllers, dt, arb)
            w.draw()
            pygame.display.flip()
            clock.tick(60)
    else:
        for stp in range(steps):
            fsm_step(w, w.controllers, dt, arb)
            if stp % every == 0:
                w.draw()
                img = Image.frombytes("RGB", w.screen.get_size(), _tobytes(w.screen, "RGB"))
                frames.append(img.resize((int(img.width * scale), int(img.height * scale))))
    if gif and frames:
        os.makedirs(os.path.dirname(gif), exist_ok=True)
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=int(every * dt * 1000), loop=0, optimize=True)
    if gif is None:
        pygame.quit()
    return w


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n", type=int, default=3, help="robots (need that many robot FSMs)")
    p.add_argument("--gif", default=ROOT + "/outputs/case_studies/gifs/corridor.gif")
    p.add_argument("--e2e", action="store_true",
                   help="run the corridor_e2e certified FSMs (world-driven req) instead")
    p.add_argument("--steps", type=int, default=6000, help="steps for the --e2e route")
    p.add_argument("--arbiter", default=None,
                   help="arbiter json override for --e2e (e.g. .../arbiter_fixed.json)")
    p.add_argument("--fsm-dir", default=E2E_FSM_DIR, help="FSM workdir for the --e2e route")
    p.add_argument("--robot-suffix", default="",
                   help='robot strategy suffix for --e2e (e.g. "_fixed" -> robot1_fixed.json)')
    a = p.parse_args()
    if a.e2e:
        gif = a.gif if a.gif != p.get_default("gif") else ROOT + "/outputs/instances/corridor/corridor_e2e.gif"
        w = run_e2e(Config(n_robots=a.n, seed=a.seed), fsm_dir=a.fsm_dir, arbiter_json=a.arbiter,
                    steps=a.steps, gif=gif, robot_suffix=a.robot_suffix)
        print("corridor_e2e:", w.metrics(), "GIF:", gif)
    else:
        fsm = ROOT + ("/outputs/two_layer_dock_n4/workdir" if a.n >= 4 else "/outputs/two_layer_dock_n3/workdir")
        w = run(Config(n_robots=a.n, seed=a.seed), fsm_dir=fsm, gif=a.gif)
        print("corridor:", w.metrics(), "GIF:", a.gif)
