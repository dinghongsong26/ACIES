"""Multi-robot warehouse simulator with a shared-resource arbiter — expressed
through the common simulator interface (contrauto.sim_iface).

Each robot is a SimAgent that only DECLARES atomic propositions + actions (+ the
REQUESTS it emits / GRANTS it consumes); both robot FSMs AND the synthesized arbiter
FSM are executed by the shared FsmController, and run_fsm performs the request->grant
handshake. No hand-written phase mirror, no per-sim execution code — that is what
makes this simulator "not a redo" of a hand-rolled per-task loop.
"""
import argparse, math, os, random, time
import pygame
from simulator.core import load_slugs, COLORS, BaseRobot, BaseGame
from contrauto.sim_iface import (
    SimAgent, SimWorld, Nav, Stationary, FsmController, Coordinator, fsm_step, run_fsm, role_of)
from contrauto.grounding import Grounding, bind_groundings
from cbf.obstacles import (boxes_as_disks, merge_obstacles,
                           other_robots_as_obstacles, warehouse_shelf_plank_rects)
from cbf.configs import get_cbf_config
from contrauto.geometry import WaypointNav

# NOTE: an EARLIER +40 shift of DOCK_Y/SHELF_TOP/SPAWN_Y ALL TOGETHER (dock/shelf-top/
# spawn-row) broke navigation -- A/B tested across 5 seeds, 5/5 stalled -- because it kept
# the dock lanes at the SAME tight clearance under the shelf columns, just moved. THIS shift
# is different: only DOCK_Y/SPAWN_Y move (+60), SHELF_TOP is untouched, so the gap between
# the shelf grid and the dock/spawn-row area only ever WIDENS, giving robots more room to
# peel off toward the shelf vs. wait near the dock without cutting across each other at the
# start -- re-validated across 5 seeds (see conversation) before this landed.
DOCK_Y, ROBOT_R, OBJ_SZ, SPEED = 620, 14, 14, 130
DOCK_W, DOCK_H = 300, 45   # a small marked rectangle, not a full-width bar (see _draw_arena)
SPAWN_Y = 550   # initial robot row -- same +60 shift as DOCK_Y, keeping their relative gap
# Side-by-side spacing for robots queued/parked near each other (dock queue, spawn row).
# The CBF config's own pairwise minimum separation is 2*ROBOT_R + margin = 30px (see
# cbf/configs.py); that is a hard collision floor, not a comfortable resting distance --
# with 3+ mutually-close robots (not just a pair) CBF avoidance has to satisfy several
# constraints from several neighbors at once, and right at that floor the equilibrium is
# unstable, so idle robots visibly jitter fighting a fine margin instead of settling.
QUEUE_SPACING = 3 * (ROBOT_R + 6)
CBF_TASK = "warehouse"
MISSION = "warehouse_arbiter"
SHELF_TOP, SHELF_ROWS, SHELF_COLS = 50, 4, 3
# NOTE: recentering this at 210 (for WarehouseGame's new 1000px-wide, panel-less window) was
# tried and reverted -- see the DOCK_Y note above; it put the dock lanes squarely under a
# shelf column in a way this layout's original 80 avoids, and that combination is what
# triggered the navigation stalls, not the centering by itself in isolation.
SHELF_ROW_H, SHELF_ROW_GAP, SHELF_COL_W, SHELF_COL_GAP, SHELF_X0 = 20, 110, 140, 80, 80
# Center the dock under the shelf grid (not an arbitrary arena-relative constant) so the
# two zones read as one aligned layout instead of two independently-placed shapes.
DOCK_X = int(SHELF_X0 + ((SHELF_COLS - 1) * (SHELF_COL_W + SHELF_COL_GAP) + SHELF_COL_W) / 2)
# Spawn pacing: two robots deliver ~0.4 goods/s (mean cycle ~4–5 s each). Spawning every
# 120 frames (~2 s) outran that and filled the shelves (peak ~11). 300 frames (~5 s) plus a
# soft active cap keeps backlog near the fleet size instead of stacking every free shelf.
SPAWN_FRAMES = 300
MAX_ACTIVE_BINS = 4


def _shelf_xy(row, col):
    return SHELF_X0 + col * (SHELF_COL_W + SHELF_COL_GAP), SHELF_TOP + row * (SHELF_ROW_H + SHELF_ROW_GAP)


# ── scenario declarations: per-robot predicates + actions ────────────────────
def _avail_bins(world, who):
    return [b for b in world.world
            if b.carrier is None and not getattr(b, "delivered", False)
            and (b.reserved is None or b.reserved == who)]


# The ONE place that reads raw warehouse state, keyed by capability token.
PROVIDERS = {
    "agent.pos":          lambda a, w: (a.x, a.y),
    "agent.carry":        lambda a, w: a.carry,
    "world.goods":        lambda a, w: w.world,
    "world.avail_bins":   lambda a, w: _avail_bins(w, a.id),   # bins free or reserved for this robot
    "geom.shelf_centers": lambda a, w: [(sx + SHELF_COL_W // 2, sy + SHELF_ROW_H // 2)
                                        for r in range(SHELF_ROWS) for c in range(SHELF_COLS)
                                        for sx, sy in [_shelf_xy(r, c)]],
    "geom.dock_y":        lambda a, w: DOCK_Y,
    "world.robot_count":  lambda a, w: len(w.robots),
}


class AtShelf(Grounding):
    # Arrival radius must exceed the closest approach the CBF allows across a shelf plank; keep it
    # identical to _do_pickup's grab radius (ROBOT_R+OBJ_SZ+14) or a robot can livelock 2 px outside.
    def evaluate(self, cap):
        pos = cap.get("agent.pos")
        # Exclude a lingering delivered item (see DELIVER_HOLD_FRAMES): it sits at the
        # DOCK, and a robot queued/dropping off there can be well within this proximity
        # radius of it -- without the exclusion, at_shelf spuriously fires for a robot
        # that is actually at the dock, confusing the FSM (observed: full stall).
        if any(cap.dist(pos, (b.x, b.y)) < ROBOT_R + OBJ_SZ + 14
               for b in cap.get("world.goods")
               if b.carrier is None and not getattr(b, "delivered", False)):
            return True
        if cap.get("agent.carry") is not None:
            return any(cap.dist(pos, c) < ROBOT_R + OBJ_SZ + 12 for c in cap.get("geom.shelf_centers"))
        return False


class HasItem(Grounding):
    def evaluate(self, cap): return cap.get("agent.carry") is not None


def _dock_lane_half_width(n):
    """How far the outermost robot's dock lane sits from DOCK_X, for a fleet of `n`.
    AtDock's X-tolerance is derived from this (not a fixed constant) so it always exactly
    covers whatever lane layout _dock_lane_x actually uses, for any fleet size."""
    return (max(n, 1) - 1) / 2 * QUEUE_SPACING + ROBOT_R + 6


def _dock_lane_x(a, world):
    """This robot's own dock lane X -- fixed for its lifetime, shared by waiting AND
    dropping off, so entering the dock is a straight walk up its own lane (pure Y motion,
    see _dock_target) instead of a lateral detour into a shared center point that funnels
    every robot through the middle of whoever else is currently queued."""
    idx = world.robots.index(a)
    n = len(world.robots)
    return DOCK_X + (idx - (n - 1) / 2) * QUEUE_SPACING


class AtDock(Grounding):
    def evaluate(self, cap):
        x, y = cap.get("agent.pos")
        if y < cap.get("geom.dock_y") - ROBOT_R - 6:
            return False
        return abs(x - DOCK_X) <= _dock_lane_half_width(cap.get("world.robot_count"))


class HasTask(Grounding):
    def evaluate(self, cap): return bool(cap.get("world.avail_bins")) or cap.get("agent.carry") is not None


PREDICATES = bind_groundings({
    "at_shelf":  AtShelf(),
    "has_item":  HasItem(),
    "at_dock":   AtDock(),
    "task":      HasTask(),
}, PROVIDERS)


def _shelf_target(a, world):
    if a.carry is not None:
        return a.pose()
    cands = _avail_bins(world, a.id)
    if not cands:
        return a.pose()
    b = min(cands, key=lambda b: math.hypot(a.x - b.x, a.y - b.y))
    if a._reserved_bin is not None and a._reserved_bin is not b:
        a._release_reservation()
    b.reserved = a.id; a._reserved_bin = b
    a.status = "-> SHELF"
    return (b.x, b.y)


def _dock_target(a, world):
    a.status = "-> DOCK"
    # Every robot has its own fixed lane (see _dock_lane_x) for its WHOLE dock visit --
    # waiting outside and dropping off both happen at this same X, so entering the dock
    # is a straight walk forward in Y, never a lateral detour into a shared center point.
    # Forcing every granted robot onto one exact (DOCK_X, ...) point used to make it cut
    # across whoever else was still queued in the middle lanes to get there.
    x = _dock_lane_x(a, world)
    if not a.has_dock_grant():
        # A dock grant is an admission token, not just permission to drop off. Stay north
        # of the AtDock threshold, in-lane, while the arbiter serves another robot.
        return (x, DOCK_Y - ROBOT_R - 32)
    return (x, DOCK_Y - ROBOT_R - 4)


def _do_pickup(a, world):
    if a.carry is None:
        for b in world.world:
            # grab radius == _at_shelf arrival radius, or the FSM can advance to pickup
            # at a spot where the grab silently fails
            if b.carrier is None and math.hypot(a.x - b.x, a.y - b.y) < ROBOT_R + OBJ_SZ + 14:
                b.carrier, b.reserved, a.carry = a.id, None, b
                a._reserved_bin, a.status = None, "PICKED"
                break


def _do_dropoff(a, world):
    if a.carry:
        a.carry.carrier = None; a.carry.y = DOCK_Y + 5; a.carry.delivered = True
        a.carry, a._just_delivered = None, True
        a.done_count += 1; a.status = "DELIVERED"


# done_when on the Nav actions: HOLD the navigation (don't advance the FSM) until the
# robot physically arrives. This replaces relying on the FSM's _stuck fallback as implicit
# nav-persistence (which stalls when a robot waits long for a free shelf), so navigation
# never registers as an off-model stall.
ACTIONS = {
    "go_to_shelf": Nav(_shelf_target, done_when=lambda a, w: PREDICATES["at_shelf"](a, w)),
    # Hold until the grasp sticks — if the FSM also asserts goto_dock, act() still runs
    # the pickup effect first; done_when keeps us here when the grab misses a frame.
    "pickup":      Stationary(_do_pickup, done_when=lambda a, w: a.carry is not None),
    "go_to_dock":  Nav(_dock_target, done_when=lambda a, w: a.y >= DOCK_Y - ROBOT_R - 6),
    "dropoff":     Stationary(_do_dropoff),
    "done":        Stationary(None),
}


class WorldObj:
    """A good on a shelf; `reserved` holds a robot id that has claimed but not yet picked it."""
    _nxt = 0
    def __init__(self, x, y):
        self.id = WorldObj._nxt; WorldObj._nxt += 1
        self.x, self.y, self.carrier, self.reserved = float(x), float(y), None, None

    def is_available(self):
        return self.carrier is None and self.reserved is None

    def draw(self, surf):
        if self.carrier is not None: return
        if getattr(self, "delivered", False):
            col = COLORS["green"]
        else:
            col = COLORS["orange"] if self.reserved is not None else COLORS["yellow"]
        r = pygame.Rect(self.x - OBJ_SZ // 2, self.y - OBJ_SZ // 2, OBJ_SZ, OBJ_SZ)
        pygame.draw.rect(surf, col, r); pygame.draw.rect(surf, COLORS["border"], r, 1)


def _cbf_obstacles(world, robots, self_robot):
    """Return (disk obstacles for free goods + peer robots, shelf rects, obstacle velocities)."""
    # A delivered item now lingers (visibly) at the dock for DELIVER_HOLD_FRAMES instead
    # of vanishing same-frame -- it must NOT count as a CBF obstacle, or it blocks the
    # exact spot every other robot needs to reach to drop off its own item, deadlocking
    # the dock (observed: progress-bounded stall once >=1 delivered item was lingering).
    bins = [d for b in world if b.carrier is None and not getattr(b, "delivered", False)
            and b is not getattr(self_robot, "_reserved_bin", None)
            for d in boxes_as_disks([(b.x, b.y)], OBJ_SZ / 2)]
    idx = robots.index(self_robot)
    peers = other_robots_as_obstacles([(r.x, r.y) for r in robots], ROBOT_R, idx)
    peer_vel = [(r.vx, r.vy) for i, r in enumerate(robots) if i != idx]
    # Pass this module's ACTUAL current shelf geometry -- these rects are real CBF
    # obstacles, not decoration, so if _shelf_xy's constants ever move (e.g. a layout
    # re-center), the collision boxes must move with them or a robot would visually
    # clip through a shelf that only moved on screen (see the caller's own comment).
    shelf_rects = warehouse_shelf_plank_rects(
        top=SHELF_TOP, x0=SHELF_X0, shelf_w=SHELF_COL_W, shelf_h=SHELF_ROW_H,
        col_gap=SHELF_COL_GAP, row_gap=SHELF_ROW_GAP, rows=SHELF_ROWS, cols=SHELF_COLS)
    return merge_obstacles(bins, peers), shelf_rects, [(0.0, 0.0)] * len(bins) + peer_vel


# ── backend: physical primitives implementing the SimAgent port ──────────────
class WarehouseRobot(BaseRobot, SimAgent):
    """Worker that reserves a bin before fetching it (so two robots don't target the same good)."""
    def __init__(self, rid, name, x, y, color_key, fsm_path):
        super().__init__(rid, name, x, y, color_key, CBF_TASK, ROBOT_R, SPEED)
        self.name = name
        self.bind_fsm(fsm_path, PREDICATES, ACTIONS)
        self.carry, self._just_delivered, self.action = None, False, {}
        self._reserved_bin, self.done_count = None, 0
        self.grants = {}
        self._nav = WaypointNav()
        self._bounds = get_cbf_config(CBF_TASK).world_bounds
        self.target = (float(x), float(y))

    def _release_reservation(self):
        if self._reserved_bin is not None:
            self._reserved_bin.reserved = None
            self._reserved_bin = None

    def pose(self):
        return (self.x, self.y)

    def has_dock_grant(self):
        return any(value for name, value in self.grants.items()
                   if role_of(name) == "dock_grant")

    def act(self, action, world):
        self._just_delivered = False
        self.action = action
        SimAgent.act(self, action, world)

    def tick(self, dt, world):
        self.tx, self.ty = self.target
        holding_pose = math.hypot(self.tx - self.x, self.ty - self.y) < 1.0
        # Stationary dropoff/done/idle: park — no CBF, no vel/acc. Previously this also
        # required being at the dock, so an idle robot ANYWHERE ELSE (no task available,
        # or between actions) still ran full CBF-reactive movement with a target equal to
        # its own position -- nominally zero velocity, but the CBF QP still nudges it
        # whenever a moving peer passes nearby, which reads as constant idle jitter. A
        # peer's own CBF still treats a parked (zero-velocity) robot as an obstacle at its
        # current position regardless (see other_robots_as_obstacles), so skipping motion
        # here doesn't remove it from anyone else's avoidance -- it only stops IT from
        # reacting to nothing.
        if holding_pose:
            self.vx = self.vy = 0.0
            return
        self.move(dt, world.speed, world.world, world.robots)

    def move(self, dt, spd, goods, robots):
        obs, shelf_rects, obs_vel = _cbf_obstacles(goods, robots, self)
        rect_tuples = [(rc.x, rc.y, rc.half_w, rc.half_h) for rc in shelf_rects]
        nav = self._nav.target((self.x, self.y), (self.tx, self.ty), rect_tuples, self._bounds, ROBOT_R)
        BaseRobot.move(self, dt, spd, obs, rect_obstacles=shelf_rects,
                       obstacle_velocities=obs_vel, nav_target=nav)
        if self.carry:
            self.carry.x, self.carry.y = self.x, self.y - ROBOT_R - OBJ_SZ // 2

    def draw(self, surf, font):
        super().draw(surf, font)
        if self.carry:
            r = pygame.Rect(int(self.x) - OBJ_SZ // 2, int(self.y) - ROBOT_R - OBJ_SZ - 2, OBJ_SZ, OBJ_SZ)
            pygame.draw.rect(surf, COLORS["yellow"], r); pygame.draw.rect(surf, COLORS["white"], r, 1)
        bs = font.render(self.status, True, COLORS["white"])
        bx = int(self.x) - bs.get_width() // 2
        pygame.draw.rect(surf, COLORS["blue"], (bx - 2, int(self.y) - ROBOT_R - 20, bs.get_width() + 4, 15), border_radius=3)
        surf.blit(bs, (bx, int(self.y) - ROBOT_R - 19))

    def reset(self):
        BaseRobot.reset(self)
        self._release_reservation()
        self.carry, self._just_delivered, self.done_count = None, False, 0
        self.ctrl.reset()
        self.target = (self.x, self.y)


class _CompositeCoordinator:
    """Steps several independent per-resource Coordinators as a single arbiter.

    this task's shelf and dock are two UNRELATED shared resources, each synthesized
    as its own small Arbiter contract (see contrauto/tasks/warehouse.yaml) so each
    one's GR(1) game stays small enough for SLUGS to solve quickly. SimWorld.arbiter()
    and fsm_step() expect a single object with a `.step(reqs, world)` method, so this
    class fans one call out to every underlying Coordinator and merges their grant
    outputs — the two FSMs never share state, so order doesn't matter.
    """

    def __init__(self, coords):
        self.coords = coords

    def step(self, reqs, world):
        merged = {}
        for c in self.coords:
            merged.update(c.step(reqs, world))
        return merged

    def reset(self):
        for c in self.coords:
            if c.ctrl is not None:
                c.ctrl.reset()


class WarehouseGame(BaseGame, SimWorld):
    """Multi-robot SimWorld with persistent arbiter Coordinators mediating the shared dock/shelf."""
    MISSION = MISSION
    SUBTITLE = "SLUGS GR(1) + verified arbiter (true FSM)"
    DELIVER_HOLD_FRAMES = 30   # ~0.5s at 60fps: how long a delivered item stays visible at the dock
    # Match corridor.py/charger.py's window (no side panel there either) instead of BaseGame's
    # default 1280x800-with-a-380px-panel: the stats now render as a single top HUD line (see
    # draw()), so the panel width has nothing left to hold.
    #
    # CONTENT_W/H is the arena's NATURAL, untransformed size -- shelf/dock/robots are still
    # drawn at exactly the same SHELF_X0/DOCK_Y/SPAWN_Y physics coordinates that empirically
    # work (see the DOCK_Y note above: moving those broke navigation). Centering "shelf/dock
    # in the middle of the window" and "more gap under the HUD line" are achieved WITHOUT
    # touching any of those coordinates, by rendering the arena onto its own CONTENT_W x
    # CONTENT_H surface first and blitting THAT surface onto the real (larger) window at a
    # fixed pixel offset (see draw()) -- a camera pan, not a world edit. CBF/obstacle geometry
    # (cbf.obstacles.warehouse_shelf_plank_rects) reads the SAME untransformed constants, so
    # it is automatically still exactly aligned with what's drawn; nothing to keep in sync.
    CONTENT_W, CONTENT_H = 740, 685   # 685 = new DOCK_Y(620) + DOCK_H(45) + 20px margin
    CONTENT_DX, CONTENT_DY = 130, 50   # (1000-740)//2, and the top-gap shift
    W, H, ARENA_W = 1000, CONTENT_H + CONTENT_DY + 10, CONTENT_W

    def __init__(self, robot_defs, arbiter_paths, max_active_bins=None, headless=False,
                spawn_frames=None):
        # Batch FSM execution has no pygame event/render loop. Select the dummy
        # backend before BaseGame initializes pygame so it cannot leave a frozen
        # visible window behind; live validation passes `headless=False`.
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        super().__init__()
        # Bump BaseGame's default panel-scale fonts (18/22/28) now that all this game's text
        # (robot status tags, the Dock label, the top HUD line) renders directly on the arena
        # at full window resolution instead of in a cramped 380px side panel.
        self.fS = pygame.font.Font(None, 22)
        self.fM = pygame.font.Font(None, 34)   # top HUD line
        self.fL = pygame.font.Font(None, 44)   # "Dock" label -- the one other font, bigger still
        self.spawn_frames = spawn_frames if spawn_frames is not None else SPAWN_FRAMES
        for rid, name, x, y, col, fsm_path in robot_defs:
            self.robots.append(WarehouseRobot(rid, name, x, y, col, fsm_path))
        self.controllers = {r.name: r.ctrl for r in self.robots}
        self.selected = self.robots[0]
        self.max_active_bins = max_active_bins if max_active_bins is not None else MAX_ACTIVE_BINS
        # One Coordinator PER arbiter FSM (shelf, dock, ... — arbiter_paths is a list, kept
        # generic rather than hardcoding "two" in case a task ever splits further). Detect
        # request/grant vars by ROLE, not by a raw substring: a synthesized arbiter may name a
        # request `r1_req_shelf` (abbreviated "req"), which the old `"request" in v` test missed, so
        # the arbiter got no inputs and never granted (robots wait forever, delivered 0). role_of maps
        # req/request and grant/granted alike, matching how the robot FSMs are bound.
        self._req_vars, self._grant_vars, coords = [], [], []
        for path in arbiter_paths:
            ad = load_slugs(path)
            req_vars = [v for v in ad["variables"]
                       if role_of(v) in ("shelf_request", "dock_request", "request")]
            grant_vars = [v for v in ad["variables"]
                         if role_of(v) in ("shelf_grant", "dock_grant", "grant")]
            if grant_vars and not req_vars:
                print(f"[warehouse] warning: arbiter {path} has grant outputs {grant_vars} but NO "
                      f"request inputs were recognized in {ad['variables']}; it can never grant. Check "
                      f"the arbiter's request variable naming against role_of.")
            self._req_vars += req_vars
            self._grant_vars += grant_vars
            coords.append(Coordinator(ad, req_vars, grant_vars, apply=self._grant_apply))
        # one persistent composite Coordinator (each underlying FSM keeps state across steps)
        self._coord = _CompositeCoordinator(coords)
        self.last_grants = {v: 0 for v in self._grant_vars}
        self.world, self.delivered, self.violations = [], 0, 0
        self._frame = 0
        self._n_spawned = 0

    # ── SimWorld interface ───────────────────────────────────────────────────
    def agents(self):
        return self.robots

    def arbiter(self):
        return self._coord                          # persistent generic Coordinator

    def _grant_apply(self, out, world):
        """Record one arbiter's grants and count any double-grant (mutex breach).

        Called once per underlying Coordinator (see _CompositeCoordinator), so `out`
        holds only THAT resource's grant variables — merge, never replace, or the
        second call would wipe out the first resource's grants for this step."""
        self.last_grants.update(out)
        for robot in self.robots:
            robot.grants.update({name: out[name] for name in robot.GRANTS if name in out})
        shelf = [v for v in self._grant_vars if role_of(v) == "shelf_grant"]
        dock = [v for v in self._grant_vars if role_of(v) == "dock_grant"]
        # Fall back to all grant vars when roles are undifferentiated ("grant").
        if not shelf and not dock:
            shelf = list(self._grant_vars)
        if (sum(1 for v in shelf if out.get(v)) > 1 or
                sum(1 for v in dock if out.get(v)) > 1):
            self.violations += 1
        return out

    def step(self, dt):                              # spawn + cleanup (end of fsm_step)
        self._frame += 1
        # Keep at least one task available, but never flood shelves faster than robots clear them.
        if ((self._frame % self.spawn_frames == 0 or not self.world)
                and len(self.world) < self.max_active_bins):
            self.spawn()
        # A dropped-off item used to vanish the very next frame -- faster than any
        # periodic GIF sample could ever catch, so a delivery was invisible even though
        # it happened. Hold it, visibly, at the dock for DELIVER_HOLD_FRAMES before
        # actually removing it (excluded from _avail_bins the whole time via `delivered`,
        # so it can't be mistaken for a fetchable shelf item while it lingers).
        still, delivered_now = [], 0
        for o in self.world:
            if getattr(o, "delivered", False):
                o._deliver_wait = getattr(o, "_deliver_wait", 0) + 1
                if o._deliver_wait >= self.DELIVER_HOLD_FRAMES:
                    delivered_now += 1
                    continue
            still.append(o)
        self.world = still
        self.delivered += delivered_now

    def metrics(self):
        return {"delivered": self.delivered, "violations": self.violations,
                "per_robot": {r.name: r.done_count for r in self.robots}, "active": len(self.world)}

    # ── live-render driver ────────────────────────────────────────────────────
    def update(self, dt):
        if self.paused: return
        fsm_step(self, self.controllers, dt, self.arbiter())

    def spawn(self):
        spots = [(sx + SHELF_COL_W // 2, sy + SHELF_ROW_H // 2)
                 for r in range(SHELF_ROWS) for c in range(SHELF_COLS) for sx, sy in [_shelf_xy(r, c)]]
        occupied = {s for b in self.world for s in spots if math.hypot(b.x - s[0], b.y - s[1]) < OBJ_SZ + 2}
        free = [s for s in spots if s not in occupied]
        if not free:
            return
        # The very first two items spawn in the LEFT or MIDDLE shelf column (any row) instead
        # of a fully random slot, so a GIF's opening frames read as deliberate; every later
        # spawn (once the fleet is already busy) stays fully random.
        pick = None
        if self._n_spawned < 2:
            col_w = SHELF_COL_W + SHELF_COL_GAP
            left_x = SHELF_X0 + SHELF_COL_W // 2
            mid_x = SHELF_X0 + col_w + SHELF_COL_W // 2
            candidates = [s for s in free if abs(s[0] - left_x) < 1 or abs(s[0] - mid_x) < 1]
            if candidates:
                pick = random.choice(candidates)
        if pick is None:
            pick = random.choice(free)
        self._n_spawned += 1
        self.world.append(WorldObj(*pick))

    def _draw_arena(self):
        self._draw_grid()
        for row in range(SHELF_ROWS):
            for col in range(SHELF_COLS):
                sx, sy = _shelf_xy(row, col)
                sr = pygame.Rect(sx, sy, SHELF_COL_W, SHELF_ROW_H)
                pygame.draw.rect(self.screen, COLORS["shelf"], sr)
                pygame.draw.rect(self.screen, COLORS["border"], sr, 2)
        dock_rect = pygame.Rect(DOCK_X - DOCK_W // 2, DOCK_Y, DOCK_W, DOCK_H)
        pygame.draw.rect(self.screen, COLORS["dock"], dock_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLORS["border"], dock_rect, 2, border_radius=6)
        label = self.fL.render("Dock", True, COLORS["white"])
        self.screen.blit(label, label.get_rect(center=dock_rect.center))

    def _draw_objects(self):
        for o in self.world: o.draw(self.screen)

    def _draw_panel(self):
        pass

    def draw(self):
        # Render the arena at its natural, physics-aligned size onto its own surface, then
        # blit that whole surface onto the real (larger, centered) window at a fixed offset
        # -- see the CONTENT_W/H note on the class. self.screen is swapped only for the
        # duration of the existing _draw_arena/_draw_objects/robot.draw calls, which already
        # just draw onto "self.screen" and need no changes themselves.
        content = pygame.Surface((self.CONTENT_W, self.CONTENT_H))
        content.fill(COLORS["bg"])
        real_screen = self.screen
        self.screen = content
        self._draw_arena()
        self._draw_objects()
        for r in self.robots:
            r.draw(self.screen, self.fS)
        self.screen = real_screen
        self.screen.fill(COLORS["bg"])
        self.screen.blit(content, (self.CONTENT_DX, self.CONTENT_DY))
        # HUD stays fixed at the window's own top-left (not shifted with the arena) so
        # centering/shifting the arena below it only ever widens the gap between them.
        el = int(time.time() - self.t0)
        hud = (f"Items {len(self.world)} active   Delivered {self.delivered}   "
               f"Mutex viol {self.violations}   Time {el // 60:02d}:{el % 60:02d}   "
               f"State {'PAUSED' if self.paused else 'RUNNING'}")
        stats = self.fM.render(hud, True, COLORS["white"])
        self.screen.blit(stats, (16, 12))
        pygame.display.flip()

    def _reset(self):
        super()._reset()
        self._coord.reset()
        self.world.clear(); self.delivered = 0; self.violations = 0; self._frame = 0
        self._n_spawned = 0
        self.last_grants = {v: 0 for v in self._grant_vars}
        for robot in self.robots:
            robot.grants = {}


def headless(robot1, robot2, arbiter, steps=8000, dt=1 / 60.0,
             assumption_monitor=None, visualize=False):
    """Run the two-robot + dock arbiter FSMs; optionally render the validation live.

    `assumption_monitor` (optional): channel-5 AssumptionMonitorSet fed the observed environment
    each step; a run-time assumption violation surfaces as `assumption_feedback` in the metrics."""
    xs2 = _spawn_row_x(2); defs = [(1, 'Robot1', xs2[0], SPAWN_Y, 'red', robot1), (2, 'Robot2', xs2[1], SPAWN_Y, 'blue', robot2)]
    g = WarehouseGame(defs, [arbiter], headless=not visualize)
    return _run_validation(g, steps, dt, assumption_monitor, visualize)


def _spawn_row_x(n):
    """X coordinates for n robots lined up in a row, centered directly above the shared
    dock -- previously they spawned spread across the whole arena width, which read as
    arbitrary; starting queued above the (now dock-aligned) dock/shelf column makes the
    initial frame legible as "a fleet about to work this dock," not a random scatter."""
    return [int(DOCK_X + (i - (n - 1) / 2) * QUEUE_SPACING) for i in range(n)]


def _spawn_frames_for(n):
    """Scale the spawn interval down as the fleet grows, so per-robot item supply stays
    roughly constant instead of thinning out. SPAWN_FRAMES (300 = ~5s) was calibrated for
    the original 2-robot fleet's ~0.4 goods/s delivery rate; a bigger fleet clears items
    proportionally faster, so it needs proportionally faster spawning too. Floored at 60
    frames (~1s) so spawning never outruns the shelf's own SHELF_ROWS*SHELF_COLS capacity."""
    return max(60, round(SPAWN_FRAMES * 2 / max(n, 1)))


def headless_n(robot_paths, arbiter, steps=8000, dt=1 / 60.0,
               assumption_monitor=None, visualize=False, spawn_frames=None):
    """Run N robots + the dock arbiter FSM; optionally render the validation live.

    `robot_paths`: ordered list of FSM json paths [robot1.json, robot2.json, ...].
    Robots are spread evenly across the arena width (world_bounds x-range).
    max_active_bins is scaled to 2 * len(robot_paths) so the shelf backlog
    stays proportional to fleet size; spawn_frames likewise (see _spawn_frames_for)."""
    n = len(robot_paths)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'gold', 'teal']
    xs = _spawn_row_x(n)
    defs = [(i + 1, f'Robot{i + 1}', xs[i], SPAWN_Y, colors[i % len(colors)], robot_paths[i])
            for i in range(n)]
    g = WarehouseGame(defs, [arbiter], max_active_bins=max(4, 2 * n),
                       headless=not visualize,
                       spawn_frames=spawn_frames if spawn_frames is not None else _spawn_frames_for(n))
    return _run_validation(g, steps, dt, assumption_monitor, visualize)


def _run_validation(game, steps, dt, assumption_monitor, visualize):
    """Execute the finite harness, pumping pygame only for an opted-in live view."""
    def _render(_step, _world):
        if not visualize:
            return None
        game.clock.tick(60)
        if not game.handle():
            return False
        game.draw()
        return True

    try:
        return run_fsm(game, game.controllers, steps, dt, on_step=_render,
                       assumption_monitor=assumption_monitor)
    finally:
        if visualize:
            pygame.quit()


def render_gif(robot1, robot2, arbiter, gif_path, steps=3000, every=18,
               scale=0.6, dt=1 / 60.0):
    """Drive the SAME loop as headless() but capture an animated GIF (like charger.py).

    Renders offscreen: the SDL dummy video driver is selected before the pygame window
    is created (exactly the trick charger.py uses at import time), so no window opens.
    Frames are grabbed from the board surface every `every` steps via PIL and saved as
    an animated GIF. Returns the run's metrics dict.

    NOTE: must be called before any real pygame display exists in this process (the
    dummy driver is chosen at display-init time); the --gif CLI path satisfies this.
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # offscreen, as in charger.py
    from PIL import Image
    _tobytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    xs2 = _spawn_row_x(2); defs = [(1, 'Robot1', xs2[0], SPAWN_Y, 'red', robot1), (2, 'Robot2', xs2[1], SPAWN_Y, 'blue', robot2)]
    g = WarehouseGame(defs, [arbiter])
    frames = []

    def _grab(step, world):
        if step % every == 0:
            g.draw()                                    # BaseGame.draw: arena+robots+panel
            img = Image.frombytes("RGB", g.screen.get_size(), _tobytes(g.screen, "RGB"))
            frames.append(img.resize((int(img.width * scale), int(img.height * scale))))

    metrics = run_fsm(g, g.controllers, steps, dt, on_step=_grab)
    if frames:
        d = os.path.dirname(os.path.abspath(gif_path))
        if d:
            os.makedirs(d, exist_ok=True)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=int(every * dt * 1000), loop=0, optimize=True)
    return metrics


if __name__ == "__main__":
    W = "outputs/warehouse_n2/workdir/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot1', default=W + 'robot1.json')
    parser.add_argument('--robot2', default=W + 'robot2.json')
    parser.add_argument('--arbiter', default=W + 'arbiter.json')
    parser.add_argument('--headless', type=int, default=0)
    parser.add_argument('--gif', default=None,
                        help="render an animated GIF to this path (offscreen; "
                             "steps = --headless if given, else 3000)")
    args = parser.parse_args()
    if args.gif:
        m = render_gif(args.robot1, args.robot2, args.arbiter, args.gif,
                       steps=args.headless or 3000)
        print("warehouse:", m, "GIF:", args.gif)
    elif args.headless:
        print(headless(args.robot1, args.robot2, args.arbiter,
                       steps=args.headless))
    else:
        xs2 = _spawn_row_x(2); defs = [(1, 'Robot1', xs2[0], SPAWN_Y, 'red', args.robot1), (2, 'Robot2', xs2[1], SPAWN_Y, 'blue', args.robot2)]
        WarehouseGame(defs, [args.arbiter]).run()

VOCAB_NOTES = (
    "at_shelf/at_dock: position sensors; persist while the robot stays (movement is DURATIVE "
    "— use justice G F(at_shelf | !go_to_shelf), never next-step arrival). has_item: rises "
    "after pickup, falls after dropoff. dock_grant: raised by the Arbiter in response to "
    "dock_req; assume fairness G F(dock_grant | !dock_req). The Arbiter guarantees only the "
    "request-conditioned persistence G((dock_grant & dock_req & X(dock_req)) -> X(dock_grant)); "
    "any other grant persistence a robot relies on belongs in System assumptions (folded). "
    "task: env task stream justice G F(task)."
)

# Coordination signals the Arbiter edge carries (bound by role_of, not by PREDICATES/ACTIONS):
# per-robot names take the r<i>_ prefix like every other grounding.
SIGNALS = {"requests": ["dock_req"], "grants": ["dock_grant"]}

# Durative-action physics, used by contrauto/translation_check.py to mechanically complete the
# translator's persistence / fairness / frame formulas. action -> (sensed effect literal it waits
# for, grant it must keep holding while it runs or None).
ACTION_EFFECTS = {
    "go_to_shelf": ("at_shelf", None),
    "go_to_dock": ("at_dock", None),
    "pickup": ("has_item", None),
    "dropoff": ("!has_item", "dock_grant"),
}
# sensed literal -> the action whose absence keeps it unchanged (frame axiom).
FRAME = {"at_shelf": "go_to_dock", "at_dock": "go_to_shelf",
         "has_item": "dropoff", "!has_item": "pickup"}
# Shared (un-prefixed) sensors.
GLOBAL_SENSORS = ["task"]
