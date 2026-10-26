"""TWO-LAYER warehouse — the SEAM between the two layers we'd kept separate.

  • TASK layer  : each robot's DISCRETE logic is the platform-synthesized worker FSM
                  (refinement + well-sep gate produced outputs/warehouse_hl/workdir/robot.json).
                  FsmController drives go_to_shelf -> pickup -> go_to_dock -> dropoff.
  • PHYSICAL layer: WHICH equivalent good each robot fetches, and in what order, is decided by
                  assign_decomposer.solve (geometry t_min vs deadline; change-target / reorder /
                  relax). Execution = OBB nav + collision; the deadline is VERIFIED in sim.

So: task layer = "is the discrete behaviour correct"; physical layer = "is it feasible in time".
Both run in ONE simulation here.

Run:  python simulator/two_layer_warehouse.py            # headless report + GIF
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import sys, math, random
from dataclasses import dataclass, replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from PIL import Image
from contrauto.sim_iface import (SimAgent, SimWorld, Nav, Stationary, FsmController,
                                 fsm_step, load_fsm)
from contrauto.grounding import Grounding, bind_groundings
from contrauto.geometry import WaypointNav
from contrauto.assign_decomposer import solve, path_time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_FSM = ROOT + "/outputs/warehouse_hl/workdir/robot.json"   # TASK-layer synthesized worker
_tobytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
RR, OBJ, SPEED = 13, 13, 155


@dataclass
class Config:
    name: str
    n_robots: int = 4
    n_demands: int = 6
    equiv: int = 2            # equivalent goods per demand (the change-target choices)
    rows: int = 4
    cols: int = 4
    W: int = 1120
    H: int = 760
    seed: int = 7
    deadline: float = 35.0
    dock_hw: float = 84.0       # half-size of the dock block; shrink it to create unload
    dock_hh: float = 58.0       # contention (a bottleneck the single-robot judica can't see)


# ── random shelves + random dock block (same physical world as scalable_warehouse) ──
class Layout:
    """Random oriented shelves + a random dock block, with one pickable cell per shelf."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.bounds = (0.0, float(cfg.W), 0.0, float(cfg.H))
        THIN, n, m, top = 9, cfg.rows * cfg.cols, 60, 80
        self.dock_hw, self.dock_hh = cfg.dock_hw, cfg.dock_hh
        drad = math.hypot(self.dock_hw, self.dock_hh)
        dx = random.uniform(m + self.dock_hw, cfg.W - m - self.dock_hw)
        dy = random.uniform(top + self.dock_hh, cfg.H - m - self.dock_hh)
        self.dock = (dx, dy)
        self.rects, self.cells = [], []
        tries = 0
        while len(self.rects) < n and tries < 9000:
            tries += 1
            ang = random.uniform(0, math.pi)
            half_long = random.uniform(34, 92)
            rad = math.hypot(half_long, THIN)
            x = random.uniform(m + rad, cfg.W - m - rad)
            y = random.uniform(top + rad, cfg.H - m - rad)
            if math.hypot(x - dx, y - dy) < rad + drad:
                continue
            self.rects.append((x, y, half_long, THIN, ang))
            t = random.uniform(-(half_long - 12), half_long - 12)
            self.cells.append((x + t * math.cos(ang), y + t * math.sin(ang)))


def _near(a, xy, r):
    return math.hypot(a.x - xy[0], a.y - xy[1]) < r


def _obb_aabb(o):
    """Axis-aligned envelope of an oriented shelf — the physical assign layer estimates t_min
    on AABBs (geometry.path_time is AABB-only); execution still avoids the true OBB."""
    cx, cy, hw, hh, ang = o
    c, s = abs(math.cos(ang)), abs(math.sin(ang))
    return (cx, cy, hw * c + hh * s, hw * s + hh * c)


class Cargo:
    """A good instance for a demand; `chosen` marks the equivalent the assign layer fetches."""
    def __init__(self, name, demand, cell, dcolor):
        self.name, self.demand, self.dcolor = name, demand, dcolor
        self.x, self.y = cell
        self.carrier = None
        self.delivered = False
        self.chosen = False                # picked by the assign layer (vs unused equivalent)


# ── predicates/actions: the robot fetches the goods in ITS assigned queue (physical layer),
#    but WHEN to pick/drop/move is the task FSM's call (FsmController). ──
def _current(a):
    return a.queue[a.qi] if a.qi < len(a.queue) else None


def _at_shelf(a, w):
    if a.carry is not None:
        return _near(a, a._pickup_xy, RR + OBJ + 10)          # persist after pickup (EXP-011)
    b = _current(a)
    return b is not None and _near(a, (b.x, b.y), RR + OBJ + 8)


def _shelf_target(a, w):
    if a.carry is not None:
        return a.pose()
    b = _current(a)
    return (b.x, b.y) if b is not None else a.pose()


def _do_pick(a, w):
    b = _current(a)
    if b is not None and a.carry is None and _near(a, (b.x, b.y), RR + OBJ + 8):
        a._pickup_xy = (a.x, a.y)
        b.carrier, a.carry = a.id, b


def _at_dock(a, w):
    lay = w.lay
    return (abs(a.x - lay.dock[0]) <= lay.dock_hw + RR and
            abs(a.y - lay.dock[1]) <= lay.dock_hh + RR)


def _do_drop(a, w):
    if a.carry is not None and _at_dock(a, w):
        a.carry.delivered, a.carry.carrier, a.carry, a._jd = True, None, None, True
        a.done_count += 1
        a.qi += 1
        if a.qi >= len(a.queue) and a.finish_t is None:
            a.finish_t = w.t


# The ONE place that reads raw two-layer-dock state, keyed by capability token. (_at_shelf/_at_dock
# stay as functions because the actions still call them; the groundings below mirror their logic
# reading only through the gate.)
PROVIDERS = {
    "agent.pos":             lambda a, w: (a.x, a.y),
    "agent.carry":           lambda a, w: a.carry,
    "agent.pickup_xy":       lambda a, w: a._pickup_xy,
    "agent.current_box_pos": lambda a, w: (lambda b: (b.x, b.y) if b is not None else None)(_current(a)),
    "agent.qi":              lambda a, w: a.qi,
    "agent.queue_len":       lambda a, w: len(a.queue),
    "agent.jd":              lambda a, w: a._jd,
    "geom.dock_box":         lambda a, w: (w.lay.dock[0], w.lay.dock[1], w.lay.dock_hw, w.lay.dock_hh),
}


class AtShelf(Grounding):
    def evaluate(self, cap):
        if cap.get("agent.carry") is not None:
            return cap.dist(cap.get("agent.pos"), cap.get("agent.pickup_xy")) < RR + OBJ + 10   # persist after pickup
        bpos = cap.get("agent.current_box_pos")
        return bpos is not None and cap.dist(cap.get("agent.pos"), bpos) < RR + OBJ + 8


class HasItem(Grounding):
    def evaluate(self, cap): return cap.get("agent.carry") is not None


class AtDock(Grounding):
    def evaluate(self, cap):
        x, y = cap.get("agent.pos")
        dx, dy, hw, hh = cap.get("geom.dock_box")
        return abs(x - dx) <= hw + RR and abs(y - dy) <= hh + RR


class Delivered(Grounding):
    def evaluate(self, cap): return cap.get("agent.jd")


class HasTask(Grounding):
    def evaluate(self, cap): return (cap.get("agent.qi") < cap.get("agent.queue_len")) or (cap.get("agent.carry") is not None)


PRED = bind_groundings({
    "at_shelf":  AtShelf(),
    "has_item":  HasItem(),
    "at_dock":   AtDock(),
    "delivered": Delivered(),
    "task":      HasTask(),
}, PROVIDERS)
ACT = {
    "go_to_shelf": Nav(_shelf_target, done_when=lambda a, w: _at_shelf(a, w)),
    "pickup":      Stationary(_do_pick),
    "go_to_dock":  Nav(lambda a, w: w.lay.dock, done_when=_at_dock),
    "dropoff":     Stationary(_do_drop),
    "done":        Stationary(None),
}

ROBOT_COLORS = [(225, 70, 70), (70, 130, 225), (80, 200, 100), (225, 165, 50), (165, 100, 220),
                (70, 200, 205), (225, 130, 190), (150, 150, 150), (120, 200, 60), (230, 100, 60)]
DEMAND_COLORS = [(240, 215, 70), (250, 150, 60), (120, 220, 250), (200, 120, 240),
                 (120, 240, 160), (250, 110, 150), (180, 220, 90), (250, 200, 120)]


class Robot(SimAgent):
    """Worker fetching its assigned queue: task FSM decides when, WaypointNav decides the path."""
    def __init__(self, rid, color, fsm, lay):
        self.id, self.name, self.color = rid, f"r{rid}", color
        ang0 = rid * 1.9
        self.x = lay.dock[0] + lay.dock_hw * 0.7 * math.cos(ang0)
        self.y = lay.dock[1] + lay.dock_hh * 0.7 * math.sin(ang0)
        self.carry = None
        self._pickup_xy = (self.x, self.y)
        self._jd = False
        self.done_count = 0
        self.queue, self.qi, self.finish_t = [], 0, None
        self.target = (self.x, self.y)
        self.action = {}
        self._nav = WaypointNav(obb=True)
        self.bind_fsm(fsm, PRED, ACT)

    def pose(self):
        return (self.x, self.y)

    def act(self, action, world):
        self._jd = False
        self.action = action
        SimAgent.act(self, action, world)

    def tick(self, dt, world):
        lay = world.lay
        wp = self._nav.target((self.x, self.y), self.target, lay.rects, lay.bounds, RR)
        dx, dy = wp[0] - self.x, wp[1] - self.y
        d = math.hypot(dx, dy) or 1.0
        vx, vy = SPEED * dx / d, SPEED * dy / d
        for o in world.robots:
            if o is self:
                continue
            ox, oy = self.x - o.x, self.y - o.y
            od = math.hypot(ox, oy) or 1.0
            if od < 2 * RR + 10:
                vx += 140 * ox / od
                vy += 140 * oy / od
        self.x = min(lay.cfg.W - RR, max(RR, self.x + vx * dt))
        self.y = min(lay.cfg.H - RR, max(RR, self.y + vy * dt))
        if self.carry is not None:
            self.carry.x, self.carry.y = self.x, self.y - RR - OBJ // 2


class TwoLayerWarehouse(SimWorld):
    """SimWorld coupling the task layer (worker FSM) to the physical layer (deadline-aware assign)."""
    MISSION = "two_layer_warehouse"

    def __init__(self, cfg, verbose=True):
        random.seed(cfg.seed)
        self.cfg = cfg
        self.lay = Layout(cfg)
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.W, cfg.H))
        self.font = pygame.font.SysFont(None, 24)
        self.fbig = pygame.font.SysFont(None, 30)
        fsm = load_fsm(WORKER_FSM)
        self.robots = [Robot(i + 1, ROBOT_COLORS[i % len(ROBOT_COLORS)], fsm, self.lay)
                       for i in range(cfg.n_robots)]
        self.controllers = {r.name: r.ctrl for r in self.robots}

        # ── build demands: each = `equiv` identical goods on distinct random shelf cells ──
        cells = list(self.lay.cells)
        random.shuffle(cells)
        self.boxes, self.demands = {}, {}
        k = 0
        for d in range(cfg.n_demands):
            insts = []
            for mi in range(cfg.equiv):
                if k >= len(cells):
                    break
                nm = f"D{d}#{mi}"
                self.boxes[nm] = Cargo(nm, f"D{d}", cells[k], DEMAND_COLORS[d % len(DEMAND_COLORS)])
                k += 1
                insts.append(nm)
            if insts:
                self.demands[f"D{d}"] = insts

        # ── PHYSICAL layer: deadline-aware assignment (change-target / reorder / relax) ──
        robot_pos = {r.name: (r.x, r.y) for r in self.robots}
        groups = {d: [(self.boxes[n].x, self.boxes[n].y) for n in insts]
                  for d, insts in self.demands.items()}
        aabb = [_obb_aabb(o) for o in self.lay.rects]
        self.report = solve(robot_pos, groups, aabb, self.lay.bounds, cfg.deadline, dock=self.lay.dock)
        self._apply_assignment()

        self.t = 0.0
        self.delivered = 0
        if verbose:
            self._print_report()

    def _apply_assignment(self):
        idx = {r.name: r for r in self.robots}
        for r in self.robots:
            r.queue = []
        for b in self.boxes.values():
            b.chosen = False
        if self.report["assignment"]:
            for rname, seq in self.report["assignment"].items():
                if rname in idx:
                    idx[rname].queue = [self.boxes[n] for n in seq]
                    for n in seq:
                        self.boxes[n].chosen = True

    def _print_report(self):
        rep = self.report
        tag = (f"  (relaxed {rep['base_deadline']:.0f}->{rep['deadline']:.0f}s, ×{rep['relax_x']})"
               if rep["status"] == "REPAIRED" else "")
        print(f"\n=== PHYSICAL layer (assign + deadline): {rep['status']}{tag} ===")
        if not rep["assignment"]:
            return
        for d, insts in self.demands.items():
            chosen = [n for n in insts if self.boxes[n].chosen]
            tmins = {n: min(path_time((r.x, r.y), (self.boxes[n].x, self.boxes[n].y),
                                      [_obb_aabb(o) for o in self.lay.rects], self.lay.bounds)
                            for r in self.robots) for n in insts}
            opts = ", ".join(f"{n}{'*' if self.boxes[n].chosen else ''}(best {tmins[n]:.1f}s)" for n in insts)
            print(f"  {d}: chose {chosen[0] if chosen else '—'}   [{opts}]")
        for r in self.robots:
            print(f"  {r.name}: {[b.name for b in r.queue]}")

    def agents(self):
        return self.robots

    def arbiter(self):
        return None

    def step(self, dt):
        self.t += dt
        self.delivered = sum(r.done_count for r in self.robots)

    def update(self, dt):
        fsm_step(self, self.controllers, dt, None)

    def all_done(self):
        return all(r.finish_t is not None for r in self.robots)

    def metrics(self):
        return {"delivered": self.delivered, "robots": len(self.robots)}

    def draw(self):
        s, lay, cfg = self.screen, self.lay, self.cfg
        s.fill((18, 20, 28))
        for gx in range(0, cfg.W, 40):
            pygame.draw.line(s, (28, 31, 42), (gx, 0), (gx, cfg.H))
        for gy in range(0, cfg.H, 40):
            pygame.draw.line(s, (28, 31, 42), (0, gy), (cfg.W, gy))
        for (cx, cy, hw, hh, ang) in lay.rects:
            c, sn = math.cos(ang), math.sin(ang)
            corners = [(cx + dx * hw * c - dy * hh * sn, cy + dx * hw * sn + dy * hh * c)
                       for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            pygame.draw.polygon(s, (100, 65, 30), corners)
            pygame.draw.polygon(s, (75, 80, 100), corners, 2)
        dx0, dy0 = int(lay.dock[0] - lay.dock_hw), int(lay.dock[1] - lay.dock_hh)
        pygame.draw.rect(s, (45, 105, 155), (dx0, dy0, lay.dock_hw * 2, lay.dock_hh * 2))
        pygame.draw.rect(s, (120, 180, 220), (dx0, dy0, lay.dock_hw * 2, lay.dock_hh * 2), 2)
        s.blit(self.font.render("DOCK", True, (255, 255, 255)), (int(lay.dock[0]) - 22, int(lay.dock[1]) - 8))
        for b in self.boxes.values():
            if b.delivered or b.carrier is not None:
                continue
            col = b.dcolor if b.chosen else (70, 72, 80)         # unused equivalent goods = dim grey
            pygame.draw.rect(s, col, (int(b.x) - OBJ // 2, int(b.y) - OBJ // 2, OBJ, OBJ))
            pygame.draw.rect(s, (75, 80, 100), (int(b.x) - OBJ // 2, int(b.y) - OBJ // 2, OBJ, OBJ), 1)
        for r in self.robots:
            ix, iy = int(r.x), int(r.y)
            pygame.draw.circle(s, r.color, (ix, iy), RR)
            pygame.draw.circle(s, (255, 255, 255), (ix, iy), RR, 2)
            if r.carry is not None:
                pygame.draw.rect(s, r.carry.dcolor, (ix - OBJ // 2, iy - RR - OBJ - 2, OBJ, OBJ))
        rep = self.report
        hud = (f"TASK: FSM-driven   PHYSICAL: {rep['status']}   "
               f"deadline {rep['deadline']:.0f}s   t={self.t:4.1f}s   delivered {self.delivered}")
        surf = self.fbig.render(hud, True, (235, 235, 245))
        pygame.draw.rect(s, (12, 14, 20), (10, 8, surf.get_width() + 16, 32))
        s.blit(surf, (18, 12))
        pygame.display.flip()


def _execute(w, max_steps=5000, dt=1 / 60.0, gif=None, every=5, scale=0.55):
    """Run the FSM-driven simulation to completion; optionally capture GIF frames."""
    frames = []
    for i in range(max_steps):
        w.update(dt)
        w.step(dt)
        if gif and i % every == 0:
            w.draw()
            img = Image.frombytes("RGB", w.screen.get_size(), _tobytes(w.screen, "RGB"))
            frames.append(img.resize((int(img.width * scale), int(img.height * scale))))
        if w.all_done():
            break
    return frames


def _oracle(w):
    """Execution oracle: per-robot (finish_t, within?) vs the (relaxed) deadline."""
    dl = w.report["deadline"]
    rows = [(r.name, r.finish_t, (r.finish_t is not None and r.finish_t <= dl + 1e-6)) for r in w.robots]
    worst = max((r.finish_t for r in w.robots if r.finish_t is not None), default=None)
    return dl, rows, worst


def run(cfg, max_steps=5000, dt=1 / 60.0, gif=None, every=5, scale=0.55):
    w = TwoLayerWarehouse(cfg)
    if not w.report["assignment"]:
        print("PHYSICAL layer UNSATISFIABLE — no feasible assignment within deadline.")
        return w
    frames = _execute(w, max_steps, dt, gif, every, scale)
    dl, rows, _ = _oracle(w)
    total = sum(len(r.queue) for r in w.robots)
    print(f"\n=== EXECUTION verify (FSM-driven, deadline {dl:.1f}s) ===")
    for name, ft, within in rows:
        print(f"  {name}: " + (f"finished {ft:.1f}s ({'≤' if within else '>'} {dl:.1f}s)" if ft is not None else "DNF"))
    print(f"{'ALL within deadline ✓' if all(w_ for _, _, w_ in rows) else 'check above ✗'}  "
          f"-> delivered {w.delivered}/{total}")
    if gif and frames:
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=int(every * dt * 1000), loop=0, optimize=True)
        print(f"GIF: {gif}  ({len(frames)} frames, {os.path.getsize(gif)//1024} KB)")
    return w


def run_closed_loop(cfg, max_rounds=3, max_steps=6000, dt=1 / 60.0):
    """FULL two-layer loop with the PHYSICAL execution-oracle feedback wired in.

    Each round: assign (CBF-provable deadline judica) -> FSM-driven execution -> the
    execution oracle checks who missed the deadline (catches multi-robot congestion the
    single-robot judica can't see). On a miss, feedback is routed BOTH ways, as discussed:
      • PHYSICAL: relax the shared time budget to what execution actually needed.
      • TASK:     re-solve the assignment under that budget -> reassign robots / change to a
                  different equivalent target (rebalance away from the congested robot).
    Repeat until the oracle is clean or the round budget is exhausted.
    """
    cur, last = cfg, None
    for rnd in range(max_rounds):
        w = TwoLayerWarehouse(cur, verbose=(rnd == 0))
        last = w
        if not w.report["assignment"]:
            print(f"[round {rnd}] PHYSICAL UNSATISFIABLE — stop.")
            return w
        _execute(w, max_steps, dt)
        dl, rows, worst = _oracle(w)
        viol = [(n, ft) for n, ft, ok in rows if not ok]
        print(f"\n[round {rnd}] deadline {dl:.1f}s | exec worst "
              f"{worst:.1f}s | delivered {w.delivered}/{sum(len(r.queue) for r in w.robots)} | "
              f"violators {viol if viol else 'none'}")
        if not viol:
            print(f"[round {rnd}] ✓ execution oracle clean — closed loop CONVERGED")
            return w
        new_dl = (worst if worst else dl) * 1.15
        print(f"[round {rnd}] execution-oracle feedback (congestion not seen by design-time judica):")
        print(f"   → PHYSICAL: relax shared deadline {dl:.1f} → {new_dl:.1f}s (the time execution really needs)")
        print(f"   → TASK:     re-solve assignment under new budget (reassign / change equivalent target)")
        cur = replace(cur, deadline=new_dl)
    print(f"[closed loop] exhausted {max_rounds} rounds without a clean oracle.")
    return last


if __name__ == "__main__":
    if "--loop" in sys.argv:
        # small dock -> unload congestion the single-robot judica can't predict -> the
        # execution-oracle feedback loop has something to catch and repair.
        run_closed_loop(Config("two_layer_loop", n_robots=6, n_demands=8, equiv=2,
                               seed=7, deadline=12.0, dock_hw=30.0, dock_hh=22.0))
    else:
        out = os.path.join(ROOT, "test_gifs/two_layer_warehouse.gif")
        run(Config("two_layer", n_robots=4, n_demands=6, equiv=2, seed=7, deadline=35.0), gif=out)
