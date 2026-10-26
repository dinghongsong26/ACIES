"""Physical-layer end-to-end validation of the HIERARCHICAL arbiter tree (N=8).

Runs the two_layer_dock physical world with EIGHT worker FSMs coordinated by the
two-level arbiter tree synthesized in outputs/hierarchy_replay/:

    RootArb                 -- arbitrates between the two GROUPS
    GroupArb1 (robots 1-4)  -- arbitrates within its group, gated by the root
    GroupArb2 (robots 5-8)

TreeArbCoordinator wraps the three tree FSMs into ONE composite arbiter with the
same interface the fsm_step driver expects from world.arbiter() (step(reqs, world)
-> grants), reusing the certified per-tick wiring from experiments/
hierarchy_behavioral.py exactly:

    1. RootArb steps on the groups' PREVIOUS-tick greq outputs (the demand
       aggregator in the group contracts is one-step DELAYED by design) -> ggrant(t).
    2. Each GroupArb steps on {req(t), ggrant(t)} -> grant(t); its greq output is
       buffered for the root's next tick.

Everything here is ADDITIVE: TwoLayerDock/DockRobot are reused unchanged; the only
divergences are (a) the tree coordinator replaces the single flat arbiter FSM,
(b) an even spawn ring OUTSIDE the dock (the parent's ring is inside the dock and
clusters at N=8), and (c) a DETERMINISTIC greedy assignment replaces
assign_decomposer.solve (which consults an LLM; this validation must be LLM-free).

Run:  python simulator/tree_dock.py [--gif outputs/instances/hierarchical_dock/tree8.gif]
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from PIL import Image
from contrauto import report
from contrauto.sim_iface import FsmController, load_fsm
from contrauto.assign_decomposer import path_time
from simulator.two_layer_dock import TwoLayerDock, DockRobot
from simulator.two_layer_warehouse import (Config, Layout, Cargo, _obb_aabb,
                                           RR, OBJ, ROBOT_COLORS, DEMAND_COLORS, _tobytes)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE_DIR = ROOT + "/outputs/hierarchy_replay"
GROUPS = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}


class TreeArbCoordinator:
    """Composite two-level arbiter for the SIM side (drop-in for Coordinator).

    Implements the interface fsm_step expects from world.arbiter():
    step(reqs, world) -> {grant_r{i}: bit}. The wiring is byte-identical to the
    behavioral harness (experiments/hierarchy_behavioral.py TreeArb): root first on
    the buffered previous greq, then the groups on the root's current ggrant."""
    name = "hierarchical tree (2x GroupArb + RootArb)"

    def __init__(self, fsm_dir=TREE_DIR, groups=None, apply=None):
        self.groups = groups or GROUPS
        self.apply = apply
        self.g = {}
        for k, rs in self.groups.items():
            self.g[k] = FsmController(
                load_fsm(f"{fsm_dir}/grouparb{k}.json"),
                in_vars=[f"req_r{i}" for i in rs] + [f"ggrant_g{k}"],
                out_vars=[f"grant_r{i}" for i in rs] + [f"greq_g{k}"])
        self.root = FsmController(
            load_fsm(f"{fsm_dir}/rootarb.json"),
            in_vars=[f"greq_g{k}" for k in self.groups],
            out_vars=[f"ggrant_g{k}" for k in self.groups])
        self.reset()

    def reset(self):
        for c in self.g.values():
            c.reset()
        self.root.reset()
        self.prev_greq = {k: 0 for k in self.groups}
        self.last_root = {f"ggrant_g{k}": 0 for k in self.groups}

    def step(self, reqs, world):
        rout = self.root.step({f"greq_g{k}": self.prev_greq[k] for k in self.groups})
        self.last_root = rout
        out = {}
        for k, rs in self.groups.items():
            env = {f"req_r{i}": reqs.get(f"req_r{i}", 0) for i in rs}
            env[f"ggrant_g{k}"] = rout.get(f"ggrant_g{k}", 0)
            gout = self.g[k].step(env)
            for i in rs:
                out[f"grant_r{i}"] = gout.get(f"grant_r{i}", 0)
            self.prev_greq[k] = gout.get(f"greq_g{k}", 0)
        return self.apply(out, world) if self.apply else out


def deterministic_assign(robot_pos, demand_insts, aabb, bounds, deadline, dock):
    """LLM-FREE replacement for assign_decomposer.solve (which prompts an LLM).

    Greedy balanced assignment: demands hardest-first; each goes to a robot with the
    currently smallest task COUNT (so with n_demands == n_robots every robot gets
    exactly one delivery -> every robot must request the dock), choosing the
    (robot, equivalent-instance) pair with the smallest route-time bound
    (path_time chain start -> good -> dock). Returns a solve()-shaped report dict."""
    names = list(robot_pos)
    load = {r: 0.0 for r in names}
    cnt = {r: 0 for r in names}
    cur = dict(robot_pos)
    assign = {r: [] for r in names}

    def best_cost(d):
        return min(path_time(robot_pos[r], xy, aabb, bounds)
                   + path_time(xy, dock, aabb, bounds)
                   for r in names for _, xy in demand_insts[d])

    for d in sorted(demand_insts, key=best_cost, reverse=True):
        mn = min(cnt.values())
        best = None
        for r in [r for r in names if cnt[r] == mn]:
            for nm, xy in demand_insts[d]:
                t = (load[r] + path_time(cur[r], xy, aabb, bounds)
                     + path_time(xy, dock, aabb, bounds))
                if best is None or t < best[0]:
                    best = (t, r, nm)
        if best is None or not math.isfinite(best[0]):
            return {"status": "UNSATISFIABLE", "assignment": None,
                    "deadline": float(deadline), "base_deadline": float(deadline),
                    "relax_x": 1.0, "phases": []}
        t, r, nm = best
        assign[r].append(nm)
        load[r], cur[r] = t, dock
        cnt[r] += 1
    bound = max(load.values()) if load else 0.0
    used = max(float(deadline), 1.4 * bound + 30.0)    # cushion: dock-grant waiting time
    return {"status": "SATISFIED" if used <= float(deadline) else "REPAIRED",
            "assignment": assign, "deadline": round(used, 1),
            "base_deadline": float(deadline),
            "relax_x": round(used / float(deadline), 2),
            "phases": [], "det_bound": round(bound, 1)}


class TreeDock(TwoLayerDock):
    """TwoLayerDock physical world driven by the TREE arbiter instead of one flat
    arbiter FSM. Re-implements __init__ (additively; the parent hardcodes
    arbiter.json + an LLM-backed assignment) but reuses every other parent method."""

    def __init__(self, cfg, fsm_dir=TREE_DIR, groups=None, verbose=True):
        random.seed(cfg.seed)
        self.cfg = cfg
        self.lay = Layout(cfg)
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.W, cfg.H))
        self.font = pygame.font.SysFont(None, 24)
        self.fbig = pygame.font.SysFont(None, 30)
        self.groups = groups or GROUPS
        self.robots = [DockRobot(i + 1, ROBOT_COLORS[i % len(ROBOT_COLORS)],
                                 f"{fsm_dir}/robot{i+1}.json", self.lay)
                       for i in range(cfg.n_robots)]
        self._spawn_ring()
        self.controllers = {r.name: r.ctrl for r in self.robots}

        # composite tree arbiter = the coordination control unit (3 FSMs, one adapter)
        self._req_vars = [f"req_r{r.id}" for r in self.robots]
        self._grant_vars = [f"grant_r{r.id}" for r in self.robots]
        self._coord = TreeArbCoordinator(fsm_dir, self.groups, apply=self._grant_apply)
        self.last_grants = {v: 0 for v in self._grant_vars}
        self.violations = 0
        self.grant_ticks = {r.name: 0 for r in self.robots}     # starvation check
        self.co_occupancy_ticks = 0                             # physical dock mutex check
        self.max_dock_carriers = 0

        # demands -> equivalent goods on random shelves (same policy as the parent)
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
        robot_pos = {r.name: (r.x, r.y) for r in self.robots}
        aabb = [_obb_aabb(o) for o in self.lay.rects]
        self.report = deterministic_assign(
            robot_pos,
            {d: [(n, (self.boxes[n].x, self.boxes[n].y)) for n in insts]
             for d, insts in self.demands.items()},
            aabb, self.lay.bounds, cfg.deadline, self.lay.dock)
        self._apply_assignment()
        self.t = 0.0
        self.delivered = 0
        if verbose:
            self._print_report()

    def _spawn_ring(self):
        """Even spawn ring OUTSIDE the dock's at_dock margin: the parent's ring sits
        inside the dock (breaking the FSMs' initial !at_resource pin) and clusters at
        N=8. 4.5*RR clearance keeps every spawn out of the dock+RR margin box."""
        lay, n = self.lay, len(self.robots)
        rx, ry = lay.dock_hw + 4.5 * RR, lay.dock_hh + 4.5 * RR
        for r in self.robots:
            a = 2 * math.pi * (r.id - 1) / n
            r.x = min(self.cfg.W - RR, max(RR, lay.dock[0] + rx * math.cos(a)))
            r.y = min(self.cfg.H - RR, max(RR, lay.dock[1] + ry * math.sin(a)))
            r._pickup_xy = (r.x, r.y)
            r.target = (r.x, r.y)

    def _grant_apply(self, out, world):
        out = TwoLayerDock._grant_apply(self, out, world)
        for r in self.robots:
            if out.get(f"grant_r{r.id}"):
                self.grant_ticks[r.name] += 1
        return out

    def step(self, dt):
        lay = self.lay
        carriers = sum(1 for r in self.robots
                       if r.carry is not None
                       and abs(r.x - lay.dock[0]) <= lay.dock_hw
                       and abs(r.y - lay.dock[1]) <= lay.dock_hh)
        self.max_dock_carriers = max(self.max_dock_carriers, carriers)
        if carriers >= 2:
            self.co_occupancy_ticks += 1
        TwoLayerDock.step(self, dt)

    def metrics(self):
        per = {r.name: r.done_count for r in self.robots}
        grp = {k: sum(per[f"r{i}"] for i in rs) for k, rs in self.groups.items()}
        return {"delivered": self.delivered,
                "per_robot": per,
                "per_group": grp,
                "grant_mutex_violations": self.violations,
                "co_occupancy_ticks": self.co_occupancy_ticks,
                "max_dock_carriers": self.max_dock_carriers,
                "grant_ticks": dict(self.grant_ticks),
                "starved": [n for n, g in self.grant_ticks.items() if g == 0]}

    def draw(self):
        TwoLayerDock.draw(self)
        gg = self._coord.last_root
        txt = ("TWO-LEVEL ARBITER TREE   RootArb ggrant: "
               f"g1={gg.get('ggrant_g1', 0)}  g2={gg.get('ggrant_g2', 0)}   "
               f"(GroupArb1: r1-r4 | GroupArb2: r5-r8)")
        surf = self.font.render(txt, True, (200, 220, 245))
        pygame.draw.rect(self.screen, (12, 14, 20),
                         (10, self.cfg.H - 34, surf.get_width() + 16, 26))
        self.screen.blit(surf, (18, self.cfg.H - 30))
        pygame.display.flip()


def run_tree8(fsm_dir=TREE_DIR, gif=None, max_steps=8000, dt=1 / 60.0, every=8,
              scale=0.5, seed=3, cfg=None, stop_when_done=None):
    """N=8 dock run under the tree arbiter. Headless (gif=None) runs the FULL horizon
    (idle ticks still exercise the arbiter mutex); a GIF run stops at completion."""
    cfg = cfg or Config("hierarchical_dock8", n_robots=8, n_demands=8, equiv=2,
                        seed=seed, deadline=90.0, dock_hw=58.0, dock_hh=40.0)
    w = TreeDock(cfg, fsm_dir)
    if not w.report["assignment"]:
        print("PHYSICAL layer UNSATISFIABLE.")
        return w
    if stop_when_done is None:
        stop_when_done = bool(gif)
    frames = []
    for i in range(max_steps):
        w.update(dt)
        w.step(dt)
        if gif and i % every == 0:
            w.draw()
            img = Image.frombytes("RGB", w.screen.get_size(), _tobytes(w.screen, "RGB"))
            frames.append(img.resize((int(img.width * scale), int(img.height * scale))))
        if stop_when_done and w.all_done():
            break
    m = w.metrics()
    dl = w.report["deadline"]
    report.section(f"hierarchical tree arbiter x dock (N=8, {i + 1} steps, deadline {dl:.1f}s)")
    for r in w.robots:
        ft = r.finish_t
        report.step(r.done_count > 0,
                    f"{r.name} (group {1 if r.id <= 4 else 2})",
                    f"delivered {r.done_count}  · granted {w.grant_ticks[r.name]} ticks  · "
                    + (f"finished {ft:.1f}s" if ft is not None else "DNF"))
    report.step(all(v > 0 for v in m["per_group"].values()),
                "(a) deliveries in BOTH groups", f"group1={m['per_group'][1]}  group2={m['per_group'][2]}")
    report.step(m["grant_mutex_violations"] == 0,
                "(b) dock mutex - grant level (certified property)",
                f"{m['grant_mutex_violations']} ticks with >1 simultaneous grant")
    report.step(m["co_occupancy_ticks"] == 0,
                "(b) dock mutex - physical occupancy (>=2 carriers inside dock rect)",
                f"{m['co_occupancy_ticks']} ticks  (max simultaneous carriers {m['max_dock_carriers']})")
    report.step(not m["starved"],
                "(c) no starvation (every robot granted at least once)",
                f"starved: {m['starved'] or 'none'}")
    if gif and frames:
        os.makedirs(os.path.dirname(gif), exist_ok=True)
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=int(every * dt * 1000), loop=0, optimize=True)
        report.info("GIF", f"{gif}  ({len(frames)} frames, {os.path.getsize(gif)//1024} KB)")
    return w


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gif", default=None)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()
    w = run_tree8(gif=a.gif, max_steps=a.steps, seed=a.seed)
    print("metrics:", w.metrics())
