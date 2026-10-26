"""two_layer WITH a genuine shared resource: 2 workers + capacity-1 shared DOCK + a
synthesized Arbiter (request->grant), on the SAME physical layer (assign deadline +
execution feedback). EVERY synthesized FSM — both worker FSMs AND the arbiter FSM — is a
control unit in the simulator (framework principle: every produced FSM is wired into
physical execution; the arbiter is not special).

task layer  : outputs/two_layer_dock_n2/workdir/{robot1,robot2,arbiter}.json
              (LLM contracts -> refinement -> well-sep gate -> SLUGS FSMs)
physical    : assign deadline judica (CBF bound) + OBB nav + dock mutex via the arbiter

Variable binding is role-based via bind_fsm (which uses bind_by_role/role_of). role_of handles
this contract's names (e.g. `at_pickup`, `dock_req`, `req`/`request`), so a renamed synthesis
still binds without hardcoding a prefix.

Run:  python simulator/two_layer_dock.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from PIL import Image
from contrauto import report
from contrauto.sim_iface import (SimAgent, SimWorld, Nav, Stationary,
                                 Coordinator, fsm_step, load_fsm)
from contrauto.assign_decomposer import solve, path_time
import numpy as np
from dataclasses import replace
from contrauto.stl_cbf import in_circle, out_circle, out_obb, Always, Eventually, And, stl_cbf_step
from simulator.two_layer_warehouse import (
    Config, Layout, Cargo, _obb_aabb, _near, PRED, ACT,
    _current, _at_shelf, _shelf_target, _do_pick, _at_dock, _do_drop,
    RR, OBJ, SPEED, ROBOT_COLORS, DEMAND_COLORS, _tobytes)
from contrauto.geometry import WaypointNav

PREDICATES = PRED
ACTIONS = ACT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSM_DIR = ROOT + "/outputs/two_layer_dock_n2/workdir"


def _task(a, w):
    return (a.qi < len(a.queue)) or (a.carry is not None)


class DockRobot(SimAgent):
    """Worker driven by its synthesized FSM (robot{rid}.json), which requests the shared
    dock before delivering. Variables bound by actual name (r{rid}_ prefix)."""
    def __init__(self, rid, color, fsm_path, lay):
        self.id, self.name, self.color = rid, f"r{rid}", color
        # NAMING-AGNOSTIC binding (bind_fsm): maps whatever names this synthesis used
        # (r1_/suffix1, move_/goto_, req/request, dock_grant/granted, at_pickup/at_shelf) to
        # canonical roles via role_of, so a renamed synthesis still binds.
        self.bind_fsm(fsm_path, PRED, ACT)
        ang0 = rid * 2.3
        self.x = lay.dock[0] + lay.dock_hw * 0.9 * math.cos(ang0)
        self.y = lay.dock[1] + lay.dock_hh * 0.9 * math.sin(ang0)
        self.carry = None
        self._pickup_xy = (self.x, self.y)
        self._jd = False
        self.done_count = 0
        self.queue, self.qi, self.finish_t = [], 0, None
        self.target = (self.x, self.y)
        self._nav = WaypointNav(obb=True)
        self.vel = np.zeros(2)            # physical state for the CBF-QP
        self._max_slack = 0.0             # peer-avoidance infeasibility signal (repair trigger)
        self.priority = rid               # higher priority = others yield to it (task-coord repair)

    def pose(self):
        return (self.x, self.y)

    def act(self, action, world):
        self._jd = False
        SimAgent.act(self, action, world)

    def tick(self, dt, world):
        # PHYSICAL layer = STL/CBF timed-reach contract (by construction). The A* path gives
        # convex segments (around the non-convex shelves); each waypoint gets a sub-deadline
        # (cumulative length / speed + margin) encoded as F[0,sub_dl](reach wp), plus
        # G(avoid peers). The CBF-QP enforces the timed reach; slack = how far behind schedule
        # = the deadline-infeasibility / repair signal.
        lay = world.lay
        wp = self._nav.target((self.x, self.y), self.target, lay.rects, lay.bounds, RR)  # next convex waypoint
        p = np.array([self.x, self.y], float)
        # one timed-reach leaf for the CURRENT segment (sub-deadline = segment length / speed +
        # margin) — WaypointNav advances to the next waypoint as each is reached. Avoiding the
        # "all waypoints at once" conflict that over-constrains the QP.
        seg_dl = world.t + float(np.linalg.norm(np.array(wp, float) - p)) / SPEED * 1.3 + dt
        # asymmetric (priority-based) avoidance: only yield to HIGHER-priority peers. Unique
        # priorities => exactly one of each pair yields => no symmetric deadlock, and a
        # high-priority robot is NOT slowed by avoidance (this is the task-coordination lever).
        # SHELF (wall) avoidance: G(out_obb(shelf)) for every nearby shelf — the constraint the
        # CBF-QP was missing (it routed around walls via A* waypoints but the QP itself could cut
        # corners into a shelf). Each shelf is an oriented box; keep the robot centre RR clear.
        shelf_rows = [Always(0.0, 1e9, out_obb(cx, cy, hw, hh, ang, margin=RR + 5))
                      for (cx, cy, hw, hh, ang) in lay.rects
                      if (cx - p[0]) ** 2 + (cy - p[1]) ** 2 < (max(hw, hh) + 140) ** 2]
        formula = And([Eventually(world.t, seg_dl, in_circle(wp[0], wp[1], RR))]
                      + [Always(0.0, 1e9, out_circle(o.x, o.y, 2 * RR + 6))
                         for o in world.robots if o is not self and o.priority > self.priority]
                      + shelf_rows)
        dirn = np.array(wp, float) - p
        nrm = float(np.linalg.norm(dirn)) or 1.0
        u_nom = 6.0 * (dirn / nrm * SPEED - self.vel)
        p_new, self.vel, _, slack = stl_cbf_step(formula, p, self.vel, world.t, u_nom, dt, v_max=SPEED)
        self.x = min(lay.cfg.W - RR, max(RR, float(p_new[0])))
        self.y = min(lay.cfg.H - RR, max(RR, float(p_new[1])))
        self._max_slack = max(self._max_slack, slack)
        if self.carry is not None:
            self.carry.x, self.carry.y = self.x, self.y - RR - OBJ // 2


class TwoLayerDock(SimWorld):
    """Two-layer dock world: worker FSMs + an arbiter FSM enforcing the capacity-1 dock mutex."""
    def __init__(self, cfg, fsm_dir=FSM_DIR, verbose=True, model="gemini"):
        random.seed(cfg.seed)
        self.cfg = cfg
        self.lay = Layout(cfg)
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.W, cfg.H))
        self.font = pygame.font.SysFont(None, 24)
        self.fbig = pygame.font.SysFont(None, 30)
        self.robots = [DockRobot(i + 1, ROBOT_COLORS[i % len(ROBOT_COLORS)],
                                 f"{fsm_dir}/robot{i+1}.json", self.lay)
                       for i in range(cfg.n_robots)]
        self.controllers = {r.name: r.ctrl for r in self.robots}

        # arbiter FSM = a control unit, wired exactly like the workers
        ad = load_fsm(f"{fsm_dir}/arbiter.json")
        self._req_vars = [v for v in ad["variables"] if "req" in v]
        self._grant_vars = [v for v in ad["variables"] if "grant" in v]
        self._coord = Coordinator(ad, self._req_vars, self._grant_vars, apply=self._grant_apply)
        self.last_grants = {v: 0 for v in self._grant_vars}
        self.violations = 0

        # demands -> equivalent goods on random shelves -> deadline assignment (physical layer)
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
        groups = {d: [(self.boxes[n].x, self.boxes[n].y) for n in insts]
                  for d, insts in self.demands.items()}
        aabb = [_obb_aabb(o) for o in self.lay.rects]
        self.report = solve(robot_pos, groups, aabb, self.lay.bounds, cfg.deadline,
                            model=model, dock=self.lay.dock)
        self._apply_assignment()
        self.t = 0.0
        self.delivered = 0
        if verbose:
            self._print_report()

    def _apply_assignment(self):
        """Load each robot's delivery queue from the solver's assignment and mark chosen goods."""
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
        tag = (f"relaxed {rep['base_deadline']:.0f}->{rep['deadline']:.0f}s"
               if rep["status"] == "REPAIRED" else "")
        report.section("assign + deadline")
        report.step(rep["status"] != "UNSATISFIABLE", f"deadline judica: {rep['status']}", tag)
        for r in self.robots:
            report.info(r.name, str([b.name for b in r.queue]))
        self._physical_certificate()

    def _physical_certificate(self):
        """Option A: emit each robot's PHYSICAL-LAYER contract certificate — the deadline as a
        guaranteed property of a satisfied CBF contract. The mission route (start -> good ->
        dock -> good -> dock ...) is structured as a chain of TV-FCBF reach contracts (each
        segment's interval = the CBF-provable trapezoidal reach time) ∧ TV-CBF shelf-avoid
        contracts; the provable completion bound is re-checked against the deadline. These are
        the Level-2 contract objects an HHCN would trace up to the task-layer contracts."""
        from contrauto.cbf_contracts import physical_certificate
        aabb = [_obb_aabb(o) for o in self.lay.rects]
        dl = self.report["deadline"]
        report.section("physical contract certificate (RC reach ∧ CBC avoid)")
        for r in self.robots:
            legs = []
            for b in r.queue:
                legs += [(b.x, b.y), self.lay.dock]
            cert = physical_certificate((r.x, r.y), legs, aabb, self.lay.bounds, dl)
            report.step(cert["ok"], f"{r.name}  ({cert['n_reach']} RC reach segs)",
                        (f"provable completion ≤ {cert['bound']:.1f}s ≤ deadline {dl:.1f}s ✓"
                         if cert["ok"] else
                         (f"provable {cert['bound']:.1f}s > deadline {dl:.1f}s → repair"
                          if cert["bound"] != float("inf") else
                          f"route unreachable → repair  {cert['issues']}")))

    def _grant_apply(self, out, world):
        """Record the arbiter's grants and count any >1 simultaneous grant (dock mutex breach)."""
        self.last_grants = out
        if sum(1 for v in self._grant_vars if out.get(v)) > 1:
            self.violations += 1               # >1 robot granted the capacity-1 dock = mutex breach
        return out

    def agents(self):
        return self.robots

    def arbiter(self):
        return self._coord

    def step(self, dt):
        self.t += dt
        self.delivered = sum(r.done_count for r in self.robots)

    def update(self, dt):
        fsm_step(self, self.controllers, dt, self.arbiter())

    def all_done(self):
        return all(r.finish_t is not None for r in self.robots)

    def metrics(self):
        return {"delivered": self.delivered, "violations": self.violations}

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
        held = sum(1 for v in self.last_grants.values() if v)
        col = (180, 70, 70) if held > 1 else ((60, 140, 90) if held == 1 else (45, 105, 155))
        pygame.draw.rect(s, col, (dx0, dy0, lay.dock_hw * 2, lay.dock_hh * 2))
        pygame.draw.rect(s, (120, 180, 220), (dx0, dy0, lay.dock_hw * 2, lay.dock_hh * 2), 2)
        s.blit(self.font.render(f"DOCK (cap 1, held {held})", True, (255, 255, 255)),
               (int(lay.dock[0]) - 60, int(lay.dock[1]) - 8))
        for b in self.boxes.values():
            if b.delivered or b.carrier is not None:
                continue
            cc = b.dcolor if b.chosen else (70, 72, 80)
            pygame.draw.rect(s, cc, (int(b.x) - OBJ // 2, int(b.y) - OBJ // 2, OBJ, OBJ))
        for r in self.robots:
            ix, iy = int(r.x), int(r.y)
            pygame.draw.circle(s, r.color, (ix, iy), RR)
            pygame.draw.circle(s, (255, 255, 255), (ix, iy), RR, 2)
            if r.carry is not None:
                pygame.draw.rect(s, r.carry.dcolor, (ix - OBJ // 2, iy - RR - OBJ - 2, OBJ, OBJ))
        rep = self.report
        hud = (f"TASK: {len(self.robots)} worker FSM + ARBITER (cap-1 dock)   PHYSICAL: {rep['status']}   "
               f"t={self.t:4.1f}s  delivered {self.delivered}  mutex viol {self.violations}")
        surf = self.fbig.render(hud, True, (235, 235, 245))
        pygame.draw.rect(s, (12, 14, 20), (10, 8, surf.get_width() + 16, 32))
        s.blit(surf, (18, 12))
        pygame.display.flip()


def run(cfg, fsm_dir=FSM_DIR, max_steps=8000, dt=1 / 60.0, gif=None, every=5, scale=0.5,
        model="gemini"):
    w = TwoLayerDock(cfg, fsm_dir, model=model)
    if not w.report["assignment"]:
        print("PHYSICAL layer UNSATISFIABLE."); return w
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
    dl = w.report["deadline"]
    total = sum(len(r.queue) for r in w.robots)
    report.section(f"execution (FSM-driven, deadline {dl:.1f}s)")
    for r in w.robots:
        ft = r.finish_t
        within = ft is not None and ft <= dl
        report.step(within, f"{r.name}  ({len(r.queue)} goods)",
                    (f"finished {ft:.1f}s {'≤' if within else '>'} {dl:.1f}s" if ft is not None else "DNF")
                    + f"  · peer-avoid slack {r._max_slack:.0f}")
    if gif and frames:
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=int(every * dt * 1000), loop=0, optimize=True)
        report.info("GIF", f"{gif}  ({len(frames)} frames, {os.path.getsize(gif)//1024} KB)")
    return w   # the caller (main.py orchestrator, or __main__) prints the final summary


def run_closed_loop(cfg, fsm_dir=FSM_DIR, max_rounds=6, max_steps=8000, dt=1 / 60.0,
                    model="gemini"):
    """PHYSICAL-layer repair loop (escalation-ladder steps 1-3 — all physical). When the
    execution oracle finds a robot missed the deadline, prefer to ELIMINATE the avoidance cost
    by COORDINATION before merely relaxing the requirement:

      1. PHYSICAL coordination (yielding) — raise the offender's collision-avoidance priority
         so peers yield to it (a CBF-QP parameter; does NOT touch the FSM). Same deadline.
      2. PHYSICAL relax — if it is already top priority and still over, the deadline is
         genuinely unreachable: relax the shared deadline + re-solve the assignment.

    NB both tiers are PHYSICAL (yielding priority and deadline are physical-execution/planning
    knobs; the synthesized FSM is untouched). A STRUCTURAL failure that survives this loop is
    escalated to the TASK layer by run_with_task_repair (ladder step 4). Returns a dict
    {converged, world, delivered, mutex_viol, rounds_used, final_deadline, history}."""
    cur, last, prio, coord_tries = cfg, None, None, 0
    history = []
    for rnd in range(max_rounds):
        w = TwoLayerDock(cur, fsm_dir, verbose=(rnd == 0), model=model)
        last = w
        if not w.report["assignment"]:
            report.step(False, f"round {rnd}", "PHYSICAL UNSATISFIABLE — stop")
            return {"converged": False, "world": w, "delivered": 0, "mutex_viol": w.violations,
                    "rounds_used": rnd + 1, "final_deadline": cur.deadline, "history": history,
                    "unsat": True}
        if prio is None:
            prio = {r.name: r.id for r in w.robots}
        for r in w.robots:
            r.priority = prio[r.name]                       # apply current yielding priorities
        for i in range(max_steps):
            w.update(dt)
            w.step(dt)
            if w.all_done():
                break
        dl = w.report["deadline"]
        rows = [(r.name, r.finish_t, r._max_slack) for r in w.robots]
        history.append({"round": rnd, "deadline": dl, "rows": rows, "delivered": w.delivered})
        viol = [n for n, ft, _ in rows if ft is None or ft > dl + 1e-6]
        report.section(f"round {rnd}  ·  deadline {dl:.1f}s  ·  priorities {prio}")
        for n, ft, s in rows:
            ok = ft is not None and ft <= dl + 1e-6
            report.step(ok, n, (f"finished {ft:.1f}s" if ft is not None else "DNF") + f"  · slack {s:.0f}")
        if not viol:
            report.summary(True, f"CONVERGED (round {rnd})",
                           {"delivered": w.delivered, "mutex_viol": w.violations})
            return {"converged": True, "world": w, "delivered": w.delivered,
                    "mutex_viol": w.violations, "rounds_used": rnd + 1,
                    "final_deadline": dl, "history": history}
        # worst offender = a DNF first, else the largest finish time
        dnf = [n for n, ft, _ in rows if ft is None]
        worst_name = dnf[0] if dnf else max((r for r in rows if r[1] is not None), key=lambda r: r[1])[0]
        if coord_tries < len(prio):
            # TIER 1 — PHYSICAL coordination (yielding): peers yield to the offender. Bounded by
            # #robots: priority is zero-sum (helping one delays another), so if a full round of
            # re-prioritization still misses, coordination alone can't fix it.
            prio[worst_name] = max(prio.values()) + 1
            coord_tries += 1
            report.info(f"repair → PHYSICAL coordination ({coord_tries}/{len(prio)})",
                        f"raise {worst_name} yielding priority (peers give way) — same deadline")
        else:
            # TIER 2 — PHYSICAL relax: coordination exhausted (whack-a-mole) ⇒ the deadline is
            # genuinely unreachable for this congestion, not a yielding artifact. Relax + re-solve.
            worst = max((ft for _, ft, _ in rows if ft is not None), default=dl)
            new_dl = worst * 1.4
            report.info("repair → PHYSICAL relax",
                        f"coordination exhausted → deadline unreachable → relax {dl:.1f}→{new_dl:.1f}s + re-solve")
            cur = replace(cur, deadline=new_dl)
            coord_tries = 0                                 # let coordination retry under the new budget
    report.summary(False, "NOT CONVERGED (physical repair exhausted)", {"rounds_used": max_rounds})
    return {"converged": False, "world": last, "delivered": last.delivered if last else 0,
            "mutex_viol": last.violations if last else 0, "rounds_used": max_rounds,
            "final_deadline": cur.deadline, "history": history}


def classify_structural_failure(history, robot_names):
    """Decide whether a NOT-CONVERGED physical run is STRUCTURAL (escalate to TASK re-synthesis,
    ladder step 4) rather than quantitative (which steps 1-3 should have handled). Returns
    {structural, kind, message}; `message` is actionable feedback for the decomposer. Signals:
      starvation  — a robot never finished in ANY round (the arbiter never served it);
      no_progress — total delivered never increased across rounds (deadlock / livelock);
      diverging   — the deadline was relaxed yet it still never converged (decomposition likely
                    infeasible / wrong ordering)."""
    if not history:
        return {"structural": False, "kind": "none", "message": ""}
    # starvation: a robot that is DNF (finish_t is None) in every recorded round
    never_done = []
    for name in robot_names:
        if all(any(n == name and ft is None for n, ft, _ in h["rows"]) for h in history):
            never_done.append(name)
    if never_done:
        return {"structural": True, "kind": "starvation",
                "message": (f"Robot(s) {never_done} never completed any delivery in {len(history)} "
                            f"physical-repair rounds — the arbiter never granted them the dock "
                            f"(starvation). Re-synthesize the arbiter with a FAIRNESS guarantee so "
                            f"every requesting robot is EVENTUALLY granted (no robot is starved).")}
    delivered = [h["delivered"] for h in history]
    if len(delivered) >= 2 and max(delivered) == delivered[0] and delivered[0] < len(robot_names):
        return {"structural": True, "kind": "no_progress",
                "message": (f"Delivery count stayed at {delivered[0]} across {len(history)} rounds "
                            f"despite priority/deadline repair — a deadlock / circular wait in the "
                            f"coordination protocol. Re-synthesize to break the circular dependency "
                            f"(e.g. a strict request/grant ordering that cannot cycle).")}
    deadlines = [h["deadline"] for h in history]
    if len(deadlines) >= 2 and deadlines[-1] > deadlines[0] + 1e-6:
        return {"structural": True, "kind": "diverging",
                "message": (f"Deadline was relaxed {deadlines[0]:.1f}->{deadlines[-1]:.1f}s yet the "
                            f"system still did not converge — the task decomposition is likely "
                            f"infeasible. Re-split the tasks so each robot's load is reachable, or "
                            f"correct the delivery ordering.")}
    return {"structural": False, "kind": "quantitative", "message": ""}


def run_with_task_repair(task, cfg, fsm_dir, model="gemini", outdir=None,
                         max_task_rounds=2, **phys_kw):
    """OUTER (task-level) repair loop — escalation-ladder step 4. Run the PHYSICAL layer with its
    inner repair (run_closed_loop); if it converges, done. If it fails and the failure is
    STRUCTURAL (classify_structural_failure), RE-SYNTHESIZE the FSM with that structural feedback
    (ContrAuto physical_feedback channel) and retry. This is the previously-missing physical->task
    feedback edge. Returns the final run_closed_loop result dict (+ task_rounds)."""
    from contrauto.pipeline import ContrAuto
    feedback = ""
    result = None
    for tr in range(max_task_rounds):
        if tr > 0:   # tr==0 uses the existing FSM; later rounds re-synthesize with feedback
            report.section(f"TASK-LEVEL REPAIR round {tr} — re-synthesize FSM with structural feedback")
            ContrAuto(task=task, model=model, outdir=outdir, physical_feedback=feedback).run()
        result = run_closed_loop(cfg, fsm_dir=fsm_dir, model=model, **phys_kw)
        result["task_rounds"] = tr + 1
        if result["converged"]:
            return result
        names = [n for n, *_ in result["history"][-1]["rows"]] if result["history"] else []
        diag = classify_structural_failure(result["history"], names)
        if not diag["structural"]:
            report.summary(False, "physical repair exhausted (non-structural — no FSM change)", {})
            return result
        report.info(f"escalate → TASK-LEVEL repair ({diag['kind']})", diag["message"])
        feedback += ("\n- " + diag["message"]) if feedback else diag["message"]
    report.summary(False, "TASK REPAIR exhausted", {"task_rounds": max_task_rounds})
    return result


if __name__ == "__main__":
    if "--n3" in sys.argv:                     # N=3: 3 workers + cap-1 dock + 3-way arbiter
        w = run(Config("two_layer_dock3", n_robots=3, n_demands=6, equiv=2, seed=7,
                       deadline=55.0, dock_hw=50.0, dock_hh=36.0),
                fsm_dir=ROOT + "/outputs/two_layer_dock_n3/workdir",
                gif=os.path.join(ROOT, "test_gifs/two_layer_dock3.gif"))
    else:
        w = run(Config("two_layer_dock", n_robots=2, n_demands=6, equiv=2, seed=7,
                       deadline=40.0, dock_hw=46.0, dock_hh=34.0),
                gif=os.path.join(ROOT, "test_gifs/two_layer_dock.gif"))
    if w.report.get("assignment"):
        report.summary(w.delivered > 0 and w.violations == 0, "PHYSICAL EXECUTION",
                       {"delivered": w.delivered, "mutex_viol": w.violations})
