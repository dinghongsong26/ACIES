"""Patrol-responder simulator — expressed through the common simulator interface
(contrauto.sim_iface). A single robot patrols zones A/B/C, inspects an alerted zone,
and files a report. Unlike warehouse the atomic propositions are NOT
warehouse-shaped (at_A/go_to_A/inspect/report) — the scenario declares SENSORS/ACTIONS
keyed directly by the FSM's own variable names (the interface is name-based), proving
the same FsmController/run_fsm drive a structurally different contract.
"""
import argparse, math, random
import pygame
from simulator.core import load_slugs, COLORS, Btn, BaseRobot, BaseGame
from contrauto.sim_iface import SimAgent, SimWorld, Nav, Stationary, FsmController, fsm_step, run_fsm
from contrauto.grounding import Grounding, bind_groundings
from cbf.motion import step_double_integrator_toward_target_cbf

ROBOT_R, SPEED, ZONE_R = 16, 120, 58
INSPECT_T, REPORT_T = 1.5, 1.0
ZONE_XY = {'A': (150, 210), 'B': (450, 210), 'C': (750, 210)}
REPORT_XY = (450, 630)
HOME_XY = (450, 490)
MISSION = CBF_TASK = "patrol"


# ── scenario declarations: name-based predicates + actions ───────────────────
def _zone(a):
    for k, (cx, cy) in ZONE_XY.items():
        if math.hypot(a.x - cx, a.y - cy) < ZONE_R:
            return k
    return None


def _at_report(a):
    return math.hypot(a.x - REPORT_XY[0], a.y - REPORT_XY[1]) < ZONE_R * 0.7


# The ONE place that reads raw patrol state, keyed by capability token.
PROVIDERS = {
    "agent.pos":             lambda a, w: (a.x, a.y),
    "agent.report_ack":      lambda a, w: a.report_ack,
    "agent.inspection_done": lambda a, w: a.inspection_done,
    "world.alerts":          lambda a, w: w.alerts,        # dict zone -> bool
    "geom.zones":            lambda a, w: ZONE_XY,         # dict zone -> (cx, cy)
    "geom.zone_r":           lambda a, w: ZONE_R,
}


def _zone_via(cap):
    """Which zone the agent is in, read through the gate (mirrors _zone, used by tick)."""
    pos = cap.get("agent.pos")
    zr = cap.get("geom.zone_r")
    for k, cxy in cap.get("geom.zones").items():
        if cap.dist(pos, cxy) < zr:
            return k
    return None


# Keyed by the FSM's OWN variable names (regenerated contract uses go_A/inspect/report
# + system-memory output has_inspected; env input `ack` = report acknowledged).
class AtZone(Grounding):
    def __init__(self, z): self.z = z
    def evaluate(self, cap): return _zone_via(cap) == self.z


class Alert(Grounding):
    def __init__(self, z): self.z = z
    def evaluate(self, cap): return cap.get("world.alerts")[self.z]


class ReportAck(Grounding):
    def evaluate(self, cap): return cap.get("agent.report_ack")


class InspectionDone(Grounding):
    def evaluate(self, cap): return cap.get("agent.inspection_done")


PREDICATES = bind_groundings({
    "at_A": AtZone('A'), "at_B": AtZone('B'), "at_C": AtZone('C'),
    "alert_A": Alert('A'), "alert_B": Alert('B'), "alert_C": Alert('C'),
    "ack": ReportAck(), "report_ack": ReportAck(), "inspection_done": InspectionDone(),
}, PROVIDERS)


def _zone_nav(z):
    return Nav(lambda a, w: ZONE_XY[z])


# has_inspected is a SYSTEM memory output, not a physical action -> not in ACTIONS
# (so SimAgent.act ignores it); it only participates in the FSM's own state.
ACTIONS = {
    "go_A": _zone_nav('A'), "go_to_A": _zone_nav('A'),
    "go_B": _zone_nav('B'), "go_to_B": _zone_nav('B'),
    "go_C": _zone_nav('C'), "go_to_C": _zone_nav('C'),
    "inspect": Stationary(None),               # hold; timer + alert-clear in tick()
    "report":  Nav(lambda a, w: REPORT_XY),    # go to station; ack timer in tick()
}


class PatrolRobot(BaseRobot, SimAgent):
    """Single patrol responder; binds only the FSM's own sensor/action variable names."""
    def __init__(self, rid, name, x, y, color_key, fsm_path):
        super().__init__(rid, name, x, y, color_key, CBF_TASK, ROBOT_R, SPEED)
        self.name = name
        data = load_slugs(fsm_path)
        V = data["variables"]
        self.SENSORS = {v: PREDICATES[v] for v in V if v in PREDICATES}
        self.ACTIONS = {v: ACTIONS[v] for v in V if v in ACTIONS}
        in_vars = [v for v in V if v in PREDICATES]
        out_vars = [v for v in V if v in ACTIONS]
        self.ctrl = FsmController(data, in_vars, out_vars)
        self.action = {}
        self._inspect_t, self._report_t = 0.0, 0.0
        self.inspection_done, self.report_ack = False, False
        self.cleared = 0
        self.target = (float(x), float(y))

    def pose(self):
        return (self.x, self.y)

    def act(self, action, world):
        self.action = action
        SimAgent.act(self, action, world)      # Nav -> target / Stationary(inspect) -> hold

    def tick(self, dt, world):
        a = self.action
        self.report_ack = False                # one-shot ack (env input `ack`)
        z = _zone(self)
        # inspect: the FSM commands inspect for one step at the alerted zone; clear the
        # alert immediately (the strategy latches has_inspected and moves on — it does
        # NOT wait for a physical dwell timer, so the effect must be step-aligned).
        if a.get("inspect") and z and world.alerts.get(z):
            world.alerts[z] = False
            self.cleared += 1
        # report: acknowledge as soon as the robot is filing at the station
        if a.get("report") and _at_report(self):
            self.report_ack = True
        # if idle (no action), drift home
        if not any(a.values()):
            self.target = (self.hx, self.hy)
        self.x, self.y, self.vx, self.vy = step_double_integrator_toward_target_cbf(
            self._cbf_task, self.x, self.y, self.vx, self.vy,
            self.target[0], self.target[1], dt, 1.0, self._base_speed, self._robot_radius,
            (), min_dist=1.5)

    def reset(self):
        BaseRobot.reset(self)
        self._inspect_t = self._report_t = 0.0
        self.inspection_done = self.report_ack = False
        self.cleared = 0
        self.ctrl.reset()
        self.target = (self.x, self.y)

    def draw(self, surf, font):
        ix, iy = int(self.x), int(self.y)
        if math.hypot(self.tx - self.x, self.ty - self.y) > 12:
            pygame.draw.line(surf, (80, 80, 110), (ix, iy), (int(self.target[0]), int(self.target[1])), 1)
        pygame.draw.circle(surf, self.color, (ix, iy), ROBOT_R)
        pygame.draw.circle(surf, COLORS["white"], (ix, iy), ROBOT_R, 2)
        surf.blit(font.render(self.name[0], True, COLORS["white"]),
                  font.render(self.name[0], True, COLORS["white"]).get_rect(center=(ix, iy)))


class PatrolGame(BaseGame, SimWorld):
    """Patrol SimWorld: raises zone alerts and runs the responder FSM that clears them."""
    MISSION = MISSION
    SUBTITLE = "Patrol Responder (true FSM)"

    def __init__(self, robot_defs):
        super().__init__()
        self.fS = pygame.font.Font(None, 17)
        self.fT = pygame.font.Font(None, 42)
        for rid, name, x, y, col, path in robot_defs:
            self.robots.append(PatrolRobot(rid, name, x, y, col, path))
        self.controllers = {r.name: r.ctrl for r in self.robots}
        self.selected = self.robots[0]
        self.alerts = {'A': False, 'B': False, 'C': False}
        self._frame, self.alert_frames = 0, 360
        self.cleared = 0
        px = self.PANEL_X + 10
        self.btns = {
            "pause": Btn(px, 640, 120, 28, "PAUSE", COLORS["blue"]),
            "fast":  Btn(px + 130, 640, 120, 28, "FASTER", COLORS["orange"]),
            "reset": Btn(px, 674, 120, 28, "RESET", COLORS["red"]),
            "spawn": Btn(px + 130, 674, 120, 28, "+ALERT", COLORS["zone"]),
        }
        for i, r in enumerate(self.robots):
            self.btns[f"s{r.id}"] = Btn(px, 100 + i * 36, 250, 28, f">> {r.name}", COLORS[robot_defs[i][4]])

    # ── SimWorld interface ───────────────────────────────────────────────────
    def agents(self):
        return self.robots

    def arbiter(self):
        return None

    def step(self, dt):
        self._frame += 1
        if self._frame % self.alert_frames == 0 or not any(self.alerts.values()):
            self.spawn()
        self.cleared = sum(r.cleared for r in self.robots)

    def metrics(self):
        return {"cleared": self.cleared, "active_alerts": sum(self.alerts.values())}

    def spawn(self):
        free = [k for k in ('A', 'B', 'C') if not self.alerts[k]]
        if free:
            self.alerts[random.choice(free)] = True

    def update(self, dt):
        if self.paused: return
        fsm_step(self, self.controllers, dt * self.speed, None)

    def _handle_key(self, key):
        for k, code in (('A', pygame.K_a), ('B', pygame.K_b), ('C', pygame.K_c)):
            if key == code: self.alerts[k] = True

    def _draw_arena(self):
        self._draw_grid()
        for key, (cx, cy) in ZONE_XY.items():
            fill = COLORS["alert"] if self.alerts.get(key) else COLORS["zone"]
            pygame.draw.circle(self.screen, fill, (cx, cy), ZONE_R)
            pygame.draw.circle(self.screen, COLORS["border"], (cx, cy), ZONE_R, 3)
            self.screen.blit(self.fM.render(f"ZONE {key}", True, COLORS["white"]),
                             self.fM.render(f"ZONE {key}", True, COLORS["white"]).get_rect(center=(cx, cy - 10)))
        pygame.draw.circle(self.screen, COLORS["cyan"], REPORT_XY, 20, 3)
        pygame.draw.line(self.screen, COLORS["border"], (self.ARENA_W, 0), (self.ARENA_W, self.H), 3)

    def draw(self):
        self.screen.fill(COLORS["bg"])
        self._draw_arena()
        for r in self.robots:
            r.draw(self.screen, self.fS)
        self._draw_panel()
        pygame.display.flip()

    def _draw_panel(self):
        self._draw_panel_header()
        self._draw_panel_stats(extra_stats=[f"Cleared {self.cleared}",
                                            f"Alerts  {sum(self.alerts.values())}"])
        self._draw_panel_controls()

    def _reset(self):
        super()._reset()
        self.alerts = {'A': False, 'B': False, 'C': False}
        self.cleared = 0; self._frame = 0


def headless(fsm_path, steps=6000, dt=1 / 60.0):
    """Run the patrol FSM without a display; return metrics (cleared alerts)."""
    g = PatrolGame([(1, 'P1', HOME_XY[0], HOME_XY[1], 'blue', fsm_path)])
    return run_fsm(g, g.controllers, steps, dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsm', default='outputs/patrol_regen/workdir/patrolrobot.json')
    parser.add_argument('--headless', type=int, default=0)
    args = parser.parse_args()
    if args.headless:
        print(headless(args.fsm, steps=args.headless))
    else:
        PatrolGame([(1, 'P1', HOME_XY[0], HOME_XY[1], 'blue', args.fsm)]).run()

VOCAB_NOTES = (
    "at_A/at_B/at_C: position sensors; persist while the robot stays (movement is DURATIVE — "
    "use justice G F(at_x | !go_x), never next-step arrival). alert_*: latched until "
    "inspected. inspection_done / report_ack: ONE-STEP acknowledgement pulses — react the "
    "step they appear."
)
