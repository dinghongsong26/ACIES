"""C (angle-3, GENERALITY): a shared CHARGING STATION only ONE robot may use at a time.

A THIRD resource-conflict form, distinct from dock (a device) and corridor (a passage):
- The trigger is a CONTINUOUS physical state — the battery level — abstracted to a discrete
  `battery_low` predicate (battery < threshold).
- The task is CHARGING (single-stage, terminal): drive to the station, charge until full.
This is its OWN synthesized FSM (task `charger`, propositions battery_low / at_charger /
charging + req/grant), NOT the dock FSM — the arbiter here gates "charging", and the robot's
objective is `G F(charging)` (recharge indefinitely), while "charging restores the battery" is a
world reaction. Layout is RANDOM per seed: the charging station and each robot's rest point vary.

Physical layer: grant is occupancy. Only the reserved robot may enter the pad; others wait
at per-robot slots or stay at rest. CBF priority follows the reservation.

Role mapping (bind_fsm via the role_of extension): battery_low / at_charger / go_to_charger /
charging + request / grant.

Run:  python simulator/charger.py [--seed N]
"""
import os
if os.environ.get("CHARGER_HEADLESS", "0") == "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import sys
import math
import random
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
from PIL import Image

from contrauto.sim_iface import SimAgent, SimWorld, Nav, Stationary, Coordinator, fsm_step, load_fsm
from contrauto.grounding import Grounding, bind_groundings
from contrauto.stl_cbf import in_circle, out_circle, Always, Eventually, And, stl_cbf_step
from contrauto.geometry import WaypointNav

RR, SPEED = 18, 155
WAIT_R = 88.0                     # standoff: outside at_charger (R+16) and peer bubble
CHARGE_DWELL_STEPS = 120          # ~2s at 60fps: visibly hold still & occupy the pad after charging fires
REST_MARGIN = 110.0               # keep rest points off the arena walls
REST_MIN_DIST = 280.0             # a rest point must be at least this far from the charger
_tobytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSM_DIR = ROOT + "/outputs/charger_n3/workdir"


REST_MIN_SEP = 160.0              # rest points should also stay apart from EACH OTHER


def _far_point(cfg: "Config", charger, avoid=(), min_dist: float = REST_MIN_DIST,
               min_sep: float = REST_MIN_SEP, tries: int = 30):
    """A random point at least `min_dist` from `charger` AND `min_sep` from every point in
    `avoid` (other robots' current rest spots), safely inside the arena.

    Rejection sampling over the whole arena (not a fixed-radius ring around the charger,
    clamped afterward) so every returned point is reliably that far away regardless of
    where the charger sits or how big the arena is -- the old radius+clamp approach could
    silently pull a point back close to the charger once clamped to the arena edge. Without
    the `avoid` check, each robot's own point was individually far from the charger but
    nothing stopped two of them from landing right on top of each other."""
    cx, cy = charger
    x = y = 0.0
    for _ in range(tries):
        x = random.uniform(REST_MARGIN, cfg.W - REST_MARGIN)
        y = random.uniform(REST_MARGIN, cfg.H - REST_MARGIN)
        if math.hypot(x - cx, y - cy) < min_dist:
            continue
        if all(math.hypot(x - ax, y - ay) >= min_sep for ax, ay in avoid):
            return (x, y)
    return (x, y)   # arena too small/crowded to satisfy both after `tries` attempts: best effort


@dataclass
class Config:
    name: str = "charger"
    n_robots: int = 3
    n_recharges: int = 2          # how many full recharges per robot before "done"
    W: int = 1000
    H: int = 700
    seed: int = 3
    low_threshold: float = 0.32   # battery below this -> battery_low
    drain: float = 0.0016         # battery lost per step while NOT charging
    charge: float = 0.010         # battery gained per step while charging


class Layout:
    """A single shared charging station at a random position; each robot rests at a random point."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bounds = (0.0, float(cfg.W), 0.0, float(cfg.H))
        m = 110.0
        self.charger = (cfg.W / 2.0, cfg.H / 2.0)   # fixed at the arena center, not random
        self.rects = []           # no walls (open floor) — CBF only does peer avoidance
        # rest points spread around the station (robots work/idle there, draining, between charges)
        self.rest = []
        self.wait = []
        for i in range(cfg.n_robots):
            self.rest.append(_far_point(cfg, self.charger, avoid=self.rest))
            # fixed 120°-apart wait slots so queued robots do not stack on the pad
            wang = 2 * math.pi * i / max(1, cfg.n_robots) + 0.35
            wx = min(cfg.W - m, max(m, self.charger[0] + WAIT_R * math.cos(wang)))
            wy = min(cfg.H - m, max(m, self.charger[1] + WAIT_R * math.sin(wang)))
            self.wait.append((wx, wy))


def _at(a, pt, tol=RR + 16):
    return (a.x - pt[0]) ** 2 + (a.y - pt[1]) ** 2 <= tol * tol


def _reserved_for(a, w):
    return getattr(w, "_reserve", None) is a


def _do_charge(a, w):
    """Charging restores the battery in ONE step — matching the world reaction the FSM assumes
    (charging -> X(not battery_low)) — and counts the recharge. (A slow per-step top-up would let
    the robot leave the instant it crosses the low threshold, never reaching 'full', so the FSM's
    one-step recovery must be mirrored physically.) Pad occupancy is reserved: being on the
    square without the grant/reservation does not charge.

    The LOGICAL charge is still instantaneous (the contract's timing assumption above is not
    renegotiable), but that used to make the visit itself imperceptible -- `_charging` is only
    ever true for the one step this fires. CHARGE_DWELL_STEPS adds a purely PHYSICAL hold
    afterward (see ChargerRobot.tick): the robot visibly sits still on the pad, and the arbiter
    reservation (see ChargerWorld.step) is held for the same span, so nothing else can move onto
    the pad while it dwells. A freshly-picked rest point (not the same one every time) is also
    queued here so leaving the pad reads as heading somewhere new, not back to a fixed 'home'."""
    if _at(a, w.lay.charger) and _reserved_for(a, w):
        a._charging = True
        if not a._counted_full:
            a._charge_ramp_from = a.battery   # pre-charge level, for the visual ramp only
            a.battery = 1.0
            a.n_charged += 1
            a._counted_full = True
            a._just_charged = True
            a._charge_dwell = CHARGE_DWELL_STEPS
            a.rest = _far_point(w.cfg, w.lay.charger,
                               avoid=[r.rest for r in w.robots if r is not a])


# predicates/actions keyed by charger roles (role_of maps battery_low/at_charger/charging/go_to_charger)
# The ONE place that reads raw charger state, keyed by capability token.
PROVIDERS = {
    "agent.pos":            lambda a, w: (a.x, a.y),
    "agent.battery":        lambda a, w: a.battery,
    "world.low_threshold":  lambda a, w: w.cfg.low_threshold,
    "geom.point:charger":   lambda a, w: w.lay.charger,
}


class BatteryLow(Grounding):
    def evaluate(self, cap): return cap.get("agent.battery") < cap.get("world.low_threshold")


class AtCharger(Grounding):
    def evaluate(self, cap):
        return cap.dist(cap.get("agent.pos"), cap.get("geom.point:charger")) <= RR + 16


PRED = bind_groundings({
    "battery_low": BatteryLow(),
    "at_charger":  AtCharger(),
}, PROVIDERS)
def _goto_done(a, w):
    """Hold Nav only while this robot actually owns the pad. A lazy/blocked goto
    (sent home or to a wait slot) must not freeze the FSM — otherwise battery_low
    is never observed and no one ever requests."""
    return (not _reserved_for(a, w)) or _at(a, w.lay.charger)


ACT = {
    "go_to_charger": Nav(lambda a, w: w.lay.charger, done_when=_goto_done),
    "charging":      Stationary(_do_charge),
}

PREDICATES = PRED
ACTIONS = ACT

ROBOT_COLORS = [(225, 70, 70), (70, 130, 225), (80, 200, 100), (225, 165, 50), (165, 100, 220)]


class ChargerRobot(SimAgent):
    """Worker driven by the charger FSM; the granted station gates `charging` (one at a time)."""
    def __init__(self, rid, fsm_path, lay, cfg, color):
        self.id, self.name, self.color = rid, f"r{rid}", color
        self.bind_fsm(fsm_path, PRED, ACT)
        self.rest = lay.rest[rid - 1]
        self.x, self.y = self.rest
        self.battery = random.uniform(0.55, 1.0)     # start with varied charge
        self.display_battery = self.battery   # cosmetic-only: ramps during the charge dwell
        self._charge_ramp_from = self.battery
        self._charging = False
        self._charge_dwell = 0        # steps left to visibly sit still after charging fires
        self._counted_full = False
        self._just_charged = False
        self.n_charged = 0
        self.target = (self.x, self.y)
        self._nav = WaypointNav(obb=True)
        self.vel = np.zeros(2)
        self.priority = rid

    def pose(self):
        return (self.x, self.y)

    def _requesting(self):
        return any(self.ctrl.requests(self.REQUESTS).values())

    def _retarget(self, action, world):
        """Grant-gated targets: only the reserved robot may aim at the pad."""
        mine = _reserved_for(self, world)
        self.priority = 1000 if mine else self.id
        if mine:
            self.target = world.lay.charger
            return
        # Only actively wanting the pad soon routes through the nearby wait slot; a robot
        # that JUST finished dwelling (battery full, not requesting) has no reason to loiter
        # near the charger just because it is still physically standing there -- send it all
        # the way to its rest point. The old `_at(self, charger)` clause sent every
        # freshly-charged robot to "wait" first (close by) purely because it hadn't moved
        # yet, which read as "never actually leaving" the charger.
        if self._requesting():
            self.target = world.lay.wait[self.id - 1]
            return
        self.target = self.rest

    def act(self, action, world):
        self._charging = False
        # reset the "counted this full charge" latch once we leave full (so the NEXT charge counts)
        if self.battery < 0.999:
            self._counted_full = False
        SimAgent.act(self, action, world)
        self._retarget(action, world)

    def tick(self, dt, world):
        lay = world.lay
        if self._charge_dwell > 0:
            # Sit still on the pad for a beat instead of the FSM's one-step charge reading
            # as a blink-and-you-miss-it touch; no CBF/nav this tick, velocity pinned to 0.
            # The LOGICAL battery (drives battery_low) still jumped to 1.0 instantly in
            # _do_charge -- that timing is the verified contract's own assumption and is not
            # renegotiable. display_battery is a separate, purely cosmetic value that ramps
            # from the pre-charge level up to full over the dwell, so the bar reads as an
            # actual charge-up instead of an instant jump.
            self._charge_dwell -= 1
            frac = 1.0 - self._charge_dwell / CHARGE_DWELL_STEPS
            self.display_battery = self._charge_ramp_from + (1.0 - self._charge_ramp_from) * frac
            self.x, self.y = lay.charger
            self.vel = np.zeros(2)
            return
        self.display_battery = self.battery
        # holding() / reapply may re-aim at the pad; keep occupancy gated every frame
        mine = _reserved_for(self, world)
        self.priority = 1000 if mine else self.id
        if mine:
            self.target = lay.charger
        elif self._requesting():
            self.target = lay.wait[self.id - 1]
        else:
            self.target = self.rest
        # Already at (or effectively at) the target: hold still instead of running CBF
        # avoidance against nothing -- a queued/parked robot with nowhere to go was still
        # reacting to every passing peer, which read as never quite settling ("queueing
        # should stop and stand still", not keep jittering in place).
        if (self.x - self.target[0]) ** 2 + (self.y - self.target[1]) ** 2 < 4.0:
            self.vel = np.zeros(2)
            return
        wp = self._nav.target((self.x, self.y), self.target, lay.rects, lay.bounds, RR)
        p = np.array([self.x, self.y], float)
        seg_dl = world.t + float(np.linalg.norm(np.array(wp, float) - p)) / SPEED * 1.3 + dt
        formula = And([Eventually(world.t, seg_dl, in_circle(wp[0], wp[1], RR))]
                      + [Always(0.0, 1e9, out_circle(o.x, o.y, 2 * RR + 6))
                         for o in world.robots if o is not self and o.priority > self.priority])
        dirn = np.array(wp, float) - p
        nrm = float(np.linalg.norm(dirn)) or 1.0
        u_nom = 6.0 * (dirn / nrm * SPEED - self.vel)
        p_new, self.vel, _, _ = stl_cbf_step(formula, p, self.vel, world.t, u_nom, dt, v_max=SPEED)
        self.x = min(lay.cfg.W - RR, max(RR, float(p_new[0])))
        self.y = min(lay.cfg.H - RR, max(RR, float(p_new[1])))


class ChargerWorld(SimWorld):
    """SimWorld reusing the synthesized charger arbiter FSM to enforce one-robot-at-a-time charging."""
    def __init__(self, cfg, fsm_dir=FSM_DIR, headless=False):
        random.seed(cfg.seed)
        self.cfg = cfg
        self.lay = Layout(cfg)
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.W, cfg.H))
        self.font = pygame.font.SysFont(None, 40)
        self.font_charger = pygame.font.SysFont(None, 52)
        self.robots = [ChargerRobot(i + 1, f"{fsm_dir}/robot{i+1}.json", self.lay, cfg,
                                    ROBOT_COLORS[i % len(ROBOT_COLORS)]) for i in range(cfg.n_robots)]
        self.controllers = {r.name: r.ctrl for r in self.robots}
        ad = load_fsm(f"{fsm_dir}/arbiter.json")
        rq = [v for v in ad["variables"] if "req" in v.lower()]
        gr = [v for v in ad["variables"] if "grant" in v.lower()]
        self.last_grants = {}
        self._reserve = None           # robot that currently owns the pad
        self._reserve_charged = False
        self._coord = Coordinator(ad, rq, gr, apply=self._grant_apply)
        self.violations = 0
        self.max_charging = 0
        self.t = 0.0

    def agents(self):
        return self.robots

    def arbiter(self):
        return self._coord

    def _grant_apply(self, out, world):
        """Sticky reservation: a 1-step arbiter pulse claims the pad until that robot
        finishes a charge (or drops its request away from the pad). Injected grants stay
        high for the holder so the robot FSM can charge on arrival, not only on the pulse."""
        if self._reserve is None:
            for r in self.robots:
                if r.GRANTS and any(out.get(g, 0) for g in r.GRANTS):
                    self._reserve = r
                    self._reserve_charged = False
                    break
        self.last_grants = {k: int(self._reserve is not None and k in self._reserve.GRANTS)
                            for k in out}
        return self.last_grants

    def step(self, dt):
        self.t += dt
        for r in self.robots:                         # continuous battery dynamics
            if not r._charging and r._charge_dwell <= 0:   # freshly full during dwell: don't drain
                r.battery = max(0.0, r.battery - self.cfg.drain)
        n_charging = sum(1 for r in self.robots if r._charging)   # capacity-1 mutex monitor
        self.max_charging = max(self.max_charging, n_charging)
        if n_charging > 1:
            self.violations += 1
        holder = self._reserve
        if holder is not None:
            if holder._charging or holder._just_charged:
                self._reserve_charged = True
            # Keep the reservation through the visual dwell (see ChargerRobot.tick) so the
            # arbiter can't hand the pad to someone else while the holder is still sitting on it.
            done = self._reserve_charged and not holder._charging and holder._charge_dwell <= 0
            gone = (not holder._requesting()) and (not _at(holder, self.lay.charger))
            if done or gone:
                holder._just_charged = False
                self._reserve = None
                self._reserve_charged = False

    def all_done(self):
        return all(r.n_charged >= self.cfg.n_recharges for r in self.robots)

    def metrics(self):
        recharges = sum(r.n_charged for r in self.robots)
        return {"total_recharges": recharges, "delivered": recharges,
                "charger_violations": self.violations, "violations": self.violations,
                "max_charging": self.max_charging,
                "per_robot": [r.n_charged for r in self.robots]}

    def draw(self):
        s, lay = self.screen, self.lay
        s.fill((24, 26, 32))
        cx, cy = int(lay.charger[0]), int(lay.charger[1])
        # Display-only count: includes the visual dwell, not just the FSM's one-step
        # _charging pulse, so the HUD/box color reflect what a viewer actually sees on the
        # pad. The mutex-violation monitor in step() deliberately still counts _charging
        # alone -- that one must stay tied to the verified contract's own instant, or it
        # would flag a false violation for every routine dwell overlap.
        n_charging_display = sum(1 for r in self.robots if r._charging or r._charge_dwell > 0)
        col = (60, 170, 90) if n_charging_display <= 1 else (210, 70, 70)   # green free / red if >1
        box = 84
        half = box // 2
        pygame.draw.rect(s, col, (cx - half, cy - half, box, box), 0, border_radius=10)
        pygame.draw.rect(s, (240, 240, 245), (cx - half, cy - half, box, box), 3, border_radius=10)
        label = self.font_charger.render("charger", True, col)
        s.blit(label, (cx - label.get_width() // 2, cy + half + 10))
        for r in self.robots:
            pygame.draw.circle(s, r.color, (int(r.x), int(r.y)), RR)
            # battery bar above the robot: green>low, red when low
            bw = 26
            pygame.draw.rect(s, (60, 60, 66), (int(r.x) - bw // 2, int(r.y) - RR - 14, bw, 6))
            bc = (230, 80, 70) if r.battery < self.cfg.low_threshold else (90, 210, 110)
            pygame.draw.rect(s, bc, (int(r.x) - bw // 2, int(r.y) - RR - 14, int(bw * r.display_battery), 6))
            if r._charging or r._charge_dwell > 0:
                pygame.draw.circle(s, (240, 230, 90), (int(r.x), int(r.y)), RR + 5, 2)
        hud = (f"t={self.t:5.1f}s   charging now: {n_charging_display}   "
               f"mutex violations: {self.violations}   recharges: {sum(r.n_charged for r in self.robots)}")
        s.blit(self.font.render(hud, True, (235, 235, 240)), (16, 12))


def run(cfg, fsm_dir=FSM_DIR, gif=None, max_steps=20000, dt=1 / 60.0, every=18, scale=0.6,
    visualize=True):
    """Run the charger world to completion (optionally capture a GIF); return the world.

    `visualize=True` opens and updates the live pygame window. `visualize=False` runs
    the same finite simulation off-screen. Supplying `gif` additionally samples frames
    into an animated GIF file; a GIF is an export artifact, not the live window itself.
    """
    w = ChargerWorld(cfg, fsm_dir, headless=(not visualize or gif is not None))
    arb = w.arbiter()
    frames = []
    clock = pygame.time.Clock()
    if visualize and gif is None:
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
            pygame.event.pump()
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
    pygame.quit()
    return w


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--recharges", type=int, default=2,
                   help="full charges per robot before the run stops (default 2)")
    p.add_argument("--steps", type=int, default=20000,
                   help="hard cap on simulation steps (default 20000)")
    p.add_argument("--gif", default=ROOT + "/outputs/case_studies/gifs/charger.gif")
    a = p.parse_args()
    w = run(Config(n_robots=a.n, seed=a.seed, n_recharges=a.recharges),
            gif=a.gif or None, max_steps=a.steps)
    print("charger:", w.metrics(), "GIF:", a.gif)
