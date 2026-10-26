"""Optional 2D pygame rendering helpers (BaseGame / BaseRobot). NOT the simulator base.

The one simulator base is SimAgent / SimWorld in `contrauto/sim_iface.py`: it is
rendering- and dimension-agnostic, and every scenario (2D or 3D) is a subclass that
implements pose()/tick() and declares its sensors/actions. The classes here just provide
a shared 2D board (window, control panel, circle robots) that a 2D scenario MAY mix in for
an interactive view (e.g. `class WarehouseRobot(BaseRobot, SimAgent)`). A 3D scenario brings
its own viewer and does not use this file.
"""
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")   # hide pygame's import banner

import json, math, sys, time
from pathlib import Path

import pygame

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cbf.motion import step_double_integrator_toward_target_cbf


def load_slugs(path):
    """Load a SLUGS FSM JSON, skipping any preamble before the leading '{'."""
    with open(path) as f:
        content = f.read()
    return json.loads(content[content.index('{'):])


COLORS = {
    "bg":     (15,  17,  25),
    "grid":   (28,  31,  42),
    "panel":  (22,  24,  35),
    "border": (75,  80, 100),
    "shelf":  (100, 65,  30),
    "dock":   (45, 105, 155),
    "white":  (255, 255, 255),
    "black":  (0,   0,   0),
    "yellow": (255, 215,  50),
    "green":  (70,  210,  70),
    "red":    (210,  65,  65),
    "blue":   (70,  120, 210),
    "purple": (150,  90, 210),
    "orange": (210, 130,  45),
    "cyan":   (70,  190, 210),
    "gray":   (150, 150, 150),
    "gold":   (255, 195,   0),
    "info":   (110, 150, 245),
    "dim":    (60,   63,  80),
    "zone":   (35,   75, 115),
    "alert":  (160,  40,  40),
    "teal":   (0,   150, 136),
    "success":(40,  160,  80),
}


class Btn:
    """Clickable rectangular control-panel button."""
    def __init__(self, x, y, w, h, text, col):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.col  = col

    def hit(self, pos, clicked):
        return clicked and self.rect.collidepoint(pos)

    def draw(self, surf, font):
        pygame.draw.rect(surf, self.col,          self.rect, border_radius=4)
        pygame.draw.rect(surf, COLORS["border"],  self.rect, 2, border_radius=4)
        t = font.render(self.text, True, COLORS["white"])
        surf.blit(t, t.get_rect(center=self.rect.center))


class BaseRobot:
    """A 2D circle robot driven by a double-integrator CBF toward a target (tx, ty)."""
    def __init__(self, rid, name, x, y, color_key, cbf_task, robot_radius, speed):
        self.id    = rid
        self.name  = name
        self.x     = float(x)
        self.y     = float(y)
        self.hx    = float(x)
        self.hy    = float(y)
        self.tx    = float(x)
        self.ty    = float(y)
        self.vx    = 0.0
        self.vy    = 0.0
        self.color = COLORS[color_key]
        self.status = "IDLE"
        self.action = {}
        self._cbf_task    = cbf_task
        self._robot_radius = robot_radius
        self._base_speed   = speed

    def move(self, dt, speed_scale, obstacles, *, rect_obstacles=(), obstacle_velocities=None,
             nav_target=None):
        """Advance one CBF step toward the target, avoiding the given obstacles."""
        # nav_target (a geometric waypoint) overrides the final goal as the CBF
        # target so the robot routes around obstacles; self.tx/ty stay the true
        # goal for FSM/observe logic.
        tx, ty = nav_target if nav_target is not None else (self.tx, self.ty)
        self.x, self.y, self.vx, self.vy = step_double_integrator_toward_target_cbf(
            self._cbf_task,
            self.x, self.y, self.vx, self.vy,
            tx, ty,
            dt, speed_scale, self._base_speed, self._robot_radius,
            obstacles,
            obstacle_velocities=obstacle_velocities,
            rect_obstacles=rect_obstacles,
            min_dist=1.0,
        )

    def draw(self, surf, font):
        ix, iy = int(self.x), int(self.y)
        pygame.draw.circle(surf, self.color, (ix, iy), self._robot_radius)
        pygame.draw.circle(surf, COLORS["white"], (ix, iy), self._robot_radius, 2)
        lbl = font.render(self.name[0], True, COLORS["black"])
        surf.blit(lbl, lbl.get_rect(center=(ix, iy)))

    def snap_home(self, threshold=4.0):
        """Snap to home and zero velocity if within threshold; return whether snapped."""
        if math.hypot(self.hx - self.x, self.hy - self.y) < threshold:
            self.x, self.y = self.hx, self.hy
            self.tx, self.ty = self.hx, self.hy
            self.vx, self.vy = 0.0, 0.0
            return True
        return False

    def reset(self):
        self.x, self.y = self.hx, self.hy
        self.tx, self.ty = self.hx, self.hy
        self.vx, self.vy = 0.0, 0.0
        self.status = "IDLE"
        self.action = {}


class BaseGame:
    """Shared 2D board: window, grid arena, control panel, and the run loop.

    Subclasses override the _draw_*/update/spawn hooks to render a scenario; a 2D
    SimWorld may mix this in (e.g. `class WarehouseGame(BaseGame, SimWorld)`).
    """
    W       = 1280
    H       = 800
    ARENA_W = 900
    PANEL_X = 900
    MISSION = "contrauto"
    TITLE   = "ContrAuto"
    SUBTITLE = "SLUGS GR(1) Controller"

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption(f"ContrAuto -- {self.MISSION}")
        self.clock = pygame.time.Clock()
        self.fS = pygame.font.Font(None, 18)
        self.fM = pygame.font.Font(None, 22)
        self.fL = pygame.font.Font(None, 28)
        self.fT = pygame.font.Font(None, 40)

        self.robots   = []
        self.selected = None
        self.paused   = False
        self.speed    = 1.0
        self.t0       = time.time()
        self.btns     = {}

    def _init_control_btns(self):
        px = self.PANEL_X + 10
        self.btns["pause"] = Btn(px,     630, 130, 32, "PAUSE",  COLORS["blue"])
        self.btns["fast"]  = Btn(px+140, 630, 130, 32, "FASTER", COLORS["orange"])
        self.btns["reset"] = Btn(px,     668, 130, 32, "RESET",  COLORS["red"])
        self.btns["spawn"] = Btn(px+140, 668, 130, 32, "+SPAWN", COLORS["green"])

    def _init_robot_btns(self, robot_defs):
        px = self.PANEL_X + 10
        for i, r in enumerate(self.robots):
            self.btns[f"s{r.id}"] = Btn(
                px, 100 + i * 38, 270, 30,
                f">> {r.name}", COLORS[robot_defs[i][4]])

    def run(self):
        """Interactive main loop: handle input, update, draw at 60 FPS until quit."""
        while True:
            dt = self.clock.tick(60) / 1000.0
            if not self.handle():
                break
            self.update(dt)
            self.draw()
        pygame.quit()

    def handle(self):
        """Process one frame of mouse/keyboard input; return False to quit."""
        mp = pygame.mouse.get_pos()
        clicked = False
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.MOUSEBUTTONDOWN:
                clicked = True
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if e.key == pygame.K_f:
                    self.speed = 2.0 if self.speed == 1.0 else 1.0
                if e.key == pygame.K_r:
                    self._reset()
                if e.key == pygame.K_s:
                    self.spawn()
                self._handle_key(e.key)
                for idx, r in enumerate(self.robots):
                    if e.key == pygame.K_1 + idx:
                        self.selected = r

        if self.btns.get("pause") and self.btns["pause"].hit(mp, clicked):
            self.paused = not self.paused
        if self.btns.get("fast") and self.btns["fast"].hit(mp, clicked):
            self.speed = 2.0 if self.speed == 1.0 else 1.0
        if self.btns.get("reset") and self.btns["reset"].hit(mp, clicked):
            self._reset()
        if self.btns.get("spawn") and self.btns["spawn"].hit(mp, clicked):
            self.spawn()
        for r in self.robots:
            if self.btns.get(f"s{r.id}") and self.btns[f"s{r.id}"].hit(mp, clicked):
                self.selected = r
        self._handle_click(mp, clicked)
        return True

    def _handle_key(self, key):
        pass

    def _handle_click(self, mp, clicked):
        pass

    def draw(self):
        self.screen.fill(COLORS["bg"])
        self._draw_arena()
        self._draw_objects()
        for r in self.robots:
            r.draw(self.screen, self.fS)
        self._draw_panel()
        pygame.display.flip()

    def _draw_grid(self):
        for x in range(0, self.ARENA_W, 60):
            pygame.draw.line(self.screen, COLORS["grid"], (x, 0), (x, self.H))
        for y in range(0, self.H, 60):
            pygame.draw.line(self.screen, COLORS["grid"], (0, y), (self.ARENA_W, y))

    def _draw_panel_header(self):
        px = self.PANEL_X + 10
        pygame.draw.rect(self.screen, COLORS["panel"],
                         pygame.Rect(self.PANEL_X, 0, self.W - self.PANEL_X, self.H))
        self.screen.blit(self.fT.render(self.TITLE, True, COLORS["cyan"]), (px, 10))
        self.screen.blit(self.fS.render(self.SUBTITLE, True, COLORS["info"]), (px, 50))
        self.screen.blit(self.fS.render(f"Mission: {self.MISSION}", True, COLORS["dim"]), (px, 66))
        self.screen.blit(self.fM.render("-- SELECT ROBOT --", True, COLORS["dim"]), (px, 78))

    def _draw_panel_stats(self, y_start=308, extra_stats=None):
        px = self.PANEL_X + 10
        el = int(time.time() - self.t0)
        self.screen.blit(self.fM.render("-- STATS --", True, COLORS["dim"]), (px, y_start))
        lines = [
            f"Time    {el // 60:02d}:{el % 60:02d}",
            f"Speed   {'2x' if self.speed > 1 else '1x'}",
            f"State   {'PAUSED' if self.paused else 'RUNNING'}",
        ]
        if extra_stats:
            lines = extra_stats + lines
        for i, s in enumerate(lines):
            self.screen.blit(self.fM.render(s, True, COLORS["white"]),
                             (px, y_start + 22 + i * 22))

    def _draw_panel_controls(self, y_start=610):
        px = self.PANEL_X + 10
        self.screen.blit(self.fM.render("-- CONTROLS --", True, COLORS["dim"]), (px, y_start))
        for btn in self.btns.values():
            if not btn.text.startswith(">>"):
                btn.draw(self.screen, self.fS)
        self.screen.blit(self.fM.render("-- KEYS --", True, COLORS["dim"]), (px, y_start + 100))
        for i, h in enumerate(["SPACE pause", "F speed", "S spawn", "R reset", "ESC quit"]):
            self.screen.blit(self.fS.render(h, True, COLORS["dim"]),
                             (px, y_start + 120 + i * 14))

    def _draw_arena(self):
        self._draw_grid()

    def _draw_objects(self):
        pass

    def _draw_panel(self):
        self._draw_panel_header()

    def update(self, dt):
        pass

    def spawn(self):
        pass

    def _reset(self):
        for r in self.robots:
            r.reset()
        self.t0 = time.time()
