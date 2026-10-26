"""Publication figure: three-panel simulator filmstrip (scene + trajectories + req/grant
timeline) -> outputs/sim_filmstrip_v2.pdf.  Each panel is a REAL simulator run with the
robots' trajectories overlaid on a world snapshot and a request/grant timeline below.

  (a) dock (terminal), N=3   : simulator/two_layer_dock.py TwoLayerDock with the
      outputs/two_layer_dock_n3/workdir FSMs (3 worker FSMs + flat pulse arbiter).
      NB the world's init calls the LLM-based deadline/assignment solver
      (contrauto.assign_decomposer.solve), so regeneration needs network + the LLM key.
  (b) corridor (passage), N=3: simulator/corridor.py E2ECorridorWorld with the CERTIFIED
      corridor_e2e7 FSMs run AS-IS (world-driven req; hold-until-release passage
      arbiter; repeated shuttle service -- see experiments/instances/corridor/run.sh).
      The plain CorridorWorld + dock FSMs route is NOT used: the dock3 arbiter emits
      1-step grant PULSES (grant toggles 1,0,1,0 even for a lone requester), and the
      corridor's hold-at-entrance gate then keeps every robot pinned at the corridor
      mouth forever (0 crossings at any seed; cf. the July-09 outputs/instances/
      corridor/corridor.gif, frozen at the mouth at t=266s).  Hold-until-release --
      the property panel (b) is about -- only exists in the e2e passage arbiter.
  (c) charger (terminal, physical trigger), N=3: simulator/charger.py ChargerWorld with
      the certified outputs/charger_n3/workdir FSMs and n_recharges=1 -- the SAME config
      the repo's own evaluation harness runs (experiments/env_transfer.py run_charger;
      done=true / 0 violations documented for seeds 1..5 in env_transfer_results.json).
      With n_recharges>=2 the certified service is one-shot in practice (the shipped
      case_studies charger GIFs end at 3 total recharges after 20000 steps), so one
      full recharge per robot IS the complete certified run.  Charger events are 1-4
      sim steps long (arrive -> req -> grant -> ONE-step charge, mirroring the
      contract's one-step recovery `charging -> X !battery_low`), so panel (c)'s
      bars get a small minimum drawn width (positions stay time-accurate).

Timeline semantics (per robot row):
  light band  = request high   (a/c: the robot FSM's request output bit, read from the
                                controller's current node via FsmController.requests();
                                b: the world-driven demand r.req -- the same signal the
                                arbiter consumes)
  solid bar   = grant high     (world.last_grants)
  dark tick   = (b) robot inside the single-lane corridor (_in_corridor)
                (c) robot charging (r._charging)

Panels + layout are selectable with --panels (default "a,b" -> the paper's SINGLE-COLUMN
two-panel figure, (a) above (b), stacked so the 100 s corridor timeline keeps its width;
"a,b,c" reproduces the earlier full-page three-panel version side by side).

Run:  python experiments/sim_filmstrip.py [--panels a,b] [--out outputs/sim_filmstrip_v2.pdf]
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"          # offscreen pygame, BEFORE any display init
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import sys
import argparse
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

DT = 1 / 60.0
HUD_CROP = 44                 # px: strip the baked-in HUD text band at the top of a frame
SCENE_ASPECT = 1.6            # common displayed w/h for ALL scenes -> identical panel dims

# Font sizes per target column width.  "wide" = the multi-column full-page figure;
# "single" = a single IEEE column (~3.4 in): everything shrinks toward the ~6 pt floor.
FONTS = {
    "wide":   dict(base=8, title=8, anno=6.5, tick=7, ylab=7, xlab=7),
    "single": dict(base=6, title=6.5, anno=6, tick=6, ylab=6, xlab=6),
}


# ── run + log ─────────────────────────────────────────────────────────────────
def _grab(world):
    """One offscreen pygame frame as an (H, W, 3) uint8 array."""
    from simulator.two_layer_warehouse import _tobytes
    world.draw()
    wpx, hpx = world.screen.get_size()
    return np.frombuffer(_tobytes(world.screen, "RGB"), np.uint8).reshape(hpx, wpx, 3).copy()


def run_dock(fsm_dir, seed=10, max_steps=8000, dt=DT):
    """Panel (a): TwoLayerDock N=3 (the __main__ --n3 config, but with a seed picked
    from a visual sweep of the random shelf layouts: seed 10 gives an evenly spread,
    uncluttered arrangement with the dock clearly visible), logging per step:
    time, per-robot (x, y), request bit, grant bit; plus offscreen frames."""
    from simulator.two_layer_dock import TwoLayerDock, Config
    cfg = Config("two_layer_dock3", n_robots=3, n_demands=6, equiv=2, seed=seed,
                 deadline=55.0, dock_hw=50.0, dock_hh=36.0)
    w = TwoLayerDock(cfg, fsm_dir, verbose=False)
    if not w.report["assignment"]:
        raise SystemExit("dock: PHYSICAL layer UNSATISFIABLE (LLM assignment failed)")
    n = len(w.robots)
    log = dict(t=[], pos=[[] for _ in range(n)], req=[[] for _ in range(n)],
               grant=[[] for _ in range(n)], tick=None)
    frames, every = [], 200            # adaptively thinned frame ring (memory bound)
    steps_done = 0
    for i in range(max_steps):
        w.update(dt)                   # fsm_step: observe -> arbiter -> FSM -> act -> tick
        w.step(dt)                     # (the module's own run() loop pattern)
        log["t"].append(w.t)
        for k, r in enumerate(w.robots):
            log["pos"][k].append((r.x, r.y))
            log["req"][k].append(int(any(w.controllers[r.name].requests(r.REQUESTS).values())))
            log["grant"][k].append(int(any(w.last_grants.get(g, 0) for g in r.GRANTS)))
        if i % every == 0:
            frames.append((i, _grab(w)))
            if len(frames) > 24:       # thin: keep memory ~constant on long runs
                frames = frames[::2]
                every *= 2
        steps_done = i + 1
        if w.all_done():
            break
    tgt = int(steps_done * 2 / 3)      # snapshot at ~2/3 of the run
    frame = min(frames, key=lambda f: abs(f[0] - tgt))[1]
    met = w.metrics()
    lay = w.lay
    return dict(log=log, frame=frame, metrics=met, steps=steps_done,
                colors=[tuple(c / 255 for c in r.color) for r in w.robots],
                landmarks=[(lay.dock[0] - lay.dock_hw, lay.dock[1] - lay.dock_hh),
                           (lay.dock[0] + lay.dock_hw, lay.dock[1] + lay.dock_hh)])


def run_corridor(fsm_dir, seed=3, steps=6000, dt=DT):
    """Panel (b): E2ECorridorWorld on the certified corridor_e2e7 FSMs (run AS-IS),
    logging per step: time, per-robot (x, y), world req, grant, inside-corridor flag."""
    from simulator.corridor import E2ECorridorWorld, Config, _in_corridor
    from contrauto.sim_iface import fsm_step
    w = E2ECorridorWorld(Config(n_robots=3, seed=seed), fsm_dir)
    arb = w.arbiter()
    n = len(w.robots)
    log = dict(t=[], pos=[[] for _ in range(n)], req=[[] for _ in range(n)],
               grant=[[] for _ in range(n)], tick=[[] for _ in range(n)])
    frame = None
    for i in range(steps):
        fsm_step(w, w.controllers, dt, arb)   # the module's own run_e2e() loop pattern
        log["t"].append(w.t)
        for k, r in enumerate(w.robots):
            log["pos"][k].append((r.x, r.y))
            log["req"][k].append(int(r.req))
            log["grant"][k].append(int(w.last_grants.get(f"grant_r{r.id}", 0)))
            log["tick"][k].append(int(_in_corridor(r, w)))
        if i == int(steps * 2 / 3):           # snapshot at 2/3 of the run
            frame = _grab(w)
    met = w.metrics()
    lay = w.lay
    x0, y0, x1, y1 = lay.corridor
    return dict(log=log, frame=frame, metrics=met, steps=steps,
                colors=[tuple(c / 255 for c in r.color) for r in w.robots],
                landmarks=[(x0, y0), (x1, y1), lay.pickup_a, lay.deliver_b])


def run_charger(fsm_dir, seed=3, max_steps=12000, dt=DT):
    """Panel (c): ChargerWorld on the certified charger FSMs, n_recharges=1 (the
    evaluation-harness config -- see the module docstring), run to all_done().
    Logs per step: time, per-robot (x, y), request output bit, grant bit, charging
    flag (r._charging -- the world's own capacity-1 mutex monitor signal)."""
    from simulator.charger import ChargerWorld, Config
    from contrauto.sim_iface import fsm_step
    w = ChargerWorld(Config(n_robots=3, seed=seed, n_recharges=1), fsm_dir)
    arb = w.arbiter()
    n = len(w.robots)
    log = dict(t=[], pos=[[] for _ in range(n)], req=[[] for _ in range(n)],
               grant=[[] for _ in range(n)], tick=[[] for _ in range(n)])
    frames, every = [], 40             # short run: capture often, thin adaptively
    steps_done = 0
    for i in range(max_steps):
        fsm_step(w, w.controllers, dt, arb)   # the module's own run() loop pattern
        log["t"].append(w.t)
        for k, r in enumerate(w.robots):
            log["pos"][k].append((r.x, r.y))
            log["req"][k].append(int(any(w.controllers[r.name].requests(r.REQUESTS).values())))
            log["grant"][k].append(int(any(w.last_grants.get(g, 0) for g in r.GRANTS)))
            log["tick"][k].append(int(r._charging))
        if i % every == 0:
            frames.append((i, _grab(w)))
            if len(frames) > 24:
                frames = frames[::2]
                every *= 2
        steps_done = i + 1
        if w.all_done():
            break
    tgt = int(steps_done * 2 / 3)
    frame = min(frames, key=lambda f: abs(f[0] - tgt))[1]
    lay = w.lay
    # blank the baked "<zap> charger" caption (SysFont renders the zap glyph as a tofu
    # box); the figure adds its own serif "charging station" label at render time
    cx, cy = int(lay.charger[0]), int(lay.charger[1])
    frame[max(0, cy + 28):cy + 56, max(0, cx - 38):cx + 96] = frame[-3, 3]
    met = dict(w.metrics())
    met["recharges_per_robot"] = [r.n_charged for r in w.robots]
    return dict(log=log, frame=frame, metrics=met, steps=steps_done,
                colors=[tuple(c / 255 for c in r.color) for r in w.robots],
                landmarks=[lay.charger] + list(lay.rest))


# ── rendering ─────────────────────────────────────────────────────────────────
def _spans(flags, t, dt=DT):
    """Merge consecutive high steps into (t_start, duration) spans for broken_barh."""
    out, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = t[i] - dt          # the step covers (t-dt, t]
        elif not v and start is not None:
            out.append((start, t[i - 1] - start))
            start = None
    if start is not None:
        out.append((start, t[-1] - start))
    return out


def _crop_box(panel, aspect=SCENE_ASPECT):
    """Crop rectangle (x0, x1, y0, y1) around trajectories + landmarks, padded and
    forced to EXACTLY the target aspect (the same for both panels, so the two scenes
    render at identical width x height).  The box may extend beyond the frame; the
    overhang is letterboxed with the scene's own background colour (axes facecolor).
    y0 stays below the baked HUD band whenever the content fits."""
    h, wpx = panel["frame"].shape[:2]
    pts = np.concatenate([np.asarray(p) for p in panel["log"]["pos"]]
                         + [np.asarray(panel["landmarks"], float)])
    m = 42.0
    x0, x1 = max(0, pts[:, 0].min() - m), min(wpx, pts[:, 0].max() + m)
    y0, y1 = max(HUD_CROP, pts[:, 1].min() - m), min(h, pts[:, 1].max() + m)
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < aspect:                    # widen (centered); shift inside if it fits
        pad = (aspect * ch - cw) / 2
        x0, x1 = x0 - pad, x1 + pad
        if x1 - x0 <= wpx:
            if x0 < 0:
                x1, x0 = x1 - x0, 0
            if x1 > wpx:
                x0, x1 = x0 - (x1 - wpx), wpx
    else:                                   # heighten (centered); keep HUD out if possible
        pad = (cw / aspect - ch) / 2
        y0, y1 = y0 - pad, y1 + pad
        if y1 - y0 <= h - HUD_CROP:
            if y0 < HUD_CROP:
                y1, y0 = y1 + (HUD_CROP - y0), HUD_CROP
            if y1 > h:
                y0, y1 = y0 - (y1 - h), h
    return x0, x1, y0, y1


def draw_scene(ax, panel, title, fonts=None):
    fonts = fonts or FONTS["wide"]
    frame = panel["frame"].copy()
    h, wpx = frame.shape[:2]
    bg = frame[h - 3, 3]                    # plain background pixel (bottom-left corner)
    frame[:HUD_CROP] = bg                   # blank the baked HUD text band
    ax.set_facecolor(tuple(bg / 255))       # letterbox colour for out-of-frame overhang
    x0, x1, y0, y1 = _crop_box(panel)
    ax.imshow(frame, extent=(0, wpx, h, 0), interpolation="bilinear", zorder=0)
    for k, traj in enumerate(panel["log"]["pos"]):
        pts = np.asarray(traj)[::3]
        if len(pts) < 3:
            continue
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        alphas = np.linspace(0.15, 0.9, len(segs))
        rgba = np.tile((*panel["colors"][k], 1.0), (len(segs), 1))
        rgba[:, 3] = alphas
        ax.add_collection(LineCollection(segs, colors=rgba, linewidths=1.3,
                                         capstyle="round", zorder=3))
        ax.plot(*pts[0], "o", ms=2.6, mfc=panel["colors"][k], mec="white",
                mew=0.5, zorder=4)
        tail = pts[-min(4, len(pts) - 1)]
        ax.add_patch(FancyArrowPatch(tail, pts[-1], arrowstyle="-|>", mutation_scale=10,
                                     color=panel["colors"][k], lw=0, zorder=4))
    for txt, (ax_, ay_) in panel.get("anno", []):
        ax.text(ax_, ay_, txt, color="white", fontsize=fonts["anno"], ha="center",
                va="center", zorder=5, bbox=dict(boxstyle="round,pad=0.25", fc="black",
                                                  alpha=0.55, ec="none"))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)                 # image coords: y down
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=fonts["title"], pad=3)


def draw_timeline(ax, panel, xlabel, min_event=0.0, fonts=None):
    """min_event: minimum DRAWN duration (s) for req/grant spans -- panel (c)'s events
    are 1-4 sim steps (0.02-0.07 s) and would be sub-pixel otherwise; positions are
    kept time-accurate (span is widened symmetrically around its true extent)."""
    fonts = fonts or FONTS["wide"]
    log = panel["log"]
    t = np.asarray(log["t"])
    n = len(log["pos"])

    def _draw(flags, y, hgt, alpha, c):
        for t0, dur in _spans(flags, t):
            if dur < min_event:
                t0, dur = t0 - (min_event - dur) / 2, min_event
            ax.broken_barh([(t0, dur)], (y, hgt), facecolors=c, alpha=alpha, linewidth=0)

    for k in range(n):
        c = panel["colors"][k]
        _draw(log["req"][k], k - 0.36, 0.72, 0.22, c)     # light band: request high
        _draw(log["grant"][k], k - 0.18, 0.36, 0.95, c)   # solid bar: grant high
        if log["tick"] is not None:                       # dark tick: inside / charging
            tin = t[np.asarray(log["tick"][k], bool)]
            if len(tin):
                ax.vlines(tin, k + 0.20, k + 0.36, colors="0.1", lw=0.7, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"R{k+1}" for k in range(n)], fontsize=fonts["ylab"])
    ax.set_ylim(n - 0.5 + 0.15, -0.65)                    # R1 on top
    ax.set_xlim(0, t[-1] + max(min_event, 0.01 * t[-1]))  # margin: last event not clipped
    ax.tick_params(axis="x", labelsize=fonts["tick"], length=2, pad=1.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, fontsize=fonts["xlab"], labelpad=1.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)


def check(panel, name, tick_desc=None):
    """Sanity checks on the logged data (printed; figure is only as good as these).
    If the panel has a tick signal, also count tick steps and how many fall inside
    that robot's grant (must be ALL of them for (b) inside-corridor / (c) charging)."""
    log = panel["log"]
    g = np.asarray([log["grant"][k] for k in range(len(log["grant"]))])
    overlap = int((g.sum(axis=0) > 1).sum())
    print(f"[{name}] steps with >1 simultaneous grant: {overlap}")
    if log["tick"] is not None and tick_desc:
        total, in_grant = 0, 0
        for k in range(g.shape[0]):
            tk = np.asarray(log["tick"][k], bool)
            total += int(tk.sum())
            in_grant += int((tk & g[k].astype(bool)).sum())
        print(f"[{name}] {tick_desc} ticks: {total}, of which inside a grant: {in_grant}")
    print(f"[{name}] metrics: {panel['metrics']}")
    return overlap


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=3, help="corridor world seed (panel b)")
    ap.add_argument("--dock-seed", type=int, default=10,
                    help="dock world seed (panel a; drives the random shelf layout)")
    ap.add_argument("--charger-seed", type=int, default=3,
                    help="charger world seed (panel c; station + rest-point layout)")
    ap.add_argument("--steps", type=int, default=6000, help="corridor steps (panel b)")
    ap.add_argument("--dock-fsm-dir", default=ROOT + "/outputs/two_layer_dock_n3/workdir")
    ap.add_argument("--cor-fsm-dir", default=ROOT + "/outputs/corridor_e2e7/workdir")
    ap.add_argument("--charger-fsm-dir", default=ROOT + "/outputs/charger_n3/workdir")
    ap.add_argument("--out", default=ROOT + "/outputs/sim_filmstrip_v2.pdf")
    ap.add_argument("--panels", default="a,b",
                    help='which panels to draw, in order (default "a,b" -> the '
                         'single-column stacked figure; "a,b,c" -> the full-page '
                         'three-panel version)')
    ap.add_argument("--layout", choices=["auto", "stacked", "row"], default="auto",
                    help='auto: stacked (a over b) for <=2 panels, side-by-side for 3')
    ap.add_argument("--cache", default=None,
                    help="pickle path: reuse logged runs if it exists, else run + save "
                         "(dev convenience; the dock run costs an LLM call)")
    a = ap.parse_args()

    want = [p.strip().lower() for p in a.panels.split(",") if p.strip()]
    if not set(want) <= {"a", "b", "c"} or not want:
        raise SystemExit(f"--panels must be a comma list from a,b,c; got {a.panels!r}")

    # run only what's needed (each entry: builder + args), cache keyed by panel set
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as f:
            built = pickle.load(f)
        print(f"loaded cached runs from {a.cache}")
    else:
        built = {}
        if "a" in want:
            print(">>> panel (a): two_layer_dock N=3, seed", a.dock_seed, "...")
            built["a"] = run_dock(a.dock_fsm_dir, seed=a.dock_seed)
        if "b" in want:
            print(">>> panel (b): corridor e2e7 N=3, seed", a.seed, "...")
            built["b"] = run_corridor(a.cor_fsm_dir, seed=a.seed, steps=a.steps)
        if "c" in want:
            print(">>> panel (c): charger N=3, seed", a.charger_seed, "...")
            built["c"] = run_charger(a.charger_fsm_dir, seed=a.charger_seed)
        if a.cache:
            with open(a.cache, "wb") as f:
                pickle.dump(built, f)
            print(f"cached runs -> {a.cache}")

    # panel registry: (world dict, title, timeline x-label, tick description, min_event)
    SPEC = {
        "a": ("(a) dock (terminal), N=3, flat arbiter",
              "time (s) — req light, grant solid", None, 0.0, "dock"),
        "b": ("(b) corridor (passage), N=3, hold-until-release",
              "time (s) — req light, grant solid; dark tick: inside corridor",
              "inside-corridor", 0.0, "corridor"),
        "c": ("(c) charger (terminal, physical trigger), N=3",
              "time (s) — req light, grant solid; dark tick: charging",
              "charging", 0.12, "charger"),
    }
    for p in want:
        check(built[p], f"{p}:{SPEC[p][4]}", tick_desc=SPEC[p][2])

    # annotations are render-time (so a cached run can be re-styled): the dock frame
    # already carries a baked "DOCK (cap 1, held n)" label -> no extra label in (a);
    # (b) labels the single-lane corridor just above its rectangle (landmarks[0..1]);
    # (c) labels the station (landmarks[0]).
    if "a" in built:
        built["a"]["anno"] = []
    if "b" in built:
        (cx0, cy0), (cx1, _cy1) = built["b"]["landmarks"][0], built["b"]["landmarks"][1]
        built["b"]["anno"] = [("single-lane corridor", ((cx0 + cx1) / 2, cy0 - 36))]
    if "c" in built:
        st = built["c"]["landmarks"][0]
        built["c"]["anno"] = [("charging station (cap 1)", (st[0], st[1] + 60))]

    layout = a.layout
    if layout == "auto":
        layout = "row" if len(want) >= 3 else "stacked"
    single_col = layout == "stacked"
    fonts = FONTS["single"] if single_col else FONTS["wide"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": fonts["base"],
        "pdf.fonttype": 42,
    })

    m = len(want)
    if layout == "stacked":
        # single IEEE column: panels stacked, each a scene row over a timeline row,
        # every scene spanning the full ~3.4 in column width.
        fig = plt.figure(figsize=(3.4, 2.15 * m), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.02, hspace=0.05)
        gs = fig.add_gridspec(2 * m, 1, height_ratios=[1.55, 1.0] * m)
        scene_ax = [fig.add_subplot(gs[2 * i, 0]) for i in range(m)]
        tl_ax = [fig.add_subplot(gs[2 * i + 1, 0]) for i in range(m)]
    else:
        fig = plt.figure(figsize=(3.5 * m, 3.2), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.02)
        gs = fig.add_gridspec(2, m, height_ratios=[1.55, 1.0])
        scene_ax = [fig.add_subplot(gs[0, j]) for j in range(m)]
        tl_ax = [fig.add_subplot(gs[1, j]) for j in range(m)]

    for i, p in enumerate(want):
        title, xlabel, _tick, min_event, _name = SPEC[p]
        draw_scene(scene_ax[i], built[p], title, fonts=fonts)
        draw_timeline(tl_ax[i], built[p], xlabel, min_event=min_event, fonts=fonts)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"wrote {a.out}  (panels {','.join(want)}, layout {layout}, "
          f"{fig.get_size_inches()[0]:.2f}x{fig.get_size_inches()[1]:.2f} in)")


if __name__ == "__main__":
    main()
