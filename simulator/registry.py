"""Task -> physical-layer simulator adapters.

Each adapter takes the task layer's output dir (`<outdir>/workdir`, holding the synthesized
FSM json files) and runs the matching simulator, returning a metrics dict. This is the seam
that lets the orchestrator (main.py) hand a task's synthesized FSMs to the physical layer
without the two layers knowing each other's internals — the binding is naming-agnostic
(role_of / bind_by_role), so the same simulator reuses whatever names the synthesis chose.
"""
import os


def _gif_path(d):
    return os.path.join(os.path.dirname(d), "physical_simulation.gif")


def _render_game(factory, gif, steps, monitor=None):
    """Capture a BaseGame-backed finite validation without opening a live window."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from simulator.gif_utils import render_fsm_gif
    return render_fsm_gif(factory(), gif, steps, 1 / 60.0, assumption_monitor=monitor)


def _monitor(d):
    """Build the channel-5 assumption monitor from the decomposition's specs.ltl in workdir `d`
    (watches each contract's X-free assumptions). None if the contract or module is unavailable."""
    try:
        from contrauto.assumption_monitor import monitor_from_ltl
        ltl = os.path.join(d, "specs.ltl")
        if os.path.exists(ltl):
            return monitor_from_ltl(open(ltl).read())
    except Exception:
        pass
    return None


def _patrol(d, model=None, visualize=False):
    if visualize:
        from simulator.patrol import PatrolGame, HOME_XY
        return _render_game(lambda: PatrolGame(
            [(1, "P1", HOME_XY[0], HOME_XY[1], "blue", f"{d}/patrolrobot.json")]), _gif_path(d), 6000)
    from simulator.patrol import headless
    return headless(f"{d}/patrolrobot.json")


def _warehouse(d, model=None, visualize=False, n_robots=2):
    if n_robots != 2:
        return _warehouse_n(n_robots)(d, model=model, visualize=visualize)
    if visualize:
        from simulator.warehouse import WarehouseGame, SPAWN_Y, _spawn_row_x
        xs = _spawn_row_x(2)
        defs = [(1, "Robot1", xs[0], SPAWN_Y, "red", f"{d}/robot1.json"),
                (2, "Robot2", xs[1], SPAWN_Y, "blue", f"{d}/robot2.json")]
        return _render_game(lambda: WarehouseGame(defs, [f"{d}/arbiter.json"]),
                            _gif_path(d), 8000, _monitor(d))
    from simulator.warehouse import headless
    return headless(f"{d}/robot1.json", f"{d}/robot2.json", f"{d}/arbiter.json",
                    assumption_monitor=_monitor(d), visualize=visualize)


def _warehouse_n(n):
    """Build an adapter for an n-robot warehouse task (closes over n)."""
    def run_n(d, model=None, visualize=False):
        if visualize:
            from simulator.warehouse import WarehouseGame, SPAWN_Y, _spawn_frames_for, _spawn_row_x
            colors = ["red", "blue", "green", "purple", "orange", "cyan", "gold", "teal"]
            xs = _spawn_row_x(n)
            defs = [(i + 1, f"Robot{i + 1}", xs[i], SPAWN_Y,
                     colors[i % len(colors)], f"{d}/robot{i + 1}.json") for i in range(n)]
            return _render_game(lambda: WarehouseGame(
                                    defs, [f"{d}/arbiter.json"],
                                    max_active_bins=max(4, 2 * n), spawn_frames=_spawn_frames_for(n)),
                                _gif_path(d), 8000, _monitor(d))
        from simulator.warehouse import headless_n
        paths = [f"{d}/robot{i}.json" for i in range(1, n + 1)]
        return headless_n(paths, f"{d}/arbiter.json",
                          assumption_monitor=_monitor(d), visualize=visualize)
    return run_n


def _static(d, model=None, visualize=False):
    if visualize:
        from simulator.static_2r_3b import StaticDeliveryGame, _defs, CBF_TASK
        return _render_game(lambda: StaticDeliveryGame(_defs(f"{d}/robot1.json", f"{d}/robot2.json"), CBF_TASK),
                            _gif_path(d), 8000, _monitor(d))
    from simulator.static_2r_3b import headless
    return headless(f"{d}/robot1.json", f"{d}/robot2.json", assumption_monitor=_monitor(d))


def _dynamic(d, model=None, visualize=False):
    if visualize:
        from simulator.dynamic_2r_3b import DeliveryGame, _defs, CBF_TASK
        return _render_game(lambda: DeliveryGame(_defs(f"{d}/robot1.json", f"{d}/robot2.json"), CBF_TASK,
                                                   f"{d}/dispatcher.json"), _gif_path(d), 8000, _monitor(d))
    from simulator.dynamic_2r_3b import headless
    return headless(f"{d}/robot1.json", f"{d}/robot2.json", f"{d}/dispatcher.json",
                    assumption_monitor=_monitor(d))


def _dock(default_n):
    """Build a two_layer_dock adapter for n robots (closes over the per-n Config)."""
    def run_dock(d, model=None, visualize=False, n_robots=None):
        from simulator.two_layer_dock import run, Config
        n = n_robots or default_n
        name = "two_layer_dock" + ("" if n == 2 else str(n))
        if visualize:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        w = run(Config(name, n_robots=n, n_demands=4 if n >= 3 else 6, equiv=2, seed=3,
                       deadline=120.0, dock_hw=58.0, dock_hh=40.0), fsm_dir=d,
            model=model or "gemini", gif=_gif_path(d) if visualize else None)
        return w.metrics()
    return run_dock


def _charger(d, model=None, visualize=False, n_robots=3):
    from simulator.charger import run, Config
    w = run(Config(n_robots=n_robots), fsm_dir=d, visualize=False,
            gif=_gif_path(d) if visualize else None)
    return w.metrics()


def _corridor(d, model=None, visualize=False, n_robots=3):
    from simulator.corridor import run_e2e, Config
    if visualize:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    w = run_e2e(Config(n_robots=n_robots), fsm_dir=d, gif=_gif_path(d) if visualize else None)
    return w.metrics()


# NOTE on "corridor": simulator/corridor.py has TWO physical-layer harnesses; `_corridor`
# below binds the one that actually drives corridor.yaml's own synthesis output
# (outputs/corridor_n<N>/workdir, over corridor.yaml's OWN vocabulary at_b/task_arrived/
# crossing/req/grant -- `at_a` was removed, see corridor.yaml's own note on why):
#   - E2ECorridorWorld/run_e2e(): what `_corridor` calls, with `fsm_dir` pointed at the
#     task's own outdir -- this DOES drive corridor.yaml's own FSM correctly.
#   - CorridorWorld/run(): a separate, OLDER harness whose PRED/ACT are the DOCK's
#     vocabulary (at_shelf/at_dock/pickup/dropoff) -- built to reuse a two_layer_dock FSM
#     physically as a corridor, not to drive corridor.yaml's own FSM. Not registered here;
#     only reachable via `python simulator/corridor.py` directly.
# The grounding vocabulary lookup below (_VOCAB_MODULES) still reads this module's
# top-level PREDICATES/ACTIONS, which belong to the OLDER dock-vocabulary harness, not
# E2ERobot's real corridor.yaml vocabulary -- a still-open gap (a correct grounding check
# needs its own PRED/ACT keyed on E2ERobot's actual sensors).


# task name -> (adapter, FSM file names it expects in the workdir)
SIMULATORS = {
    "patrol":          (_patrol,      ["patrolrobot.json"]),
    "warehouse":       (_warehouse,   ["robot1.json", "robot2.json", "arbiter.json"]),
    "warehouse5":      (_warehouse_n(5), ["robot1.json", "robot2.json", "robot3.json", "robot4.json", "robot5.json", "arbiter.json"]),
    "corridor":        (_corridor,    ["robot1.json", "robot2.json", "robot3.json", "arbiter.json"]),
    "static_2r_3b":    (_static,      ["robot1.json", "robot2.json"]),
    "dynamic_2r_3b":   (_dynamic,     ["robot1.json", "robot2.json", "dispatcher.json"]),
    "two_layer_dock":  (_dock(2),     ["robot1.json", "robot2.json", "arbiter.json"]),
    "charger":         (_charger,     ["robot1.json", "robot2.json", "robot3.json", "arbiter.json"]),
}
#  corridor charger warehouse patrol static_2r_3b dynamic_2r_3b


# task name -> the simulator module that declares its grounding vocabulary (atomic
# propositions / predicates + actions). One source of truth per simulator: the LTL the
# decomposer writes must be over exactly these APs, else bind_by_role/role_of cannot map a
# contract variable to a physical grounding and the FSM silently deadlocks in execution.
_VOCAB_MODULES = {
    "patrol":        "simulator.patrol",
    "warehouse":     "simulator.warehouse",
    "warehouse5":    "simulator.warehouse",
    "corridor":      "simulator.corridor",
    "static_2r_3b":  "simulator.static_2r_3b",
    "dynamic_2r_3b": "simulator.dynamic_2r_3b",
    "two_layer_dock":  "simulator.two_layer_dock",
    "charger":        "simulator.charger"
}


def grounding_vocab(task: str):
    """Return {'sensors': [...], 'actions': [...]} -- the simulator's atomic propositions
    (predicate grounding names) and action names for `task`, or None if the task has no
    registered simulator vocabulary. This is the vocabulary the decomposer must write its
    contracts over so every variable binds to a physical grounding (see role_of/bind_by_role).
    """
    mod_name = _VOCAB_MODULES.get(task)
    if not mod_name:
        return None
    import importlib
    try:
        mod = importlib.import_module(mod_name)
        preds = getattr(mod, "PREDICATES", None)
        acts = getattr(mod, "ACTIONS", None)
        if preds is None or acts is None:
            return None
        out = {"sensors": sorted(preds.keys()), "actions": sorted(acts.keys()),
               "source": getattr(mod, "__file__", None)}
        for attr, key in (("SIGNALS", "signals"), ("ACTION_EFFECTS", "action_effects"),
                          ("FRAME", "frame"), ("GLOBAL_SENSORS", "globals")):
            value = getattr(mod, attr, None)
            if value:
                out[key] = value
        notes = getattr(mod, "VOCAB_NOTES", None)
        if notes:
            out["notes"] = notes
        return out
    except Exception:
        return None


def grounding_binder(task: str):
    """Optional per-task grounding predicate: the simulator family's OWN variable binder
    (e.g. box_delivery._classify via `grounds`), so the design-time vocabulary check judges
    bindability with the exact function the runtime uses — no drift between check and
    binding. Returns a callable var->bool, or None (caller falls back to vocab + role_of)."""
    mod_name = _VOCAB_MODULES.get(task)
    if not mod_name:
        return None
    import importlib
    try:
        return getattr(importlib.import_module(mod_name), "grounds", None)
    except Exception:
        return None


def fsm_files(task: str, n_robots: int | None = None) -> list[str]:
    """Return the FSM files required by ``task``, expanding scalable worker families."""
    if n_robots is not None and task in {"charger", "corridor", "warehouse", "two_layer_dock"}:
        # warehouse has a single Arbiter (the dock -- the only real capacity-limited
        # resource; the shelf needs no arbiter, see contrauto/tasks/warehouse.yaml),
        # same as every other scalable task.
        return [f"robot{index}.json" for index in range(1, n_robots + 1)] + ["arbiter.json"]
    return SIMULATORS.get(task, (None, []))[1]


def has_fsm(task: str, fsm_dir: str, n_robots: int | None = None) -> bool:
    """True if the task layer's FSMs already exist in fsm_dir (so synthesis can be reused)."""
    if not os.path.isdir(fsm_dir):
        return False
    needed = fsm_files(task, n_robots)
    if needed:
        return all(os.path.exists(os.path.join(fsm_dir, f)) for f in needed)
    return any(f.endswith(".json") for f in os.listdir(fsm_dir))
