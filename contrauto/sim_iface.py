"""
Simulator-agnostic interface for driving synthesized FSMs (the runtime port).

SimAgent and SimWorld below are THE one simulator base: every scenario, current or future,
is a subclass. The base is rendering- and dimension-agnostic. An agent's only required
primitives are pose() (a position; 2D or 3D, the base does not care) and tick() (actuate
toward self.target). It declares SENSORS (atomic propositions) and ACTIONS (Nav/Stationary);
the FSM driver (FsmController, run_fsm) depends only on these ABCs, never on a concrete
simulator. Swap simulators (2D pygame, 3D Gazebo/Isaac) by writing a different subclass;
the driver and the scenario logic are unchanged.

Rendering is NOT part of the base. A 2D scenario may mix in the pygame helpers in
simulator/core.py for an interactive board; a 3D scenario brings its own viewer.

The contract-respecting rules (a non-navigation action HOLDS position so location
APs persist) live ONCE in SimAgent.act().
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ── scenario action declarations ─────────────────────────────────────────────
@dataclass
class Nav:
    """A navigation action: move toward target(agent, world) -> (x, y).

    Optional `done_when(agent, world) -> bool`: while it is False the driver KEEPS this
    action and does NOT advance the FSM (action-holding) — the discrete one-step action
    is held until the continuous world finishes it (e.g. arrived). Default None == the
    legacy behaviour (advance every step; the contract's persistence assumptions hold it).
    """
    target: Callable
    done_when: Optional[Callable] = None


@dataclass
class Stationary:
    """A non-moving action: HOLD position; optionally run effect(agent, world).

    Optional `done_when(agent, world) -> bool`: hold this action until it is True (e.g. an
    inspection finished). Default None == legacy (one step). See Nav.done_when.
    """
    effect: Optional[Callable] = None
    done_when: Optional[Callable] = None


# ── FSM executor (name-based; no keyword role guessing) ───────────────────────
class FsmController:
    """Execute a synthesized SLUGS strategy (load_slugs json) directly.

    Matches the FSM's OWN variable names against the agent's observe()/action dicts,
    given the input/output variable partition (the agent knows it from its SENSORS/
    GRANTS = inputs and ACTIONS/REQUESTS = outputs). Moore execution: transition to
    the successor whose INPUT bits match the observed env, then emit the current
    node's OUTPUT bits.
    """

    def __init__(self, data: dict, in_vars, out_vars):
        self.nodes = data["nodes"]
        self.idx = {v: i for i, v in enumerate(data["variables"])}
        self.in_vars = [v for v in in_vars if v in self.idx]
        self.out_vars = [v for v in out_vars if v in self.idx]
        self.current = "0"
        self._stuck = 0

    def _outputs(self, nid) -> Dict[str, int]:
        st = self.nodes[str(nid)]["state"]
        return {v: st[self.idx[v]] for v in self.out_vars}

    def requests(self, names) -> Dict[str, int]:
        """Request OUTPUT bits of the current node (the arbiter needs these first)."""
        o = self._outputs(self.current)
        return {v: o.get(v, 0) for v in names if v in self.idx}

    def _inputs_match(self, nid, env: Dict[str, int]) -> bool:
        st = self.nodes[str(nid)]["state"]
        return all(st[self.idx[v]] == int(bool(env.get(v, 0))) for v in self.in_vars)

    def _progress_score(self, nid) -> tuple:
        """Prefer a successor that actually acts (grant / nav / charge) over a lazy idle
        edge that is also input-legal. Rank is the SLUGS justice index — used only as a
        tie-break so we stay on the current goal when scores match."""
        st = self.nodes[str(nid)]["state"]
        acts = sum(st[self.idx[v]] for v in self.out_vars)
        rank = int(self.nodes[str(nid)].get("rank", 0) or 0)
        return (acts, rank)

    def _pick(self, candidates) -> Optional[str]:
        best, score = None, None
        for n in candidates:
            nid = str(n)
            sc = self._progress_score(nid)
            if score is None or sc > score:
                best, score = nid, sc
        return best

    def step(self, env: Dict[str, int]) -> Dict[str, int]:
        matched = [n for n in self.nodes[self.current]["trans"] if self._inputs_match(n, env)]
        nxt = self._pick(matched)
        if nxt is None:
            # Off-model env (e.g. CBF shoved the body off a pad the node still thinks
            # it occupies): re-sync to any node whose INPUT bits match, don't freeze.
            nxt = self._pick(n for n in self.nodes if self._inputs_match(n, env))
        if nxt is not None:
            self.current, self._stuck = nxt, 0
        else:
            self._stuck += 1                      # nothing matches -> hold node (fallback)
        return self._outputs(self.current)

    def reset(self):
        self.current = "0"
        self._stuck = 0


def load_fsm(path: str) -> dict:
    """Parse a SLUGS --jsonOutput strategy (skips the non-JSON banner prefix)."""
    c = open(path).read()
    return json.loads(c[c.index("{"):])


def role_of(var: str) -> Optional[str]:
    """Map an FSM variable name (LLM-chosen, possibly suffixed r1/r2/1/2) to a canonical
    ROLE, so a scenario can bind canonical predicates to whatever names the contract used.
    Order matters: request/grant before shelf/dock; actions before locations."""
    s = var.lower()
    if "request" in s or "_req" in s or s.endswith("req") or s.startswith("req"):  # request / `req` / `req_r1`
        return "shelf_request" if "shelf" in s else ("dock_request" if "dock" in s else "request")
    if "grant" in s:                                                # grant / granted
        return "shelf_grant" if "shelf" in s else ("dock_grant" if "dock" in s else "grant")
    if "done" in s:    return "done"
    if "drop" in s:    return "dropoff"
    if "deliver" in s: return "delivered"
    if "task" in s:    return "task"
    # Tokenize so a robot prefix does not hide the verb: r1_go_pickup / r1_go_dock
    # must count as NAV (startswith("go") only matches unprefixed go_pickup).
    toks = s.replace("-", "_").split("_")
    nav = ("nav" in toks or "move" in toks or "head" in toks
           or any(t.startswith("go") for t in toks))
    # pickup-point / shelf: distinguish LOCATION (at_*) vs NAV (goto_/move_) vs the grab ACTION.
    # (must come AFTER nav is computed; `pick` alone is the action, `at_pick`/`move_pick` are not)
    if "shelf" in s or "pick" in s:
        if nav:        return "go_to_shelf"                         # goto_pickup, move_pickup, head_shelf
        if "at" in s:  return "at_shelf"                            # at_pickup, at_shelf
        return "pickup"                                             # pickup (the grab)
    if "dock" in s or "resource" in s:    return "go_to_dock" if nav else "at_dock"  # generator uses `resource`
    if "charg" in s or "battery" in s:                        # angle-3 C: shared charging station
        if "battery" in s: return "battery_low"
        if nav:            return "go_to_charger"
        if "at" in s:      return "at_charger"
        return "charging"
    if "item" in s:    return "has_item"
    return None


def bind_by_role(fsm_vars, in_roles, out_roles, predicates, actions):
    """Build name-keyed SENSORS/ACTIONS + in/out var lists for one agent, by mapping each
    FSM variable to a role (role_of) and looking up the canonical predicate/action.
    Returns (sensors{var:pred}, action_map{var:Nav|Stationary}, in_vars, out_vars,
    request_vars, grant_vars)."""
    sensors, action_map = {}, {}
    in_vars, out_vars, request_vars, grant_vars = [], [], [], []
    for v in fsm_vars:
        r = role_of(v)
        if r in predicates:                      # sensor input
            sensors[v] = predicates[r]; in_vars.append(v)
        elif r in ("shelf_grant", "dock_grant", "grant"):
            in_vars.append(v); grant_vars.append(v)
        elif r in actions:                       # action output
            action_map[v] = actions[r]; out_vars.append(v)
        elif r in ("shelf_request", "dock_request", "request"):
            out_vars.append(v); request_vars.append(v)
    return sensors, action_map, in_vars, out_vars, request_vars, grant_vars


# ── generic coordinator (unifies the arbiter and the dispatcher) ──────────────
class Coordinator:
    """A shared-resource/assignment coordinator backed by a synthesized FSM, with
    DECLARED wiring — this is the single mechanism behind both warehouse's arbiter
    (request -> grant) and dynamic's dispatcher (done -> assign). Adding a coordinator
    means declaring its `source`/`apply`, not writing a new adapter class.

      source(reqs, world) -> {in_var: bit}   how the FSM's inputs are obtained
                                             (default: the agents' collected REQUESTS)
      apply(out, world)   -> {grant: bit}    how the FSM's outputs take effect; whatever
                                             it RETURNS is injected into agents' GRANTS env
                                             (dispatcher-style coordinators mutate world
                                             state instead and return {}).
    """

    def __init__(self, fsm: Optional[dict] = None, in_vars=None, out_vars=None,
                 source: Optional[Callable] = None, apply: Optional[Callable] = None):
        self.ctrl = FsmController(fsm, in_vars or [], out_vars or []) if fsm else None
        self.source = source
        self.apply = apply

    def step(self, reqs: Dict[str, int], world) -> Dict[str, int]:
        env = self.source(reqs, world) if self.source else reqs
        out = self.ctrl.step(env) if self.ctrl is not None else dict(env)
        return self.apply(out, world) if self.apply else out


# ── the port: abstract base classes ──────────────────────────────────────────
class SimAgent(ABC):
    """One robot the FSM drives. A scenario subclass declares SENSORS/ACTIONS;
    a backend implements pose()/hold()/tick()/_move_to()."""

    name: str = "robot"
    SENSORS: Dict[str, Callable] = {}     # ap_name -> (agent, world) -> bool
    ACTIONS: Dict[str, object] = {}       # action_name -> Nav | Stationary
    REQUESTS: List[str] = []              # output AP names emitted to the arbiter
    GRANTS: List[str] = []                # input AP names consumed from the arbiter

    def bind_fsm(self, fsm, predicates, actions, name=None):
        """Wire an FSM to this scenario's predicates/actions by role. `fsm` is either a path to a
        SLUGS json strategy or an already-loaded strategy dict. Sets SENSORS/ACTIONS/REQUESTS/
        GRANTS and the FsmController, so a subclass __init__ calls this once and then sets its pose
        state instead of repeating the bind_by_role boilerplate."""
        data = load_fsm(fsm) if isinstance(fsm, str) else fsm
        s, am, iv, ov, rq, gr = bind_by_role(data["variables"], None, None, predicates, actions)
        self.SENSORS, self.ACTIONS, self.REQUESTS, self.GRANTS = s, am, rq, gr
        self.ctrl = FsmController(data, iv, ov)
        if name is not None:
            self.name = name
        return self

    @property
    def in_vars(self) -> List[str]:
        return list(self.SENSORS.keys()) + list(self.GRANTS)

    @property
    def out_vars(self) -> List[str]:
        return list(self.ACTIONS.keys()) + list(self.REQUESTS)

    def observe(self, world) -> Dict[str, int]:
        return {ap: int(bool(pred(self, world))) for ap, pred in self.SENSORS.items()}

    def act(self, action: Dict[str, int], world):
        """Apply true action bits. Stationary effects always run (so a grasp is not
        shadowed when the FSM also asserts a Nav, e.g. pickup & goto_dock). Motion:
        unfinished Stationary (done_when False) holds pose; else the first Nav wins;
        else hold. Location APs still persist under a pure Stationary step."""
        active = [n for n, v in action.items() if v and n in self.ACTIONS]
        if not active:
            self._cur = None
            self.target = self.pose()
            return

        for n in active:
            spec = self.ACTIONS[n]
            if isinstance(spec, Stationary) and spec.effect is not None:
                spec.effect(self, world)

        for n in active:
            spec = self.ACTIONS[n]
            if isinstance(spec, Stationary):
                dw = getattr(spec, "done_when", None)
                if dw is not None and not dw(self, world):
                    self._cur = n
                    self.target = self.pose()
                    return

        nav = next((n for n in active if isinstance(self.ACTIONS[n], Nav)), None)
        if nav is not None:
            self._cur = nav
            self.target = self.ACTIONS[nav].target(self, world)
            return

        self._cur = active[0]
        self.target = self.pose()

    def holding(self, world) -> bool:
        """True if the current action declares a `done_when` that is not yet satisfied —
        the driver then KEEPS this action and does not advance the FSM. Default (no
        done_when) is always False, i.e. the legacy advance-every-step behaviour."""
        spec = self.ACTIONS.get(getattr(self, "_cur", None))
        dw = getattr(spec, "done_when", None)
        return bool(dw) and not dw(self, world)

    def reapply(self, world):
        """Re-issue the held action WITHOUT advancing the FSM. Nav recomputes its
        target; Stationary holds pose and, when done_when is still pending, retries
        its effect (so a one-frame miss does not permanently skip a grasp)."""
        spec = self.ACTIONS.get(getattr(self, "_cur", None))
        if isinstance(spec, Nav):
            self.target = spec.target(self, world)
        elif isinstance(spec, Stationary):
            self.target = self.pose()
            dw = getattr(spec, "done_when", None)
            if spec.effect is not None and dw is not None and not dw(self, world):
                spec.effect(self, world)
        else:
            self.target = self.pose()

    # physical primitives — a backend implements these
    @abstractmethod
    def pose(self): ...
    @abstractmethod
    def tick(self, dt, world): ...     # actuate toward self.target


class SimWorld(ABC):
    @abstractmethod
    def agents(self) -> List[SimAgent]: ...
    def arbiter(self):                 # ArbiterController or None (shared resources)
        return None
    @abstractmethod
    def step(self, dt): ...            # advance world + spawn
    @abstractmethod
    def metrics(self) -> dict: ...
    def reset(self):
        pass


# ── generic FSM-execution driver (handles the arbiter request/grant handshake) ─
def fsm_step(world: SimWorld, controllers: Dict[str, FsmController], dt: float, arb=None):
    """One simulation step: observe -> (arbiter request/grant) -> FSM step -> act ->
    tick -> world.step. Shared by the headless run_fsm loop and a live render loop."""
    envs = {a.name: a.observe(world) for a in world.agents()}
    if arb is not None:
        reqs = {}
        for a in world.agents():
            reqs.update(controllers[a.name].requests(a.REQUESTS))
        grants = arb.step(reqs, world)
        for a in world.agents():
            envs[a.name].update({g: grants.get(g, 0) for g in a.GRANTS})
    for a in world.agents():
        if a.holding(world):
            a.reapply(world)                     # keep the current action (retarget); don't advance FSM
        else:
            a.act(controllers[a.name].step(envs[a.name]), world)
    for a in world.agents():
        a.tick(dt, world)
    world.step(dt)


def run_fsm(world: SimWorld, controllers: Dict[str, FsmController],
            steps: int, dt: float = 1 / 60.0, on_step: Optional[Callable] = None,
            invariants: Optional[List[Callable]] = None, assumption_monitor=None):
    """Drive the FSMs headless for `steps` steps; return world.metrics().

    If `invariants` is given (each a `callable(world) -> Optional[str]` returning an error
    message or None), they are checked AFTER every step; the returned metrics gain
    `n_violations` and a sample `invariant_violations` list. This is the physical-layer
    auto-check: catch sim/contract mismatches (e.g. a permanently stuck FSM, an off-model
    sensor) and CBF safety breaches (robots overlapping) instead of trusting the metrics.

    If `assumption_monitor` (a contrauto.assumption_monitor.AssumptionMonitorSet) is given, each
    step it is fed the OBSERVED environment (every agent's sensor inputs ∪ its current action
    outputs). On a violation the returned metrics gain `assumption_violations` (count) and
    `assumption_feedback` (the channel-5 decomposer feedback) — i.e. the environment could NOT
    discharge a contract assumption at run time, which the task-level loop turns into a re-decompose.
    """
    arb = world.arbiter()
    violations = []
    # Bounded-progress sentinel (the runtime surrogate for liveness, which is not
    # monitorable on finite traces): a certified controller that makes NO physical
    # progress for `progress_patience` steps is reported as a channel-5 violation.
    # Catches ALL starvation variants mechanism-agnostically — cheap-exit goal hacking
    # (done-mashing), inter-FSM handshake standoffs, and legal idling — including the
    # ones no design-time certificate or assumption monitor can see.
    progress_patience = 1500
    _progress = lambda mm: mm.get("delivered", 0) + mm.get("cleared", 0)
    last_p, last_p_step, stall_at = 0, 0, None
    for s in range(steps):
        fsm_step(world, controllers, dt, arb)
        if s % 100 == 0 and s > 0:
            p = _progress(world.metrics())
            if p > last_p:
                last_p, last_p_step = p, s
            elif stall_at is None and s - last_p_step >= progress_patience:
                stall_at = s
        if invariants:
            for inv in invariants:
                msg = inv(world)
                if msg:
                    violations.append((s, msg))
        if assumption_monitor is not None:
            obs = {}
            for a in world.agents():
                env = a.observe(world) or {}                 # sensor inputs (+ grants)
                act = getattr(a, "action", {}) or {}         # current action outputs
                for k, v in {**env, **act}.items():
                    obs[k] = 1 if v else 0
            assumption_monitor.update(obs, s)
        if on_step is not None and on_step(s, world) is False:
            break
    m = world.metrics()
    if invariants is not None:
        m = {**m, "n_violations": len(violations), "invariant_violations": violations[:20]}
    if assumption_monitor is not None:
        m = {**m, "assumption_violations": len(assumption_monitor.violations),
             "assumption_feedback": assumption_monitor.feedback()}
    # Off-script sentinel: a controller whose observed input valuation has NO matching
    # successor freezes in place (FsmController._stuck climbs). That means physics produced
    # an environment state OUTSIDE the contract's assumed set — a timing-semantics mismatch
    # (class 8) that X-ful assumptions hide from the monitor. Probe-verified: the warehouse
    # one-then-stall froze at the delivered pulse (has_item already low), stuck forever.
    stuck = {n: c._stuck for n, c in controllers.items() if getattr(c, "_stuck", 0) >= 50}
    if stuck:
        m = {**m, "offscript_violations": len(stuck),
             "offscript_feedback": (
                 "[OFF-SCRIPT CONTROLLER] (channel 5, runtime): controller(s) "
                 f"{sorted(stuck)} observed input valuations with NO matching strategy "
                 f"successor and froze (stuck steps: {stuck}). The physical environment "
                 "produced a state outside the contract's ASSUMED environment — the assumed "
                 "timing of the sensors mismatches the physics (e.g. assuming `delivered` "
                 "implies still-carrying, when the pulse and the carry-drop coincide). "
                 "REPAIR: align the assumptions with the grounding TIMING SEMANTICS notes; "
                 "do not couple a one-step pulse to states that fall on the same step.")}
    if stall_at is not None:
        m = {**m, "progress_violations": 1, "progress_stalled_at": stall_at,
             "progress_feedback": (
                 "[BOUNDED-PROGRESS VIOLATION] (channel 5, runtime): the certified "
                 f"controllers made no physical progress for {progress_patience}+ steps "
                 f"(stalled from step {last_p_step}, progress={last_p}). The strategy is "
                 "formally winning but physically idle — typical causes: a goal output "
                 "without a sensor WITNESS precondition (e.g. missing G(done -> delivered), "
                 "so the goal is satisfiable by asserting the action vacuously), an "
                 "inter-agent handshake standoff (each FSM waiting for the other's signal), "
                 "or reliance on an environment justice that never has to fire. REPAIR: gate "
                 "every completion/goal action on the sensor that witnesses its physical "
                 "effect, make preconditions chain back to real actions, and keep handshake "
                 "formulas consistent with the grounding timing semantics.")}
    return m


# ── physical-layer invariants (reusable across scenarios) ────────────────────
def separation_invariant(min_dist: float):
    """No two agents' centres come closer than `min_dist` (CBF safety / no overlap)."""
    def check(world):
        ps = [a.pose() for a in world.agents()]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                d = math.hypot(ps[i][0] - ps[j][0], ps[i][1] - ps[j][1])
                if d < min_dist:
                    return f"agents {i},{j} within {d:.1f} < {min_dist}"
        return None
    return check


def no_stall_invariant(controllers: Dict[str, FsmController], limit: int = 600):
    """No FSM stays off-model (no matching successor) for more than `limit` steps — catches
    dead-end / degenerate strategies and sims that violate the contract's assumptions."""
    def check(world):
        for name, c in controllers.items():
            if getattr(c, "_stuck", 0) > limit:
                return f"{name} FSM stalled {c._stuck} steps (dead-end / off-model env)"
        return None
    return check
