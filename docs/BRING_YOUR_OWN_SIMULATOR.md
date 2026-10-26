# Bring your own simulator

This guide explains how to connect a new simulation environment to the framework. The two layers
are decoupled, so the task layer (contracts, synthesis, the three task feedbacks) and the physical
machinery (planner, CBF contracts, certificate, the two physical feedbacks) are reused as they are.
You implement a thin adapter that exposes your environment as a set of predicates and actions plus
two small classes, and you register it.

The relevant interface lives in `contrauto/sim_iface.py`. The runner finds your simulator through
`simulator/registry.py`.

## The base (one class to subclass)

There is a single simulator base: `SimAgent` and `SimWorld` in `contrauto/sim_iface.py`. Every
scenario is a subclass, and the base is rendering- and dimension-agnostic. An agent only has to
implement `pose()` (a position, 2D or 3D) and `tick()` (drive toward `self.target`), and declare its
predicates and actions. This is what lets the same machinery extend from the 2D pygame simulators
here to a 3D engine later: you write a new backend, nothing else changes.

Rendering is not part of the base. A 2D scenario may mix in the pygame board in `simulator/core.py`
(`BaseGame` / `BaseRobot`) for an interactive view; a 3D scenario brings its own viewer. `core.py` is
an optional 2D renderer, not a base class.

## What you reuse without changes

- The FSM controller (`FsmController`) and the main loop (`fsm_step`, `run_fsm`), including the
  arbiter request and grant handshake for shared resources.
- Name binding by role (`bind_by_role`, `role_of`). The synthesizer names variables differently on
  every run, so you do not hand map names. You provide predicates and actions keyed by role
  (location `at_*`, navigation `goto_*`, action `pickup`/`dropoff`, protocol `request`/`grant`) and
  the binder connects whatever names the FSM used.
- The whole task layer (translator, decomposer, CHASE, SLUGS, well-separatedness gate) and the
  five feedbacks.

## What you implement

### 1. Predicates (the atomic propositions)

A dictionary that maps each atomic proposition name to a function `(agent, world) -> bool`. These
ground the discrete propositions used by the contracts in your environment's physical state.

```python
def _at_goal(a, w):
    return math.hypot(a.x - w.goal[0], a.y - w.goal[1]) < 8.0

PREDICATES = {
    "at_goal":  _at_goal,
    "has_item": lambda a, w: a.carry is not None,
    # one entry per atomic proposition the contracts use, by role
}
```

### 2. Actions (the actuators)

A dictionary that maps each action name to either a `Nav` (move toward a target) or a `Stationary`
(an in place effect such as pickup or dropoff).

```python
def _goal_target(a, w):
    return w.goal

def _do_pick(a, w):
    a.carry = w.nearest_item(a)

ACTIONS = {
    "go_to_goal": Nav(_goal_target, done_when=_at_goal),
    "pick":       Stationary(_do_pick),
    # one entry per action the FSM can output, by role
}
```

`Nav(target, done_when)`: `target(agent, world)` returns an `(x, y)` goal, and `done_when(agent,
world)` returns True when the action is complete. `Stationary(effect)`: `effect(agent, world)`
applies an in place change.

### 3. A SimAgent subclass (the physical backend for one robot)

```python
class MyRobot(SimAgent):
    def __init__(self, rid, fsm_path, x0, y0):
        # bind_fsm (on the SimAgent base) loads the FSM and wires it to PREDICATES/ACTIONS
        # by role, setting SENSORS/ACTIONS/REQUESTS/GRANTS and the controller.
        self.bind_fsm(fsm_path, PREDICATES, ACTIONS, name=f"r{rid}")
        self.x, self.y, self.carry = x0, y0, None
        self.target = (x0, y0)

    def pose(self):
        return (self.x, self.y)

    def tick(self, dt, world):
        # Drive toward self.target. Either reuse the CBF quadratic program
        # (see simulator/two_layer_dock.py tick for the pattern) or translate
        # self.target into your environment's own control or action API here.
        ...
```

### 4. A SimWorld subclass (the backend world)

```python
class MyWorld(SimWorld):
    def __init__(self, fsm_dir):
        self.goal = (...)
        self.robots = [MyRobot(1, f"{fsm_dir}/robot1.json", ...)]

    def agents(self):
        return self.robots

    def arbiter(self):
        return None                      # or an ArbiterController for a shared resource

    def step(self, dt):
        ...                              # advance the world, spawn new tasks

    def metrics(self):
        return {"delivered": ..., "violations": ...}
```

### 5. An adapter, registered

```python
# in simulator/mysim.py
def headless(fsm_dir, steps=8000, dt=1 / 60.0):
    w = MyWorld(fsm_dir)
    controllers = {a.name: a.ctrl for a in w.agents()}
    return run_fsm(w, controllers, steps, dt)   # run_fsm pulls the arbiter from world.arbiter()

# in simulator/registry.py
from simulator.mysim import headless as _mysim
SIMULATORS["mytask"] = (_mysim, ["robot1.json"])   # the FSM files it expects in the workdir
```

The adapter is a function `fsm_dir -> metrics dict`. The second item of the registry tuple is the
list of FSM JSON files the simulator needs; the runner uses it to decide whether an FSM already
exists for reuse.

## Requirements on the environment

1. Observable state for predicates. The environment must expose enough state to evaluate the atomic
   propositions the contracts use. If it does not define them, you write them as functions of its
   state inside `PREDICATES`.
2. A way to drive an agent. The physical layer produces an acceleration for a double integrator. The
   environment should accept an acceleration, a velocity, or a position target, and you map the
   physical output to that API inside `tick`. If the action space is different (discrete grid moves,
   high level skills, an external game API), you adapt it in `tick`: the `Nav` target and `done_when`
   say where to go, and `tick` translates that into the environment's control.
3. Geometry for the planner and the CBF. The A* planner and the CBF need obstacle geometry and
   bounds, expressed as rectangles and an `(xmin, xmax, ymin, ymax)` bound. Convert your layout into
   that form, or replace the planner.
4. Role coverage. The predicates and actions you provide must cover the roles the task uses
   (location, navigation, the in place action, and the protocol signals for a shared resource).
   Exact names do not matter because `bind_by_role` maps by role, but the roles must be present.
5. Shared resources. If the scenario has a shared resource, provide an arbiter and the request and
   grant protocol through `world.arbiter()` and each agent's `REQUESTS` and `GRANTS`.

## Step by step

1. Write a task YAML in `contrauto/tasks/<name>.yaml` (architecture, requirements, optional map).
2. Synthesize the FSMs once: `python main.py --task <name> --no-sim`. This exercises the task layer
   and its three feedbacks, and writes the FSM JSON files under `outputs/<name>/workdir/`.
3. Implement `PREDICATES`, `ACTIONS`, the `SimAgent` subclass, and the `SimWorld` subclass in
   `simulator/<name>.py`.
4. Add the adapter to `simulator/registry.py`.
5. Run both layers: `python main.py --task <name>`.

## What can be hard

- No continuous control. If the environment has no continuous control (a purely discrete or abstract
  simulator, or an external engine), the CBF physical layer does not apply directly. You either run
  the task layer only, or write a different `tick`.
- Partial observability. If a predicate cannot be computed from the available state, you need state
  estimation before you can ground it.
- Different dynamics. The CBF here assumes a double integrator (see `contrauto/stl_cbf.py` and the
  rectangle and circle barriers in `cbf/filter.py`). A car, a Dubins vehicle, or a drone needs the
  CBF relative degree and dynamics adapted.

## Summary

You do not touch the task layer, the feedbacks, the planner, the CBF contracts, the certificate, or
the FSM loop. You provide predicates, actions, a `tick` that drives toward a target, a `SimWorld`
wrapper, and one registry line. The two hard requirements on the environment are that you can compute
the propositions from its state and that you can drive an agent toward a target while reading its
geometry.
