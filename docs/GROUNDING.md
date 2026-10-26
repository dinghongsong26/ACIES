# Grounding: how a simulator decides an atomic proposition

A synthesized FSM speaks in atomic propositions (APs) such as `at_shelf`, `has_item`, or
`battery_low`. These are symbols. Before a simulator can run an FSM it has to decide, every step,
whether each symbol is true in the actual world. That decision is called grounding.

This document defines one format for writing a grounding, used by every simulator in the repo, so that
adding a new simulator or a new proposition is a mechanical step and the contract between a proposition
and the world cannot silently drift.

The base classes live in `contrauto/grounding.py`. Worked examples live in `simulator/warehouse.py`
and in every other file under `simulator/` that binds an FSM. The same format applies to a 3D drone
world; the 3D example below is written out to show that 2D and 3D groundings are written separately.

## The idea

A grounding reads world state only through a gate. Each simulator declares a `providers` map from a
capability token (a string such as `agent.pos` or `geom.point:dock`) to a small function that fetches
that value from the agent or world. A grounding is a subclass of `Grounding` and implements
`evaluate(cap)`, where `cap` is the gate. It reads what it needs with `cap.get(token)`.

If a grounding asks for a token the simulator did not declare, `cap.get` raises immediately and names
the missing token. So "what a grounding reads" and "what it requires from the world" are the same
thing. There is no separate hand written list to keep in sync, and a grounding cannot depend on
something the world does not expose without failing loudly and specifically.

```python
from contrauto.grounding import Grounding, bind_groundings

def _providers():
    return {
        "agent.xyz":        lambda a, w: a.pose(),
        "geom.point:dock":  lambda a, w: a.cfg.dock_xyz,
        "geom.reach":       lambda a, w: a.cfg.interaction_reach,
    }

class AtDock(Grounding):
    def evaluate(self, cap):
        return cap.dist(cap.get("agent.xyz"), cap.get("geom.point:dock")) <= cap.get("geom.reach")

PREDICATES = bind_groundings({"at_dock": AtDock()}, _providers())
```

`bind_groundings` returns the `{name: predicate}` map that `SimAgent.bind_fsm` already consumes, where
each predicate is a plain `(agent, world) -> bool` callable that runs `evaluate` behind the gate.
Nothing in the FSM executor changes.

## What the gate gives you

Read a token the world did not declare and you get a clear failure at the point of the read, not an
opaque `AttributeError` deep in a step:

```
MissingCapability: world does not provide capability 'scalar:heat'; provides: ['agent.xyz', ...]
```

Because reads go through the gate, the gate also records every token a scenario actually used
(`cap.used`). The real requirement set is derived from what runs, not declared by hand.

## Providers are the one place that touches raw state

Inside `evaluate` you never write `a.x` or `w.cfg.dock_xyz` directly. Those raw reads live only in the
`providers` map, one entry per capability token. That map is the simulator's contract: it is the full
list of what the world can answer, in one place. The rule of thumb is that a grounding contains logic
(a comparison, a threshold, a latch), and a provider contains a single raw read.

## Tokens are per simulator, not global

The tokens are each simulator's own vocabulary. They are not a shared global list. A 2D pygame world
exposes pixel positions and a goods list, so its tokens read like `agent.pos`, `agent.carry`,
`world.goods`. A 3D drone world exposes a pose and payload flags, so its tokens read like `agent.xyz`,
`agent.payload`. This is expected. The same proposition `at_shelf` is a different grounding in each
world because the two worlds hold different state.

What is shared across simulators is the format (subclass `Grounding`, read through `cap.get`, declare
`providers`), not the grounding instances. Do not try to make one grounding serve both 2D and 3D. Write
one per world.

## Worked example: at_shelf in 2D and in 3D

Same proposition, two worlds, two groundings, one format.

2D (`simulator/warehouse.py`). The world holds a goods list on shelves, so its providers answer the
robot pixel position, its carried item, and the goods.

```python
PROVIDERS = {
    "agent.pos":          lambda a, w: (a.x, a.y),
    "agent.carry":        lambda a, w: a.carry,
    "world.free_goods":   lambda a, w: _free_goods(w),
    "geom.shelf_centers": lambda a, w: [...],
}

class AtShelf(Grounding):
    def evaluate(self, cap):
        pos = cap.get("agent.pos")
        if any(cap.dist(pos, (b.x, b.y)) < ROBOT_R + OBJ_SZ + 6 for b in cap.get("world.free_goods")):
            return True
        if cap.get("agent.carry") is not None:
            return any(cap.dist(pos, c) < ROBOT_R + OBJ_SZ + 12 for c in cap.get("geom.shelf_centers"))
        return False
```

3D (a drone world). The world holds a drone pose and a fixed shelf point, so its providers answer the
pose, the named point, and the arrival radius.

```python
def _warehouse_3d_providers():
    return {
        "agent.xyz":        lambda a, w: np.asarray(a.pose()),
        "geom.point:shelf": lambda a, w: a.cfg.shelf_xyz,
        "geom.reach":       lambda a, w: a.cfg.interaction_reach,
    }

class AtShelf3D(Grounding):
    def evaluate(self, cap):
        return cap.dist(cap.get("agent.xyz"), cap.get("geom.point:shelf")) <= cap.get("geom.reach")
```

Both compute "the agent is at the shelf", but the 2D version has pygame goods logic that has no 3D
counterpart, and the 3D version uses a meter distance the 2D version does not. Writing them separately
keeps each one readable.

## How to add a new simulator

1. Decide what state your world can answer, and pick a token for each. For example a thermal drone
   world might answer `agent.xyz`, `scalar:heat`, and `flag:hotspot`.
2. Write a `providers` map, one entry per token, each a `lambda a, w: <raw read>`.
3. For each proposition your FSMs use, write a `Grounding` subclass whose `evaluate(cap)` reads only
   through `cap.get`.
4. Bind them with `bind_groundings({...}, providers)` and hand the result to `SimAgent.bind_fsm`,
   exactly as the warehouse simulators do.
5. Implement the physical primitives the shared base needs (position, step, move). For a 2D world
   subclass the 2D base, for a drone world subclass `DroneAgent` and `PyBulletWorld`.

If a grounding reads a token your providers do not list, the gate tells you the exact token to add.

## Adding a new proposition

Write one `Grounding` subclass whose `evaluate` reads through `cap.get`. If it needs a reading your
providers do not yet answer, add one entry to the `providers` map. No other grounding and no other
simulator is affected.

## The action side

Grounding covers the forward direction, from a proposition to a truth value. The reverse direction,
from an FSM action symbol such as `pickup` or `go_to_dock` to a physical command, is handled by the
`Nav` and `Stationary` action wrappers in `contrauto/sim_iface.py`, declared in each simulator's
`ACTIONS` map alongside its predicates. This document covers the predicate side, which is where truth
is decided.
