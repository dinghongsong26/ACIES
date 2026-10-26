"""Grounding: one enforced format for deciding an atomic proposition against a simulator's state.

A synthesized FSM speaks in atomic propositions (APs) such as `at_shelf` or `battery_low`. These are
symbols. Grounding is the mapping from a symbol to a truth value read off the concrete world, decided
every step. This module gives that mapping ONE shape, and enforces the contract between a proposition
and the world by construction.

The rule: a grounding may read world state ONLY through a Capabilities gate. Each simulator declares a
`providers` map from a capability token (a string like "agent.pos" or "geom.point:dock") to a function
that fetches it. A grounding's `evaluate(cap)` calls `cap.get(token)`. If it asks for a token the
simulator did not declare, the gate raises immediately and names the missing token. So "what a
grounding reads" and "what it requires" are the same thing: there is no separate hand written list to
drift or lie. The set of tokens actually read is recorded in `cap.used`, so requirements are derived,
not declared.

Two simulators (for example a 2D pygame world and a 3D pybullet world) write their own providers and
their own groundings. They are not shared: a 2D world exposes pixel positions and a goods list, a 3D
world exposes drone poses and payload flags. What is shared is this format and the gate, not the
instances. See docs/GROUNDING.md.
"""
from __future__ import annotations

import math


class MissingCapability(KeyError):
    """Raised when a grounding reads a capability token the simulator did not declare in `providers`."""


class Capabilities:
    """The gate a grounding reads the world through.

    Built from a simulator's `providers` map (token -> callable(agent, world) -> value). `get(token)`
    returns the value if the simulator declared the token, or raises MissingCapability naming it. Every
    token read is recorded in `used`, so a scenario's real requirement set is derived from what runs.
    """

    def __init__(self, providers: dict):
        self._providers = dict(providers)
        self.used: set = set()
        self._agent = None
        self._world = None

    @property
    def provides(self) -> set:
        return set(self._providers)

    def _bind(self, agent, world) -> "Capabilities":
        self._agent, self._world = agent, world
        return self

    def get(self, token: str):
        self.used.add(token)
        fn = self._providers.get(token)
        if fn is None:
            raise MissingCapability(
                f"world does not provide capability {token!r}; provides: {sorted(self._providers)}"
            )
        return fn(self._agent, self._world)

    @staticmethod
    def dist(p, q) -> float:
        """Euclidean distance, dimension agnostic (2D pixels or 3D meters)."""
        return math.sqrt(sum((float(pi) - float(qi)) ** 2 for pi, qi in zip(p, q)))


class Grounding:
    """One AP's grounding. Subclass and implement `evaluate(cap)`, reading world state only via `cap`."""

    def evaluate(self, cap: Capabilities) -> bool:
        raise NotImplementedError


def bind_groundings(mapping: dict, providers: dict) -> dict:
    """Wrap each grounding into the `(agent, world) -> bool` callable that SimAgent.bind_fsm consumes.

    Every call reads world state only through a Capabilities gate built from `providers`, so a grounding
    that reads an undeclared token raises MissingCapability instead of failing with an opaque error. One
    gate is shared across the scenario's groundings, so `gate.used` accumulates the whole scenario's real
    requirement set. The gate is returned too (as the `.capabilities` attribute of the mapping) for
    inspection; callers that only need the predicate map can ignore it.
    """
    gate = Capabilities(providers)

    def wrap(g: Grounding):
        return lambda agent, world: bool(g.evaluate(gate._bind(agent, world)))

    bound = {name: wrap(g) for name, g in mapping.items()}
    return bound
