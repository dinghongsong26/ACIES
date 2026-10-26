"""Task specification schema: three-file YAML system (physics, map, task)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml


# ---------------------------------------------------------------------------
# Physics spec
# ---------------------------------------------------------------------------

@dataclass
class PhysicsSpec:
    """One node's physics: dynamics type and its input/output propositions."""

    name: str
    type: str                       # "differential_drive" | "virtual"
    geometry: dict | None
    dynamics: dict | None
    prop_inputs: list[str]
    prop_outputs: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> PhysicsSpec:
        """Load a physics spec from YAML."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        props = raw.get("propositions", {})
        return cls(
            name=raw.get("name", path.stem),
            type=raw.get("type", "virtual"),
            geometry=raw.get("geometry"),
            dynamics=raw.get("dynamics"),
            prop_inputs=props.get("inputs", raw.get("observables", [])),
            prop_outputs=props.get("outputs", raw.get("actions", [])),
        )


# ---------------------------------------------------------------------------
# Map spec
# ---------------------------------------------------------------------------

@dataclass
class MapRegion:
    """A named map region (rectangle or circle) with interaction metadata."""

    name: str
    shape: str                      # "rectangle" | "circle"
    position: list[float]
    size: list[float] | None
    radius: float | None
    interaction: str
    navigable: bool
    color: list[int]


@dataclass
class MapSpec:
    """The world map: bounds, regions, and named spawn points."""

    name: str
    dimensions: int
    bounds: list[float]
    regions: list[MapRegion]
    spawn_points: dict[str, list[float]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> MapSpec:
        """Load a map spec from YAML, expanding region templates."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        templates: dict[str, dict] = raw.get("templates", {})

        regions: list[MapRegion] = []
        for r in raw.get("regions", []):
            # Start from template defaults, then override with region fields.
            merged: dict = {}
            tpl_name = r.get("template")
            if tpl_name and tpl_name in templates:
                merged.update(templates[tpl_name])
            # Every explicit key on the region (except 'template') overrides.
            for k, v in r.items():
                if k != "template":
                    merged[k] = v

            regions.append(MapRegion(
                name=merged["name"],
                shape=merged.get("shape", "rectangle"),
                position=merged.get("position", [0, 0]),
                size=merged.get("size"),
                radius=merged.get("radius"),
                interaction=merged.get("interaction", ""),
                navigable=merged.get("navigable", True),
                color=merged.get("color", [128, 128, 128]),
            ))

        spawn_points: dict[str, list[float]] = {}
        for k, v in raw.get("spawn_points", {}).items():
            spawn_points[k] = list(v)

        return cls(
            name=raw.get("name", path.stem),
            dimensions=raw.get("dimensions", 2),
            bounds=raw.get("bounds", []),
            regions=regions,
            spawn_points=spawn_points,
        )


# ---------------------------------------------------------------------------
# Architecture specs
# ---------------------------------------------------------------------------

@dataclass
class NodeSpec:
    """One architecture node (robot or coordinator) with its merged propositions."""

    name: str
    physics: PhysicsSpec | None
    color: str
    role: str
    prop_inputs: list[str]          # merged: physics + overrides
    prop_outputs: list[str]         # merged: physics + overrides


@dataclass
class EdgeSpec:
    """A directed-pair communication edge between two nodes."""

    from_node: str
    to_node: str
    interface: str
    # Structured communication signals carried by this edge. Each signal is the SINGLE
    # source of truth for one communication variable shared by the two endpoints — so the
    # two controllers can never disagree on who owns/drives it. See _expand_edge_interfaces.
    #   {"base": "grant", "dir": "backward"}  # base name + direction
    #   dir "forward"  => from -> to  (from's OUTPUT, to's INPUT)
    #   dir "backward" => to -> from  (to's OUTPUT, from's INPUT)
    signals: list[dict] | None = None


# ---------------------------------------------------------------------------
# Task spec (top-level)
# ---------------------------------------------------------------------------

def _node_prefix(node_name: str) -> str:
    """Derive a short prefix from a node name, e.g. Robot1 -> r1_."""
    name = node_name.lower()
    if name[-1].isdigit():
        return "r" + name[-1] + "_"
    return name + "_"


def _find_project_root(start: Path) -> Path:
    """Walk up from *start* until we find a directory containing ``maps/``
    or ``physics/``.  Falls back to *start*'s parent if nothing is found."""
    current = start if start.is_dir() else start.parent
    for ancestor in [current, *current.parents]:
        if (ancestor / "maps").is_dir() or (ancestor / "physics").is_dir():
            return ancestor
    return start.parent


@dataclass
class TaskSpec:
    """Top-level task: map, nodes, edges, and natural-language requirements."""

    name: str
    map_spec: MapSpec
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    requirements: list[str]
    # Contract-writing guidance for the decomposer only (not requirements to translate).
    decomposer_notes: list[str] = field(default_factory=list)
    # Files this task was loaded from (the task YAML, then any map / physics YAML it references).
    sources: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskSpec:
        """Load a task and its referenced map/physics, then prefix props and expand edges."""
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _find_project_root(path)
        sources = [str(path)]

        # --- Map ---
        map_rel = raw.get("map")
        if map_rel:
            map_path = (root / map_rel).resolve()
            map_spec = MapSpec.from_yaml(map_path)
            sources.append(str(map_path))
        else:
            map_spec = MapSpec(
                name="",
                dimensions=2,
                bounds=[],
                regions=[],
                spawn_points={},
            )

        # --- Nodes ---
        arch = raw.get("architecture", {})
        nodes: list[NodeSpec] = []
        for n in arch.get("nodes", []):
            physics: PhysicsSpec | None = None
            physics_rel = n.get("physics")
            if physics_rel:
                physics_path = (root / physics_rel).resolve()
                physics = PhysicsSpec.from_yaml(physics_path)
                if str(physics_path) not in sources:
                    sources.append(str(physics_path))

            # Merge propositions: start from physics, override if
            # the node declares its own lists.
            node_props = n.get("propositions", {})
            if "inputs" in node_props:
                p_in = list(node_props["inputs"])
            elif "observables" in n:
                p_in = list(n["observables"])
            elif physics:
                p_in = list(physics.prop_inputs)
            else:
                p_in = []

            if "outputs" in node_props:
                p_out = list(node_props["outputs"])
            elif "actions" in n:
                p_out = list(n["actions"])
            elif physics:
                p_out = list(physics.prop_outputs)
            else:
                p_out = []

            nodes.append(NodeSpec(
                name=n["name"],
                physics=physics,
                color=n.get("color", "gray"),
                role=n.get("role", "robot"),
                prop_inputs=p_in,
                prop_outputs=p_out,
            ))

        # --- Edges ---
        edges: list[EdgeSpec] = []
        for e in arch.get("edges", []):
            edges.append(EdgeSpec(
                from_node=e.get("from", e.get("from_node", "")),
                to_node=e.get("to", e.get("to_node", "")),
                interface=e.get("interface", ""),
                signals=list(e.get("signals", []) or []),
            ))

        # --- Requirements ---
        requirements: list[str] = raw.get("requirements", [])

        task = cls(
            name=raw.get("name", path.stem),
            map_spec=map_spec,
            nodes=nodes,
            edges=edges,
            requirements=requirements,
            decomposer_notes=list(raw.get("decomposer_notes", []) or []),
            sources=sources,
        )
        task._apply_prefixes()
        task._expand_edge_interfaces()
        return task

    def _expand_edge_interfaces(self) -> None:
        """Turn structured edge signals into per-node input/output variables, with a SINGLE
        source of truth per communication variable.

        A signal declared on an edge becomes the SAME variable on BOTH endpoints — the
        producer's output and the consumer's input — so a controller's interface can never
        disagree about who owns/drives a communication variable (the failure mode behind the
        N>=5 refinement non-convergence: the LLM kept reassigning grant's owner/direction).

        The variable name is derived deterministically from the FROM node, e.g. base "grant"
        on edge Robot1<->Arbiter -> grant_r1 (or take an explicit signal["var"]). Runs AFTER
        _apply_prefixes so these names are not prefixed a second time.
        """
        by_name = {n.name: n for n in self.nodes}

        def _add(lst: list[str], v: str) -> None:
            if v not in lst:
                lst.append(v)

        for e in self.edges:
            frm = by_name.get(e.from_node)
            to = by_name.get(e.to_node)
            if not (frm and to and e.signals):
                continue
            fid = _node_prefix(e.from_node).rstrip("_")        # Robot1 -> "r1"
            for sig in e.signals:
                base = sig.get("var") or sig.get("base") or sig.get("name")
                if not base:
                    continue
                var = sig["var"] if sig.get("var") else f"{base}_{fid}"
                direction = sig.get("dir", "forward")
                if direction == "world":
                    # World-owned signal: the ENVIRONMENT drives it; both endpoints
                    # only observe it (inputs on both sides, and hence a System input,
                    # since the skeleton derives System inputs as inputs-minus-outputs).
                    # Use for demand/release signals the requirement attributes to the
                    # world -- putting them on a robot's output side creates obligations
                    # no component can discharge (see the corridor case study).
                    _add(frm.prop_inputs, var)
                    _add(to.prop_inputs, var)
                elif direction == "forward":                   # from -> to
                    _add(frm.prop_outputs, var)
                    _add(to.prop_inputs, var)
                else:                                          # backward: to -> from
                    _add(to.prop_outputs, var)
                    _add(frm.prop_inputs, var)

    def _apply_prefixes(self) -> None:
        """Add node-name prefixes to propositions when multiple physical
        nodes share the same physics template.  Coordinators (whose
        propositions are already globally unique) are left untouched."""
        physical = [n for n in self.nodes
                    if not (n.physics and n.physics.type == "virtual")]
        if len(physical) <= 1:
            return
        for node in physical:
            prefix = _node_prefix(node.name)
            node.prop_inputs = [f"{prefix}{p}" for p in node.prop_inputs]
            node.prop_outputs = [f"{prefix}{p}" for p in node.prop_outputs]

    def all_propositions(self) -> list[str]:
        """Return a flat, deduplicated list of every atomic proposition
        across all nodes (inputs + outputs, prefixed)."""
        props: list[str] = []
        for node in self.nodes:
            for p in node.prop_inputs + node.prop_outputs:
                if p not in props:
                    props.append(p)
        return props

    def with_robot_count(self, count: int) -> "TaskSpec":
        """Return a task with ``Robot1``'s worker interface expanded to ``count`` robots."""
        if count < 1:
            raise ValueError("robot count must be at least 1")
        workers = sorted((node for node in self.nodes if re.fullmatch(r"Robot\d+", node.name)),
                         key=lambda node: int(node.name[5:]))
        if not workers:
            raise ValueError(f"task {self.name!r} has no RobotN worker template")
        template = workers[0]
        worker_names = {node.name for node in workers}
        other_nodes = [deepcopy(node) for node in self.nodes if node.name not in worker_names]

        def rename(value: str, index: int) -> str:
            return re.sub(r"^r\d+_", f"r{index}_", value)

        nodes = []
        for index in range(1, count + 1):
            node = deepcopy(template)
            node.name = f"Robot{index}"
            node.prop_inputs = [rename(value, index) for value in node.prop_inputs]
            node.prop_outputs = [rename(value, index) for value in node.prop_outputs]
            nodes.append(node)

        template_edges = [edge for edge in self.edges
                          if template.name in (edge.from_node, edge.to_node)]
        edges = [deepcopy(edge) for edge in self.edges
                 if edge.from_node not in worker_names and edge.to_node not in worker_names]
        for index in range(1, count + 1):
            for edge in template_edges:
                clone = deepcopy(edge)
                clone.from_node = f"Robot{index}" if clone.from_node == template.name else clone.from_node
                clone.to_node = f"Robot{index}" if clone.to_node == template.name else clone.to_node
                for signal in clone.signals or []:
                    if signal.get("var"):
                        signal["var"] = rename(signal["var"], index)
                edges.append(clone)

        # Remove the original task's generated req_rN/grant_rN-style variables before
        # expanding the new edge set; otherwise reducing N leaves stale Arbiter ports.
        signal_bases = {
            signal.get("base") or signal.get("name")
            for edge in template_edges for signal in (edge.signals or [])
            if not signal.get("var") and (signal.get("base") or signal.get("name"))
        }
        generated = re.compile(r"^(?:" + "|".join(map(re.escape, signal_bases)) + r")_r\d+$") \
            if signal_bases else None
        result = TaskSpec(self.name, deepcopy(self.map_spec), nodes + other_nodes, edges,
                          list(self.requirements), list(self.decomposer_notes),
                          list(self.sources))
        if generated:
            for node in result.nodes:
                node.prop_inputs = [value for value in node.prop_inputs if not generated.fullmatch(value)]
                node.prop_outputs = [value for value in node.prop_outputs if not generated.fullmatch(value)]
        result._expand_edge_interfaces()
        return result

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, include_decomposer_notes: bool = False) -> str:
        """Render the task (architecture, map, requirements) as LLM-prompt text; the
        decomposer additionally passes include_decomposer_notes=True."""
        lines: list[str] = [f"SYSTEM: {self.name}", ""]

        # Architecture
        lines.append("ARCHITECTURE:")
        lines.append("  Nodes:")
        for n in self.nodes:
            parts = [f"{n.name} (role: {n.role})"]
            if n.physics:
                parts.append(f"physics={n.physics.name}")
            if n.prop_inputs:
                parts.append(f"prop_inputs={n.prop_inputs}")
            if n.prop_outputs:
                parts.append(f"prop_outputs={n.prop_outputs}")
            lines.append(f"    - {', '.join(parts)}")

        if self.edges:
            lines.append("  Edges:")
            for e in self.edges:
                lines.append(
                    f"    - {e.from_node} <-> {e.to_node}: {e.interface}"
                )

        # Resources (from map regions)
        if self.map_spec.regions:
            lines.append("")
            lines.append(f"MAP: {self.map_spec.name}")
            lines.append(f"  bounds: {self.map_spec.bounds}")
            lines.append("  Regions:")
            for r in self.map_spec.regions:
                dim = (f"size={r.size}" if r.size
                       else f"radius={r.radius}" if r.radius
                       else "")
                desc = (f"{r.name}: {r.shape} at {r.position}"
                        + (f", {dim}" if dim else "")
                        + f", interaction={r.interaction}")
                lines.append(f"    - {desc}")
            if self.map_spec.spawn_points:
                lines.append("  Spawn points:")
                for name, pos in self.map_spec.spawn_points.items():
                    lines.append(f"    - {name}: {pos}")

        # Requirements
        lines.append("")
        lines.append("REQUIREMENTS:")
        for i, req in enumerate(self.requirements, 1):
            lines.append(f"  {i}. {req}")

        if include_decomposer_notes and self.decomposer_notes:
            lines.append("")
            lines.append("CONTRACT-WRITING NOTES:")
            for note in self.decomposer_notes:
                lines.append(f"  - {note}")

        return "\n".join(lines)
