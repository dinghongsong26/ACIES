#!/bin/bash
# Case-study instance HIERARCHICAL DOCK: the RQ4 2^N coordination wall is a
# FLAT-architecture artifact, not a limit of certified decomposition.
#
# The flat gen_8r_1res decomposition certifies with a single arbiter of |M| = 3072
# (state ~ 2^N). Replacing that one node with a TWO-LEVEL ARBITER TREE -- two group
# arbiters (robots 1-4 / 5-8) under a root arbiter -- keeps every certificate and
# shrinks the coordination kernel to 288 + 288 + 12 = 588 states (largest single
# component 288, a 10.7x reduction):
#
#   refinement : tree composition refines the ORIGINAL flat System contract (2x true)
#   synthesis  : GroupArb1 |M|=288, GroupArb2 |M|=288, RootArb |M|=12 (all GR(1) ok)
#   gate       : all three arbiters well-separated
#
# The group-level demand/grant signals (greq_gk / ggrant_gk) are NEW internal
# interface variables invented at this level; the discharge chain stays acyclic
# (robot <- group <- root, root assumption-free), mirroring the flat terminal family.
# Design notes that matter: the upward demand aggregator must be one-step DELAYED
# (an instantaneous iff clashes with the canonical initial pin at t=0).
#
# This instance is the certified feasibility demo for paper-1 (RQ4 extension /
# future work); the AUTOMATIC version (state-cap trigger -> LLM re-architects the
# node) is paper-2 material.
set -u
cd "$(git rev-parse --show-toplevel)"
set -a; source ./.env 2>/dev/null; set +a
CE="${PYTHON:-python}"  # chase_env python (see the top-level README)
OUT=outputs/instances/hierarchical_dock
mkdir -p "$OUT"

# Requires the flat certified run to exist (produced by the RQ4/P0 sweeps):
#   outputs/sweep/gen_8r_1res_arbiter_justice__seed0/workdir/_tmp.ltl
# To regenerate it (COSTS LLM CALLS, ~$0.4):
#   $CE experiments/sweep_scalability.py --cells 8,1,arbiter,justice --seeds 1 --model gemini

echo ">>> [hierarchical_dock] tree certificates (refinement + synthesis + gate; no LLM)"
$CE experiments/hierarchy_replay.py --synth 2>&1 | grep -E "tree8" | tee "$OUT/certificates.txt"
echo ">>> [hierarchical_dock] artifacts -> outputs/hierarchy_replay/ (tree.ltl, FSM jsons)"
cp outputs/hierarchy_replay/tree.ltl "$OUT/tree.ltl" 2>/dev/null
echo ">>> [hierarchical_dock] done -> $OUT"

# ── PHYSICAL layer: N=8 dock simulation under the TWO-LEVEL arbiter tree ──────
# Requires the tree FSMs in outputs/hierarchy_replay/ (grouparb1/2.json,
# rootarb.json, robot1..8.json — robots 2-8 synthesize from the existing
# .structuredSlugs via ToolPipeline.synthesize_controller; SLUGS only, no LLM).
echo ">>> [hierarchical_dock] physical headless verification (8000 steps; both-group deliveries + dock mutex + no starvation)"
$CE simulator/tree_dock.py --steps 8000 2>&1 | tee "$OUT/physical_headless.txt"
echo ">>> [hierarchical_dock] GIF render -> $OUT/tree8.gif"
$CE simulator/tree_dock.py --gif "$OUT/tree8.gif" 2>&1 | tail -3
ls -la "$OUT/tree8.gif"
