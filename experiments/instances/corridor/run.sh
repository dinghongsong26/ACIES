#!/bin/bash
# Case-study instance CORRIDOR: a PASSAGE resource (single-lane corridor) -- the grant
# must be HELD for the whole crossing (hold-until-release lock), unlike the terminal
# dock/charger where a memoryless mutex+justice arbiter suffices.
# Three phases: (a) SYNTHESIS (commented out) -> (b) EVALUATION -> (c) GIF.
cd "$(git rev-parse --show-toplevel)"
set -a; source ./.env 2>/dev/null; set +a
CE="${PYTHON:-python}"  # chase_env python (see the top-level README)
OUT=outputs/instances/corridor
mkdir -p "$OUT"

# ── (a) SYNTHESIS ─────────────────────────────────────────────────────────────
# FULL LLM END-TO-END — CERTIFIED (2026-07-09): with req declared a WORLD signal
# (dir: world in contrauto/tasks/corridor.yaml) and the three general contract-design
# rules in the decomposer prompt (no-U / world-signals-only-in-assumptions /
# liveness-where-forceable), the LLM decomposition passes refinement + GR(1)
# synthesis + the well-separatedness gate in a SINGLE round (~$0.28):
#   Arbiter |M|=24 (== the hand-authored passage arbiter), Robots |M|=34 each,
#   hold-until-release G(grant_i & req_i -> X grant_i) invented by the LLM.
# Reproduce (COSTS LLM CALLS, ~$0.3; results land in outputs/corridor_e2e):
#   $CE main.py --task contrauto/tasks/corridor.yaml --model gemini --resynth \
#       --no-sim --outdir outputs/corridor_e2e
# WHY the ownership matters (the diagnosis trail, reproducible offline at zero cost —
# variants baseline..v12 in experiments/corridor_replay.py): with req on the robots'
# OUTPUT side, "eventual release" is a robot obligation that holds only if it is
# eventually granted, while the arbiter's fairness needs eventual release — a circular
# liveness discharge that non-circular contract refinement rightly rejects (the
# counterexample is the mutual-default trace: req held forever, nothing granted,
# nobody at fault). Legacy no-LLM route (hand-authored arbiter spec, SLUGS only):
#   $CE experiments/passage_arbiter.py --n 2 3 4 5   -> outputs/corridor_direct/

# ── (b) EVALUATION ────────────────────────────────────────────────────────────
# Passage workload harness: mutex / NON-PREEMPTION / starvation under adversarial
# crossing streams; ours (passage arbiter) vs terminal-dock arbiter and baselines.
echo ">>> [corridor] passage-workload harness (rq2_passage)"
$CE experiments/rq2_passage.py --out "$OUT/rq2_passage_results.json"

# ── (c) GIF ───────────────────────────────────────────────────────────────────
echo ">>> [corridor] rendering simulation GIF (offscreen)"
$CE simulator/corridor.py --seed 3 --gif "$OUT/corridor.gif"
# E2E-FSM physical run (2026-07-10): the corridor_e2e certified FSMs with WORLD-driven
# req (simulator/corridor.py run_e2e / --e2e).  NOTE the certified arbiter.json can NOT
# drive a service-coupled world -- its encoding G(grant->req) + G(grant & req -> X grant)
# (unprimed req) makes every granting state losing, so the extracted strategy NEVER
# grants and the loop deadlocks (0 crossings; see the module note in simulator/corridor.py).
# arbiter_fixed.json = same contract with the antecedent primed (grant & req & X req ->
# X grant), re-synthesized offline with slugs --biasForAction (no LLM).  Result @6000
# steps: crossings 8/8/8, 0 corridor mutex violations, max 1 inside, 0 ungranted ticks:
$CE simulator/corridor.py --e2e --seed 3 --steps 6000 \
    --arbiter outputs/corridor_e2e/workdir/arbiter_fixed.json   # -> $OUT/corridor_e2e.gif
# corridor_e2e4 (2026-07-11): the re-certified FSMs fix the arbiter (primed hold rule +
# env justice GF !(grant_i & req_i), 36/60 granting nodes, genuinely eager) but the repair
# rounds DROPPED the robots' justice guarantee GF(!req_i | crossing_i) (present in
# robot{i}_core.slugsin, absent in robot{i}.slugsin / _tmp.ltl), so the certified
# robot{i}.json have ZERO crossing=1 nodes out of 32 -- one grant fires, no robot ever
# crosses, the lock is never released (grant_edges=1, crossings=0).  Refinement still
# passes because the System assumption GF(!req_i) discharges the service liveness without
# service.  robot{i}_fixed.json = robot{i}.slugsin + that one justice line, slugs only
# (no LLM); they come out node-for-node IDENTICAL to the certified corridor_e2e robots.
# Result @6000 steps: crossings 8/8/8, 0 violations, max 1 inside, 0 ungranted ticks:
$CE simulator/corridor.py --e2e --seed 3 --steps 6000 --fsm-dir outputs/corridor_e2e4/workdir \
    --robot-suffix _fixed --gif "$OUT/corridor_e2e4.gif"        # certified e2e4 arbiter.json
# corridor_e2e6 (2026-07-11): FINAL pipeline FSMs (--biasForAction default, all
# certificates + both activity gates green) run AS-IS -- no fixed artifacts needed.
# NOTE the e2e6 robot contract adds the one-step crossing dynamics assumption
# G(crossing -> X at_b); with at_b persistence + G(!(at_b & crossing)) this makes at_b
# ABSORBING: each robot crosses AT MOST ONCE ever and parks on side B (a one-shot
# relocation service, not a shuttle -- crossings [1,1,1] IS the contract maximum).
# The adapter auto-detects this world model from the strategy table (E2ERobot.one_shot)
# and raises at_b the tick after the crossing output; feeding at_b=0 there is an illegal
# env move (it caused ~5900/6000 off-model stalls per robot before the fix).
# Result @6000 steps: crossings 1/1/1, 0 violations, 0 ungranted ticks, 0 stalls:
$CE simulator/corridor.py --e2e --seed 3 --steps 3000 --fsm-dir outputs/corridor_e2e6/workdir \
    --gif "$OUT/corridor_e2e6.gif"
# corridor_e2e7 (2026-07-11, GRAND SLAM): certified FSMs run AS-IS with FULL repeated
# shuttle service.  The contract keeps the one-step physics G(crossing -> X at_b) but
# swaps at_b persistence for GF(!at_b) ("being at side B is temporary"), and crossing is
# now FORCED by G(req & grant & !at_a & !at_b -> crossing); crossing nodes are RECURRENT
# in the strategy graph.  The adapter's world-model detection is two-flag (E2ERobot):
# forced_atb (all crossing-node successors have at_b=1) x one_shot (crossing nodes not
# reachable again) -> e2e7 is the pulse-shuttle case: raise at_b the tick after the
# crossing, DROP it when the robot leaves side B on the return lane (holding it forever
# would violate GF(!at_b) -- the mirror image of the e2e6 bug).
# Result @6000 steps: crossings 8/8/8, 0 violations, 0 ungranted ticks, 0 stalls:
$CE simulator/corridor.py --e2e --seed 3 --steps 6000 --fsm-dir outputs/corridor_e2e7/workdir \
    --gif "$OUT/corridor_e2e7.gif"
echo ">>> [corridor] done -> $OUT"
