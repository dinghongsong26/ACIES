#!/bin/bash
# Case-study instance CHARGER: the TERMINAL resource GROUNDED in a continuous physical
# state -- a shared charging station only one robot may use at a time; the trigger is
# the battery level abstracted to a discrete `battery_low` predicate.
# Three phases: (a) SYNTHESIS (commented out: costs LLM calls) -> (b) EVALUATION -> (c) GIF.
cd "$(git rev-parse --show-toplevel)"
set -a; source ./.env 2>/dev/null; set +a
CE="${PYTHON:-python}"  # chase_env python (see the top-level README)
OUT=outputs/instances/charger
mkdir -p "$OUT"

# ── (a) SYNTHESIS ─────────────────────────────────────────────────────────────
# Would (re)produce the FSMs via the LLM decomposition pipeline. COSTS LLM CALLS,
# so it stays commented out; the certified artifacts already exist at:
#   outputs/charger_n3/workdir/{robot1,robot2,robot3,arbiter}.json  (+ specs.ltl)
# Intermediate results of that run:
#   outputs/charger_n3/decomposer_output.json   contract rounds + gate catches
#   outputs/charger_n3/translator_output.json   NL -> proposition/role translation
#   outputs/charger_n3/result.json              token usage / elapsed time / FSM list
#
# $CE main.py --task charger --n 3 --resynth

# ── (b) EVALUATION ────────────────────────────────────────────────────────────
# Environment transfer: ONE synthesized controller on K random maps (station and rest
# points re-randomized per seed); charger part only via the --task flag.
echo ">>> [charger] environment-transfer harness (env_transfer --task charger_3r)"
$CE experiments/env_transfer.py --task charger_3r --seeds 8 \
  --out "$OUT/env_transfer_charger.json"

# ── (c) GIF ───────────────────────────────────────────────────────────────────
echo ">>> [charger] rendering simulation GIF (offscreen)"
$CE simulator/charger.py --seed 3 --gif "$OUT/charger.gif"
echo ">>> [charger] done -> $OUT"
