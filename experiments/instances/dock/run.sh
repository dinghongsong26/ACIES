#!/bin/bash
# Case-study instance DOCK: a TERMINAL shared resource (two warehouse robots + one
# shared loading dock/shelf, synthesized worker FSMs + arbiter FSM).
# Three phases: (a) SYNTHESIS (commented out: costs LLM calls) -> (b) EVALUATION -> (c) GIF.
cd "$(git rev-parse --show-toplevel)"
set -a; source ./.env 2>/dev/null; set +a
CE="${PYTHON:-python}"  # chase_env python (see the top-level README)
OUT=outputs/instances/dock
mkdir -p "$OUT"

# ── (a) SYNTHESIS ─────────────────────────────────────────────────────────────
# Would (re)produce the FSMs via the LLM decomposition pipeline. COSTS LLM CALLS,
# so it stays commented out; the certified artifacts already exist at:
#   golden/warehouse/{robot1,robot2,arbiter}.json   committed known-good copy (+ specs.ltl)
#   outputs/warehouse_n2/workdir/*.json              identical working copy (+ .slugsin/.smv/.chase)
# The dock-family arbiters the RQ2 harness compares against live in
#   outputs/sweep/gen_{2..5}r_1res_arbiter_justice__seed0/workdir/arbiter.json
#
# $CE main.py --task warehouse --n 2 --resynth

# ── (b) EVALUATION ────────────────────────────────────────────────────────────
# Terminal-resource guarantees (mutex / no-spurious-grant / no-starvation) under an
# ADVERSARIAL request scheduler, ours vs plan baselines (greedy / priority / TDMA).
# Quick variant:  add  --n 3 --sample 20000
echo ">>> [dock] adversarial-scheduler harness (rq2_adversarial)"
$CE experiments/rq2_adversarial.py --out "$OUT/rq2_results.json"

# The LLM-authored arbitration rule (longest-wait-first) row, run through BOTH the
# terminal and the passage harnesses (~1 min).
echo ">>> [dock] LLM-rule row (rq2_llm_row)"
$CE experiments/rq2_llm_row.py | tee "$OUT/rq2_llm_row.txt"

# ── (c) GIF ───────────────────────────────────────────────────────────────────
echo ">>> [dock] rendering simulation GIF (offscreen)"
$CE simulator/warehouse.py \
  --robot1 golden/warehouse/robot1.json \
  --robot2 golden/warehouse/robot2.json \
  --arbiter golden/warehouse/arbiter.json \
  --headless 3000 --gif "$OUT/dock.gif"
echo ">>> [dock] done -> $OUT"
