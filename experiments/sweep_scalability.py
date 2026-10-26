"""Scalability sweep for the resource-conflict decomposition.

Generates resource-conflict tasks over (N robots × M shared resources × coordinator ×
coordination profile), runs the TASK (discrete) layer only, and records one metrics row per
run: realizability, number of LLM feedback rounds, refinement/synthesis pass counts, FSM
sizes, wall-clock, tokens, and $ cost.

Env: source the toolchain first (NUXMV / SLUGS / LOGICS_TOOL / DYLD_LIBRARY_PATH + an LLM
key), e.g.  `set -a; source /path/to/.env; set +a`.

Examples:
    # one validation cell
    python experiments/sweep_scalability.py --cells 5,1,arbiter,justice --model gemini

    # a small grid, 1 seed each
    python experiments/sweep_scalability.py \
        --grid-N 3,4,5 --grid-M 1 --profile justice --seeds 1 --model gemini

Each cell calls main.py --no-sim so synthesis is the only thing measured. Results append to
experiments/sweep_results.csv (and a per-run JSON snapshot) so a crash mid-sweep loses nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contrauto.tasks.gen_resource_task import gen_resource_task, task_name  # noqa: E402

RESULTS_CSV = ROOT / "experiments" / "sweep_results.csv"
FIELDS = [
    "task", "N", "M", "coordinator", "profile", "seed", "model",
    "realizable", "n_robots_synth", "fsm_states_total", "fsm_states_per",
    "feedback_rounds", "refinement_pass", "refinement_fail",
    "synthesis_pass", "synthesis_fail",
    "elapsed_sec", "input_tokens", "output_tokens", "cost_usd", "error",
]


def _count_steps(step_logs: list[dict]) -> dict:
    """Derive scalability metrics from the decomposer's per-step log."""
    def _is(step, name):  # decomposer step labels look like 'Step4:Refinement'
        return name.lower() in step.get("step", "").lower()

    rounds = sum(1 for s in step_logs if _is(s, "GenerateContracts"))
    ref_pass = sum(1 for s in step_logs if _is(s, "Refinement") and s.get("passed"))
    ref_fail = sum(1 for s in step_logs if _is(s, "Refinement") and not s.get("passed"))
    syn_pass = sum(1 for s in step_logs if _is(s, "Synthesis") and s.get("passed"))
    syn_fail = sum(1 for s in step_logs if _is(s, "Synthesis") and not s.get("passed"))
    return {
        "feedback_rounds": rounds,
        "refinement_pass": ref_pass, "refinement_fail": ref_fail,
        "synthesis_pass": syn_pass, "synthesis_fail": syn_fail,
    }


def _fsm_sizes(fsm_files: dict) -> tuple[int, str]:
    """Total FSM states and a per-controller breakdown string."""
    total, per = 0, []
    for name, path in sorted(fsm_files.items()):
        try:
            data = json.load(open(path))
            # SLUGS-style FSM json: a dict of states (possibly under a 'nodes'/'states' key).
            if isinstance(data, dict):
                states = data.get("nodes") or data.get("states") or data
                n = len(states) if isinstance(states, (dict, list)) else 0
            else:
                n = len(data)
            total += n
            per.append(f"{name}:{n}")
        except Exception:
            per.append(f"{name}:?")
    return total, " ".join(per)


def run_cell(N, M, coordinator, profile, seed, model, outroot, k=2) -> dict:
    name = task_name(N, M, coordinator, profile)
    label = f"{name}__seed{seed}"
    yaml_path = gen_resource_task(N, M, coordinator, profile, k=k)
    outdir = Path(outroot) / label

    row = {f: "" for f in FIELDS}
    row.update({"task": name, "N": N, "M": M, "coordinator": coordinator,
                "profile": profile, "seed": seed, "model": model})

    print(f"\n{'='*70}\n  CELL  {label}   (N={N} M={M} {coordinator}/{profile})\n{'='*70}")
    t0 = time.time()
    try:
        # Force re-synthesis so each seed is an independent measurement.
        cmd = [sys.executable, str(ROOT / "main.py"), "--task", yaml_path,
               "--no-sim", "--resynth", "--model", model, "--outdir", str(outdir)]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            row["error"] = f"exit{proc.returncode}: {proc.stderr.strip()[-300:]}"
    except subprocess.TimeoutExpired:
        row["error"] = "timeout"
    row["elapsed_sec"] = round(time.time() - t0, 1)

    # Parse whatever the run produced (result.json + decomposer_output.json).
    res_path = outdir / "result.json"
    dec_path = outdir / "decomposer_output.json"
    if res_path.exists():
        res = json.load(open(res_path))
        fsm_files = res.get("fsm_files", {})
        tot, per = _fsm_sizes(fsm_files)
        tu = res.get("token_usage", {})
        row.update({
            "realizable": bool(fsm_files),
            "n_robots_synth": len(fsm_files),
            "fsm_states_total": tot, "fsm_states_per": per,
            "input_tokens": tu.get("total_input_tokens", ""),
            "output_tokens": tu.get("total_output_tokens", ""),
            "cost_usd": tu.get("total_cost_usd", ""),
        })
    else:
        row["realizable"] = False
    if dec_path.exists():
        dec = json.load(open(dec_path))
        row.update(_count_steps(dec.get("step_logs", [])))

    _append_row(row)
    print(f"  -> realizable={row['realizable']} rounds={row.get('feedback_rounds')} "
          f"elapsed={row['elapsed_sec']}s cost=${row.get('cost_usd')}")
    return row


def _append_row(row: dict) -> None:
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def _parse_cells(spec: str):
    """'5,1,arbiter,justice ; 3,2,arbiter,priority' -> list of tuples."""
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        N, M, coord, prof = [x.strip() for x in chunk.split(",")]
        out.append((int(N), int(M), coord, prof))
    return out


def main():
    p = argparse.ArgumentParser(description="Resource-conflict scalability sweep")
    p.add_argument("--cells", help="explicit cells 'N,M,coord,profile' separated by ';'")
    p.add_argument("--grid-N", help="comma list of N for a grid, e.g. 3,4,5")
    p.add_argument("--grid-M", default="1", help="comma list of M for a grid")
    p.add_argument("--coordinator", default="arbiter")
    p.add_argument("--profile", default="justice")
    p.add_argument("--seeds", type=int, default=1, help="repeats per cell")
    p.add_argument("--model", default="gemini")
    p.add_argument("-k", type=int, default=2)
    p.add_argument("--outroot", default="outputs/sweep")
    a = p.parse_args()

    cells = []
    if a.cells:
        cells = _parse_cells(a.cells)
    elif a.grid_N:
        Ns = [int(x) for x in a.grid_N.split(",")]
        Ms = [int(x) for x in a.grid_M.split(",")]
        cells = [(N, M, a.coordinator, a.profile) for N in Ns for M in Ms]
    else:
        p.error("provide --cells or --grid-N")

    print(f"Sweep: {len(cells)} cell(s) × {a.seeds} seed(s) = {len(cells)*a.seeds} runs, model={a.model}")
    for (N, M, coord, prof) in cells:
        for seed in range(a.seeds):
            run_cell(N, M, coord, prof, seed, a.model, a.outroot, k=a.k)
    print(f"\nDone. Metrics -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
