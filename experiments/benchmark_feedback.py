"""Benchmark: compare refinement repair WITH vs WITHOUT DFA-based feedback.

Runs each task twice (use_dfa_feedback True/False) and measures success + time + tokens.

Usage (set env vars first, see .env.example):
    python experiments/benchmark_feedback.py --task warehouse --model gemini
    python experiments/benchmark_feedback.py --all --model gemini
    python experiments/benchmark_feedback.py --all --model gemini --runs 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contrauto.pipeline import ContrAuto

MODELS = ["gpt", "claude", "gemini", "deepseek", "glm", "kimi"]


def _task_names():
    return sorted(p.stem for p in Path("contrauto/tasks").glob("*.yaml"))


def run_single(task_name, model_key, use_feedback, run_id):
    label = "WITH_FEEDBACK" if use_feedback else "NO_FEEDBACK"
    outdir = f"benchmark_outputs/{task_name}_{model_key}_{label}_run{run_id}"

    t0 = time.time()
    try:
        result = ContrAuto(task=task_name, outdir=outdir, model=model_key,
                           use_dfa_feedback=use_feedback).run()
        elapsed = time.time() - t0
        success = bool(result.get("fsm_files"))
    except Exception as e:
        elapsed = time.time() - t0
        result = {"error": str(e)}
        success = False

    record = {
        "task": task_name,
        "model": model_key,
        "feedback": use_feedback,
        "run_id": run_id,
        "success": success,
        "elapsed_sec": round(elapsed, 1),
        "token_usage": result.get("token_usage", {}),
        "error": result.get("error"),
    }

    summary_path = Path(outdir) / "benchmark_record.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  [{label}] {task_name} / {model_key} / run {run_id}")
    print(f"  Success: {success}  |  Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")
    return record


def main():
    parser = argparse.ArgumentParser(description="Benchmark feedback vs no-feedback")
    parser.add_argument("--task", default=None, choices=_task_names())
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--model", default="gemini", choices=MODELS)
    parser.add_argument("--runs", type=int, default=1, help="Runs per configuration")
    args = parser.parse_args()

    tasks = _task_names() if args.all else [args.task or "warehouse"]
    records = []
    for task in tasks:
        for run_id in range(1, args.runs + 1):
            for use_fb in [True, False]:
                records.append(run_single(task, args.model, use_fb, run_id))

    out_path = Path("benchmark_outputs/summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Task':<18} {'Feedback':<12} {'Success':<10} {'Time(s)'}")
    print("-" * 70)
    for r in records:
        fb = "YES" if r["feedback"] else "NO"
        print(f"{r['task']:<18} {fb:<12} {r['success']!s:<10} {r['elapsed_sec']}")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
