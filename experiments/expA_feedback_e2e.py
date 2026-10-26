"""Experiment A: end-to-end convergence WITH vs WITHOUT the feedback harness.

Unlike benchmark_feedback.py (task layer only), this drives the FULL two-layer
pipeline via `main.py` as a subprocess — so each run includes the simulator
(delivered count) and the channel-5 monitor loop — and harvests metrics from the
tee'd log. Runs are resumable: a finished run (log ends with RUN COMPLETE) is
skipped, so the batch can be interrupted and relaunched.

Usage (from repo root):
    python experiments/expA_feedback_e2e.py --runs 3 --model gemini
    python experiments/expA_feedback_e2e.py --tasks warehouse --runs 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# warehouse first: it is the focus task (refinement-diagnosis fix validation).
DEFAULT_TASKS = ["warehouse", "patrol", "static_2r_3b", "dynamic_2r_3b"]

# Toolchain locations come from the environment (see .env.example); no defaults here.
TOOLCHAIN = {k: os.environ[k] for k in
             ("NUXMV", "SLUGS", "SLUGS_COMP", "LOGICS_TOOL", "DYLD_LIBRARY_PATH")
             if k in os.environ}


def _load_env():
    env = os.environ.copy()
    env.update(TOOLCHAIN)
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def parse_log(log_path: Path) -> dict:
    """Harvest metrics from one run's plain-text log (report.py piped output)."""
    text = log_path.read_text(errors="replace")
    m = {}
    m["attempts"] = len(re.findall(r"\[Decomposer\] Attempt \d+", text))
    # Last synthesis step line wins (channel 5 may re-synthesize).
    synth_lines = re.findall(r"[✓✗]\s+synthesis\s+—\s+(.*)", text)
    m["fsm"] = bool(synth_lines) and "no FSM" not in synth_lines[-1]
    m["channel5_fired"] = text.count("violation (channel 5)")
    # Last PHYSICAL EXECUTION summary: "✓ PHYSICAL EXECUTION   ·   delivered 2   violations 0"
    phys = re.findall(r"PHYSICAL EXECUTION\s+·\s+(.*)", text)
    if phys:
        for k, v in re.findall(r"(\w+)\s+(-?[\d.]+)", phys[-1]):
            m[k] = float(v) if "." in v else int(v)
    return m


def run_one(task, cond, run_id, model, base, timeout, env):
    tag = f"{task}_{cond}_run{run_id}"
    outdir = base / tag
    log = base / f"{tag}.log"

    if log.exists() and "RUN COMPLETE" in log.read_text(errors="replace"):
        print(f"[skip] {tag} (already complete)")
        return None

    cmd = [sys.executable, "-u", "main.py", "--task", task, "--resynth",
           "--model", model, "--save-attempts", "--outdir", str(outdir)]
    if cond == "nofb":
        cmd.append("--no-feedback")

    print(f"[run ] {tag} ...", flush=True)
    t0 = time.time()
    timed_out = False
    with open(log, "w") as fh:
        try:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = -1, True
    elapsed = round(time.time() - t0, 1)
    with open(log, "a") as fh:
        fh.write(f"\nRUN COMPLETE rc={rc} timed_out={timed_out} elapsed={elapsed}\n")

    rec = {"task": task, "cond": cond, "run": run_id, "model": model,
           "rc": rc, "timed_out": timed_out, "elapsed_s": elapsed}
    rec.update(parse_log(log))
    status = "FSM+sim ok" if rec.get("fsm") else "no FSM"
    print(f"[done] {tag}: {status}  attempts={rec.get('attempts')} "
          f"delivered={rec.get('delivered', rec.get('cleared', '-'))} ({elapsed}s)", flush=True)
    return rec


def harvest_all(base: Path) -> list:
    """Rebuild the full results list from every finished log (idempotent)."""
    out = []
    for log in sorted(base.glob("*_run*.log")):
        text = log.read_text(errors="replace")
        done = re.search(r"RUN COMPLETE rc=(-?\d+) timed_out=(\w+) elapsed=([\d.]+)", text)
        if not done:
            continue
        stem = log.stem
        mt = re.match(r"(.+)_(fb|nofb)_run(\d+)$", stem)
        if not mt:
            continue
        rec = {"task": mt.group(1), "cond": mt.group(2), "run": int(mt.group(3)),
               "rc": int(done.group(1)), "timed_out": done.group(2) == "True",
               "elapsed_s": float(done.group(3))}
        rec.update(parse_log(log))
        out.append(rec)
    return out


def summarize(records):
    from collections import defaultdict
    cells = defaultdict(list)
    for r in records:
        cells[(r["task"], r["cond"])].append(r)
    print(f"\n{'task':<16}{'cond':<6}{'conv':<8}{'attempts':<10}{'delivered':<10}{'ch5':<5}")
    for (task, cond), rs in sorted(cells.items()):
        conv = sum(1 for r in rs if r.get("fsm"))
        att = [r.get("attempts", 0) for r in rs]
        del_ = [r.get("delivered", r.get("cleared", 0)) for r in rs]
        ch5 = sum(r.get("channel5_fired", 0) for r in rs)
        print(f"{task:<16}{cond:<6}{conv}/{len(rs):<6}"
              f"{sum(att)/len(att):<10.1f}{sum(del_)/len(del_):<10.1f}{ch5:<5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default="gemini")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--base", default="outputs/expA_0730")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()

    base = ROOT / a.base
    base.mkdir(parents=True, exist_ok=True)
    results_path = base / "results.json"

    if a.summarize_only:
        records = harvest_all(base)
        summarize(records)
        return

    env = _load_env()
    # run-index OUTER loop: after each sweep every cell has equal N, so interim
    # summaries are balanced instead of exhausting one cell first.
    for run_id in range(1, a.runs + 1):
        for task in a.tasks:
            for cond in ("fb", "nofb"):
                run_one(task, cond, run_id, a.model, base, a.timeout, env)
                records = harvest_all(base)
                results_path.write_text(json.dumps(records, indent=2))
    summarize(harvest_all(base))


if __name__ == "__main__":
    main()
