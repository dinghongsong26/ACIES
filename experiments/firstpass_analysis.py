"""First-pass acceptance analysis over the sweep grid (the paper's planned headline).

Reads experiments/sweep_results.csv (written by sweep_scalability.py), derives per-run
first-pass acceptance -- the raw LLM draft clears every certificate with no repair,
i.e. realizable && feedback_rounds == 1 -- and reports:

  1. a per-cell table (N, M, profile): first-pass rate, repair rounds, failures;
  2. a logistic regression of first-pass acceptance on standardized difficulty axes
     (N, M, profile dummies vs the justice baseline), the analog of the
     complexity-regression in the NL-to-LTL benchmark of Bayat et al. (arXiv:2606.30163).

'priority' rows are the labeled NEGATIVE instances (strict-priority + justice is
unrealizable): they are excluded from the acceptance regression and reported separately
as "correctly reported unrealizable" (no FSM emitted) vs "hallucinated a controller".
Rows with an infrastructure error (timeout / nonzero exit) are excluded from both.

Run:  python experiments/firstpass_analysis.py [--csv experiments/sweep_results.csv]
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("error", "").strip():
                continue  # infra failure (timeout etc.), not a decomposition outcome
            try:
                rows.append({
                    "N": int(r["N"]), "M": int(r["M"]),
                    "profile": r["profile"],
                    "realizable": r["realizable"].strip().lower() == "true",
                    "rounds": int(r["feedback_rounds"] or 0),
                    "model": r.get("model", ""),
                })
            except (KeyError, ValueError):
                continue
    return rows


def cell_table(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["profile"], r["M"], r["N"])].append(r)
    print(f"{'profile':12s} {'M':>2s} {'N':>2s} {'runs':>5s} {'first-pass':>10s} "
          f"{'accepted':>9s} {'med.rounds':>10s}")
    for key in sorted(cells):
        rs = cells[key]
        fp = sum(1 for x in rs if x["realizable"] and x["rounds"] == 1)
        acc = sum(1 for x in rs if x["realizable"])
        rounds = sorted(x["rounds"] for x in rs if x["realizable"])
        med = rounds[len(rounds) // 2] if rounds else "-"
        print(f"{key[0]:12s} {key[1]:>2d} {key[2]:>2d} {len(rs):>5d} "
              f"{fp:>4d}/{len(rs):<5d} {acc:>4d}/{len(rs):<4d} {str(med):>10s}")


def logistic_irls(X, y, iters=50, tol=1e-8):
    """Plain Newton-Raphson IRLS; X includes the intercept column."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = p * (1 - p)
        # ridge epsilon keeps the Hessian invertible under (quasi-)separation
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def regression(rows):
    pos = [r for r in rows if r["profile"] != "priority"]
    if len(pos) < 20:
        print(f"[regression skipped: only {len(pos)} usable rows]")
        return
    profiles = sorted({r["profile"] for r in pos} - {"justice"})
    feats, names = [], []
    N = np.array([r["N"] for r in pos], float)
    M = np.array([r["M"] for r in pos], float)
    for name, v in (("N", N), ("M", M)):
        feats.append((v - v.mean()) / (v.std() or 1.0))
        names.append(name)
    for pr in profiles:  # dummies vs justice baseline (already 0/1, leave unscaled)
        feats.append(np.array([1.0 if r["profile"] == pr else 0.0 for r in pos]))
        names.append(f"profile={pr}")
    X = np.column_stack([np.ones(len(pos))] + feats)
    y = np.array([1.0 if (r["realizable"] and r["rounds"] == 1) else 0.0 for r in pos])
    beta = logistic_irls(X, y)
    print(f"\nLogistic regression of FIRST-PASS acceptance ({len(pos)} runs, "
          f"{int(y.sum())} accepted). Standardized coefficients:")
    print(f"  {'(intercept)':16s} {beta[0]:+.3f}")
    for name, b in zip(names, beta[1:]):
        print(f"  {name:16s} {b:+.3f}")
    print("  (negative = this axis makes first-pass acceptance less likely)")


def negatives(rows):
    neg = [r for r in rows if r["profile"] == "priority"]
    if not neg:
        return
    correct = sum(1 for r in neg if not r["realizable"])
    print(f"\nNegative instances (priority, expected unrealizable): "
          f"{correct}/{len(neg)} correctly reported unrealizable; "
          f"{len(neg) - correct} emitted a controller anyway.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "experiments/sweep_results.csv"))
    a = ap.parse_args()
    rows = load_rows(a.csv)
    print(f"{len(rows)} usable runs from {a.csv}\n")
    cell_table(rows)
    regression(rows)
    negatives(rows)


if __name__ == "__main__":
    main()
