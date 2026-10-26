"""Environment transfer: one synthesized controller, many random physical maps.

Our correctness guarantees live at the DISCRETE contract layer (the arbiter and
worker FSMs are defined over atomic propositions such as at_dock / crossing / grant,
not over coordinates), so a controller synthesized once is independent of the map
geometry; the physical layer (CBF-QP) re-solves per map. This experiment makes that
concrete: we take ONE synthesized controller per task and run it on K random map
layouts (one per seed -- the simulators randomize shelf/dock/station/rest positions by
seed), reporting, per map, whether the task completes and whether mutual exclusion
holds. A controller that completes with zero resource violations on every random map
demonstrates transfer across physical environments for a fixed task.

Run:  python experiments/env_transfer.py [--seeds 8]
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_dock(n, seed, steps):
    from simulator.two_layer_dock import run, Config
    if n == 2:
        cfg = Config("two_layer_dock", n_robots=2, n_demands=4, equiv=2, seed=seed,
                     deadline=40.0, dock_hw=46.0, dock_hh=34.0)
        fsm = ROOT + "/outputs/two_layer_dock_n2/workdir"
    else:
        cfg = Config("two_layer_dock3", n_robots=3, n_demands=4, equiv=2, seed=seed,
                     deadline=55.0, dock_hw=50.0, dock_hh=36.0)
        fsm = ROOT + "/outputs/two_layer_dock_n3/workdir"
    w = run(cfg, fsm_dir=fsm, max_steps=steps)
    m = w.metrics()
    feasible = bool(w.report.get("assignment"))
    return {"done": bool(w.all_done()), "violations": int(m.get("violations", -1)),
            "delivered": int(m.get("delivered", -1)), "feasible": feasible}


def run_charger(n, seed, steps):
    from simulator.charger import run, Config
    w = run(Config(n_robots=n, seed=seed, n_recharges=1), max_steps=steps)
    m = w.metrics()
    return {"done": bool(w.all_done()),
            "violations": int(m.get("charger_violations", -1)),
            "recharges": int(m.get("total_recharges", -1)),
            "max_concurrent": int(m.get("max_charging", -1)), "feasible": True}


TASKS = [
    ("dock_2r",   lambda s, st: run_dock(2, s, st),   6000),
    ("charger_3r", lambda s, st: run_charger(3, s, st), 12000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=f"{ROOT}/experiments/env_transfer_results.json")
    ap.add_argument("--task", default=None, choices=[t[0] for t in TASKS],
                    help="run only this task (default: all tasks)")
    a = ap.parse_args()
    seeds = list(range(1, a.seeds + 1))
    results = {}
    tasks = [t for t in TASKS if a.task is None or t[0] == a.task]
    for name, fn, steps in tasks:
        runs = []
        for s in seeds:
            try:
                r = fn(s, steps)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"}
            r["seed"] = s
            runs.append(r)
            tag = ("ok" if r.get("done") and r.get("violations") == 0
                   else "err" if "error" in r else "partial")
            print(f"[{name}] seed {s}: {tag} {r}", flush=True)
        ok = [r for r in runs if r.get("feasible") and r.get("violations") == 0]
        done = [r for r in runs if r.get("done")]
        results[name] = {
            "maps": len(seeds),
            "feasible_maps": sum(1 for r in runs if r.get("feasible")),
            "violation_free_maps": len(ok),
            "task_completed_maps": len(done),
            "runs": runs,
        }
        print(f"  => {name}: {len(ok)}/{len(seeds)} maps violation-free, "
              f"{len(done)}/{len(seeds)} completed the task", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
