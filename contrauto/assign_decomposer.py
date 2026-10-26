"""
Task-level assignment decomposer (LLM + geometry-prompt + unified feedback chain).

L1 = task layer. The LLM gets COARSE geometry (robot/target positions, obstacle
boxes, straight-line distances) and assigns/orders targets per robot. Geometry
VALIDATES on demand (plan_waypoints detour time per hop) — straight-line close
≠ reachable. On infeasibility a NATURAL-LANGUAGE reason is fed back and the LLM
freely REDESIGNS by either:
    - reassign        : give a target to a different robot
    - change target   : cover a goal via a different EQUIVALENT instance
                        (e.g. another access point of the same shelf / dock slot)
No fixed priority between the two — the LLM explores both.

Unified fallback chain (solve()):
    L1 free explore (reassign / change-target) within deadline
      ├─ escalation A (geometric PROOF): some goal's every equivalent instance is
      │    unreachable within deadline by EVERY robot  -> L1 cannot help, relax now
      ├─ escalation B (budget): L1 explored l1_rounds with no feasible assignment
      └─ relax: bump the ONE shared deadline (xfactor) and re-run L1
    report: SATISFIED / REPAIRED(relaxed Ts->Tr) / UNSATISFIABLE

Targets may declare EQUIVALENCE CLASSES (a goal satisfiable by any one instance):
    targets = {"G0": (x, y)}                    # single instance
    targets = {"G0": [(x1,y1), (x2,y2)]}        # equivalent access points
    targets = {"G0": {"a": (x1,y1), "b": ...}}  # named instances
Each GROUP must be covered by exactly ONE of its instances.

Run:  set -a; source .env.example; set +a
      cd contrauto && python3 assign_decomposer.py
"""
from __future__ import annotations

import math

try:
    from geometry import plan_waypoints            # run from contrauto/
except ModuleNotFoundError:
    from contrauto.geometry import plan_waypoints   # imported as a package

R, VMAX, A_MAX = 12.0, 130.0, 400.0     # VMAX/A_MAX match the stl_cbf double-integrator limits


def _trapezoid_time(L, vmax=VMAX, a_max=A_MAX):
    """Time to traverse a straight length L from rest to rest under |v|<=vmax, |a|<=a_max
    (a trapezoidal / triangular speed profile). This is the CBF-provable reach-time bound
    for one segment of the 2nd-order dynamics."""
    if L <= 0:
        return 0.0
    d_acc = vmax * vmax / a_max          # distance to accelerate to vmax AND decelerate back
    if L <= d_acc:
        return 2.0 * math.sqrt(L / a_max)   # triangular: never reaches vmax
    return L / vmax + vmax / a_max          # trapezoidal: accel + cruise + decel


def path_time(a, b, rects, bounds):
    """CBF-PROVABLE deadline criterion (replaces the old optimistic L/VMAX, which assumed
    full-speed straight-line travel and UNDER-estimated execution — see verify②, where the
    geometric estimate said 9s feasible but the run took 9.8s). We sum the per-segment
    trapezoid time under the 2nd-order limits (decelerate-to-turn at each waypoint), giving a
    SOUND UPPER BOUND on the executed reach time: real turns need not stop, so actual <= this."""
    wp = plan_waypoints(tuple(a), tuple(b), rects, bounds, R)
    if not wp:
        return math.inf
    return sum(_trapezoid_time(math.hypot(wp[i + 1][0] - wp[i][0], wp[i + 1][1] - wp[i][1]))
               for i in range(len(wp) - 1))


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_xy(v):
    return isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v)


def norm_targets(targets):
    """Normalize the target spec into (inst_pos, inst_group, groups).

    inst_pos:   instance_name -> (x, y)
    inst_group: instance_name -> group_name
    groups:     group_name    -> [instance_names]
    """
    inst_pos, inst_group, groups = {}, {}, {}
    for g, val in targets.items():
        if isinstance(val, dict):
            items = [(name, tuple(p)) for name, p in val.items()]
        elif _is_xy(val):
            items = [(g, tuple(val))]
        else:                                    # list of positions
            items = [(f"{g}#{i}", tuple(p)) for i, p in enumerate(val)]
        names = []
        for name, pos in items:
            inst_pos[name], inst_group[name] = pos, g
            names.append(name)
        groups[g] = names
    return inst_pos, inst_group, groups


def build_prompt(robots, norm, rects, bounds, deadline, dock=None):
    """Build the (system, user) decomposer prompt from robots, targets, obstacles, deadline."""
    inst_pos, inst_group, groups = norm
    rb = "; ".join(f"{r}@({int(p[0])},{int(p[1])})" for r, p in robots.items())
    gl = []
    for g, names in groups.items():
        opts = " | ".join(f"{n}@({int(inst_pos[n][0])},{int(inst_pos[n][1])})" for n in names)
        gl.append(f"  {g}: {opts}" + ("   (equivalent access points — pick ONE)" if len(names) > 1 else ""))
    ob = "; ".join(f"box[x {int(c-hw)}..{int(c+hw)}, y {int(cy-hh)}..{int(cy+hh)}]"
                   for (c, cy, hw, hh) in rects) or "(none)"
    dm = []
    for r, rp in robots.items():
        dm.append(f"  {r}: " + ", ".join(f"{n}={int(euclid(rp, p))}" for n, p in inst_pos.items()))
    system = (
        "You are a task-level decomposer for a multi-robot delivery system. Cover "
        "EVERY goal-group by assigning exactly ONE of its instances to exactly one "
        "robot, and give each robot an ordered list of instance names, so each robot "
        "reaches its targets (in order, around obstacles) within the per-robot deadline.\n"
        "You may freely (a) REASSIGN a goal to a different robot, and/or (b) for a "
        "group with several equivalent instances, CHANGE which instance you use "
        "(equivalent access points cost different detours). Balance load.\n"
        "IMPORTANT: straight-line distance close does NOT mean reachable — obstacles "
        "force detours, so a euclidean-near target behind a wall may be far/unreachable.\n"
        "Reply with ONLY JSON: {\"r1\":[ordered instance names], \"r2\":[...], ...}")
    dockline = ""
    if dock is not None:
        dockline = (f"\nDELIVERY: each goal is PICKED at its instance AND then carried to "
                    f"the dock D@({int(dock[0])},{int(dock[1])}); the per-robot deadline "
                    f"covers the whole pick+deliver tour (returns to dock between goals).")
    user = (f"Robots: {rb}\nGoal groups (cover each exactly once):\n" + "\n".join(gl) +
            f"\nObstacles: {ob}{dockline}\nPer-robot deadline: {deadline}s (robot speed {int(VMAX)} u/s)\n"
            f"Straight-line distances robot->instance:\n" + "\n".join(dm))
    return system, user


def validate(assignment, robots, norm, rects, bounds, deadline, dock=None):
    """Geometrically check an assignment: per-robot deadlines and exactly-once coverage.

    Returns:
        (issues, t_by_robot): a list of natural-language problems (empty if feasible,
        fed back to the LLM) and each robot's cumulative tour time.
    """
    inst_pos, inst_group, groups = norm
    issues, t_by_robot, covered = [], {}, {}
    for r, seq in assignment.items():
        if r not in robots:
            issues.append(f"unknown robot '{r}'"); continue
        p, t = robots[r], 0.0
        for iname in seq:
            if iname not in inst_pos:
                issues.append(f"{r}: unknown instance '{iname}'"); continue
            g = inst_group[iname]
            covered[g] = covered.get(g, 0) + 1
            dt = path_time(p, inst_pos[iname], rects, bounds)
            if dt == math.inf:
                alt = [n for n in groups[g] if n != iname]
                hint = f" (try another access point of {g}: {alt})" if alt else ""
                issues.append(f"{r}->{iname}: UNREACHABLE around obstacles{hint}")
                break
            t += dt; p = inst_pos[iname]
            if dock is not None:                       # carry to dock, then continue
                back = path_time(p, dock, rects, bounds)
                if back == math.inf:
                    issues.append(f"{r}->{iname}->dock: dock UNREACHABLE around obstacles")
                    break
                t += back; p = dock
            if t > deadline:
                alt = [n for n in groups[g] if n != iname]
                hint = (f" — try a closer access point of {g} {alt}, or give {g} to "
                        f"another robot") if alt else " — reassign to another robot"
                issues.append(f"{r}->{iname}: cumulative {t:.1f}s > deadline {deadline}s "
                              f"(detour, not straight line){hint}")
                break
        t_by_robot[r] = round(t, 1)
    for g, names in groups.items():
        c = covered.get(g, 0)
        if c == 0:
            issues.append(f"group {g} not covered (pick exactly ONE of {names})")
        elif c > 1:
            issues.append(f"group {g} covered {c}x (pick exactly ONE instance of {names})")
    return issues, t_by_robot


def uncoverable_groups(robots, norm, rects, bounds, deadline, dock=None):
    """Escalation A: groups whose EVERY instance is, from EVERY robot start, a direct
    pick(+deliver) longer than the deadline. This direct cost is a lower bound on when
    a group can be served in any order, so it PROVES no L1 assignment can cover it."""
    inst_pos, inst_group, groups = norm
    bad = []
    for g, names in groups.items():
        best = math.inf
        for rp in robots.values():
            for n in names:
                c = path_time(rp, inst_pos[n], rects, bounds)
                if dock is not None and c != math.inf:
                    c += path_time(inst_pos[n], dock, rects, bounds)
                best = min(best, c)
                if best <= deadline:
                    break
            if best <= deadline:
                break
        if best > deadline:
            bad.append((g, best))
    return bad


def decompose(robots, norm, rects, bounds, deadline, model="gemini", rounds=4,
              seed_assignment=None, dock=None):
    """L1 inner loop: LLM assign + on-demand validate + NL-feedback redesign.

    seed_assignment (test only): attempt 1 uses it WITHOUT calling the LLM, to
    deterministically exercise the redesign branch with a real LLM responding to
    real natural-language feedback. Returns (assignment_or_None, log)."""
    try:
        from llm import chat, parse_json_response
    except ModuleNotFoundError:
        from contrauto.llm import chat, parse_json_response
    system, user = build_prompt(robots, norm, rects, bounds, deadline, dock=dock)
    log = []
    for attempt in range(rounds):
        if attempt == 0 and seed_assignment is not None:
            asg = seed_assignment
        else:
            try:
                txt, _ = chat(model, system, user)
                asg = parse_json_response(txt)
            except Exception as e:                     # API timeout / non-JSON -> retry
                log.append((attempt + 1, "LLM-ERR", {}, [type(e).__name__ + ": " + str(e)[:50]]))
                user = build_prompt(robots, norm, rects, bounds, deadline, dock=dock)[1] + \
                    "\n\nReply with ONLY valid JSON."
                continue
        issues, tb = validate(asg, robots, norm, rects, bounds, deadline, dock=dock)
        log.append((attempt + 1, asg, tb, issues))
        if not issues:
            return asg, log
        fb = ("Your assignment is INFEASIBLE:\n  - " + "\n  - ".join(issues) +
              "\nRedesign by REASSIGNING goals to other robots and/or CHANGING which "
              "equivalent instance you use (closer access point), and/or reordering. "
              "Cover each group exactly once, each robot within the deadline. Reply ONLY JSON.")
        user = build_prompt(robots, norm, rects, bounds, deadline, dock=dock)[1] + "\n\n" + fb
    return None, log


def solve(robots, targets, rects, bounds, deadline, model="gemini",
          l1_rounds=4, max_relax=3, relax_factor=1.5, seed_assignment=None, dock=None):
    """Unified fallback chain. Returns a report dict:
       {status, assignment, deadline (used), base_deadline, relax_x, phases}."""
    norm = norm_targets(targets)
    cur = float(deadline)
    phases = []
    for k in range(max_relax + 1):
        bad = uncoverable_groups(robots, norm, rects, bounds, cur, dock=dock)
        if bad:
            reason = ("escalation A (geometric proof): " +
                      "; ".join(f"{g}: every access point > {cur:.1f}s for all robots "
                                f"(best {b:.1f}s)" for g, b in bad))
            phases.append({"deadline": round(cur, 1), "outcome": "L1 skipped", "reason": reason,
                           "log": []})
        else:
            asg, log = decompose(robots, norm, rects, bounds, cur, model=model, rounds=l1_rounds,
                                 seed_assignment=(seed_assignment if k == 0 else None), dock=dock)
            if asg is not None:
                status = "SATISFIED" if k == 0 else "REPAIRED"
                phases.append({"deadline": round(cur, 1), "outcome": "FEASIBLE", "reason": "", "log": log})
                return {"status": status, "assignment": asg, "deadline": round(cur, 1),
                        "base_deadline": float(deadline), "relax_x": round(cur / deadline, 2),
                        "norm": norm, "phases": phases}
            reason = f"escalation B (budget): L1 explored {l1_rounds} rounds, no feasible assignment"
            phases.append({"deadline": round(cur, 1), "outcome": "L1 exhausted", "reason": reason, "log": log})
        if k == max_relax:
            break
        cur *= relax_factor                       # relax the ONE shared deadline
    return {"status": "UNSATISFIABLE", "assignment": None, "deadline": round(cur, 1),
            "base_deadline": float(deadline), "relax_x": round(cur / deadline, 2),
            "norm": norm, "phases": phases}


def print_report(rep, title=""):
    """Pretty-print a solve() report: each phase's attempts, then the final verdict."""
    if title:
        print(f"\n############ {title} ############")
    for ph in rep["phases"]:
        print(f"  [deadline {ph['deadline']}s] -> {ph['outcome']}"
              + (f"  ({ph['reason']})" if ph["reason"] else ""))
        for attempt, a, tb, issues in ph["log"]:
            src = "SEEDED" if (attempt == 1 and a is not None and rep["phases"][0] is ph
                              and rep.get("_seeded")) else "LLM"
            print(f"      attempt {attempt}: {a if a != 'PARSE-ERR' else 'PARSE ERR'}"
                  + (f"  times={tb}" if tb else ""))
            print(f"         -> {'FEASIBLE ✓' if not issues else issues}")
    s = rep["status"]
    tail = (f"  (relaxed {rep['base_deadline']:.1f}s -> {rep['deadline']:.1f}s, "
            f"×{rep['relax_x']})") if s == "REPAIRED" else ""
    print(f"==> {s}{tail}: {rep['assignment'] if rep['assignment'] else '—'}")


if __name__ == "__main__":
    bounds = (0, 700, 0, 600)

    # ---- (1) pocket trap, single-instance: LLM should avoid it first try ----
    rects = [(190, 300, 12, 95), (250, 218, 78, 12), (250, 382, 78, 12)]
    robots = {"r1": (90, 300), "r2": (430, 300)}
    targets = {"T0": (250, 300), "T1": (120, 520)}
    print_report(solve(robots, targets, rects, bounds, 3.0), "1. pocket trap (LLM free)")

    # ---- (2) CHANGE EQUIVALENT TARGET: one group, two access points; seed the bad
    #         (trapped) instance -> real LLM must switch to the open instance ----
    rects2 = [(300, 300, 15, 150)]                 # vertical wall x285..315, y150..450
    robots2 = {"r1": (90, 300)}
    targets2 = {"G": [(360, 300),                   # G#0: just past the wall -> long detour for r1
                      (130, 170)]}                  # G#1: open, near r1
    print("\nGeometric ground truth (change-target case):")
    for n, p in norm_targets(targets2)[0].items():
        print(f"  r1->{n}: t_min={path_time(robots2['r1'], p, rects2, bounds):.2f}s")
    rep2 = solve(robots2, targets2, rects2, bounds, 2.0, seed_assignment={"r1": ["G#0"]})
    rep2["_seeded"] = True
    print_report(rep2, "2. change equivalent target (seeded G#0 -> LLM switches instance)")

    # ---- (3) RELAX escalation: target unreachable-in-deadline by all robots at base
    #         deadline -> escalation A -> relax bumps deadline -> becomes feasible ----
    rects3 = []
    robots3 = {"r1": (50, 50)}
    targets3 = {"F": (650, 550)}                    # far: t_min ~ 6.6s
    tmin = path_time(robots3["r1"], targets3["F"], rects3, bounds)
    print(f"\nGeometric ground truth (relax case): r1->F t_min={tmin:.2f}s")
    print_report(solve(robots3, targets3, rects3, bounds, 2.0, max_relax=4, relax_factor=1.6),
                 "3. relax escalation (tight deadline -> A -> relax -> REPAIRED)")
