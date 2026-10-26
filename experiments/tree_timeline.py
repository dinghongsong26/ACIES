#!/usr/bin/env python3
"""Two-level grant timelines from the LLM-decomposed tree (task tree_dock6, S2).

Lockstep discrete co-simulation of the 9 certified FSMs (6 robots, 2 group arbiters,
1 root) under a simple abstract plant (navigation arrives after a few steps, pickup
takes effect next step). Produces the request/grant timeline panel showing: per-robot
requests/grants never overlapping within a group, group demands, and the root
alternating the two groups — the live analogue of the hand-built panel (c) in Fig. 2.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from contrauto.sim_iface import FsmController

W = os.environ.get("TREE_WORKDIR", "outputs/s2_tree_pro1/workdir")
OUT = os.environ.get("TREE_TIMELINE_OUT", "tree_timeline_e2e.pdf")
STEPS = 220
NAV_DELAY = 4


def load(name, ins, outs):
    c = open(f"{W}/{name}.json").read()
    return FsmController(json.loads(c[c.index("{"):]), ins, outs)


robots = []
for k in range(1, 7):
    robots.append(load(f"robot{k}",
                       [f"at_pickup_{k}", f"at_dock_{k}", f"has_item_{k}",
                        f"task_{k}", f"grant_{k}"],
                       [f"nav_pickup_{k}", f"nav_dock_{k}", f"pickup_{k}",
                        f"dropoff_{k}", f"req_{k}"]))
g1 = load("grouparb1", ["req_1", "req_2", "req_3", "group_grant_1"],
          ["grant_1", "grant_2", "grant_3", "group_req_1"])
g2 = load("grouparb2", ["req_4", "req_5", "req_6", "group_grant_2"],
          ["grant_4", "grant_5", "grant_6", "group_req_2"])
root = load("rootarb", ["group_req_1", "group_req_2"],
            ["group_grant_1", "group_grant_2"])

# plant state per robot
plant = [dict(at_p=0, at_d=0, item=0, nav_p_since=None, nav_d_since=None)
         for _ in range(6)]
sig = {f"grant_{k}": 0 for k in range(1, 7)}
sig.update(group_grant_1=0, group_grant_2=0, group_req_1=0, group_req_2=0)
for k in range(1, 7):
    sig[f"req_{k}"] = 0

trace = {name: [] for name in
         [f"req_{k}" for k in range(1, 7)] + [f"grant_{k}" for k in range(1, 7)]
         + ["group_req_1", "group_req_2", "group_grant_1", "group_grant_2"]}

for s in range(STEPS):
    # robots step (read current grants + plant sensors)
    outs = {}
    for i, r in enumerate(robots):
        k = i + 1
        p = plant[i]
        inp = {f"at_pickup_{k}": p["at_p"], f"at_dock_{k}": p["at_d"],
               f"has_item_{k}": p["item"], f"task_{k}": 1,
               f"grant_{k}": sig[f"grant_{k}"]}
        outs.update(r.step(inp))
    for k in range(1, 7):
        sig[f"req_{k}"] = outs.get(f"req_{k}", 0)
    # group arbiters (read robot reqs + current root grant)
    o1 = g1.step({"req_1": sig["req_1"], "req_2": sig["req_2"],
                  "req_3": sig["req_3"], "group_grant_1": sig["group_grant_1"]})
    o2 = g2.step({"req_4": sig["req_4"], "req_5": sig["req_5"],
                  "req_6": sig["req_6"], "group_grant_2": sig["group_grant_2"]})
    sig.update({k: v for k, v in o1.items()})
    sig.update({k: v for k, v in o2.items()})
    # root (reads group demands)
    orо = root.step({"group_req_1": sig["group_req_1"],
                     "group_req_2": sig["group_req_2"]})
    sig.update(orо)
    # plant dynamics from robot actions
    for i in range(6):
        k = i + 1
        p = plant[i]
        if outs.get(f"nav_pickup_{k}"):
            p["nav_p_since"] = p["nav_p_since"] if p["nav_p_since"] is not None else s
            if s - p["nav_p_since"] >= NAV_DELAY:
                p["at_p"], p["at_d"] = 1, 0
        else:
            p["nav_p_since"] = None
        if outs.get(f"nav_dock_{k}"):
            p["nav_d_since"] = p["nav_d_since"] if p["nav_d_since"] is not None else s
            if s - p["nav_d_since"] >= NAV_DELAY:
                p["at_d"], p["at_p"] = 1, 0
        else:
            p["nav_d_since"] = None
        if outs.get(f"pickup_{k}") and p["at_p"]:
            p["item"] = 1
        if outs.get(f"dropoff_{k}") and p["at_d"] and p["item"]:
            p["item"] = 0
    for name in trace:
        trace[name].append(sig.get(name, 0))

# ── plot ─────────────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "STIXGeneral"
fig, ax = plt.subplots(figsize=(7.0, 2.6))
rows = ([("root: g$_2$", "group_grant_2", "#7a1f9c"), ("root: g$_1$", "group_grant_1", "#7a1f9c"),
         ("dem$_2$", "group_req_2", "#b8860b"), ("dem$_1$", "group_req_1", "#b8860b")]
        + [(f"R{k}", (f"req_{k}", f"grant_{k}"), "#1a6b3c") for k in range(6, 0, -1)])
y = 0
labels, yticks = [], []
for label, key, col in rows:
    if isinstance(key, tuple):
        rq, gt = key
        spans = [(t, 1) for t, v in enumerate(trace[rq]) if v]
        ax.broken_barh(spans, (y + 0.08, 0.30), color="#bbbbbb")
        spans = [(t, 1) for t, v in enumerate(trace[gt]) if v]
        ax.broken_barh(spans, (y + 0.42, 0.50), color=col)
    else:
        spans = [(t, 1) for t, v in enumerate(trace[key]) if v]
        ax.broken_barh(spans, (y + 0.15, 0.7), color=col)
    labels.append(label); yticks.append(y + 0.5)
    y += 1
ax.set_yticks(yticks); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlim(0, STEPS); ax.set_ylim(0, y)
ax.set_xlabel("step", fontsize=8)
ax.tick_params(axis="x", labelsize=7)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
print("grants overlap within group1:",
      any(trace["grant_1"][t] + trace["grant_2"][t] + trace["grant_3"][t] > 1
          for t in range(STEPS)))
print("root both groups:", any(trace["group_grant_1"][t] + trace["group_grant_2"][t] > 1
                               for t in range(STEPS)))
EOF_MARKER_NOT_USED = None
