"""Integration test: the patrol scenario on the common interface. The patrol contract
is structurally different from warehouse (APs at_A/B/C, alert_A/B/C, inspection_done,
report_ack -> go_to_A/B/C, inspect, report) — so this proves the same FsmController/
run_fsm drive a non-warehouse-shaped FSM. The robot must clear alerts over time.

Run:  SDL_VIDEODRIVER=dummy python3 tests/test_patrol_sim.py
"""
import glob
import json
import os
import sys

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_patrol_fsm():
    """The patrol robot FSM = the workdir json whose variables include the patrol APs."""
    for wd in ("outputs/patrol_regen/workdir", "outputs/patrol/workdir"):
        for f in sorted(glob.glob(os.path.join(ROOT, wd, "*.json"))):
            try:
                v = json.load(open(f)).get("variables", [])
            except Exception:
                continue
            if ("go_A" in v or "go_to_A" in v) and "inspect" in v:
                return f
    return None


def test_patrol_fsm_exists_and_nonvacuous():
    f = _find_patrol_fsm()
    if not f:
        pytest.skip("needs synthesized patrol FSMs (run: python main.py --task patrol --no-sim)")
    from contrauto.sim_iface import FsmController, load_fsm
    d = load_fsm(f)
    V = d["variables"]
    ins = [v for v in V if v.startswith("at_") or v.startswith("alert_")
           or v in ("inspection_done", "report_ack", "ack")]
    outs = [v for v in V if v not in ins]
    c = FsmController(d, ins, outs)
    # over a realistic alert/arrival feedback loop, at least one action must fire
    fired = set()
    env = {v: 0 for v in ins}
    env_alert = next((v for v in ins if v.startswith("alert_")), None)
    for _ in range(40):
        if env_alert:
            env[env_alert] = 1
        o = c.step(env)
        acts = [k for k, val in o.items() if val]
        fired.update(acts)
        env = {v: 0 for v in ins}
        for z in ("A", "B", "C"):
            if f"go_{z}" in acts or f"go_to_{z}" in acts:
                if f"at_{z}" in env: env[f"at_{z}"] = 1
        if "report" in acts:
            for k in ("ack", "report_ack"):
                if k in env: env[k] = 1
        if "inspect" in acts and "inspection_done" in env:
            env["inspection_done"] = 1
    assert fired, "patrol FSM emitted no actions over 40 steps (vacuous)"


def test_patrol_clears_alerts():
    f = _find_patrol_fsm()
    if not f:
        pytest.skip("needs synthesized patrol FSMs (run: python main.py --task patrol --no-sim)")
    from simulator.patrol import headless
    m = headless(f, steps=6000)
    assert m["cleared"] > 0, f"patrol cleared no alerts: {m}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all patrol-sim tests passed")
