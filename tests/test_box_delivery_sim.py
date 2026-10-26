"""Integration tests: static_2r_3b and dynamic_2r_3b on the common interface
(simulator/box_delivery.py). static = two independent warehouse-isomorphic robot FSMs;
dynamic = two robots + a synthesized Dispatcher FSM driven like warehouse's arbiter.
Both robots must deliver.

Run:  SDL_VIDEODRIVER=dummy python3 tests/test_box_delivery_sim.py
"""
import os
import sys

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_static_both_robots_deliver():
    from simulator.static_2r_3b import headless, W
    if not os.path.exists(W + "robot1.json"):
        pytest.skip("needs synthesized static_2r_3b FSMs (run: python main.py --task static_2r_3b --no-sim)")
    m = headless(steps=8000)
    assert m["delivered"] > 0, f"no deliveries: {m}"
    assert all(v > 0 for v in m["per_robot"].values()), f"a robot starved: {m['per_robot']}"


def test_dynamic_both_robots_deliver():
    from simulator.dynamic_2r_3b import headless, W
    if not os.path.exists(W + "robot1.json"):
        pytest.skip("needs synthesized dynamic_2r_3b FSMs (run: python main.py --task dynamic_2r_3b --no-sim)")
    m = headless(steps=8000)
    assert m["delivered"] > 0, f"no deliveries: {m}"
    assert all(v > 0 for v in m["per_robot"].values()), f"a robot starved: {m['per_robot']}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all box-delivery tests passed")
