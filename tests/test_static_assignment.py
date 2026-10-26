"""static_2r_3b TRUE static assignment test.

Robot1 is statically assigned box1 + box3 ; Robot2 is statically assigned box2.
We assert, headless:
  * both robots deliver (>0),
  * each robot ONLY delivered boxes it owns (foreign deliveries == 0).

Run:  SDL_VIDEODRIVER=dummy python3 tests/test_static_assignment.py
"""
import os
import sys

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_static_assignment_both_deliver_only_own_boxes():
    from simulator.static_2r_3b import headless, W
    if not (os.path.exists(W + "robot1.json") and os.path.exists(W + "robot2.json")):
        pytest.skip("needs synthesized static_2r_3b FSMs (run: python main.py --task static_2r_3b --no-sim)")

    m = headless(steps=8000)

    # both robots make progress
    assert m["delivered"] > 0, f"no deliveries: {m}"
    assert all(v > 0 for v in m["per_robot"].values()), f"a robot starved: {m['per_robot']}"

    # static assignment respected: each robot delivered ONLY its own boxes
    assert all(v == 0 for v in m["foreign_per_robot"].values()), \
        f"a robot delivered a box it does not own: {m['foreign_per_robot']}"
    # owned tally must equal the total delivery tally for each robot
    for name, total in m["per_robot"].items():
        assert m["owned_per_robot"][name] == total, \
            f"{name}: owned {m['owned_per_robot'][name]} != delivered {total}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all static-assignment tests passed")
