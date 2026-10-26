# Golden FSMs + Contracts (model-independent physical-layer tests)

These are **verified, frozen** task-layer outputs (GR(1) contracts in `specs.ltl` and the
SLUGS-synthesized FSM JSONs) for all four simulator tasks. Use them to test the **physical
layer (CBF execution) directly**, decoupled from any LLM.

## Why this exists

ContrAuto has two layers:

- **Task layer (discrete):** an LLM writes GR(1) contracts → SLUGS synthesizes one FSM per agent.
- **Physical layer (continuous):** the CBF controller **only reads the FSM JSON**; it never touches the LLM.

When a *weaker model* writes a contract that is too weak, SLUGS can return a **legal-but-lazy
strategy**, e.g. a warehouse FSM that **never activates `navigate_to_shelf`**, so the robot
never fetches an item and `delivered` stays `0` (the robot looks idle / barely moves). That is a
**task-layer (contract) problem, not a simulator bug**.

These golden files let you verify the physical layer works regardless of which model you swap in.

## Verified results (headless)

| Task           | Command flag(s)                          | Result (headless)              |
|----------------|------------------------------------------|---------------------------------|
| warehouse      | `--robot1 --robot2 --arbiter`            | delivered 25–39, violations 0  |
| patrol         | `--fsm`                                   | cleared 5–8                    |
| static_2r_3b   | `--robot1 --robot2`                      | delivered 22 (11+11)           |
| dynamic_2r_3b  | `--robot1 --robot2 --dispatcher`         | delivered 10 (5+5)             |

(Numbers vary with step count; the point is **> 0 and stable**, vs a bad FSM that yields 0.)

## Run with the live pygame window

```bash
python -m simulator --task warehouse     --robot1  golden/warehouse/robot1.json \
                                         --robot2  golden/warehouse/robot2.json \
                                         --arbiter golden/warehouse/arbiter.json
python -m simulator --task patrol        --fsm     golden/patrol/patrolrobot.json
python -m simulator --task static_2r_3b  --robot1  golden/static_2r_3b/robot1.json \
                                         --robot2  golden/static_2r_3b/robot2.json
python -m simulator --task dynamic_2r_3b --robot1  golden/dynamic_2r_3b/robot1.json \
                                         --robot2  golden/dynamic_2r_3b/robot2.json \
                                         --dispatcher golden/dynamic_2r_3b/dispatcher.json
```

## Run headless (prints metrics, no window)

```bash
python -m simulator.warehouse     --robot1 golden/warehouse/robot1.json \
                                  --robot2 golden/warehouse/robot2.json \
                                  --arbiter golden/warehouse/arbiter.json --headless 6000
python -m simulator.patrol        --fsm golden/patrol/patrolrobot.json --headless 6000
python -m simulator.static_2r_3b  --robot1 golden/static_2r_3b/robot1.json \
                                  --robot2 golden/static_2r_3b/robot2.json --headless 6000
python -m simulator.dynamic_2r_3b --robot1 golden/dynamic_2r_3b/robot1.json \
                                  --robot2 golden/dynamic_2r_3b/robot2.json \
                                  --dispatcher golden/dynamic_2r_3b/dispatcher.json --headless 3000
```

## Important

- **Do not `--resynth` over these**, and do not point a weak-model run at the same workdir:
  that overwrites a good FSM with a degenerate one and `delivered` drops to 0.
- Each folder also ships the `specs.ltl` that produced its FSM, so you can inspect or
  re-synthesize the contract with SLUGS independently of the LLM.
- `golden/warehouse/` currently ships a single combined `arbiter.json` from an earlier,
  pre-split version of the task (before shelf/dock became two independent arbiters). The
  live `warehouse` task now needs `shelfarbiter.json` + `dockarbiter.json` separately — this
  golden fixture is stale and due for regeneration (`--resynth` once, on a KNOWN-good model,
  then copy the result back here) before relying on it for a physical-layer-only test.
