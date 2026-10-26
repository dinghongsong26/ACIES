# Case-study instances

Four case studies, one per resource form. Each directory holds a `run.sh` that runs
the evaluation and renders a GIF; synthesis lines stay commented inside because they
cost LLM calls and the certified FSMs already exist. Artifacts land under
`outputs/instances/<name>/`; the synthesized FSMs and pipeline intermediates each
instance consumes are listed in its section.

Common pattern: the LLM decomposition produces one worker contract per robot plus an
arbiter contract; every contract is synthesized to an FSM with SLUGS and certified by
model checking; the physical layer executes the FSMs under the CBF-QP controller.

## dock (terminal resource)

Two warehouse robots deliver to one shared loading dock. A terminal resource gates a
single operation, so a memoryless mutex + no-spurious-grant + justice arbiter with an
unconstrained environment suffices. This is the canonical instance the others vary.

- Simulator: `simulator/warehouse.py`; FSMs: `golden/warehouse/` (frozen) or
  `outputs/warehouse_n2/workdir/`.
- Feeds: the RQ2 adversarial table (arbiter holds every guarantee where each plan
  baseline breaks one) and the environment-transfer table.

```bash
bash experiments/instances/dock/run.sh
```

## charger (terminal resource, physically grounded trigger)

Three robots share one charging station. The request trigger is a continuous
physical state, battery level, abstracted to the predicate `battery_low`; the
objective is recurrent (`G F charging`), and "charging restores the battery" is a
world reaction the contract assumes and the simulator realizes. Its own synthesized
decomposition, not a re-skin of the dock FSMs.

- Simulator: `simulator/charger.py` (random station layout per seed, per-step battery
  drain); FSMs: `outputs/charger_n3/workdir/`.
- Feeds: generality across resource forms; environment transfer (`charger_3r`).

```bash
bash experiments/instances/charger/run.sh
```

## corridor (passage resource, hold-until-release)

A single-lane corridor only one robot may occupy. Unlike a terminal resource, the
grant must be held for the whole crossing: the naive contract is unrealizable, and
the counterstrategy exhibits exactly the revocation state (`crossing = 1, grant = 0`).
The passage form adds a hold-until-release lock, still inside GR(1):

```
mutex                 !(grant_i & grant_j)          i != j
grant only requester  grant_i -> req_i
hold until release    (grant_i & req_i) -> X grant_i
no starvation         GF(!req_i | grant_i)
ENV: finite crossing  GF !(req_i & grant_i)
```

- Simulator: `simulator/corridor.py`. The live route is `run_e2e` with the
  certified corridor FSMs (`outputs/corridor_e2e*/workdir/`): the request is a
  world-driven input and the arbiter holds each grant for the whole crossing.
  The dock-FSM route exists in the module but cannot sustain a crossing (the
  terminal arbiter pulses its grants, so robots wait at the corridor mouth),
  which is exactly the terminal/passage mismatch the RQ2 passage row
  quantifies. The direct SLUGS passage arbiters (N=2..5, no LLM) are in
  `outputs/corridor_direct/` via `experiments/passage_arbiter.py`.
- Feeds: the RQ2 passage row (the terminal arbiter preempts a standing lock on
  passage workloads; the passage arbiter does not) and the localization narrative
  (the counterstrategy pinpoints the missing guarantee).

```bash
bash experiments/instances/corridor/run.sh
```

## hierarchical dock (two-level arbiter tree)

The flat arbiter's state grows as ~2^N (3072 states at N=8). Splitting it into two
group arbiters and a root arbiter keeps every certificate while shrinking the
coordination kernel: 3072 flat vs 288 + 288 + 12 = 588 tree at N=8 (hand-built), and
the full LLM pipeline certifies the tree end to end at N=6 (task `tree_dock6`, 3/3
seeds, kernel 74 + 74 + 12 = 160 vs 576 flat), inventing the group-level demand/grant
interface itself.

Two design notes that matter when reproducing: the upward demand aggregator must be
one-step delayed (`G(reqs -> X greq)`), because an instantaneous aggregator clashes
with the canonical initial pin and makes the group game unrealizable; cross-group
mutual exclusion composes from the downward gate `G(grant_ri -> ggrant_gk)` plus the
root's mutex on `ggrant`, verified by refinement rather than assumed.

- Tree builder and certificate runner: `experiments/hierarchy_replay.py` (offline,
  no LLM, about a minute); artifacts in `outputs/hierarchy_replay/`.
- Feeds: the RQ4 hierarchy rows.

```bash
bash experiments/instances/hierarchical_dock/run.sh
```
