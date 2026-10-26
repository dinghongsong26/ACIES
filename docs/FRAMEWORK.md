# ContrAuto framework

ContrAuto produces running multi-robot controllers from a high-level task description. It has two
layers, and each layer has its own feedback so that a failed attempt is diagnosed and fixed rather
than discarded. The task layer produces discrete controllers (finite state machines) from
contracts. The physical layer drives the robots in continuous space and checks timing and safety.

## Inputs

A task is a single YAML file with three parts.

1. Natural-language requirements and context. Plain sentences describing what the robots should do.
2. Architecture. The component graph: nodes (workers, an arbiter, a dispatcher) and communication
   edges. The edges decide coordination. No edge between two agents means they are independent, so
   one synthesized worker FSM is reused for both. An edge means they can coordinate, for example an
   arbiter that grants access to a shared dock.
3. Map and physics. The spatial layout (regions, bounds, spawn points) and optional per-node
   dynamics.

Routing of the inputs:
- Requirements and architecture feed the task layer.
- Map, architecture, and physics feed the physical layer. The planner needs the map geometry, and
  the architecture sets how many agents run.

## Task layer (discrete)

Pipeline: the translator turns requirements plus architecture into LTL formulas. The decomposer
turns those into assume-guarantee contracts, one per component. CHASE checks the contracts. SLUGS
synthesizes one finite state machine per contract. A well-separatedness gate inspects each
synthesized strategy. The output is one controller FSM per component.

There are three feedbacks. Each one produces a formal witness, and each one is sent back to the
decomposer, which regenerates the contracts. Every diagnosis is accumulated in episodic memory, so
a later attempt has to fix all earlier mistakes at once instead of fixing the latest one and
reintroducing an old one.

1. Refinement feedback. After CHASE composes the sub-contracts, nuXMV checks that the composition
   refines the system contract. If it does not, nuXMV returns a counterexample trace. The
   decomposer receives the counterexample together with the contracts that caused it.

2. Synthesis (realizability) feedback. SLUGS runs GR(1) synthesis on each contract. If a contract
   is unrealizable, SLUGS is run again with the counterstrategy option to obtain the environment's
   winning strategy, which is the actual play in which the environment forces the system to lose.
   From that play we also report which inputs the environment can hold fixed forever, which is a
   frequent cause of unrealizable liveness guarantees. The counterstrategy goes back to the
   decomposer, with guidance on the usual fixes (add an assumption that forces the input to change,
   make the variable an output, or weaken the conflicting guarantee while keeping GR(1) form).

3. Well-separatedness feedback. A synthesized GR(1) strategy is degenerate if the system wins
   vacuously by driving the environment into a dead end, which means the specification is not
   well-separated. Each FSM is checked for such a dead end. If one is found, the dead-end witness is
   sent back so the decomposer makes the environment non-blocking (for example by adding the
   matching system-side exclusions or weakening the offending assumption).

Two earlier checks, syntax and consistency/compatibility, run in the same loop and send their error
message back as well. They are simpler pass-or-fail checks than the three diagnostic feedbacks
above.

## Physical layer (continuous)

Pipeline: the well-separated FSM becomes the controller. For the controller's current target, a
planner (grid A*) routes around the obstacles and returns waypoints. Each straight segment becomes
a Reachability Contract (RC) with a provable reach time. Peers and shelves become Control Barrier
Contracts (CBC) for avoidance. A CBF quadratic program executes the contracts at each step. A
certificate sums the provable segment times and compares them against the deadline. Then the robots
run in the simulator.

There are two feedbacks.

1. Certificate feedback (before running, from provability). The certificate computes a sound
   single-agent completion bound, the sum of the trapezoidal reach times along the route. If that
   bound already exceeds the deadline, the assignment and route are infeasible, so the system
   repairs before it simulates. This also covers the admissibility condition, namely that the
   sampling time has to exceed the worst-case reach time between adjacent regions.

2. Execution feedback (after running, from measurement). The simulation measures each robot's actual
   finish time. Multi-agent coupling, that is avoidance detours, yielding, and queueing for the
   shared dock, adds time that the single-agent certificate does not model, so a robot can still
   overrun the deadline even when its certificate passed. The measured overrun is the signal.

Repair acts on these two feedbacks in tiers, taking the lowest level that can fix the problem.
- Physical execution. The CBF quadratic program already handles instantaneous avoidance.
- Physical planning. Reassign which good goes to which robot, or relax the shared deadline, then
  re-solve.
- Physical coordination. Raise the offending robot's yielding priority so its peers give way.
- Task re-synthesis. If none of the physical tiers can fix the failure because it is structural
  (deadlock, starvation, or an infeasible decomposition), the structural diagnosis is sent up to the
  task layer and the FSM is re-synthesized. This is the physical-to-task feedback edge.

## Why this is a general framework

The two layers are decoupled. The physical layer only reads the FSM JSON files, so one synthesized
FSM is reused across different maps and instance counts. A new task is a new YAML file. A new
simulator is one adapter registered in `simulator/registry.py` that maps the task name to a
simulator function and the list of FSM files it needs. Without an adapter, the runner executes the
task layer only.

The five feedbacks do not depend on the specific simulator. The three task feedbacks come from the
contract toolchain (CHASE, nuXMV, SLUGS), and the two physical feedbacks come from the certificate
and the execution oracle. Both of these work for any registered simulator, so the same machinery
applies to warehouse, patrol, multi-robot, dock, and any new scenario.

## Memory

Memory has two roles and they are kept separate.
- Episodic memory lives within one design problem. It accumulates the distinct failures of the
  current run so that a later step never repeats an earlier one. It is reset for each new problem.
- Semantic memory lives across tasks. The summarizer distills per-mission lessons into a file that
  is loaded into the prompt of future runs.
