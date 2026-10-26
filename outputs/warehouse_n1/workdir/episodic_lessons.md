

---

<!-- mission: warehouse -->
### Lesson: warehouse
- **Failure pattern**: Decomposer fails before synthesis with GR(1) fragment violations, compatibility errors, or NuXMV crashes on undeclared variables.
- **Root cause**: Contracts violate GR(1) strict synthesis rules or interface boundaries. Specifically, assumptions apply next-step operators (`X`) to component-owned outputs (e.g., constraining `X(r1_dock_req)` in an assumption), creating unsatisfiable circular obligations. 
- **Fix strategy**:
  - **Strict Assumption Boundaries**: Assumptions must strictly constrain the environment (inputs). Never reference or apply `X` to a component's own outputs in its assumptions.
  - **Relocate Output Constraints**: Move all handshake continuity (e.g., holding a request until granted), output next-state rules, mutual-exclusion, and initial-state constraints into Guarantees.
  - **Interface Discipline**: Declare every signal explicitly as input or output in each contract; ensure names match exactly and remove stale/undeclared variables before decomposition.
  - **GR(1) Liveness**: Express liveness guarantees only as pure justice properties `G F(input_goal | !output_trigger)`. Avoid nested temporals, `Until` operators, or direct response chains (`G(p -> F(q))`) in guarantees.
