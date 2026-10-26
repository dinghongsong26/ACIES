# ContrAuto: Related Work and References

## 1. Contract-Based Design Frameworks

### CHASE (Nuzzo et al., UC Berkeley/USC)
- **Paper**: Nuzzo, P., Lora, M., Feldman, Y. A., & Sangiovanni-Vincentelli, A. L. "CHASE: Contract-based requirement engineering for cyber-physical system design." DATE 2018.
- **Link**: https://chase-cps.github.io/
- **Summary**: The foundational framework for A/G contract-based CPS design. Supports consistency, compatibility, refinement checking (via NuXmv) and GR(1) synthesis (via Slugs). All variables must be **fully predefined by the user**, with no automated variable discovery. ContrAuto builds directly on CHASE's toolchain.
- **Relevance**: Our tool automates the human steps of CHASE: variable design, contract writing, and repair after verification failure.

### Pacti (Incer, Sangiovanni-Vincentelli et al.)
- **Paper**: Incer, I., et al. "Pacti: Assume-Guarantee Contracts for Efficient Compositional Analysis and Design." ACM TECS, 2024.
- **Link**: https://www.pacti.org/
- **Summary**: Python library for A/G contract algebra: composition, refinement, **quotient** operations. Variables matched by name across components. Demonstrated verification of **198 components in <5 minutes**. Variables are user-defined; Pacti provides algebraic operations.
- **Relevance**: Quotient operation can derive missing sub-contracts. Naming-by-convention pattern validates our approach. Scalability benchmark (198 components) is encouraging for 100+ agent target.

### OCRA (FBK, Cimatti, Dorigatti, Tonetta)
- **Paper**: Cimatti, A., Dorigatti, M., & Tonetta, S. "OCRA: A tool for checking the refinement of temporal contracts." ASE 2013.
- **Link**: https://es-static.fbk.eu/tools/ocra/
- **Summary**: Tool for checking refinement of temporal contracts. User defines component hierarchy with explicit INPUT/OUTPUT port declarations. OCRA checks refinement; it does not generate contracts or discover variables.

### AGREE (Collins Aerospace / Loonwerks)
- **Paper**: Cofer, D., et al. "Compositional formal analysis based on conventional engineering models." 2020.
- **Link**: https://github.com/loonwerks/AGREE
- **Summary**: Built on AADL architecture models. Variables come from AADL port declarations (**all predefined by the system architect**). Supports `eq` helper variables (also user-defined). Industry-grade but requires significant manual specification effort.

### Interface Automata (de Alfaro & Henzinger)
- **Paper**: de Alfaro, L. & Henzinger, T. A. "Interface automata." FSE 2001.
- **Summary**: Actions partitioned into input/output/internal (**all fully specified by designer**). Optimistic composition semantics. Key lesson: interface theories always start with a predefined alphabet.

### Hypercontracts (Incer, Benveniste, Sangiovanni-Vincentelli)
- **Paper**: Incer, I., et al. "From Interface Automata to Hypercontracts." FM 2022.
- **Summary**: Generalize interface automata to hyperproperties. Assume all components share the same action alphabet (predefined globally).

---

## 2. Automated Contract Decomposition

### DecomposeContract (DC) Algorithm
- **Paper**: Iannopollo, A., Tripakis, S., & Sangiovanni-Vincentelli, A. L. "Specification Decomposition for Synthesis from Libraries of LTL A/G Contracts." DATE 2018.
- **Link**: https://github.com/ianno/pyco
- **Summary**: Given a system-level A/G contract with all variables predefined, automatically **partitions controlled variables** into independent subsets using a model checker. Variables are fully predefined; the automation is in partitioning, not discovery.
- **Relevance**: Demonstrates that variable decomposition can be automated once variables are known. The dependency-detection approach could complement LLM-based decomposition.

### CONDEnSe
- **Paper**: Santos, T., Saleh, M., et al. "CONDEnSe: Contract Based Design Synthesis." DATE 2019.
- **Summary**: Semi-automated methodology: user provides architecture decomposition and high-level contract; tool derives sub-contract bounds. Variables predefined; bounds are derived.

### Contract-Based Specification Refinement and Repair
- **Paper**: Mallozzi, P., Incer, I., Nuzzo, P., et al. "Contract-Based Specification Refinement and Repair for Mission Planning." 2022.
- **Link**: https://arxiv.org/abs/2211.11908
- **Summary**: Uses Pacti's quotient operation to derive and repair sub-contracts from system-level specs. Directly relevant to our feedback loop: quotient provides a formal way to compute what a sub-contract "should" guarantee.

---

## 3. LLM for Formal Specification

### NL2TL (MIT REALM)
- **Paper**: Chen, Y., et al. "NL2TL: Transforming Natural Languages to Temporal Logics using Large Language Models." 2023.
- **Link**: https://arxiv.org/abs/2305.07766
- **Summary**: Two-step approach: (1) AP lifting phase replaces domain terms with placeholders, (2) NL-to-TL translation. AP recognition by GPT-3 achieves **98.84-100% accuracy**. APs are extracted from NL by the LLM, but constrained by domain description.
- **Relevance**: Shows LLMs can reliably extract atomic propositions from NL when domain is constrained. Supports our hybrid approach.

### Lang2LTL (Brown University)
- **Paper**: Pan, J., et al. "Lang2LTL: Translating Natural Language Commands to Temporal Robot Task Specification." CoRL 2023.
- **Link**: https://lang2ltl.github.io/
- **Summary**: Propositions grounded to a **predefined semantic database** of landmarks. LLM does grounding/matching, not invention. Modular pipeline: RE recognition → grounding → lifted LTL.
- **Relevance**: Validates that constraining the proposition vocabulary is essential for accuracy.

### Req2LTL / Building-OnionL
- **Paper**: "Bridging Natural Language and Formal Specification." 2025.
- **Link**: https://arxiv.org/abs/2512.17334
- **Summary**: APs defined with four semantic subfields: Component, Variable, Relation, Formula. LLM extracts APs with **99.5% recall**. Human-in-the-loop correction needed for ~12%. **Hybrid approach: LLM proposes, human validates.**
- **Relevance**: Closest to our target: structured AP templates with LLM extraction. Their 4-field AP structure could inform our variable naming convention.

### ADARULE (ICSE 2026)
- **Paper**: "ADARULE: LLM-Driven Natural Language to LTL Conversion via Pattern-Adaptive Rule Induction." ICSE 2026.
- **Summary**: Feedback-guided approach using Translator + Learner LLM pair. Focuses on NL-to-LTL mapping, assuming APs already defined.
- **Relevance**: Their feedback loop pattern (learn from failures) parallels our DFA-based repair feedback.

### LTLGuard (AIT, 2026)
- **Paper**: "LTLGuard: Formalizing LTL Specifications with Compact Language Models." 2026.
- **Link**: https://arxiv.org/abs/2603.05728
- **Summary**: Uses compact models (4B-14B) with formal consistency checking. Constrained generation + automated reasoning for conflict-free specifications.

### Grammar-Forced Translation (2025)
- **Paper**: "Grammar-Forced Translation of Natural Language to Temporal Logic using Large Language Models." 2025.
- **Link**: https://arxiv.org/abs/2512.16814
- **Summary**: Forces LLM output to conform to a formal grammar. APs from predefined vocabulary.

---

## 4. Multi-Agent Contracts

### Quantitative A/G Contracts for Multi-Agent Systems
- **Paper**: "Contract-based Design and Verification of Multi-Agent Systems with Quantitative Temporal Requirements." AAAI 2025.
- **Link**: https://arxiv.org/abs/2412.13114
- **Summary**: Agent outputs must be disjoint: `O_a ∩ O_a' = {}`. User predefines local specs, shared spec, combination function. Tested with 3-5 agents.
- **Relevance**: The disjoint output requirement is important for our multi-robot scenarios.

### Compositional STL Synthesis for Multi-Robot (Liu, Saoud, Dimarogonas; AGC, KTH)
- **Paper**: Liu, S., Saoud, A., Dimarogonas, D. V. "Controller Synthesis of Collaborative Signal Temporal Logic Tasks for Multiagent Systems via Assume-Guarantee Contracts." IEEE TAC 70(9), 2025 (arXiv 2309.13499, 2023).
- **Link**: https://arxiv.org/abs/2309.13499
- **Summary**: Continuous-time A/G contracts (AGC) for *collaborative* STL: each agent's task depends on neighbors. Casts STL as prescribed-performance (funnel) control, proves a compositionality result (local contracts ⇒ global task), and gives a closed-form distributed controller. A *different* controller approach from CBF-QP.
- **Relevance**: The multi-agent compositionality (local ⇒ global) is the reference for our shared-resource coupling, though we realize contracts with CBF-QP (below), not PPC funnels.

---

## 4b. Physical-Layer CBF Contracts (Nuzzo lineage: the layer ContrAuto's physical level follows)

### Contract Embeddings for Layered Control Architectures (Naik, Pinto, Nuzzo)
- **Paper**: Naik, N. V., Pinto, A., Nuzzo, P. "Contract Embeddings for Layered Control Architectures." ACM TECS 24(5s), Art. 138, 2025.
- **Link**: https://doi.org/10.1145/3764587
- **Summary**: The journal formalization of our exact architecture: discrete **LTL_f** planning contracts on top, continuous **Control Barrier Contracts (CBC, invariance)** and **Reachability Contracts (RC, finite-time reach via FTRF)** below. A conservative approximation Ψ_CT^DT (cell-occupancy over sampling intervals) is the contract *embedding* that traceably refines the LTL_f plan down to a CBC⊗RC control spec (Thm 2/7), implemented by a CBF-QP. Necessary conditions: cell adjacency + admissibility (T_s > T_r^max).
- **Relevance**: This is the formal backbone of ContrAuto's physical layer. `contrauto/cbf_contracts.py` implements its CBC/RC (the FTRF reach-time bound `T_r ≤ |r₀|^{1-δ}/(η(1-δ))`). Our additions over it: LLM-automated contract generation, multi-agent shared-resource coordination (dock mutex/arbiter), and the physical→task repair loop (they return UNSAT; we re-synthesize).

### CBF Contracts for Vehicular Mission Planning (Waqas, Naik, Ioannou, Nuzzo)
- **Paper**: Waqas, M., Naik, N. V., Ioannou, P. A., Nuzzo, P. "Control Barrier Function Contracts for Vehicular Mission Planning Under Signal Temporal Logic Specifications." arXiv 2209.07478, 2022 (conf. version: "Contract-Based Control Synthesis with Barrier Functions for Vehicular Mission Planning").
- **Link**: https://arxiv.org/abs/2209.07478
- **Summary**: Earlier conference-line work: a mission-level STL spec is decomposed into time-interval predicates, mapped to an aggregation of contracts realized by time-varying CBFs, solved by QP. Introduces TV-CBF / TV-FCBF contracts and their composition compatibility.
- **Relevance**: The precursor to the LCA journal paper above; established the STL → time-varying-CBF-contract → QP pattern our physical layer uses.

---

## 5. Formal Verification Background

### Predicate Abstraction / CEGAR
- **Summary**: Predicates extracted from model/spec, not invented from scratch. CEGAR loop: start rough, refine on counterexample. Directly applicable to our LLM-based retry loop.

### Hierarchical System Design with Vertical Contracts
- **Paper**: "Hierarchical System Design with Vertical Contracts." FORMATS 2018.
- **Summary**: Primary scalability strategy for 100+ components: hierarchical decomposition, each level adds only relevant variables. Developers never consider all components at once.

---

## 6. Survey Papers

### Inside the Scaffold
- **Paper**: "Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures." arXiv:2604.03515, April 2025.
- **Summary**: Formal taxonomy of agent scaffold patterns at the source code level. Directly relevant for positioning ContrAuto's architectural choices.

### Formal Requirements Engineering and LLMs
- **Paper**: "Formal Requirements Engineering and LLMs: A Two-Way Roadmap." Information and Software Technology, 2025.
- **Link**: https://doi.org/10.1016/j.infsof.2025.107662
- **Summary**: Surveys the intersection of formal RE and LLMs. Identifies the "proposition grounding gap" as a key challenge.

---

## Key Conclusions

1. **No existing framework automatically discovers variable names.** Every successful system (CHASE, OCRA, AGREE, Pacti) requires user-defined variables. The automation is in decomposition, verification, and synthesis, not in deciding what to observe/control.

2. **LLMs can extract APs from NL with 95-100% accuracy** when given a constrained domain vocabulary (NL2TL, Req2LTL). This supports a hybrid approach: architecture defines the vocabulary, LLM uses it.

3. **Interface variables must be predefined and name-matched** (Pacti validates by name). Internal variables have more freedom.

4. **Quotient operation** (Pacti, Mallozzi et al.) can formally derive what a missing sub-contract should guarantee. This is complementary to our LLM + DFA feedback approach.

5. **Hierarchical decomposition** is the only proven path to 100+ components. No framework has demonstrated flat decomposition at that scale.

6. **CEGAR-like feedback** (start rough, refine on failure) is the standard pattern; our DFA-based feedback loop is a natural instance of this.

---

## 7. Three-Source Atomic Proposition Taxonomy (New Contribution)

ContrAuto introduces a three-source categorization of atomic propositions for contract-based CPS design. No existing work provides a comparable taxonomy.

### The Three Sources

**Source 1, Simulator-grounded propositions** (fixed, physically grounded):
- Observables: `at_shelf`, `has_item`, `at_dock` (from sensors/predicates over physical state)
- Actions: `go_to_shelf`, `pickup`, `dropoff` (from actuators/effectors)
- These are determined by the robot's physical capabilities and the simulator API.

**Source 2, Architecture-derived propositions** (fixed, structurally grounded):
- Protocol signals: `shelf_request`, `shelf_grant`, `task`, `done`
- Derived from edges in the architecture graph (communication channels between components).
- Determined by the system topology, not by requirements.

**Source 3, LLM-proposed propositions** (free, designed by the agent):
- Intermediate abstractions: `del_1`, `all_done`, `returning_home`
- Invented by the LLM to enable contract decomposition.
- Not directly observable from the simulator or derivable from the architecture.

### Literature Comparison

| Existing Classification | Dimension | Missing from Ours? |
|---|---|---|
| Pacti, OCRA, AGREE: input/output | Controllability direction | Covered (Source 1 splits into obs/act) |
| Kress-Gazit 2009: sensor/robot | Two-way controllability | Missing architecture + LLM sources |
| AADL: data/event/flow ports | Communication mechanism | Orthogonal |
| Menghi et al. 2021: Location/Environment/Action | Semantic content | Different axes; no protocol or AI-invented |
| CEGAR: uniform predicate pool | Sufficiency | No source distinction at all |
| NL2TL, Lang2LTL: lifted/grounded | Syntactic role | LLM only translates, does not invent |

### Key References for This Taxonomy

- Menghi, C., et al. "Specification Patterns for Robotic Missions." IEEE TSE, 2021. Three-category classification (Location/Environment/Action), closest structural analog but different axes.
- Kress-Gazit, H., et al. "Where do I go? Synthesizing robot controllers from temporal logic specifications." IEEE TRO, 2009. Sensor vs. robot proposition partition for GR(1) synthesis.
- Incer, I., et al. "Pacti: Scaling Assume-Guarantee Reasoning." ACM TCPS, 2024. I/O contract framework, variables matched by name.
- Bhatia, A., et al. "NL2TL: Transforming Natural Languages to Temporal Logics using LLMs." 2023. LLM extracts APs from NL but does not invent new ones.
- Alrahman, Y.A. & Piterman, N. "Correct-by-Design Teamwork Plans." 2023. Automatically introduces minimal agent interactions, closest to architecture-derived concept but at automaton level.
- Liang, Z., et al. "VisualPredicator." ICLR 2025. Learns neuro-symbolic predicates from robot interaction, closest to LLM-proposed concept but for planning, not contracts.
