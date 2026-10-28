# ACIES

**A Self-Evolving Agentic Contract-Based Framework for Cyber-Physical System Control Synthesis.**

[![Paper](https://img.shields.io/badge/Paper-REALM%202026-red.svg)](https://aclanthology.org/2026.realm-1.36/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![Overview of ACIES: within episode k, the Translator produces LTL, the Decomposer produces A/G contracts, formal verification (syntax check, consistency & compatibility, refinement with CHASE + nuXmv) and GR(1) synthesis with Slugs feed counterexamples and unrealizable cores back to the Decomposer; the Summarizer distills the episode log into a lesson stored in memory and recalled at the start of episode k+1](figure/acies_overview.png)

ACIES turns natural-language requirements into provably correct
controllers for cyber-physical systems. In each synthesis episode, a
**Translator** agent formalizes the requirements in temporal logic, a
**Decomposer** agent structures them into a system-level contract and
component-level contracts, and formal verification (CHASE + nuXmv) and
GR(1) synthesis (SLUGS) return a diagnostic for every failed check that
drives contract repair. Across episodes, a **Summarizer** agent distills
the verification outcomes into lessons stored in memory.

**Paper:** [ACIES: A Self-Evolving Agentic Contract-Based Framework for
Cyber-Physical System Control Synthesis](https://aclanthology.org/2026.realm-1.36/)
([PDF](https://aclanthology.org/2026.realm-1.36.pdf)), REALM 2026 (ACL Anthology).

## Installation

```bash
git clone https://github.com/dinghongsong26/ACIES.git
cd ACIES
conda create -n acies python=3.10 -y && conda activate acies
pip install -r requirements.txt
```

The symbolic tools are installed separately and carry their own licenses:
[CHASE](https://chase-cps.github.io) (`logics_tool`), nuXmv (free for
non-commercial use, not redistributable) and SLUGS (BSD). ACIES only
invokes them as external executables, located through environment
variables. Copy [`.env.example`](.env.example) to `.env`, fill in one API
key and the tool paths, and source it:

```bash
cp .env.example .env      # GEMINI_API_KEY, NUXMV, SLUGS, SLUGS_COMP, LOGICS_TOOL
source .env
```

`.env` is gitignored, so keys are never committed.

## Quick start

<p align="center">
  <a href="figure/acies_demo.mp4"><img src="figure/acies_demo.gif" width="820"
     alt="Terminal recording of an ACIES run on the two-robot warehouse task: Translator output with requirement tags, then Decomposer attempts with Steps 1 to 12 and the synthesized FSMs"></a>
</p>

<p align="center">
  <img src="figure/warehouse_n2.gif" width="420"
       alt="Two AGVs fetching items from the shelves and delivering them to a shared dock; the synthesized Arbiter FSM grants the dock to one robot at a time, with zero mutual-exclusion violations">
</p>

`main.py` is the single entry point. It runs the task layer (synthesize,
or reuse existing FSMs), then the physical layer (execute them in the
matching simulator and export a GIF).

```bash
python main.py --task warehouse --n 1              # full two-layer run
python main.py --task warehouse --n 2 --no-sim     # task layer only
python main.py --task warehouse --n 2 --resynth    # force re-synthesis
python main.py --task warehouse --n 1 --retranslate  # ignore the Translator cache
```

A run prints one block per agent and one line per check, in this order:

| Step | Check | Tool |
|---|---|---|
| Translator | NL requirements → LTL; coverage, lint, translation realizability | LLM, Python, CHASE `logics_tool` + SLUGS |
| 1 | Contracts generated | LLM |
| 2 | Grounding: every variable binds to the simulator vocabulary | Python |
| 3 | Syntax check | CHASE `logics_tool` |
| 4–8 | GR(1) fragment, initial values, variable ownership, mission anchoring, goal witness | Python |
| 9 | Mechanical completion (assumption discharge, persistence, cycle folding) | Python + SLUGS |
| 10 | Consistency & compatibility | CHASE `logics_tool` + nuXmv |
| 11 | Refinement of the System contract | CHASE `logics_tool` + nuXmv IC3 |
| 12 | FSM synthesis, one per contract | CHASE `logics_tool` + SLUGS |

Artifacts land under `outputs/<task>_n<N>/`: the Translator and
Decomposer outputs, the contracts (`specs.ltl`), the synthesized FSMs
(`workdir/*.json`), `result.json` with time and token cost, and
`physical_simulation.gif`. [`outputs/warehouse_n1/`](outputs/warehouse_n1/)
is a complete single-robot example run (Gemini 3.1 Pro, 263 s end to end,
47k tokens, $0.38).

To run a simulator on existing FSMs without any LLM call, for example the
frozen two-robot warehouse controllers in [`golden/`](golden/):

```bash
python -m simulator --task warehouse --robot1 golden/warehouse/robot1.json \
    --robot2 golden/warehouse/robot2.json --arbiter golden/warehouse/arbiter.json
```

**Models (`--model`).** `gemini` (default, Gemini 3.1 Pro), `gpt`,
`claude`, `deepseek`, `glm`, `kimi`; the full alias table is in
[`contrauto/llm.py`](contrauto/llm.py), and any LiteLLM model id is
accepted as well.

## Tasks

A task is one YAML in [`contrauto/tasks/`](contrauto/tasks/): an
architecture (nodes and communication edges) and plain natural-language
requirements. No formula is written by hand.

| Task (`--task`) | Scenario |
|---|---|
| `warehouse` (`--n N`) | N AGVs fetch items from a shelf and deliver them to a capacity-limited dock; an arbiter grants dock access. |
| `charger` (`--n N`) | N robots share one charging pad under an arbiter. |
| `corridor` (`--n N`) | N robots cross a single-lane corridor under an arbiter. |

The worker-scalable tasks declare one `Robot1` template, cloned into
`Robot1..RobotN` by `--n`. To define a new task, copy
[`warehouse.yaml`](contrauto/tasks/warehouse.yaml) and edit its
requirements; connecting a simulator is described in
[docs/BRING_YOUR_OWN_SIMULATOR.md](docs/BRING_YOUR_OWN_SIMULATOR.md).

## Repository layout

| Path | Contents |
|---|---|
| [`contrauto/`](contrauto/) | The task layer: agents (Translator, Decomposer, Summarizer), prompts, mechanical checks, feedback and QuickXplain cores, the tool pipeline around CHASE / nuXmv / SLUGS |
| [`simulator/`](simulator/) | Continuous simulators (pygame), FSM binding, GIF export |
| [`cbf/`](cbf/) | The CBF-QP filter and motion model |
| [`physics/`](physics/), [`maps/`](maps/) | Robot dynamics and map layouts |
| [`golden/`](golden/) | Frozen, verified contracts and FSMs for testing the physical layer without an LLM |
| [`experiments/`](experiments/) | Evaluation harnesses ([README](experiments/README.md)) |
| [`tests/`](tests/) | `pytest` suite |

## Documentation

- [Framework](docs/FRAMEWORK.md): the two layers and the certificate ladder.
- [Grounding](docs/GROUNDING.md): how contract variables bind to simulator sensors and actions.
- [Bring your own simulator](docs/BRING_YOUR_OWN_SIMULATOR.md): the adapter interface.
- [References](references.md): the contract-based design lineage ACIES builds on.

## Tests

```bash
pytest
```

The mechanics tests need no external tools; the simulator tests skip
until FSMs have been synthesized (for example by
`python main.py --task warehouse --n 2 --no-sim`).

## Citation

```bibtex
@inproceedings{xiao-etal-2026-acies,
  title     = {{ACIES}: A Self-Evolving Agentic Contract-Based Framework for
               Cyber-Physical System Control Synthesis},
  author    = {Xiao, Yifeng and Ni, Zhiyu and Song, Dinghong and Zhou, Mi and
               Nuzzo, Pierluigi},
  editor    = {Kamalloo, Ehsan and Gontier, Nicolas and Rozanov, Nikolai and
               Hashemzadeh, Maryam and Arora, Akhil},
  booktitle = {Proceedings of the Second Workshop for {RE}search on Agent
               Language Models ({REALM} 2026)},
  month     = oct,
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2026.realm-1.36/},
  pages     = {541--550},
  isbn      = {979-8-89176-485-9}
}
```

## License

MIT, see [`LICENSE`](LICENSE). nuXmv, SLUGS and CHASE are not bundled
and keep their own licenses.
