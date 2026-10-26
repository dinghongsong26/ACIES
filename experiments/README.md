# Experiments

Every script here reads synthesized artifacts from `outputs/` (or synthesizes its own
inputs where noted) and writes result files back under `outputs/` or
`experiments/`-local JSON/CSV. Result data files are gitignored; only the scripts are
tracked. Run everything from the repo root with the `chase_env` environment active
and `.env` sourced.

## Case-study instances

`instances/` holds one `run.sh` per case study (dock, charger, corridor,
hierarchical dock) and a single [README](instances/README.md) describing all four.

## Paper harnesses

| Script | What it measures |
| --- | --- |
| `rq1_localized.py` | Localized vs unlocalized feedback: repair rounds to certification per task. |
| `rq2_adversarial.py` | The synthesized arbiter vs plan baselines (greedy, fixed-priority, static TDMA) under adversarial request streams; exhaustive at N=2. |
| `rq2_llm_row.py` | One LLM-authored arbitration rule (longest-wait-first) run through both the terminal and passage harnesses. |
| `rq2_passage_v2.py` | Passage (hold-until-release) workload with service-coupled request withdrawal. Replaces the numbers of `rq2_passage.py`, which is kept because both this script and `rq2_llm_row.py` import its `FsmArb` executor. |
| `sweep_scalability.py` | The N robots x M resources x coordinator-profile sweep over generated tasks (`contrauto/tasks/generated/`). |
| `firstpass_analysis.py` | First-pass acceptance statistics over the sweep CSV. |
| `env_transfer.py` | One synthesized controller executed on K randomized maps; guarantees live at the discrete layer, geometry is re-solved by the CBF-QP. |
| `expA_feedback_e2e.py` | End-to-end feedback ablation runner (harness on vs off). |
| `benchmark_feedback.py` | Refinement repair with vs without the DFA-based diagnoser: success, time, tokens. |

## Hierarchical arbiter tree

| Script | Role |
| --- | --- |
| `hierarchy_replay.py` | Builds the two-level tree from the flat N=8 contract and runs all certificates (offline, no LLM). |
| `hierarchy_behavioral.py` | Steps the three tree FSMs together and checks the composed behavior. |
| `tree_timeline.py` | Renders the two-level grant timeline from the LLM-decomposed `tree_dock6` run. |

## Passage arbiter (no LLM)

| Script | Role |
| --- | --- |
| `passage_arbiter.py` | Direct SLUGS synthesis of the hold-until-release passage arbiter for N=2..5. |
| `corridor_replay.py` | Replays the corridor counterstrategy material through the tool pipeline. |

## Figures

| Script | Output |
| --- | --- |
| `draw_architecture.py` | `figure/ContrAuto.png` (two-layer scheme). |
| `sim_filmstrip.py` | The paper's certified-executions figure: scene + trajectories + request/grant timelines for the dock (N=3) and corridor e2e (N=3) runs. Needs `.env` (panel (a) makes one LLM assignment call). |
| `plot_all_model.py`, `plot_all_task.py` | Latency/cost bar charts per model and per task. |
