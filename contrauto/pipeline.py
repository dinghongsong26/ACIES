"""ContrAuto pipeline: TaskSpec → Translator → Decomposer → FSM."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from contrauto import report
from contrauto.llm import calc_cost, model_name
from contrauto.schema import TaskSpec
from contrauto.agents.translator import TranslatorAgent
from contrauto.agents.decomposer import DecomposerAgent


def _grounding_vocab_for(task_name: str):
    """The target simulator's grounding vocabulary (atomic propositions + actions) for `task_name`,
    provided to the decomposer so every contract variable binds to a physical grounding. Returns
    None if the simulator layer is unavailable or the task has no registered vocabulary."""
    try:
        from simulator.registry import grounding_vocab
        return grounding_vocab(task_name)
    except Exception:
        return None


def _grounding_binder_for(task_name: str):
    """The simulator family's own variable binder for `task_name` (see registry.grounding_binder),
    or None -> the decomposer's vocab + role_of fallback."""
    try:
        from simulator.registry import grounding_binder
        return grounding_binder(task_name)
    except Exception:
        return None


class ContrAuto:
    """Drives the task layer: Translator -> Decomposer, writing artifacts to outdir."""

    def __init__(
        self,
        task: TaskSpec | str,
        model: str = "gemini",
        outdir: str | None = None,
        use_dfa_feedback: bool = True,
        skip_translator: bool = False,
        retranslate: bool = False,
        save_attempts: bool = False,
        physical_feedback: str = "",
        localized_feedback: bool = True,
        enable_completion: bool = True,
        feedback_modes: dict | None = None,
    ):
        if isinstance(task, str):
            if task.endswith((".yaml", ".yml")):
                self.task = TaskSpec.from_yaml(task)
            else:
                yaml_path = Path(__file__).parent / "tasks" / f"{task}.yaml"
                self.task = TaskSpec.from_yaml(str(yaml_path))
        else:
            self.task = task

        self.model = model
        self.outdir = outdir or f"outputs/{self.task.name}"
        self.skip_translator = skip_translator
        self.retranslate = retranslate
        Path(self.outdir).mkdir(parents=True, exist_ok=True)

        self.translator = TranslatorAgent(model=model,
                                          ground_vocab=_grounding_vocab_for(self.task.name))
        self.decomposer = DecomposerAgent(
            model=model,
            workdir=os.path.join(self.outdir, "workdir"),
            mission=self.task.name,
            use_dfa_feedback=use_dfa_feedback,
            save_attempts=save_attempts,
            physical_feedback=physical_feedback,
            localized_feedback=localized_feedback,
            ground_vocab=_grounding_vocab_for(self.task.name),
            ground_binder=_grounding_binder_for(self.task.name),
            enable_completion=enable_completion,
            feedback_modes=feedback_modes,
        )

    def _load_predefined_translator_output(self) -> dict:
        """Load cached translator output from previous runs."""
        search_paths = [
            Path(self.outdir) / "translator_output.json",
            Path(f"outputs/{self.task.name}/translator_output.json"),
        ]
        for p in search_paths:
            if p.exists():
                report.note("cached translator output", str(p))
                return json.load(open(p))
        raise ValueError(
            f"No cached translator output for '{self.task.name}'. "
            f"Run once without --skip-translator first."
        )

    def run(self) -> dict:
        """Run the full pipeline; write specs/FSMs/result.json and return the result dict."""
        t0 = time.time()

        if self.skip_translator:
            report.agent("Translator", "skipped (--skip-translator)")
            tra_out = self._load_predefined_translator_output()
        else:
            tra_out = self.translator.run(
                self.task, cache_path=os.path.join(self.outdir, "translator_output.json"),
                refresh=self.retranslate)
        self._save_json("translator_output.json", tra_out)

        dec_out = self.decomposer.run(tra_out, self.task)
        self._save_json("decomposer_output.json",
                        {k: v for k, v in dec_out.items() if k != "ltl_content"})
        if dec_out.get("success") and dec_out.get("ltl_content"):
            with open(os.path.join(self.outdir, "specs.ltl"), "w") as f:
                f.write(dec_out["ltl_content"])

        elapsed = time.time() - t0

        result = {
            "task": self.task.name,
            "outdir": self.outdir,
            "success": dec_out.get("success", False),
            "ltl_file": str(self.decomposer.ltl_file),
            "fsm_files": dec_out.get("fsm_files", {}),
            "elapsed_sec": round(elapsed, 1),
        }

        tra_usage = tra_out.get("usage", {})
        dec_usage = dec_out.get("usage", {})
        tra_in = tra_usage.get("input_tokens", 0)
        tra_out_t = tra_usage.get("output_tokens", 0)
        dec_in = dec_usage.get("input_tokens", 0)
        dec_out_t = dec_usage.get("output_tokens", 0)

        mn = model_name(self.model)
        tra_cost = calc_cost(self.model, tra_in, tra_out_t)
        dec_cost = calc_cost(self.model, dec_in, dec_out_t)


        result["token_usage"] = {
            "translator": {"model": mn, "input_tokens": tra_in, "output_tokens": tra_out_t, "cost_usd": round(tra_cost, 6)},
            "decomposer": {"model": mn, "input_tokens": dec_in, "output_tokens": dec_out_t, "cost_usd": round(dec_cost, 6)},
            "total_input_tokens": tra_in + dec_in,
            "total_output_tokens": tra_out_t + dec_out_t,
            "total_cost_usd": round(tra_cost + dec_cost, 6),
        }

        mins, secs = divmod(int(elapsed), 60)
        report.agent("Done", f"{mins}m {secs}s" if mins else f"{secs}s")
        report.note("specs", result["ltl_file"])
        for name, path in dec_out.get("fsm_files", {}).items():
            report.note(f"FSM {name}", path)
        report.note("tokens", f"{mn} · Translator in {tra_in:,} / out {tra_out_t:,}"
                              f" · Decomposer in {dec_in:,} / out {dec_out_t:,}"
                              f" · ${tra_cost + dec_cost:.4f}")
        self._save_json("result.json", result)
        return result

    def _save_json(self, filename, data):
        path = os.path.join(self.outdir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
