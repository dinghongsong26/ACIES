"""External-tool orchestration for the decomposer's symbolic pipeline.

This module isolates the low-level subprocess plumbing that drives the three
external tools the decomposer depends on:

  * ``logics_tool`` (CHASE) -- saturate / compose / verify / refine / synthesize
  * ``nuXmv``               -- IC3 LTL model checking of the generated .smv files
  * ``SLUGS`` / ``slugs_comp`` -- GR(1) FSM synthesis and counter-strategy extraction

It also owns the chase-command-file generation and the small helpers that parse
the raw tool output (NuXMV counterexamples). The decomposer keeps the LLM loop,
step orchestration, and feedback-prompt construction; ``ToolPipeline`` only runs
tools and reports structured results back.

The tool executables are resolved from the environment so deployments can point
at custom builds without code changes.
"""

import os
import re
import subprocess
from pathlib import Path

from contrauto import report

LOGICS_TOOL = os.environ.get("LOGICS_TOOL", "logics_tool")
SLUGS = os.environ.get("SLUGS", "slugs")
SLUGS_COMP = os.environ.get("SLUGS_COMP", "slugs_comp")
NUXMV = os.environ.get("NUXMV", "nuxmv")
# The FINAL controller synthesis (synthesize_controller, --explicitStrategy) solves a much
# bigger GR(1) game than the realizability-only probes elsewhere in this file (e.g. an
# Arbiter coordinating N robots over MULTIPLE shared resources): 120s is enough for a
# single-resource arbiter but not a 2-resource x 5-robot one. Configurable so a scaled task
# is not stuck with the probe budget.
SLUGS_SYNTH_TIMEOUT = int(os.environ.get("SLUGS_SYNTH_TIMEOUT", "300"))


class ToolPipeline:
    """Runs the CHASE / nuXmv / SLUGS toolchain for a single design problem.

    Holds the workdir and the canonical .ltl / .chase file paths so the
    decomposer and this class agree on where artifacts live. All methods are
    pure tool invocations or tool-output parsing; none touch the LLM.
    """

    def __init__(self, workdir: Path, mission: str, ltl_file: Path,
                 ver_file: Path, syn_file: Path):
        """Bind the pipeline to a workdir and the shared artifact paths.

        Args:
            workdir: Directory where all tool inputs/outputs are written.
            mission: Mission name, used to name refinement_<mission>.smv.
            ltl_file: Path to specs.ltl (the contract source).
            ver_file: Path to verification.chase.
            syn_file: Path to synthesis.chase.
        """
        self.workdir = Path(workdir)
        self.mission = mission
        self.ltl_file = ltl_file
        self.ver_file = ver_file
        self.syn_file = syn_file

    def _save(self, path, content: str):
        """Write `content` to `path` (overwriting)."""
        Path(path).write_text(content)

    # ══════════════════════════════════════════════════════════════════════════
    # Syntax check (logics_tool show)
    # ══════════════════════════════════════════════════════════════════════════
    def parse_ltl(self, content: str):
        """Syntax-check `content` via logics_tool using a minimal 'show' command.

        Returns:
            (ok: bool, output: str) where output is combined stdout/stderr.
        """
        tmp_ltl = self.workdir / "_tmp.ltl"
        tmp_chase = self.workdir / "_tmp.chase"
        self._save(tmp_ltl, content)
        self._save(tmp_chase, "show\n")
        r = subprocess.run(
            [LOGICS_TOOL, "-i", str(tmp_ltl),
             "-o", str(self.workdir), "-c", str(tmp_chase)],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0, r.stderr + r.stdout

    # ══════════════════════════════════════════════════════════════════════════
    # Chase command file generation
    # ══════════════════════════════════════════════════════════════════════════
    def write_chase_command_files(self, contract_content: str) -> dict:
        """Parse CONTRACT names and write the three sets of chase command files.

          synthesis.chase       saturate + synthesize each sub-contract
          verification.chase    saturate all -> compose -> refinement check
          verify_<name>.chase   one per contract for CC check (Step 3)

        Args:
            contract_content: Raw .ltl contract text.

        Returns:
            ci = {system, controllers, verify_files}.
        """
        names = re.findall(r'CONTRACT\s+(\w+)\s*:', contract_content) or ["Robot", "System"]
        sys_name = next((n for n in names if n.lower() == "system"), names[-1])
        ctrls = [n for n in names if n != sys_name] or [names[0]]

        # Do NOT `saturate` before `synthesize`: saturation turns (A,G) into
        # (A, ¬A∨G); the saturated guarantee ¬A∨G is not GR(1)-decomposable, so the
        # SlugsPrinter emits EMPTY [SYS_*] -> a vacuous "do nothing" FSM. CHASE's own
        # demos synthesize WITHOUT saturate (saturate is only for refinement). See
        # docs/EXPERIMENTS.md EXP-004..006.
        self._save(self.syn_file,
            "\n".join(f"synthesize {c} {c.lower()};" for c in ctrls) + "\n"
        )

        ver = [f"saturate {c};" for c in ctrls] + [f"saturate {sys_name};"]
        if len(ctrls) == 1:
            composed = ctrls[0]
        else:
            prev = ctrls[0]
            for i, c in enumerate(ctrls[1:], start=1):
                tag = f"Comp{i}"
                ver.append(f"compose {prev} {c} {tag};")
                prev = tag
            composed = prev
        ver.append(f"refinement {composed} {sys_name} refinement_{self.mission}.smv;")
        self._save(self.ver_file, "\n".join(ver) + "\n")

        verify_files = {}
        for name in names:
            path = self.workdir / f"verify_{name.lower()}.chase"
            self._save(path, f"verify {name} {name.lower()};\n")
            verify_files[name] = path

        return {
            "system": sys_name,
            "controllers": ctrls,
            "verify_files": verify_files,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Consistency & Compatibility (logics_tool verify_<name>.chase -> NuXMV)
    # ══════════════════════════════════════════════════════════════════════════
    def check_consistency_compatibility(self, ci: dict):
        """Run the per-contract consistency/compatibility check.

        For each entry in ci["verify_files"] {ContractName: verify_<name>.chase}:
          1. logics_tool -i specs.ltl -c verify_<name>.chase -V  ->  <name>.smv
          2. NuXMV <name>.smv  ->  pass if exactly 2xfalse, 0xtrue

        Returns:
            (all_passed: bool, error_summary: str).
        """
        errors = []

        for contract_name, chase_file in ci["verify_files"].items():
            report.status(f"Checking {contract_name} (NuXMV)")

            r = subprocess.run(
                [LOGICS_TOOL, "-i", str(self.ltl_file),
                 "-o", str(self.workdir), "-c", str(chase_file), "-V"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                errors.append(f"{contract_name}: logics_tool failed:\n{r.stdout + r.stderr}")
                continue

            smv_file = self.workdir / f"{contract_name.lower()}.smv"
            if not smv_file.exists():
                errors.append(f"{contract_name}: .smv not found at {smv_file}")
                continue

            passed, nuxmv_out = self.run_nuxmv_cc(smv_file)
            if not passed:
                # The first 8 lines of NuXMV output are the version banner — slicing them
                # sent the LLM a copyright notice as "diagnosis". Report the verdict lines.
                verdicts = [l.strip() for l in nuxmv_out.splitlines()
                            if "is true" in l or "is false" in l]
                labelled = [f"  spec {i+1}: {'UNSAT-check FAILED (is true)' if 'is true' in v else 'ok (is false)'}"
                            f" — {v[:180]}"
                            for i, v in enumerate(verdicts)]
                err_short = "\n".join(labelled) or nuxmv_out[-400:]
                errors.append(f"{contract_name}: FAILED (expected 2xfalse):\n{err_short}")

        return (False, "\n\n".join(errors)) if errors else (True, "")

    def run_nuxmv_cc(self, smv_file: Path):
        """Run NuXMV in IC3 mode on a consistency/compatibility .smv file.

        PASS: exactly 2x'is false', 0x'is true'. Timeout treated as PASS (no
        counterexample found).

        Returns:
            (passed: bool, output: str).
        """
        if not smv_file.exists():
            report.note("warning: SMV not found", str(smv_file))
            return False, f"SMV not found: {smv_file}"

        script_file = smv_file.with_suffix(".cmd")
        script_file.write_text(
            f"read_model -i {smv_file}\n"
            "flatten_hierarchy\n"
            "encode_variables\n"
            "build_boolean_model\n"
            "check_ltlspec_ic3\n"
            "quit\n"
        )

        try:
            r = subprocess.run(
                [NUXMV, "-source", str(script_file)],
                capture_output=True, text=True, timeout=120,
            )
            out = r.stdout + r.stderr
            true_count = len(re.findall(r"\bis true\b", out))
            false_count = len(re.findall(r"\bis false\b", out))
            return (false_count == 2 and true_count == 0), out

        except subprocess.TimeoutExpired:
            report.note("NuXMV IC3 timeout", "treated as pass (no counterexample found)")
            return True, "timeout: no counterexample found"

    # ══════════════════════════════════════════════════════════════════════════
    # Refinement (logics_tool verification.chase -> NuXMV)
    # ══════════════════════════════════════════════════════════════════════════
    def run_verification(self):
        """Run logics_tool on verification.chase to emit refinement_<mission>.smv.

        Returns:
            (ok: bool, output: str) where output is combined stderr/stdout.
        """
        r = subprocess.run(
            [LOGICS_TOOL, "-i", str(self.ltl_file),
             "-o", str(self.workdir), "-c", str(self.ver_file), "-V"],
            capture_output=True, text=True, timeout=120,
        )
        return r.returncode == 0, r.stderr + r.stdout

    def run_nuxmv_refinement(self, smv_file: Path):
        """Run NuXMV in IC3 mode on refinement_<mission>.smv.

        PASS: 2x'is true', 0x'is false'. Timeout treated as PASS (no
        counterexample found).

        Returns:
            (passed: bool, output: str).
        """
        if not smv_file.exists():
            report.note("warning: SMV not found", str(smv_file))
            return False, f"SMV not found: {smv_file}"

        script_file = smv_file.with_suffix(".cmd")
        script_file.write_text(
            f"read_model -i {smv_file}\n"
            "flatten_hierarchy\n"
            "encode_variables\n"
            "build_boolean_model\n"
            "check_ltlspec_ic3\n"
            "quit\n"
        )

        try:
            r = subprocess.run(
                [NUXMV, "-source", str(script_file)],
                capture_output=True, text=True, timeout=120,
            )
            out = r.stdout + r.stderr
            true_cnt = len(re.findall(r"\bis true\b", out))
            false_cnt = len(re.findall(r"\bis false\b", out))
            return (true_cnt == 2 and false_cnt == 0), out

        except subprocess.TimeoutExpired:
            report.note("NuXMV IC3 timeout", "treated as pass (no counterexample found)")
            return True, "timeout: no counterexample found"

    # ══════════════════════════════════════════════════════════════════════════
    # FSM synthesis (logics_tool synthesis.chase -> SLUGS_COMP -> SLUGS)
    # ══════════════════════════════════════════════════════════════════════════
    def run_synthesis(self):
        """Run logics_tool on synthesis.chase to emit per-controller .structuredSlugs.

        Returns:
            (ok: bool, stderr: str).
        """
        r = subprocess.run(
            [LOGICS_TOOL, "-i", str(self.ltl_file),
             "-o", str(self.workdir), "-c", str(self.syn_file)],
            capture_output=True, text=True, timeout=120,
        )
        return r.returncode == 0, r.stderr

    def synthesize_controller(self, ctrl: str):
        """Synthesize one controller's FSM via SLUGS_COMP then SLUGS.

          <ctrl>.structuredSlugs  --SLUGS_COMP-->  <ctrl>.slugsin
          <ctrl>.slugsin          --SLUGS------->  <ctrl>.json

        Args:
            ctrl: Controller (contract) name.

        Returns:
            dict describing the outcome, with key "status" one of:
              "missing"      -- .structuredSlugs not produced
              "error"        -- SLUGS reported an error (has "detail")
              "unrealizable" -- GR(1) synthesis unrealizable
              "empty"        -- SLUGS produced an empty FSM
              "ok"           -- FSM written (has "json_path": Path)
        """
        cl = ctrl.lower()
        ss = self.workdir / f"{cl}.structuredSlugs"
        si = self.workdir / f"{cl}.slugsin"
        jo = self.workdir / f"{cl}.json"

        if not ss.exists():
            report.note(f"warning: {ss.name} not found")
            return {"status": "missing"}

        r1 = subprocess.run(
            ["python3", SLUGS_COMP, str(ss)],
            capture_output=True, text=True, timeout=60,
        )
        self._save(si, r1.stdout)

        # --biasForAction: among the certified winning strategies, prefer one that
        # ACTS. Without it slugs may extract a lazy winner that discharges a liveness
        # goal vacuously through an environment assumption (e.g. an arbiter that never
        # serves one robot, covering that robot's fairness via the world's release
        # justice) — the activity gate then rejects it, but no contract edit can steer
        # the extractor, so the repair loop cannot converge (corridor E2E #5:
        # 9 rounds, 'Arbiter never asserts grant_r1' every round).
        try:
            r2 = subprocess.run(
                [SLUGS, "--explicitStrategy", "--biasForAction", "--jsonOutput", str(si)],
                capture_output=True, text=True, timeout=SLUGS_SYNTH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}

        if "error" in r2.stderr.lower():
            return {"status": "error", "detail": r2.stderr[:300]}

        if "unrealizable" in r2.stdout.lower() or "unrealizable" in r2.stderr.lower():
            return {"status": "unrealizable"}

        self._save(jo, r2.stdout)
        if not (jo.exists() and jo.stat().st_size > 0):
            report.note(f"warning: empty FSM for {ctrl}")
            return {"status": "empty"}

        return {"status": "ok", "json_path": jo}

    def check_realizable(self, ctrl: str, content: str):
        """Realizability oracle for one controller on a GIVEN .ltl variant (for core search).

        Writes `content` to a throwaway .ltl, synthesizes ONLY `ctrl` into a `_core`-suffixed
        basename (so the canonical synthesis.chase / <ctrl>.structuredSlugs are untouched),
        and runs SLUGS for a realizable/unrealizable verdict only.

        Returns:
            True  -- realizable
            False -- unrealizable
            None  -- inconclusive (tool error / no structuredSlugs / no verdict string)
        """
        cl = ctrl.lower()
        base = f"{cl}_core"
        tmp_ltl = self.workdir / f"_{base}.ltl"
        tmp_syn = self.workdir / f"_{base}.chase"
        ss = self.workdir / f"{base}.structuredSlugs"
        si = self.workdir / f"{base}.slugsin"
        # Remove stale artifacts first: a previous subset's .structuredSlugs left behind would
        # make a crashed/failed logics_tool run look like a valid verdict (false localization).
        for stale in (ss, si):
            if stale.exists():
                stale.unlink()
        self._save(tmp_ltl, content)
        self._save(tmp_syn, f"synthesize {ctrl} {base};\n")

        subprocess.run(
            [LOGICS_TOOL, "-i", str(tmp_ltl), "-o", str(self.workdir), "-c", str(tmp_syn)],
            capture_output=True, text=True, timeout=120,
        )
        if not ss.exists():
            return None

        r1 = subprocess.run(["python3", SLUGS_COMP, str(ss)],
                            capture_output=True, text=True, timeout=60)
        self._save(si, r1.stdout)
        try:
            r2 = subprocess.run([SLUGS, str(si)], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return None
        out = (r2.stdout + r2.stderr).lower()
        if "unrealizable" in out:
            return False
        if "realizable" in out:
            return True
        return None

    def check_consistent(self, ctrl: str, content: str):
        """Consistency/compatibility oracle for one controller on a GIVEN .ltl variant.

        The analogue of :meth:`check_realizable` for the Step-3 consistency check: writes
        `content`, verifies ONLY `ctrl` into a `_ccore`-suffixed .smv, and runs nuXmv (IC3).
        A variant PASSES (returns True) when it is both consistent and compatible (the same
        ``2x false`` criterion the pipeline's Step 3 uses).

        Returns:
            True  -- consistent and compatible
            False -- inconsistent (or incompatible)
            None  -- inconclusive (tool error / no .smv produced)
        """
        cl = ctrl.lower()
        base = f"{cl}_ccore"
        tmp_ltl = self.workdir / f"_{base}.ltl"
        tmp_ver = self.workdir / f"_{base}.chase"
        smv = self.workdir / f"{base}.smv"
        if smv.exists():
            smv.unlink()                       # avoid a stale .smv masking a failed logics_tool run
        self._save(tmp_ltl, content)
        self._save(tmp_ver, f"verify {ctrl} {base};\n")
        subprocess.run(
            [LOGICS_TOOL, "-i", str(tmp_ltl), "-o", str(self.workdir), "-c", str(tmp_ver), "-V"],
            capture_output=True, text=True, timeout=60,
        )
        if not smv.exists():
            return None
        passed, _out = self.run_nuxmv_cc(smv)
        return bool(passed)

    def counterstrategy_held_inputs(self, ctrl: str, in_vars: set) -> dict:
        """Inputs the environment holds CONSTANT across its winning counterstrategy for `ctrl`.

        Runs `slugs --counterStrategy` on the controller's .slugsin (produced earlier by
        :meth:`synthesize_controller`) and returns {input_var: "TRUE"|"FALSE"} for every input
        that never changes value in the counterstrategy — the frequent root cause of an
        unrealizable liveness guarantee, and the seed for assumption mining (gr1_assume).
        Empty if there is no .slugsin / no counterstrategy / nothing held.
        """
        si = self.workdir / f"{ctrl.lower()}.slugsin"
        if not si.exists():
            return {}
        try:
            r = subprocess.run([SLUGS, "--counterStrategy", str(si)],
                               capture_output=True, text=True, timeout=120)
        except Exception:
            return {}
        lines = [l.strip() for l in (r.stdout + r.stderr).splitlines()
                 if l.startswith("State") or "successors" in l]
        vals = {}
        for l in lines:
            m = re.search(r"-> <(.*)>", l)
            if m:
                for kv in m.group(1).split(","):
                    if ":" in kv:
                        k, v = kv.split(":")
                        vals.setdefault(k.strip(), set()).add(v.strip())
        return {k: next(iter(s)) for k, s in vals.items()
                if len(s) == 1 and k in in_vars}

    def counterstrategy_diagnosis(self, ctrls, in_vars: set) -> str:
        """Diagnose UNREALIZABLE contracts via `slugs --counterStrategy`.

        For each contract, run SLUGS in counter-strategy mode to get the
        ENVIRONMENT's winning strategy (how it forces the system to lose), and
        turn it into a concise reason the LLM can act on: a short trace + which
        INPUTS the env holds fixed (a frequent root cause of unrealizable
        liveness guarantees).

        Args:
            ctrls: Controller names known to be unrealizable.
            in_vars: Set of input-variable names (used to flag held inputs).

        Returns:
            A human-readable diagnosis string, one block per controller.
        """
        blocks = []
        for ctrl in ctrls:
            si = self.workdir / f"{ctrl.lower()}.slugsin"
            if not si.exists():
                blocks.append(f"[{ctrl}] (no .slugsin to diagnose)")
                continue
            try:
                r = subprocess.run([SLUGS, "--counterStrategy", str(si)],
                                   capture_output=True, text=True, timeout=120)
            except Exception as e:
                blocks.append(f"[{ctrl}] counterStrategy failed: {type(e).__name__}")
                continue
            lines = [l.strip() for l in (r.stdout + r.stderr).splitlines()
                     if l.startswith("State") or "successors" in l]
            trace = "\n".join(lines[:24]) or "(empty counterstrategy)"
            # which inputs does the environment hold constant across the whole play?
            vals = {}
            for l in lines:
                m = re.search(r"-> <(.*)>", l)
                if m:
                    for kv in m.group(1).split(","):
                        if ":" in kv:
                            k, v = kv.split(":")
                            vals.setdefault(k.strip(), set()).add(v.strip())
            held = {k: next(iter(s)) for k, s in vals.items()
                    if len(s) == 1 and k in in_vars}
            hint = ""
            if held:
                hint = ("\n  Environment can hold these INPUTS fixed forever: "
                        + ", ".join(f"{k}={v}" for k, v in held.items())
                        + " — so any guarantee that needs one of them to change (e.g. G F(x), "
                          "or a guarantee conditioned on it) is UNREALIZABLE unless an ASSUMPTION "
                          "forces it or it is made an output.")
            blocks.append(f"[{ctrl}] environment counterstrategy:\n{trace}{hint}")
        return "\n\n".join(blocks)

    # ══════════════════════════════════════════════════════════════════════════
    # NuXMV counterexample parsing helpers
    # ══════════════════════════════════════════════════════════════════════════
    def extract_counterexample(self, nuxmv_output: str) -> list:
        """Parse a NuXMV trace into a list of (state_header, [var=val ...])."""
        counterexample, current_state, state_info = [], None, []
        for raw in nuxmv_output.splitlines():
            line = raw.strip()
            if "-> State:" in line:
                if current_state:
                    counterexample.append((current_state, state_info))
                current_state, state_info = line, []
            elif "=" in line:
                state_info.append(line)
        if current_state:
            counterexample.append((current_state, state_info))
        return counterexample

    def format_counterexample(self, counterexample: list) -> str:
        """Render the parsed counterexample (or raw list) as indented text."""
        lines = []
        for state, variables in counterexample:
            lines.append(state)
            lines.extend(f"  {v}" for v in variables)
        return "\n".join(lines)
