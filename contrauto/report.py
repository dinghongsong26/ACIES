"""Unified, open-source-friendly terminal output for the two-layer pipeline.

Color only when stdout is a TTY (so piped logs / CI stay plain text). One small vocabulary
used by BOTH layers — the task (discrete) layer and the physical (continuous) layer — so the
orchestrated run reads as one coherent report instead of two mismatched styles.
"""
import sys
import threading
import time

_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def bold(s):  return _c("1", s)
def dim(s):   return _c("2", s)
def green(s): return _c("32", s)
def red(s):   return _c("31", s)
def cyan(s):  return _c("36", s)

_OK = lambda: green("✓")
_NO = lambda: red("✗")


VERSION = "1.0.0"

_LOGO = r"""
      _           ______    _____    ________     ______
     / \        .' ___  |  |_   _|  |_   __  |  .' ____ \
    / _ \      / .'   \_|    | |      | |_ \_|  | (___ \_|
   / ___ \     | |           | |      |  _| _    _.____`.
 _/ /   \ \_   \ `.___.'\   _| |_    _| |__/ |  | \____) |
|____| |____|   `._____.'  |_____|  |________|   \______.'"""


def logo():
    """ACIES banner (italic on terminals that support it), printed once per run."""
    lines = _LOGO.strip("\n").splitlines()
    for ln in lines[:-1]:
        print(_c("1;3;36", ln))
    print(_c("1;3;36", lines[-1]) + dim(f"  version {VERSION}"))


def layer(n: int, title: str):
    """Big header marking one of the two layers."""
    bar = "═" * 64
    print(f"\n{cyan(bar)}")
    print(cyan(bold(f" LAYER {n}  ·  {title}")))
    print(f"{cyan(bar)}")


def section(title: str):
    """Lighter sub-header within a layer."""
    print(f"\n{bold('── ' + title + ' ')}{dim('─' * max(3, 56 - len(title)))}")


def step(ok: bool, label: str, detail: str = ""):
    """A pass/fail line: ✓/✗ + label (+ dim detail)."""
    line = f"   {_OK() if ok else _NO()}  {label}"
    if detail:
        line += dim(f"  — {detail}")
    print(line)


def info(label: str, detail: str = ""):
    """A neutral (non pass/fail) info line: · label (+ dim detail)."""
    print(f"   {dim('·')}  {label}" + (dim(f"  {detail}") if detail else ""))


def summary(ok: bool, title: str, kvs: dict):
    """Final one-line verdict for a layer: ✓/✗ TITLE   k v   k v ..."""
    kv = "   ".join(f"{k} {bold(str(v))}" for k, v in kvs.items())
    print(f"\n{_OK() if ok else _NO()} {bold(title)}   {dim('·')}   {kv}")


# ── Agent-style lines (task layer) ───────────────────────────────────────────
# Claude-Code-like layout: a `⏺ Agent` header, then its results hanging off one `⎿` branch:
#
#   ⏺ Decomposer  LTL → contracts → FSM
#     ⎿  ✓ Step 1  contracts: Robot1, Arbiter, System
#        ✓ Step 2  syntax
#
# Every line goes through _emit so it never collides with a running spinner.

_lock = threading.RLock()
_spin_line = False          # a spinner frame is currently drawn on the last line
_branch_open = False        # the next result line takes the `⎿` elbow
_last_elapsed = None        # seconds the last spinner ran; shown on the next result line


def _emit(line: str):
    global _spin_line
    with _lock:
        if _spin_line:
            sys.stdout.write("\r\033[K")
            _spin_line = False
        print(line, flush=True)


def agent(name: str, subtitle: str = ""):
    """`⏺ Name  subtitle` header opening one agent's block."""
    global _branch_open
    _branch_open = True
    _emit(f"\n{_c('35', '⏺')} {bold(name)}" + (dim(f"  {subtitle}") if subtitle else ""))


def branch():
    """Start a new `⎿` group under the current agent (e.g. one decomposition attempt)."""
    global _branch_open
    _branch_open = True


def _fmt_secs(secs: float) -> str:
    secs = int(secs)
    return f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"


def _result(mark: str, text: str, detail: str = ""):
    global _branch_open, _last_elapsed
    _stop_status()
    elbow = dim("  ⎿  ") if _branch_open else "     "
    _branch_open = False
    took = dim(f" ({_fmt_secs(_last_elapsed)})") if _last_elapsed is not None else ""
    _last_elapsed = None
    _emit(f"{elbow}{mark} {text}{took}" + (dim(f"  {detail}") if detail else ""))


def took(seconds: float):
    """Make the next result line show `(Xs)` for a wait that spanned several spinners."""
    global _last_elapsed
    _last_elapsed = seconds


def ok(text: str, detail: str = ""):   _result(_OK(), text, detail)
def fail(text: str, detail: str = ""): _result(_NO(), text, detail)
def note(text: str, detail: str = ""): _result(dim("·"), text, detail)


def read(path: str, summary: str = ""):
    """Claude-Code-style file read: `⏺ Read(path)` with its summary on a `⎿` line."""
    _emit(f"{_c('32', '⏺')} {bold('Read')}({path})")
    if summary:
        _emit(dim("  ⎿  ") + summary)


def body(text: str):
    """Indented free text (multi-line diagnostics) under the current result."""
    for ln in str(text).splitlines():
        _emit(dim(f"       {ln}"))


def log(text: str):
    """A plain line that is safe to print while a spinner is running."""
    _emit(text)


_status = None             # the spinner started by status(), stopped by the next result line


def status(label: str):
    """Show `✻ label… (12s)` until the next ok/fail/note line (or the next status) replaces it --
    for slow steps that are not one `with` block (e.g. a series of NuXMV / SLUGS calls)."""
    global _status
    # A status that replaces another (e.g. Step 3's per-contract checks) keeps the clock
    # running, so the step's result line reports the whole step's time.
    t0 = _status._t0 if _status is not None else None
    _stop_status()
    _status = spinner(label).__enter__()
    if t0 is not None:
        _status._t0 = t0


def _stop_status():
    global _status
    if _status is not None:
        _status.__exit__(None, None, None)
        _status = None


class spinner:
    """`✻ Translating… (12s)` -- an animated status line while a slow call runs.

    TTY only: piped logs / CI get nothing (no carriage-return noise). The line is erased
    when the block exits, so the result line printed next replaces it.
    """
    _FRAMES = "·✢✳✶✻✽✻✶✳✢"

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        _stop_status()             # one animated line at a time
        self._t0 = time.monotonic()
        if _TTY:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        global _spin_line
        i = 0
        while not self._stop.is_set():
            elapsed = _fmt_secs(time.monotonic() - self._t0)
            frame = self._FRAMES[i % len(self._FRAMES)]
            with _lock:
                sys.stdout.write(f"\r\033[K  {_c('33', frame)} {self.label}… "
                                 f"{dim(f'({elapsed})')}")
                sys.stdout.flush()
                _spin_line = True
            i += 1
            self._stop.wait(0.12)

    def __exit__(self, *exc):
        global _spin_line, _last_elapsed
        # Keep the wait visible: the next ok/fail/note line carries "(2m 6s)".
        _last_elapsed = time.monotonic() - self._t0
        if self._t:
            self._stop.set()
            self._t.join()
            with _lock:
                if _spin_line:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                    _spin_line = False
        return False
