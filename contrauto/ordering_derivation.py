"""Legacy shim for the removed causal-flow mechanism.

An earlier prototype derived required action orderings from an LLM-emitted
CAUSAL FLOW block; solver-driven persistence mining (contrauto/completion.py)
subsumed it. The only surviving duty is stripping any FLOW block a model
still emits so that clean CHASE .ltl text reaches the toolchain. The full
derivation code lives in git history (pre paper-v10).
"""
import re


def strip_flow(text: str) -> str:
    """Remove the CAUSAL FLOW block so only the CHASE .ltl reaches the toolchain."""
    out, in_flow = [], False
    for line in text.splitlines():
        if re.match(r"\s*(CAUSAL\s+)?FLOW\s*:", line, re.I):
            in_flow = True
            continue
        if re.match(r"\s*(NAME\s*:|CONTRACT\b)", line):
            in_flow = False
        if in_flow:
            continue
        out.append(line)
    return "\n".join(out)
