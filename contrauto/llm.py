"""Lightweight LLM interface via LiteLLM.

Replaces the OpenAI Agents SDK + per-provider AsyncOpenAI client setup.
All models are called through a single function.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

import litellm

from contrauto import report

# Quiet the third-party noise that otherwise interleaves with the pipeline's own output
# (LiteLLM's per-call DeprecationWarning about temperature/top_p on Gemini 3+, debug banners).
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

MODEL_ALIASES = {
    "gpt":      "openai/gpt-5.5",
    "claude":   "anthropic/claude-opus-5",
    "gemini":   "gemini/gemini-3.1-pro-preview",
    "gemini3":  "gemini/gemini-3-pro-preview",
    "gemini25": "gemini/gemini-2.5-pro",
    "flash":    "gemini/gemini-3.5-flash",   # separate per-model daily quota from 3.1-pro
    "flash3":   "gemini/gemini-3-flash-preview",
    "deepseek": "deepseek/deepseek-v4-pro",
    "glm":      "openai/glm-5.1",
    "kimi":     "openai/kimi-k2.6",
    "kimifast": "openai/kimi-k2.7-code-highspeed",
    "kimik3":   "openai/kimi-k3",
    "groq":     "groq/llama-3.3-70b-versatile",
}

# Default per-call timeout (s) for models whose single calls routinely exceed 300s.
SLOW_MODEL_TIMEOUTS = {
    "deepseek": 900,
}

CUSTOM_ENDPOINTS = {
    "glm":      "https://api.z.ai/api/coding/paas/v4",
    "kimi":     "https://api.moonshot.cn/v1",
    "kimifast": "https://api.moonshot.cn/v1",
    "kimik3":   "https://api.moonshot.cn/v1",
}

TOKEN_PRICES = {
    "deepseek-v4-pro":          {"input": 0.43,  "output": 0.87},
    "kimi-k2.6":                {"input": 0.95,  "output": 4.00},
    "glm-5.1":                  {"input": 1.40,  "output": 4.40},
    "gemini-3.1-pro-preview":   {"input": 2.00,  "output": 12.00},
    "gpt-5.5":                  {"input": 5.00,  "output": 30.00},
}

MAX_RETRIES = 3


def _run_with_hard_timeout(fn, timeout):
    """Run `fn()` on a daemon thread and enforce a REAL wall-clock cutoff.

    litellm's own `timeout=` kwarg was observed to not bound total call duration for at
    least one provider (anthropic/claude-opus-5): a single call sat on one established
    connection for 30+ minutes with zero output, never raising, so neither MAX_RETRIES nor
    a caller's `deadline` ever got a chance to run (both only fire from inside the `except`
    block below). This is the actual backstop -- independent of whatever the SDK/provider
    does internally.

    The thread is daemonized and NOT joined on timeout: a `ThreadPoolExecutor` (or any
    non-daemon Thread) left running would make Python's own atexit machinery hang waiting
    to join it when the process tries to exit, i.e. the fix would trade a hung LLM call for
    a hung process shutdown. A daemon thread that never finishes is simply abandoned when
    the process exits.
    """
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:      # noqa: BLE001 - relayed to the caller's thread
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"hard timeout after {timeout:.0f}s -- call still running in background")
    if "error" in box:
        raise box["error"]
    return box["value"]


def resolve_model(alias: str) -> str:
    """Expand a short alias to its full provider/model id (pass through if unknown)."""
    return MODEL_ALIASES.get(alias, alias)


def model_name(alias: str) -> str:
    """The bare model name (no provider prefix), used for pricing lookup and display."""
    full = resolve_model(alias)
    return full.split("/", 1)[-1] if "/" in full else full


def chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    deadline: float | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, dict]:
    """Single system+user completion with retry/backoff.

    `deadline` (optional): a `time.monotonic()` timestamp after which this call stops
    retrying and raises immediately instead of sleeping into another rate-limit/backoff
    wait -- lets a caller with its own overall time budget (e.g. DecomposerAgent's
    DECOMPOSER_TIME_BUDGET_SEC) bail out of a single call that would otherwise burn the
    whole budget alone (observed: a rate-limited call retrying 20x65s = ~21 minutes).

    Returns:
        (text, usage) where usage has input_tokens / output_tokens.
    """
    model_id = resolve_model(model)
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    if model.startswith("kimi"):
        # api.moonshot.cn rejects any temperature other than 1 for the kimi models.
        temperature = 1.0

    if model in CUSTOM_ENDPOINTS:
        kwargs["api_base"] = CUSTOM_ENDPOINTS[model]
        api_key_env = "KIMI_API_KEY" if model.startswith("kimi") else {"glm": "GLM_API_KEY"}.get(model)
        if api_key_env:
            kwargs["api_key"] = os.environ.get(api_key_env, "")

    # Robust to transient endpoint timeouts / hiccups: generous timeout + retry with backoff,
    # so one flaky network call does not crash a whole pipeline run.
    last_err = None
    response = None
    # A rate-limit (e.g. groq's per-minute token budget) is not a hard failure -- wait out the
    # window and retry many times (independent of MAX_RETRIES), so a whole multi-attempt run does
    # not die on one throttle. Other errors keep the short backoff / few retries (usually terminal).
    attempt = 0
    rate_retries = 0
    # Per-call time budget. 300s covers every normal decomposition call on the current models;
    # a slow strong model (kimi-k3's 4-contract generation can exceed 600s) gets more via
    # LLM_TIMEOUT. The previous fixed 1800s budget let one wedged endpoint stall a pipeline
    # run for a silent half hour per retry (observed: a 1h hang inside a single call).
    # deepseek-v4-pro reasons 25k-37k tokens on the warehouse translator call alone (measured
    # 305s and 462s), so it defaults to 900s; LLM_TIMEOUT still overrides every model.
    call_timeout = int(os.environ.get("LLM_TIMEOUT", SLOW_MODEL_TIMEOUTS.get(model, 300)))
    while True:
        try:
            response = _run_with_hard_timeout(
                lambda: litellm.completion(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    # temperature=temperature,
                    timeout=call_timeout,
                    **kwargs,
                ),
                call_timeout + 30,   # grace period so litellm's own timeout gets first chance
            )
            break
        except Exception as e:
            last_err = e
            if deadline is not None and time.monotonic() >= deadline:
                report.log(report.dim("     [llm] deadline reached — not retrying further"))
                raise
            is_rate = "ratelimit" in type(e).__name__.lower() or "rate limit" in str(e).lower()
            if is_rate and rate_retries < 20:
                rate_retries += 1
                report.log(report.dim(f"     [llm] rate-limited ({rate_retries}/20) — waiting 65s"))
                time.sleep(65)         # clear the per-minute token window, then retry
                continue
            attempt += 1
            if attempt < MAX_RETRIES:
                report.log(report.dim(f"     [llm] {type(e).__name__} (attempt {attempt}/{MAX_RETRIES})"
                                      f" — retrying: {' '.join(str(e).split())[:300]}"))
                time.sleep(8 * attempt)
                continue
            raise
    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, usage


def calc_cost(model_alias: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a call, from TOKEN_PRICES (0 if the model is unpriced)."""
    name = model_name(model_alias)
    prices = TOKEN_PRICES.get(name, {"input": 0, "output": 0})
    return (input_tokens / 1_000_000 * prices["input"]) + \
           (output_tokens / 1_000_000 * prices["output"])


def parse_json_response(text: str) -> dict:
    """Parse a JSON object from an LLM reply, stripping ```fences``` first."""
    clean = re.sub(r'```[a-z]*\n?|```', '', text).strip()
    return json.loads(clean)
