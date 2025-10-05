import time
from functools import wraps
from typing import Any, Dict, Optional, Tuple
from agaie.observability.metrics import AGENT_TTFB, TOOL_LATENCY, TOOL_ERRORS, RATE_LIMIT_ERRORS
from openai import RateLimitError as OpenAIRateLimitError
from langchain_core.callbacks import BaseCallbackHandler


class TTFBCallback(BaseCallbackHandler):
    """
    LangChain callback that captures time-to-first-token for streaming LLM calls.
    Records TTFB once per request using the outer 'request_context' dict.

    Rationale: we use a callback to more precisely have insights into the LLM workflow and timings 
    instead of having to measure the end-to-end, application latency.

    :param lane: Logical lane name ("query", "ingest", etc.) for labeling.
    :param model_name: Model identifier used for labeling.
    :param request_context: Mutable dict living at the request boundary where we stash times.
    """
    def __init__(self, lane: str, model_name: str, request_context: Dict[str, Any]):
        self.lane = lane
        self.model_name = model_name
        self.ctx = request_context

    def on_llm_start(self, *args, **kwargs):
        self.ctx.setdefault("req_start", time.perf_counter())   # in case we forget to set it, we initialize it here
        # mark LLM start; outer layer should have set ctx["req_start"]
        self.ctx.setdefault("llm_start", time.perf_counter())

    def on_llm_new_token(self, token: str, **kwargs):
        # first token = TTFB
        if not self.ctx.get("ttfb_recorded"):
            ttfb = time.perf_counter() - self.ctx.get("req_start", time.perf_counter())
            AGENT_TTFB.labels(lane=self.lane, model=self.model_name).observe(ttfb)
            self.ctx["ttfb_recorded"] = True

    def on_llm_error(self, error: BaseException, **kwargs):
        # Count 429s at the LLM boundary if thrown here
        if isinstance(error, OpenAIRateLimitError):
            RATE_LIMIT_ERRORS.labels(component="llm", provider="openai", model_or_name=self.model_name).inc()


class ToolMetricsCallback(BaseCallbackHandler):
    """
    Record per-tool latency and errors using LangChain tool lifecycle hooks.

    We key timers by run_id to support concurrent tool calls.

    :param None: Construct with no arguments; attach via config={"callbacks":[...]}.
    """

    def __init__(self):
        # Map run_id -> (start_time, tool_name)
        self._starts: Dict[Any, Tuple[float, str]] = {}

    # Tool lifecycle --------------------------------------------------------

    def on_tool_start(self, serialized, input_str: str | None = None, run_id=None, **kwargs) -> None:
        # 'serialized' is typically a dict with {"name": "<tool_name>", ...}
        name = "tool"
        if isinstance(serialized, dict):
            name = serialized.get("name") or serialized.get("id") or "tool"
        self._starts[run_id] = (time.perf_counter(), name)

    def on_tool_end(self, output=None, run_id=None, **kwargs) -> None:
        start, name = self._starts.pop(run_id, (None, "tool"))
        if start is not None:
            TOOL_LATENCY.labels(tool=name).observe(time.perf_counter() - start)

    def on_tool_error(self, error: BaseException, run_id=None, **kwargs) -> None:
        start, name = self._starts.pop(run_id, (None, "tool"))
        if start is not None:
            TOOL_LATENCY.labels(tool=name).observe(time.perf_counter() - start)

        # Classify error for counters
        http_status = "unknown"
        err_type = type(error).__name__

        # Heuristic extraction from HTTP client exceptions (requests/httpx style)
        status = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(status, int):
            if status == 429:
                http_status = "429"
                RATE_LIMIT_ERRORS.labels(
                    component="tool", provider="unknown", model_or_name=name
                ).inc()
            elif 500 <= status <= 599:
                http_status = "5xx"
            elif 400 <= status <= 499:
                http_status = "4xx"
            else:
                http_status = str(status)
        elif isinstance(error, OpenAIRateLimitError):
            http_status = "429"
            RATE_LIMIT_ERRORS.labels(
                component="tool", provider="openai", model_or_name=name
            ).inc()

        TOOL_ERRORS.labels(tool=name, error_type=err_type, http_status=http_status).inc()



