import time
from functools import wraps
from typing import Any, Dict, Optional
from src.agaie.observability.metrics import AGENT_TTFB, TOOL_LATENCY, TOOL_ERRORS, RATE_LIMIT_ERRORS
from openai import RateLimitError as OpenAIRateLimitError
from langchain_core.callbacks import BaseCallbackHandler


def instrument_tool(tool_fn, tool_name: Optional[str] = None):
    """
    Decorator to record tool latency and error categories.

    :param tool_fn: The tool function to wrap.
    :param tool_name: Optional explicit tool name for labeling.
    """

    name = tool_name or getattr(tool_fn, "__name__", "tool")

    @wraps(tool_fn)
    def _wrapped(*args, **kwargs):
        start = time.perf_counter()

        try:
            return tool_fn(*args, **kwargs)
        except Exception as e:
            http_status = "unknown"
            err_type = type(e).__name__

            # Heuristics to classify HTTP error types
            status = getattr(getattr(e, "response", None), "status_code", None)
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
            elif isinstance(e, OpenAIRateLimitError):
                http_status = "429"
                RATE_LIMIT_ERRORS.labels(
                    component="tool", provider="openai", model_or_name=name
                ).inc()

            
            TOOL_ERRORS.labels(tool=name, error_type=err_type, http_status=http_status).inc()
            
            raise  # so that it doesn't go silently
        
        finally:
            TOOL_LATENCY.labels(tool=name).observe(time.perf_counter() - start)
    
    return _wrapped


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




