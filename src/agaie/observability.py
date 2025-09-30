import time
from functools import wraps
from typing import Any, Dict, Optional
from src.agaie.metrics import AGENT_TFFB, AGENT_LATENCY, TOOL_LATENCY, TOOL_ERRORS, RATE_LIMIT_ERRORS
from openai import RateLimitError as OpenAIRateLimitError


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





