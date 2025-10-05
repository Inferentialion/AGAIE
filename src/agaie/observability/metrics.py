from prometheus_client import Counter, Histogram


# adjust to our observed latency envelope (typical range of end-to-end response latency)

AGENT_TTFB = Histogram(
    "agent_ttfb_seconds",
    "Time-to-first-byte for agent responses",
    ["lane", "model"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2, 3, 5, 8, 13, 21)  # how many observerd values are <= n seconds
)

AGENT_LATENCY = Histogram(
    "agent_request_seconds",
    "End-to-end agent request latency",
    ["lane", "model"],
    buckets=(0.1, 0.2, 0.5, 1, 2, 3, 5, 8, 13, 21, 34)
)

TOOL_LATENCY  = Histogram(
    "tool_call_seconds",
    "Latency per tool invocation",
    ["tool"],
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)
)

TOOL_ERRORS = Counter(
    "tool_call_errors_total",
    "Tool call errors by type and HTTP status class",
    ["tool", "error_type", "http_status"],

)

RATE_LIMIT_ERRORS = Counter(
    "rate_limit_errors_total",
    "Rate limit (429) errors seen across components",
    ["component", "provider", "model_or_name"]
)

