"""
- Best practice in production: use explicit lanes (ingest + query) with automatic retrieval, 
    and optionally expose the same retriever as a tool for follow-up hops.

- Why: determinism, observability, cost/SLO control, and easier testing.

- Why LangGraph over hand-rolled FastAPI logic: you get a real state machine (typed state, 
    conditional edges, retries, timeouts, parallel steps) and built-in tracing, instead of 
    re-implementing orchestration inside HTTP handlers.
"""

# TODO: first think of the "app / entry layer", and how it would connect to the ingest / query lanes (e.g. a FastAPI layer for ingestion / query)

