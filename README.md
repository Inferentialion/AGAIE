## Steps to run the app

1) Bring up Weaviate, and Prometheus + Grafana
```
docker compose up -d
``` 

2) Run the app
```
uv run --active uvicorn --app-dir src agaie.app:app --host 0.0.0.0 --port 8080
```

3) Open the UIs
- Prometheus targets: http://localhost:9090/targets -- job agaie-agent should be "UP".
- Grafana: http://localhost:3000 --> Dashboards --> AGAIE --> Agent SLOs.


4) Sanity checks
```
docker compose ps

docker exec -it $(docker ps --filter name=prometheus --format '{{.ID}}') \
  wget -qO- http://host.docker.internal:8080/metrics | head
```

5) Send a query to test the system
```
curl -s -X POST http://localhost:8080/query   -H 'Content-Type: application/json'   -d '{"question":"What is AAPL?","index_name":"finance-news","top_k":3,"dense_k":10,"sparse_k":30}'
```


## AGAIE Workflow

```
                           ┌───────────────────────────────────────────────────────┐
                           │                    FRONTEND (Public)                  │
                           │   /query  (/chat)   [Q&A over corpus]                 │
                           │   /sources  [choose source / "bring your own data"]   │
                           └───────────────────────────────────────────────────────┘

        ┌─────────────────────────────── INGEST LANE (Backend plumbing) ───────────────────────────────┐
          [POST /sources]  (Public or Admin)
                │  creates job_id + bg task
                ▼
          [Ingestion Jobs]  ──► fetch guardian | rss | url | pdf | alphavantage
                ▼
          [Doc Normalizer]  ──► chunk(400/40), metadata, (optional NER tickers)
                ▼
          writes: data/processed/<job_id>/normalized.jsonl & chunks.jsonl
                │
                ├───(Private) [GET /jobs/{job_id}]  ← poll status
                │
                └───(Private) [POST /index/build {job_id, index_name}]
                              └──► builds/updates persistent stores:
                                      ┌───────────────────────────────┐
                                      │ [Dense Index / Vector DB]     │  (embeddings)
                                      └───────────────────────────────┘
                                      ┌───────────────────────────────┐
                                      │ [Sparse Index]                │  (BM25 or SPLADE)
                                      └───────────────────────────────┘

                (Optional automation: after /sources succeeds,
                a background hook auto-calls /index/build)
        └──────────────────────────────────────────────────────────────────────────────────────────────┘


        ┌─────────────────────────────── QUERY LANE (User path) ───────────────────────────────────────┐
          (Public) [POST /query]  or  [POST /chat]
                ▼
          [LangGraph Supervisor]
                │
                ├─► Planner Node      → rewrites question / spawns subqueries
                │
                ├─► Retriever Node    → calls:
                │        ┌────────────────────────────────────────────────────────────┐
                │        │ [Hybrid Fusion Retriever]  (dense + sparse, RRF/weighted) │
                │        └────────────────────────────────────────────────────────────┘
                │                      ▲                       ▲
                │                      │                       │
                │             Dense Index / Vector DB     Sparse Index (BM25/SPLADE)
                │             ----------------------      -------------------------
                │
                ├─► (optional) ColBERT Reranker  → top-k precision bump
                │
                ├─► Answerer Node   → synthesize grounded answer + citations
                │
                └─► Evaluator Node  → quick faithfulness/coverage check
                        └─ if low confidence → retry Planner/Retriever once

          returns:  { answer, citations[], confidence }
        └──────────────────────────────────────────────────────────────────────────────────────────────┘


Legend:
- Public endpoints (frontend):  /query (/chat), optionally /sources
- Private/ops endpoints:        /jobs/{id}, /index/build
- Optional automation:          auto-trigger /index/build when /sources job succeeds
- Same hybrid retriever can also be exposed as a Tool for agentic follow-ups if desired

```


 ## RAG

 Description: builds a dense+BM25 hybrid index from the extracted and processed chunks.jsonl files (data/processed/...), exposes a fusion retriever (RRF/weighted), and provides a debug /search API to inspect top‑K results with scores.


 ## Potential issues and solutions

 If you are running this from linux (as I am) and are having an error with ```uvicorn agaie.app:app --reaload --port X```,
 it most likely related to inotify tracking too many files (e.g. if you have the venv in the project directory, or due to caches, etc.). 

 A hot fix is to use ```WATCHFILES_FORCE_POLLING=1 uvicorn agaie.app:app --reload --port 8001``` so that it uses uvicorn's polling mode which is more relaxed.



## Alpha Vantage API

### What Information Does the Alpha Vantage News API Return?

For a given ticker like AAPL, we’d query:

```https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=...```

and then we decide:

- Do we just want the latest N articles (limit=50, sort=LATEST)?

- Or all articles in a time window (time_from=20250820T0000&time_to=20250823T2359)?

- The API will return up to limit articles in that scope. The free plan caps limit at 1000 per call.

When we invoke the function=NEWS_SENTIMENT endpoint, Alpha Vantage returns a JSON feed with these key fields (exposed per article):

- url — the article URL

- title — headline text

- time_published — timestamp in a compact format like 20240822T153036 or 20240822T1530

- authors — list or string of author names

- summary — short excerpt or summary of the article

- overall_sentiment_score and overall_sentiment_label — a numeric score and human-readable label (e.g., "POSITIVE", "NEGATIVE", "NEUTRAL")

- ticker_sentiment — an array linking tickers mentioned in the article, each with its own relevance and sentiment metrics

- topics — topic tags assigned to the article, such as earnings, mergers_and_acquisitions, economy_monetary, etc.

- Additional metadata like source, source_domain, category_within_source, and banner_image may also be provided, depending on the article’s origin.

This structure is intentionally designed for trading: it provides entities (tickers, topics), sentiment labels, and a timestamp, allowing you to build pipelines and event studies without needing to extract full article text.

Example:
<pre> 
json
{
  "items": 1,
  "sentiment_score_definition": "The sentiment score is based on a 0 to 1 scale, where scores closer to 1 indicate more positive sentiment, and closer to 0 indicate more negative sentiment.",
  "relevance_score_definition": "The relevance score quantifies how closely the article relates to the mentioned ticker, from 0 (not relevant) to 1 (highly relevant).",
  "feed": [
    {
      "title": "Apple beats earnings expectations, announces $90B buyback",
      "url": "https://www.cnbc.com/2025/05/02/apple-earnings-q1.html",
      "time_published": "20250502T213000",
      "authors": ["Steve Kovach"],
      "summary": "Apple reported quarterly earnings that exceeded Wall Street expectations and announced a $90 billion stock buyback program.",
      "banner_image": "https://image.cnbcfm.com/api/v1/image/106875123-1625257120216-apple-earnings.jpg",
      "source": "CNBC",
      "category_within_source": "Business",
      "source_domain": "cnbc.com",
      "overall_sentiment_score": 0.65,
      "overall_sentiment_label": "Positive",
      "topics": [
        {"topic": "earnings", "relevance_score": "0.90"},
        {"topic": "buybacks", "relevance_score": "0.85"}
      ],
      "ticker_sentiment": [
        {
          "ticker": "AAPL",
          "relevance_score": "0.95",
          "ticker_sentiment_score": "0.72",
          "ticker_sentiment_label": "Positive"
        },
        {
          "ticker": "MSFT",
          "relevance_score": "0.15",
          "ticker_sentiment_score": "0.10",
          "ticker_sentiment_label": "Neutral"
        }
      ]
    }
  ]
}
 </pre>


