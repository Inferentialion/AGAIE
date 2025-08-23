## Conceptual AGAIE Workflow

```
[Sources API] ──► [Ingestion Jobs]
   (guardian|rss|url|pdf)      │
                               ▼
                        [Doc Normalizer]
                   (chunk, metadata, NER tickers)
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
          [Dense Index / Vector DB]   [Sparse Index]
                (embeddings)            (BM25 or SPLADE)
                  │                         │
                  └─────► [Hybrid Fusion Retriever] ◄─────┘
                                  │
                           [ColBERT Reranker]
                                  │
                    [LangGraph Agents: Planner → Retriever
                         → Answerer → Evaluator]
                                  │
                            [/chat | /rag APIs]
```



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