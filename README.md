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



