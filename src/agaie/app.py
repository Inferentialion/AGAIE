from __future__ import annotations

import io
import json
import hashlib
import re
import time
import uuid
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Annotated
from urllib.parse import urlencode

import os
import requests
import feedparser
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from pathlib import Path

from trafilatura import fetch_url as tf_fetch_url, extract as tf_extract

PROJECT_ROOT = Path(__file__).resolve().parents[2]   
load_dotenv(PROJECT_ROOT / ".env")                   
load_dotenv()                                       

DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))

RAW_DIR  = DATA_ROOT / "raw"
PROC_DIR = DATA_ROOT / "processed"
JOBS_DIR = DATA_ROOT / "jobs"

DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))

RAW_DIR  = DATA_ROOT / "raw"
PROC_DIR = DATA_ROOT / "processed"
JOBS_DIR = DATA_ROOT / "jobs"
for d in (RAW_DIR, PROC_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


GUARDIAN_API_KEY = (os.environ.get("GUARDIAN_API_KEY") or "").strip()

ALPHA_VANTAGE_API_KEY = (os.environ.get("ALPHA_VANTAGE_API_KEY") or "").strip()

# ---------- utils ----------
def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")[:120]

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def save_jsonlines(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chunk_text(text: str, chunk_tokens: int = 400, overlap_tokens: int = 40) -> List[str]:
    """Whitespace-token chunker; swap for tokenizer-aware later."""
    words = text.split()
    if not words:
        return []
    chunks, step = [], chunk_tokens - overlap_tokens
    for start in range(0, len(words), step):
        end = start + chunk_tokens
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks

def job_path_for(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"

def write_job_file(job_id: str, payload: Dict[str, Any]) -> Path:
    path = job_path_for(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"job_id": job_id, "status": "queued", "payload": payload, "started_at": now_iso()}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path

def update_job(job_id: str, **kw) -> None:
    path = job_path_for(job_id)
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(kw)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _alphavantage_time_to_iso(t: Optional[str]) -> Optional[str]:
    """
    Convert Alpha Vantage 'time_published' (e.g., '20240822T153036' or '20240822T1530')
    into an ISO8601 'YYYY-MM-DDTHH:MM:SSZ'.

    :param t: Alpha Vantage time string without timezone.
    :return: ISO8601 UTC string or None on parse failure.
    """
    if not t:
        return None
    t = t.strip()
    fmts = ["%Y%m%dT%H%M%S", "%Y%m%dT%H%M"]
    for fmt in fmts:
        try:
            return dt.datetime.strptime(t, fmt).replace(microsecond=0).isoformat() + "Z"
        except Exception:
            continue
    return None

# ---------- Pydantic payloads ----------
class GuardianParams(BaseModel):
    type: Literal["guardian"] = "guardian"
    section: str = Field(default="business", description="Guardian section id (e.g., 'business')")
    tags: Optional[List[str]] = Field(default=None, description="Optional Guardian tag ids")
    from_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    to_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    page_size: int = Field(default=100, ge=1, le=200)  # API commonly caps at 200
    max_pages: int = Field(default=3, ge=1, le=50)

class AlphaVantageParams(BaseModel):
    type: Literal["alphavantage"] = "alphavantage"
    tickers: Optional[List[str]] = Field(
        default=None,
        description="List of symbols to filter (e.g., ['AAPL', 'MSFT'] or CRYPTO:/FOREX prefixes)."
    )
    topics: Optional[List[str]] = Field(
        default=None,
        description="Alpha Vantage topics, e.g. ['earnings','mergers_and_acquisitions','economy_monetary']."
    )
    time_from: Optional[str] = Field(
        default=None, description="Lower bound in YYYYMMDDTHHMM (UTC)."
    )
    time_to: Optional[str] = Field(
        default=None, description="Upper bound in YYYYMMDDTHHMM (UTC)."
    )
    sort: Optional[Literal["LATEST", "EARLIEST", "RELEVANCE"]] = "LATEST"
    limit: int = Field(default=100, ge=1, le=1000)
    fetch_full_text: bool = Field(
        default=False,
        description="If true, attempt Trafilatura extraction; else rely on AV summary/title.",
    )
    explode_by_ticker: bool = Field(
        default=False,
        description="If true, emit one NormalizedDoc per (article,ticker) with per-ticker sentiment.",
    )
    rate_limit_sleep_sec: int = Field(
        default=13, ge=0, le=120, description="Sleep between paged/windowed calls if needed."
    )

class RSSParams(BaseModel):
    type: Literal["rss"] = "rss"
    url: HttpUrl
    limit: int = Field(default=20, ge=1, le=200)

class URLParams(BaseModel):
    type: Literal["url"] = "url"
    url: HttpUrl

class PDFParams(BaseModel):
    type: Literal["pdf"] = "pdf"
    url: HttpUrl

SourcePayload = Annotated[GuardianParams | RSSParams | URLParams | PDFParams | AlphaVantageParams,
                          Field(discriminator='type')]  # thus, our SourcePayload is one of the above pydantic models

# ---------- normalized doc model ----------
@dataclass
class NormalizedDoc:
    id: str
    source_type: str
    source_url: str
    title: Optional[str]
    author: Optional[str]
    published_at: Optional[str]
    text: str
    metadata: Dict[str, Any]

# ---------- loaders ----------
def ingest_alphavantage(p: AlphaVantageParams, job_id: str) -> List[NormalizedDoc]:
    """
    Ingest market news via Alpha Vantage NEWS_SENTIMENT.

    :param p: AlphaVantageParams with filters (tickers, topics, time range).
    :param job_id: Ingestion job identifier.
    :return: List of NormalizedDoc items.
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY env var")

    base = "https://www.alphavantage.co/query"
    query: Dict[str, Any] = {
        "function": "NEWS_SENTIMENT",
        "apikey": ALPHA_VANTAGE_API_KEY,
        "sort": p.sort or "LATEST",
        "limit": min(1000, int(p.limit)),
    }
    if p.tickers:
        query["tickers"] = ",".join([t.strip() for t in p.tickers if t.strip()])
    if p.topics:
        query["topics"] = ",".join([t.strip() for t in p.topics if t.strip()])
    if p.time_from:
        query["time_from"] = p.time_from
    if p.time_to:
        query["time_to"] = p.time_to

    # Prepare dirs and fire request
    raw_dir = RAW_DIR / "alphavantage" / job_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base}?{urlencode(query)}"
    r = requests.get(url, timeout=30)
    try:
        r.raise_for_status()
    except Exception as e:
        # Write the raw text for debugging before raising
        (raw_dir / "error_http.txt").write_text(r.text, encoding="utf-8")
        raise

    payload = r.json()
    # Alpha Vantage sends "Note" or "Information" when you hit limits/quotas.
    if isinstance(payload, dict) and any(k in payload for k in ("Note", "Information", "Error Message")):
        (raw_dir / "error_api.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Fail fast; the job file will capture this.
        msg = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "Alpha Vantage error"
        raise RuntimeError(f"Alpha Vantage returned a limit/error message: {msg}")

    # Persist raw for auditability
    (raw_dir / "response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    feed = payload.get("feed") or []
    docs: List[NormalizedDoc] = []
    seen_ids: set[str] = set()

    for item in feed:
        url = item.get("url") or ""
        title = item.get("title")
        pub_iso = _alphavantage_time_to_iso(item.get("time_published"))
        authors = item.get("authors")
        authors_str = ", ".join(authors) if isinstance(authors, list) else (authors or None)
        summary = (item.get("summary") or "").strip()

        # Minimal text for downstream NLP; optionally augment with full text
        text = summary
        if p.fetch_full_text and url:
            try:
                extracted = extract_with_trafilatura(url) or ""
                if extracted.strip():
                    text = extracted
            except Exception:
                pass
        if not (text or title):
            # If nothing to index, skip
            continue

        topics = item.get("topics") or []  # list of {"topic": "...", "relevance_score": "..."}
        overall_score = item.get("overall_sentiment_score")
        overall_label = item.get("overall_sentiment_label")
        ticker_sentiment = item.get("ticker_sentiment") or []  # list of {"ticker": "...", "relevance_score": "...", ...}

        if p.explode_by_ticker and ticker_sentiment:
            for ts in ticker_sentiment:
                tk = ts.get("ticker")
                comp_id = sha1((url or "") + "::" + (tk or ""))
                if comp_id in seen_ids:
                    continue
                seen_ids.add(comp_id)
                docs.append(
                    NormalizedDoc(
                        id=comp_id,
                        source_type="alphavantage",
                        source_url=url,
                        title=title,
                        author=authors_str,
                        published_at=pub_iso,
                        text=text or (title or ""),
                        metadata={
                            "source": item.get("source"),
                            "source_domain": item.get("source_domain"),
                            "category_within_source": item.get("category_within_source"),
                            "banner_image": item.get("banner_image"),
                            "topics": topics,
                            "overall_sentiment_score": overall_score,
                            "overall_sentiment_label": overall_label,
                            "ticker": tk,
                            "ticker_relevance": ts.get("relevance_score"),
                            "ticker_sentiment_score": ts.get("ticker_sentiment_score"),
                            "ticker_sentiment_label": ts.get("ticker_sentiment_label"),
                        },
                    )
                )
        else:
            # Article-level record with all tickers
            doc_id = sha1(url or (title or "") + str(pub_iso))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            docs.append(
                NormalizedDoc(
                    id=doc_id,
                    source_type="alphavantage",
                    source_url=url,
                    title=title,
                    author=authors_str,
                    published_at=pub_iso,
                    text=text or (title or ""),
                    metadata={
                        "source": item.get("source"),
                        "source_domain": item.get("source_domain"),
                        "category_within_source": item.get("category_within_source"),
                        "banner_image": item.get("banner_image"),
                        "topics": topics,
                        "overall_sentiment_score": overall_score,
                        "overall_sentiment_label": overall_label,
                        "tickers": [ts.get("ticker") for ts in ticker_sentiment if ts.get("ticker")],
                        "ticker_sentiment": ticker_sentiment,
                    },
                )
            )

    return docs

def ingest_guardian(p: GuardianParams, job_id: str) -> List[NormalizedDoc]:
    if not GUARDIAN_API_KEY:
        raise RuntimeError("Missing GUARDIAN_API_KEY env var")

    base = "https://content.guardianapis.com/search"
    params = {
        "page-size": min(200, p.page_size),
        "order-by": "newest",
        "show-fields": "bodyText,headline,byline,trailText",
    }
    if p.section:
        params["section"] = p.section
    if p.tags:
        params["tag"] = ",".join(p.tags)
    if p.from_date:
        params["from-date"] = p.from_date
    if p.to_date:
        params["to-date"] = p.to_date

    docs: List[NormalizedDoc] = []
    raw_dir = RAW_DIR / "guardian" / job_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    for page in range(1, p.max_pages + 1):
        q = params | {"api-key": GUARDIAN_API_KEY, "page": page}
        r = requests.get(base, params=q, timeout=30)
        r.raise_for_status()
        payload = r.json()

        # Save raw page
        (raw_dir / f"page_{page}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        results = payload.get("response", {}).get("results", [])
        for it in results:
            fields = it.get("fields") or {}
            text = (fields.get("bodyText") or "").strip()
            if not text:
                continue
            title = fields.get("headline") or it.get("webTitle")
            author = fields.get("byline")
            pub = it.get("webPublicationDate")
            url = it.get("webUrl")
            doc_id = sha1(url or (title or "") + str(pub))
            docs.append(
                NormalizedDoc(
                    id=doc_id,
                    source_type="guardian",
                    source_url=url,
                    title=title,
                    author=author,
                    published_at=pub,
                    text=text,
                    metadata={
                        "sectionId": it.get("sectionId"),
                        "sectionName": it.get("sectionName"),
                        "pillarName": it.get("pillarName"),
                        "trailText": fields.get("trailText"),
                        "apiUrl": it.get("apiUrl"),
                    },
                )
            )

        meta = payload.get("response", {})
        if meta.get("currentPage") >= meta.get("pages", 1):
            break
        time.sleep(0.2)

    return docs

def extract_with_trafilatura(url: str) -> Optional[str]:
    html = tf_fetch_url(url)
    if not html:
        return None
    return tf_extract(html, url=url, favor_precision=True)

def ingest_rss(p: RSSParams, job_id: str) -> List[NormalizedDoc]:
    feed = feedparser.parse(p.url)
    entries = feed.entries[: p.limit]
    docs: List[NormalizedDoc] = []
    raw_dir = RAW_DIR / "rss" / job_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Optional: save feed href or summary
    if getattr(feed, "href", None):
        (raw_dir / "feed.txt").write_text(str(feed.href), encoding="utf-8")

    for e in entries:
        link = getattr(e, "link", None)
        if not link:
            continue
        text = extract_with_trafilatura(link) or ""
        if not text.strip():
            continue
        title = getattr(e, "title", None)
        author = getattr(e, "author", None) if hasattr(e, "author") else None
        pub = getattr(e, "published", None) if hasattr(e, "published") else None
        doc_id = sha1(link)
        docs.append(
            NormalizedDoc(
                id=doc_id,
                source_type="rss",
                source_url=link,
                title=title,
                author=author,
                published_at=pub,
                text=text,
                metadata={"feed": p.url},
            )
        )
    return docs

def ingest_url(p: URLParams, job_id: str) -> List[NormalizedDoc]:
    text = extract_with_trafilatura(p.url) or ""
    if not text.strip():
        return []
    return [
        NormalizedDoc(
            id=sha1(p.url),
            source_type="url",
            source_url=p.url,
            title=slugify(p.url),
            author=None,
            published_at=None,
            text=text,
            metadata={},
        )
    ]

def ingest_pdf(p: PDFParams, job_id: str) -> List[NormalizedDoc]:
    r = requests.get(p.url, timeout=60)
    r.raise_for_status()
    raw_dir = RAW_DIR / "pdf" / job_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{slugify(p.url)}.pdf"
    raw_path.write_bytes(r.content)

    pages_text: List[str] = []
    reader = PdfReader(io.BytesIO(r.content))
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages_text.append(t)
    txt = "\n".join(pages_text).strip()
    if not txt:
        return []
    return [
        NormalizedDoc(
            id=sha1(p.url),
            source_type="pdf",
            source_url=p.url,
            title=slugify(p.url),
            author=None,
            published_at=None,
            text=txt,
            metadata={},
        )
    ]

# ---------- orchestrator ----------
def persist_processed(docs: List[NormalizedDoc], job_id: str) -> Dict[str, Any]:
    """Save normalized docs (JSONL) and chunked docs (JSONL with metadata)."""
    out_dir = PROC_DIR / job_id
    norm_path = out_dir / "normalized.jsonl"
    chunk_path = out_dir / "chunks.jsonl"

    norm_rows, chunk_rows = [], []
    for d in docs:
        norm_rows.append(asdict(d))
        for idx, ch in enumerate(chunk_text(d.text, 400, 40)):
            chunk_rows.append(
                {
                    "id": f"{d.id}::{idx}",
                    "parent_id": d.id,
                    "text": ch,
                    "source_type": d.source_type,
                    "source_url": d.source_url,
                    "title": d.title,
                    "author": d.author,
                    "published_at": d.published_at,
                    "metadata": d.metadata,
                }
            )
    save_jsonlines(norm_path, norm_rows)
    save_jsonlines(chunk_path, chunk_rows)
    return {
        "normalized": len(norm_rows),
        "chunks": len(chunk_rows),
        "paths": {"normalized": str(norm_path), "chunks": str(chunk_path)},
    }

def run_ingest_job(job_id: str, payload: Dict[str, Any]) -> None:
    update_job(job_id, status="running")
    try:
        kind = payload.get("type")
        if kind == "guardian":
            docs = ingest_guardian(GuardianParams(**payload), job_id)
        elif kind == "rss":
            docs = ingest_rss(RSSParams(**payload), job_id)
        elif kind == "url":
            docs = ingest_url(URLParams(**payload), job_id)
        elif kind == "pdf":
            docs = ingest_pdf(PDFParams(**payload), job_id)
        elif kind == "alphavantage":
            docs = ingest_alphavantage(AlphaVantageParams(**payload), job_id)
        else:
            raise ValueError(f"Unknown source type: {kind}")

        stats = persist_processed(docs, job_id)
        update_job(job_id, status="succeeded", finished_at=now_iso(), stats=stats)
    except Exception as e:
        update_job(job_id, status="failed", finished_at=now_iso(), error=str(e))

# ---------- FastAPI ----------
app = FastAPI(title="Corpus-agnostic Ingestion Manager", version="0.1.0")

@app.post("/sources")
def add_source(payload: SourcePayload, bg: BackgroundTasks):
    if isinstance(payload, GuardianParams) and not GUARDIAN_API_KEY:
        raise HTTPException(status_code=400, detail="Missing GUARDIAN_API_KEY")
    if isinstance(payload, AlphaVantageParams) and not ALPHA_VANTAGE_API_KEY:
        raise HTTPException(status_code=400, detail="Missing ALPHA_VANTAGE_API_KEY")

    job_id = uuid.uuid4().hex
    job_file = write_job_file(job_id, payload.model_dump())
    bg.add_task(run_ingest_job, job_id, payload.model_dump())
    return {"job_id": job_id, "job_file": str(job_file)}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    path = job_path_for(job_id)
    if not path.exists():
        raise HTTPException(404, "job not found")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

