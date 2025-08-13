# app/main.py
from __future__ import annotations

import io
import json
import hashlib
import re
import time
import uuid
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal

import os  # kept only for environment variables
import requests
import feedparser
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from pathlib import Path

# Robust article-text extraction
from trafilatura import fetch_url as tf_fetch_url, extract as tf_extract

# ---------- constants & dirs ----------
BASE_DIR = Path(__file__).resolve().parents[1]

# Load .env if present (keeps env concerns separate from paths)
load_dotenv(BASE_DIR / ".env")

RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
JOBS_DIR = BASE_DIR / "data" / "jobs"
for d in (RAW_DIR, PROC_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

GUARDIAN_API_KEY = (os.environ.get("GUARDIAN_API_KEY") or "").strip()

# ---------- small utils ----------
def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9\\-]+", "-", s.lower()).strip("-")[:120]

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

# ---------- Pydantic payloads ----------
class GuardianParams(BaseModel):
    type: Literal["guardian"] = "guardian"
    section: str = Field(default="business", description="Guardian section id (e.g., 'business')")
    tags: Optional[List[str]] = Field(default=None, description="Optional Guardian tag ids")
    from_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    to_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    page_size: int = Field(default=100, ge=1, le=200)  # API commonly caps at 200
    max_pages: int = Field(default=3, ge=1, le=50)

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

SourcePayload = GuardianParams | RSSParams | URLParams | PDFParams

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
        time.sleep(0.2)  # be polite

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

