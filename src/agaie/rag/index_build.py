from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import weaviate
from weaviate.classes.config import Configure, Property, DataType

from agaie.app import DATA_ROOT
from dotenv import load_dotenv


load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", None)  # optional for local
WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "FinanceNewsChunk")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PROC_DIR = DATA_ROOT / "processed"
INDEX_DIR = DATA_ROOT / "indexes"

REQUIRED_FIELDS = {"id", "text"}

def read_chunks_jsonl(chunks_path: Path) -> List[Dict[str, Any]]:
    """Contract that checks required fields between ingestion and indexing as well as json validity."""
    rows = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)            
            except Exception as e:
                raise ValueError(f"JSONL parse error at line {ln}: {e}")
            missing = REQUIRED_FIELDS - set(obj)
            if missing:
                raise ValueError(f"Missing required fields {missing} at line {ln}")
            if not (obj.get("text") or "").strip():
                # skip empty chunks; they only add noise
                continue
            rows.append(obj)
        
        if not rows:
            raise ValueError("No valid chunk rows found.")
        
        return rows
    

META_KEYS = [
    "parent_id", "source_type", "source_url", "title",
    "author", "published_at", "metadata"
]

def rows_to_nodes(rows) -> List[TextNode]:
    """Put the already split corpus into nodes (for llamaindex interfacing). For each "row" (splitted unit),
    set the metadata to a separate metadata key. Each node carries the text + metadata for retrieval and citation."""
    
    nodes = []

    for r in rows:
        # flatten our metadata shape
        meta = {k: r.get(k) for k in META_KEYS if r.get(k) is not None}
        if isinstance(meta.get("metadata"), dict):
            # merge nested metadata into top-level meta
            extra = meta.pop("metadata")
            meta.update(extra)
        
        node = TextNode(
            id_=r["id"],
            text=r["text"],
            metadata=meta,
        )

        nodes.append(node)
    
    return nodes


def make_embedder(model_name=None) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=model_name or EMBED_MODEL)


def build_vector_store(class_name="FinanceNewsChunk"):

    client = weaviate.connect_to_local(host="http://localhost:8080")

    # v4 client
    client = weaviate.connect_to_local(
        host=WEAVIATE_URL,  # if using remote + API key, use connect_to_weaviate_cloud / connect_to_custom
    ) if WEAVIATE_API_KEY is None else weaviate.connect_to_custom(
        http_host=WEAVIATE_URL.replace("http://", "").replace("https://", ""),
        http_secure=WEAVIATE_URL.startswith("https"),
        auth_client_secret=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        )

    # Define schema for FinanceNewsChunk
    class_obj = {
        "class": class_name,
        "description": "Chunks of finance news articles with metadata",
        "vectorizer": "none",   # we provide embeddings
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "title", "dataType": ["string"]},
            {"name": "author", "dataType": ["string"]},
            {"name": "source_type", "dataType": ["string"]},
            {"name": "published_at", "dataType": ["date"]},
        ]
    }

    if not client.schema.exists("FinanceNewsChunk"):
        client.schema.create_class(class_obj)

    print(client.is_ready())  # True if server is up

    return WeaviateVectorStore(
        weaviate_client=client,
        class_name=class_name,
        text_key="text",
    )


def build_vector_index(chunks_path, index_name = "finance-news", use_weaviate=False, embedding_model_name = None) -> VectorStoreIndex:
    """Build, persist and update the vector index.

    Create/update VectorStoreIndex at data/indexes/{index_name}.
    If it already exists, we load and insert new nodes."""

    rows = read_chunks_jsonl(chunks_path=chunks_path)
    nodes = rows_to_nodes(rows)
    embedder = make_embedder(embedding_model_name)

    if use_weaviate:
        # Stateless remote-backed path (safe for multi-process ingestion).
        vector_store = build_vector_store()
        storage = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedder)
        index.insert_nodes(nodes)
        return index
    
    persist_dir = INDEX_DIR / index_name
    persist_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageContext.from_defaults(persist_dir=str(persist_dir))

    index_path_files = {p.name for p in persist_dir.glob("*")}
    print(index_path_files)
    has_index = "docstore.json" in index_path_files or "vector_store.json" in index_path_files
    
    if has_index:
        index = load_index_from_storage(storage, embed_model=embedder)
        index.insert_nodes(nodes)
    else:
        documents = [Document(text=n.text, metadata=n.metadata) for n in nodes]
        index = VectorStoreIndex.from_documents(documents, storage_context=storage, embed_model=embedder)

    index.storage_context.persist()

    return index

def load_index(index_name: str, use_weaviate: bool = False, embed_model = None):
    embedder = make_embedder()
    
    if use_weaviate:
        # attach to remote store to reaccess it
        vs = build_vector_store()
        return VectorStoreIndex.from_vector_store(vs, embed_model=embedder)
    else:
        persist_dir = INDEX_DIR / index_name
        storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(storage, embed_model=embed_model)
    

def make_hybrid_fusion_retriever():
    """WIP"""
    pass

def hybrid_search():
    """WIP"""
    pass


def test_retrieval(index_name, query, k):
    vector_store = build_vector_store(class_name="FinanceNewsChunk")
    # TODO: we don't need StorageContext here, and we don't need load_index_from_storage,
    # as we are using weaviate as the single source of truth, aiming a a multiprocess approach later on (stateless).

    