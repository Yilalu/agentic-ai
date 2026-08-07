"""Load Chroma and retrieve policy chunks for the agents."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from src.schemas import Domain, Source

_POLICY_ID = re.compile(r"\*\*Policy ID:\*\*\s*([A-Z0-9-]+)", re.IGNORECASE)


@lru_cache(maxsize=1)
def create_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def load_vector_store() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory was not found: {CHROMA_DIR}. "
            "Run python -m scripts.build_vector_db first."
        )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embedding_model(),
        persist_directory=str(CHROMA_DIR),
    )
    if vector_store._collection.count() == 0:
        raise ValueError(
            f"The {COLLECTION_NAME} ChromaDB collection is empty. "
            "Run python -m scripts.build_vector_db again."
        )
    return vector_store


def get_vector_store_status() -> dict[str, Any]:
    try:
        vector_store = load_vector_store()
        return {
            "success": True,
            "collection_name": COLLECTION_NAME,
            "persist_directory": str(CHROMA_DIR),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "document_count": vector_store._collection.count(),
        }
    except Exception as error:
        return {"success": False, "error": str(error)}


def collection_size() -> int:
    status = get_vector_store_status()
    if not status.get("success"):
        raise FileNotFoundError(status.get("error", "vector store unavailable"))
    return int(status["document_count"])


def domain_filter(domain: Domain | str, doc_types: list[str] | None = None) -> dict:
    value = domain.value if isinstance(domain, Domain) else str(domain)
    base = {"domain": {"$in": [value, "shared"]}}
    if not doc_types:
        return base
    return {"$and": [base, {"doc_type": {"$in": list(doc_types)}}]}


def retrieved_ids(sources: list[Source]) -> set[str]:
    ids: set[str] = set()
    for source in sources:
        ids.add(source.doc_id)
        ids.add(source.doc_id.split("#", 1)[0])
    return ids


def format_sources(sources: list[Source]) -> str:
    if not sources:
        return "(no policy excerpts retrieved)"
    return "\n\n".join(f"[{s.doc_id}] {s.title}\n{s.content}" for s in sources)


def retrieve(
    query: str,
    domain: str | Domain | None = None,
    doc_types: list[str] | None = None,
    k: int = 4,
) -> list[Source]:
    if not query or not str(query).strip():
        return []

    domain_value = domain.value if isinstance(domain, Domain) else (domain or "")
    metadata_filter = domain_filter(domain_value, doc_types) if domain_value else None

    store = load_vector_store()
    try:
        hits = store.similarity_search_with_score(
            query=query.strip(),
            k=k,
            filter=metadata_filter,
        )
    except Exception:
        hits = store.similarity_search_with_score(
            query=query.strip(),
            k=k,
            filter={"domain": {"$in": [domain_value, "shared"]}} if domain_value else None,
        )

    sources: list[Source] = []
    for index, (document, distance) in enumerate(hits):
        content = document.page_content or ""
        meta = document.metadata or {}
        doc_id = meta.get("doc_id")
        if not doc_id:
            match = _POLICY_ID.search(content)
            parent = match.group(1).upper() if match else "POL-UNKNOWN"
            doc_id = f"{parent}#{index}"
        title = next(
            (line.strip().lstrip("# ").strip() for line in content.splitlines() if line.strip()),
            doc_id,
        )
        try:
            score = max(0.0, 1.0 - float(distance))
        except (TypeError, ValueError):
            score = None
        sources.append(
            Source(
                doc_id=doc_id,
                title=title[:120],
                domain=meta.get("domain") or domain_value or "shared",
                doc_type=meta.get("doc_type") or "policy",
                content=content,
                score=score,
            )
        )
    return sources


if __name__ == "__main__":
    print(get_vector_store_status())
    print(retrieve("duplicate card charge refund", domain="card", k=2))
