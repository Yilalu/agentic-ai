"""Build / rebuild the local Chroma index from policies/*.md."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    KNOWLEDGE_BASE_DIR,
)
from src.retriever import create_embedding_model, load_vector_store

POLICY_ID_RE = re.compile(r"\*\*Policy ID:\*\*\s*([A-Z0-9-]+)", re.IGNORECASE)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    body = parts[2].lstrip("\n")
    return meta, body


def build_vector_database(rebuild: bool = True) -> None:
    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        load_vector_store.cache_clear()
        create_embedding_model.cache_clear()

    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
    )
    documents = loader.load()

    for document in documents:
        source_path = Path(document.metadata["source"])
        filename = source_path.name
        meta, body = parse_frontmatter(document.page_content)

        doc_id = meta.get("doc_id") or ""
        if not doc_id:
            match = POLICY_ID_RE.search(document.page_content)
            doc_id = match.group(1).upper() if match else filename.replace(".md", "").upper()

        domain = (meta.get("domain") or "shared").lower()
        doc_type = (meta.get("doc_type") or "policy").lower()
        title = meta.get("title") or filename

        document.page_content = body or document.page_content
        document.metadata["doc_id"] = doc_id
        document.metadata["domain"] = domain
        document.metadata["doc_type"] = doc_type
        document.metadata["title"] = title
        document.metadata["category"] = domain
        document.metadata["source_file"] = filename
        document.metadata["document_type"] = doc_type

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    per_doc_count: dict[str, int] = {}
    for chunk in chunks:
        parent = chunk.metadata.get("doc_id", "POL-UNKNOWN")
        index = per_doc_count.get(parent, 0)
        per_doc_count[parent] = index + 1
        chunk.metadata["doc_id"] = f"{parent}#{index}"
        chunk.metadata["parent_doc_id"] = parent

    Chroma.from_documents(
        documents=chunks,
        embedding=create_embedding_model(),
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print(f"Stored {len(chunks)} policy chunks in ChromaDB at {CHROMA_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Policies indexed: {sorted(per_doc_count)}")

    load_vector_store.cache_clear()


if __name__ == "__main__":
    build_vector_database()
