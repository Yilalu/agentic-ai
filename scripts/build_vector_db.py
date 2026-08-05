"""Build / rebuild the local Chroma index from policies/*.md."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "policies"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "banking_policies"

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

    loader = DirectoryLoader(
        str(POLICY_DIR),
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    per_doc_count: dict[str, int] = {}
    for chunk in chunks:
        parent = chunk.metadata.get("doc_id", "POL-UNKNOWN")
        index = per_doc_count.get(parent, 0)
        per_doc_count[parent] = index + 1
        chunk.metadata["doc_id"] = f"{parent}#{index}"
        chunk.metadata["parent_doc_id"] = parent

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print(f"Stored {len(chunks)} policy chunks in ChromaDB at {CHROMA_DIR}")
    print(f"Policies indexed: {sorted(per_doc_count)}")

    try:
        from src.retriever import load_vector_store

        load_vector_store.cache_clear()
    except Exception:
        pass


if __name__ == "__main__":
    build_vector_database()
