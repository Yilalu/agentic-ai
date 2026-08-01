from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "policies"
CHROMA_DIR = BASE_DIR / "chroma_db"


def get_category(filename: str) -> str:
    filename = filename.lower()

    if "card_dispute" in filename:
        return "card_dispute"
    if "fee_refund" in filename:
        return "fee_refund"
    if "failed_transfer" in filename:
        return "failed_transfer"
    if "account_lockout" in filename:
        return "account_lockout"
    if "fraud" in filename:
        return "fraud"
    if "identity_verification" in filename:
        return "identity_verification"

    return "general"


def build_vector_database() -> None:
    loader = DirectoryLoader(
        str(POLICY_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    for document in documents:
        source_path = Path(document.metadata["source"])
        filename = source_path.name

        document.metadata["category"] = get_category(filename)
        document.metadata["source_file"] = filename
        document.metadata["document_type"] = "banking_policy"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="banking_policies",
    )

    print(f"Stored {len(chunks)} policy chunks in ChromaDB.")


if __name__ == "__main__":
    build_vector_database()