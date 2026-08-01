from pathlib import Path
from typing import Any
from typing import List, Optional


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "banking_policies"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Categories that should match the metadata created by
# scripts/build_vector_db.py.
ALLOWED_CATEGORIES = {
    "card_dispute",
    "fee_refund",
    "failed_transfer",
    "account_lockout",
    "fraud",
    "identity_verification",
    "general",
}


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the Sentence Transformer embedding model.

    The same embedding model must be used when:
    1. Building the ChromaDB collection.
    2. Searching the ChromaDB collection.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# ---------------------------------------------------------
# Vector-store loading
# ---------------------------------------------------------

def load_vector_store() -> Chroma:
    """
    Load the existing ChromaDB banking-policy collection.

    Raises:
        FileNotFoundError:
            If the chroma_db directory does not exist.

        ValueError:
            If the directory exists but the collection is empty.
    """

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory was not found: {CHROMA_DIR}. "
            "Run scripts/build_vector_db.py first."
        )

    embeddings = create_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    collection_count = vector_store._collection.count()

    if collection_count == 0:
        raise ValueError(
            "The banking_policies ChromaDB collection is empty. "
            "Run scripts/build_vector_db.py again."
        )

    return vector_store


# ---------------------------------------------------------
# Result formatting
# ---------------------------------------------------------

def format_document(
    document: Any,
    score: Optional[float] = None,
) -> dict[str, Any]:
    """
    Convert a LangChain Document into a JSON-friendly dictionary.
    """

    metadata = document.metadata or {}

    result = {
        "content": document.page_content,
        "policy_id": metadata.get("policy_id"),
        "category": metadata.get("category"),

        "source_file": metadata.get(
            "source_file",
            metadata.get("source"),
        ),
        "document_type": metadata.get("document_type"),
        "metadata": metadata,
    }

    if score is not None:
        result["distance_score"] = float(score)

    return result


# ---------------------------------------------------------
# Policy retrieval
# ---------------------------------------------------------

def retrieve_policy(
    query: str,
    category: Optional[str],
    number_of_results: int = 3,
) -> dict[str, Any]:
    """
    Retrieve relevant banking-policy chunks from ChromaDB.

    Args:
        query:
            The user question or banking issue.

        category:
            Optional metadata category used to restrict the search.

            Supported values:
            - card_dispute
            - fee_refund
            - failed_transfer
            - account_lockout
            - fraud
            - identity_verification
            - general

        number_of_results:
            Maximum number of policy chunks to return.

    Returns:
        A dictionary containing:
        - success
        - query
        - category
        - count
        - results
        - error, when retrieval fails
    """

    if not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "error": "The retrieval query cannot be empty.",
            "results": [],
        }

    if number_of_results < 1:
        return {
            "success": False,
            "error": "number_of_results must be at least 1.",
            "results": [],
        }

    normalized_category = None

    if category is not None:
        normalized_category = category.strip().lower()

        if normalized_category not in ALLOWED_CATEGORIES:
            return {
                "success": False,
                "error": (
                    f"Unsupported policy category: {category}. "
                    f"Allowed categories are: "
                    f"{sorted(ALLOWED_CATEGORIES)}"
                ),
                "results": [],
            }

    try:
        vector_store = load_vector_store()

        metadata_filter = None

        if normalized_category:
            metadata_filter = {
                "category": normalized_category,
            }

        documents_with_scores = (
            vector_store.similarity_search_with_score(
                query=query.strip(),
                k=number_of_results,
                filter=metadata_filter,
            )
        )

        results = [
            format_document(document, score)
            for document, score in documents_with_scores
        ]

        return {
            "success": True,
            "query": query.strip(),
            "category": normalized_category,
            "count": len(results),
            "results": results,
        }

    except (FileNotFoundError, ValueError) as error:
        return {
            "success": False,
            "error": str(error),
            "results": [],
        }

    except Exception as error:
        return {
            "success": False,
            "error": (
                "Policy retrieval failed because of an unexpected error: "
                f"{error}"
            ),
            "results": [],
        }


# ---------------------------------------------------------
# Multi-category retrieval
# ---------------------------------------------------------

def retrieve_policies_by_categories(
    query: str,
    categories: List[str],
    number_of_results_per_category: int = 2,
) -> dict[str, Any]:
    """
    Retrieve policy chunks from multiple metadata categories.

    This is useful when one request requires more than one policy.

    Example:
        An account lockout caused by suspected fraud may need:
        - account_lockout
        - fraud
        - identity_verification
    """

    if not categories:
        return {
            "success": False,
            "error": "At least one policy category is required.",
            "results": [],
        }

    combined_results = []
    errors = []

    for category in categories:
        response = retrieve_policy(
            query=query,
            category=category,
            number_of_results=number_of_results_per_category,
        )

        if response["success"]:
            combined_results.extend(response["results"])
        else:
            errors.append(
                {
                    "category": category,
                    "error": response["error"],
                }
            )

    # Remove duplicate chunks.
    unique_results = []
    seen_chunks = set()

    for result in combined_results:
        duplicate_key = (
            result.get("source_file"),
            result.get("content"),
        )

        if duplicate_key not in seen_chunks:
            seen_chunks.add(duplicate_key)
            unique_results.append(result)

    return {
        "success": len(unique_results) > 0,
        "query": query,
        "categories": categories,
        "count": len(unique_results),
        "results": unique_results,
        "errors": errors,
    }


# ---------------------------------------------------------
# Issue-type category mapping
# ---------------------------------------------------------

def get_policy_categories_for_issue(
    issue_type: str,
) -> list[str]:
    """
    Map a triage issue type to the policy categories that should
    be retrieved.

    This can later be called by your LangGraph retrieval node.
    """

    issue_policy_map = {
        "card_dispute": [
            "card_dispute",
            "identity_verification",
        ],
        "fee_refund": [
            "fee_refund",
            "identity_verification",
        ],
        "failed_transfer": [
            "failed_transfer",
            "identity_verification",
        ],
        "account_lockout": [
            "account_lockout",
            "identity_verification",
        ],
        "suspected_fraud": [
            "fraud",
            "identity_verification",
        ],
        "fraud": [
            "fraud",
            "identity_verification",
        ],
    }

    normalized_issue_type = issue_type.strip().lower()

    return issue_policy_map.get(
        normalized_issue_type,
        ["general"],
    )


def retrieve_policies_for_issue(
    query: str,
    issue_type: str,
    number_of_results_per_category: int = 2,
) -> dict[str, Any]:
    """
    Retrieve the appropriate policies for a classified issue type.

    Example:
        issue_type="suspected_fraud"

    Retrieves:
        fraud policy
        identity-verification policy
    """

    categories = get_policy_categories_for_issue(issue_type)

    return retrieve_policies_by_categories(
        query=query,
        categories=categories,
        number_of_results_per_category=(
            number_of_results_per_category
        ),
    )


# ---------------------------------------------------------
# Debugging helper
# ---------------------------------------------------------

def get_vector_store_status() -> dict[str, Any]:
    """
    Return basic information about the stored ChromaDB collection.

    This is useful for debugging and displaying system status
    in Streamlit.
    """

    try:
        vector_store = load_vector_store()
        collection_count = vector_store._collection.count()

        return {
            "success": True,
            "collection_name": COLLECTION_NAME,
            "persist_directory": str(CHROMA_DIR),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "document_count": collection_count,
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }


# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------

if __name__ == "__main__":
    print("\nVector-store status:")
    print(get_vector_store_status())

    print("\nCard-dispute retrieval:")
    card_result = retrieve_policy(
        query=(
            "The customer says a card transaction was unauthorized. "
            "What information is required and when is human review needed?"
        ),
        category="card_dispute",
        number_of_results=3,
    )
    print(card_result)

    print("\nFraud and identity-policy retrieval:")
    fraud_result = retrieve_policies_for_issue(
        query=(
            "The customer reports multiple unauthorized transactions "
            "and possible account takeover."
        ),
        issue_type="suspected_fraud",
        number_of_results_per_category=2,
    )
    print(fraud_result)