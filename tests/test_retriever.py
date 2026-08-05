"""Quick checks that the Chroma policy index loads and retrieves."""

from src.retriever import collection_size, get_vector_store_status, retrieve


def main() -> None:
    print("\nVector database:")
    print(get_vector_store_status())
    print("\nChunks:", collection_size())
    print("\nCard retrieval:")
    for source in retrieve("duplicate card charge refund", domain="card", k=2):
        print(source.doc_id, source.title)


if __name__ == "__main__":
    main()
