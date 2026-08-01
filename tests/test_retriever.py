from src.retriever import (
    get_policy_categories_for_issue,
    get_vector_store_status,
    retrieve_policies_for_issue,
    retrieve_policy,
)


def main() -> None:
    print("\nVector database:")
    print(get_vector_store_status())

    print("\nCard-dispute policy:")
    print(
        retrieve_policy(
            query="Can an unauthorized card transaction be disputed?",
            category="card_dispute",
            number_of_results=2,
        )
    )

    print("\nFraud policies:")
    print(
        retrieve_policies_for_issue(
            query=(
                "Customer reports multiple transactions "
                "that they did not authorize."
            ),
            issue_type="suspected_fraud",
        )
    )

    print("\nMapped categories:")
    print(
        get_policy_categories_for_issue(
            "account_lockout"
        )
    )


if __name__ == "__main__":
    main()