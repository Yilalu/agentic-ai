from src.tools import (
    get_customer_accounts,
    get_customer_banking_profile,
    get_customer_fee_requests,
    get_customer_fraud_alerts,
    get_customer_support_cases,
    get_customer_transactions,
    lookup_account,
    lookup_customer,
    lookup_transaction,
)


def main():
    print("\nCustomer:")
    print(lookup_customer.invoke({"customer_id": 1}))

    print("\nAccount:")
    print(lookup_account.invoke({"account_id": 1}))

    print("\nTransaction:")
    print(
        lookup_transaction.invoke(
            {"transaction_id": 1}
        )
    )

    print("\nCustomer accounts:")
    print(
        get_customer_accounts.invoke(
            {"customer_id": 1}
        )
    )

    print("\nCustomer transactions:")
    print(
        get_customer_transactions.invoke(
            {"customer_id": 1}
        )
    )

    print("\nSupport cases:")
    print(
        get_customer_support_cases.invoke(
            {"customer_id": 1}
        )
    )

    print("\nFraud alerts:")
    print(
        get_customer_fraud_alerts.invoke(
            {"customer_id": 1}
        )
    )

    print("\nFee requests:")
    print(
        get_customer_fee_requests.invoke(
            {"customer_id": 1}
        )
    )

    print("\nComplete customer profile:")
    print(
        get_customer_banking_profile.invoke(
            {"customer_id": 1}
        )
    )


if __name__ == "__main__":
    main()