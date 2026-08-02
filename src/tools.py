from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from .config import CUSTOMER_DATA, ACCOUNT_DATA, TRANSACTIONS_DATA, SUPPORT_CASES_DATA, FRAUD_ALERTS_DATA, FEE_REQUESTS_DATA

# ---------------------------------------------------------
# Column definitions
# ---------------------------------------------------------

CUSTOMER_COLUMNS = [
    "customer_id",
    "customer_name",
    "customer_email",
]

ACCOUNT_COLUMNS = [
    "account_id",
    "customer_id",
    "account_type",
    "balance",
    "status",
]

TRANSACTION_COLUMNS = [
    "transaction_id",
    "customer_id",
    "account_id",
    "transaction_type",
    "amount",
    "status",
]

SUPPORT_CASE_COLUMNS = [
    "case_id",
    "customer_id",
    "transaction_id",
    "issue_type",
    "status",
    "assigned_team",
]

FRAUD_ALERT_COLUMNS = [
    "alert_id",
    "customer_id",
    "transaction_id",
    "risk_level",
    "reason",
    "status",
]

FEE_REQUEST_COLUMNS = [
    "request_id",
    "customer_id",
    "transaction_id",
    "fee_type",
    "amount",
    "status",
]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def load_csv(file_path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Load a CSV file that does not contain a header row.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file was not found: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {file_path}")

    try:
        dataframe = pd.read_csv(
            file_path,
            header=None,
            names=columns,
        )
    except pd.errors.ParserError as error:
        raise ValueError(
            f"Could not read {file_path.name}: {error}"
        ) from error

    return dataframe


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Convert pandas and NumPy values into regular Python values.

    This makes the result easier to store in LangGraph state
    and serialize as JSON.
    """

    cleaned_record = {}

    for key, value in record.items():
        if pd.isna(value):
            cleaned_record[key] = None
        elif hasattr(value, "item"):
            cleaned_record[key] = value.item()
        else:
            cleaned_record[key] = value

    return cleaned_record


def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame into JSON-friendly dictionaries.
    """

    return [
        convert_record(record)
        for record in dataframe.to_dict(orient="records")
    ]


def success_response(**kwargs: Any) -> dict[str, Any]:
    """
    Create a successful tool response.
    """

    return {
        "success": True,
        **kwargs,
    }


def error_response(message: str) -> dict[str, Any]:
    """
    Create a failed tool response.
    """

    return {
        "success": False,
        "error": message,
    }


# ---------------------------------------------------------
# Customer tools
# ---------------------------------------------------------

@tool
def lookup_customer(customer_id: int) -> dict[str, Any]:
    """
    Retrieve one customer using a customer ID.
    """

    try:
        customers = load_csv(
            CUSTOMER_DATA,
            CUSTOMER_COLUMNS,
        )

        result = customers[
            customers["customer_id"] == customer_id
        ]

        if result.empty:
            return error_response(
                f"Customer {customer_id} was not found."
            )

        customer = convert_record(
            result.iloc[0].to_dict()
        )

        return success_response(
            customer=customer
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def search_customer_by_email(
    customer_email: str,
) -> dict[str, Any]:
    """
    Retrieve one customer using an email address.
    """

    try:
        customers = load_csv(
            CUSTOMER_DATA,
            CUSTOMER_COLUMNS,
        )

        normalized_email = customer_email.strip().lower()

        result = customers[
            customers["customer_email"]
            .astype(str)
            .str.strip()
            .str.lower()
            == normalized_email
        ]

        if result.empty:
            return error_response(
                f"No customer was found with email "
                f"{customer_email}."
            )

        customer = convert_record(
            result.iloc[0].to_dict()
        )

        return success_response(
            customer=customer
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


# ---------------------------------------------------------
# Account tools
# ---------------------------------------------------------

@tool
def lookup_account(account_id: int) -> dict[str, Any]:
    """
    Retrieve one bank account using an account ID.
    """

    try:
        accounts = load_csv(
            ACCOUNT_DATA,
            ACCOUNT_COLUMNS,
        )

        result = accounts[
            accounts["account_id"] == account_id
        ]

        if result.empty:
            return error_response(
                f"Account {account_id} was not found."
            )

        account = convert_record(
            result.iloc[0].to_dict()
        )

        return success_response(
            account=account
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_customer_accounts(
    customer_id: int,
) -> dict[str, Any]:
    """
    Retrieve all accounts belonging to a customer.
    """

    try:
        accounts = load_csv(
            ACCOUNT_DATA,
            ACCOUNT_COLUMNS,
        )

        result = accounts[
            accounts["customer_id"] == customer_id
        ]

        if result.empty:
            return error_response(
                f"No accounts were found for customer "
                f"{customer_id}."
            )

        records = dataframe_to_records(result)

        return success_response(
            customer_id=customer_id,
            count=len(records),
            accounts=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


# ---------------------------------------------------------
# Transaction tools
# ---------------------------------------------------------

@tool
def lookup_transaction(
    transaction_id: int,
) -> dict[str, Any]:
    """
    Retrieve one transaction using a transaction ID.
    """

    try:
        transactions = load_csv(
            TRANSACTIONS_DATA,
            TRANSACTION_COLUMNS,
        )

        result = transactions[
            transactions["transaction_id"]
            == transaction_id
        ]

        if result.empty:
            return error_response(
                f"Transaction {transaction_id} "
                f"was not found."
            )

        transaction = convert_record(
            result.iloc[0].to_dict()
        )

        return success_response(
            transaction=transaction
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_customer_transactions(
    customer_id: int,
) -> dict[str, Any]:
    """
    Retrieve all transactions belonging to a customer.
    """

    try:
        transactions = load_csv(
            TRANSACTIONS_DATA,
            TRANSACTION_COLUMNS,
        )

        result = transactions[
            transactions["customer_id"] == customer_id
        ]

        if result.empty:
            return error_response(
                f"No transactions were found for "
                f"customer {customer_id}."
            )

        records = dataframe_to_records(result)

        return success_response(
            customer_id=customer_id,
            count=len(records),
            transactions=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_account_transactions(
    account_id: int,
) -> dict[str, Any]:
    """
    Retrieve all transactions belonging to an account.
    """

    try:
        transactions = load_csv(
            TRANSACTIONS_DATA,
            TRANSACTION_COLUMNS,
        )

        result = transactions[
            transactions["account_id"] == account_id
        ]

        if result.empty:
            return error_response(
                f"No transactions were found for "
                f"account {account_id}."
            )

        records = dataframe_to_records(result)

        return success_response(
            account_id=account_id,
            count=len(records),
            transactions=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


# ---------------------------------------------------------
# Support-case tools
# ---------------------------------------------------------

@tool
def lookup_support_case(
    case_id: int,
) -> dict[str, Any]:
    """
    Retrieve one support case using a case ID.
    """

    try:
        cases = load_csv(
            SUPPORT_CASES_DATA,
            SUPPORT_CASE_COLUMNS,
        )

        result = cases[
            cases["case_id"] == case_id
        ]

        if result.empty:
            return error_response(
                f"Support case {case_id} was not found."
            )

        support_case = convert_record(
            result.iloc[0].to_dict()
        )

        return success_response(
            support_case=support_case
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_customer_support_cases(
    customer_id: int,
) -> dict[str, Any]:
    """
    Retrieve all support cases belonging to a customer.
    """

    try:
        cases = load_csv(
            SUPPORT_CASES_DATA,
            SUPPORT_CASE_COLUMNS,
        )

        result = cases[
            cases["customer_id"] == customer_id
        ]

        if result.empty:
            return success_response(
                customer_id=customer_id,
                count=0,
                support_cases=[],
            )

        records = dataframe_to_records(result)

        return success_response(
            customer_id=customer_id,
            count=len(records),
            support_cases=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_transaction_support_cases(
    transaction_id: int,
) -> dict[str, Any]:
    """
    Retrieve support cases connected to a transaction.
    """

    try:
        cases = load_csv(
            SUPPORT_CASES_DATA,
            SUPPORT_CASE_COLUMNS,
        )

        result = cases[
            cases["transaction_id"] == transaction_id
        ]

        records = dataframe_to_records(result)

        return success_response(
            transaction_id=transaction_id,
            count=len(records),
            support_cases=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


# ---------------------------------------------------------
# Fraud-alert tools
# ---------------------------------------------------------

@tool
def lookup_fraud_alert(
    alert_id: int,
) -> dict[str, Any]:
    """
    Retrieve one fraud alert using an alert ID.
    """

    try:
        alerts = load_csv(
            FRAUD_ALERTS_DATA,
            FRAUD_ALERT_COLUMNS,
        )

        result = alerts[
            alerts["alert_id"] == alert_id
        ]

        if result.empty:
            return error_response(
                f"Fraud alert {alert_id} was not found."
            )

        fraud_alert = convert_record(
            result.iloc[0].to_dict()
        )

        return success_response(
            fraud_alert=fraud_alert
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_customer_fraud_alerts(
    customer_id: int,
) -> dict[str, Any]:
    """
    Retrieve all fraud alerts belonging to a customer.
    """

    try:
        alerts = load_csv(
            FRAUD_ALERTS_DATA,
            FRAUD_ALERT_COLUMNS,
        )

        result = alerts[
            alerts["customer_id"] == customer_id
        ]

        records = dataframe_to_records(result)

        active_alerts = result[
            result["status"].isin(
                ["Open", "Investigating"]
            )
        ]

        high_risk_alerts = result[
            result["risk_level"] == "High"
        ]

        return success_response(
            customer_id=customer_id,
            count=len(records),
            active_alert_count=len(active_alerts),
            high_risk_alert_count=len(high_risk_alerts),
            fraud_alerts=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_transaction_fraud_alerts(
    transaction_id: int,
) -> dict[str, Any]:
    """
    Retrieve fraud alerts connected to a transaction.
    """

    try:
        alerts = load_csv(
            FRAUD_ALERTS_DATA,
            FRAUD_ALERT_COLUMNS,
        )

        result = alerts[
            alerts["transaction_id"]
            == transaction_id
        ]

        records = dataframe_to_records(result)

        return success_response(
            transaction_id=transaction_id,
            count=len(records),
            fraud_alerts=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


# ---------------------------------------------------------
# Fee-request tools
# ---------------------------------------------------------

@tool
def lookup_fee_request(
    request_id: int,
) -> dict[str, Any]:
    """
    Retrieve one fee refund request using a request ID.
    """

    try:
        requests = load_csv(
            FEE_REQUESTS_DATA,
            FEE_REQUEST_COLUMNS,
        )

        result = requests[
            requests["request_id"] == request_id
        ]

        if result.empty:
            return error_response(
                f"Fee request {request_id} was not found."
            )

        fee_request = convert_record(
            result.iloc[0].to_dict()
        )

        return success_response(
            fee_request=fee_request
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_customer_fee_requests(
    customer_id: int,
) -> dict[str, Any]:
    """
    Retrieve all fee requests belonging to a customer.
    """

    try:
        requests = load_csv(
            FEE_REQUESTS_DATA,
            FEE_REQUEST_COLUMNS,
        )

        result = requests[
            requests["customer_id"] == customer_id
        ]

        records = dataframe_to_records(result)

        approved_requests = result[
            result["status"] == "Approved"
        ]

        pending_requests = result[
            result["status"] == "Pending"
        ]

        return success_response(
            customer_id=customer_id,
            count=len(records),
            approved_count=len(approved_requests),
            pending_count=len(pending_requests),
            fee_requests=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


@tool
def get_transaction_fee_requests(
    transaction_id: int,
) -> dict[str, Any]:
    """
    Retrieve fee requests connected to a transaction.
    """

    try:
        requests = load_csv(
            FEE_REQUESTS_DATA,
            FEE_REQUEST_COLUMNS,
        )

        result = requests[
            requests["transaction_id"]
            == transaction_id
        ]

        records = dataframe_to_records(result)

        return success_response(
            transaction_id=transaction_id,
            count=len(records),
            fee_requests=records,
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


# ---------------------------------------------------------
# Combined banking record tool
# ---------------------------------------------------------

@tool
def get_customer_banking_profile(
    customer_id: int,
) -> dict[str, Any]:
    """
    Retrieve a combined banking profile for a customer.

    This includes customer information, accounts, transactions,
    support cases, fraud alerts, and fee requests.
    """

    try:
        customers = load_csv(
            CUSTOMER_DATA,
            CUSTOMER_COLUMNS,
        )
        accounts = load_csv(
            ACCOUNT_DATA,
            ACCOUNT_COLUMNS,
        )
        transactions = load_csv(
            TRANSACTIONS_DATA,
            TRANSACTION_COLUMNS,
        )
        support_cases = load_csv(
            SUPPORT_CASES_DATA,
            SUPPORT_CASE_COLUMNS,
        )
        fraud_alerts = load_csv(
            FRAUD_ALERTS_DATA,
            FRAUD_ALERT_COLUMNS,
        )
        fee_requests = load_csv(
            FEE_REQUESTS_DATA,
            FEE_REQUEST_COLUMNS,
        )

        customer_result = customers[
            customers["customer_id"] == customer_id
        ]

        if customer_result.empty:
            return error_response(
                f"Customer {customer_id} was not found."
            )

        customer = convert_record(
            customer_result.iloc[0].to_dict()
        )

        customer_accounts = accounts[
            accounts["customer_id"] == customer_id
        ]

        customer_transactions = transactions[
            transactions["customer_id"] == customer_id
        ]

        customer_cases = support_cases[
            support_cases["customer_id"] == customer_id
        ]

        customer_alerts = fraud_alerts[
            fraud_alerts["customer_id"] == customer_id
        ]

        customer_fee_requests = fee_requests[
            fee_requests["customer_id"] == customer_id
        ]

        active_fraud_alerts = customer_alerts[
            customer_alerts["status"].isin(
                ["Open", "Investigating"]
            )
        ]

        return success_response(
            customer=customer,
            accounts=dataframe_to_records(
                customer_accounts
            ),
            transactions=dataframe_to_records(
                customer_transactions
            ),
            support_cases=dataframe_to_records(
                customer_cases
            ),
            fraud_alerts=dataframe_to_records(
                customer_alerts
            ),
            fee_requests=dataframe_to_records(
                customer_fee_requests
            ),
            summary={
                "account_count": len(customer_accounts),
                "transaction_count": len(
                    customer_transactions
                ),
                "support_case_count": len(
                    customer_cases
                ),
                "fraud_alert_count": len(
                    customer_alerts
                ),
                "active_fraud_alert_count": len(
                    active_fraud_alerts
                ),
                "fee_request_count": len(
                    customer_fee_requests
                ),
            },
        )

    except (FileNotFoundError, ValueError) as error:
        return error_response(str(error))


# ---------------------------------------------------------
# Tool list for agents
# ---------------------------------------------------------

BANKING_TOOLS = [
    lookup_customer,
    search_customer_by_email,
    lookup_account,
    get_customer_accounts,
    lookup_transaction,
    get_customer_transactions,
    get_account_transactions,
    lookup_support_case,
    get_customer_support_cases,
    get_transaction_support_cases,
    lookup_fraud_alert,
    get_customer_fraud_alerts,
    get_transaction_fraud_alerts,
    lookup_fee_request,
    get_customer_fee_requests,
    get_transaction_fee_requests,
    get_customer_banking_profile,
]