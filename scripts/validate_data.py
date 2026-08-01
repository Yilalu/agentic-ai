from pathlib import Path
import re
import sys

import pandas as pd


# Assumes this file is inside scripts/ and the CSV files are inside data/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CUSTOMER_FILE = DATA_DIR / "customer_data.csv"
ACCOUNT_FILE = DATA_DIR / "account_data.csv"
TRANSACTION_FILE = DATA_DIR / "transaction_data.csv"


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


ALLOWED_ACCOUNT_TYPES = {"Checking", "Savings", "Credit"}
ALLOWED_ACCOUNT_STATUSES = {"Active", "Locked", "Inactive"}
ALLOWED_TRANSACTION_TYPES = {"Deposit", "Withdrawal", "Transfer"}
ALLOWED_TRANSACTION_STATUSES = {"Completed", "Pending", "Failed"}

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def print_result(test_name: str, passed: bool, details: str = "") -> None:
    """Print one validation result."""
    label = "PASS" if passed else "FAIL"
    message = f"[{label}] {test_name}"

    if details:
        message += f": {details}"

    print(message)


def load_csv(file_path: Path, column_names: list[str]) -> pd.DataFrame:
    """
    Load a CSV generated without a header row.

    Raises an error when the file is missing, empty, or has the wrong
    number of columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"File is empty: {file_path}")

    dataframe = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
    )

    if dataframe.shape[1] != len(column_names):
        raise ValueError(
            f"{file_path.name} should contain {len(column_names)} columns, "
            f"but {dataframe.shape[1]} columns were read."
        )

    return dataframe


def validate_required_values(
    dataframe: pd.DataFrame,
    dataframe_name: str,
) -> int:
    """Check for missing values and return the number of errors."""
    missing_count = int(dataframe.isna().sum().sum())

    print_result(
        f"{dataframe_name} has no missing values",
        missing_count == 0,
        f"{missing_count} missing value(s)",
    )

    return missing_count


def validate_unique_id(
    dataframe: pd.DataFrame,
    id_column: str,
    dataframe_name: str,
) -> int:
    """Check that the primary ID column contains unique values."""
    duplicate_count = int(dataframe[id_column].duplicated().sum())

    print_result(
        f"{dataframe_name}.{id_column} is unique",
        duplicate_count == 0,
        f"{duplicate_count} duplicate ID(s)",
    )

    return duplicate_count


def validate_positive_integer_ids(
    dataframe: pd.DataFrame,
    id_columns: list[str],
    dataframe_name: str,
) -> int:
    """Check that ID columns contain positive integers."""
    errors = 0

    for column in id_columns:
        numeric_values = pd.to_numeric(dataframe[column], errors="coerce")

        invalid_mask = (
            numeric_values.isna()
            | (numeric_values <= 0)
            | (numeric_values % 1 != 0)
        )

        invalid_count = int(invalid_mask.sum())
        errors += invalid_count

        print_result(
            f"{dataframe_name}.{column} contains positive integers",
            invalid_count == 0,
            f"{invalid_count} invalid value(s)",
        )

    return errors


def validate_allowed_values(
    dataframe: pd.DataFrame,
    column: str,
    allowed_values: set[str],
    dataframe_name: str,
) -> int:
    """Check categorical values against an allowed set."""
    invalid_mask = ~dataframe[column].isin(allowed_values)
    invalid_count = int(invalid_mask.sum())

    invalid_values = sorted(
        dataframe.loc[invalid_mask, column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    details = f"{invalid_count} invalid value(s)"
    if invalid_values:
        details += f"; found: {invalid_values}"

    print_result(
        f"{dataframe_name}.{column} uses allowed values",
        invalid_count == 0,
        details,
    )

    return invalid_count


def validate_positive_amounts(
    dataframe: pd.DataFrame,
    column: str,
    dataframe_name: str,
) -> int:
    """Check that a monetary column is numeric and greater than zero."""
    numeric_values = pd.to_numeric(dataframe[column], errors="coerce")
    invalid_mask = numeric_values.isna() | (numeric_values <= 0)
    invalid_count = int(invalid_mask.sum())

    print_result(
        f"{dataframe_name}.{column} contains positive numeric values",
        invalid_count == 0,
        f"{invalid_count} invalid value(s)",
    )

    return invalid_count


def validate_customer_data(customers: pd.DataFrame) -> int:
    """Validate customer_data.csv."""
    print("\n--- Customer Data Validation ---")
    errors = 0

    errors += validate_required_values(customers, "customers")
    errors += validate_unique_id(customers, "customer_id", "customers")
    errors += validate_positive_integer_ids(
        customers,
        ["customer_id"],
        "customers",
    )

    invalid_names = (
        customers["customer_name"].astype(str).str.strip().eq("")
    )
    invalid_name_count = int(invalid_names.sum())
    errors += invalid_name_count

    print_result(
        "customers.customer_name is not blank",
        invalid_name_count == 0,
        f"{invalid_name_count} blank name(s)",
    )

    invalid_email_mask = ~customers["customer_email"].astype(str).map(
        lambda email: bool(EMAIL_PATTERN.match(email))
    )
    invalid_email_count = int(invalid_email_mask.sum())
    errors += invalid_email_count

    print_result(
        "customers.customer_email has a valid format",
        invalid_email_count == 0,
        f"{invalid_email_count} invalid email(s)",
    )

    duplicate_email_count = int(
        customers["customer_email"].duplicated().sum()
    )
    errors += duplicate_email_count

    print_result(
        "customers.customer_email is unique",
        duplicate_email_count == 0,
        f"{duplicate_email_count} duplicate email(s)",
    )

    return errors


def validate_account_data(
    accounts: pd.DataFrame,
    customers: pd.DataFrame,
) -> int:
    """Validate account_data.csv and its customer references."""
    print("\n--- Account Data Validation ---")
    errors = 0

    errors += validate_required_values(accounts, "accounts")
    errors += validate_unique_id(accounts, "account_id", "accounts")
    errors += validate_positive_integer_ids(
        accounts,
        ["account_id", "customer_id"],
        "accounts",
    )
    errors += validate_positive_amounts(accounts, "balance", "accounts")

    errors += validate_allowed_values(
        accounts,
        "account_type",
        ALLOWED_ACCOUNT_TYPES,
        "accounts",
    )
    errors += validate_allowed_values(
        accounts,
        "status",
        ALLOWED_ACCOUNT_STATUSES,
        "accounts",
    )

    invalid_customer_mask = ~accounts["customer_id"].isin(
        customers["customer_id"]
    )
    invalid_customer_count = int(invalid_customer_mask.sum())
    errors += invalid_customer_count

    print_result(
        "Every account references an existing customer",
        invalid_customer_count == 0,
        f"{invalid_customer_count} invalid customer reference(s)",
    )

    customers_without_accounts = customers.loc[
        ~customers["customer_id"].isin(accounts["customer_id"]),
        "customer_id",
    ].tolist()

    print_result(
        "Customer-account coverage check",
        True,
        (
            f"{len(customers_without_accounts)} customer(s) have no account. "
            "This is allowed, but review it if every customer should have one."
        ),
    )

    return errors


def validate_transaction_data(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
) -> int:
    """Validate transaction_data.csv and its foreign-key relationships."""
    print("\n--- Transaction Data Validation ---")
    errors = 0

    errors += validate_required_values(transactions, "transactions")
    errors += validate_unique_id(
        transactions,
        "transaction_id",
        "transactions",
    )
    errors += validate_positive_integer_ids(
        transactions,
        ["transaction_id", "customer_id", "account_id"],
        "transactions",
    )
    errors += validate_positive_amounts(
        transactions,
        "amount",
        "transactions",
    )

    errors += validate_allowed_values(
        transactions,
        "transaction_type",
        ALLOWED_TRANSACTION_TYPES,
        "transactions",
    )
    errors += validate_allowed_values(
        transactions,
        "status",
        ALLOWED_TRANSACTION_STATUSES,
        "transactions",
    )

    invalid_customer_mask = ~transactions["customer_id"].isin(
        customers["customer_id"]
    )
    invalid_customer_count = int(invalid_customer_mask.sum())
    errors += invalid_customer_count

    print_result(
        "Every transaction references an existing customer",
        invalid_customer_count == 0,
        f"{invalid_customer_count} invalid customer reference(s)",
    )

    invalid_account_mask = ~transactions["account_id"].isin(
        accounts["account_id"]
    )
    invalid_account_count = int(invalid_account_mask.sum())
    errors += invalid_account_count

    print_result(
        "Every transaction references an existing account",
        invalid_account_count == 0,
        f"{invalid_account_count} invalid account reference(s)",
    )

    # Check that each transaction's account belongs to the same customer.
    account_owner_map = accounts.set_index("account_id")["customer_id"]
    expected_customer_ids = transactions["account_id"].map(account_owner_map)

    ownership_mismatch_mask = (
        expected_customer_ids.notna()
        & (transactions["customer_id"] != expected_customer_ids)
    )
    ownership_mismatch_count = int(ownership_mismatch_mask.sum())
    errors += ownership_mismatch_count

    print_result(
        "Each transaction account belongs to its stated customer",
        ownership_mismatch_count == 0,
        f"{ownership_mismatch_count} ownership mismatch(es)",
    )

    return errors


def main() -> None:
    """Load all files, run validation, and report the final result."""
    print(f"Reading CSV files from: {DATA_DIR}")

    try:
        customers = load_csv(CUSTOMER_FILE, CUSTOMER_COLUMNS)
        accounts = load_csv(ACCOUNT_FILE, ACCOUNT_COLUMNS)
        transactions = load_csv(
            TRANSACTION_FILE,
            TRANSACTION_COLUMNS,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\n[FAIL] Could not load the data: {error}")
        sys.exit(1)

    print("\n--- Row Counts ---")
    print(f"Customers: {len(customers)}")
    print(f"Accounts: {len(accounts)}")
    print(f"Transactions: {len(transactions)}")

    total_errors = 0
    total_errors += validate_customer_data(customers)
    total_errors += validate_account_data(accounts, customers)
    total_errors += validate_transaction_data(
        transactions,
        customers,
        accounts,
    )

    print("\n--- Final Result ---")

    if total_errors == 0:
        print("[PASS] All required validation checks passed.")
        sys.exit(0)

    print(f"[FAIL] Validation found {total_errors} total error(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()