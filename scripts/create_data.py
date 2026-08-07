"""Build the local demo bank database used by the tools."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import BANK_DB, DATA_DIR


def setup_bank_db(db_path: str | Path | None = None) -> None:
    path = Path(db_path) if db_path else BANK_DB
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.executescript(
        """
        DROP TABLE IF EXISTS tickets;
        DROP TABLE IF EXISTS fee_waivers;
        DROP TABLE IF EXISTS credit_profiles;
        DROP TABLE IF EXISTS loans;
        DROP TABLE IF EXISTS cards;
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS accounts;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY, full_name TEXT, tier TEXT,
            age INTEGER, since TEXT, fraud_hold INTEGER
        );
        CREATE TABLE accounts (
            account_id TEXT PRIMARY KEY, customer_id TEXT, product TEXT,
            balance REAL, status TEXT
        );
        CREATE TABLE transactions (
            transaction_id TEXT PRIMARY KEY, account_id TEXT, posted_on TEXT,
            descriptor TEXT, amount REAL, channel TEXT, status TEXT
        );
        CREATE TABLE cards (
            card_id TEXT PRIMARY KEY, account_id TEXT, last_four TEXT,
            card_type TEXT, status TEXT, expires_on TEXT
        );
        CREATE TABLE loans (
            loan_id TEXT PRIMARY KEY, customer_id TEXT, loan_type TEXT,
            balance REAL, rate REAL, monthly_payment REAL, status TEXT,
            days_past_due INTEGER
        );
        CREATE TABLE fee_waivers (
            waiver_id TEXT PRIMARY KEY, customer_id TEXT, amount REAL,
            waived_on TEXT, reason TEXT
        );
        CREATE TABLE credit_profiles (
            customer_id TEXT PRIMARY KEY, credit_score INTEGER,
            annual_income REAL, monthly_debt REAL, employment_months INTEGER,
            delinquencies_24m INTEGER
        );
        CREATE TABLE tickets (
            ticket_id TEXT PRIMARY KEY, session_id TEXT, customer_id TEXT,
            domain TEXT, status TEXT, reason TEXT, queue TEXT, created_at TEXT
        );
        """
    )

    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?)",
        [
            ("CUST-001", "Alex Rivera", "standard", 34, "2019-03-12", 0),
            ("CUST-002", "Jordan Lee", "premier", 41, "2015-08-01", 0),
            ("CUST-003", "Sam Patel", "standard", 29, "2021-11-20", 0),
            ("CUST-004", "Casey Nguyen", "standard", 52, "2012-05-04", 0),
            ("CUST-005", "Riley Brooks", "standard", 68, "2008-01-15", 1),
            ("CUST-006", "Morgan Blake", "standard", 38, "2020-06-01", 0),
        ],
    )
    conn.executemany(
        "INSERT INTO accounts VALUES (?,?,?,?,?)",
        [
            ("ACCT-001", "CUST-001", "checking", 1842.55, "active"),
            ("ACCT-002", "CUST-002", "checking", 9200.00, "active"),
            ("ACCT-003", "CUST-003", "savings", 450.10, "active"),
            ("ACCT-004", "CUST-004", "checking", 312.77, "active"),
            ("ACCT-005", "CUST-005", "checking", 1105.00, "active"),
            ("ACCT-006", "CUST-006", "checking", 640.00, "locked"),
        ],
    )
    conn.executemany(
        "INSERT INTO transactions VALUES (?,?,?,?,?,?,?)",
        [
            # Happy path / revision: duplicate Summit Outdoors charge
            ("TXN-1001", "ACCT-001", "2026-07-29", "Summit Outdoors", -42.00, "card", "posted"),
            ("TXN-1002", "ACCT-001", "2026-07-29", "Summit Outdoors", -42.00, "card", "posted"),
            ("TXN-1003", "ACCT-001", "2026-07-28", "City Transit", -3.50, "card", "posted"),
            # Fraud scenario: unauthorized Northgate charge on CUST-001
            ("TXN-1101", "ACCT-001", "2026-07-30", "Northgate Electronics", -812.44, "card", "posted"),
            # Branching path: six overdraft fees in one day (daily cap is three)
            ("TXN-2001", "ACCT-004", "2026-07-30", "Overdraft Fee", -35.00, "fee", "posted"),
            ("TXN-2002", "ACCT-004", "2026-07-30", "Overdraft Fee", -35.00, "fee", "posted"),
            ("TXN-2003", "ACCT-004", "2026-07-30", "Overdraft Fee", -35.00, "fee", "posted"),
            ("TXN-2004", "ACCT-004", "2026-07-30", "Overdraft Fee", -35.00, "fee", "posted"),
            ("TXN-2005", "ACCT-004", "2026-07-30", "Overdraft Fee", -35.00, "fee", "posted"),
            ("TXN-2006", "ACCT-004", "2026-07-30", "Overdraft Fee", -35.00, "fee", "posted"),
            ("TXN-3001", "ACCT-005", "2026-07-31", "Unknown Merchant XYZ", -250.00, "card", "posted"),
            ("TXN-2007", "ACCT-002", "2026-07-20", "Payroll Deposit", 4200.00, "ach", "posted"),
        ],
    )
    conn.executemany(
        "INSERT INTO cards VALUES (?,?,?,?,?,?)",
        [
            ("CARD-001", "ACCT-001", "4421", "debit", "active", "2028-04"),
            ("CARD-002", "ACCT-004", "7788", "debit", "active", "2027-11"),
            ("CARD-003", "ACCT-005", "1190", "debit", "active", "2026-12"),
            ("CARD-004", "ACCT-006", "3344", "debit", "active", "2027-08"),
        ],
    )
    conn.executemany(
        "INSERT INTO loans VALUES (?,?,?,?,?,?,?,?)",
        [("LOAN-001", "CUST-002", "auto", 9400.00, 5.9, 285.00, "current", 0)],
    )
    conn.executemany(
        "INSERT INTO fee_waivers VALUES (?,?,?,?,?)",
        [("FW-001", "CUST-004", 35.00, "2026-03-01", "courtesy waiver")],
    )
    conn.executemany(
        "INSERT INTO credit_profiles VALUES (?,?,?,?,?,?)",
        [
            ("CUST-001", 710, 62000, 900, 36, 0),
            ("CUST-002", 780, 98000, 700, 84, 0),
            ("CUST-003", 640, 41000, 1200, 10, 0),
            ("CUST-004", 620, 38000, 1600, 48, 1),
            ("CUST-005", 700, 54000, 800, 120, 0),
            ("CUST-006", 690, 48000, 950, 24, 0),
        ],
    )
    conn.commit()
    conn.close()
    print(f"Bank demo DB ready: {path}")


if __name__ == "__main__":
    setup_bank_db()
