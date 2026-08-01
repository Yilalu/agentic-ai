import random as random

def create_customer_data(filename, number_of_rows):
    with open(filename, 'w') as f:
        for n in range(number_of_rows):
            #generate customer data
            customer_id = n + 1
            customer_name = f"Customer {customer_id}"
            customer_email = f"customer{customer_id}@example.com"
            
            f.write(f"{customer_id},{customer_name},{customer_email}\n")





def create_account_data(filename, number_of_rows, number_of_customers):
    with open(filename, 'w') as f:
        for n in range(number_of_rows):
            #generate account data
            account_id = n + 1
            customer_id = random.randint(1, number_of_customers)
            account_type = random.choice(['Checking', 'Savings', 'Credit'])
            balance = round(random.uniform(100.0, 10000.0), 2)
            status = random.choice(['Active', 'Locked', 'Inactive'])
            f.write(f"{account_id},{customer_id},{account_type},{balance},{status}\n")

def create_transaction_data(filename, account_filename, number_of_rows, number_of_customers):
    customer_accounts = {}
    with open(account_filename, 'r') as af:
        for line in af:
            parts = line.strip().split(',')
            account_id = int(parts[0])
            customer_id = int(parts[1])
            if customer_id not in customer_accounts:
                customer_accounts[customer_id] = []
            customer_accounts[customer_id].append(account_id)

    with open(filename, 'w') as f:
        for n in range(number_of_rows):
            #generate transaction data

            transaction_id = n + 1
            account_id = -1
            while account_id == -1:
                customer_id = random.randint(1, number_of_customers)
                if customer_id in customer_accounts:
                    account_id = random.choice(customer_accounts[customer_id])
            transaction_type = random.choice(['Deposit', 'Withdrawal', 'Transfer'])
            amount = round(random.uniform(10.0, 1000.0), 2)
            status = random.choice(['Completed', 'Pending', 'Failed'])
        
            f.write(f"{transaction_id},{customer_id},{account_id},{transaction_type},{amount},{status}\n")


def create_support_cases_data(filename, transaction_filename, number_of_rows):
    import random

    transactions = []

    with open(transaction_filename, "r") as tf:
        for line in tf:
            parts = line.strip().split(",")

            transactions.append({
                "customer_id": int(parts[1]),
                "transaction_id": int(parts[0])
            })

    issue_types = [
        "Card Dispute",
        "Failed Transfer",
        "Fee Refund",
        "Account Lockout",
        "Fraud Investigation"
    ]

    statuses = [
        "Open",
        "Pending",
        "Resolved",
        "Escalated",
        "Closed"
    ]

    assigned_teams = [
        "Customer Support",
        "Fraud Team",
        "Account Services",
        "Operations"
    ]

    with open(filename, "w") as f:
        for n in range(number_of_rows):

            transaction = random.choice(transactions)

            case_id = n + 1
            customer_id = transaction["customer_id"]
            transaction_id = transaction["transaction_id"]

            issue_type = random.choice(issue_types)
            status = random.choice(statuses)
            assigned_team = random.choice(assigned_teams)

            f.write(
                f"{case_id},"
                f"{customer_id},"
                f"{transaction_id},"
                f"{issue_type},"
                f"{status},"
                f"{assigned_team}\n"
            )

def create_fraud_alerts_data(filename, transaction_filename, number_of_rows):
    import random

    transactions = []

    with open(transaction_filename, "r") as tf:
        for line in tf:
            parts = line.strip().split(",")

            transactions.append({
                "transaction_id": int(parts[0]),
                "customer_id": int(parts[1])
            })

    risk_levels = [
        "Low",
        "Medium",
        "High"
    ]

    reasons = [
        "Large Purchase",
        "Foreign Transaction",
        "Multiple Failed Logins",
        "Unusual Spending Pattern",
        "Multiple Transactions"
    ]

    statuses = [
        "Open",
        "Investigating",
        "Resolved",
        "Closed"
    ]

    with open(filename, "w") as f:

        for n in range(number_of_rows):

            transaction = random.choice(transactions)

            alert_id = n + 1
            customer_id = transaction["customer_id"]
            transaction_id = transaction["transaction_id"]

            risk_level = random.choice(risk_levels)
            reason = random.choice(reasons)
            status = random.choice(statuses)

            f.write(
                f"{alert_id},"
                f"{customer_id},"
                f"{transaction_id},"
                f"{risk_level},"
                f"{reason},"
                f"{status}\n"
            )

def create_fee_requests_data(filename, transaction_filename, number_of_rows):
    import random

    transactions = []

    with open(transaction_filename, "r") as tf:
        for line in tf:
            parts = line.strip().split(",")

            transactions.append({
                "transaction_id": int(parts[0]),
                "customer_id": int(parts[1])
            })

    fee_types = [
        "ATM Fee",
        "Overdraft Fee",
        "Transfer Fee",
        "Maintenance Fee"
    ]

    statuses = [
        "Pending",
        "Approved",
        "Rejected"
    ]

    with open(filename, "w") as f:
        for n in range(number_of_rows):
            transaction = random.choice(transactions)

            request_id = n + 1
            customer_id = transaction["customer_id"]
            transaction_id = transaction["transaction_id"]

            fee_type = random.choice(fee_types)
            amount = round(random.uniform(5, 75), 2)
            status = random.choice(statuses)

            f.write(
                f"{request_id},"
                f"{customer_id},"
                f"{transaction_id},"
                f"{fee_type},"
                f"{amount},"
                f"{status}\n"
            )
            
if __name__ == "__main__":
    create_customer_data("../data/customer_data.csv", 100)
    create_account_data("../data/account_data.csv", 200, 100)
    create_transaction_data("../data/transaction_data.csv", "../data/account_data.csv", 2000, 100)
    create_support_cases_data("../data/support_cases_data.csv", "../data/transaction_data.csv", 300)
    create_fraud_alerts_data("../data/fraud_alerts_data.csv", "../data/transaction_data.csv", 150)
    create_fee_requests_data("../data/fee_requests_data.csv", "../data/transaction_data.csv", 200)

