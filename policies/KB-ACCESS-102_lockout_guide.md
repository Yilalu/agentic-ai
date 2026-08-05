---
doc_id: KB-ACCESS-102
title: Online Banking Lockout Troubleshooting
domain: account
doc_type: kb_article
department: digital_banking
effective_date: 2025-05-25
requires_human_approval: false
authority_level: support_representative
---

# Online Banking Lockout Troubleshooting

## Identify which lock the customer has hit

The error message tells you which state the profile is in, and each has a
different remedy.

| Message the customer sees | State | Remedy |
| --- | --- | --- |
| "Your password is incorrect" | Not locked yet | Password reset |
| "Your profile is temporarily locked" | Password lockout | Clears in 30 minutes, or clear it after verification |
| "We could not verify it's you" | MFA failure | Three failures create a security hold |
| "Contact us to continue" | Security hold | Release after two-factor verification |
| "This profile is restricted" | Fraud restriction | Do not release; route to fraud operations |

## Multi-factor delivery problems

Most MFA failures are delivery problems rather than wrong codes. Work through
these in order.

1. Confirm the phone number on file is the one the customer holds now. A number
   ported to a new carrier can silently stop receiving short-code messages.
2. Check whether the customer is abroad. Some carriers do not deliver
   short-code SMS internationally; switch the customer to email or app approval.
3. Confirm the customer is entering the most recent code. Codes expire in five
   minutes and customers often read an older message.
4. Check the device clock on authenticator-app setups. A clock skew over 60
   seconds invalidates every generated code.

## When reset is the wrong answer

Do not reset credentials when the customer is also asking to change the phone
number or email on file in the same interaction. That combination requires the
72-hour cooling period described in POL-ACCESS-006. Explain the delay as a
protection rather than an obstacle; customers accept it well when told it exists
to stop takeovers.

## Business versus retail portals

Business banking uses a separate portal and separate credentials. A business
customer trying retail credentials will see a generic failure message and will
insist their password is correct, because it is, on the other portal. Confirm
which product the customer holds before troubleshooting anything.
