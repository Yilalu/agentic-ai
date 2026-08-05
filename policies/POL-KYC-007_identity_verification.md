---
doc_id: POL-KYC-007
title: Customer Identity Verification Standard
domain: shared
doc_type: policy
department: risk_governance
effective_date: 2025-01-15
requires_human_approval: false
authority_level: support_representative
---

# Customer Identity Verification Standard

## Two-factor standard

Before discussing account details or taking any action, verify the customer with
two independent factors drawn from different groups below. Two factors from the
same group do not satisfy the standard.

**Group A, something the customer knows**
Full account number, recent transaction amount and merchant, last deposit amount,
or the security question answer on file.

**Group B, something the customer has**
One-time passcode delivered to the phone or email already on file, or an approval
tap in the mobile app.

**Group C, something the customer is**
Voice biometric match at or above the 0.85 confidence threshold, or in-branch
government photo identification.

## Information that never satisfies verification

Date of birth, mailing address, mother's maiden name, and the last four digits of
the Social Security number are widely available in breach data. They may be used
as a soft check but never count toward the two-factor requirement.

## Failure handling

After two failed verification attempts in one interaction, do not continue
guiding the caller toward a correct answer. Offer branch verification instead
and note the failed attempt on the profile. Three failed verifications within
seven days place a security hold on the profile.

## Reduced verification cases

A single factor is sufficient for these narrow cases, because the action is
protective and the cost of a false positive is low:

- Blocking a card the caller reports as stolen.
- Placing a temporary hold on the account at the caller's request.
- Accepting a fraud report for later investigation, without disclosing any
  account information back to the caller.

Never disclose balances, transaction history, or contact details on file under
reduced verification. Confirming or denying a detail the caller supplies is a
disclosure.
