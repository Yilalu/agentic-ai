---
doc_id: POL-AUTH-008
title: Escalation and Approval Authority Matrix
domain: shared
doc_type: policy
department: risk_governance
effective_date: 2025-07-01
requires_human_approval: true
authority_level: risk_governance
---

# Escalation and Approval Authority Matrix

## Purpose

This matrix defines which actions an automated assistant acting in a support
capacity may complete alone and which require a named human approver. It is the
controlling document when another policy is silent or when two policies
conflict.

## The automated approval limit

An automated assistant may complete a refund **only** when all of the following
hold:

1. The refund amount is **$50.00 or less**.
2. The request is not a loan matter.
3. The request is not a fraud matter.
4. The action is supported by a citation to a retrieved policy document.

A refund of exactly $50.00 is within the limit. A refund of $50.01 is not.
Everything outside those four conditions is a recommendation that stops for a
human decision.

## Actions that never execute without a named human approver

| Action | Minimum approver | Reason |
| --- | --- | --- |
| Any refund above $50.00 | Support Supervisor | Above the automated limit |
| **Any loan decision, approve or decline, at any amount** | Loan Officer | Credit decisions are never automated |
| Provisional credit, any amount | Fraud Analyst II | Moves funds before investigation |
| Goodwill scam reimbursement, any amount | Fraud Manager | Discretionary loss |
| Wire recall submission | Payment Operations Officer | Creates indemnity obligation |
| Account closure | Deposit Operations Manager | Irreversible for the customer |
| Contact-detail change plus credential reset | Digital Banking Security | Account takeover signature |
| Any action on an account with a fraud hold | Fraud Operations | Hold exists for a reason |

Loan decisions deserve emphasis. A recommendation to approve or decline credit
is never automated regardless of amount, regardless of how confident the
assessment is, and regardless of whether the recommendation is favourable to the
customer. A declined application carries fair-lending and adverse-action notice
obligations that only a loan officer may discharge.

An assistant may **recommend** any of these actions with a full justification and
policy citation. It may not record them as done. Interfaces must describe such
recommendations as awaiting approval and must not tell the customer the action
has occurred.

## Actions pre-authorized for the automated assistant

- Refunds of $50.00 or less on card and deposit accounts.
- Card block and standard-mail reissue.
- Password reset and MFA re-enrollment after two-factor verification.
- Opening a dispute record.
- Any information-only response.

## Mandatory escalation triggers

Escalate regardless of the action requested when any of these is true:

- The customer is 65 or older and reports financial exploitation.
- Aggregate loss or disputed amount is $5,000 or more.
- The customer states or implies an intent to pursue legal action or has
  contacted a regulator, an attorney, or the media.
- The same issue has been reported three or more times without resolution.
- The account carries a fraud, legal, or estate hold.
- A death, incapacity, bankruptcy, or power-of-attorney matter is disclosed.
- The customer requests a written response from a compliance officer.

## Documentation standard

Every escalation must record the trigger that fired, the evidence gathered, the
recommended action, and the policy citation supporting it. An escalation without
a citation is returned to the originator for rework.
