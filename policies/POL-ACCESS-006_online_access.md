---
doc_id: POL-ACCESS-006
title: Online Banking Access, Lockout, and Credential Reset
domain: account
doc_type: procedure
department: digital_banking
effective_date: 2025-05-20
requires_human_approval: false
authority_level: support_representative
---

# Online Banking Access, Lockout, and Credential Reset

## Lockout thresholds

Online banking locks a profile after **five** consecutive failed password
attempts. The lock clears automatically after 30 minutes. A representative may
clear it sooner after identity verification.

Three consecutive failed multi-factor challenges place a **security hold** on the
profile. A security hold does not clear on a timer and must be released by a
representative after two-factor identity verification, because repeated MFA
failure is a common signature of an account takeover attempt.

## Credential reset

Resetting a password or re-enrolling a device is a read-and-write action on the
customer's profile but does not move money. A support representative may perform
it after verification. The reset link expires in 15 minutes and may be sent only
to the contact point already on file, never to a new address supplied during the
call.

If the customer asks to change the email address or phone number on file **and**
reset credentials in the same interaction, stop. That combination is the most
common account takeover pattern. Complete the identity verification, make the
contact change, then impose a **72-hour cooling period** before the credential
reset. The cooling period may be shortened only by digital banking security.

## Dormant and restricted profiles

Profiles unused for 18 months are marked dormant and require re-enrollment with
full identity verification. Profiles restricted by the fraud team cannot be
reset by support at all; route those cases to fraud operations.

## Common non-lockout causes

Before resetting anything, confirm the customer is not hitting one of these,
which reset will not fix:

- Using the business banking portal with retail credentials, or the reverse.
- An expired temporary password, which is valid for 24 hours only.
- Browser autofill supplying an old password, producing failures the customer
  does not realize are happening.
- A card block, which does not affect online access at all and often confuses
  customers into thinking they are locked out.
