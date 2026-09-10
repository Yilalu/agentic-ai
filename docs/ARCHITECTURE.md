# Architecture

This document goes deeper than the [README](../README.md) into how the
graph, agents, schemas, and data stores fit together. Read it if you're
extending the workflow, adding a new domain, or debugging a routing
decision.

## Contents

- [Design principles](#design-principles)
- [The graph](#the-graph)
- [Shared state](#shared-state)
- [Schemas](#schemas)
- [Agents](#agents)
- [Routing rules](#routing-rules)
- [Tools](#tools)
- [Retrieval (RAG)](#retrieval-rag)
- [Data stores](#data-stores)
- [Failure handling](#failure-handling)
- [Extending the system](#extending-the-system)

## Design principles

1. **Roles never call each other.** A specialist doesn't call the critic; a
   node calls the specialist, returns state, and a separate node calls the
   critic. This keeps every hop visible in the trace and makes each role
   independently testable.
2. **Application code owns the thresholds; the model owns the drafting.**
   Whether an action needs human approval (`$50` refund cap, "loans always
   need an officer") is a function in `src/graph/routes.py`, not a prompt
   instruction. The model can be wrong or inconsistent about a threshold;
   code can't.
3. **Grounding is checkable, so check it deterministically first.** The
   critic's citation guard is plain Python running before any LLM call. Only
   claims that survive a fact-checkable test (a cited doc ID must be among
   the retrieved chunks) get spent on a model's judgment.
4. **Every failure mode degrades to "ask a human," never to "guess."** No
   API key, a failed SQL query, an empty vector store, a chat model outage —
   all of these produce a conservative fallback (keyword triage, a holding
   reply, an escalate verdict), never a fabricated answer.
5. **Nothing here touches a real banking system.** Every write
   (`process_refund`, `request_human_approval`, `create_escalation`,
   `record_approval_decision`) is explicitly simulated and labeled as such
   in its return value.

## The graph

Built in [src/graph/build.py](../src/graph/build.py) with LangGraph's
`StateGraph`. Nodes and edges:

```
START → intake → triage
triage --[route_after_triage]--> ask_user | card_agent | loan_agent
                                | account_agent | fraud_agent | out_of_scope
                                | escalated
ask_user → wait_for_user → triage                (clarification loop)
card_agent | loan_agent | account_agent → critic
fraud_agent → escalated                          (no critic — nothing to validate)
out_of_scope --[route_after_out_of_scope]--> escalated | END
critic --[route_after_critic]--> card_agent | loan_agent | account_agent
                                | resolved | human_approval | escalated
resolved | human_approval | escalated → END
```

- **Checkpointing**: `MemorySaver` with a custom `JsonPlusSerializer` that
  knows how to (de)serialize the project's Pydantic schemas
  (`CHECKPOINTED_TYPES` in `build.py`). Each Streamlit `thread_id` is a
  LangGraph thread; the checkpointer is what lets `resume_turn` continue a
  paused graph exactly where `interrupt()` suspended it.
- **Pause / resume**: `wait_for_user_node` calls LangGraph's `interrupt()`,
  which suspends the graph and returns control to the caller with an
  `__interrupt__` payload. `app.py` reads that payload via
  `pending_question()`, shows it as a question, and the next user message is
  sent back in via `resume_turn()` → `Command(resume=answer)`.
- **Public entry points** (all in `src/graph/build.py`):
  - `run_turn(message, thread_id)` — start or continue a thread with a new
    customer message.
  - `resume_turn(answer, thread_id)` — continue a thread paused at
    `wait_for_user`.
  - `snapshot(thread_id)` — full accumulated `ChatState` for a thread, used
    by the UI to render the trace.
  - `pending_question(result)` — extracts the interrupt prompt, if any.
  - `get_graph()` — the compiled graph, cached with `functools.lru_cache`;
    `reset_graph()` clears it (used by tests after wiring changes).

## Shared state

[src/state.py](../src/state.py) defines `ChatState`, a `TypedDict` that is
the *only* channel through which nodes communicate. Fields fall into six
groups:

| Group | Fields |
|---|---|
| Conversation | `session_id`, `messages` (LangGraph message reducer), `customer_message` |
| Triage | `triage`, `domain`, `clarifications` |
| Evidence | `customer_record`, `account_records`, `transactions`, `cards`, `loans`, `credit_profile`, `fee_history`, `loan_assessment`, `fraud_risk`, `sources` |
| Specialist | `draft`, `retry_count`, `critic_feedback` |
| Critic | `critique`, `critique_history`, `draft_history` |
| Outcome | `outcome`, `outcome_summary`, `pending_approval`, `ticket_id`, `escalation_id`, `questions` |
| Observability | `trace` (append-only), `tool_calls` (append-only), `errors` (append-only), `degraded` |

`new_turn(session_id, customer_message)` resets every per-turn field except
`messages`, whose reducer (`add_messages`) is what carries the conversation
forward across turns on the same thread.

## Schemas

All defined with Pydantic in [src/schemas.py](../src/schemas.py):

- **`Domain`** — `card | loan | account | fraud | out_of_scope`.
- **`TriageResult`** — domain, `bank_related` flag, intent, extracted
  `customer_id` / `account_id` / `card_last_four` / `amount`,
  `missing_info` (questions the model wants answered), `reasoning`.
- **`ProposedAction`** — `action_type` (`refund | reissue_card |
  reset_access | open_dispute | loan_decision | information_only`),
  description, amount, and the policy `citation` authorizing it.
- **`Draft`** — the specialist's output: `reply` (customer-facing text),
  optional `action`, `citations`, internal `notes`, and a `confidence`
  score.
- **`Critique`** — the critic's verdict (`Verdict.approve | revise |
  escalate`), `grounded` flag, `problems`/`fixes` lists, optional
  `escalation_reason`, `rationale`.
- **`Source`** — one retrieved policy chunk: `doc_id`, `title`, `domain`,
  `doc_type`, `content`, similarity `score`.
- **`ToolCall`** — a record of one tool invocation (name, args, `ok`,
  `result`/`error`, `attempts`) — this is what populates the trace panel.
- **`TraceEvent`** — `(node, role, detail, at)`, one per meaningful step.
- **`PendingApproval`** — a queued action awaiting a human decision
  (`approval_id`, `action_type`, `amount`, `reason`, `status`).

## Agents

Located in [src/agents/](../src/agents/). Every agent function returns
`(result, error)` — `error` is a string when the LLM call failed and a
deterministic fallback was used instead, so the calling node can flag the
turn as `degraded`.

### Triage — [triage.py](../src/agents/triage.py)

- Calls `TRIAGE_PROMPT` for structured `TriageResult` output.
- **Never trusts the model alone** for identifiers or routing-critical
  fields: `_fill_identifiers` regex-extracts customer/account IDs, card last
  four, and dollar amounts from the raw text as a backstop; `_settle_domain`
  overrides the model's domain choice when a higher-priority keyword family
  (fraud > loan > card > account) is present in the text; `_settle_missing_info`
  recomputes `missing_info` from hard application rules (`_required_missing`)
  rather than trusting whatever the model asked for, so it can't manufacture
  or skip a clarification the graph depends on.
- **`_heuristic`** is the full deterministic fallback used when
  `GEMINI_API_KEY` is unset or the call fails: keyword-matches a domain, or
  returns `out_of_scope` (with `bank_related` set appropriately) when
  nothing in the message reads as banking at all.

### Specialists — [specialists.py](../src/agents/specialists.py)

`run_card_agent`, `run_loan_agent`, `run_account_agent` share one
implementation (`_draft`) parameterized by a domain-specific prompt
(`CARD_PROMPT`, `LOAN_PROMPT`, `ACCOUNT_PROMPT`). Each:

1. Formats the evidence gathered for it (`format_evidence`) — account
   balances, transactions, cards, loans, the loan eligibility assessment,
   fee-waiver history.
2. Formats retrieved policy chunks (`format_sources`).
3. On a revision (`attempt > 0`), injects a `REVISION_BLOCK` containing the
   previous reply and the critic's specific problems/fixes.
4. Calls the LLM for a structured `Draft`.
5. Runs `settle_citations`, which prunes any citation the retriever didn't
   actually return (defense in depth alongside the critic's guard) and
   implements the `FORCE_BAD_DRAFTS` demo switch that strips citations to
   deliberately trigger a revision.
6. Falls back to `_fallback_draft` — a safe, low-confidence holding reply
   with an `information_only` action — if the LLM is unavailable.

### Fraud — [fraud.py](../src/agents/fraud.py)

Structurally different from the other three: it **never reaches the
critic**. There is no automated resolution to validate — only a case to
hand to a human investigator — so `fraud_agent` wires directly to
`escalated` in the graph. Whatever action the model proposes is forcibly
overwritten to `information_only` before the draft leaves this agent, so a
fraud path can never carry a money-moving action even if the model tries.

### Critic — [critic.py](../src/agents/critic.py)

Two layers, `run_critic`:

1. **`citation_guard`** (deterministic, no model call) rejects a draft if:
   - no policy was retrieved at all;
   - the draft cites nothing;
   - a citation doesn't appear among the retrieved doc IDs;
   - a proposed non-informational action has no citation, or cites
     something not retrieved;
   - the reply asserts a consequential action is *already done*
     (`DONE_CLAIMS` regex) — nothing is done until the approval/threshold
     check downstream has run.
   A guard failure returns a `revise` verdict without ever calling the LLM.
2. If the guard passes, `CRITIC_PROMPT` is called for a structured
   `Critique` with verdict `approve | revise | escalate`. The prompt is told
   the automated limit and how many revisions remain, but the actual
   approval-limit *decision* still happens later in `route_after_critic` —
   the critic judges quality and groundedness, not authorization.
3. If the model is unavailable, `_fallback_critique` returns `revise` (if
   the guard already found problems) or `escalate` (if the guard passed but
   the model didn't run) — approval is never the fallback outcome.

## Routing rules

[src/graph/routes.py](../src/graph/routes.py) — pure functions of
`ChatState`, so any route the graph took can be explained and replayed from
state alone:

- **`route_after_triage`** — `out_of_scope` domain → `out_of_scope` node.
  Missing required info → `ask_user`, unless `clarifications >=
  MAX_CLARIFICATIONS` (then `escalated`). Otherwise → `{domain}_agent`.
- **`route_after_out_of_scope`** — `bank_related` → `escalated` (a person
  can still help); otherwise → `END` (nothing to hand off, the decline is
  the whole outcome).
- **`needs_human_approval`** — the threshold logic, independent of the
  critic's verdict:
  - domain is `loan` → always `True` (every credit decision needs an
    officer, POL-AUTH-008);
  - a `refund` action → `True` only if `amount > MAX_APPROVAL_LIMIT`;
  - anything else that isn't `information_only`, `reset_access`, or
    `open_dispute` → `True` (not pre-authorized for automation).
- **`route_after_critic`** — `escalate` verdict → `escalated`. `revise`
  verdict → back to the originating specialist, unless
  `retry_count > MAX_REVISION_ATTEMPTS` (then `escalated`, the loop's
  termination condition). `approve` → `human_approval` if
  `needs_human_approval`, else `resolved`.

## Tools

Split by effect in [src/tools/](../src/tools/):

- **`readonly.py`** — `lookup_customer`, `lookup_accounts`,
  `lookup_transactions`, `lookup_cards`, `lookup_loans`,
  `lookup_fee_waivers`, `lookup_ticket`, `lookup_specialist_queue` (all
  query `data/bank.db` via `sqlite3`), plus `search_policy` (wraps
  `retrieve`), `assess_loan_eligibility`, and `score_fraud_risk`.
  `assess_loan_eligibility` and `score_fraud_risk` are **deterministic
  computations**, not model calls — a credit or fraud-risk recommendation
  must be reproducible and explainable, so it's code, not an LLM's
  impression.
- **`actions.py`** — the four simulated side-effect tools:
  `process_refund`, `request_human_approval`, `create_escalation`,
  `record_approval_decision`. Every one appends an entry to
  `storage/action_log.json` (`_log`), and `create_escalation` also writes a
  row to the `tickets` table. Every return value carries a
  `SIMULATED: recorded locally only, no real banking system was contacted`
  note.
- **`runner.py`** — `call_tool(tool, args)` is the single choke point every
  node uses to invoke a tool. It retries transient failures up to
  `MAX_TOOL_ATTEMPTS` with backoff, honors the `FORCE_TOOL_FAILURE` demo
  switch, and always returns `(result, ToolCall)` so a failure is visible in
  the trace and the calling node can decide whether to continue on partial
  evidence or degrade.

## Retrieval (RAG)

[src/retriever.py](../src/retriever.py):

- `load_vector_store()` opens a persisted **Chroma** collection at
  `chroma_db/` using `sentence-transformers/all-MiniLM-L6-v2` embeddings
  (via `langchain-huggingface`), built ahead of time by
  `scripts/build_vector_db.py` from the Markdown files in `policies/`.
- `retrieve(query, domain, doc_types, k=4)` runs a similarity search
  filtered to `{"domain": {"$in": [domain, "shared"]}}` — each specialist
  only sees its own domain's policies plus documents tagged `shared`
  (e.g. POL-AUTH-008, the approval-authority policy every domain needs).
- Each hit becomes a `Source` with a `doc_id` parsed from the chunk's
  frontmatter (`**Policy ID:** POL-...`) or synthesized as
  `POL-UNKNOWN#<index>` if missing.
- `retrieved_ids(sources)` returns the set of valid doc IDs (including the
  bare policy ID without a `#chunk` suffix) — this is exactly what the
  critic's citation guard and `settle_citations` check citations against.

`scripts/build_vector_db.py` reads every `policies/*.md` file, parses a
simple frontmatter block (`parse_frontmatter`) for `domain`/`doc_type`
metadata, chunks the body (`CHUNK_SIZE=700`, `CHUNK_OVERLAP=100` in
`config.py`), and writes the collection with `rebuild=True` by default —
**re-run it whenever a policy file changes**, since chunk boundaries must
stay stable for a given index.

## Data stores

- **`data/bank.db`** (SQLite, created by
  [scripts/create_data.py](../scripts/create_data.py)) — demo tables:
  `customers`, `accounts`, `transactions`, `cards`, `loans`,
  `fee_waivers`, `credit_profiles`, `tickets`. Seeded with six demo
  customers (`CUST-001`..`CUST-006`) covering a plain account, a premier
  tier, an overdraft scenario, a fraud-hold customer, and a locked-out
  customer.
- **`chroma_db/`** — persisted Chroma vector index over `policies/*.md`.
- **`storage/action_log.json`** — append-only JSON audit log of every
  simulated side effect (refunds, approvals, escalations, human decisions).
- **`policies/`** — the actual source of truth for RAG: policy documents
  (`POL-*`) and knowledge-base guides (`KB-*`) covering Regulation E, lost
  cards, fee waivers, scam claims, wire transfers, KYC, loan hardship and
  underwriting, online access, card disputes, and approval authority.

## Failure handling

Every external dependency degrades rather than crashes the turn:

| Failure | Where handled | Behavior |
|---|---|---|
| No `GEMINI_API_KEY` / model call fails | `src/model.py` raises `LLMUnavailable` | Triage falls back to keyword classification; specialists issue a low-confidence holding reply; the critic falls back to `revise`/`escalate`, never `approve` |
| SQL / bank DB error | `src/tools/readonly.py` raises `ToolError`, caught by `call_tool` | Retried up to `MAX_TOOL_ATTEMPTS`; on terminal failure the node continues on partial evidence and records the error in `state["errors"]` |
| Vector store missing/empty | `src/retriever.py` raises `FileNotFoundError`/`ValueError` | `_retrieve` in `nodes.py` catches any exception, returns no sources, and logs it to the trace — the citation guard then correctly rejects any draft as ungrounded |
| Approval queue tool fails | `human_approval_node` | Falls through to `create_escalation` instead of silently resolving |
| Demo injection flags | `FORCE_BAD_DRAFTS`, `FORCE_TOOL_FAILURE`, `FORCE_LLM_FAILURE` in `config.py` | Used only by the scripted scenarios to deterministically exercise these paths in the UI |

Every degraded turn sets `state["degraded"] = True` and appends to
`state["errors"]`, which the UI's trace panel surfaces.

## Extending the system

**Adding a new domain** (e.g. "savings"):
1. Add it to the `Domain` enum in `schemas.py`.
2. Write a system prompt in `prompts.py` and build it with
   `specialist_prompt(...)`.
3. Add a `run_savings_agent` in `specialists.py` (or a new module) using the
   shared `_draft` helper.
4. Add a `savings_agent_node` in `graph/nodes.py` (reuse `_specialist_node`)
   and wire it into `graph/build.py` (`add_node`, the triage conditional
   edge, and the `-> critic` edge).
5. Add routing: extend `route_after_triage`'s implicit `f"{domain}_agent"`
   mapping (already generic) and update `QUEUES` in `nodes.py` for its
   escalation queue.
6. Tag relevant policy docs in `policies/` with `domain: savings` (or
   `shared`) frontmatter and rebuild the vector index.

**Changing the automated approval limit**: edit `MAX_APPROVAL_LIMIT` in
`src/config.py` — `route_after_critic` / `needs_human_approval` and the
`resolved_node` messaging all read from this single constant.

**Adding a new tool**: define it with `@tool` in `readonly.py` (no side
effects) or `actions.py` (side effects, must log via `_log` and return a
`SIMULATED` note), then call it through `call_tool(...)` from a node — never
invoke a tool directly, or it won't appear in the trace.
