# Agentic AI — Bank Support Assistant

A multi-agent customer support system for a fictional bank, built with
[LangGraph](https://langchain-ai.github.io/langgraph/) and a
[Streamlit](https://streamlit.io/) chat UI.

A customer message is **triaged**, handed to a **domain specialist** that
retrieves policy and drafts a reply, checked by a **critic**, and then either
**resolved automatically**, **parked for a human to approve**, or
**escalated** to a specialist queue. Every step — retrieval, tool calls,
drafts, verdicts — is recorded in a shared state object so the UI can show
exactly how each answer was produced.

> This is a demonstration / portfolio project. All banking actions (refunds,
> card reissue, approvals, escalations) are **simulated**: they write to a
> local SQLite database and a JSON action log, never to a real system.

Detail info about the graph, agents, and data flow
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Table of contents

- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Configuration](#configuration)
- [Demo scenarios](#demo-scenarios)
- [Testing](#testing)
- [Key concepts](#key-concepts)

---

## How it works

```
START → intake → triage ─┬─→ ask_user → wait_for_user → triage   (missing info)
                          ├─→ card_agent    ─┐
                          ├─→ loan_agent     ─┼─→ critic ─┬─→ resolved
                          ├─→ account_agent ─┘            ├─→ human_approval
                          ├─→ fraud_agent  ──────────────→ escalated
                          └─→ out_of_scope ──────────────→ escalated | END
```

1. **Triage** — an LLM (or a deterministic keyword fallback if no API key is
   configured) classifies the request into `card`, `loan`, `account`,
   `fraud`, or `out_of_scope`, extracts identifiers (customer ID, account ID,
   amount), and flags anything still missing.
2. **Ask / wait for user** — if required information is missing, the graph
   asks a question and **pauses** (a LangGraph `interrupt`), resuming from
   the same point once the customer replies.
3. **Specialist agent** (card / loan / account) — pulls the relevant bank
   records (accounts, transactions, cards, loans, fee history) via read-only
   tools, retrieves relevant policy chunks from a Chroma vector store, and
   drafts a reply plus an optional proposed action (e.g. `refund`,
   `reissue_card`).
4. **Fraud agent** — a structural exception: it never proposes an automated
   action and never reaches the critic. Every fraud report is scored for
   risk and handed straight to a human investigator.
5. **Critic** — validates the draft in two layers:
   - a **deterministic citation guard** (no model call) that rejects
     unsupported claims, invented citations, or replies that falsely claim
     an action is already done;
   - if the guard passes, an **LLM review** that returns `approve`,
     `revise`, or `escalate`.
   A `revise` verdict sends the draft back to the same specialist with
   feedback, up to `MAX_REVISION_ATTEMPTS` times before it escalates.
6. **Outcome** — application code (not the model) makes the final call:
   - **resolved** — refunds at or under the automated limit go through
     automatically;
   - **human_approval** — anything over the limit, or any loan decision,
     is queued for a person and the workflow stops;
   - **escalated** — fraud, out-of-scope-but-bank-related requests, spent
     retry/clarification budgets, or critic escalations are handed to a
     specialist queue.

Every hop appends a `TraceEvent` to the shared state, so the "How this was
decided" panel in the UI is a literal readout of the graph's own log rather
than a second explanation generated after the fact.

## Project layout

```
app.py                  Streamlit UI (chat, approval cards, decision trace)
main.py                 Entry point: `streamlit run main.py`
src/
  config.py             Paths, env vars, limits, demo failure-injection flags
  schemas.py            Pydantic models: TriageResult, Draft, Critique, ...
  state.py               LangGraph ChatState (the shared state object)
  model.py              Gemini chat model wrapper + structured-output helper
  retriever.py          Chroma vector store loading and semantic retrieval
  prompts.py            All LLM prompt templates (triage, specialists, critic)
  scenarios.py          Scripted demo scenarios shown in the sidebar
  agents/
    triage.py           Classification + regex/keyword fallback
    specialists.py      Card / loan / account drafting agents
    fraud.py            Fraud agent (always escalates)
    critic.py           Citation guard + LLM-based critique
  graph/
    build.py            Graph wiring, checkpointer, run/resume/snapshot API
    nodes.py            Node implementations (one function per graph node)
    routes.py           Pure routing functions (state -> next node)
  tools/
    readonly.py         Read-only bank/policy lookup tools (SQLite + Chroma)
    actions.py          Simulated side-effect tools (refund, approval, ticket)
    runner.py           Uniform tool invocation with retry + audit trail
ui/
  components.py         Streamlit rendering helpers (cards, panels, banners)
scripts/
  create_data.py        Creates data/bank.db with demo customers/accounts/etc.
  build_vector_db.py    Chunks policies/*.md into the Chroma vector index
policies/                Markdown policy & knowledge-base documents (source of truth for RAG)
data/bank.db             Demo SQLite database (customers, accounts, transactions, ...)
chroma_db/                Persisted Chroma vector index
storage/action_log.json  Append-only audit log of every simulated side effect
tests/                   pytest suite (model, retriever, end-to-end workflow)
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build the demo bank database
python -c "from scripts.create_data import setup_bank_db; setup_bank_db()"

# Chunk policies/*.md and build the Chroma vector index
python -m scripts.build_vector_db
```

Create a `.env` file in the project root with:

```
GEMINI_API_KEY=your-key-here
# Optional overrides:
# GEMINI_MODEL=gemini-2.0-flash
# EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

`GEMINI_API_KEY` is **optional**. Without it, every LLM call raises
`LLMUnavailable` internally and each agent falls back to a deterministic
path (keyword classification for triage, a safe holding reply for
specialists, an escalate-by-default verdict for the critic) — the app stays
fully functional, just less capable.

## Running the app

```bash
streamlit run main.py
```

Open the sidebar to:
- see whether Gemini is configured and how many policy chunks are indexed;
- run one of the 12 scripted **demo scenarios** (happy path, over-limit
  approval, missing info, fraud, loan, revision loops, tool/model failure,
  out-of-scope);
- start a new conversation thread.

Each turn shows the assistant's reply, an outcome banner, an approval card
(for actions pending human sign-off) or escalation card, and an expandable
"How this was decided" trace with every tool call, retrieval, draft, and
verdict for that turn.

## Configuration

All configuration lives in [src/config.py](src/config.py):

| Setting | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | from `.env` | Enables Gemini-backed agents; empty = deterministic fallbacks |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Chat model used for structured outputs |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for the vector store |
| `MAX_REVISION_ATTEMPTS` | `2` | How many times a draft can be sent back to a specialist |
| `MAX_CLARIFICATIONS` | `3` | How many times triage can ask the customer for missing info before escalating |
| `MAX_TOOL_ATTEMPTS` | `3` | Retries for a failing tool call |
| `MAX_APPROVAL_LIMIT` | `50.0` | Refunds at or under this amount can resolve automatically; above it, a human must approve |

`FORCE_BAD_DRAFTS`, `FORCE_TOOL_FAILURE`, and `FORCE_LLM_FAILURE` are
demo-only switches (also in `config.py`) used by the scripted scenarios in
the sidebar to deterministically trigger the revision loop, a tool failure,
or a model outage.

## Demo scenarios

[src/scenarios.py](src/scenarios.py) defines 12 scripted conversations, each
with an expected route through the graph, used both by the Streamlit
sidebar and by the test suite:

1. Happy path — automatic refund
2. Branching — fees over the cap need a supervisor
3. Missing information — pause and resume
4. Fraud — bypasses the critic entirely
5. Loan — always needs a loan officer
6. Revision loop — one send-back, then accepted
7. Tool failure — customer lookup unavailable
8. Model failure — chat model outage, keyword fallback
9. Out of scope — not a banking question at all
10. Out of scope — banking question no specialist covers
11. Revision loop — two send-backs, accepted on the third try
12. Revision loop exhausted — budget spent, case escalates

## Testing

```bash
pytest tests/test_workflow.py   # end-to-end graph behavior for every scenario
pytest                          # full suite (model, retriever, workflow)
```

## Key concepts

- **Everything flows through `ChatState`** ([src/state.py](src/state.py)).
  Agents never call each other directly; a node reads state, calls an
  agent/tool, and returns a partial state update. This is what makes the
  UI's decision trace a complete, replayable account of a turn.
- **Routing is pure and separate from reasoning**
  ([src/graph/routes.py](src/graph/routes.py)). Each router is a function of
  state only, so "why did this go to human_approval?" always has a
  deterministic, inspectable answer — application thresholds (like the $50
  limit) are enforced in code, never left to the model's judgment.
- **The critic's citation guard runs before any model call.** Whether a
  cited document was actually retrieved is a checkable fact, not an opinion,
  so it's enforced deterministically and only a draft that passes gets a
  model-based review.
- **Every side effect is simulated and logged.** `src/tools/actions.py`
  writes refunds, approvals, and escalations to `data/bank.db` and
  `storage/action_log.json`, and every result is labeled
  `SIMULATED: recorded locally only`.
- **Graceful degradation, not crashes.** A missing API key, a failed tool
  call, or a vector store outage all fall back to conservative deterministic
  behavior (see `LLMUnavailable` in [src/model.py](src/model.py)) rather than
  raising through to the UI.
