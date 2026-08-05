# agentic-ai

Bank support assistant: triage → specialist → critic → resolve / approve / escalate.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build demo bank DB + policy vector index
python -c "from scripts.create_data import setup_bank_db; setup_bank_db()"
python -m scripts.build_vector_db
```

Put `GEMINI_API_KEY` in `.env` (optional — without it, keyword / holding fallbacks run).

## Run UI

```bash
streamlit run main.py
```

## Tests

```bash
pytest tests/test_workflow.py
```
