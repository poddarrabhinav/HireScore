# HireScore

A FastAPI app that ranks resumes against a job description using a three-stage pipeline.

## Pipeline

**Stage 1 — Skill Match** (choice of two modes)
- **Skill Match**: exact + fuzzy matching using skill expansions
- **ATS**: BM25Okapi scoring via `rank_bm25` — same algorithm used by Elasticsearch/Solr

**Stage 2 — Semantic Relevance**
- OpenAI `text-embedding-3-small` or `text-embedding-3-large`
- Scores resumes against a focused query built from JD skills and role type

**Stage 3 — LLM Judge** (optional, requires `OPENAI_API_KEY`)
- OpenAI structured review: strengths, weaknesses, unknowns, values alignment, verdict

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY to .env
uvicorn app.main:app --reload
```

- App: http://localhost:8000
- Settings: http://localhost:8000/settings
- API docs: http://localhost:8000/docs
- Eval: http://localhost:8000/eval

## Configuration

Edit `config.yaml` to change thresholds, embedding profile, OpenAI models, and scoring weights. Stage 1 mode and embedding profile can also be changed from the Settings page in the UI.

## Notes

- Scanned PDFs (image-only) are not supported — text must be extractable
- Stage 3 is skipped if `OPENAI_API_KEY` is not set; Stage 2 embeddings also require it
