# Resume Scorer

Resume Scorer is a FastAPI-based resume ranking prototype that matches resumes against a job description using a three-stage pipeline:

1. Stage 1: structured skill match
2. Stage 2: semantic relevance with sentence-transformer embeddings
3. Stage 3: optional OpenAI judge for strengths, weaknesses, unknowns, and hiring verdict

The app is designed as a demoable hiring POC: upload a JD and resumes, inspect extracted skills, remove noisy Stage 1 skills when needed, and review a ranked shortlist. The end-user UI is intentionally streamlined; scoring settings stay backend-managed instead of being exposed as controls in the product flow.

## Current Architecture

### Stage 1: Skill Match
- Extracts JD `core_skills`, `adjacent_skills`, `soft_skills`, and company-value signals
- Extracts resume `skills_keywords`, `project_keywords`, and inferred related skills with a deterministic parser-based heuristic, not per-resume LLM calls
- Uses a structured skill score instead of raw BM25
- Combines:
  - exact skill overlap
  - fuzzy skill overlap
  - skill-level embedding similarity
- Applies canonical skill expansion for terms like `ASR`, `RAG`, `microservices`, `FastAPI`, and `Kubernetes`

### Stage 2: Semantic Relevance
- Uses sentence-transformer embeddings
- Builds a focused semantic query from role type, JD skills, and a condensed JD summary
- Uses a hybrid of full-resume similarity and best resume-chunk similarity
- Runs on `cuda`, `mps`, or `cpu` depending on availability and config

### Stage 3: LLM Judge
- Uses OpenAI for structured review
- Produces:
  - strengths
  - weaknesses
  - unknowns
  - values alignment
  - verdict

## Why This Design

This project uses a staged ranking pipeline because hiring relevance is not purely lexical and not purely semantic.

- Stage 1 is fast and recruiter-friendly: skill coverage is explainable
- Stage 2 catches synonymy and buried relevance that exact matching misses
- Stage 3 adds deeper qualitative reasoning only after earlier filtering

This makes the system easier to explain and cheaper to run than sending every resume directly to an LLM.

## Configuration

System behavior is driven by [config.yaml](/Users/sidpoddra/Desktop/ResumeScorer/config.yaml).

Key settings include:
- OpenAI models
- embedding profile
- embedding device selection
- Stage 1, 2, and 3 thresholds
- Stage 1 skill-match weights
- final scoring weights (`alpha`, `beta`)
- prompt template names

These settings are meant for engineering/configuration, not recruiter-facing use. The main scoring UI stays focused on the workflow, while advanced controls are separated onto an internal settings page.

## Product UI

The UI is split into two clear surfaces:

- a left intake rail for JD, company values, and resumes
- a right review workspace for extracted signals, ranked candidates, semantic context, and optional LLM judging
- a separate internal settings page at [http://localhost:8000/settings](http://localhost:8000/settings) for local threshold/profile overrides

This keeps the live demo focused on screening outcomes rather than exposing tuning knobs to the user.

## Prompt Templates

Prompt templates live in [/Users/sidpoddra/Desktop/ResumeScorer/prompts](/Users/sidpoddra/Desktop/ResumeScorer/prompts):

- [judge_prompt.jinja](/Users/sidpoddra/Desktop/ResumeScorer/prompts/judge_prompt.jinja)
- [keyword_prompt.jinja](/Users/sidpoddra/Desktop/ResumeScorer/prompts/keyword_prompt.jinja)

## Project Structure

```text
ResumeScorer/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── parser.py
│   ├── models.py
│   └── pipeline/
│       ├── __init__.py
│       ├── runner.py
│       ├── stage1_skill_match.py
│       ├── stage2_sbert.py
│       └── stage3_agent.py
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── prompts/
├── eval/
│   ├── job_description.txt
│   ├── company_values.txt
│   └── resumes/
├── temp/
│   └── parser_debug/
├── config.yaml
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Install torch first if needed
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install project dependencies
pip install -r requirements.txt

# 3. Create local env
cp .env.example .env

# 4. Add your OpenAI key
# OPENAI_API_KEY=...

# 5. Run the app
uvicorn app.main:app --reload
```

Open the app at:

- [http://localhost:8000](http://localhost:8000)
- internal settings: [http://localhost:8000/settings](http://localhost:8000/settings)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Runtime Notes

### Embedding Device
The embedding model auto-selects:
- `cuda` if available
- `mps` if available
- otherwise `cpu`

You can override this in [config.yaml](/Users/sidpoddra/Desktop/ResumeScorer/config.yaml) with:

```yaml
embeddings:
  device: cuda
```

### Embedding Cache
Sentence-transformer models are cached locally so they are not downloaded every run.

Configured in:
- [config.yaml](/Users/sidpoddra/Desktop/ResumeScorer/config.yaml)

### Parser Debug Output
Parsed PDF text is written to:
- [/Users/sidpoddra/Desktop/ResumeScorer/temp/parser_debug](/Users/sidpoddra/Desktop/ResumeScorer/temp/parser_debug)

This is useful for debugging extraction quality.

## API

### `GET /health`

Returns basic local status including whether the LLM is configured.

### `POST /score/batch`

Inputs:
- `job_description`
- `company_values`
- `resumes`
- optional JD / values files

Returns:
- ranked candidates
- stage scores
- extracted skill groups
- pipeline stats

Note:
- pipeline thresholds and model choices are supplied by backend configuration even though they are not exposed as form controls in the UI

### `POST /score/judge-single`

Runs the OpenAI judge on a single candidate from the current results view.

## Evaluation

Use the evaluation page at:
- [http://localhost:8000/eval](http://localhost:8000/eval)

Upload:
- a JD or JD file
- a ZIP of resumes
- a JSON file with target final scores

The app will run the current pipeline and return:
- predicted vs target score comparisons
- MAE and RMSE
- counts within `0.10` and `0.20` absolute error

## Limitations

- Resume extraction still depends on `pdfminer`; scanned PDFs need OCR
- JD-side LLM keyword extraction can still introduce noisy adjacent skills if prompts are too broad
- The evaluation set is synthetic / manually curated
- The single-candidate judge depends on `OPENAI_API_KEY`
- Stage 1 internal score fields still use some legacy names for compatibility, even though the logic is now skill-match based

## Next Improvements

- Rename remaining legacy `bm25_*` internal fields to `skill_match_*`
- Add skill ontology / canonical mapping file instead of only prompt-driven inference
- Cache JD keyword extraction results
- Add offline evaluation summaries for multiple role families
- Add recruiter feedback-driven weight calibration
