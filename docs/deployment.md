# Local and pilot deployment

## Local setup

Install `requirements-local.txt`, copy `.env.example` to `.env`, and provide a
local `ADMIN_USERNAME`, `ADMIN_PASSWORD`, and `SECRET_KEY`. The default `MODEL_BACKEND=indicbert`
requires the supplied Run 1 adapter checkpoint at
`checkpoints/indicbert_lora/run1/`. The outer project ZIP includes the
`checkpoints/archives/grieveai_indicbert_v2_run1.zip` asset; extract the inner
archive at the repository root to restore that path. Model files are ignored by
Git under the repository's checkpoint policy.

The adapter archive includes its tokenizer, LoRA weights, trained category,
subcategory and priority heads, taxonomy and synthetic metrics. It does not
include the base transformer weights. The first inference needs network access
to download `ai4bharat/IndicBERTv2-MLM-only` from Hugging Face, or a local Hugging
Face cache populated with that model. The application uses CUDA when available
and CPU otherwise. If the adapter, base model, or taxonomy cannot load, health
reports `degraded` and submission returns a clear model error. TF-IDF is only
selected by explicitly setting `MODEL_BACKEND=baseline`; MuRIL remains an
optional research backend.

Run database migrations with `alembic upgrade head` and start Flask with
`python -m app.app`. `/api/v1/health` checks both the database and configured
classifier. A direct inference example and test command are in the root README.
The initial `ML_CONFIDENCE_THRESHOLD=0.70` is a configurable demo gate, not an
optimality claim; low-confidence records require human review.

The app defaults to SQLite at `app/instance/grieveai.db` if `DATABASE_URL` is
unset or blank. Alembic owns schema changes; startup does not call
`create_all()`.

Structured grievance analysis uses local Ollama first (`OLLAMA_BASE_URL`,
default `http://127.0.0.1:11434`; `OLLAMA_MODEL`, default `qwen3:8b`). Install
and start that model separately. If Ollama is unavailable or returns invalid
output, the app uses configured OpenRouter models when an API key is present,
then deterministic analysis. All providers receive PII-redacted text, and
LLM analysis remains advisory to classifier confidence gating and configured
department routing.

## PostgreSQL

Set a PostgreSQL SQLAlchemy URL in `DATABASE_URL`, install its driver (included
in `requirements-local.txt`), and run `alembic upgrade head` before app startup.
The same SQLAlchemy model and migrations are used. PostgreSQL runtime was not
available for verification in this environment; validate against a disposable
PostgreSQL instance before using a pilot database.

For any deployment, configure `SECRET_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, database credentials,
and provider keys through environment variables or a secret manager. Do not
commit `.env`, credentials, trained model binaries, or real records. Before a
real pilot, add HTTPS, institutional authentication, CSRF protection, backups,
logging, and an institution-approved taxonomy, department mapping, and SLA.
Current development and evaluation use synthetic grievance data. Real
institutional data was not available/confirmed for this project phase.
