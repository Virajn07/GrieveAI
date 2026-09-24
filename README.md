# GrieveAI

**Explainable Hindi-English grievance triage and department routing** is an API-first Flask prototype for a future VCET student grievance workflow. This repository is in development; no VCET pilot has been conducted, and no official taxonomy, department mapping, priority policy, or SLA has been approved here.

## Scope and current behavior

The prototype supports English, Devanagari Hindi, and Romanised Hinglish; redacts configured direct identifiers; predicts one of seven configurable categories and a category-valid subcategory; estimates priority; flags possible duplicates; and sends low-confidence submissions to human review. Higher-confidence cases receive a deterministic category-to-department recommendation from `config/departments.json`. The optional LLM service summarizes text only; it does not choose departments. Similar complaints are not automatically merged.

The local web app uses Flask, Jinja, SQLAlchemy, and SQLite by default. `/api/v1` is the integration boundary for a future institution client. PostgreSQL can be selected with `DATABASE_URL` and the appropriate driver. No direct VCET system integration is implemented.

## Architecture

```text
Student Jinja UI / external API client
                  ↓
         Flask REST endpoints
                  ↓
   preprocessing + inference + workflow
                  ↓
       SQLAlchemy / SQLite or PostgreSQL
```

See [architecture](docs/architecture.md), [API](docs/api.md), [data schema](docs/data_schema.md), [evaluation](docs/evaluation.md), [pilot protocol](docs/pilot_protocol.md), [deployment](docs/deployment.md), and [model training](docs/model_training.md).

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
Copy-Item .env.example .env
```

Set a long random `ADMIN_TOKEN` in `.env` before using the admin inbox at `/` (`/admin/login`). For a deployment beyond localhost, also set a strong `SECRET_KEY`, keep `FLASK_DEBUG=0`, and put the app behind HTTPS and institutional authentication. The token header/session is a demonstration control, not a complete institutional identity system.

To rebuild the local baseline checkpoint with the installed scikit-learn version:

```powershell
python train_baseline.py
```

Then start the app:

```powershell
alembic upgrade head
python -m app.app
```

`DATABASE_URL` selects the database. If omitted or blank, Flask and Alembic both
use the local SQLite database at `app/instance/grieveai.db`. Run
`alembic upgrade head` after pulling schema changes; application startup never
creates or upgrades tables implicitly. For PostgreSQL, provide a SQLAlchemy URL
and install the matching driver, then run the same migration command. PostgreSQL
has not been exercised in this development environment.

Student form: `http://127.0.0.1:5000/submit`
Admin inbox: `http://127.0.0.1:5000/`
Health: `http://127.0.0.1:5000/api/v1/health`

## Tests and end-to-end smoke

No external API key is required. Tests use the local synthetic baseline checkpoint, an in-memory SQLite database and deterministic LLM fallback:

```powershell
python -m unittest discover -s tests -v
```

The full smoke test submits text with synthetic contact identifiers, verifies redaction/language/prediction/priority/duplicate handling and confidence review, retrieves the ticket, changes status, overrides department, and checks persisted audit events.

## Synthetic data and classical baselines

The checked-in `data/processed/grievances_synthetic.csv` and generated files are **SYNTHETIC / DEMO DATA**. They are not VCET student grievances and their metrics are not research results.
Baseline joblib checkpoints are generated artifacts and are ignored by Git; build
them locally with `python train_baseline.py` before starting the app.

```powershell
python generate_synthetic_grievances.py --per_combo 15
python train_baseline.py
python train_baselines.py
```

The app's lightweight local baseline uses TF-IDF + Logistic Regression and Ridge priority regression. The separately evaluated research baseline comparison includes keyword rules and TF-IDF + Linear SVM. Synthetic metrics are labelled as pipeline validation. Routing accuracy requires validated gold department labels; override rate requires observed eligible routing decisions. Unavailable metrics remain null.

## MuRIL + LoRA on Colab

Install `requirements-colab.txt` in a GPU runtime with Hugging Face model access. First run the one-epoch pipeline/reload check:

```bash
python src/train.py \
  --data data/processed/grievances_synthetic.csv \
  --taxonomy config/taxonomy.json \
  --smoke_test --batch_size 8 \
  --output_dir checkpoints/muril_lora/smoke_test
```

After reviewing that smoke run, use the intended 3–4 epoch experiment:

```bash
python src/train.py \
  --data data/processed/grievances_synthetic.csv \
  --taxonomy config/taxonomy.json \
  --epochs 4 --batch_size 16 --seed 42 \
  --output_dir checkpoints/muril_lora/run1
```

Training uses duplicate-group-aware train/validation/test partitions and saves the adapter, task heads, tokenizer, taxonomy, training config, and metrics. No MuRIL run is represented as complete unless it is actually executed. Synthetic training is only a pipeline exercise.

## Configuration

Copy `.env.example` to `.env`; never commit secrets. Main settings include `MODEL_BACKEND`, `BASELINE_CHECKPOINT_DIR`, `MURIL_CHECKPOINT_DIR`, `TAXONOMY_CONFIG`, `DEPARTMENTS_CONFIG`, `SLA_CONFIG`, `CONFIDENCE_THRESHOLD`, `DEDUPE_THRESHOLD`, `DATABASE_URL`, `ADMIN_TOKEN`, optional LLM API settings, optional fastText model and optional institution-provided name list.

The taxonomy, department mapping, thresholds and SLA values under `config/` are configurable examples. Mappings/SLA defaults are proposed demos and require approval by the VCET grievance cell. Threshold tuning should use validation data and be frozen for held-out evaluation.

## API quick example

```bash
curl -X POST http://127.0.0.1:5000/api/v1/grievances \
  -H 'Content-Type: application/json' \
  -d '{"text":"Library Wi-Fi has not worked for three days."}'
```

See [docs/api.md](docs/api.md) for endpoint details and admin authentication.

## Pilot data and research limits

VCET data must be authorized, de-identified, quality-checked, and stored outside Git. The import schema is documented in [docs/data_schema.md](docs/data_schema.md). Do not fabricate unknown labels. Use separate duplicate-aware training, validation, and held-out test sets. Report category/subcategory Macro-F1, high-priority recall, routing accuracy, override rate, summary factuality, and time-to-route only when the corresponding validated labels or observed events exist. Current priority heuristics, SLA defaults, language confidence and demo mappings are not official institutional policy or calibrated research measures.

## Security and repository hygiene

`.env`, database files, raw data, Python caches, generated logs and all model
checkpoints are gitignored. Before publishing, inspect staged files with
`git status` and `git diff --cached`; do not add secrets, local databases,
student records or checkpoints. See [GitHub setup](docs/GITHUB_SETUP.md) for
Windows PowerShell commands. The checked-in CSV is synthetic demonstration data.
