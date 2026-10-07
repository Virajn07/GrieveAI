# GrieveAI

## Explainable Near-Real-Time Hindi-English Code-Mixed Grievance Triage and Department Routing

GrieveAI is an AI-assisted grievance triage and routing system designed for higher-education institutions.

The project focuses on processing student grievances written in:

- English
- Devanagari Hindi
- Romanised Hindi / Hinglish
- Hindi-English code-mixed text

The system is being developed as a case study for the **VCET Student Grievance Redressal workflow**.

---

## Local setup

Use Python 3.10 or newer. From the repository root, create and activate a
virtual environment, then install `requirements-local.txt`. Place the provided
IndicBERT Run 1 archive at the path described above and extract it into the
repository root so its files land under `checkpoints/indicbert_lora/run1/`.
Copy `.env.example` to `.env`, set a local `ADMIN_USERNAME`, `ADMIN_PASSWORD`, and stable `SECRET_KEY`, and keep `MODEL_BACKEND=indicbert`. The first inference downloads the public base
model if it is not already cached; GPU is used when available, otherwise CPU.

Start the local app with `python -m app.app`. Check model and database readiness
at `/api/v1/health`. A direct model check is:

```powershell
python -c "from src.inference import predict_grievance; print(predict_grievance('Sir meri attendance 68% hai aur teacher ne mujhe exam ke liye allow nahi kiya.'))"
```

Run the test suite with `python -m unittest discover -s tests -v`. Tests do not
require Ollama or an API key; LLM provider behavior is mocked. Local Ollama
(`OLLAMA_BASE_URL=http://127.0.0.1:11434`, `OLLAMA_MODEL=qwen3:8b`) is the
primary structured-analysis provider. If it is unavailable, configured
OpenRouter models are tried before deterministic fallback. Set
`MODEL_BACKEND=baseline` only to explicitly select TF-IDF for development or
comparison.

## Environment variables

Create an untracked local `.env` from `.env.example` and configure:

```dotenv
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<your-local-password>
SECRET_KEY=<a-long-random-session-secret>
MODEL_BACKEND=indicbert
```

The application hashes `ADMIN_PASSWORD` with Werkzeug when it starts; the
plaintext is not saved in the database or stored in the Flask configuration.
Without both administrator variables, protected admin functions fail closed
and the sign-in page displays a setup message. Do not commit `.env` or real
credentials. The local admin session survives refresh and can be ended with
Sign out. The configured confidence threshold is read from the existing
threshold configuration.

### Local administrator access

Sign in with the configured username and password at `/admin/login`. The
application uses Flask sessions for protected dashboard, review and audit
routes. There is no built-in demo password and no token-based admin login.
The student submission page is `/submit`; administrators sign in at
`/admin/login`. The dashboard and individual case review pages use the
existing database records and audit log. If there are no records, the inbox
shows an empty state. The checked-in training corpus is synthetic and is not
counted as live campus grievances or VCET activity.

### Local synthetic demo data

To populate the local database with a repeatable set of 297 clearly synthetic
demo grievances (nine per taxonomy subcategory), run:

```powershell
python -m app.seed_demo
```

The command is safe to rerun: if its `DEMO-*` records already exist, it leaves
the database unchanged. To replace only those seeded records, run:

```powershell
python -m app.seed_demo --reset
```

Reset deletes only grievance rows whose acknowledgement starts with `DEMO-`
and their audit entries. Other local submissions are preserved. The seeded
text comes from the checked-in synthetic training corpus; records are marked
as synthetic in their acknowledgement, model version, analysis and audit
history. They are for local UI demonstration only.

## Project Objective

The goal of GrieveAI is not simply to classify a grievance.

The system is designed to transform raw student complaints into structured, explainable and actionable information that can support the institutional grievance redressal workflow.

A typical workflow is:

Student grievance
        ↓
Language / script detection
        ↓
PII detection and redaction
        ↓
Category classification
        ↓
Subcategory classification
        ↓
Priority estimation
        ↓
Confidence gate
        ↓
Duplicate / recurring grievance detection
        ↓
Explanation
        ↓
Summary and action recommendation
        ↓
Department routing
        ↓
Ticket / SLA tracking
        ↓
Administrator dashboard
        ↓
Resolution and audit trail

Human administrators remain responsible for final decisions.

---

# Current Taxonomy

GrieveAI currently uses **7 major categories and 33 subcategories**.

## 1. Academics

- `timetable_scheduling`
- `teaching_quality`
- `faculty_conduct`
- `syllabus_course_progress`
- `attendance`

## 2. Examinations

- `hall_ticket`
- `marks_discrepancy`
- `revaluation`
- `examination_schedule`
- `examination_process`

## 3. Fees / Accounts

- `fee_discrepancy`
- `refund`
- `scholarship`
- `payment_transaction`
- `accounts_financial_documentation`

## 4. IT / Library

- `portal_account_access`
- `wifi_network`
- `software_license`
- `library_book_availability`
- `library_digital_resources`

## 5. Infrastructure

- `classroom_lab_maintenance`
- `electrical_issues`
- `sanitation_cleanliness`
- `parking_access`
- `construction_facility_disruption`

## 6. Transport

- `commuting_accessibility`
- `transport_academic_conflict`
- `travel_safety_access`

## 7. Canteen

- `food_quality`
- `food_hygiene`
- `pricing`
- `menu_variety`
- `food_service`

Cross-domain grievances are intended to be represented through a primary issue together with contextual information such as contributing causes, affected areas and recommended actions.

---

# Machine Learning

The current primary transformer model is:

**IndicBERT v2 + LoRA**

Model:

`ai4bharat/IndicBERTv2-MLM-only`

The model is fine-tuned using parameter-efficient LoRA adapters.

The architecture contains:

- Category classification head
- Hierarchical subcategory classification head
- Priority regression head

Subcategory prediction is constrained according to the selected category.

### Runtime and checkpoint setup

`src/indicbert_inference.py` loads the trained Run 1 adapter and task heads.
The supplied project ZIP contains the checkpoint archive at
`checkpoints/archives/grieveai_indicbert_v2_run1.zip`; large checkpoint files
are excluded from Git by repository policy. Extract that archive into
`checkpoints/indicbert_lora/run1/`, preserving its `adapter/`, `tokenizer/`,
`heads.pt`, `taxonomy.json`, and `metrics.json` structure. The adapter archive
does not include the base transformer weights: the first inference needs
`ai4bharat/IndicBERTv2-MLM-only` available in the Hugging Face cache or network.

The Flask default is `MODEL_BACKEND=indicbert`. A missing or incompatible model
returns an explicit load error and degraded health status; it never silently
switches to TF-IDF. Set `MODEL_BACKEND=baseline` only when you intentionally want
the TF-IDF + Logistic Regression baseline. The MuRIL + LoRA implementation is
retained as a secondary experiment (`MODEL_BACKEND=muril`).

The Run 1 checkpoint was trained with the project’s 7-category/33-subcategory
taxonomy, the saved tokenizer, 128-token truncation, attention-mask mean
pooling, LoRA rank 8/alpha 16/dropout 0.1 on query/value projections, and three
classification/regression heads. Inference checks the checkpoint taxonomy
against `config/taxonomy.json` and masks subcategories to the predicted parent.
The configured 0.70 confidence threshold is an initial demo setting, not a
scientifically calibrated optimum.

---

# Model Experiments

Presentation-ready results, exact train/validation/test splits, current
baseline comparisons, language-wise evaluation, limitations, and a reproducible
IndicBERT evaluation command are documented in
[`docs/PRESENTATION_METRICS.md`](docs/PRESENTATION_METRICS.md). All available
results are from synthetic data; they are not VCET pilot performance. Earlier
notebook comparison figures use different or incompletely identified runs and
must not be combined as an apples-to-apples model ranking.

---

# 60K Stress Test

A larger synthetic dataset containing:

- 60,000 records
- 7 categories
- 33 subcategories
- English
- Devanagari Hindi
- Romanised Hinglish

was used to test training scalability.

The 3-epoch IndicBERT v2 + LoRA stress test produced:

- Category Macro-F1: **100.0%**
- Subcategory Macro-F1: **100.0%**
- Priority MAE: **0.783**
- High-priority recall: **0.0%**

These results demonstrate scalability on the synthetic benchmark.

They are **not representative of real-world VCET performance** because the dataset is artificially generated.

---

# Why IndicBERT v2?

The project requires multilingual and code-mixed processing involving Indian languages and English.

IndicBERT v2 is therefore being evaluated as the primary language representation model because its pretraining is targeted toward Indian-language NLP.

The final deployment model will be selected based on evaluation using human-annotated VCET data rather than synthetic data alone.

---

# Baseline and Research Comparisons

The project maintains multiple model approaches for comparison:

### Classical baseline

TF-IDF features with Logistic Regression provide a strong and interpretable baseline.

### MuRIL

MuRIL + LoRA was evaluated for multilingual Indian-language representation.

### IndicBERT v2

IndicBERT v2 + LoRA currently provides the strongest synthetic subcategory results among the tested transformer approaches.

Keeping these experiments allows the final study to compare classical and transformer-based approaches rather than reporting only a single model.

---

# On-demand Explainability

GrieveAI is intended to provide explanations for classification decisions.

Administrators can request SHAP token attribution for the classifier selected
by `MODEL_BACKEND`. Explanations are generated on demand, so normal submission
does not pay the SHAP computation cost.

The goal is to preserve the original language/script of the grievance where practical.

---

# Duplicate and Recurring Grievance Detection

The system identifies similar grievances and stores duplicate candidates and
recurring-cluster IDs with the ticket.

This allows multiple individual complaints to be grouped into recurring institutional issues.

For example:

    42 students report problems with the same Wi-Fi service.

Instead of treating these only as 42 independent tickets, the system can identify the recurring issue and provide administrators with an aggregated view.

---

# LLM-Assisted Analysis

The LLM layer is separate from the primary classifier.

IndicBERT performs structured classification locally.

An LLM is intended for tasks such as:

- Grievance-group summarisation
- Root-cause/theme extraction
- Administrative report generation
- Action recommendations
- Natural-language summaries

The optional OpenRouter-compatible layer returns validated structured analysis
when configured. A deterministic fallback keeps submissions working without a
key or network access.

PII redaction will occur before sensitive grievance content is sent to an external LLM service.

LLM recommendations are advisory and remain subject to human review.

---

# Backend and Application

Backend:

- Python
- Flask
- Flask-SQLAlchemy
- PostgreSQL

Application components:

- Student grievance submission
- Grievance processing pipeline
- Admin dashboard
- Classification results
- Confidence indicators
- SHAP explanations
- Duplicate/recurring grievance groups
- Department routing
- Ticket management
- SLA tracking
- Human override
- Audit trail
- Report generation

---

# Database

SQLite is the local default; PostgreSQL is available through `DATABASE_URL` and
Alembic migrations.

The database will store structured grievance information such as:

- Grievance text / appropriately protected representation
- Language
- Category
- Subcategory
- Priority
- Confidence
- Duplicate / recurring group
- Routing recommendation
- Ticket status
- SLA information
- Human overrides
- Audit information
- Model version

The trained ML model itself is stored separately from PostgreSQL.

---

# Model Deployment Concept

The selected model checkpoint is stored separately from PostgreSQL and loaded
by the inference module. IndicBERT v2 + LoRA is the runtime default; the
TF-IDF model is available as an explicit baseline, and MuRIL remains an optional
research backend.

Conceptually:

    Student
       ↓
    Flask application
       ↓
    Configured local inference model
       ↓
    Classification
       ↓
    PostgreSQL
       ↓
    Admin dashboard

The model will not automatically retrain whenever a new grievance arrives.

Instead, corrected and human-validated grievances can later be collected into a new training dataset and used for controlled model updates.

Model versions will be maintained separately.

---

# Real VCET Data

Synthetic data is currently being used for development and experimentation.

The next major research stage is evaluation using de-identified and appropriately authorized VCET grievance data.

The intended process is:

1. Obtain appropriate institutional permission.
2. De-identify sensitive information.
3. Establish the final annotation guidelines.
4. Human-annotate a representative dataset.
5. Create train/validation/test splits.
6. Fine-tune IndicBERT v2.
7. Evaluate on held-out real VCET grievances.
8. Perform error analysis.
9. Evaluate confidence-based human handoff.
10. Evaluate department routing.
11. Conduct pilot testing.
12. Document results for the research paper.

Synthetic results will not be presented as real VCET performance.

---

# Evaluation Plan

The final evaluation will include more than classification accuracy.

Planned metrics include:

### Classification

- Category Macro-F1
- Subcategory Macro-F1
- Per-class precision
- Per-class recall
- Confusion matrices

### Priority

- Priority MAE
- High-priority recall

### Workflow

- Department routing accuracy
- Human override rate
- Confidence-gate performance
- Duplicate/recurring grievance detection
- Time-to-route reduction

### LLM output

- Summary factuality
- Human evaluation
- Action recommendation usefulness

---

# Research Positioning

The project is positioned around the combination of:

- Higher-education grievance processing
- Hindi-English code-mixed text
- Fine-grained hierarchical classification
- Explainability
- Confidence-aware human handoff
- Duplicate/recurring grievance aggregation
- LLM-assisted administrative analysis
- Institutional department routing
- End-to-end grievance workflow evaluation

The research contribution will ultimately be evaluated using real, appropriately authorized and human-annotated institutional data.

---

# Repository Structure

```text
GrieveAI/
│
├── config/
│   └── taxonomy.json
│
├── data/
│   └── README.md
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── baseline_classifier.py
│   └── ...
│
├── tests/
│
├── experiments/
│   ├── README.md
│   └── notebooks/
│
├── checkpoints/
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
