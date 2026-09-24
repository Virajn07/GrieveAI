# Architecture

GrieveAI is an API-first Flask application. Jinja pages and external clients call the same Flask endpoints, which currently orchestrate the inference and ticket persistence workflow in `app/routes.py`. SQLAlchemy stores tickets and audit events; `DATABASE_URL` selects SQLite locally or PostgreSQL when a PostgreSQL SQLAlchemy driver is installed.

The ML modules are independent of templates: `src/pipeline.py` coordinates redaction, language metadata, classifier inference and the confidence gate without Flask/DB calls; `src/language_id.py` implements the privacy and language primitives; `src/baseline_classifier.py` provides the local model; `src/model.py` provides optional MuRIL+LoRA; `src/priority_dedup.py` flags similar recent submissions; `src/llm_report.py` summarizes when configured and maps departments using `config/departments.json`. The LLM cannot choose a department. Category, subcategory, department, threshold and SLA examples are configurable and explicitly proposed/demo values.

Pipeline order is PII redaction → language/script metadata → hierarchical category/subcategory prediction → priority → duplicate flag → confidence gate → summary and configured routing recommendation → ticket and audit persistence. Low-confidence submissions stay unassigned for authorized human review. Explanations are generated on demand and failures do not affect inference. Authorized admins can correct labels/routes, change state, and record summary factuality reviews. Duplicate matches are never merged automatically.

This repo has no direct institutional integration. The `/api/v1` routes are the intended integration boundary.
