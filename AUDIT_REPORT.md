# GrieveAI repository readiness review

**Review date:** 2026-09-24
**Scope:** reproducible local setup, GitHub safety, database migration, API/UI boundary, configuration and MuRIL readiness.
**Training status:** MuRIL/LoRA training was not started.

## Verified in this checkout

- The repository was initialized locally with Git. No commit or remote was created; no push was attempted.
- The repository contains a generated synthetic CSV (1,323 rows), an empty raw-data placeholder, and a notebook with no stored cell outputs. The synthetic data scan found no email, phone or roll-number shaped values.
- A path-only scan for common provider-key formats found no matches. `.env.example` contains blank secret placeholders and example configuration only. Local SQLite files, raw files, Python caches, logs and all model checkpoints are ignored.
- Alembic revision `0001_initial` creates the `grievances` and `audit_log` tables. A new SQLite database upgraded successfully and recorded the expected revision.
- Application startup no longer calls `db.create_all()`. `DATABASE_URL` now treats unset and blank values consistently, defaulting to `app/instance/grieveai.db` in Flask and Alembic.
- The browser submission handler and REST submission handler use separate response adapters around one service in `app/services.py`. The API handler does not call the browser handler or render a template.
- A real local Flask process returned `ok` from `/api/v1/health`; `/submit` loaded; a synthetic API ticket was created and tracked; its direct identifiers were redacted before storage/response.
- `python -m unittest discover -s tests -v`: 12 tests passed. `python -m compileall -q app src tests`: passed.
- The Colab requirements now list the requested training packages. The one-epoch smoke command is consistent with the argparse options and the checkpoint reload branch in `src/train.py`; it was not executed.

## Configuration and provenance

Taxonomy, category-to-department mapping, demo SLA rules, confidence/dedup thresholds, model backend/checkpoint, database URL, optional student-name list, LLM providers, model names, endpoint URLs and timeouts use config files or environment variables. Configured category, department and SLA values are explicitly demo/proposed examples, not approved institutional policy. The provider defaults are vendor API endpoints, not VCET integrations.

Metrics derived from the synthetic dataset are pipeline checks only. Routing accuracy remains unavailable without validated gold routing labels. Override rates are observational workflow counts, not model accuracy. No VCET pilot metrics or real records are included or claimed.

## Remaining limitations

- PostgreSQL server migration and runtime were not tested here; local verification used SQLite only. Use the same Alembic command with a PostgreSQL URL and installed driver in an environment where PostgreSQL is available.
- The initial migration supports a fresh database and safely recognizes an existing database with the current schema. It is not a repair script for arbitrary manually altered schemas; inspect and back up a pilot database before adopting it.
- Admin authentication remains a shared-token demonstration control, not institutional identity management. Production use needs HTTPS, secure session-cookie settings, CSRF protection for browser mutations, and institution-managed authentication.
- Deprecation warnings remain for naive `datetime.utcnow()` usage. They did not fail tests, but should be addressed before broader deployment.
- MuRIL/LoRA dependencies beyond the checked-in Colab manifest are not installed here; there is no checkpoint, trained model or claimed training result.
- Metrics for the synthetic benchmark are not included in this review because the task was readiness verification, not model evaluation.

See [GitHub setup](docs/GITHUB_SETUP.md), [deployment](docs/deployment.md), and [model training](docs/model_training.md) for operating instructions and constraints.
