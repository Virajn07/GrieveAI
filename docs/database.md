# Database configuration

GrieveAI uses SQLAlchemy with Alembic migrations. SQLite remains the local default; when `DATABASE_URL` is empty, the app uses `app/instance/grieveai.db`.

For a PostgreSQL pilot, install the application dependencies (which include `psycopg[binary]`) and set a connection URL without committing credentials:

```dotenv
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require
```

URL-encode reserved characters in usernames or passwords. Keep the URL in the deployment secret store or an untracked `.env` file. Do not point a real VCET database at the synthetic demo dataset.

Initialize or upgrade either database with:

```text
alembic upgrade head
```

The migrations add nullable JSON model explanations, structured LLM analysis and similarity context, a model-version field, a recurring-cluster identifier, and indexes for department, category, priority, duplicate reference, recurring groups and SLA deadline. Existing grievance text is the PII-redacted text produced by the submission pipeline; the raw request text is not persisted. Original model labels remain separate from final category/subcategory/priority, and human corrections plus route/status events remain in `audit_log`.
