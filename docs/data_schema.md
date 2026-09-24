# Data schema and pilot import contract

The SQLAlchemy `Grievance` table stores redacted text, acknowledgement number, language/script metadata, original predicted and currently approved labels, priority, confidence and gate threshold, current/model department, duplicate candidate, summary factuality review, timestamps and status. `AuditLog` stores actor, event, timestamp and concise change detail; raw grievance text should not be written to event details.

Suggested pilot CSV columns: `text,language,category,subcategory,priority,department,timestamp,ticket_id`. Labels can be blank when unknown and must not be inferred to fill a research target. Before any import, obtain authorization, remove direct identifiers, document provenance, and validate labels with the grievance cell. No real VCET sample data is checked in.

The checked-in `data/processed/grievances_synthetic.csv` is synthetic/demo only and must never be represented as institutional records.
