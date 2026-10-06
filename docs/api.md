# REST API

All request bodies are JSON unless noted. Grievance text is limited to 2,000 characters. Admin routes require `X-ADMIN-TOKEN` matching the configured `ADMIN_TOKEN`; admin operations return 503 when no token is configured and 401 for an invalid token.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Database/service health. |
| POST | `/api/v1/grievances` | Submit `{ "text": "..." }`; returns acknowledgement, prediction, gate decision and tracking URL. |
| GET | `/api/v1/grievances/<ack_number>` | Public ticket status and routing summary. |
| POST | `/api/v1/grievances/<ack_number>/status` | Authorized state change; accepts `status`, optional `actor`, `note`. |
| POST | `/api/v1/grievances/<ack_number>/override` | Authorized route override; accepts configured `department`, optional `actor`, `note`. |
| POST | `/api/v1/grievances/<ack_number>/classification` | Authorized category/subcategory/priority correction; taxonomy validated and audited. |
| POST | `/api/v1/grievances/<ack_number>/summary-review` | Authorized factuality rating (`factual`, `partially_factual`, `inaccurate`) and optional note. |
| GET | `/api/v1/grievances/<ack_number>/explanation` | Authorized on-demand feature/token explanation; failures return unavailable without affecting tickets. |
| GET | `/api/v1/review-queue` | Authorized low-confidence inbox. |
| GET | `/api/v1/admin/metrics` | Observed counts and override rate; routing accuracy stays null until gold labels exist. |
| GET | `/api/v1/admin/analytics` | Authorized database-backed workload counts, category/priority distributions, high-priority/review/duplicate counts and recurring clusters. |

`GET /` is the administrator's database-backed inbox and dashboard; `GET /track/<ack_number>` is the student tracking page. `GET /api/v1/grievances/<ack_number>` exposes ticket status, prediction/current labels, confidence, model version, duplicate and recurring references, and routing. The separate analytics endpoint includes recurring group occurrence counts. Admins can accept the stored prediction or correct its labels through the same classification endpoint: posting the unchanged predicted labels accepts them, clears the review flag and writes an acceptance audit event; changed labels preserve the original `predicted_*` values and write a correction audit event.

Legacy `/submit`, `/api/grievances/<ack_number>`, and `/admin/...` paths remain available. Admins sign in at `/admin/login`; the browser session is HttpOnly and SameSite=Lax. The shared token is for demonstration use only.
