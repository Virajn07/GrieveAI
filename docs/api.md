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

Legacy `/submit`, `/api/grievances/<ack_number>`, and `/admin/...` paths remain available. `GET /`, `/submit`, and `/track/<ack_number>` serve the current Jinja clients. Admins sign in at `/admin/login`; the browser session is HttpOnly and SameSite=Lax. The shared token is for demonstration use only.
