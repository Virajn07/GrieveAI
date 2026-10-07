# GrieveAI final QA report

Date: 2026-10-07

## Results

- Full suite: **PASS** — 38 passed, 1 skipped, 5 subtests passed.
- LLM report tests: **PASS** — 9 passed, 5 subtests passed.
- IndicBERT end-to-end integration: **PASS** — 1 passed with `GRIEVEAI_RUN_MODEL_INTEGRATION=1`.
- Live Ollama: **PASS** — local `qwen3:8b` responded through `/api/chat`; application/browser flow reported provider `Ollama · qwen3:8b`.
- Redaction and provider fallback: **PASS in automated tests** — outbound request checks cover redacted name, phone, email, and student identifier; failure/timeout/malformed output paths fall back. A separate browser smoke with Ollama stopped was not run.
- SHAP: **PASS** — case detail generated token attributions in the browser.
- `git diff --check`: **PASS**.

## Provider and safeguards

Provider order is local Ollama, configured OpenRouter fallback, then deterministic fallback. Ollama uses `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`) and `OLLAMA_MODEL` (default `qwen3:8b`). Grievance text is redacted before LLM use; output parsing validates the structured fields and department. Existing IndicBERT classification, confidence gating, duplicate/recurring detection, SHAP, routing, review, override, audit, authentication, and dashboard paths remain in place.

## Data and repository

Browser QA used synthetic records in an isolated database copy. The configured application database was restored to its recorded pre-QA count. No model checkpoints were modified. No credentials were added. No commit or push had been performed when this report was prepared.

## Remaining limitation

The live Ollama-unavailable browser smoke was not run. Automated connection-failure and deterministic-fallback tests pass. Validate this specific outage path during deployment if it is a release requirement.
