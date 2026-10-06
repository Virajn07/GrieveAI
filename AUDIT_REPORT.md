# GrieveAI integration review

**Review date:** 2026-10-06
**Review branch:** `codex/grieveai-review-integration`
**Scope:** merge the local project snapshot with Somesh2006/GrieveAI, make inference and workflow coherent, evaluate the baseline using leakage-safe synthetic-data splits, and verify the integrated app.

## Repository and merge

- The working project was a source snapshot without Git history. It was preserved as the first commit in an isolated review repository before integration.
- The friend repository was fetched as remote `friend`, branch `member-backend`, at `de77f47fd1516e30af60ac2b321e1072ddda4c58`.
- The two repositories had unrelated histories and overlapping files. The friend backend was selected for overlapping backend files; non-overlapping local project materials were retained. The original snapshot commit remains available in this branch history.
- The integrated result is on `codex/grieveai-review-integration`; no push or force-push was performed.

## Implemented

- Added one redacting, language-aware inference API for application use, model-version tracking, configurable confidence review, and lazy model loading.
- Kept deterministic category-to-department routing in control. OpenRouter analysis is structured, validated, and advisory; unavailable credentials or invalid responses use a deterministic fallback.
- Added similarity matching, recurring-cluster context, optional embedding support with TF-IDF fallback, and database migration `0003_inference_context` for model and analysis context.
- Added on-demand feature explanations, additional India-oriented identifier redaction, and separate storage for model predictions versus administrator corrections.
- Reworked baseline training to use shared duplicate-aware train/validation/test splits. Retrained the TF-IDF baseline using the pinned scikit-learn version compatible with the checked-in model format.
- Added synthetic-corpus and app-workflow analytics to the API and dashboard, explicitly separating training corpus descriptions from real workflow counts.
- Updated setup, database, deployment, evaluation, architecture, schema, and priority-analysis documentation.

## Verification

- `python -m unittest discover -s tests -v`: **26 tests passed** after the final fix.
- `python -m compileall -q app src tests`: passed.
- `git diff --check`: passed.
- The suite exercises a SQLite migration from an empty database, submission, redaction, tracking, audit, routing, admin correction, analytics and dashboard rendering. LLM behavior is tested with mocked responses and deterministic no-key fallback.
- The synthetic split contains 922 train, 299 validation, and 338 test examples, with no exact/linked duplicate overlap between splits.
- Held-out test metrics: category accuracy **0.8550**, category macro F1 **0.8734**, subcategory macro F1 **0.6515**, raw priority MAE **0.5739**. At priority >=4, only 28/338 examples are positive; rounded-output recall is **0.0714** (3/28), with precision **0.6667**. These generated-data results do not establish real-world performance.

## Remaining limitations

- The source materials do not contain the requested IndicBERT v2 + LoRA checkpoint or its compatible inference implementation. This integration therefore uses the runnable TF-IDF baseline; a MuRIL adapter exists but was not validated against a supplied trained checkpoint. The requested transformer-specific production inference and SHAP analysis cannot be verified without those assets.
- The checked-in benchmark is synthetic and extremely imbalanced for high priorities. The model is a demonstration, not a validated grievance triage system. Never use these metrics to claim VCET performance.
- Only SQLite runtime and migration were exercised. PostgreSQL needs a separately configured server/driver validation.
- OpenRouter was mocked; no external API request was made. Sentence-transformer behavior is optional and was not tested with downloaded weights; TF-IDF fallback is covered.
- No VCET records were imported. The static synthetic dataset has no timestamps, so it cannot provide daily trends. The dashboard marks workflow status counts as unavailable for corpus data.

## Next steps for a maintainer

1. Review this branch against the original snapshot and friend remote; keep the branches separate until the integration is accepted.
2. Supply the approved IndicBERT/LoRA checkpoint, training configuration, tokenizer files, and model license; implement and validate its loader behind the inference interface.
3. Evaluate on a separately collected and institution-approved labeled set before enabling operational routing; have staff approve taxonomy, department mapping, priority policy, and thresholds.
4. Run a PostgreSQL deployment check and configure production identity, HTTPS, CSRF protections, backups, logging, and secret management before any public deployment.
