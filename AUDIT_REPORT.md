# GrieveAI integration and IndicBERT review

**Review date:** 2026-10-06
**Branch:** `codex/grieveai-review-integration`

## Asset audit

- The supplied `GrieveAI.zip` contains `checkpoints/archives/grieveai_indicbert_v2_run1.zip`. Its ZIP integrity check passed. The inner archive contains the saved fast-tokenizer JSON/config, `adapter/adapter_model.safetensors`, `adapter/adapter_config.json`, `heads.pt`, `taxonomy.json`, and `metrics.json`.
- The trained model is `ai4bharat/IndicBERTv2-MLM-only` with PEFT LoRA: rank 8, alpha 16, dropout 0.1, `query`/`value` target modules, feature-extraction task type, no bias. The notebook specifies max length 128, masked mean pooling over `last_hidden_state`, and category, subcategory, and scalar priority heads. It masks the subcategory logits using the predicted category at inference.
- Checkpoint taxonomy matches `config/taxonomy.json` exactly: 7 categories, 33 subcategories. The tokenizer was saved alongside the adapter; inference uses the serialized tokenizer JSON to support the archive’s `TokenizersBackend` format on the installed Transformers version.
- `heads.pt` and the LoRA adapter are the trained task-specific weights. The base transformer weights are not in the supplied checkpoint; they are fetched from/cached from the Hugging Face model repository. The training files did not pin or record a base-model revision. This verification loaded current Hub snapshot `8598f13fe52443bc3fc054fcd665944560145b5c`; each prediction records its short revision in `model_version`.
- The same project ZIP also contains a MuRIL adapter archive and a separate 60K stress-test archive (about 1.03 GB compressed, with a 1.11 GB `best_model.pt`). The deployed primary is the Run 1 IndicBERT checkpoint because it is the stated best experiment; no transformer retraining was performed.
- Run 1 `metrics.json` reports category Macro-F1 **1.000**, subcategory Macro-F1 **0.9122**, priority MAE **0.5395**. These are synthetic-only results.

## Integration completed

- Added `src/indicbert_inference.py`, a cached CPU/GPU loader that validates model ID, LoRA settings, taxonomy label order and task-head state. It reproduces the notebook’s pooling and hierarchical mask and returns category, subcategory, rounded/raw priority, confidence and full model version.
- `src/inference.py` now defaults to IndicBERT. TF-IDF is available only when explicitly selected with `MODEL_BACKEND=baseline`; it is also used solely as a fallback vectorizer for duplicate/recurring similarity when optional sentence embeddings are unavailable. A model-load failure is visible through health and submission responses.
- The preprocessing path detects language/script before redaction, then sends only PII-redacted text to model inference and persistence. Hindi, English and Hinglish examples were exercised.
- Existing SQLAlchemy schema retains original model labels in `predicted_category`, `predicted_subcategory`, and `predicted_priority`; administrator corrections update current labels separately. The model version, confidence/gate, duplicate/related matches, recurring group and advisory analysis are stored without replacing the original prediction.
- The Flask submission flow uses IndicBERT, confidence review, duplicate and recurring detection, DB persistence, and deterministic department routing. OpenRouter remains advisory and falls back without credentials. Dashboard shows prediction vs corrected labels, confidence/review, model version, department, similarity context and structured LLM advisory. SHAP explanation is on demand and bound to the selected classifier.
- Added local setup and checkpoint instructions. Adapter files are extracted into the ignored local checkpoint folder and are not committed, following `.gitignore` policy.

## Verification

- Real IndicBERT inference loaded on CPU with the local adapter and downloaded base model. The requested Hinglish example produced `Academics / attendance`, priority 2, confidence **0.884108**, model `indicbert_lora_run1+base-8598f13f`.
- An opt-in real-model Flask integration test submitted seven English/Hindi/Hinglish examples, verified all seven expected top-level categories, confidence gate, duplicate/recurring grouping, persistence across a fresh Flask app instance, administrator correction with original prediction preserved, audit record, health status and dashboard model version.
- Normal offline tests cover migrations, confidence gate, API, classifier identity, label masking, duplicate/recurring behavior and mocked/no-key LLM fallback. See final test output in the review task.
- Database integration was exercised with SQLite. PostgreSQL server runtime was not available.

## Remaining limitations

- The base-model revision used during the original training is not recorded, so this integration cannot prove byte-identical reconstruction of the original base weights. It verifies with the currently available Hub snapshot and records that resolved revision per prediction.
- Synthetic metrics do not establish real institutional performance. Real institutional data was not available/confirmed for this project phase. The 0.70 confidence gate is a configurable demo setting, not a scientifically tuned threshold.
- OpenRouter was tested through mocked behavior and deterministic no-key fallback; no live API key was used. A live on-demand SHAP explanation was generated from the actual IndicBERT checkpoint after disabling Transformers’ unused optional TensorFlow backend.
- No PostgreSQL instance was available for runtime verification. Sentence-transformer embeddings are optional; the production workflow has a TF-IDF similarity fallback independent of the selected classifier.
