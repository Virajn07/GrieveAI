# GrieveAI - implementation review and fast-build decisions

> This document records findings and implementation decisions from the initial
> source review. For the current test, migration, Git hygiene and training
> readiness status, see [AUDIT_REPORT.md](AUDIT_REPORT.md).

## Source review

The revised project plan is directionally correct: it names triage/routing in the title, defines Hindi-English code-mixed input, makes VCET the target workflow, and places the contribution around evaluation + workflow rather than Flask/LoRA alone.

## Critical gaps found in the original ZIP

1. `src/language_id.py` was a stub rather than fastText + robust Hinglish detection.
2. `src/explainability.py` was only a TODO; there was no real SHAP path.
3. Duplicate detection used TF-IDF, despite the plan calling for sentence-embedding similarity.
4. The local baseline treated priority as 1-5 classification, while the plan calls for priority regression.
5. The MuRIL training script saved the whole model state, including frozen base weights, instead of a compact PEFT adapter + heads.
6. There was no MuRIL inference loader, so the Flask application could not actually swap to the trained model without additional implementation.
7. Evaluation did not report macro-F1 and did not prevent duplicate-group leakage between splits.
8. Ticketing had submission/audit creation but no status-tracking or admin status/override workflow.
9. Routing was effectively category-to-department mapping, not an LLM decision. The implementation now makes that explicit and keeps routing deterministic.
10. The dashboard was read-only and did not expose a meaningful override workflow.
11. The implementation roadmap in the PDF assumed a real 4-6 week VCET pilot; until access/sign-off is obtained, this should be presented as the intended pilot plan, not as completed work.
12. The workflow diagram in the PDF mentions voice/Marathi, while the written scope makes Marathi a stretch addition and defines English/Devanagari Hindi/Romanised Hinglish as the core scope. The MVP now follows the written scope and leaves Marathi/voice as stretch features.

## What was implemented in this revised repository

- Privacy-first redaction for email, phone and roll numbers, with optional institution-supplied name list.
- Hinglish detection heuristic plus optional fastText augmentation.
- Stronger TF-IDF local baseline with hierarchical sub-category selection and Ridge priority regression.
- Validation-based confidence threshold calibration.
- Duplicate-aware group split for model evaluation.
- Classical keyword-rules and TF-IDF+SVM comparison script.
- MuRIL+LoRA compact checkpoint saving (adapter + tokenizer + heads + metrics).
- MuRIL inference loader with hierarchy masking.
- SHAP wrapper for the transformer path.
- Optional multilingual sentence-embedding duplicate detection with a local TF-IDF fallback.
- Deterministic department routing and optional LLM factual summarisation.
- Student submission page, acknowledgement number, tracking page, admin dashboard, status changes, override log and audit trail.
- Admin token hook for protected workflow actions.
- Separate local and Colab requirement files.

## What still requires VCET / research data

- Final seven-category / sub-category taxonomy approval.
- Ground-truth department mapping and SLA values.
- Human-annotated priority labels.
- Human-annotated routing labels / gold department.
- Real VCET validation/test data.
- Live-pilot timing study and human review/override measurements.
- Human review of LLM summary factuality.

Synthetic data is only a bootstrapping/demo dataset. Its scores must not be presented as evidence of real institutional performance.

## Fast execution order

### Stage 1 - demo-ready (today)

```bash
pip install -r requirements-local.txt
python generate_synthetic_grievances.py --per_combo 15
python train_baseline.py
python train_baselines.py
python -m app.app
```

### Stage 2 - research model (Colab T4)

```bash
pip install -r requirements-colab.txt
python src/train.py \
  --data data/processed/grievances_synthetic.csv \
  --taxonomy config/taxonomy.json \
  --epochs 4 \
  --batch_size 16 \
  --output_dir checkpoints/muril_lora/run1
```

Then configure `MODEL_BACKEND=muril` and `MURIL_CHECKPOINT_DIR=...`.

### Stage 3 - actual major-project evidence

Replace/augment synthetic data with de-identified VCET grievances, have at least two annotators label category/sub-category/priority/department, calculate annotator agreement, freeze a held-out test set, tune the confidence threshold only on validation data, and then run the live-pilot timing/override study.

## Recommended claim language

Use: "GrieveAI implements explainable, confidence-aware Hindi-English code-mixed grievance triage and department routing for a VCET case-study workflow."

Avoid claiming a new algorithm solely because MuRIL, LoRA, SHAP, Flask or an LLM are used. The defensible project contribution is the end-to-end workflow and its measured evaluation in the target institutional setting.
