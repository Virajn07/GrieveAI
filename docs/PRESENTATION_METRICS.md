# GrieveAI prototype metrics for presentation

**Reviewed:** 2026-10-07  
**Primary model:** IndicBERT v2 + LoRA Run 1  
**Evidence scope:** synthetic data only. None of these figures are VCET real-world results.

## Recommended presentation table

All rows below use the same current 1,559-row synthetic corpus and the frozen duplicate-aware seed-42 split unless the row is marked †. High-priority recall in this comparison uses the raw score cutoff `priority >= 4`; this is the common definition available from each current experiment.

| Model | Category Macro-F1 | Subcategory Macro-F1 | Priority MAE | High-priority recall |
|---|---:|---:|---:|---:|
| Keyword rules | 0.5466 | 0.1830 | 0.7278 | 0.0000 |
| TF-IDF + Logistic Regression / Ridge | 0.8734 | 0.6515 | 0.5739 | 0.0000 |
| TF-IDF + Linear SVM / Ridge | 0.8641 | 0.6954 | 0.5739 | 0.0000 |
| **IndicBERT v2 + LoRA Run 1** | **1.0000** | **0.9122** | **0.5395** | **0.0000** |
| MuRIL + LoRA diagnostic† | 0.0300 | 0.0012 | 2.4027 | 0.0000 |

† The MuRIL diagnostic used an older 1,323-row dataset and a different 801/275/247 split. It is recorded for completeness and is **not an apples-to-apples comparison**. Its source and limitations are in [the diagnostic report](experiments/muril_lora_synthetic_seed42_diagnostic_results.md). The historical README/notebook claims of MuRIL 0.9859 / 0.3046 refer to a separate run without a dataset hash and are not used as comparable results.

**Best baseline:** TF-IDF + Logistic Regression has the strongest current-split baseline category Macro-F1 (0.8734). TF-IDF + Linear SVM has the strongest baseline subcategory Macro-F1 (0.6954). These are different baseline checkpoints/configurations.

**Priority definition note:** the Logistic Regression metrics artifact also records rounded, clipped app-output recall of 0.0714 and raw-score-cutoff recall of 0.0000. IndicBERT's training evaluator reports raw-score recall. Do not compare the rounded LogReg recall to the raw-score rows in the table. The raw cutoff is used above for consistency. Priority 4–5 cases are uncommon in this synthetic dataset.

## Primary IndicBERT result

The primary result is the saved Run 1 checkpoint evaluated against the held-out 338-example test partition from the current dataset. The archived `checkpoints/indicbert_lora/run1/metrics.json` is labeled `ACTUAL / RECORDED`; it does not itself include dataset hash or split size. The training notebook identifies the seed-42 922/299/338 split and evaluates the saved best checkpoint on `test_loader`. The current CSV hash matches the hash in the baseline run manifest. A fresh local evaluation using that split and the resolved base revision below **reproduced** the archived test values (minor floating-point difference in priority MAE).

| Metric | Validation, n=299 | Held-out test, n=338 | Evidence |
|---|---:|---:|---|
| Category accuracy | 1.0000 | 1.0000 | REPRODUCED |
| Category Macro-F1 | 1.0000 | 1.0000 | REPRODUCED; test also recorded |
| Subcategory Macro-F1 | 0.8634 | 0.9122 | REPRODUCED; test also recorded |
| Oracle subcategory Macro-F1 (gold parent) | 0.8634 | 0.9122 | REPRODUCED; test also recorded |
| Subcategory accuracy | 0.8829 | 0.9438 | REPRODUCED |
| Priority MAE on raw output | 0.5074 | 0.5395 | REPRODUCED; test also recorded |
| High-priority recall (raw score >= 4) | 0.0690 | 0.0000 | REPRODUCED; test also recorded |

The complete per-category, per-subcategory, per-language, and confidence-gate output is in [indicbert_seed42_reproduced.json](indicbert_seed42_reproduced.json). Labels with zero examples in a partition are marked with support 0 in that file; their precision/recall/F1 values are not evaluated and should be read as N/A, not as observed model failures.

### Per-class test findings

Category precision, recall, and F1 were 1.0000 for each of the seven categories on this synthetic test split. For subcategories, 28 of the 32 labels present in the test split had precision/recall/F1 of 1.0000. The remaining supported labels were:

| Subcategory | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `payment_transaction` | 10 | 0.5263 | 1.0000 | 0.6897 |
| `accounts_financial_documentation` | 9 | 0.0000 | 0.0000 | 0.0000 |
| `food_service` | 5 | 0.3333 | 1.0000 | 0.5000 |
| `menu_variety` | 10 | 0.0000 | 0.0000 | 0.0000 |
| `construction_facility_disruption` | 0 | N/A | N/A | N/A |

### Reproducing the evaluation

From the repository root, with the local Run 1 checkpoint and the base snapshot cached:

```powershell
python scripts/evaluate_indicbert.py --offline --output docs/indicbert_seed42_reproduced.json
```

The script checks duplicate-group leakage, uses the shared deterministic split helper, pins the observed base revision, reports validation and test scores, and prints or saves full label/language/gate details. Do not replace the archived checkpoint metrics with new results; retain both if checkpoint, dataset hash, split, or base revision changes.

## Dataset summary

| Property | Value | Evidence |
|---|---:|---|
| Total synthetic records | 1,559 | ACTUAL |
| Categories / subcategories | 7 / 33 | ACTUAL |
| English (`en`) | 526 (33.7%) | ACTUAL corpus label |
| Devanagari Hindi (`hi`) | 513 (32.9%) | ACTUAL corpus label |
| Hinglish (`hinglish`) | 520 (33.4%) | ACTUAL corpus label |
| Train / validation / test | 922 / 299 / 338 | ACTUAL frozen split |
| Duplicate-reference rows (`duplicate_of`) | 74 | ACTUAL |
| Exact repeated-text rows after the first copy | 1,004 | ACTUAL; generated synthetic duplication |
| Priority 1 / 2 / 3 / 4 / 5 | 130 / 671 / 613 / 137 / 8 | ACTUAL |

Dataset SHA-256: `02bbc1e393544efe4a1513c395813bba70d0e754e140ab5366931ae880803ea6`. Splits group duplicate references and exact normalized-text copies so these groups do not cross train/validation/test boundaries. The large number of repeated templates remains a material limitation even with group separation.

## Primary model configuration and size

| Setting | Value | Evidence |
|---|---|---|
| Base model | `ai4bharat/IndicBERTv2-MLM-only` | ACTUAL checkpoint config |
| LoRA | rank 8, alpha 16, dropout 0.1; query/value modules; feature-extraction task | ACTUAL checkpoint config |
| Learning rate / epochs / batch | 2e-4 / 10 / 16 | ACTUAL training notebook |
| Maximum sequence length | 128 | ACTUAL training notebook |
| Optimizer | AdamW; weight decay 0.01 | ACTUAL training notebook |
| Total parameters including task heads | 278,367,785 | COMPUTED from logged encoder parameters plus serialized head shapes |
| Trainable LoRA parameters | 294,912 | ACTUAL notebook log |
| Trainable task-head parameters | 31,529 | COMPUTED from serialized head tensors |
| Total trainable parameters | 326,441 (0.1173%) | COMPUTED from LoRA plus task heads |
| Local checkpoint bundle including tokenizer | 9,073,963 bytes (about 8.65 MiB) | ACTUAL local files; excludes base model |
| Base-model download/cache | 1.12 GB in notebook output | ACTUAL notebook output; not in checkpoint bundle |

The checkpoint is Run 1 at `checkpoints/indicbert_lora/run1/`. Adapter SHA-256: `C1FAEBA412A54601A08EDD70E9868C5AFD84F42B47025AC193B23BB2AE9B468A`; task-head SHA-256: `F8D2DBF019F34364DA08ADA09046D292CC7684E150D1D92CDB0927B065A27623`. The base revision used for this review/reproduction is `8598f13fe52443bc3fc054fcd665944560145b5c`. The original training artifact did not record its base revision, so byte-identical reconstruction of the training-time base cannot be proven.

The notebook saved Run 1's metrics after evaluating the held-out test loader. Its final saved values are category Macro-F1 1.0000, subcategory Macro-F1 0.9121767, oracle subcategory Macro-F1 0.9121767, priority MAE 0.5394806, and high-priority raw-score recall 0.0. The values above are marked reproduced on the current cached base snapshot; category/subcategory test metrics match, and the fresh raw priority MAE is 0.5394798.

### Separate 60K synthetic stress run

The notebook records 48,000 train / 6,000 validation / 6,000 test rows, seed 42, 3 epochs, batch 64, learning rate 2e-4, max length 128. Its held-out result is category Macro-F1 1.0000, subcategory Macro-F1 1.0000, oracle subcategory Macro-F1 1.0000, priority MAE 0.7826212, and high-priority recall 0.0000. This is a separate generated stress corpus, not the Run 1 checkpoint result or an institutional benchmark. Per-language or per-class outputs for this stress run are not available.

## Baseline comparison provenance

Keyword and TF-IDF + Linear SVM results were freshly reproduced with `python train_baselines.py --seed 42`. TF-IDF + Logistic Regression was re-evaluated from the existing saved baseline checkpoint against the same held-out split; all saved metrics matched. Baseline split manifest and dataset hash match the current corpus.

| Model | Dataset / split | Category Macro-F1 | Subcategory Macro-F1 | Priority MAE | Raw high-priority recall |
|---|---|---:|---:|---:|---:|
| Keyword rules | Current 1,559 rows; test 338 | 0.5466 | 0.1830 | 0.7278 | 0.0000 |
| TF-IDF + Logistic Regression / Ridge | Current 1,559 rows; test 338 | 0.8734 | 0.6515 | 0.5739 | 0.0000 |
| TF-IDF + Linear SVM / Ridge | Current 1,559 rows; test 338 | 0.8641 | 0.6954 | 0.5739 | 0.0000 |
| IndicBERT v2 + LoRA Run 1 | Current 1,559 rows; test 338 | 1.0000 | 0.9122 | 0.5395 | 0.0000 |
| MuRIL + LoRA diagnostic† | Older 1,323 rows; test 247 | 0.0300 | 0.0012 | 2.4027 | 0.0000 |

† Not directly comparable because it used a different dataset hash (`c982a99147589e75990361a2917465ada7aedc3ddfbbe2b6359d0c2a3ffa8dcc`) and split. No MuRIL result on the current hashed dataset/split is available. The old notebook's other MuRIL run has no recorded dataset hash/split and is excluded.

Routing accuracy is **NOT AVAILABLE**: this dataset has no validated gold department label. A production department-routing score must not be inferred from category mapping.

## Language-wise results

These are reproduced on the same synthetic held-out test split. The labels `en`, `hi`, and `hinglish` come from the synthetic corpus, not human language annotation. All three groups had category accuracy / Macro-F1 of 1.0000 in this split; these are not real-world language performance guarantees.

| Test language group | N | Category Macro-F1 | Subcategory Macro-F1 | Subcategory accuracy | Priority MAE |
|---|---:|---:|---:|---:|---:|
| English | 138 | 1.0000 | 0.8727 | 0.8623 | 0.5388 |
| Hindi / Devanagari | 98 | 1.0000 | 1.0000 | 1.0000 | 0.5491 |
| Hinglish / Romanised Hindi | 102 | 1.0000 | 1.0000 | 1.0000 | 0.5311 |

The exact language-wise subcategory accuracy is included in the JSON reproduction artifact. Do not use these small synthetic subgroup metrics as language parity evidence.

## Priority and confidence gate

On the held-out test split, 28/338 examples have gold priority 4 or 5. The model's high-priority raw score recall is 0/28. The default confidence threshold is 0.70 and is explicitly a demo setting, not a calibrated threshold.

At 0.70, 324/338 test examples are above threshold and 14/338 (4.14%) are below threshold. In this particular synthetic test partition, both groups have category accuracy and Macro-F1 1.0000; subcategory accuracy is 0.9414 above the threshold and 1.0000 below it. The below-threshold group is only 14 examples. The validation partition had 299/299 above threshold and no below-threshold examples. Therefore:

- **Confidence review rate at this test threshold:** 4.14% (REPRODUCED synthetic split).
- **Evidence that the threshold improves accuracy:** NOT AVAILABLE.
- **Threshold calibration or expected institutional review rate:** NOT AVAILABLE.

## Duplicate and recurring detection

Duplicate and recurring issue detection is implemented and is exercised by unit and end-to-end tests. Current similarity settings are duplicate 0.85, related 0.65, and recurring 0.80. The synthetic dataset contains 74 explicit duplicate-reference rows. **Formal detection precision, recall, F1, accuracy, and ground-truth cluster evaluation are NOT AVAILABLE.** Do not present the number of detected runtime clusters as detection quality.

## Explainability and system latency

SHAP is reachable through the protected `GET /api/v1/grievances/<ack_number>/explanation` endpoint and is also available as an on-demand classifier method. Three local direct-model explanations succeeded (3/3); measured mean 7.228 s, minimum 2.926 s, maximum 15.767 s. Representative tokens included `wif` / `disconnect` for a WiFi report and `ticket` / `hall` for a hall-ticket report. Token fragments appeared for code-mixed inputs. Semantic quality was not formally rated; SHAP is an explanation aid and has not been shown to improve classification.

Local timing measurements on 2026-10-07, Python 3.12.5, PyTorch 2.4.1 CPU, Transformers 4.45.1, PEFT 0.21.0, SHAP 0.51.0, scikit-learn 1.5.2; no CUDA GPU. These local-development timings are not production SLAs.

| Operation | Samples | Mean | Min | Max | Notes |
|---|---:|---:|---:|---:|---|
| Warm single-example IndicBERT prediction | 8 | 0.171 s | 0.135 s | 0.360 s | After one warm-up; synthetic examples |
| On-demand SHAP explanation | 3 | 7.228 s | 2.926 s | 15.767 s | Direct model calls; each builds the SHAP explainer |
| TF-IDF duplicate/related search | 8 | 0.0068 s | 0.0062 s | 0.0079 s | Fallback vectorizer over 200 synthetic prior rows |
| SQLite ORM insert + commit | 8 | 0.0012 s | 0.0004 s | 0.0043 s | In-memory SQLite, one minimal synthetic row per commit |
| Full `/api/v1/grievances` submit, including inference, similarity work and SQLite persistence | 8 | 1.678 s including cold load | 0.143 s | 12.093 s | First request 12.093 s loaded the model; remaining 7 averaged 0.190 s (min 0.143, max 0.399 s) |

The local API timing run persisted all 8 synthetic submissions to a temporary in-memory SQLite database. The separate insert/commit and duplicate-search timings are isolated microbenchmarks, not portions subtracted from the full endpoint time. PostgreSQL latency was not measured. The first request includes model initialization; warm API latency is more representative of an already-loaded local process.

## OpenRouter status

OpenRouter is optional, uses environment variables, and receives the PII-redacted grievance text. Responses are schema-checked; provider errors or invalid responses fall back deterministically and do not interrupt submission. Success, invalid-response, timeout/failure and redaction behavior are covered by mocked tests. **No live OpenRouter key/request was used** in this review. The generated analysis is advisory and cannot overwrite the classifier result or deterministic department route.

## Safe claims for the presentation

### Supported by experiment or implementation

- IndicBERT v2 + LoRA was evaluated on generated synthetic grievance data; the Run 1 held-out test scores above were reproduced on the current frozen split and cached base snapshot.
- The demo flow accepts English, Devanagari Hindi and Romanised Hinglish inputs; this is also covered by the synthetic real-model integration test.
- Low-confidence cases can be queued for administrator review; an administrator can accept or correct a result while preserving the original prediction and writing an audit event.
- The admin interface can request token explanations, inspect duplicate/recurring context and show database-backed demo workload counts.
- OpenRouter summaries are optional, PII-redacted, validated, advisory, mocked in tests and have a deterministic fallback.

### Not yet validated

- Real VCET accuracy, category/subcategory/priority or department-routing performance.
- Student satisfaction, reduction in routing time, grievance-resolution outcomes, or institutional impact.
- Production-scale throughput, availability, security, PostgreSQL latency, model memory use on deployment hardware, or any production SLA.
- Confidence calibration, SHAP explanation quality, or duplicate/recurring precision/recall/F1.
- Any real-world benefit from using IndicBERT, SHAP, confidence gating, duplicate detection or LLM summaries.

## Final metric availability checklist

| Requested metric | Status |
|---|---|
| Primary test Category Macro-F1 | **REPRODUCED:** 1.0000, synthetic test n=338 |
| Primary test Subcategory Macro-F1 | **REPRODUCED:** 0.9122, synthetic test n=338 |
| Oracle Subcategory Macro-F1 | **REPRODUCED:** 0.9122, synthetic test n=338 |
| Priority MAE | **REPRODUCED:** 0.5395, raw 1–5 scale |
| High-priority recall | **REPRODUCED:** 0.0000 at raw score >=4 |
| Trainable parameters | **COMPUTED from actual notebook/artifacts:** 326,441 total (LoRA + heads) |
| Best baseline | **REPRODUCED:** TF-IDF + Logistic Regression category Macro-F1 0.8734; TF-IDF + SVM subcategory Macro-F1 0.6954 |
| Language-wise results | **REPRODUCED on synthetic test groups:** see table and JSON |
| Average inference latency | **MEASURED locally:** 0.171 s, n=8 warm examples |
| SHAP latency | **MEASURED locally:** 7.228 s mean, n=3 |
| Duplicate detection metrics | **NOT AVAILABLE:** no formal ground-truth evaluation |
| Confidence-gate metrics | **REPRODUCED descriptively:** 14/338 below 0.70; threshold not calibrated |
| Real VCET metrics | **NOT AVAILABLE** |
