# MuRIL + LoRA one-epoch diagnostic results

This is a synthetic-data development experiment, not VCET performance. It ran from source commit `f67a1947da6d4b749e0e041150ceb962916c55af` using the checked-in dataset and taxonomy. The only notebook-runtime adjustments were upgrading the preinstalled incompatible optional `torchao` package and saving artifacts under a separate temporary Colab directory; the training source was unchanged.

## Reproducibility

- Dataset: `data/processed/grievances_synthetic.csv`, 1,323 rows, seven categories and 28 subcategories.
- Frozen duplicate-aware split: 801 train / 275 validation / 247 test; no duplicate-group leakage.
- GPU: Tesla T4; Python 3.13.15; PyTorch 2.11.0+cu128 / CUDA 12.8; Transformers 5.16.1; PEFT 0.20.0; scikit-learn 1.6.1; pandas 2.2.3; torchao 0.18.0.
- LoRA: rank 8, alpha 16, dropout 0.1, query/value targets; 322,596 trainable parameters.
- Training: seed 42, batch 8, max length 128, AdamW, LR 2e-4, weight decay 0.01; one epoch; cross-entropy/focal gamma 0 for category and gold-parent-masked subcategory; 0.5-weighted Huber priority loss.
- Checkpoint and experiment manifest were written separately to `/content/GrieveAI/checkpoints/diagnostic_1_epoch/` in the Colab runtime. Google Drive mount failed, so those runtime files are temporary. This report preserves the metrics and run configuration in the repository.

## Baseline and MuRIL metrics

Metrics use the same frozen split. The TF-IDF + SVM baseline values are reproduced by `train_baselines.py` in the same Colab run.

| Model / split | Category Macro-F1 | Subcategory Macro-F1 | Subcategory Macro-F1 given gold parent | Priority MAE | High-priority recall |
|---|---:|---:|---:|---:|---:|
| TF-IDF + SVM / validation | 0.85718 | 0.74366 | — | 0.60099 | 0.15000 |
| TF-IDF + SVM / test | 0.83249 | 0.76003 | — | 0.54557 | 0.15217 |
| MuRIL + LoRA / validation | 0.03941 | 0.00158 | 0.08149 | 2.40476 | 0.00000 |
| MuRIL + LoRA / test | 0.03002 | 0.00123 | 0.06491 | 2.40269 | 0.00000 |

Mean training loss for the canonical epoch was 4.3969. The checkpoint saved and reloaded successfully. A sample prediction was `Examinations / marks_discrepancy / priority 1`.

## Diagnostic findings

- Taxonomy validation passed: all dataset category/subcategory pairs map to one configured parent; each category has four children. The same loaded taxonomy order is used for label indices and model heads.
- Forward pass and gold-parent subcategory mask assertions passed; the tokenizer's `token_type_ids` are accepted and forwarded.
- After the one-epoch checkpoint, category predictions collapsed to `Examinations` on every train, validation, and test row. This explains the near-zero macro-F1. Gold-parent subcategory F1 remained low, so category mistakes alone do not explain the child-label failure.
- On a representative training batch, the saved model had nonzero gradients through 48 LoRA gradient tensors and the task heads. Trainable parameter count and checkpoint reload were correct.
- A seeded instrumented repeat recorded 101 batches. Category loss moved from 1.9457 for the first ten to 1.9433 for the last ten; masked subcategory loss moved from 1.3842 to 1.3742. Total loss was flat (4.4035 to 4.4292). The last batch had non-finite gradients, was skipped by GradScaler, and halved the scale from 65,536 to 32,768; the other 100 batches were finite.
- A full-precision one-epoch comparison had zero non-finite gradient batches and reproduced the collapse and nearly identical losses. Thus AMP overflow is a real isolated issue but is not the main cause.
- A diagnostic with task-head LR 1e-3 and LoRA LR 2e-4 (FP32) improved priority MAE, but category F1 remained 0.02978 on validation and subcategory F1 0.00158. It did not support changing the shared-head LR as a sufficient fix.

## Follow-up representation diagnostics

- A frozen MuRIL linear probe trained on the 801-row training split yielded pooler and CLS category Macro-F1 of 0.03061 validation / 0.02337 test. Masked-mean pooling was higher but still weak at 0.14804 / 0.13040.
- A separate one-epoch masked-mean pooling ablation used the same seed, split, optimizer, learning rate, losses and batch size, with a separate checkpoint directory `/content/GrieveAI/checkpoints/ablation_mean_pooling/`. It produced category Macro-F1 0.03941 validation / 0.03002 test; subcategory Macro-F1 0.00158 / 0.00123; gold-parent subcategory Macro-F1 0.11207 / 0.10352; priority MAE 2.40982 / 2.40759; high-priority recall 0. The checkpoint reloaded successfully.
- The pooling probe suggests some distinction in token-average features, but the full ablation did not improve category F1 and remained near-random. Pooling is therefore not a sufficient root cause and was not changed in the repository.

## Decision

The controlled one-epoch run is not learning the classification tasks sensibly and is far below the TF-IDF + SVM synthetic baseline. No three-epoch run was started. Mapping, mask, forward pass, gradients, and checkpoint reload checks did not reveal a label-pipeline defect. The available evidence rules out taxonomy/index mismatch, inference reload, AMP overflow, a too-low task-head LR, and pooling alone as sufficient explanations. The best-supported diagnosis is that this MuRIL + LoRA representation/training setup is failing to form category-discriminative features on this synthetic corpus: the category predictions collapse despite valid labels, masks, gradients and substantial parameter updates, while frozen linear probes are also weak. The exact cause within that representation/adaptation path remains unresolved. No speculative model or architecture change was made. Next, instrument category/subcategory head updates and per-task logits/accuracy by batch, then run a preregistered optimizer/objective diagnostic before any longer training.

## Runtime warnings

PEFT initially could not initialize because the Colab image had preinstalled `torchao` 0.10.0, which is below PEFT 0.20.0's accepted version. Upgrading to 0.18.0 allowed the forward pass and training; `requirements-colab.txt` now declares the minimum. That build emitted optional compiled-kernel warnings that did not stop training. Transformers also reported unused masked-language-model head weights while loading the encoder, which is expected for a different downstream task. The runtime used unauthenticated public Hugging Face downloads and emitted a deprecated GradScaler API warning.
