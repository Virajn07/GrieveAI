# Evaluation protocol

`train_baseline.py` evaluates the local TF-IDF + Logistic Regression classifier on a duplicate-group-aware train/validation split. `train_baselines.py` compares keyword rules with TF-IDF + Linear SVM. `src/train.py` trains MuRIL + LoRA. Current synthetic outputs are pipeline checks, not VCET pilot results.

Report category and subcategory Macro-F1, high-priority recall (priority ≥ 4), and priority MAE. Routing accuracy requires a validated gold department column. The live app's override-rate denominator is automatically routed tickets (manual-review tickets are excluded); the numerator is distinct automatically routed tickets with at least one recorded route override. Both must remain unavailable when their evidence/denominator is absent. Also collect summary factuality ratings and submission/prediction/routing timestamps before reporting factuality or time-to-route results.

Tune `CONFIDENCE_THRESHOLD` only on validation data; freeze it before held-out testing. Report language-specific results and data provenance. Keep duplicate groups together across partitions. Never present synthetic scores as VCET evidence.
