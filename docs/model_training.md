# Model training

The Flask application uses the already-trained IndicBERT v2 + LoRA Run 1 checkpoint as its primary model. Do not retrain it for normal integration or inference. The archive contains the PEFT adapter, tokenizer, three task heads, taxonomy, and synthetic test metrics; it does not contain the base transformer weights. The base model `ai4bharat/IndicBERTv2-MLM-only` must be downloaded from Hugging Face on first load or already exist in the local Hugging Face cache. See the root README for the local checkpoint directory and environment settings.

The TF-IDF + Logistic Regression category/subcategory model and Ridge priority regressor are a comparison baseline, explicitly selected with `MODEL_BACKEND=baseline`. `train_baseline.py` trains that baseline, while `train_baselines.py` separately implements keyword-rule and TF-IDF + LinearSVC comparisons.

The secondary MuRIL research path remains available. Its experiment and training module are retained for comparison; it is not selected by default. Current performance figures are from synthetic data only and are not VCET results.
