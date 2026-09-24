# Model training

The default web model is the small local TF-IDF/logistic category and subcategory classifier plus Ridge priority regression. It runs without a GPU. `train_baselines.py` separately implements the required keyword-rule and TF-IDF + LinearSVC comparisons.

For the MuRIL research path, install `requirements-colab.txt` in a GPU Colab runtime and run one epoch first to check data loading, tokenization, LoRA, forward/loss/evaluation and checkpoint output. Then run the planned 3–4 epoch experiment with fixed seed and a documented split. The code requires Transformers, PEFT and access to the Hugging Face MuRIL weights. No MuRIL run or checkpoint reload has been completed in this environment. Do not call one-epoch smoke results final research results.
