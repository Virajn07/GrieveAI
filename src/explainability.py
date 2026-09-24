"""Explainability adapters.

For the local baseline, exact linear coefficients are used. For the MuRIL
backend, SHAP can be enabled with a text masker. The wrapper deliberately
returns a small token/importance list suitable for the admin dashboard.
"""

from __future__ import annotations


def build_transformer_shap_explainer(model, tokenizer, device="cpu", max_evals=120):
    import numpy as np
    import shap
    import torch

    def predict_fn(texts):
        enc = tokenizer(
            list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            return torch.softmax(out["category_logits"], dim=-1).detach().cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    return shap.Explainer(predict_fn, masker, algorithm="partition", max_evals=max_evals)


def explain_transformer(explainer, text, predicted_class_idx, top_k=8):
    values = explainer([text])
    tokens = list(values.data[0])
    scores = values.values[0]
    if scores.ndim == 2:
        pred_scores = scores[:, int(predicted_class_idx)]
    else:
        pred_scores = scores
    ranked = sorted(zip(tokens, pred_scores), key=lambda x: abs(float(x[1])), reverse=True)
    return [(str(tok), round(float(score), 4)) for tok, score in ranked[:top_k]]
