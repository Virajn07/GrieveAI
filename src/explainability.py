"""
SHAP explainability wrapper (Week 3).

Plan:
    - Wrap the trained GrieveAIClassifier's category prediction in a
      shap.Explainer using shap's Text masker, so token attributions are
      computed directly over the tokenizer's own tokens - meaning the
      highlighted words are shown in whatever script (Hindi/Hinglish/
      English) the grievance was originally written in, with no
      back-translation step.
    - Render as red/orange/green highlights in the dashboard template,
      matching the scheme from the original design doc: red = strongest
      signal for the predicted category, orange = context, green = neutral.
"""

# TODO: implement once src/train.py has produced a checkpoint to explain.
# Rough shape of what this will look like:
#
# import shap
# from transformers import AutoTokenizer
#
# def build_explainer(model, tokenizer):
#     def predict_fn(texts):
#         ...tokenize, run model.category_head, return softmax probs...
#     masker = shap.maskers.Text(tokenizer)
#     return shap.Explainer(predict_fn, masker)
#
# def explain_grievance(explainer, text):
#     shap_values = explainer([text])
#     return shap_values  # token -> attribution score, per category
