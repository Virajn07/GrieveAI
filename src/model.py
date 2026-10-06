"""
GrieveAI classifier model.

Architecture:
    MuRIL encoder (google/muril-base-cased), LoRA-adapted
      -> pooled [CLS] representation
      -> category_head      (7-way:  Academics, Examinations, Fees_Accounts,
                                       IT_Library, Infrastructure, Transport, Canteen)
      -> subcategory_head    (28-way: flat over all sub-categories; at inference
                                       time we mask logits to only the sub-categories
                                       valid under the predicted category)
      -> priority_head       (1-dim regression, target range 1-5)

Only the LoRA adapter weights and the three heads are trained; the base
MuRIL weights stay frozen. This is what makes fine-tuning feasible on a
free-tier Colab GPU.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

MURIL_CHECKPOINT = "google/muril-base-cased"


def load_taxonomy(path="config/taxonomy.json"):
    with open(path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    categories = list(taxonomy["categories"].keys())
    subcategories = []
    subcat_to_category = {}
    category_to_subcat_indices = {cat: [] for cat in categories}
    for cat in categories:
        for sub in taxonomy["categories"][cat]["subcategories"]:
            idx = len(subcategories)
            subcategories.append(sub)
            subcat_to_category[sub] = cat
            category_to_subcat_indices[cat].append(idx)
    return {
        "categories": categories,
        "subcategories": subcategories,
        "subcat_to_category": subcat_to_category,
        "category_to_subcat_indices": category_to_subcat_indices,
    }


class FocalLoss(nn.Module):
    """Focuses training on hard/rare examples - see Lin et al. 2017.
    Reduces to standard cross-entropy when gamma=0."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce)
        return loss.mean()


class GrieveAIClassifier(nn.Module):
    def __init__(self, num_categories, num_subcategories, encoder_name=MURIL_CHECKPOINT,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, freeze_base=True):
        super().__init__()

        base_encoder = AutoModel.from_pretrained(encoder_name)

        if freeze_base:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "value"],  # MuRIL is BERT-architecture
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.encoder = get_peft_model(base_encoder, lora_config)
        else:
            self.encoder = base_encoder

        hidden_size = base_encoder.config.hidden_size

        self.category_head = nn.Linear(hidden_size, num_categories)
        self.subcategory_head = nn.Linear(hidden_size, num_subcategories)
        self.priority_head = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # MuRIL/BERT: use the pooler_output (tanh-activated [CLS]) when available,
        # else mean-pool the last hidden state over non-padding tokens.
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        pooled = self.dropout(pooled)

        category_logits = self.category_head(pooled)
        subcategory_logits = self.subcategory_head(pooled)
        priority_pred = self.priority_head(pooled).squeeze(-1)

        return {
            "category_logits": category_logits,
            "subcategory_logits": subcategory_logits,
            "priority_pred": priority_pred,
            "pooled": pooled,  # exposed for SHAP / embedding-based dedup later
        }

    def mask_subcategory_logits(self, subcategory_logits, predicted_category_idx, taxonomy):
        """At inference, zero out sub-category logits that don't belong to the
        predicted top-level category, so the model can't e.g. predict a
        Canteen sub-category for a grievance it classified as Transport."""
        valid_indices = taxonomy["category_to_subcat_indices"][taxonomy["categories"][predicted_category_idx]]
        mask = torch.full_like(subcategory_logits, float("-inf"))
        mask[..., valid_indices] = subcategory_logits[..., valid_indices]
        return mask

    def trainable_parameters(self):
        """Returns only the parameters that actually get updated: LoRA
        adapter weights + the three heads. Useful for sanity-checking that
        the base MuRIL weights are indeed frozen before a training run."""
        params = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                params.append((n, p.numel()))
        return params
