"""MuRIL + LoRA multi-task classifier.

Heads:
    category (7-way), sub-category (33-way), priority (1-5 regression).

The sub-category head is masked by the predicted top-level category at
inference so an invalid hierarchy cannot be returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer


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
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce

        if self.weight is not None:
            sample_weight = self.weight[targets]
            loss = loss * sample_weight

        return loss.mean()


class GrieveAIClassifier(nn.Module):
    def __init__(
        self,
        num_categories,
        num_subcategories,
        encoder_name=MURIL_CHECKPOINT,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()

        base_encoder = AutoModel.from_pretrained(encoder_name)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none",
        )

        self.encoder = get_peft_model(base_encoder, lora_config)

        hidden_size = base_encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.category_head = nn.Linear(
            hidden_size,
            num_categories,
        )

        self.subcategory_head = nn.Linear(
            hidden_size,
            num_subcategories,
        )

        self.priority_head = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        **_ignored_extra_tokenizer_fields,
    ):
        # Preserve BERT-style segment IDs when supplied.
        # Ignore unrelated tokenizer metadata so the model remains
        # compatible with tokenizer output.
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)

        # ---------------------------------------------------------
        # IMPORTANT:
        # Always use masked mean pooling.
        #
        # We intentionally do NOT use pooler_output here because
        # the tiny-set diagnostics showed that MuRIL's pooler
        # representation was much less useful for this classifier.
        # ---------------------------------------------------------
        last_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1e-9)

        pooled = self.dropout(pooled)

        return {
            "category_logits": self.category_head(pooled),
            "subcategory_logits": self.subcategory_head(pooled),
            "priority_pred": self.priority_head(pooled).squeeze(-1),
            "pooled": pooled,
        }

    @staticmethod
    def mask_subcategory_logits(
        subcategory_logits,
        predicted_category_idx,
        taxonomy,
    ):
        valid_indices = taxonomy["category_to_subcat_indices"][
            taxonomy["categories"][predicted_category_idx]
        ]

        mask = torch.full_like(
            subcategory_logits,
            float("-inf"),
        )

        mask[..., valid_indices] = subcategory_logits[..., valid_indices]

        return mask

    def trainable_parameters(self):
        return [
            (name, param.numel())
            for name, param in self.named_parameters()
            if param.requires_grad
        ]


class MuRILInference:
    """Loads a compact PEFT adapter + classifier heads for Flask inference."""

    def __init__(
        self,
        checkpoint_dir,
        taxonomy_path="config/taxonomy.json",
        device=None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)

        self.taxonomy = load_taxonomy(taxonomy_path)

        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        # Load tokenizer saved alongside the trained model.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_dir / "tokenizer"
        )

        # Load base MuRIL encoder.
        base = AutoModel.from_pretrained(
            MURIL_CHECKPOINT
        )

        # Load trained LoRA adapter.
        self.encoder = PeftModel.from_pretrained(
            base,
            self.checkpoint_dir / "adapter",
        )

        hidden_size = base.config.hidden_size

        # Recreate classifier heads.
        self.category_head = nn.Linear(
            hidden_size,
            len(self.taxonomy["categories"]),
        )

        self.subcategory_head = nn.Linear(
            hidden_size,
            len(self.taxonomy["subcategories"]),
        )

        self.priority_head = nn.Linear(
            hidden_size,
            1,
        )

        # Load trained head weights.
        heads = torch.load(
            self.checkpoint_dir / "heads.pt",
            map_location="cpu",
        )

        self.category_head.load_state_dict(
            heads["category_head"]
        )

        self.subcategory_head.load_state_dict(
            heads["subcategory_head"]
        )

        self.priority_head.load_state_dict(
            heads["priority_head"]
        )

        # Move everything to the selected device.
        self.encoder.to(self.device)
        self.category_head.to(self.device)
        self.subcategory_head.to(self.device)
        self.priority_head.to(self.device)

        # Evaluation mode.
        self.encoder.eval()
        self.category_head.eval()
        self.subcategory_head.eval()
        self.priority_head.eval()

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        **_ignored_extra_tokenizer_fields,
    ):
        # Preserve the same tokenizer inputs used during training.
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)

        # ---------------------------------------------------------
        # IMPORTANT:
        # Use EXACTLY the same masked mean pooling as training.
        # This keeps the training and inference representations
        # consistent.
        # ---------------------------------------------------------
        last_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()

        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1e-9)

        return {
            "category_logits": self.category_head(pooled),
            "subcategory_logits": self.subcategory_head(pooled),
            "priority_pred": self.priority_head(pooled).squeeze(-1),
            "pooled": pooled,
        }

    def _forward(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        enc = {
            key: value.to(self.device)
            for key, value in enc.items()
        }

        with torch.no_grad():
            out = self(**enc)

        return (
            out["category_logits"],
            out["subcategory_logits"],
            out["priority_pred"],
        )

    def predict(self, text):
        cat_logits, sub_logits, priority = self._forward(text)

        # ---------------------------------------------------------
        # Category prediction
        # ---------------------------------------------------------
        cat_probs = torch.softmax(
            cat_logits,
            dim=-1,
        )[0]

        cat_idx = int(
            torch.argmax(cat_probs)
        )

        # ---------------------------------------------------------
        # Hierarchical subcategory prediction
        # ---------------------------------------------------------
        sub_logits = GrieveAIClassifier.mask_subcategory_logits(
            sub_logits,
            cat_idx,
            self.taxonomy,
        )

        sub_probs = torch.softmax(
            sub_logits,
            dim=-1,
        )[0]

        sub_idx = int(
            torch.argmax(sub_probs)
        )

        # ---------------------------------------------------------
        # Priority prediction
        # ---------------------------------------------------------
        priority_raw = float(
            priority.item()
        )

        priority_value = int(
            max(
                1,
                min(
                    5,
                    round(priority_raw),
                ),
            )
        )

        return {
            "category": self.taxonomy["categories"][cat_idx],
            "subcategory": self.taxonomy["subcategories"][sub_idx],
            "priority": priority_value,
            "priority_raw": round(
                priority_raw,
                3,
            ),
            "confidence": round(
                float(cat_probs[cat_idx]),
                3,
            ),
            "subcategory_confidence": round(
                float(sub_probs[sub_idx]),
                3,
            ),
        }

    def explain(
        self,
        text: str,
        top_k: int = 8,
    ):
        from src.explainability import (
            build_transformer_shap_explainer,
            explain_transformer,
        )

        prediction = self.predict(text)

        class_idx = self.taxonomy["categories"].index(
            prediction["category"]
        )

        explainer = build_transformer_shap_explainer(
            self,
            self.tokenizer,
            self.device,
        )

        return explain_transformer(
            explainer,
            text,
            predicted_class_idx=class_idx,
            top_k=top_k,
        )