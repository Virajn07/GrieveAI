"""Inference adapter for the trained IndicBERT v2 + LoRA Run 1 checkpoint.

The base transformer is fetched from the Hugging Face model named in the
saved PEFT adapter configuration. Adapter weights, task heads, tokenizer and
taxonomy are expected in the local checkpoint directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# This project loads Transformers through PyTorch. Disable its optional TF
# backend before imports so SHAP/Transformers do not probe a mismatched local
# TensorFlow installation during on-demand explanations.
os.environ.setdefault("USE_TF", "0")

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModel, PreTrainedTokenizerFast


EXPECTED_BASE_MODEL = "ai4bharat/IndicBERTv2-MLM-only"
DEFAULT_MAX_LENGTH = 128


def _taxonomy(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read IndicBERT taxonomy: {path}") from exc
    categories = list(value.get("categories", {}).keys())
    subcategories: list[str] = []
    category_to_subcategory_indices: dict[str, list[int]] = {}
    for category in categories:
        children = value["categories"][category].get("subcategories", [])
        if not children:
            raise RuntimeError(f"IndicBERT taxonomy category {category!r} has no subcategories")
        category_to_subcategory_indices[category] = []
        for child in children:
            category_to_subcategory_indices[category].append(len(subcategories))
            subcategories.append(child)
    if len(set(subcategories)) != len(subcategories):
        raise RuntimeError("IndicBERT taxonomy contains duplicate subcategory labels")
    return {
        "source": value,
        "categories": categories,
        "subcategories": subcategories,
        "category_to_subcategory_indices": category_to_subcategory_indices,
    }


class IndicBERTInference:
    """Cached, CPU/GPU-safe inference matching the IndicBERT training notebook."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints/indicbert_lora/run1",
        taxonomy_path: str | Path = "config/taxonomy.json",
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        required = (
            self.checkpoint_dir / "tokenizer" / "tokenizer.json",
            self.checkpoint_dir / "tokenizer" / "tokenizer_config.json",
            self.checkpoint_dir / "adapter" / "adapter_config.json",
            self.checkpoint_dir / "adapter" / "adapter_model.safetensors",
            self.checkpoint_dir / "heads.pt",
            self.checkpoint_dir / "taxonomy.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "IndicBERT checkpoint is incomplete. Missing: " + ", ".join(missing)
            )

        self.taxonomy = _taxonomy(self.checkpoint_dir / "taxonomy.json")
        project_taxonomy = _taxonomy(Path(taxonomy_path))
        if self.taxonomy["source"] != project_taxonomy["source"]:
            raise RuntimeError(
                "IndicBERT checkpoint taxonomy does not match TAXONOMY_CONFIG; "
                "refusing to load incompatible label ordering."
            )

        adapter_config = json.loads(
            (self.checkpoint_dir / "adapter" / "adapter_config.json").read_text(encoding="utf-8")
        )
        base_model_name = adapter_config.get("base_model_name_or_path")
        if base_model_name != EXPECTED_BASE_MODEL:
            raise RuntimeError(
                "Unexpected IndicBERT adapter base model "
                f"{base_model_name!r}; expected {EXPECTED_BASE_MODEL!r}."
            )
        expected_lora = {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["query", "value"],
            "task_type": "FEATURE_EXTRACTION",
            "bias": "none",
        }
        mismatched_lora = {
            key: (adapter_config.get(key), expected)
            for key, expected in expected_lora.items()
            if adapter_config.get(key) != expected
        }
        if mismatched_lora:
            raise RuntimeError(
                "IndicBERT adapter LoRA settings do not match its training notebook: "
                f"{mismatched_lora}"
            )
        self.base_model_name = base_model_name
        configured_model_version = os.getenv("INDICBERT_MODEL_VERSION", "").strip()
        self.max_length = int(os.getenv("INDICBERT_MAX_LENGTH", str(DEFAULT_MAX_LENGTH)))
        if not 1 <= self.max_length <= 512:
            raise ValueError("INDICBERT_MAX_LENGTH must be between 1 and 512")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            tokenizer_dir = self.checkpoint_dir / "tokenizer"
            tokenizer_config = json.loads(
                (tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            tokenizer_data = json.loads(
                (tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8")
            )
            known_specials = {
                tokenizer_config.get(name)
                for name in ("unk_token", "cls_token", "sep_token", "pad_token", "mask_token")
            }
            additional_specials = [
                item["content"]
                for item in tokenizer_data.get("added_tokens", [])
                if item.get("special") and item["content"] not in known_specials
            ]
            # TokenizersBackend checkpoints are readable across the project's
            # older Transformers installs via the serialized fast-tokenizer JSON.
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
                unk_token=tokenizer_config.get("unk_token"),
                cls_token=tokenizer_config.get("cls_token"),
                sep_token=tokenizer_config.get("sep_token"),
                pad_token=tokenizer_config.get("pad_token"),
                mask_token=tokenizer_config.get("mask_token"),
                additional_special_tokens=additional_specials,
                model_max_length=self.max_length,
            )
            revision = os.getenv("INDICBERT_BASE_REVISION", "").strip() or None
            try:
                base_path = snapshot_download(
                    repo_id=self.base_model_name,
                    revision=revision,
                    local_files_only=True,
                )
            except Exception:
                base_path = snapshot_download(
                    repo_id=self.base_model_name,
                    revision=revision,
                )
            self.base_model_revision = Path(base_path).name
            base = AutoModel.from_pretrained(base_path, local_files_only=True)
            self.encoder = PeftModel.from_pretrained(
                base,
                self.checkpoint_dir / "adapter",
                is_trainable=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load IndicBERT v2 + LoRA. The local adapter is present, "
                "but the base model must be cached or downloadable from Hugging Face. "
                f"Details: {exc}"
            ) from exc
        self.model_version = configured_model_version or (
            f"indicbert_lora_run1+base-{self.base_model_revision[:8]}"
        )

        hidden_size = int(base.config.hidden_size)
        self.category_head = torch.nn.Linear(hidden_size, len(self.taxonomy["categories"]))
        self.subcategory_head = torch.nn.Linear(hidden_size, len(self.taxonomy["subcategories"]))
        self.priority_head = torch.nn.Linear(hidden_size, 1)
        try:
            heads = torch.load(self.checkpoint_dir / "heads.pt", map_location="cpu", weights_only=True)
        except TypeError:  # Compatibility with older supported PyTorch builds.
            heads = torch.load(self.checkpoint_dir / "heads.pt", map_location="cpu")
        expected_heads = {"category_head", "subcategory_head", "priority_head"}
        if set(heads) != expected_heads:
            raise RuntimeError(f"IndicBERT heads file must contain {sorted(expected_heads)}")
        self.category_head.load_state_dict(heads["category_head"], strict=True)
        self.subcategory_head.load_state_dict(heads["subcategory_head"], strict=True)
        self.priority_head.load_state_dict(heads["priority_head"], strict=True)

        for module in (self.encoder, self.category_head, self.subcategory_head, self.priority_head):
            module.to(self.device)
            module.eval()

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_inputs = {
            key: inputs[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in inputs
        }
        outputs = self.encoder(**encoder_inputs)
        mask = encoder_inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return {
            "category_logits": self.category_head(pooled),
            "subcategory_logits": self.subcategory_head(pooled),
            "priority_pred": self.priority_head(pooled).squeeze(-1),
        }

    def _forward(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            return self(**encoded)

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        output = self._forward(text)
        category_probabilities = torch.softmax(output["category_logits"], dim=-1)[0]
        category_index = int(category_probabilities.argmax().item())
        allowed = self.taxonomy["category_to_subcategory_indices"][
            self.taxonomy["categories"][category_index]
        ]
        subcategory_logits = output["subcategory_logits"][0].clone()
        invalid = torch.ones_like(subcategory_logits, dtype=torch.bool)
        invalid[allowed] = False
        subcategory_logits[invalid] = float("-inf")
        subcategory_probabilities = torch.softmax(subcategory_logits, dim=-1)
        subcategory_index = int(subcategory_probabilities.argmax().item())
        priority_raw = float(output["priority_pred"][0].item())
        return {
            "category": self.taxonomy["categories"][category_index],
            "subcategory": self.taxonomy["subcategories"][subcategory_index],
            "priority": max(1, min(5, round(priority_raw))),
            "priority_raw": round(priority_raw, 3),
            "confidence": round(float(category_probabilities[category_index].item()), 6),
            "subcategory_confidence": round(float(subcategory_probabilities[subcategory_index].item()), 6),
            "model_version": self.model_version,
        }

    def explain(self, text: str, top_k: int = 8) -> list[tuple[str, float]]:
        from src.explainability import build_transformer_shap_explainer, explain_transformer

        prediction = self.predict(text)
        category_index = self.taxonomy["categories"].index(prediction["category"])
        explainer = build_transformer_shap_explainer(self, self.tokenizer, self.device)
        return explain_transformer(explainer, text, predicted_class_idx=category_index, top_k=top_k)
