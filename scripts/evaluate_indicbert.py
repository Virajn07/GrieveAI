"""Re-evaluate the saved IndicBERT checkpoint on this repo's fixed synthetic split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    recall_score,
)

from src.data_splits import has_group_leakage, make_splits
from src.indicbert_inference import IndicBERTInference


def score_partition(frame, model, batch_size: int, threshold: float):
    categories = model.taxonomy["categories"]
    subcategories = model.taxonomy["subcategories"]
    category_to_subcategory_indices = model.taxonomy["category_to_subcategory_indices"]
    texts = frame["text"].astype(str).tolist()
    gold_categories = frame["category"].astype(str).tolist()
    gold_subcategories = frame["subcategory"].astype(str).tolist()
    gold_priorities = frame["priority"].astype(int).tolist()
    languages = frame["language"].astype(str).tolist()

    predicted_categories = []
    predicted_subcategories = []
    oracle_subcategories = []
    predicted_priorities = []
    confidences = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        encoded = model.tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=model.max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(model.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.inference_mode():
            output = model(**encoded)
        category_probabilities = torch.softmax(output["category_logits"], dim=-1).cpu()
        subcategory_logits = output["subcategory_logits"].cpu()
        priority_predictions = output["priority_pred"].cpu().tolist()

        for row, category_index in enumerate(category_probabilities.argmax(-1).tolist()):
            predicted_category = categories[category_index]
            predicted_categories.append(predicted_category)
            confidences.append(float(category_probabilities[row, category_index]))
            predicted_priorities.append(float(priority_predictions[row]))

            for parent, destination in (
                (predicted_category, predicted_subcategories),
                (gold_categories[offset + row], oracle_subcategories),
            ):
                allowed = category_to_subcategory_indices[parent]
                masked_logits = torch.full_like(subcategory_logits[row], float("-inf"))
                masked_logits[allowed] = subcategory_logits[row][allowed]
                destination.append(subcategories[int(masked_logits.argmax().item())])

    def classification_metrics(indices):
        if not indices:
            return {"n": 0, "category_accuracy": None, "category_macro_f1": None}
        return {
            "n": len(indices),
            "category_accuracy": float(
                accuracy_score(
                    [gold_categories[i] for i in indices],
                    [predicted_categories[i] for i in indices],
                )
            ),
            "category_macro_f1": float(
                f1_score(
                    [gold_categories[i] for i in indices],
                    [predicted_categories[i] for i in indices],
                    average="macro",
                    zero_division=0,
                )
            ),
            "subcategory_accuracy": float(
                accuracy_score(
                    [gold_subcategories[i] for i in indices],
                    [predicted_subcategories[i] for i in indices],
                )
            ),
        }

    category_precision, category_recall, category_f1, category_support = (
        precision_recall_fscore_support(
            gold_categories,
            predicted_categories,
            labels=categories,
            zero_division=0,
        )
    )
    subcategory_precision, subcategory_recall, subcategory_f1, subcategory_support = (
        precision_recall_fscore_support(
            gold_subcategories,
            predicted_subcategories,
            labels=subcategories,
            zero_division=0,
        )
    )
    gold_high = [value >= 4 for value in gold_priorities]
    predicted_high = [value >= 4 for value in predicted_priorities]
    confidence_high = [i for i, value in enumerate(confidences) if value >= threshold]
    confidence_low = [i for i, value in enumerate(confidences) if value < threshold]

    overall = {
        "n": len(frame),
        "category_accuracy": float(accuracy_score(gold_categories, predicted_categories)),
        "category_macro_f1": float(
            f1_score(gold_categories, predicted_categories, average="macro", zero_division=0)
        ),
        "subcategory_macro_f1": float(
            f1_score(gold_subcategories, predicted_subcategories, average="macro", zero_division=0)
        ),
        "subcategory_accuracy": float(accuracy_score(gold_subcategories, predicted_subcategories)),
        "oracle_subcategory_macro_f1": float(
            f1_score(gold_subcategories, oracle_subcategories, average="macro", zero_division=0)
        ),
        "priority_mae_raw": float(mean_absolute_error(gold_priorities, predicted_priorities)),
        "high_priority_recall_raw_ge_4": float(recall_score(gold_high, predicted_high, zero_division=0)),
    }
    per_category = {
        label: {
            "support": int(category_support[i]),
            "precision": float(category_precision[i]),
            "recall": float(category_recall[i]),
            "f1": float(category_f1[i]),
        }
        for i, label in enumerate(categories)
    }
    per_subcategory = {
        label: {
            "support": int(subcategory_support[i]),
            "precision": float(subcategory_precision[i]),
            "recall": float(subcategory_recall[i]),
            "f1": float(subcategory_f1[i]),
        }
        for i, label in enumerate(subcategories)
    }
    language_metrics = {}
    for language in sorted(set(languages)):
        indices = [i for i, value in enumerate(languages) if value == language]
        language_metrics[language] = {
            **classification_metrics(indices),
            "subcategory_macro_f1": float(
                f1_score(
                    [gold_subcategories[i] for i in indices],
                    [predicted_subcategories[i] for i in indices],
                    average="macro",
                    zero_division=0,
                )
            ),
            "subcategory_accuracy": float(
                accuracy_score(
                    [gold_subcategories[i] for i in indices],
                    [predicted_subcategories[i] for i in indices],
                )
            ),
            "priority_mae_raw": float(
                mean_absolute_error(
                    [gold_priorities[i] for i in indices],
                    [predicted_priorities[i] for i in indices],
                )
            ),
        }

    return {
        "overall": overall,
        "language": language_metrics,
        "confidence_gate": {
            "threshold": threshold,
            "above_threshold": classification_metrics(confidence_high),
            "below_threshold": classification_metrics(confidence_low),
            "fraction_below_threshold": len(confidence_low) / len(frame) if len(frame) else None,
        },
        "per_category": per_category,
        "per_subcategory": per_subcategory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "data/processed/grievances_synthetic.csv"))
    parser.add_argument("--taxonomy", default=str(ROOT / "config/taxonomy.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/indicbert_lora/run1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--base-revision", default="8598f13fe52443bc3fc054fcd665944560145b5c")
    parser.add_argument("--output", help="optional JSON file for the full per-label results")
    parser.add_argument("--offline", action="store_true", help="use only the Hugging Face cache")
    args = parser.parse_args()
    os.environ["INDICBERT_BASE_REVISION"] = args.base_revision
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    dataset_path = Path(args.dataset)
    dataset = pd.read_csv(dataset_path)
    train, validation, test = make_splits(dataset, seed=args.seed)
    if has_group_leakage((train, validation, test)):
        raise RuntimeError("Duplicate or exact-text leakage found across evaluation splits")

    model = IndicBERTInference(args.checkpoint, args.taxonomy)
    encoder_parameter_count = sum(parameter.numel() for parameter in model.encoder.parameters())
    lora_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.encoder.named_parameters()
        if "lora_" in name
    )
    head_parameter_count = sum(
        parameter.numel()
        for head in (model.category_head, model.subcategory_head, model.priority_head)
        for parameter in head.parameters()
    )
    total_parameter_count = encoder_parameter_count + head_parameter_count
    trainable_parameter_count = lora_parameter_count + head_parameter_count
    checkpoint_dir = Path(args.checkpoint)
    result = {
        "data_scope": "synthetic only; not VCET performance",
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "split_seed": args.seed,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "base_model": model.base_model_name,
        "resolved_model_version": model.model_version,
        "device": str(model.device),
        "confidence_threshold": args.confidence_threshold,
        "parameters": {
            "total_encoder_and_task_heads": total_parameter_count,
            "trainable_lora_adapter": lora_parameter_count,
            "trainable_task_heads": head_parameter_count,
            "trainable_total": trainable_parameter_count,
            "trainable_percent": 100 * trainable_parameter_count / total_parameter_count,
        },
        "checkpoint_bytes_without_base_model": sum(
            path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
        ),
        "validation": score_partition(validation, model, args.batch_size, args.confidence_threshold),
        "test": score_partition(test, model, args.batch_size, args.confidence_threshold),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
