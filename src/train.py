"""Train MuRIL + LoRA and produce a compact inference checkpoint.

The split is duplicate-group aware so synthetic duplicate pairs cannot leak
between train and validation. Evaluation reports macro-F1 and priority MAE.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, mean_absolute_error, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_splits import has_group_leakage, make_splits, validate_taxonomy_labels
from src.model import FocalLoss, GrieveAIClassifier, MURIL_CHECKPOINT, load_taxonomy


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GrievanceDataset(Dataset):
    def __init__(self, df, tokenizer, taxonomy, max_length=128):
        self.texts = df["text"].astype(str).tolist()
        self.categories = df["category"].astype(str).tolist()
        self.subcategories = df["subcategory"].astype(str).tolist()
        self.priorities = df["priority"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cat_to_idx = {c: i for i, c in enumerate(taxonomy["categories"])}
        self.subcat_to_idx = {s: i for i, s in enumerate(taxonomy["subcategories"])}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "category_label": torch.tensor(self.cat_to_idx[self.categories[idx]], dtype=torch.long),
            "subcategory_label": torch.tensor(self.subcat_to_idx[self.subcategories[idx]], dtype=torch.long),
            "priority_label": torch.tensor(self.priorities[idx], dtype=torch.float),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


def evaluate(model, loader, device, taxonomy):
    model.eval()
    y_cat, p_cat, y_sub, p_sub, y_sub_oracle, p_sub_oracle = [], [], [], [], [], []
    y_pri, p_pri, high_true, high_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            out = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            cat = out["category_logits"].argmax(-1)
            masked_sub = torch.stack([
                model.mask_subcategory_logits(out["subcategory_logits"][i], int(cat[i]), taxonomy)
                for i in range(len(cat))
            ])
            sub = masked_sub.argmax(-1)
            oracle_masked_sub = torch.stack([
                model.mask_subcategory_logits(out["subcategory_logits"][i], int(batch["category_label"][i]), taxonomy)
                for i in range(len(batch["category_label"]))
            ])
            oracle_sub = oracle_masked_sub.argmax(-1)
            pri = out["priority_pred"]
            y_cat.extend(batch["category_label"].tolist()); p_cat.extend(cat.cpu().tolist())
            y_sub.extend(batch["subcategory_label"].tolist()); p_sub.extend(sub.cpu().tolist())
            y_sub_oracle.extend(batch["subcategory_label"].tolist()); p_sub_oracle.extend(oracle_sub.cpu().tolist())
            y_pri.extend(batch["priority_label"].tolist()); p_pri.extend(pri.cpu().tolist())
            high_true.extend((batch["priority_label"] >= 4).int().tolist())
            high_pred.extend((pri >= 4).int().tolist())
    return {
        "dataset_kind": "training_dataset_metrics_require_data_provenance_review",
        "category_macro_f1": f1_score(y_cat, p_cat, average="macro"),
        "subcategory_macro_f1": f1_score(y_sub, p_sub, average="macro"),
        "subcategory_macro_f1_given_gold_category": f1_score(y_sub_oracle, p_sub_oracle, average="macro"),
        "priority_mae": mean_absolute_error(y_pri, p_pri),
        "priority_recall_high": recall_score(high_true, high_pred, zero_division=0),
        "routing_accuracy": None,
        "routing_accuracy_note": "Requires validated gold department labels.",
        "override_rate": None,
        "override_rate_note": "Requires observed eligible human routing decisions.",
    }


def save_checkpoint(model, tokenizer, taxonomy, output_dir, metrics, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    model.encoder.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    torch.save({
        "category_head": model.category_head.state_dict(),
        "subcategory_head": model.subcategory_head.state_dict(),
        "priority_head": model.priority_head.state_dict(),
    }, output_dir / "heads.pt")
    (output_dir / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main(args):
    if args.smoke_test:
        args.epochs = 1
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taxonomy = load_taxonomy(args.taxonomy)
    df = pd.read_csv(args.data)
    required = {"id", "text", "category", "subcategory", "priority", "duplicate_of"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    taxonomy_json = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    validate_taxonomy_labels(df, taxonomy_json)
    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    print(f"Device: {device} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MURIL_CHECKPOINT)
    train_ds = GrievanceDataset(train_df, tokenizer, taxonomy, args.max_length)
    val_ds = GrievanceDataset(val_df, tokenizer, taxonomy, args.max_length)
    test_ds = GrievanceDataset(test_df, tokenizer, taxonomy, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    model = GrieveAIClassifier(len(taxonomy["categories"]), len(taxonomy["subcategories"]))
    model.to(device)
    print(f"Trainable parameters: {sum(n for _, n in model.trainable_parameters()):,}")

    # The synthetic category and subcategory labels are near-balanced; ordinary
    # cross-entropy (gamma=0) is the controlled default, with focal loss opt-in.
    cat_loss = FocalLoss(gamma=args.focal_gamma)
    sub_loss = FocalLoss(gamma=args.focal_gamma)
    pri_loss = torch.nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0
    best_metrics = {}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cat_labels = batch["category_label"].to(device)
            sub_labels = batch["subcategory_label"].to(device)
            pri_labels = batch["priority_label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                out = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                conditional_sub_logits = torch.stack([
                    model.mask_subcategory_logits(out["subcategory_logits"][i], int(cat_labels[i]), taxonomy)
                    for i in range(len(cat_labels))
                ])
                loss = cat_loss(out["category_logits"], cat_labels) + sub_loss(conditional_sub_logits, sub_labels) + 0.5 * pri_loss(out["priority_pred"], pri_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())

        metrics = evaluate(model, val_loader, device, taxonomy)
        metrics["dataset_kind"] = "synthetic_demo_pipeline_validation" if "synthetic" in Path(args.data).name.lower() else "unspecified_dataset_provenance"
        metrics["research_claim"] = "not_vcet_pilot_results" if metrics["dataset_kind"].startswith("synthetic") else "verify_provenance_before_reporting"
        print(f"Epoch {epoch}/{args.epochs} loss={total_loss/len(train_loader):.4f} metrics={metrics}")
        if metrics["category_macro_f1"] + metrics["subcategory_macro_f1"] > best_f1:
            best_f1 = metrics["category_macro_f1"] + metrics["subcategory_macro_f1"]
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint candidate")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, taxonomy)
    test_metrics["dataset_kind"] = "synthetic_demo_pipeline_validation" if "synthetic" in Path(args.data).name.lower() else "unspecified_dataset_provenance"
    test_metrics["research_claim"] = "not_vcet_pilot_results" if test_metrics["dataset_kind"].startswith("synthetic") else "verify_provenance_before_reporting"
    best_metrics["validation"] = best_metrics.copy()
    best_metrics["test"] = test_metrics
    best_metrics["split"] = {"train_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df), "duplicate_group_leakage": has_group_leakage((train_df, val_df, test_df))}
    experiment_config = vars(args).copy()
    import hashlib
    import platform
    from importlib.metadata import PackageNotFoundError, version
    experiment_config.update({
        "model_identifier": MURIL_CHECKPOINT,
        "tokenizer_identifier": MURIL_CHECKPOINT,
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "taxonomy_sha256": hashlib.sha256(Path(args.taxonomy).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
    })
    for package in ("torch", "transformers", "peft", "scikit-learn", "pandas"):
        try:
            experiment_config[f"{package}_version"] = version(package)
        except PackageNotFoundError:
            experiment_config[f"{package}_version"] = None
    import subprocess
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    experiment_config["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    experiment_config["git_working_tree_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    lora = model.encoder.peft_config["default"]
    experiment_config["lora_config"] = {
        "r": lora.r,
        "alpha": lora.lora_alpha,
        "dropout": lora.lora_dropout,
        "target_modules": sorted(str(module) for module in lora.target_modules),
    }
    save_checkpoint(model, tokenizer, taxonomy, args.output_dir, best_metrics, experiment_config)
    if args.smoke_test:
        from src.model import MuRILInference
        reloaded = MuRILInference(args.output_dir, args.taxonomy, device=device)
        sample = str(test_df.iloc[0]["text"])
        print(f"Reloaded smoke prediction: {reloaded.predict(sample)}")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/grievances_synthetic.csv")
    parser.add_argument("--taxonomy", default="config/taxonomy.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal_gamma", type=float, default=0.0, help="Use >0 only when justified by measured class imbalance")
    parser.add_argument("--output_dir", default="checkpoints/muril_lora/run1")
    parser.add_argument("--smoke_test", action="store_true", help="Run one epoch and reload the saved checkpoint for an inference check")
    main(parser.parse_args())
