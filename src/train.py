"""
GrieveAI training script.

Run from the repo root (this is what notebooks/01_train_classifier.ipynb calls):

    python src/train.py \
        --data data/processed/grievances_synthetic.csv \
        --taxonomy config/taxonomy.json \
        --epochs 4 \
        --batch_size 16 \
        --output_dir checkpoints/run1

On the synthetic dataset (~1,300 rows) this trains in a few minutes on a
free-tier Colab T4 GPU. Swap --data to point at real VCET data later with
no other code changes, as long as the CSV has the same column schema:
id, text, language, category, subcategory, priority, duplicate_of
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import GrieveAIClassifier, FocalLoss, load_taxonomy, MURIL_CHECKPOINT


class GrievanceDataset(Dataset):
    def __init__(self, df, tokenizer, taxonomy, max_length=128):
        self.texts = df["text"].tolist()
        self.categories = df["category"].tolist()
        self.subcategories = df["subcategory"].tolist()
        self.priorities = df["priority"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cat_to_idx = {c: i for i, c in enumerate(taxonomy["categories"])}
        self.subcat_to_idx = {s: i for i, s in enumerate(taxonomy["subcategories"])}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "category_label": torch.tensor(self.cat_to_idx[self.categories[idx]], dtype=torch.long),
            "subcategory_label": torch.tensor(self.subcat_to_idx[self.subcategories[idx]], dtype=torch.long),
            "priority_label": torch.tensor(self.priorities[idx], dtype=torch.float),
        }


def evaluate(model, loader, device):
    model.eval()
    correct_cat, correct_sub, total = 0, 0, 0
    priority_abs_err = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids, attention_mask)

            cat_pred = out["category_logits"].argmax(-1).cpu()
            sub_pred = out["subcategory_logits"].argmax(-1).cpu()
            pri_pred = out["priority_pred"].cpu()

            correct_cat += (cat_pred == batch["category_label"]).sum().item()
            correct_sub += (sub_pred == batch["subcategory_label"]).sum().item()
            priority_abs_err += (pri_pred - batch["priority_label"]).abs().sum().item()
            total += len(batch["category_label"])

    return {
        "category_acc": correct_cat / total,
        "subcategory_acc": correct_sub / total,
        "priority_mae": priority_abs_err / total,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected. Training MuRIL on CPU is very slow - "
              "make sure the Colab runtime is set to a GPU (Runtime > Change runtime type > T4 GPU).")

    taxonomy = load_taxonomy(args.taxonomy)
    print(f"Loaded taxonomy: {len(taxonomy['categories'])} categories, "
          f"{len(taxonomy['subcategories'])} sub-categories")

    df = pd.read_csv(args.data)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(MURIL_CHECKPOINT)
    train_ds = GrievanceDataset(train_df, tokenizer, taxonomy)
    val_ds = GrievanceDataset(val_df, tokenizer, taxonomy)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = GrieveAIClassifier(
        num_categories=len(taxonomy["categories"]),
        num_subcategories=len(taxonomy["subcategories"]),
    ).to(device)

    trainable = model.trainable_parameters()
    total_trainable = sum(n for _, n in trainable)
    print(f"Trainable parameters (LoRA adapters + heads only): {total_trainable:,}")

    category_loss_fn = FocalLoss(gamma=2.0)
    subcategory_loss_fn = FocalLoss(gamma=2.0)
    priority_loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cat_labels = batch["category_label"].to(device)
            sub_labels = batch["subcategory_label"].to(device)
            pri_labels = batch["priority_label"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask)

            loss_cat = category_loss_fn(out["category_logits"], cat_labels)
            loss_sub = subcategory_loss_fn(out["subcategory_logits"], sub_labels)
            loss_pri = priority_loss_fn(out["priority_pred"], pri_labels)

            # Category + sub-category weighted higher than priority regression -
            # tune these weights once you see which head lags on real data.
            loss = loss_cat + loss_sub + 0.5 * loss_pri

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {running_loss / len(train_loader):.4f} "
              f"- val_category_acc: {val_metrics['category_acc']:.3f} "
              f"- val_subcategory_acc: {val_metrics['subcategory_acc']:.3f} "
              f"- val_priority_mae: {val_metrics['priority_mae']:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "grieveai_model.pt"))
    with open(os.path.join(args.output_dir, "taxonomy_used.json"), "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    print(f"Saved model + taxonomy snapshot to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/grievances_synthetic.csv")
    parser.add_argument("--taxonomy", type=str, default="config/taxonomy.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/run1")
    args = parser.parse_args()
    main(args)
