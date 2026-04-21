from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReIDTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pid_to_label: dict[int, int], transform):
        self.df = df.reset_index(drop=True)
        self.pid_to_label = pid_to_label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        pid = int(row["pid"])
        label = self.pid_to_label[pid]
        return img, label


def split_train_val_by_pid_images(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []

    for pid, group in df.groupby("pid"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_val = max(1, int(round(n * val_ratio))) if n >= 2 else 0
        val_idx = set(rng.choice(n, size=n_val, replace=False).tolist()) if n_val > 0 else set()

        val_group = group.iloc[list(val_idx)] if len(val_idx) > 0 else group.iloc[[]]
        train_group = group.drop(index=list(val_idx)) if len(val_idx) > 0 else group

        if len(train_group) == 0:
            train_group = val_group.iloc[:1].copy()
            val_group = val_group.iloc[1:].copy()

        train_parts.append(train_group)
        if len(val_group) > 0:
            val_parts.append(val_group)

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)

    return train_df, val_df


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model


def get_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "val_loss": total_loss / total_samples,
        "val_acc": total_correct / total_samples,
    }


def train_one_epoch(model, loader, device, optimizer, criterion, scaler):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "train_loss": total_loss / total_samples,
        "train_acc": total_correct / total_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    df = df[df["pid"] >= 0].reset_index(drop=True)

    train_df, val_df = split_train_val_by_pid_images(df, val_ratio=args.val_ratio, seed=args.seed)

    unique_pids = sorted(df["pid"].unique().tolist())
    pid_to_label = {int(pid): idx for idx, pid in enumerate(unique_pids)}

    with open(outdir / "pid_to_label.json", "w", encoding="utf-8") as f:
        json.dump(pid_to_label, f, indent=2)

    train_tf, val_tf = get_transforms()

    train_ds = ReIDTrainDataset(train_df, pid_to_label, train_tf)
    val_ds = ReIDTrainDataset(val_df, pid_to_label, val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(unique_pids), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_acc = -1.0
    history = []

    print(f"Device        : {device}")
    print(f"Train images  : {len(train_ds)}")
    print(f"Val images    : {len(val_ds)}")
    print(f"Num classes   : {len(unique_pids)}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_metrics = train_one_epoch(model, train_loader, device, optimizer, criterion, scaler)
        val_metrics = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }
        history.append(row)

        print(
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_acc']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "pid_to_label": pid_to_label,
            },
            outdir / "last_checkpoint.pth",
        )

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "pid_to_label": pid_to_label,
                },
                outdir / "best_checkpoint.pth",
            )
            print(f"Saved best checkpoint at epoch {epoch}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(outdir / "training_history.csv", index=False)

    print("\nSaved files:")
    print(outdir / "best_checkpoint.pth")
    print(outdir / "last_checkpoint.pth")
    print(outdir / "training_history.csv")
    print(outdir / "pid_to_label.json")


if __name__ == "__main__":
    main()