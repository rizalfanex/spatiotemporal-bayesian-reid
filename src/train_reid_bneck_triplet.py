from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MarketReIDDataset(Dataset):
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
    train_parts = []
    val_parts = []

    for pid, group in df.groupby("pid"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_val = max(1, int(round(n * val_ratio))) if n >= 2 else 0

        val_group = group.iloc[:n_val].copy()
        train_group = group.iloc[n_val:].copy()

        if len(train_group) == 0:
            train_group = val_group.iloc[:1].copy()
            val_group = val_group.iloc[1:].copy()

        train_parts.append(train_group)
        if len(val_group) > 0:
            val_parts.append(val_group)

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)
    return train_df, val_df


class RandomIdentitySampler(Sampler):
    def __init__(self, labels: list[int], batch_size: int, num_instances: int):
        self.labels = labels
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        if self.batch_size % self.num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")

        self.index_dic = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.index_dic[label].append(index)

        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = self.index_dic[pid].copy()
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = self.pids.copy()
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        targets_onehot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_onehot + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, D]
        dist_mat = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        N = labels.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = ~is_pos

        # exclude self-comparison from positives
        eye = torch.eye(N, dtype=torch.bool, device=labels.device)
        is_pos = is_pos & ~eye

        dist_ap = torch.where(is_pos, dist_mat, torch.tensor(-1e9, device=dist_mat.device)).max(dim=1)[0]
        dist_an = torch.where(is_neg, dist_mat, torch.tensor(1e9, device=dist_mat.device)).min(dim=1)[0]

        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss


class ReIDResNet50BNNeck(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        feat_dim = 2048
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        global_feat = self.gap(x).flatten(1)      # [B, 2048]
        bn_feat = self.bnneck(global_feat)        # [B, 2048]
        logits = self.classifier(bn_feat)         # [B, C]
        return global_feat, bn_feat, logits


def get_transforms():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_tf, val_tf


@torch.no_grad()
def evaluate_classification(model: nn.Module, loader: DataLoader, device: torch.device, ce_loss_fn: nn.Module):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            _, _, logits = model(images)
            loss = ce_loss_fn(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "val_loss": total_loss / total_samples,
        "val_acc": total_correct / total_samples,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ce_loss_fn: nn.Module,
    tri_loss_fn: nn.Module,
    tri_weight: float = 1.0,
):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_tri = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            global_feat, bn_feat, logits = model(images)
            ce_loss = ce_loss_fn(logits, labels)
            tri_loss = tri_loss_fn(global_feat, labels)
            loss = ce_loss + tri_weight * tri_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_ce += ce_loss.item() * labels.size(0)
        total_tri += tri_loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "train_loss": total_loss / total_samples,
        "train_ce_loss": total_ce / total_samples,
        "train_tri_loss": total_tri / total_samples,
        "train_acc": total_correct / total_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
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

    train_ds = MarketReIDDataset(train_df, pid_to_label, train_tf)
    val_ds = MarketReIDDataset(val_df, pid_to_label, val_tf)

    train_labels = [pid_to_label[int(pid)] for pid in train_df["pid"].tolist()]
    train_sampler = RandomIdentitySampler(
        labels=train_labels,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReIDResNet50BNNeck(num_classes=len(unique_pids)).to(device)

    ce_loss_fn = CrossEntropyLabelSmooth(num_classes=len(unique_pids), epsilon=0.1)
    tri_loss_fn = BatchHardTripletLoss(margin=args.triplet_margin)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_acc = -1.0
    history = []

    print(f"Device          : {device}")
    print(f"Total train imgs: {len(df)}")
    print(f"Train split imgs: {len(train_df)}")
    print(f"Val split imgs  : {len(val_df)}")
    print(f"Num classes     : {len(unique_pids)}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Num instances   : {args.num_instances}")
    print(f"Epochs          : {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            ce_loss_fn=ce_loss_fn,
            tri_loss_fn=tri_loss_fn,
            tri_weight=args.triplet_weight,
        )

        val_metrics = evaluate_classification(
            model=model,
            loader=val_loader,
            device=device,
            ce_loss_fn=ce_loss_fn,
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
        }
        history.append(row)

        print(
            f"lr={row['lr']:.6f} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"ce={row['train_ce_loss']:.4f} | "
            f"tri={row['train_tri_loss']:.4f} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_acc']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pid_to_label": pid_to_label,
            "args": vars(args),
        }

        torch.save(checkpoint, outdir / "last_checkpoint.pth")

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            torch.save(checkpoint, outdir / "best_checkpoint.pth")
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