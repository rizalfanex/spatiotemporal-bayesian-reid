from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights


class MarketImageDataset(Dataset):
    def __init__(self, csv_path: str, transform):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        return img, idx


def build_model(device: torch.device) -> tuple[nn.Module, callable]:
    weights = ResNet50_Weights.IMAGENET1K_V2
    backbone = models.resnet50(weights=weights)

    # Remove final classification layer, keep pooled visual feature
    model = nn.Sequential(*list(backbone.children())[:-1])
    model = model.to(device)
    model.eval()

    transform = weights.transforms()
    return model, transform


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    all_features = []

    for images, _ in tqdm(dataloader, desc="Extracting", leave=True):
        images = images.to(device, non_blocking=True)
        feats = model(images)              # [B, 2048, 1, 1]
        feats = feats.flatten(1)           # [B, 2048]
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-csv", type=str, required=True)
    parser.add_argument("--gallery-csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, transform = build_model(device)

    query_dataset = MarketImageDataset(args.query_csv, transform)
    gallery_dataset = MarketImageDataset(args.gallery_csv, transform)

    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Num query images   : {len(query_dataset)}")
    print(f"Num gallery images : {len(gallery_dataset)}")

    query_features = extract_features(model, query_loader, device)
    gallery_features = extract_features(model, gallery_loader, device)

    np.save(outdir / "query_features.npy", query_features)
    np.save(outdir / "gallery_features.npy", gallery_features)

    # Save aligned metadata copies
    pd.read_csv(args.query_csv).to_csv(outdir / "query_metadata.csv", index=False)
    pd.read_csv(args.gallery_csv).to_csv(outdir / "gallery_metadata.csv", index=False)

    print("\nSaved files:")
    print(outdir / "query_features.npy")
    print(outdir / "gallery_features.npy")
    print(outdir / "query_metadata.csv")
    print(outdir / "gallery_metadata.csv")

    print("\nFeature shapes:")
    print("query_features  :", query_features.shape)
    print("gallery_features:", gallery_features.shape)


if __name__ == "__main__":
    main()