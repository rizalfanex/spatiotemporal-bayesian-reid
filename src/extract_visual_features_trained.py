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
from torchvision import models, transforms


class ImageCSVDataset(Dataset):
    def __init__(self, csv_file: str, transform):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        return img


class ResNet50FeatureNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = models.resnet50(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        self.base = base

    def forward(self, x: torch.Tensor):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.base.avgpool(x)
        feat = torch.flatten(x, 1)  # [B, 2048]
        return feat


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    feats = []

    for images in tqdm(loader, desc="Extracting"):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            batch_feat = model(images)

        batch_feat = batch_feat.float()
        batch_feat = torch.nn.functional.normalize(batch_feat, p=2, dim=1)
        feats.append(batch_feat.cpu().numpy())

    return np.concatenate(feats, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-csv", type=str, required=True)
    parser.add_argument("--gallery-csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    pid_to_label = checkpoint["pid_to_label"]
    num_classes = len(pid_to_label)

    model = ResNet50FeatureNet(num_classes=num_classes)
    model.base.load_state_dict(checkpoint["model_state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tf = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    query_ds = ImageCSVDataset(args.query_csv, tf)
    gallery_ds = ImageCSVDataset(args.gallery_csv, tf)

    query_loader = DataLoader(
        query_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Using device: {device}")
    print(f"Num query images   : {len(query_ds)}")
    print(f"Num gallery images : {len(gallery_ds)}")

    query_features = extract_features(model, query_loader, device)
    gallery_features = extract_features(model, gallery_loader, device)

    query_meta = pd.read_csv(args.query_csv)
    gallery_meta = pd.read_csv(args.gallery_csv)

    np.save(outdir / "query_features.npy", query_features)
    np.save(outdir / "gallery_features.npy", gallery_features)
    query_meta.to_csv(outdir / "query_metadata.csv", index=False)
    gallery_meta.to_csv(outdir / "gallery_metadata.csv", index=False)

    print("\nSaved files:")
    print(outdir / "query_features.npy")
    print(outdir / "gallery_features.npy")
    print(outdir / "query_metadata.csv")
    print(outdir / "gallery_metadata.csv")

    print("\nFeature shapes:")
    print(f"query_features  : {query_features.shape}")
    print(f"gallery_features: {gallery_features.shape}")


if __name__ == "__main__":
    main()