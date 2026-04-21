from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


PATTERN = re.compile(
    r"^(?P<pid>-?\d+)_c(?P<camid>\d)s(?P<seqid>\d)_(?P<frameid>\d+)_(?P<idx>\d+)\.jpg(?:\.jpg)?$"
)


def parse_market1501_filename(filename: str) -> dict:
    match = PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Filename format not recognized: {filename}")

    info = match.groupdict()
    return {
        "pid": int(info["pid"]),
        "camid": int(info["camid"]),
        "seqid": int(info["seqid"]),
        "frameid": int(info["frameid"]),
        "idx": int(info["idx"]),
    }


def collect_split(root: Path, split_name: str, folder_name: str) -> pd.DataFrame:
    split_dir = root / folder_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    rows = []
    for img_path in sorted(split_dir.glob("*.jpg")):
        meta = parse_market1501_filename(img_path.name)

        # pid == -1 is junk image in Market-1501
        if meta["pid"] == -1:
            continue

        rows.append(
            {
                "split": split_name,
                "path": str(img_path.resolve()),
                "rel_path": str(img_path.relative_to(root)),
                "filename": img_path.name,
                **meta,
            }
        )

    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame, split_name: str) -> None:
    print(f"\n=== {split_name.upper()} ===")
    print(f"num_images     : {len(df)}")
    print(f"num_ids        : {df['pid'].nunique()}")
    print(f"num_cameras    : {df['camid'].nunique()}")
    print(f"seq_ids        : {sorted(df['seqid'].unique().tolist())}")
    print(f"frameid_minmax : ({df['frameid'].min()}, {df['frameid'].max()})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to Market-1501-v15.09.15",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to save generated CSV metadata",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = collect_split(root, "train", "bounding_box_train")
    query_df = collect_split(root, "query", "query")
    gallery_df = collect_split(root, "gallery", "bounding_box_test")

    all_df = pd.concat([train_df, query_df, gallery_df], ignore_index=True)

    train_df.to_csv(outdir / "market1501_train_metadata.csv", index=False)
    query_df.to_csv(outdir / "market1501_query_metadata.csv", index=False)
    gallery_df.to_csv(outdir / "market1501_gallery_metadata.csv", index=False)
    all_df.to_csv(outdir / "market1501_all_metadata.csv", index=False)

    summarize(train_df, "train")
    summarize(query_df, "query")
    summarize(gallery_df, "gallery")

    print("\nSaved files:")
    print(outdir / "market1501_train_metadata.csv")
    print(outdir / "market1501_query_metadata.csv")
    print(outdir / "market1501_gallery_metadata.csv")
    print(outdir / "market1501_all_metadata.csv")


if __name__ == "__main__":
    main()