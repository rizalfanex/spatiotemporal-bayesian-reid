from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_time_index(df: pd.DataFrame, seq_scale: int = 200000) -> pd.DataFrame:
    out = df.copy()
    out["time_idx"] = (out["seqid"] - 1) * seq_scale + out["frameid"]
    return out


def generate_positive_cross_camera_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for pid, group in df.groupby("pid"):
        group = group.sort_values("time_idx").reset_index(drop=True)
        n = len(group)

        for i in range(n):
            a = group.iloc[i]
            for j in range(i + 1, n):
                b = group.iloc[j]

                if a["camid"] == b["camid"]:
                    continue

                delta_t = abs(int(b["time_idx"]) - int(a["time_idx"]))

                if int(a["time_idx"]) <= int(b["time_idx"]):
                    cam_from, cam_to = int(a["camid"]), int(b["camid"])
                    seq_from, seq_to = int(a["seqid"]), int(b["seqid"])
                    frame_from, frame_to = int(a["frameid"]), int(b["frameid"])
                    file_from, file_to = a["filename"], b["filename"]
                else:
                    cam_from, cam_to = int(b["camid"]), int(a["camid"])
                    seq_from, seq_to = int(b["seqid"]), int(a["seqid"])
                    frame_from, frame_to = int(b["frameid"]), int(a["frameid"])
                    file_from, file_to = b["filename"], a["filename"]

                rows.append(
                    {
                        "pid": int(pid),
                        "cam_from": cam_from,
                        "cam_to": cam_to,
                        "seq_from": seq_from,
                        "seq_to": seq_to,
                        "frame_from": frame_from,
                        "frame_to": frame_to,
                        "time_from": min(int(a["time_idx"]), int(b["time_idx"])),
                        "time_to": max(int(a["time_idx"]), int(b["time_idx"])),
                        "delta_t": delta_t,
                        "file_from": file_from,
                        "file_to": file_to,
                    }
                )

    return pd.DataFrame(rows)


def summarize_transition_counts(pairs_df: pd.DataFrame) -> pd.DataFrame:
    trans = (
        pairs_df.groupby(["cam_from", "cam_to"])
        .size()
        .reset_index(name="count")
        .sort_values(["cam_from", "cam_to"])
        .reset_index(drop=True)
    )

    trans["prob_from_cam"] = (
        trans.groupby("cam_from")["count"].transform(lambda x: x / x.sum())
    )
    return trans


def summarize_delta_bins(pairs_df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 100, 500, 1000, 5000, 10000, 20000, 50000, np.inf]
    labels = [
        "0-100",
        "101-500",
        "501-1000",
        "1001-5000",
        "5001-10000",
        "10001-20000",
        "20001-50000",
        "50000+",
    ]

    temp = pairs_df.copy()
    temp["delta_bin"] = pd.cut(
        temp["delta_t"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    delta_summary = (
        temp.groupby(["cam_from", "cam_to", "delta_bin"], observed=False)
        .size()
        .reset_index(name="count")
        .sort_values(["cam_from", "cam_to", "delta_bin"])
        .reset_index(drop=True)
    )

    delta_summary["prob_within_transition"] = (
        delta_summary.groupby(["cam_from", "cam_to"])["count"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    )

    return delta_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to market1501_train_metadata.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save spatio-temporal prior files",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    df = build_time_index(df)

    pairs_df = generate_positive_cross_camera_pairs(df)
    trans_df = summarize_transition_counts(pairs_df)
    delta_df = summarize_delta_bins(pairs_df)

    pairs_df.to_csv(outdir / "positive_cross_camera_pairs.csv", index=False)
    trans_df.to_csv(outdir / "camera_transition_prior.csv", index=False)
    delta_df.to_csv(outdir / "camera_transition_delta_prior.csv", index=False)

    print("=== SPATIO-TEMPORAL PRIOR SUMMARY ===")
    print(f"train images used           : {len(df)}")
    print(f"unique train identities     : {df['pid'].nunique()}")
    print(f"positive cross-camera pairs : {len(pairs_df)}")
    print()

    print("Camera transition prior:")
    print(trans_df.head(20).to_string(index=False))
    print()

    print("Delta-time prior:")
    print(delta_df.head(30).to_string(index=False))
    print()

    print("Saved files:")
    print(outdir / "positive_cross_camera_pairs.csv")
    print(outdir / "camera_transition_prior.csv")
    print(outdir / "camera_transition_delta_prior.csv")


if __name__ == "__main__":
    main()