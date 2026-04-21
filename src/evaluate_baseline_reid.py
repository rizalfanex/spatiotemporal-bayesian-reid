from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_cmc_map(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    topk: tuple[int, ...] = (1, 5, 10),
) -> dict:
    # cosine similarity because features were already L2-normalized
    sim = query_features @ gallery_features.T  # [num_query, num_gallery]

    num_query = len(query_df)
    cmc = np.zeros(len(gallery_df), dtype=np.float64)
    ap_list = []
    valid_queries = 0

    gallery_pids = gallery_df["pid"].to_numpy()
    gallery_camids = gallery_df["camid"].to_numpy()

    for q_idx in tqdm(range(num_query), desc="Evaluating"):
        q_pid = int(query_df.iloc[q_idx]["pid"])
        q_camid = int(query_df.iloc[q_idx]["camid"])

        # Standard Market-1501 protocol:
        # remove gallery samples with same pid and same camid as query
        invalid = (gallery_pids == q_pid) & (gallery_camids == q_camid)
        valid = ~invalid

        scores = sim[q_idx][valid]
        valid_gallery_pids = gallery_pids[valid]

        order = np.argsort(-scores)
        ranked_pids = valid_gallery_pids[order]

        matches = (ranked_pids == q_pid).astype(np.int32)

        if matches.sum() == 0:
            continue

        valid_queries += 1

        # CMC
        first_match_index = np.where(matches == 1)[0][0]
        cmc[first_match_index:] += 1

        # AP
        cumulative_hits = np.cumsum(matches)
        precision_at_k = cumulative_hits / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / matches.sum()
        ap_list.append(ap)

    if valid_queries == 0:
        raise RuntimeError("No valid queries found for evaluation.")

    cmc = cmc / valid_queries
    mAP = float(np.mean(ap_list))

    results = {
        "num_query_total": int(num_query),
        "num_query_valid": int(valid_queries),
        "mAP": mAP,
    }

    for k in topk:
        results[f"Rank-{k}"] = float(cmc[k - 1])

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-features", type=str, required=True)
    parser.add_argument("--gallery-features", type=str, required=True)
    parser.add_argument("--query-meta", type=str, required=True)
    parser.add_argument("--gallery-meta", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    query_features = np.load(args.query_features)
    gallery_features = np.load(args.gallery_features)

    query_df = pd.read_csv(args.query_meta)
    gallery_df = pd.read_csv(args.gallery_meta)

    results = compute_cmc_map(
        query_features=query_features,
        gallery_features=gallery_features,
        query_df=query_df,
        gallery_df=gallery_df,
        topk=(1, 5, 10),
    )

    with open(outdir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== BASELINE RESULTS ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nSaved:")
    print(outdir / "baseline_metrics.json")


if __name__ == "__main__":
    main()