from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_time_index(df: pd.DataFrame, seq_scale: int = 200000) -> np.ndarray:
    return ((df["seqid"].to_numpy() - 1) * seq_scale + df["frameid"].to_numpy()).astype(np.int64)


def build_cam_prior_matrix(cam_prior_df: pd.DataFrame, eps: float = 1e-8) -> np.ndarray:
    # camera ids in Market-1501 are 1..6
    mat = np.full((7, 7), eps, dtype=np.float32)
    for _, row in cam_prior_df.iterrows():
        c_from = int(row["cam_from"])
        c_to = int(row["cam_to"])
        mat[c_from, c_to] = float(row["prob_from_cam"])
    return mat


def build_delta_prior_tensor(delta_prior_df: pd.DataFrame, eps: float = 1e-8) -> np.ndarray:
    # 8 bins: 0-100, 101-500, 501-1000, 1001-5000, 5001-10000, 10001-20000, 20001-50000, 50000+
    tensor = np.full((7, 7, 8), eps, dtype=np.float32)

    bin_to_idx = {
        "0-100": 0,
        "101-500": 1,
        "501-1000": 2,
        "1001-5000": 3,
        "5001-10000": 4,
        "10001-20000": 5,
        "20001-50000": 6,
        "50000+": 7,
    }

    for _, row in delta_prior_df.iterrows():
        c_from = int(row["cam_from"])
        c_to = int(row["cam_to"])
        b = str(row["delta_bin"])
        if b in bin_to_idx:
            tensor[c_from, c_to, bin_to_idx[b]] = float(row["prob_within_transition"])

    return tensor


def delta_to_bin_index(delta_t: np.ndarray) -> np.ndarray:
    # edges correspond to:
    # 0-100, 101-500, 501-1000, 1001-5000, 5001-10000, 10001-20000, 20001-50000, 50000+
    edges = np.array([100, 500, 1000, 5000, 10000, 20000, 50000], dtype=np.int64)
    return np.digitize(delta_t, edges, right=True)


def compute_metrics_with_bayesian_prior(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    cam_prior_mat: np.ndarray,
    delta_prior_tensor: np.ndarray,
    beta: float = 0.05,
    gamma: float = 0.05,
    eps: float = 1e-8,
) -> dict:
    gallery_pids = gallery_df["pid"].to_numpy()
    gallery_camids = gallery_df["camid"].to_numpy().astype(np.int64)
    gallery_time = build_time_index(gallery_df)

    num_gallery = len(gallery_df)
    num_query = len(query_df)

    cmc = np.zeros(num_gallery, dtype=np.float64)
    ap_list = []
    valid_queries = 0

    for q_idx in tqdm(range(num_query), desc="Evaluating Bayesian"):
        q_feat = query_features[q_idx]  # [2048]
        q_pid = int(query_df.iloc[q_idx]["pid"])
        q_camid = int(query_df.iloc[q_idx]["camid"])
        q_time = int((query_df.iloc[q_idx]["seqid"] - 1) * 200000 + query_df.iloc[q_idx]["frameid"])

        # visual similarity
        scores = gallery_features @ q_feat  # cosine similarity because features are normalized

        # build directional transition
        q_earlier = q_time <= gallery_time
        cam_from = np.where(q_earlier, q_camid, gallery_camids)
        cam_to = np.where(q_earlier, gallery_camids, q_camid)

        delta_t = np.abs(gallery_time - q_time)
        delta_bin_idx = delta_to_bin_index(delta_t)

        cam_prior = cam_prior_mat[cam_from, cam_to]
        delta_prior = delta_prior_tensor[cam_from, cam_to, delta_bin_idx]

        # Bayesian-style additive log prior
        adjusted_scores = (
            scores
            + beta * np.log(cam_prior + eps)
            + gamma * np.log(delta_prior + eps)
        )

        # standard Market-1501 protocol
        invalid = (gallery_pids == q_pid) & (gallery_camids == q_camid)
        valid = ~invalid

        adjusted_scores = adjusted_scores[valid]
        valid_gallery_pids = gallery_pids[valid]

        order = np.argsort(-adjusted_scores)
        ranked_pids = valid_gallery_pids[order]
        matches = (ranked_pids == q_pid).astype(np.int32)

        if matches.sum() == 0:
            continue

        valid_queries += 1

        first_match_index = np.where(matches == 1)[0][0]
        cmc[first_match_index:] += 1

        cumulative_hits = np.cumsum(matches)
        precision_at_k = cumulative_hits / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / matches.sum()
        ap_list.append(ap)

    if valid_queries == 0:
        raise RuntimeError("No valid queries found.")

    cmc = cmc / valid_queries
    mAP = float(np.mean(ap_list))

    return {
        "num_query_total": int(num_query),
        "num_query_valid": int(valid_queries),
        "beta": float(beta),
        "gamma": float(gamma),
        "mAP": mAP,
        "Rank-1": float(cmc[0]),
        "Rank-5": float(cmc[4]),
        "Rank-10": float(cmc[9]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-features", type=str, required=True)
    parser.add_argument("--gallery-features", type=str, required=True)
    parser.add_argument("--query-meta", type=str, required=True)
    parser.add_argument("--gallery-meta", type=str, required=True)
    parser.add_argument("--cam-prior", type=str, required=True)
    parser.add_argument("--delta-prior", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.05)
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    query_features = np.load(args.query_features)
    gallery_features = np.load(args.gallery_features)

    query_df = pd.read_csv(args.query_meta)
    gallery_df = pd.read_csv(args.gallery_meta)

    cam_prior_df = pd.read_csv(args.cam_prior)
    delta_prior_df = pd.read_csv(args.delta_prior)

    cam_prior_mat = build_cam_prior_matrix(cam_prior_df)
    delta_prior_tensor = build_delta_prior_tensor(delta_prior_df)

    results = compute_metrics_with_bayesian_prior(
        query_features=query_features,
        gallery_features=gallery_features,
        query_df=query_df,
        gallery_df=gallery_df,
        cam_prior_mat=cam_prior_mat,
        delta_prior_tensor=delta_prior_tensor,
        beta=args.beta,
        gamma=args.gamma,
    )

    with open(outdir / "bayesian_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== BAYESIAN RESULTS ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nSaved:")
    print(outdir / "bayesian_metrics.json")


if __name__ == "__main__":
    main()