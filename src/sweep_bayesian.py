from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_bayesian_reid import (
    build_cam_prior_matrix,
    build_delta_prior_tensor,
    compute_metrics_with_bayesian_prior,
)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-features", type=str, required=True)
    parser.add_argument("--gallery-features", type=str, required=True)
    parser.add_argument("--query-meta", type=str, required=True)
    parser.add_argument("--gallery-meta", type=str, required=True)
    parser.add_argument("--cam-prior", type=str, required=True)
    parser.add_argument("--delta-prior", type=str, required=True)
    parser.add_argument("--betas", type=str, default="0.005,0.01,0.02,0.05")
    parser.add_argument("--gammas", type=str, default="0.0,0.005,0.01,0.02")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    betas = parse_float_list(args.betas)
    gammas = parse_float_list(args.gammas)

    query_features = np.load(args.query_features)
    gallery_features = np.load(args.gallery_features)

    query_df = pd.read_csv(args.query_meta)
    gallery_df = pd.read_csv(args.gallery_meta)

    cam_prior_df = pd.read_csv(args.cam_prior)
    delta_prior_df = pd.read_csv(args.delta_prior)

    cam_prior_mat = build_cam_prior_matrix(cam_prior_df)
    delta_prior_tensor = build_delta_prior_tensor(delta_prior_df)

    rows = []

    for beta in betas:
        for gamma in gammas:
            print(f"Running beta={beta}, gamma={gamma} ...")
            metrics = compute_metrics_with_bayesian_prior(
                query_features=query_features,
                gallery_features=gallery_features,
                query_df=query_df,
                gallery_df=gallery_df,
                cam_prior_mat=cam_prior_mat,
                delta_prior_tensor=delta_prior_tensor,
                beta=beta,
                gamma=gamma,
            )
            rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "bayesian_sweep_results.csv", index=False)

    print("\n=== TOP BY RANK-1 ===")
    print(df.sort_values(["Rank-1", "mAP"], ascending=False).head(10).to_string(index=False))

    print("\n=== TOP BY mAP ===")
    print(df.sort_values(["mAP", "Rank-1"], ascending=False).head(10).to_string(index=False))

    print("\nSaved:")
    print(outdir / "bayesian_sweep_results.csv")


if __name__ == "__main__":
    main()