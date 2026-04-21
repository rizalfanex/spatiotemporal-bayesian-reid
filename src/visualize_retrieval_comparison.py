from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_prior_tables(cam_prior_csv: str | None, delta_prior_csv: str | None):
    cam_prior = {}
    delta_prior = {}

    if cam_prior_csv is not None:
        cam_df = pd.read_csv(cam_prior_csv)
        for _, row in cam_df.iterrows():
            cam_prior[(int(row["cam_from"]), int(row["cam_to"]))] = float(row["prob_from_cam"])

    if delta_prior_csv is not None:
        delta_df = pd.read_csv(delta_prior_csv)
        for _, row in delta_df.iterrows():
            delta_prior[(int(row["cam_from"]), int(row["cam_to"]), str(row["delta_bin"]))] = float(row["prob_within_transition"])

    return cam_prior, delta_prior


def get_delta_bin(delta: int) -> str:
    if delta <= 100:
        return "0-100"
    elif delta <= 500:
        return "101-500"
    elif delta <= 1000:
        return "501-1000"
    elif delta <= 5000:
        return "1001-5000"
    elif delta <= 10000:
        return "5001-10000"
    elif delta <= 20000:
        return "10001-20000"
    elif delta <= 50000:
        return "20001-50000"
    else:
        return "50000+"


def compute_baseline_scores(query_feat: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
    return gallery_feats @ query_feat


def compute_bayesian_scores(
    query_feat: np.ndarray,
    query_row: pd.Series,
    gallery_feats: np.ndarray,
    gallery_df: pd.DataFrame,
    beta: float,
    gamma: float,
    cam_prior: dict,
    delta_prior: dict,
) -> np.ndarray:
    visual_scores = gallery_feats @ query_feat
    scores = visual_scores.copy()

    q_cam = int(query_row["camid"])
    q_frame = int(query_row["frameid"])

    bonus = np.zeros(len(gallery_df), dtype=np.float32)

    for i, row in gallery_df.iterrows():
        g_cam = int(row["camid"])
        g_frame = int(row["frameid"])

        cam_prob = cam_prior.get((q_cam, g_cam), 0.0)
        delta_bin = get_delta_bin(abs(q_frame - g_frame))
        delta_prob = delta_prior.get((q_cam, g_cam, delta_bin), 0.0)

        bonus[i] = beta * cam_prob + gamma * delta_prob

    scores += bonus
    return scores


def valid_mask(query_row: pd.Series, gallery_df: pd.DataFrame) -> np.ndarray:
    q_pid = int(query_row["pid"])
    q_cam = int(query_row["camid"])

    mask = ~((gallery_df["pid"].to_numpy() == q_pid) & (gallery_df["camid"].to_numpy() == q_cam))
    return mask


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-scores)
    return order[:k]


def is_correct_top1(query_row: pd.Series, gallery_df: pd.DataFrame, ranked_idx: np.ndarray) -> bool:
    if len(ranked_idx) == 0:
        return False
    top1_pid = int(gallery_df.iloc[ranked_idx[0]]["pid"])
    return top1_pid == int(query_row["pid"])


def open_and_resize(path: str, size=(128, 256)):
    img = Image.open(path).convert("RGB")
    return img.resize(size)


def draw_border(img: Image.Image, color: tuple[int, int, int], width: int = 6):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    return img


def paste_text(draw, xy, text, font, fill=(0, 0, 0)):
    draw.text(xy, text, font=font, fill=fill)


def make_sheet(
    query_row: pd.Series,
    gallery_df: pd.DataFrame,
    baseline_idx: np.ndarray,
    bayesian_idx: np.ndarray,
    out_path: Path,
):
    font = ImageFont.load_default()

    thumb_w, thumb_h = 128, 256
    margin = 20
    header_h = 70
    row_gap = 30
    text_h = 40
    num_cols = max(len(baseline_idx), len(bayesian_idx)) + 1  # +1 for query image

    canvas_w = margin * (num_cols + 1) + thumb_w * num_cols
    canvas_h = header_h + (thumb_h + text_h) * 2 + row_gap + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = (
        f"Query PID={int(query_row['pid'])} | CAM={int(query_row['camid'])} | "
        f"FRAME={int(query_row['frameid'])}"
    )
    paste_text(draw, (margin, 20), title, font)

    # query image
    q_img = open_and_resize(query_row["path"], (thumb_w, thumb_h))
    q_img = draw_border(q_img, (0, 0, 255), width=6)
    qx = margin
    by1 = header_h
    by2 = header_h + thumb_h + text_h + row_gap
    canvas.paste(q_img, (qx, by1))
    canvas.paste(q_img, (qx, by2))
    paste_text(draw, (qx, by1 + thumb_h + 5), "QUERY", font)
    paste_text(draw, (qx, by2 + thumb_h + 5), "QUERY", font)

    paste_text(draw, (margin, by1 - 20), "Baseline Top-K", font)
    paste_text(draw, (margin, by2 - 20), "Bayesian Top-K", font)

    # baseline row
    for rank, idx in enumerate(baseline_idx, start=1):
        row = gallery_df.iloc[idx]
        img = open_and_resize(row["path"], (thumb_w, thumb_h))
        correct = int(row["pid"]) == int(query_row["pid"])
        img = draw_border(img, (0, 180, 0) if correct else (220, 0, 0), width=6)

        x = margin * (rank + 1) + thumb_w * rank
        canvas.paste(img, (x, by1))
        txt = f"R{rank} PID={int(row['pid'])} CAM={int(row['camid'])}"
        paste_text(draw, (x, by1 + thumb_h + 5), txt, font)

    # bayesian row
    for rank, idx in enumerate(bayesian_idx, start=1):
        row = gallery_df.iloc[idx]
        img = open_and_resize(row["path"], (thumb_w, thumb_h))
        correct = int(row["pid"]) == int(query_row["pid"])
        img = draw_border(img, (0, 180, 0) if correct else (220, 0, 0), width=6)

        x = margin * (rank + 1) + thumb_w * rank
        canvas.paste(img, (x, by2))
        txt = f"R{rank} PID={int(row['pid'])} CAM={int(row['camid'])}"
        paste_text(draw, (x, by2 + thumb_h + 5), txt, font)

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-features", type=str, required=True)
    parser.add_argument("--gallery-features", type=str, required=True)
    parser.add_argument("--query-meta", type=str, required=True)
    parser.add_argument("--gallery-meta", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--cam-prior", type=str, default=None)
    parser.add_argument("--delta-prior", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.005)
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    query_feats = np.load(args.query_features)
    gallery_feats = np.load(args.gallery_features)
    query_df = pd.read_csv(args.query_meta)
    gallery_df = pd.read_csv(args.gallery_meta)

    # normalize for safety
    query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
    gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)

    cam_prior, delta_prior = load_prior_tables(args.cam_prior, args.delta_prior)

    selected = []

    for qi in range(len(query_df)):
        q_row = query_df.iloc[qi]
        mask = valid_mask(q_row, gallery_df)

        valid_gallery_df = gallery_df[mask].reset_index(drop=True)
        valid_gallery_feats = gallery_feats[mask]

        base_scores = compute_baseline_scores(query_feats[qi], valid_gallery_feats)
        base_idx = topk_indices(base_scores, args.topk)
        base_ok = is_correct_top1(q_row, valid_gallery_df, base_idx)

        bayes_scores = compute_bayesian_scores(
            query_feats[qi],
            q_row,
            valid_gallery_feats,
            valid_gallery_df,
            args.beta,
            args.gamma,
            cam_prior,
            delta_prior,
        )
        bayes_idx = topk_indices(bayes_scores, args.topk)
        bayes_ok = is_correct_top1(q_row, valid_gallery_df, bayes_idx)

        if (not base_ok) and bayes_ok:
            selected.append((qi, q_row, valid_gallery_df, base_idx, bayes_idx, "improved"))

    # kalau contoh "improved" kurang, tambah contoh bayesian-correct biasa
    if len(selected) < args.num_examples:
        for qi in range(len(query_df)):
            if len(selected) >= args.num_examples:
                break

            q_row = query_df.iloc[qi]
            mask = valid_mask(q_row, gallery_df)

            valid_gallery_df = gallery_df[mask].reset_index(drop=True)
            valid_gallery_feats = gallery_feats[mask]

            base_scores = compute_baseline_scores(query_feats[qi], valid_gallery_feats)
            base_idx = topk_indices(base_scores, args.topk)
            bayes_scores = compute_bayesian_scores(
                query_feats[qi],
                q_row,
                valid_gallery_feats,
                valid_gallery_df,
                args.beta,
                args.gamma,
                cam_prior,
                delta_prior,
            )
            bayes_idx = topk_indices(bayes_scores, args.topk)

            already = any(item[0] == qi for item in selected)
            bayes_ok = is_correct_top1(q_row, valid_gallery_df, bayes_idx)
            if (not already) and bayes_ok:
                selected.append((qi, q_row, valid_gallery_df, base_idx, bayes_idx, "bayes_correct"))

    selected = selected[: args.num_examples]

    summary_rows = []

    for idx, (qi, q_row, valid_gallery_df, base_idx, bayes_idx, tag) in enumerate(selected, start=1):
        out_path = outdir / f"example_{idx:02d}_{tag}_q{qi}.png"
        make_sheet(q_row, valid_gallery_df, base_idx, bayes_idx, out_path)

        summary_rows.append({
            "example_id": idx,
            "tag": tag,
            "query_index": qi,
            "query_pid": int(q_row["pid"]),
            "query_camid": int(q_row["camid"]),
            "baseline_top1_pid": int(valid_gallery_df.iloc[base_idx[0]]["pid"]),
            "bayesian_top1_pid": int(valid_gallery_df.iloc[bayes_idx[0]]["pid"]),
            "saved_path": str(out_path),
        })

    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)

    print("Saved qualitative examples to:")
    print(outdir)
    print(outdir / "summary.csv")


if __name__ == "__main__":
    main()