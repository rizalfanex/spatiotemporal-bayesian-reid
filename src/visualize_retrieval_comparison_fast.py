from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


DELTA_BIN_LABELS = [
    "0-100",
    "101-500",
    "501-1000",
    "1001-5000",
    "5001-10000",
    "10001-20000",
    "20001-50000",
    "50000+",
]


def load_prior_tables(cam_prior_csv: str | None, delta_prior_csv: str | None, max_cam: int):
    cam_prior_mat = np.zeros((max_cam + 1, max_cam + 1), dtype=np.float32)
    delta_prior_tensor = np.zeros((max_cam + 1, max_cam + 1, len(DELTA_BIN_LABELS)), dtype=np.float32)

    if cam_prior_csv is not None:
        cam_df = pd.read_csv(cam_prior_csv)
        for _, row in cam_df.iterrows():
            cam_from = int(row["cam_from"])
            cam_to = int(row["cam_to"])
            cam_prior_mat[cam_from, cam_to] = float(row["prob_from_cam"])

    if delta_prior_csv is not None:
        delta_df = pd.read_csv(delta_prior_csv)
        label_to_idx = {label: i for i, label in enumerate(DELTA_BIN_LABELS)}
        for _, row in delta_df.iterrows():
            cam_from = int(row["cam_from"])
            cam_to = int(row["cam_to"])
            delta_bin = str(row["delta_bin"])
            idx = label_to_idx[delta_bin]
            delta_prior_tensor[cam_from, cam_to, idx] = float(row["prob_within_transition"])

    return cam_prior_mat, delta_prior_tensor


def delta_to_bin_idx(delta_arr: np.ndarray) -> np.ndarray:
    bins = np.array([100, 500, 1000, 5000, 10000, 20000, 50000], dtype=np.int64)
    return np.digitize(delta_arr, bins, right=True)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.argsort(-scores)
    partial = np.argpartition(-scores, k - 1)[:k]
    return partial[np.argsort(-scores[partial])]


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
    baseline_abs_idx: np.ndarray,
    bayesian_abs_idx: np.ndarray,
    out_path: Path,
):
    font = ImageFont.load_default()

    thumb_w, thumb_h = 128, 256
    margin = 20
    header_h = 70
    row_gap = 30
    text_h = 40
    num_cols = max(len(baseline_abs_idx), len(bayesian_abs_idx)) + 1

    canvas_w = margin * (num_cols + 1) + thumb_w * num_cols
    canvas_h = header_h + (thumb_h + text_h) * 2 + row_gap + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title = (
        f"Query PID={int(query_row['pid'])} | CAM={int(query_row['camid'])} | "
        f"FRAME={int(query_row['frameid'])}"
    )
    paste_text(draw, (margin, 20), title, font)

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

    for rank, abs_idx in enumerate(baseline_abs_idx, start=1):
        row = gallery_df.iloc[abs_idx]
        img = open_and_resize(row["path"], (thumb_w, thumb_h))
        correct = int(row["pid"]) == int(query_row["pid"])
        img = draw_border(img, (0, 180, 0) if correct else (220, 0, 0), width=6)

        x = margin * (rank + 1) + thumb_w * rank
        canvas.paste(img, (x, by1))
        txt = f"R{rank} PID={int(row['pid'])} CAM={int(row['camid'])}"
        paste_text(draw, (x, by1 + thumb_h + 5), txt, font)

    for rank, abs_idx in enumerate(bayesian_abs_idx, start=1):
        row = gallery_df.iloc[abs_idx]
        img = open_and_resize(row["path"], (thumb_w, thumb_h))
        correct = int(row["pid"]) == int(query_row["pid"])
        img = draw_border(img, (0, 180, 0) if correct else (220, 0, 0), width=6)

        x = margin * (rank + 1) + thumb_w * rank
        canvas.paste(img, (x, by2))
        txt = f"R{rank} PID={int(row['pid'])} CAM={int(row['camid'])}"
        paste_text(draw, (x, by2 + thumb_h + 5), txt, font)

    canvas.save(out_path)


def evaluate_one_query(
    q_feat: np.ndarray,
    q_pid: int,
    q_cam: int,
    q_frame: int,
    gallery_feats: np.ndarray,
    gallery_pid: np.ndarray,
    gallery_cam: np.ndarray,
    gallery_frame: np.ndarray,
    cam_prior_mat: np.ndarray,
    delta_prior_tensor: np.ndarray,
    beta: float,
    gamma: float,
    topk: int,
):
    valid_mask = ~((gallery_pid == q_pid) & (gallery_cam == q_cam))
    valid_idx = np.where(valid_mask)[0]

    g_feats = gallery_feats[valid_mask]
    g_pid = gallery_pid[valid_mask]
    g_cam = gallery_cam[valid_mask]
    g_frame = gallery_frame[valid_mask]

    base_scores = g_feats @ q_feat
    base_local_idx = topk_indices(base_scores, topk)
    base_abs_idx = valid_idx[base_local_idx]

    deltas = np.abs(g_frame - q_frame)
    delta_bin_idx = delta_to_bin_idx(deltas)

    bayes_bonus = (
        beta * cam_prior_mat[q_cam, g_cam]
        + gamma * delta_prior_tensor[q_cam, g_cam, delta_bin_idx]
    )
    bayes_scores = base_scores + bayes_bonus
    bayes_local_idx = topk_indices(bayes_scores, topk)
    bayes_abs_idx = valid_idx[bayes_local_idx]

    base_ok = gallery_pid[base_abs_idx[0]] == q_pid
    bayes_ok = gallery_pid[bayes_abs_idx[0]] == q_pid

    return base_abs_idx, bayes_abs_idx, base_ok, bayes_ok


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
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.01)
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    query_feats = np.load(args.query_features).astype(np.float32)
    gallery_feats = np.load(args.gallery_features).astype(np.float32)
    query_df = pd.read_csv(args.query_meta)
    gallery_df = pd.read_csv(args.gallery_meta)

    query_feats /= np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-12
    gallery_feats /= np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-12

    query_pid = query_df["pid"].to_numpy(dtype=np.int32)
    query_cam = query_df["camid"].to_numpy(dtype=np.int32)
    query_frame = query_df["frameid"].to_numpy(dtype=np.int32)

    gallery_pid = gallery_df["pid"].to_numpy(dtype=np.int32)
    gallery_cam = gallery_df["camid"].to_numpy(dtype=np.int32)
    gallery_frame = gallery_df["frameid"].to_numpy(dtype=np.int32)

    max_cam = int(max(query_cam.max(), gallery_cam.max()))
    cam_prior_mat, delta_prior_tensor = load_prior_tables(
        args.cam_prior, args.delta_prior, max_cam
    )

    selected = []
    selected_qidx = set()

    print("Searching improved examples...")

    for qi in tqdm(range(len(query_df))):
        base_abs_idx, bayes_abs_idx, base_ok, bayes_ok = evaluate_one_query(
            query_feats[qi],
            int(query_pid[qi]),
            int(query_cam[qi]),
            int(query_frame[qi]),
            gallery_feats,
            gallery_pid,
            gallery_cam,
            gallery_frame,
            cam_prior_mat,
            delta_prior_tensor,
            args.beta,
            args.gamma,
            args.topk,
        )

        if (not base_ok) and bayes_ok:
            selected.append((qi, base_abs_idx, bayes_abs_idx, "improved"))
            selected_qidx.add(qi)
            if len(selected) >= args.num_examples:
                break

    if len(selected) < args.num_examples:
        print("Improved examples not enough, adding bayesian-correct examples...")
        for qi in tqdm(range(len(query_df))):
            if qi in selected_qidx:
                continue

            base_abs_idx, bayes_abs_idx, base_ok, bayes_ok = evaluate_one_query(
                query_feats[qi],
                int(query_pid[qi]),
                int(query_cam[qi]),
                int(query_frame[qi]),
                gallery_feats,
                gallery_pid,
                gallery_cam,
                gallery_frame,
                cam_prior_mat,
                delta_prior_tensor,
                args.beta,
                args.gamma,
                args.topk,
            )

            if bayes_ok:
                selected.append((qi, base_abs_idx, bayes_abs_idx, "bayes_correct"))
                selected_qidx.add(qi)
                if len(selected) >= args.num_examples:
                    break

    summary_rows = []

    print("Saving qualitative sheets...")

    for idx, (qi, base_abs_idx, bayes_abs_idx, tag) in enumerate(selected, start=1):
        q_row = query_df.iloc[qi]
        out_path = outdir / f"example_{idx:02d}_{tag}_q{qi}.png"
        make_sheet(q_row, gallery_df, base_abs_idx, bayes_abs_idx, out_path)

        summary_rows.append({
            "example_id": idx,
            "tag": tag,
            "query_index": qi,
            "query_pid": int(q_row["pid"]),
            "query_camid": int(q_row["camid"]),
            "baseline_top1_pid": int(gallery_df.iloc[base_abs_idx[0]]["pid"]),
            "bayesian_top1_pid": int(gallery_df.iloc[bayes_abs_idx[0]]["pid"]),
            "saved_path": str(out_path),
        })

    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)

    print("\nSaved qualitative examples to:")
    print(outdir)
    print(outdir / "summary.csv")


if __name__ == "__main__":
    main()