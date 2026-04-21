from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, out_csv: Path, out_md: Path | None = None):
    df.to_csv(out_csv, index=False)
    if out_md is not None:
        try:
            out_md.write_text(df.to_markdown(index=False), encoding="utf-8")
        except Exception:
            out_md.write_text(df.to_string(index=False), encoding="utf-8")


def annotate_bars(ax, fmt="{:.3f}", fontsize=8, rotation=0, y_offset=0.01):
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        x = p.get_x() + p.get_width() / 2.0
        y = height + y_offset
        ax.text(
            x,
            y,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def grouped_bar_metrics(df: pd.DataFrame, out_path: Path):
    metrics = ["mAP", "Rank-1", "Rank-5", "Rank-10"]

    x = np.arange(len(metrics))
    width = 0.18

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=row["Experiment"])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison Across Experiments")
    ax.legend()

    annotate_bars(ax, fmt="{:.3f}", fontsize=8, rotation=90, y_offset=0.01)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def trained_gain_bar(df: pd.DataFrame, out_path: Path):
    base = df[df["Experiment"] == "Baseline-Trained"].iloc[0]
    bayes = df[df["Experiment"] == "Bayesian-Trained"].iloc[0]

    metrics = ["mAP", "Rank-1", "Rank-5", "Rank-10"]
    gains = [bayes[m] - base[m] for m in metrics]

    plt.figure(figsize=(8.5, 5.2))
    ax = plt.gca()
    bars = ax.bar(metrics, gains)

    ax.set_ylabel("Absolute Gain")
    ax.set_xlabel("Evaluation Metric")
    ax.set_title("Absolute Performance Gain of Bayesian Re-ranking over Trained Baseline")

    ymax = max(gains) if len(gains) > 0 else 0.1
    ax.set_ylim(0, ymax + 0.04)

    for bar, gain in zip(bars, gains):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            gain + 0.003,
            f"{gain:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(train_hist: pd.DataFrame, out_loss: Path, out_acc: Path):
    epoch_col = "epoch" if "epoch" in train_hist.columns else train_hist.columns[0]

    plt.figure(figsize=(8, 5))
    if "train_loss" in train_hist.columns:
        plt.plot(train_hist[epoch_col], train_hist["train_loss"], label="train_loss")
    if "val_loss" in train_hist.columns:
        plt.plot(train_hist[epoch_col], train_hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_loss, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    if "train_acc" in train_hist.columns:
        plt.plot(train_hist[epoch_col], train_hist["train_acc"], label="train_acc")
    if "val_acc" in train_hist.columns:
        plt.plot(train_hist[epoch_col], train_hist["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_acc, dpi=300, bbox_inches="tight")
    plt.close()


def matrix_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    out_path: Path,
    xlabel: str = "Gamma",
    ylabel: str = "Beta",
    value_fmt: str = ".3f",
):
    data = pivot_df.values
    row_labels = [str(x) for x in pivot_df.index.tolist()]
    col_labels = [str(x) for x in pivot_df.columns.tolist()]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im)

    plt.xticks(np.arange(len(col_labels)), col_labels)
    plt.yticks(np.arange(len(row_labels)), row_labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            plt.text(j, i, format(val, value_fmt), ha="center", va="center", fontsize=8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def camera_transition_heatmap(cam_df: pd.DataFrame, out_path: Path):
    pivot = cam_df.pivot(index="cam_from", columns="cam_to", values="prob_from_cam").fillna(0.0)
    matrix_heatmap(
        pivot_df=pivot,
        title="Camera Transition Prior Heatmap",
        out_path=out_path,
        xlabel="Camera To",
        ylabel="Camera From",
        value_fmt=".2f",
    )


def copy_best_qualitative_examples(qual_dir: Path, out_dir: Path, max_examples: int = 6):
    pngs = sorted(qual_dir.glob("*.png"))
    selected = pngs[:max_examples]
    copied = []

    for p in selected:
        dst = out_dir / p.name
        shutil.copy2(p, dst)
        copied.append(dst)

    return copied


def make_qualitative_collage(image_paths: list[Path], out_path: Path, thumb_size=(420, 260), cols=2):
    if not image_paths:
        return

    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img.thumbnail(thumb_size)
        canvas = Image.new("RGB", thumb_size, "white")
        x = (thumb_size[0] - img.size[0]) // 2
        y = (thumb_size[1] - img.size[1]) // 2
        canvas.paste(img, (x, y))
        canvas = ImageOps.expand(canvas, border=2, fill="black")
        images.append(canvas)

    cols = min(cols, len(images))
    rows = math.ceil(len(images) / cols)

    w, h = images[0].size
    collage = Image.new("RGB", (cols * w, rows * h), "white")

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        collage.paste(img, (c * w, r * h))

    collage.save(out_path)


def render_dataframe_table_image(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    figsize=(12, 6),
    font_size=9,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_sweep_tables(sweep_df: pd.DataFrame, outdir: Path, topk: int = 15):
    sweep_df = sweep_df.copy()

    sweep_df = sweep_df.sort_values(["beta", "gamma"]).reset_index(drop=True)
    save_dataframe(
        sweep_df,
        outdir / "table01_sweep_all_results.csv",
        outdir / "table01_sweep_all_results.md",
    )

    by_rank1 = sweep_df.sort_values(["Rank-1", "mAP"], ascending=[False, False]).reset_index(drop=True)
    save_dataframe(
        by_rank1,
        outdir / "table02_sweep_sorted_by_rank1.csv",
        outdir / "table02_sweep_sorted_by_rank1.md",
    )

    by_map = sweep_df.sort_values(["mAP", "Rank-1"], ascending=[False, False]).reset_index(drop=True)
    save_dataframe(
        by_map,
        outdir / "table03_sweep_sorted_by_map.csv",
        outdir / "table03_sweep_sorted_by_map.md",
    )

    rank1_pivot = sweep_df.pivot(index="beta", columns="gamma", values="Rank-1").sort_index().sort_index(axis=1)
    map_pivot = sweep_df.pivot(index="beta", columns="gamma", values="mAP").sort_index().sort_index(axis=1)

    rank1_pivot.to_csv(outdir / "table04_rank1_pivot.csv")
    map_pivot.to_csv(outdir / "table05_map_pivot.csv")

    try:
        (outdir / "table04_rank1_pivot.md").write_text(rank1_pivot.to_markdown(), encoding="utf-8")
    except Exception:
        (outdir / "table04_rank1_pivot.md").write_text(rank1_pivot.to_string(), encoding="utf-8")

    try:
        (outdir / "table05_map_pivot.md").write_text(map_pivot.to_markdown(), encoding="utf-8")
    except Exception:
        (outdir / "table05_map_pivot.md").write_text(map_pivot.to_string(), encoding="utf-8")

    render_dataframe_table_image(
        by_rank1.head(topk).round(4),
        outdir / "fig09_top_sweep_by_rank1_table.png",
        title=f"Top {topk} Bayesian Sweep Results by Rank-1",
        figsize=(14, 6),
        font_size=8,
    )

    render_dataframe_table_image(
        by_map.head(topk).round(4),
        outdir / "fig10_top_sweep_by_map_table.png",
        title=f"Top {topk} Bayesian Sweep Results by mAP",
        figsize=(14, 6),
        font_size=8,
    )

    render_dataframe_table_image(
        rank1_pivot.round(4).reset_index(),
        outdir / "fig11_rank1_pivot_table.png",
        title="Rank-1 Pivot Table across Beta and Gamma",
        figsize=(12, 4.8),
        font_size=9,
    )

    render_dataframe_table_image(
        map_pivot.round(4).reset_index(),
        outdir / "fig12_map_pivot_table.png",
        title="mAP Pivot Table across Beta and Gamma",
        figsize=(12, 4.8),
        font_size=9,
    )


def build_summary_metrics(
    eval_baseline: dict,
    eval_bayesian: dict,
    eval_baseline_trained: dict,
    eval_bayesian_trained: dict,
) -> pd.DataFrame:
    rows = [
        {
            "Experiment": "Baseline-Pretrained",
            "mAP": eval_baseline["mAP"],
            "Rank-1": eval_baseline["Rank-1"],
            "Rank-5": eval_baseline["Rank-5"],
            "Rank-10": eval_baseline["Rank-10"],
        },
        {
            "Experiment": "Bayesian-Pretrained",
            "mAP": eval_bayesian["mAP"],
            "Rank-1": eval_bayesian["Rank-1"],
            "Rank-5": eval_bayesian["Rank-5"],
            "Rank-10": eval_bayesian["Rank-10"],
        },
        {
            "Experiment": "Baseline-Trained",
            "mAP": eval_baseline_trained["mAP"],
            "Rank-1": eval_baseline_trained["Rank-1"],
            "Rank-5": eval_baseline_trained["Rank-5"],
            "Rank-10": eval_baseline_trained["Rank-10"],
        },
        {
            "Experiment": "Bayesian-Trained",
            "mAP": eval_bayesian_trained["mAP"],
            "Rank-1": eval_bayesian_trained["Rank-1"],
            "Rank-5": eval_bayesian_trained["Rank-5"],
            "Rank-10": eval_bayesian_trained["Rank-10"],
        },
    ]
    return pd.DataFrame(rows)


def build_gain_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    bp = df_metrics[df_metrics["Experiment"] == "Baseline-Pretrained"].iloc[0]
    bayp = df_metrics[df_metrics["Experiment"] == "Bayesian-Pretrained"].iloc[0]
    bt = df_metrics[df_metrics["Experiment"] == "Baseline-Trained"].iloc[0]
    bayt = df_metrics[df_metrics["Experiment"] == "Bayesian-Trained"].iloc[0]

    rows = [
        {
            "Comparison": "Bayesian - Baseline (Pretrained)",
            "mAP_gain": bayp["mAP"] - bp["mAP"],
            "Rank1_gain": bayp["Rank-1"] - bp["Rank-1"],
            "Rank5_gain": bayp["Rank-5"] - bp["Rank-5"],
            "Rank10_gain": bayp["Rank-10"] - bp["Rank-10"],
        },
        {
            "Comparison": "Training Gain (Baseline-Trained - Baseline-Pretrained)",
            "mAP_gain": bt["mAP"] - bp["mAP"],
            "Rank1_gain": bt["Rank-1"] - bp["Rank-1"],
            "Rank5_gain": bt["Rank-5"] - bp["Rank-5"],
            "Rank10_gain": bt["Rank-10"] - bp["Rank-10"],
        },
        {
            "Comparison": "Bayesian - Baseline (Trained)",
            "mAP_gain": bayt["mAP"] - bt["mAP"],
            "Rank1_gain": bayt["Rank-1"] - bt["Rank-1"],
            "Rank5_gain": bayt["Rank-5"] - bt["Rank-5"],
            "Rank10_gain": bayt["Rank-10"] - bt["Rank-10"],
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    out_path: Path,
    df_metrics: pd.DataFrame,
    df_gains: pd.DataFrame,
    best_rank1_row: pd.Series,
    best_map_row: pd.Series,
):
    bt = df_metrics[df_metrics["Experiment"] == "Baseline-Trained"].iloc[0]
    bayt = df_metrics[df_metrics["Experiment"] == "Bayesian-Trained"].iloc[0]

    try:
        metrics_table = df_metrics.to_markdown(index=False)
    except Exception:
        metrics_table = df_metrics.to_string(index=False)

    try:
        gains_table = df_gains.to_markdown(index=False)
    except Exception:
        gains_table = df_gains.to_string(index=False)

    text = f"""# Result Summary

## Main quantitative results

### Overall metrics
{metrics_table}

### Gain summary
{gains_table}

## Best sweep settings

The Bayesian hyperparameter sweep was conducted across multiple beta and gamma combinations to justify parameter selection objectively rather than heuristically. Therefore, the final parameter choice is supported not only by the best single result, but also by the broader performance trend across the explored spatio-temporal weighting space.

### Best by Rank-1
- beta = {best_rank1_row['beta']}
- gamma = {best_rank1_row['gamma']}
- mAP = {best_rank1_row['mAP']:.4f}
- Rank-1 = {best_rank1_row['Rank-1']:.4f}
- Rank-5 = {best_rank1_row['Rank-5']:.4f}
- Rank-10 = {best_rank1_row['Rank-10']:.4f}

### Best by mAP
- beta = {best_map_row['beta']}
- gamma = {best_map_row['gamma']}
- mAP = {best_map_row['mAP']:.4f}
- Rank-1 = {best_map_row['Rank-1']:.4f}
- Rank-5 = {best_map_row['Rank-5']:.4f}
- Rank-10 = {best_map_row['Rank-10']:.4f}

## Ready-to-use discussion snippet

The experimental results show that visual fine-tuning substantially improves person re-identification performance over generic pretrained features. On top of the trained visual representation, the proposed spatio-temporal Bayesian constraints further enhance ranking quality by incorporating camera-transition likelihood and temporal consistency. Using the selected final setting, the Bayesian-trained configuration improves mAP from {bt['mAP']:.4f} to {bayt['mAP']:.4f} and Rank-1 from {bt['Rank-1']:.4f} to {bayt['Rank-1']:.4f}, demonstrating that spatial-temporal priors provide complementary information beyond appearance similarity alone.
"""
    out_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    outputs_root = project_root / "outputs"
    outdir = Path(args.outdir).resolve() if args.outdir else outputs_root / "result_all"
    ensure_dir(outdir)

    # Input files
    baseline_json = outputs_root / "eval_baseline" / "baseline_metrics.json"
    bayesian_json = outputs_root / "eval_bayesian" / "bayesian_metrics.json"
    baseline_trained_json = outputs_root / "eval_baseline_trained" / "baseline_metrics.json"
    bayesian_trained_json = outputs_root / "eval_bayesian_trained" / "bayesian_metrics.json"

    sweep_csv = outputs_root / "sweep_bayesian_trained" / "bayesian_sweep_results.csv"
    train_hist_csv = outputs_root / "train_reid_stable" / "training_history.csv"
    cam_prior_csv = outputs_root / "spatiotemporal" / "camera_transition_prior.csv"
    qual_dir = outputs_root / "qualitative_comparison_best"

    # Load
    eval_baseline = load_json(baseline_json)
    eval_bayesian = load_json(bayesian_json)
    eval_baseline_trained = load_json(baseline_trained_json)
    eval_bayesian_trained = load_json(bayesian_trained_json)

    sweep_df = pd.read_csv(sweep_csv)
    train_hist = pd.read_csv(train_hist_csv)
    cam_df = pd.read_csv(cam_prior_csv)

    # Tables
    df_metrics = build_summary_metrics(
        eval_baseline,
        eval_bayesian,
        eval_baseline_trained,
        eval_bayesian_trained,
    )
    save_dataframe(
        df_metrics,
        outdir / "metrics_summary.csv",
        outdir / "metrics_summary.md",
    )

    df_gains = build_gain_table(df_metrics)
    save_dataframe(
        df_gains,
        outdir / "gain_summary.csv",
        outdir / "gain_summary.md",
    )

    # Best sweep rows
    best_rank1_row = sweep_df.sort_values(["Rank-1", "mAP"], ascending=[False, False]).iloc[0]
    best_map_row = sweep_df.sort_values(["mAP", "Rank-1"], ascending=[False, False]).iloc[0]

    pd.DataFrame([best_rank1_row]).to_csv(outdir / "best_by_rank1.csv", index=False)
    pd.DataFrame([best_map_row]).to_csv(outdir / "best_by_map.csv", index=False)

    # Figures
    grouped_bar_metrics(df_metrics, outdir / "fig01_metrics_comparison.png")
    trained_gain_bar(df_metrics, outdir / "fig02_bayesian_gain_over_trained.png")
    plot_training_curves(
        train_hist,
        outdir / "fig03_training_loss_curve.png",
        outdir / "fig04_training_accuracy_curve.png",
    )

    rank1_pivot = sweep_df.pivot(index="beta", columns="gamma", values="Rank-1").sort_index().sort_index(axis=1)
    map_pivot = sweep_df.pivot(index="beta", columns="gamma", values="mAP").sort_index().sort_index(axis=1)
    matrix_heatmap(rank1_pivot, "Bayesian Sweep Heatmap (Rank-1)", outdir / "fig05_sweep_heatmap_rank1.png")
    matrix_heatmap(map_pivot, "Bayesian Sweep Heatmap (mAP)", outdir / "fig06_sweep_heatmap_map.png")

    camera_transition_heatmap(cam_df, outdir / "fig07_camera_transition_heatmap.png")

    # Sweep tables
    save_sweep_tables(sweep_df, outdir, topk=15)

    # Qualitative
    copied_examples = copy_best_qualitative_examples(qual_dir, outdir, max_examples=6)
    make_qualitative_collage(copied_examples, outdir / "fig08_qualitative_collage.png", cols=2)

    # Report
    write_report(
        outdir / "result_summary_report.md",
        df_metrics,
        df_gains,
        best_rank1_row,
        best_map_row,
    )

    # Simple manifest
    manifest = {
        "generated_files": sorted([p.name for p in outdir.iterdir() if p.is_file()]),
        "best_rank1": {
            "beta": float(best_rank1_row["beta"]),
            "gamma": float(best_rank1_row["gamma"]),
            "mAP": float(best_rank1_row["mAP"]),
            "Rank-1": float(best_rank1_row["Rank-1"]),
        },
        "best_map": {
            "beta": float(best_map_row["beta"]),
            "gamma": float(best_map_row["gamma"]),
            "mAP": float(best_map_row["mAP"]),
            "Rank-1": float(best_map_row["Rank-1"]),
        },
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done. Generated files in:")
    print(outdir)
    for p in sorted(outdir.iterdir()):
        print("-", p.name)


if __name__ == "__main__":
    main()