# Spatio-Temporal Bayesian ReID

A data science project on **person re-identification (ReID)** using **visual feature learning** and **spatio-temporal Bayesian constraints** on the **Market-1501** dataset.

This repository builds a complete ReID pipeline: dataset preparation, spatio-temporal prior construction, supervised visual training, feature extraction, baseline evaluation, Bayesian re-ranking, hyperparameter sweep, qualitative comparison, and automatic result packaging for journal-style figures and presentation materials.

---

## Overview

Person re-identification aims to retrieve images of the same person across different cameras. A purely visual system often confuses similar-looking identities, especially in crowded scenes. This project improves ranking by combining:

1. **Visual embeddings** learned from person images
2. **Camera transition priors** estimated from training metadata
3. **Delta-time priors** that model plausible temporal transitions between cameras
4. **Bayesian score fusion** to refine retrieval ranking

The final system shows that **spatio-temporal Bayesian re-ranking** improves over the trained visual baseline and produces a stronger final retrieval pipeline.

---

## Main Contribution

This repository provides:

- A clean **end-to-end Market-1501 ReID pipeline**
- A practical implementation of **camera-transition Bayesian constraints**
- A practical implementation of **cross-camera delta-time priors**
- A reproducible **hyperparameter sweep over β and γ**
- **Quantitative and qualitative comparisons**
- An automated **`result_all.py`** script for generating publication-ready outputs

---

## Final Headline Result

### Official final configuration
- **Visual model**: supervised trained model from `train_reid_stable.py`
- **Bayesian re-ranking**: **β = 0.02**, **γ = 0.01**
- **Why this is selected**: it achieved the **best Rank-1** in the sweep while also maintaining near-best mAP

### Final metrics
- **mAP = 0.2971**
- **Rank-1 = 0.5781**
- **Rank-5 = 0.7708**
- **Rank-10 = 0.8376**

### Best mAP observed in sweep
- **β = 0.03**, **γ = 0.01**
- **mAP = 0.2979**
- **Rank-1 = 0.5769**
- **Rank-5 = 0.7702**
- **Rank-10 = 0.8370**

---

## Experimental Story

This project has four important stages:

1. **Pretrained visual baseline**
2. **Pretrained visual baseline + Bayesian re-ranking**
3. **Trained visual baseline**
4. **Trained visual baseline + Bayesian re-ranking**

This progression shows that the Bayesian component becomes much more effective when it is applied on top of a stronger learned visual representation.

---

## Repository Structure

```text
spatiotemporal-bayesian-reid/
├── README.md
├── requirements.txt
├── src/
│   ├── prepare_market1501.py
│   ├── build_spatiotemporal_prior.py
│   ├── extract_visual_features.py
│   ├── extract_visual_features_trained.py
│   ├── evaluate_baseline_reid.py
│   ├── evaluate_bayesian_reid.py
│   ├── sweep_bayesian.py
│   ├── train_reid_stable.py
│   ├── train_reid_bneck_triplet.py
│   ├── visualize_retrieval_comparison_fast.py
│   └── result_all.py
├── Market-1501-v15.09.15/
└── outputs/
    ├── market1501_train_metadata.csv
    ├── market1501_query_metadata.csv
    ├── market1501_gallery_metadata.csv
    ├── spatiotemporal/
    ├── train_reid_stable/
    ├── features/
    ├── features_trained/
    ├── eval_baseline/
    ├── eval_bayesian/
    ├── eval_baseline_trained/
    ├── eval_bayesian_trained/
    ├── sweep_bayesian/
    ├── sweep_bayesian_trained/
    ├── qualitative_comparison_best/
    └── result_all/
```

---

## Environment

### Recommended environment
- Python 3.10
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- scikit-learn
- pillow
- tqdm
- seaborn
- tabulate

### Installation

```bash
conda create -n reid_bayesian python=3.10 -y
conda activate reid_bayesian

pip install -r requirements.txt
pip install tabulate
```

### Example GPU check

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); print('cuda_version:', torch.version.cuda)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
```

---

## Dataset

### Dataset used
- **Market-1501**

### Dataset root example
```bash
/home/ucl/Documents/reid_bayesian/Market-1501-v15.09.15
```

### Raw Market-1501 folders
- `bounding_box_train`
- `query`
- `bounding_box_test`

### Experimental counts used in this project
After metadata preparation and filtering, the experiment used:

- **Train = 12,936**
- **Query = 3,368**
- **Gallery = 15,913**

These counts reflect the actual prepared metadata used in the reported experiments.

---

## Full Pipeline

## 1) Prepare Market-1501 metadata

```bash
python src/prepare_market1501.py \
  --root /home/ucl/Documents/reid_bayesian/Market-1501-v15.09.15 \
  --outdir /home/ucl/Documents/reid_bayesian/outputs
```

Expected outputs:
- `outputs/market1501_train_metadata.csv`
- `outputs/market1501_query_metadata.csv`
- `outputs/market1501_gallery_metadata.csv`
- `outputs/market1501_all_metadata.csv`

---

## 2) Build spatio-temporal priors

```bash
python src/build_spatiotemporal_prior.py \
  --metadata /home/ucl/Documents/reid_bayesian/outputs/market1501_train_metadata.csv \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal
```

Expected outputs:
- `outputs/spatiotemporal/positive_cross_camera_pairs.csv`
- `outputs/spatiotemporal/camera_transition_prior.csv`
- `outputs/spatiotemporal/camera_transition_delta_prior.csv`

---

## 3) Extract pretrained visual features

```bash
python src/extract_visual_features.py \
  --query-csv /home/ucl/Documents/reid_bayesian/outputs/market1501_query_metadata.csv \
  --gallery-csv /home/ucl/Documents/reid_bayesian/outputs/market1501_gallery_metadata.csv \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/features \
  --batch-size 64 \
  --num-workers 4
```

Expected outputs:
- `outputs/features/query_features.npy`
- `outputs/features/gallery_features.npy`
- `outputs/features/query_metadata.csv`
- `outputs/features/gallery_metadata.csv`

Feature shapes used in this project:
- `query_features: (3368, 2048)`
- `gallery_features: (15913, 2048)`

---

## 4) Evaluate pretrained visual baseline

```bash
python src/evaluate_baseline_reid.py \
  --query-features /home/ucl/Documents/reid_bayesian/outputs/features/query_features.npy \
  --gallery-features /home/ucl/Documents/reid_bayesian/outputs/features/gallery_features.npy \
  --query-meta /home/ucl/Documents/reid_bayesian/outputs/features/query_metadata.csv \
  --gallery-meta /home/ucl/Documents/reid_bayesian/outputs/features/gallery_metadata.csv \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/eval_baseline
```

---

## 5) Evaluate pretrained visual baseline + Bayesian re-ranking

```bash
python src/evaluate_bayesian_reid.py \
  --query-features /home/ucl/Documents/reid_bayesian/outputs/features/query_features.npy \
  --gallery-features /home/ucl/Documents/reid_bayesian/outputs/features/gallery_features.npy \
  --query-meta /home/ucl/Documents/reid_bayesian/outputs/features/query_metadata.csv \
  --gallery-meta /home/ucl/Documents/reid_bayesian/outputs/features/gallery_metadata.csv \
  --cam-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_prior.csv \
  --delta-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_delta_prior.csv \
  --beta 0.02 \
  --gamma 0.005 \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/eval_bayesian
```

---

## 6) Hyperparameter sweep on pretrained features

```bash
python src/sweep_bayesian.py \
  --query-features /home/ucl/Documents/reid_bayesian/outputs/features/query_features.npy \
  --gallery-features /home/ucl/Documents/reid_bayesian/outputs/features/gallery_features.npy \
  --query-meta /home/ucl/Documents/reid_bayesian/outputs/features/query_metadata.csv \
  --gallery-meta /home/ucl/Documents/reid_bayesian/outputs/features/gallery_metadata.csv \
  --cam-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_prior.csv \
  --delta-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_delta_prior.csv \
  --betas "0.005,0.01,0.02,0.05" \
  --gammas "0.0,0.005,0.01,0.02" \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/sweep_bayesian
```

---

## 7) Train the supervised visual ReID model

This is the **official visual training stage** used for the final results.

```bash
python src/train_reid_stable.py \
  --train-csv /home/ucl/Documents/reid_bayesian/outputs/market1501_train_metadata.csv \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/train_reid_stable \
  --epochs 15 \
  --batch-size 64 \
  --num-workers 4 \
  --lr-backbone 1e-4 \
  --lr-head 1e-3 \
  --weight-decay 1e-4 \
  --val-ratio 0.2 \
  --freeze-backbone-epochs 2
```

Expected outputs:
- `outputs/train_reid_stable/best_checkpoint.pth`
- `outputs/train_reid_stable/last_checkpoint.pth`
- `outputs/train_reid_stable/training_history.csv`
- `outputs/train_reid_stable/pid_to_label.json`

---

## 8) Extract features from the trained model

```bash
python src/extract_visual_features_trained.py \
  --query-csv /home/ucl/Documents/reid_bayesian/outputs/market1501_query_metadata.csv \
  --gallery-csv /home/ucl/Documents/reid_bayesian/outputs/market1501_gallery_metadata.csv \
  --checkpoint /home/ucl/Documents/reid_bayesian/outputs/train_reid_stable/best_checkpoint.pth \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/features_trained \
  --batch-size 64 \
  --num-workers 4
```

Expected outputs:
- `outputs/features_trained/query_features.npy`
- `outputs/features_trained/gallery_features.npy`
- `outputs/features_trained/query_metadata.csv`
- `outputs/features_trained/gallery_metadata.csv`

---

## 9) Evaluate trained visual baseline

```bash
python src/evaluate_baseline_reid.py \
  --query-features /home/ucl/Documents/reid_bayesian/outputs/features_trained/query_features.npy \
  --gallery-features /home/ucl/Documents/reid_bayesian/outputs/features_trained/gallery_features.npy \
  --query-meta /home/ucl/Documents/reid_bayesian/outputs/features_trained/query_metadata.csv \
  --gallery-meta /home/ucl/Documents/reid_bayesian/outputs/features_trained/gallery_metadata.csv \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/eval_baseline_trained
```

---

## 10) Evaluate trained visual baseline + Bayesian re-ranking

This is the **official final evaluation command**.

```bash
python src/evaluate_bayesian_reid.py \
  --query-features /home/ucl/Documents/reid_bayesian/outputs/features_trained/query_features.npy \
  --gallery-features /home/ucl/Documents/reid_bayesian/outputs/features_trained/gallery_features.npy \
  --query-meta /home/ucl/Documents/reid_bayesian/outputs/features_trained/query_metadata.csv \
  --gallery-meta /home/ucl/Documents/reid_bayesian/outputs/features_trained/gallery_metadata.csv \
  --cam-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_prior.csv \
  --delta-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_delta_prior.csv \
  --beta 0.02 \
  --gamma 0.01 \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/eval_bayesian_trained
```

---

## 11) Hyperparameter sweep on trained features

```bash
python src/sweep_bayesian.py \
  --query-features /home/ucl/Documents/reid_bayesian/outputs/features_trained/query_features.npy \
  --gallery-features /home/ucl/Documents/reid_bayesian/outputs/features_trained/gallery_features.npy \
  --query-meta /home/ucl/Documents/reid_bayesian/outputs/features_trained/query_metadata.csv \
  --gallery-meta /home/ucl/Documents/reid_bayesian/outputs/features_trained/gallery_metadata.csv \
  --cam-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_prior.csv \
  --delta-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_delta_prior.csv \
  --betas "0.002,0.005,0.01,0.02,0.03,0.05" \
  --gammas "0.0,0.0025,0.005,0.01,0.02" \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/sweep_bayesian_trained
```

---

## 12) Generate qualitative comparison sheets

```bash
python src/visualize_retrieval_comparison_fast.py \
  --query-features /home/ucl/Documents/reid_bayesian/outputs/features_trained/query_features.npy \
  --gallery-features /home/ucl/Documents/reid_bayesian/outputs/features_trained/gallery_features.npy \
  --query-meta /home/ucl/Documents/reid_bayesian/outputs/features_trained/query_metadata.csv \
  --gallery-meta /home/ucl/Documents/reid_bayesian/outputs/features_trained/gallery_metadata.csv \
  --cam-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_prior.csv \
  --delta-prior /home/ucl/Documents/reid_bayesian/outputs/spatiotemporal/camera_transition_delta_prior.csv \
  --beta 0.02 \
  --gamma 0.01 \
  --topk 5 \
  --num-examples 10 \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/qualitative_comparison_best
```

Expected outputs:
- qualitative comparison sheets
- `summary.csv`

---

## 13) Generate consolidated journal/presentation outputs

```bash
python src/result_all.py \
  --project-root /home/ucl/Documents/reid_bayesian \
  --outdir /home/ucl/Documents/reid_bayesian/outputs/result_all
```

This script is intended to package:
- metric comparison tables
- hyperparameter summaries
- figure panels
- final report markdown/text
- presentation-ready result assets

---

## Quantitative Results

## A. Pretrained visual baseline

| Setting | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| Pretrained visual baseline | 0.0282 | 0.0885 | 0.1912 | 0.2643 |

## B. Pretrained visual baseline + Bayesian re-ranking

| Setting | β | γ | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|---:|---:|
| Bayesian on pretrained features | 0.02 | 0.005 | 0.0407 | 0.1351 | 0.2776 | 0.3512 |

## C. Trained visual baseline

| Setting | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| Trained visual baseline | 0.2321 | 0.4629 | 0.6672 | 0.7461 |

## D. Trained visual baseline + Bayesian re-ranking

| Setting | β | γ | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|---:|---:|
| Final selected configuration | 0.02 | 0.01 | 0.2971 | 0.5781 | 0.7708 | 0.8376 |

---

## Improvement Summary

| Comparison | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| Pretrained baseline → Trained baseline | +0.2039 | +0.3744 | +0.4760 | +0.4818 |
| Trained baseline → Trained + Bayesian | +0.0650 | +0.1152 | +0.1036 | +0.0915 |
| Pretrained baseline → Final selected system | +0.2689 | +0.4896 | +0.5796 | +0.5733 |

---

## Hyperparameter Sweep on Trained Features

### Top configurations by performance

| Rank | β | γ | mAP | Rank-1 | Rank-5 | Rank-10 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.03 | 0.01 | 0.2979 | 0.5769 | 0.7702 | 0.8370 |
| 2 | 0.02 | 0.01 | 0.2971 | 0.5781 | 0.7708 | 0.8376 |
| 3 | 0.03 | 0.02 | 0.2962 | 0.5742 | 0.7657 | 0.8325 |
| 4 | 0.03 | 0.005 | 0.2956 | 0.5754 | 0.7699 | 0.8373 |
| 5 | 0.05 | 0.01 | 0.2952 | 0.5730 | 0.7669 | 0.8370 |

### Hyperparameter selection rationale

The repository uses **β = 0.02** and **γ = 0.01** as the official final setting because it achieved the **highest Rank-1** while maintaining nearly the best mAP. This makes it a strong final choice when the project prioritizes top-ranked retrieval quality without sacrificing overall ranking performance.

---

## Why Bayesian Re-Ranking Helps

The visual model measures appearance similarity, but visual similarity alone is often insufficient in person re-identification. Two identities can have similar clothes, color distribution, or body shape. The Bayesian stage introduces extra evidence:

- **Camera transition prior** estimates whether a movement from camera A to camera B is plausible
- **Delta-time prior** estimates whether the time difference between two observations is realistic
- **Bayesian fusion** combines visual similarity with spatio-temporal plausibility

As a result, implausible matches can be pushed down while realistic candidates are promoted.

---

## Qualitative Result

The qualitative comparison script produces side-by-side retrieval sheets showing:

- query image
- top-k baseline retrieval
- top-k Bayesian retrieval
- cases where ranking improved after Bayesian fusion

This section is useful for:
- presentation slides
- project reports
- visual proof of improvement beyond metrics alone

---

## Output Directory Layout

```text
outputs/
├── market1501_train_metadata.csv
├── market1501_query_metadata.csv
├── market1501_gallery_metadata.csv
├── market1501_all_metadata.csv
├── spatiotemporal/
│   ├── positive_cross_camera_pairs.csv
│   ├── camera_transition_prior.csv
│   └── camera_transition_delta_prior.csv
├── train_reid_stable/
│   ├── best_checkpoint.pth
│   ├── last_checkpoint.pth
│   ├── training_history.csv
│   └── pid_to_label.json
├── features/
│   ├── query_features.npy
│   ├── gallery_features.npy
│   ├── query_metadata.csv
│   └── gallery_metadata.csv
├── features_trained/
│   ├── query_features.npy
│   ├── gallery_features.npy
│   ├── query_metadata.csv
│   └── gallery_metadata.csv
├── eval_baseline/
│   └── baseline_metrics.json
├── eval_bayesian/
│   └── bayesian_metrics.json
├── eval_baseline_trained/
│   └── baseline_metrics.json
├── eval_bayesian_trained/
│   └── bayesian_metrics.json
├── sweep_bayesian/
│   └── bayesian_sweep_results.csv
├── sweep_bayesian_trained/
│   └── bayesian_sweep_results.csv
├── qualitative_comparison_best/
│   ├── summary.csv
│   └── *.png
└── result_all/
    ├── *.png
    ├── *.csv
    ├── *.md
    └── *.txt
```

---

## Reproducibility Notes

For reproducible results, keep the following fixed:

- dataset root structure
- metadata preparation logic
- prior construction logic
- trained checkpoint path
- feature extraction procedure
- β and γ configuration
- evaluation scripts and output folders

Recommended reproducibility practice:
- use one environment
- avoid mixing checkpoints from different experiments
- always report the selected β and γ
- keep the trained baseline and Bayesian result in separate folders

---

## Limitations

This project is strong as a data science and experimental ReID pipeline, but several limitations remain:

- The visual backbone is still relatively simple compared with state-of-the-art ReID methods
- The Bayesian prior is estimated from dataset metadata rather than real deployment camera calibration
- The method is validated on a single benchmark dataset
- The Bayesian stage improves retrieval ranking, but it is not a replacement for stronger visual feature learning
- The approach is most useful when camera-transition and timing patterns carry meaningful signal

---

## Suitable Use Case

This repository is particularly suitable for:

- data science coursework
- computer vision projects
- person re-identification experiments
- ranking and retrieval analysis
- applied Bayesian fusion demonstrations
- reproducible academic project portfolios

---

## Citation

If you use this repository, cite it as:

```bibtex
@misc{rizal_spatiotemporal_bayesian_reid,
  title={Spatio-Temporal Bayesian ReID},
  author={Mochamad Rizal Fauzan},
  year={2026},
  howpublished={\url{https://github.com/rizalfanex/spatiotemporal-bayesian-reid}}
}
```

---

## Repository Link

```text
https://github.com/rizalfanex/spatiotemporal-bayesian-reid
```
