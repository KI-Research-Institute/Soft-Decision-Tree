'''
Implementation for: "Soft Decision Tree classifier: explainable and extendable PyTorch implementation"
Author: Reuben R Shamir

This files analyzes the simulation experiments results
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# === Load CSV ===
RESULTS_DIR = Path("results")
csv_path = RESULTS_DIR / "sim_experiments.csv"
df = pd.read_csv(csv_path)

OUT_DIR  = Path("results/plots_by_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- load ---
df = pd.read_csv(csv_path)

# Optional fixed model order (otherwise alphabetical)
MODEL_ORDER = None
# Example:
# MODEL_ORDER = ["LR", "DT", "RF100", "RF1000", "XGB", "SDT", "SMSDT"]

ANNOT_FONTSIZE = 8

# ===== Load & aggregate =====
required = {"model", "n_features", "n_samples", "auc", "acc"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

agg = (
    df.groupby(["model", "n_features", "n_samples"])
      .agg(auc=("auc", "mean"), acc=("acc", "mean"))
      .reset_index()
)

models_all = sorted(agg["model"].unique()) if MODEL_ORDER is None else MODEL_ORDER
if len(models_all) != 7:
    raise ValueError(f"Expected exactly 7 models for a 4+3 layout, found {len(models_all)}.")

top_models    = models_all[:4]
bottom_models = models_all[4:7]

# Common axes values (sorted)
feats_all = sorted(agg["n_features"].unique())
samps_all = sorted(agg["n_samples"].unique())

def make_matrix(sub, feats, sams, metric):
    mat = np.full((len(feats), len(sams)), np.nan)
    for i, f in enumerate(feats):
        for j, s in enumerate(sams):
            val = sub.loc[(sub["n_features"] == f) & (sub["n_samples"] == s), metric]
            if not val.empty:
                mat[i, j] = float(val.values[0])
    return mat

def annotate_matrix(ax, mat, fmt="{:.3f}", fontsize=ANNOT_FONTSIZE):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize)

def draw_metric_figure(metric_name: str, filename: str):
    """Draw a 4 (top) + 3 (bottom) heatmap panel for a single metric (AUC or ACC)."""
    metric_col = "auc" if metric_name.lower().startswith("auc") else "acc"
    vmin = float(agg[metric_col].min())
    vmax = float(agg[metric_col].max())

    # Figure with 2 rows:
    # row 0 -> 4 heatmaps; row 1 -> 3 heatmaps; last column is the colorbar
    fig = plt.figure(figsize=(4*4 + 1.6, 9))
    gs = GridSpec(
        nrows=2, ncols=5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.18], height_ratios=[1, 1],
        wspace=0.08, hspace=0.18
    )

    # --- TOP ROW (4 panels) ---
    im_ref = None
    for col, m in enumerate(top_models):
        ax = fig.add_subplot(gs[0, col])
        sub = agg[agg["model"] == m]
        mat = make_matrix(sub, feats_all, samps_all, metric_col)
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="Reds", aspect="auto")
        im_ref = im

        annotate_matrix(ax, mat)

        ax.set_title(m, fontsize=11)
        # ticks (show features only on the VERY left; no sample labels on top row)
        ax.set_yticks(np.arange(len(feats_all)))
        if col == 0:
            ax.set_yticklabels(feats_all)
            ax.set_ylabel("n_features")
        else:
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(len(samps_all)))
        ax.set_xticklabels([])  # hide on top row
        ax.set_xlabel("")

    # --- BOTTOM ROW (3 panels) ---
    for col, m in enumerate(bottom_models):
        ax = fig.add_subplot(gs[1, col])
        sub = agg[agg["model"] == m]
        mat = make_matrix(sub, feats_all, samps_all, metric_col)
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="Reds", aspect="auto")
        im_ref = im

        annotate_matrix(ax, mat)

        ax.set_title(m, fontsize=11)
        # ticks (features only on leftmost bottom; samples on ALL bottom panels)
        ax.set_yticks(np.arange(len(feats_all)))
        if col == 0:
            ax.set_yticklabels(feats_all)
            ax.set_ylabel("n_features")
        else:
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(len(samps_all)))
        ax.set_xticklabels(samps_all, rotation=45, ha="right")
        ax.set_xlabel("n_samples")

    # --- Shared colorbar (spans both rows) ---
    cax = fig.add_subplot(gs[:, 4])
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_label(metric_name)

    fig.suptitle(
        f"{metric_name} heatmaps across models",
        fontsize=13, y=0.98
    )
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.resolve()}")

# === Build both figures ===
draw_metric_figure("AUC", "heatmaps_auc_4_top_3_bottom_labels_clean.png")
draw_metric_figure("Accuracy", "heatmaps_acc_4_top_3_bottom_labels_clean.png")
