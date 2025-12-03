'''
Implementation for: "Soft Decision Tree classifier: explainable and extendable PyTorch implementation"
Author: Reuben R Shamir

This file executes the multiple simulations experiment
'''

# =============================
# File: sim_experiments.py
# =============================
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import torch

# --- Your models from the second snippet ---
from SoftDecisionTree import SoftBinaryDecisionTree
from ShortMemorySoftDecisionTree import ShortMemorySoftBinaryTree

# ----------------------
# Config & Paths
# ----------------------
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TRAINED_MODELS_DIR = Path("/home/ruby/Research/data/03_methods_paper/SDT_trained_models")
TRAINED_MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Simulation grid
SEEDS = [15, 21, 42, 57, 76, 82, 63, 97, 77, 715, 721, 472, 577, 776, 872, 673, 797]
N_FEATURES_LIST = [50, 100, 200]
N_SAMPLES_LIST = [100, 500, 1000, 2000, 8000]

MAX_DEPTH = 3
EPOCHS_NUM = 300
TEMP = 0.8

# Use this settings for fast validating that everything works ####################
# SEEDS = [15, 21]
# N_FEATURES_LIST = [50, 100]
# N_SAMPLES_LIST = [500, 600]
#
# MAX_DEPTH = 2
# EPOCHS_NUM = 30
# TEMP = 0.8


def generate_split(n_samples: int, n_features: int, seed: int):
    """Simulate data like your first chunk, then split train/test & scale (fit on train)."""
    # Ensure informative + redundant <= n_features
    n_informative = min(30, max(2, n_features - 10))
    n_redundant = min(5, max(0, n_features - n_informative - 5))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        class_sep=0.8,
        flip_y=0.1,
        random_state=seed,
    )

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)

    # Internal validation for SDT
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=seed, stratify=y_train_np
    )

    # Tensors
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    y_val = torch.tensor(y_val_np, dtype=torch.long)

    return (X_train_np, y_train_np, X_test_np, y_test_np, X_tr, y_tr, X_val, y_val, X_test)


def run_one_setting(seed: int, n_features: int, n_samples: int):
    (
        X_train_np,
        y_train_np,
        X_test_np,
        y_test_np,
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_test,
    ) = generate_split(n_samples, n_features, seed)

    n_classes = int(len(np.unique(y_train_np)))
    n_features_eff = X_tr.shape[1]

    rows = []

    def add(model_name, auc, acc, elapsed):
        rows.append({
            "seed": seed,
            "n_features": n_features,
            "n_samples": n_samples,
            "model": model_name,
            "auc": float(auc),
            "acc": float(acc),
            "time_sec": float(elapsed),
        })

    # ===== Baselines (like the second file) =====
    t0 = time.time()
    lr = LogisticRegression(max_iter=600, random_state=seed)
    lr.fit(X_train_np, y_train_np)
    proba = lr.predict_proba(X_test_np)[:, 1]
    pred = lr.predict(X_test_np)
    add("LR", roc_auc_score(y_test_np, proba), accuracy_score(y_test_np, pred), time.time() - t0)

    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=seed)
    dt.fit(X_train_np, y_train_np)
    proba = dt.predict_proba(X_test_np)[:, 1]
    pred = dt.predict(X_test_np)
    add("DT", roc_auc_score(y_test_np, proba), accuracy_score(y_test_np, pred), time.time() - t0)

    t0 = time.time()
    rf100 = RandomForestClassifier(n_estimators=100, max_depth=MAX_DEPTH, random_state=seed, n_jobs=-1)
    rf100.fit(X_train_np, y_train_np)
    proba = rf100.predict_proba(X_test_np)[:, 1]
    pred = rf100.predict(X_test_np)
    add("RF100", roc_auc_score(y_test_np, proba), accuracy_score(y_test_np, pred), time.time() - t0)

    t0 = time.time()
    rf1000 = RandomForestClassifier(n_estimators=1000, max_depth=MAX_DEPTH, random_state=seed, n_jobs=-1)
    rf1000.fit(X_train_np, y_train_np)
    proba = rf1000.predict_proba(X_test_np)[:, 1]
    pred = rf1000.predict(X_test_np)
    add("RF1000", roc_auc_score(y_test_np, proba), accuracy_score(y_test_np, pred), time.time() - t0)

    t0 = time.time()
    xgb = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", use_label_encoder=False,
        max_depth=MAX_DEPTH, random_state=seed, n_estimators=200, n_jobs=-1
    )
    xgb.fit(X_train_np, y_train_np)
    proba = xgb.predict_proba(X_test_np)[:, 1]
    pred = xgb.predict(X_test_np)
    add("XGB", roc_auc_score(y_test_np, proba), accuracy_score(y_test_np, pred), time.time() - t0)

    # ===== Soft Trees (train on X_tr/y_tr, validate on X_val/y_val) =====
    batch_size = max(32, min(int(len(X_tr) // 10), 1000))

    t0 = time.time()
    sdt = SoftBinaryDecisionTree(
        input_dim=n_features_eff, num_classes=n_classes, max_depth=MAX_DEPTH, temperature=TEMP
    )
    sdt.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        model_prefix="SDT",
        output_folder=str(TRAINED_MODELS_DIR),
        num_epochs=EPOCHS_NUM,
        batch_size=batch_size,
        learning_rate=1e-3,
    )
    sdt.load_trained_model(os.path.join(TRAINED_MODELS_DIR, "SDT_soft_decision_tree.pt"))
    proba_mat, _ = sdt.predict(X_test)
    proba = proba_mat[:, 1]
    pred = np.argmax(proba_mat, axis=1)
    add("SDT", roc_auc_score(y_test_np, proba), accuracy_score(y_test_np, pred), time.time() - t0)

    t0 = time.time()
    smsdt = ShortMemorySoftBinaryTree(
        input_dim=n_features_eff, num_classes=n_classes, max_depth=MAX_DEPTH,
        temperature=TEMP, effective_transform_output_dim=min(16, n_features_eff)
    )
    smsdt.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        model_prefix="SMSDT",
        output_folder=str(TRAINED_MODELS_DIR),
        num_epochs=EPOCHS_NUM,
        batch_size=batch_size,
        learning_rate=1e-3,
    )
    smsdt.load_trained_model(os.path.join(TRAINED_MODELS_DIR, "SMSDT_short_memory_soft_decision_tree.pt"))
    proba_mat, _ = smsdt.predict(X_test)
    proba = proba_mat[:, 1]
    pred = np.argmax(proba_mat, axis=1)
    add("SMSDT", roc_auc_score(y_test_np, proba), accuracy_score(y_test_np, pred), time.time() - t0)

    return rows


def main():
    all_rows = []
    for seed in SEEDS:
        for n_features in N_FEATURES_LIST:
            for n_samples in N_SAMPLES_LIST:
                try:
                    rows = run_one_setting(seed, n_features, n_samples)
                    all_rows.extend(rows)
                    print(f"[OK] seed={seed} features={n_features} samples={n_samples}")
                except Exception as e:
                    print(f"[ERR] seed={seed} features={n_features} samples={n_samples}: {e}")

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "sim_experiments.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved raw results to {out_csv}")


if __name__ == "__main__":
    main()

