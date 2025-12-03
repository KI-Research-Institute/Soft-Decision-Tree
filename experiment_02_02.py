# experiments.py
import os, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import torch

from SoftDecisionTree import SoftBinaryDecisionTree
from ShortMemorySoftDecisionTree import ShortMemorySoftBinaryTree

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TRAINED_MODELS_DIR = Path("/home/ruby/Research/data/03_methods_paper/SDT_trained_models")
TRAINED_MODELS_DIR.mkdir(exist_ok=True, parents=True)

DATA_DIR = Path("data_prepared")
DATASETS = ["breast_cancer","pima_diabetes","heart_cleveland","thyroid_sick","dermatology","ilpd","stroke","local_disease5","heart_failure"]

SEEDS = list(range(1, 21))  # random states
EPOCHS_NUM = 360

def load_dataset(name: str):
    X_train_np = np.load(DATA_DIR / name / "X_train.npy")
    y_train_np = np.load(DATA_DIR / name / "y_train.npy")
    X_test_np  = np.load(DATA_DIR / name / "X_test.npy")
    y_test_np  = np.load(DATA_DIR / name / "y_test.npy")
    # merge then re-split (fresh random test)
    X_all = np.concatenate([X_train_np, X_test_np], axis=0)
    y_all = np.concatenate([y_train_np, y_test_np], axis=0)
    return X_all, y_all

def preprocess_train_test(X_train_np, y_train_np, X_test_np, k_features=16, corr_thr=0.9, seed=42):
    # 1) drop highly correlated features (fit on train only)
    corr = np.corrcoef(X_train_np, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(np.abs(corr))
    to_drop = [j for j in range(corr.shape[1]) if np.any(corr[:j, j] > corr_thr)]
    keep_idx = np.array([i for i in range(corr.shape[1]) if i not in set(to_drop)], dtype=int)
    X_train_np = X_train_np[:, keep_idx]
    X_test_np  = X_test_np[:,  keep_idx]

    # 2) oversample train only
    ros = RandomOverSampler(random_state=71)
    X_train_np, y_train_np = ros.fit_resample(X_train_np, y_train_np)

    # 3) SelectKBest on train; transform test
    k = min(k_features, X_train_np.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_new = selector.fit_transform(X_train_np, y_train_np)
    X_test_new  = selector.transform(X_test_np)

    # 4) scale (fit on train; transform test)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_new)
    X_test_np  = scaler.transform(X_test_new)

    return X_train_np, y_train_np, X_test_np, keep_idx  # keep_idx optional to trace back

def run_one_split(dataset_name: str, seed: int, max_depth=3, epochs_num=360):
    # fresh split
    X_all, y_all = load_dataset(dataset_name)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed,
        stratify=y_all if len(np.unique(y_all)) > 1 else None
    )

    # preprocess
    X_train_np, y_train_np, X_test_np, _ = preprocess_train_test(
        X_train_np, y_train_np, X_test_np, k_features=16, corr_thr=0.9, seed=seed
    )

    # internal val split (from processed train)
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=seed, stratify=y_train_np
    )

    # tensors for SDT
    X_tr  = torch.tensor(X_tr_np,  dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test= torch.tensor(X_test_np,dtype=torch.float32)
    y_tr  = torch.tensor(y_tr_np,  dtype=torch.long)
    y_val = torch.tensor(y_val_np, dtype=torch.long)
    n_features = X_tr.shape[1]

    results = []
    def add_result(model_name, auc, acc, duration):
        results.append({
            "dataset": dataset_name,
            "seed": seed,
            "model": model_name,
            "auc": float(auc),
            "acc": float(acc),
            "time_sec": float(duration)
        })

    # === Baselines (fit on processed train, eval on processed test) ===
    # Logistic Regression
    t0 = time.time()
    lr = LogisticRegression(max_iter=600, random_state=seed)
    lr.fit(X_train_np, y_train_np)
    y_proba = lr.predict_proba(X_test_np)[:, 1]
    y_pred  = lr.predict(X_test_np)
    add_result("LR", roc_auc_score(y_test_np, y_proba), accuracy_score(y_test_np, y_pred), time.time()-t0)

    # Decision Tree
    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    dt.fit(X_train_np, y_train_np)
    y_proba = dt.predict_proba(X_test_np)[:, 1]
    y_pred  = dt.predict(X_test_np)
    add_result("DT", roc_auc_score(y_test_np, y_proba), accuracy_score(y_test_np, y_pred), time.time()-t0)

    # RF 100
    t0 = time.time()
    rf100 = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=seed, n_jobs=-1)
    rf100.fit(X_train_np, y_train_np)
    y_proba = rf100.predict_proba(X_test_np)[:, 1]
    y_pred  = rf100.predict(X_test_np)
    add_result("RF100", roc_auc_score(y_test_np, y_proba), accuracy_score(y_test_np, y_pred), time.time()-t0)

    # RF 1000
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=1000, max_depth=max_depth, random_state=seed, n_jobs=-1)
    rf.fit(X_train_np, y_train_np)
    y_proba = rf.predict_proba(X_test_np)[:, 1]
    y_pred  = rf.predict(X_test_np)
    add_result("RF1000", roc_auc_score(y_test_np, y_proba), accuracy_score(y_test_np, y_pred), time.time()-t0)

    # XGBoost
    t0 = time.time()
    xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                        max_depth=max_depth, random_state=seed, n_estimators=200, n_jobs=-1)
    xgb.fit(X_train_np, y_train_np)
    y_proba = xgb.predict_proba(X_test_np)[:, 1]
    y_pred  = xgb.predict(X_test_np)
    add_result("XGB", roc_auc_score(y_test_np, y_proba), accuracy_score(y_test_np, y_pred), time.time()-t0)

    # === Soft Trees (train on X_tr/y_tr, validate on X_val/y_val, test on X_test) ===
    batch_size = max(32, min(int(len(X_tr) // 10), 1000))

    # SDT
    t0 = time.time()
    sdt = SoftBinaryDecisionTree(
        input_dim=n_features, num_classes=len(np.unique(y_tr_np)), max_depth=max_depth, temperature=0.8
    )
    sdt.fit(X_tr, y_tr, X_val=X_val, y_val=y_val,
            model_prefix="SDT", output_folder=str(TRAINED_MODELS_DIR),
            num_epochs=epochs_num, batch_size=batch_size, learning_rate=1e-3)
    sdt.load_trained_model(os.path.join(TRAINED_MODELS_DIR, "SDT_soft_decision_tree.pt"))
    proba, _ = sdt.predict(X_test)
    y_proba = proba[:, 1]
    y_pred  = np.argmax(proba, axis=1)
    add_result("SDT", roc_auc_score(y_test_np, y_proba), accuracy_score(y_test_np, y_pred), time.time()-t0)

    # SMSDT
    t0 = time.time()
    smsdt = ShortMemorySoftBinaryTree(
        input_dim=n_features, num_classes=len(np.unique(y_tr_np)), max_depth=max_depth,
        temperature=0.8, effective_transform_output_dim=min(16, X_tr.shape[1])
    )
    smsdt.fit(X_tr, y_tr, X_val=X_val, y_val=y_val,
              model_prefix="SMSDT", output_folder=str(TRAINED_MODELS_DIR),
              num_epochs=epochs_num, batch_size=batch_size, learning_rate=1e-3)
    smsdt.load_trained_model(os.path.join(TRAINED_MODELS_DIR, "SMSDT_short_memory_soft_decision_tree.pt"))
    proba, _ = smsdt.predict(X_test)
    y_proba = proba[:, 1]
    y_pred  = np.argmax(proba, axis=1)
    add_result("SMSDT", roc_auc_score(y_test_np, y_proba), accuracy_score(y_test_np, y_pred), time.time()-t0)

    return results

def main():
    all_rows = []
    for dname in DATASETS:
        for seed in SEEDS:
            try:
                rows = run_one_split(dname, seed, max_depth=3, epochs_num=EPOCHS_NUM)
                all_rows.extend(rows)
                print(f"[OK] {dname} seed={seed}")
            except Exception as e:
                print(f"[ERR] {dname} seed={seed}: {e}")

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "all_experiments.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved raw results to {out_csv}")

if __name__ == "__main__":
    main()
