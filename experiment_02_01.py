import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
from SoftDecisionTree import SoftBinaryDecisionTree, entropy_regularization, visualize_soft_tree, render_tree, collect_decision_probs, explain_sample, collect_soft_path_details
from ShortMemorySoftDecisionTree import ShortMemorySoftBinaryTree
from SoftDecisionTree_v2 import SoftBinaryDecisionTreeV2, entropy_regularization_v2, collect_decision_probs_v2, collect_transform_weight_penalty
from SoftDecisionTree_V03 import SoftBinaryDecisionTreeV03
from SoftDecisionTree_v4 import SoftBinaryDecisionTreeV4
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from torch.optim.lr_scheduler import LambdaLR
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import os
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from ucimlrepo import fetch_ucirepo

from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

DATA_DIR = Path("data_prepared")
DATA_DIR.mkdir(exist_ok=True, parents=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    predict_only = False

    trained_models_folder = r'/home/ruby/Research/data/03_methods_paper/SDT_trained_models'

    max_depth = 3

    dataset_code = 3

    #                        0                 1               2                   3              4          5        6              7               8
    dataset_names =   ["breast_cancer", "pima_diabetes", "heart_cleveland", "thyroid_sick", "dermatology", "ilpd", "stroke", "local_disease5", "heart_failure"]
    # initial AUC LR        0.994             0.82             0.94               0.97          1.0         0.84     0.84         1.0               0.87
    # SMSDT max_d=3         0.995             0.83             0.92                                         0.84     0.82
    # everyone success: 0, 3,
    current_dataset_name = dataset_names[dataset_code]

    X_train_np = np.array(np.load(f"data_prepared/{current_dataset_name}/X_train.npy"))
    y_train_np = np.array(np.load(f"data_prepared/{current_dataset_name}/y_train.npy"))
    X_test_np =  np.array(np.load(f"data_prepared/{current_dataset_name}/X_test.npy"))
    y_test_np = np.array(np.load(f"data_prepared/{current_dataset_name}/y_test.npy"))

    X_all = np.concatenate([X_train_np, X_test_np], axis=0)
    y_all = np.concatenate([y_train_np, y_test_np], axis=0)

    # --- Randomly re-split ---
    # Adjust test_size as needed (e.g., 0.2 for 80/20 split)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=7,  # ensures reproducibility
        stratify=y_all if len(np.unique(y_all)) > 1 else None  # keep label balance if possible
    )


    # ---- 1) Drop highly correlated columns (NumPy only) ----
    threshold = 0.9

    # correlation matrix across columns
    corr = np.corrcoef(X_train_np, rowvar=False)  # (n_features, n_features)
    np.fill_diagonal(corr, 0.0)  # ignore self-corr
    corr = np.nan_to_num(np.abs(corr))  # absolute, NaN -> 0

    to_drop_idx = []
    for j in range(corr.shape[1]):  # use upper-tri logic
        if np.any(corr[:j, j] > threshold):
            to_drop_idx.append(j)

    keep_idx = np.array([i for i in range(corr.shape[1]) if i not in set(to_drop_idx)], dtype=int)

    X_train_np = X_train_np[:, keep_idx]
    X_test_np = X_test_np[:, keep_idx]

    # (optional) keep map to original features
    # selected_corr_idx = keep_idx.copy()

    # ---- 2) Random oversampling (train only) ----
    ros = RandomOverSampler(random_state=41)
    X_train_np, y_train_np = ros.fit_resample(X_train_np, y_train_np)
    # X_test_np, y_test_np = ros.fit_resample(X_test_np, y_test_np)

    # ---- 3) SelectKBest on training set only ----
    k = min(16, X_train_np.shape[1])  # guard if fewer than 16 features remain
    print (f"selects {k} features")
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_new = selector.fit_transform(X_train_np, y_train_np)
    X_test_new = selector.transform(X_test_np)

    # (optional) indices after both steps relative to original columns:
    # final_feature_idx = keep_idx[selector.get_support(indices=True)]

    # .... all your imports and preprocessing above stay the same ...

    # ---- 4) Scale (fit on train, transform test) ----
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_new)
    X_test_np = scaler.transform(X_test_new)

    # =========================
    # NEW: make a validation split from the (preprocessed) training data
    # =========================
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np
    )

    # Convert to tensors (train/val/test)
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)

    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    y_val = torch.tensor(y_val_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    n_features = X_tr.shape[1]

    # ===== Classical baselines (unchanged): they still train on full X_train_np, evaluate on X_test_np =====
    # (If you want them to mirror the soft trees, you can switch them to X_tr_np / X_val_np too, but not required.)
    start_time = time.time()
    clf = LogisticRegression(max_iter=600, random_state=42)
    clf.fit(X_train_np, y_train_np)
    y_pred_lr = clf.predict(X_test_np)
    y_proba_lr = clf.predict_proba(X_test_np)[:, 1]
    acc_lr = accuracy_score(y_test_np, y_pred_lr)
    auc_lr = roc_auc_score(y_test_np, y_proba_lr)
    lr_time = time.time() - start_time

    # Decision Tree
    start_time = time.time()
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train_np, y_train_np)
    y_pred_dt = clf.predict(X_test_np)
    y_proba_dt = clf.predict_proba(X_test_np)[:, 1]
    acc_dt = accuracy_score(y_test_np, y_pred_dt)
    auc_dt = roc_auc_score(y_test_np, y_proba_dt)
    dt_time = time.time() - start_time

    # RF 100
    start_time = time.time()
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42)
    rf_clf.fit(X_train_np, y_train_np)
    y_pred_rf_100 = rf_clf.predict(X_test_np)
    y_proba_rf_100 = rf_clf.predict_proba(X_test_np)[:, 1]
    acc_rf_100 = accuracy_score(y_test_np, y_pred_rf_100)
    auc_rf_100 = roc_auc_score(y_test_np, y_proba_rf_100)
    random_forest_time_100 = time.time() - start_time

    # RF 1000
    start_time = time.time()
    rf_clf = RandomForestClassifier(n_estimators=1000, max_depth=max_depth, random_state=42)
    rf_clf.fit(X_train_np, y_train_np)
    y_pred_rf = rf_clf.predict(X_test_np)
    y_proba_rf = rf_clf.predict_proba(X_test_np)[:, 1]
    acc_rf = accuracy_score(y_test_np, y_pred_rf)
    auc_rf = roc_auc_score(y_test_np, y_proba_rf)
    random_forest_time = time.time() - start_time

    # XGBoost
    start_time = time.time()
    xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                            max_depth=max_depth)
    xgb_clf.fit(X_train_np, y_train_np)
    y_pred_xgb = xgb_clf.predict(X_test_np)
    y_proba_xgb = xgb_clf.predict_proba(X_test_np)[:, 1]
    acc_xgb = accuracy_score(y_test_np, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test_np, y_proba_xgb)
    xgb_time = time.time() - start_time

    # ===== 3. Soft Trees with explicit validation =====

    epochs_num = 200
    batch_size = max(32, min(int(len(X_tr) // 10), 1000))

    results = []
    for model_type in ['SDT', 'SMSDT']:

        if model_type == 'SDT':
            model = SoftBinaryDecisionTree(
                input_dim=n_features,
                num_classes=len(torch.unique(y_tr)),
                max_depth=max_depth,
                temperature=0.8
            )

            start_time = time.time()
            model.fit(
                X_tr, y_tr,
                X_val=X_val, y_val=y_val,
                model_prefix=model_type,
                output_folder=trained_models_folder,
                num_epochs=epochs_num,
                batch_size=batch_size,
                learning_rate=0.001
            )
            trained_model_filename = os.path.join(trained_models_folder, f"{model_type}_soft_decision_tree.pt")
            model.load_trained_model(trained_model_filename)
            proba, probs_dict = model.predict(X_test)
            proba_soft = proba[:, 1]
            y_pred_soft = np.argmax(proba, axis=1)
            acc_soft = accuracy_score(y_test_np, y_pred_soft)
            auc_soft = roc_auc_score(y_test_np, proba_soft)
            train_time = time.time() - start_time
            results.append([acc_soft, auc_soft, proba_soft, train_time, model_type])
            soft_tree_model = model

            feature_names = [f"f{i}" for i in range(X_tr.shape[1])]  # or your real column names
            with torch.no_grad():
                X_vis = torch.tensor(X_test_np, dtype=torch.float32).to(model.device)
                probs_dict = {}
                collect_decision_probs(model.root, X_vis, probs_dict)

            render_tree(model,
                        probs_dict,
                        filename=f"soft_tree_{current_dataset_name}_depth{max_depth}",
                        feature_names=feature_names,
                        top_n=3)

            for sample_idx in range(0, 1):
                print(f"Sample {sample_idx}:")
                shap_df = explain_sample(
                    soft_tree_model,
                    X_train_np,
                    X_test_np[sample_idx],
                    feature_names=feature_names,
                    top_k=5
                )
                path_info = collect_soft_path_details(
                    model,
                    X_sample_np=X_test_np[sample_idx],
                    feature_names=feature_names,
                    top_k=3,
                    max_depth=None  # or set an int if you want a cap
                )

                # print(f"SHAP: {shap_df}")
                # print(f"path info: {path_info}")


        if model_type == 'SMSDT':
            model = ShortMemorySoftBinaryTree(
                input_dim=n_features,
                num_classes=len(torch.unique(y_tr)),
                max_depth=max_depth,
                temperature=0.8,
                effective_transform_output_dim=min(16, X_tr.shape[1])
            )
            # NEW: pass validation explicitly
            start_time = time.time()
            model.fit(
                X_tr, y_tr,
                X_val=X_val, y_val=y_val,
                model_prefix=model_type,
                output_folder=trained_models_folder,
                num_epochs=epochs_num,
                batch_size=batch_size,
                learning_rate=0.001
            )
            trained_model_filename = os.path.join(trained_models_folder,
                                                  f"{model_type}_short_memory_soft_decision_tree.pt")
            model.load_trained_model(trained_model_filename)
            proba, probs_dict = model.predict(X_test)
            proba_soft = proba[:, 1]
            y_pred_soft = np.argmax(proba, axis=1)
            acc_soft = accuracy_score(y_test_np, y_pred_soft)
            auc_soft = roc_auc_score(y_test_np, proba_soft)
            train_time = time.time() - start_time
            results.append([acc_soft, auc_soft, proba_soft, train_time, model_type])



    # 4. Plot ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test_np, y_proba_lr, name=f"Logistic regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test_np, y_proba_dt, name=f"Decision Tree", ax=ax)
    RocCurveDisplay.from_predictions(y_test_np, y_proba_rf_100, name=f"Random Forest 100", ax=ax)
    RocCurveDisplay.from_predictions(y_test_np, y_proba_rf, name=f"Random Forest 1000", ax=ax)
    RocCurveDisplay.from_predictions(y_test_np, y_proba_xgb, name=f"XGBoost", ax=ax)
    for index in range(0, len(results)):
        RocCurveDisplay.from_predictions(y_test_np, results[index][2], name=results[index][4], ax=ax)
    plt.title("ROC Curve Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5. Print Results
    print(f"Logistic regression   → Accuracy: {acc_lr:.3f}, AUC: {auc_lr:.3f}, Time: {lr_time:.2f}s")
    print(f"Decision Tree         → Accuracy: {acc_dt:.3f}, AUC: {auc_dt:.3f}, Time: {dt_time:.2f}s")
    print(f"Random Forest  100    → Accuracy: {acc_rf_100:.3f}, AUC: {auc_rf_100:.3f}, Time: {random_forest_time_100:.2f}s")
    print(f"Random Forest  1000   → Accuracy: {acc_rf:.3f}, AUC: {auc_rf:.3f}, Time: {random_forest_time:.2f}s")
    print(f"XGBoost               → Accuracy: {acc_xgb:.3f}, AUC: {auc_xgb:.3f}, Time: {xgb_time:.2f}s")
    print(f"Soft Decision Tree    → Accuracy: {results[0][0]:.3f}, AUC: {results[0][1]:.3f}, Time: {results[0][3]:.2f}s")
    print(f"SM Soft Decision Tree → Accuracy: {results[1][0]:.3f}, AUC: {results[1][1]:.3f}, Time: {results[1][3]:.2f}s")




