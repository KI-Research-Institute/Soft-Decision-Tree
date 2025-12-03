'''
Implementation for: "Soft Decision Tree classifier: explainable and extendable PyTorch implementation"
Author: Reuben R Shamir

In this file we implement:

1. Soft decision tree node
2. Soft decision tree
3. Early stoping class
4. A method for entropy regularization
5. A method to visualize the soft decision tree
6. A method for feature importance (SHAP)

'''

import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphviz import Digraph
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import os
import copy
from typing import Optional, Callable
callback: Optional[Callable[[int, dict], None]] = None
from collections import deque


class SoftBinaryTreeNode(nn.Module):
    def __init__(self, input_dim, num_classes, depth, max_depth, temperature=1.0):
        super().__init__()
        self.depth = depth
        self.max_depth = max_depth
        self.temperature = temperature
        self.path_trace = None
        self.pruned = False

        if depth < max_depth:
            self.is_leaf = False
            self.decision = nn.Linear(input_dim, 1)
            self.left = SoftBinaryTreeNode(input_dim, num_classes, depth + 1, max_depth, temperature)
            self.right = SoftBinaryTreeNode(input_dim, num_classes, depth + 1, max_depth, temperature)
        else:
            self.is_leaf = True
            self.output_logits = nn.Parameter(torch.randn(num_classes))

    def forward(self, x, path_prob):

        if self.pruned or self.is_leaf:
            self.path_trace = path_prob.detach().clone()
            return path_prob * self.output_logits.unsqueeze(0)

        decision_score = self.decision(x).squeeze(-1)
        decision_prob = torch.sigmoid(decision_score / self.temperature)

        self.path_trace = path_prob.detach().clone()

        left_prob = path_prob * decision_prob.unsqueeze(1)
        right_prob = path_prob * (1 - decision_prob).unsqueeze(1)

        return self.left(x, left_prob) + self.right(x, right_prob)

    def prune(self, threshold=1e-4):
        if self.is_leaf:
            return

        self.left.prune(threshold)
        self.right.prune(threshold)

        if self.left.is_leaf and self.right.is_leaf:
            if self.left.path_trace is not None and self.right.path_trace is not None:
                max_prob = torch.cat([self.left.path_trace, self.right.path_trace], dim=1).max().item()
                if max_prob < threshold:
                    avg_logits = (self.left.output_logits + self.right.output_logits) / 2
                    self.output_logits = nn.Parameter(avg_logits.detach())
                    self.is_leaf = True
                    self.pruned = True
                    self.left = None
                    self.right = None


class SoftBinaryDecisionTree(nn.Module):
    def __init__(self, input_dim, num_classes, max_depth=3, temperature=1.0):
        super().__init__()
        self.root = SoftBinaryTreeNode(input_dim, num_classes, 0, max_depth, temperature)
        self.trained_model_path = None
        self.trained = False
        self.device = 'cpu'

    def forward(self, x):
        batch_size = x.size(0)
        initial_prob = torch.ones(batch_size, 1, device=x.device)
        return self.root(x, initial_prob)

    def prune(self, threshold=1e-4):
        print('prune')
        self.root.prune(threshold)

    def load_trained_model(self, trained_model_filename):
        state = torch.load(trained_model_filename, map_location=self.device)
        self.load_state_dict(state)
        self.eval()
        self.trained = True

    def fit(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            *,
            learning_rate: float = 1e-3,
            num_epochs: int = 360,
            batch_size: int = 300,
            stop_threshold: float = 1e-4,
            max_no_improvement_epochs: int = 3,
            model_prefix: str = "SDT",
            output_folder: str | None = None,
            X_val: torch.Tensor | None = None,
            y_val: torch.Tensor | None = None,
            val_loader: torch.utils.data.DataLoader | None = None,
            val_patience: int = 12,
            val_min_delta: float = 0.0,
            callback: Optional[Callable[[int, dict], None]] = None,
    ):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.to(device)

        # ---- Data ----
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=max((os.cpu_count() or 2) // 2, 2),
            persistent_workers=True
        )

        if val_loader is None and X_val is not None and y_val is not None:
            val_ds = TensorDataset(X_val, y_val)
            val_loader = DataLoader(
                val_ds,
                batch_size=min(4096, batch_size * 4),
                shuffle=False,
                pin_memory=(device.type == "cuda"),
                num_workers=max((os.cpu_count() or 2) // 2, 2),
                persistent_workers=True
            )

        use_val = val_loader is not None

        # ---- Optim, sched, loss ----
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5, verbose=True, min_lr=1e-5, threshold=1e-4
        )
        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs))
        loss_fn = torch.nn.CrossEntropyLoss()

        # AMP
        use_amp = (device.type == "cuda")
        supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ---- Early stop / best model tracking ----
        best_state = None
        best_metric = float("inf")  # val_loss when available; else train loss
        early = EarlyStopping(patience=val_patience, min_delta=val_min_delta) if use_val else None

        prev_train_loss = -1.0
        train_no_improve = 0

        entropy_weight = 0.1  # consider making this an argument

        for epoch in range(num_epochs):
            # ---- Train ----
            self.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = self(xb)
                    tree_loss = loss_fn(logits, yb)
                with torch.cuda.amp.autocast(enabled=False):
                    ent_pen = entropy_regularization(self, xb.float())  # FP32
                    loss = tree_loss + entropy_weight * ent_pen

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * xb.size(0)

            train_loss = total_loss / len(train_loader.dataset)

            # ---- Validate (if provided) ----
            val_loss = None
            if use_val:
                self.eval()
                v_total = 0.0
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    for vxb, vyb in val_loader:
                        vxb = vxb.to(device, non_blocking=True)
                        vyb = vyb.to(device, non_blocking=True)
                        v_logits = self(vxb)
                        v_loss = loss_fn(v_logits, vyb)
                        v_total += v_loss.item() * vxb.size(0)
                val_loss = v_total / len(val_loader.dataset)

            # ---- Schedulers ----
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                # Drive plateau by validation loss if available, else training loss
                scheduler.step(val_loss if use_val else train_loss)

            # ---- Choose model-selection metric ----
            current_metric = val_loss if use_val else train_loss
            improved = current_metric < best_metric - 1e-12
            if improved and (epoch > num_epochs / 3 or use_val):  # keep your “after 1/3 epochs” condition if you like
                best_metric = current_metric
                best_state = copy.deepcopy(self.state_dict())

            # ---- Early stopping ----
            stop_now = False
            if use_val:
                stop_now = early.step(val_loss)
            else:
                # Fallback: tiny-change-on-train-loss heuristic (what you had before)
                if (prev_train_loss > 0) and (abs(prev_train_loss - train_loss) < stop_threshold):
                    train_no_improve += 1
                    if train_no_improve > max_no_improvement_epochs:
                        stop_now = True
                else:
                    prev_train_loss = train_loss
                    train_no_improve = 0

            # ---- Callback ----
            if callback is not None:
                logs = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "best_metric": best_metric,
                    "stopped": stop_now
                }
                try:
                    callback(epoch + 1, logs)
                except Exception as e:
                    # make callbacks non-fatal
                    print(f"[fit][callback] Exception ignored: {e}")

            # ---- Logging (optional) ----
            if use_val:
                print(f"Epoch {epoch + 1}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} | train {train_loss:.4f}")

            if stop_now:
                print("Early stopping triggered.")
                break

        # ---- Restore best ----
        if best_state is not None:
            self.load_state_dict(best_state)

        # (Optional) prune with a dataloader — prefer using the training or validation loader
        try:
            if use_val:
                self.prune(threshold=0.01, dataloader=val_loader)
            else:
                self.prune(threshold=0.01, dataloader=train_loader)
        except TypeError:
            # if your current prune() doesn't accept dataloader yet, keep legacy
            self.prune(threshold=0.01)

        # Save
        if output_folder is not None:
            self.trained_model_path = os.path.join(output_folder, f"{model_prefix}_soft_decision_tree.pt")
            torch.save(self.state_dict(), self.trained_model_path)

        self.trained = True

    def predict(self, X_test, batch_size=4096):
        if not self.trained:
            raise RuntimeError("Call fit() or load_trained_model() first.")
        self.eval()
        device = next(self.parameters()).device

        probs = []
        with torch.no_grad():
            for i in range(0, X_test.size(0), batch_size):
                xb = X_test[i:i + batch_size].to(device, non_blocking=True)
                pb = F.softmax(self(xb), dim=1).detach().cpu().numpy()
                probs.append(pb)
        proba = np.concatenate(probs, axis=0)

        probs_dict = {}
        collect_decision_probs(self.root, X_test.to(device, non_blocking=True), probs_dict)

        return proba, probs_dict

def entropy_regularization(model, X):
    eps = 1e-4
    entropies = []
    node_stack = [(model.root, torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype))]
    while node_stack:
        node, path_prob = node_stack.pop()
        if node.is_leaf or node.pruned:
            continue
        decision_score = node.decision(X).squeeze(-1)           # fp32 now
        p = torch.sigmoid(decision_score / node.temperature)
        p = p.clamp(eps, 1 - eps)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        entropies.append(entropy.mean())
        node_stack.append((node.left,  path_prob * p.unsqueeze(1)))
        node_stack.append((node.right, path_prob * (1 - p).unsqueeze(1)))
    return torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=X.device, dtype=X.dtype)


def collect_decision_probs(node, x, probs_dict, node_id=0):
    if node.is_leaf or node.pruned:
        return

    with torch.no_grad():
        score = node.decision(x).squeeze(-1)
        decision_prob = torch.sigmoid(score / node.temperature)
        avg_prob = decision_prob.mean().item()
        probs_dict[node_id] = avg_prob

    collect_decision_probs(node.left, x, probs_dict, 2 * node_id + 1)
    collect_decision_probs(node.right, x, probs_dict, 2 * node_id + 2)


def _top_features_from_node(node, feature_names=None, top_n=3):
    """
    Return [(feat_name, weight), ...] for the top |weight| features at this node.
    For standardized inputs, |weight| is a good proxy for contribution magnitude.
    """
    if getattr(node, "is_leaf", True) or getattr(node, "pruned", False):
        return []

    if not hasattr(node, "decision") or not hasattr(node.decision, "weight"):
        return []

    w = node.decision.weight.detach().cpu().numpy().ravel()  # shape: (input_dim,)
    if w.size == 0:
        return []

    idx = np.argsort(-np.abs(w))[:min(top_n, w.size)]
    names = [feature_names[i] if (feature_names is not None and i < len(feature_names)) else f"f{i}"
             for i in idx]
    return list(zip(names, w[idx]))  # [(name, signed_weight), ...]

def visualize_soft_tree(node,
                        probs_dict,
                        graph=None,
                        node_id=0,
                        feature_names=None,
                        top_n=3):
    """
    Recursively build a Graphviz graph of the soft tree.
    - Adds Top-k contributing features (by |weight|) at each internal node.
    """
    if graph is None:
        graph = Digraph(format='png')
        graph.attr(rankdir='TB')
        graph.attr('node', shape='box')

    current_id = f"node_{node_id}"

    # Leaf or pruned node
    if node.is_leaf or node.pruned:
        label = f"{current_id}\nLeaf"
        if hasattr(node, 'output_logits'):
            logits = node.output_logits.detach().cpu().numpy()
            label += f"\nlogits={np.round(logits, 2)}"
        graph.node(current_id, label, style='filled', fillcolor='lightgray')
        return graph

    # Internal node: show depth, path prob (if available), and top-k features
    lines = [f"{current_id}", f"Depth: {node.depth}"]
    if node.path_trace is not None:
        avg_path_prob = float(node.path_trace.mean().item())
        lines.append(f"path_prob={avg_path_prob:.2f}")

    # Top-k contributing features by |weight|
    top_feats = _top_features_from_node(node, feature_names=feature_names, top_n=top_n)
    if top_feats:
        lines.append("Top features:")
        for name, w in top_feats:
            lines.append(f"• {name} ({w:+.2f})")

    graph.node(current_id, "\n".join(lines))

    # Left child
    left_id = 2 * node_id + 1
    graph = visualize_soft_tree(node.left, probs_dict, graph, left_id,
                                feature_names=feature_names, top_n=top_n)
    left_prob = probs_dict.get(node_id, 0.5)
    graph.edge(current_id, f"node_{left_id}", label=f"p={left_prob:.2f}")

    # Right child
    right_id = 2 * node_id + 2
    graph = visualize_soft_tree(node.right, probs_dict, graph, right_id,
                                feature_names=feature_names, top_n=top_n)
    graph.edge(current_id, f"node_{right_id}", label=f"1-p={1 - left_prob:.2f}")

    return graph

def render_tree(tree_model,
                probs_dict,
                filename="soft_tree_binary",
                feature_names=None,
                top_n=3):
    """
    Render the soft tree with feature names at splits.
    """
    graph = visualize_soft_tree(tree_model.root,
                                probs_dict,
                                feature_names=feature_names,
                                top_n=top_n)
    image_path = graph.render(filename, format="png", cleanup=True)
    image = Image.open(image_path)
    image.show()



def explain_sample(model, X_train_np, X_sample_np, feature_names=None, top_k=3):
    """
    Explain a single sample prediction using both SHAP and internal soft-tree structure.

    Parameters
    ----------
    model : SoftBinaryDecisionTree
        A trained soft decision tree model.
    X_train_np : np.ndarray
        Training data (used as SHAP background).
    X_sample_np : np.ndarray
        A single sample, shape (n_features,) or (1, n_features).
    feature_names : list[str], optional
        Names of features (default: f0, f1, ...).
    max_depth : int
        Max tree depth for path tracing.
    top_k : int
        Number of top-weighted features to show per node.

    Returns
    -------
    shap_contribs : pd.DataFrame
        SHAP values per feature.
    """

    import pandas as pd
    from collections import deque

    # Normalize input
    if X_sample_np.ndim == 1:
        X_sample_np = X_sample_np.reshape(1, -1)

    n_features = X_sample_np.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    def model_predict(X):
        X_t = torch.tensor(X, dtype=torch.float32).to(model.device)
        proba, _ = model.predict(X_t)  # shape: (n_samples, n_classes)
        return proba

    try:
        explainer = shap.Explainer(model_predict, X_train_np)
        shap_values = explainer(X_sample_np)
    except Exception:
        # Fallback (more general, slower)
        explainer = shap.KernelExplainer(model_predict, X_train_np[:200])
        shap_values = explainer.shap_values(X_sample_np, nsamples=200)

    # Get a 1-D contribution vector for this one sample
    import numpy as np
    import pandas as pd

    # Case A: shap.Explanation object
    if hasattr(shap_values, "values"):
        vals = shap_values.values  # could be (1, n_features) or (1, n_features, n_outputs)
        if vals.ndim == 2:
            contrib = vals[0]  # (n_features,)
        elif vals.ndim == 3:
            # pick positive class if available, else 0
            out_idx = 1 if vals.shape[2] > 1 else 0
            contrib = vals[0, :, out_idx]  # (n_features,)
        else:
            raise ValueError(f"Unexpected SHAP values ndim: {vals.ndim}")
    # Case B: KernelExplainer can return list per class
    elif isinstance(shap_values, (list, tuple)):
        # choose positive class if present, else first
        pos_idx = 1 if len(shap_values) > 1 else 0
        vec = np.array(shap_values[pos_idx])  # shape (1, n_features)
        contrib = vec[0]
    else:
        raise ValueError("Unrecognized SHAP return type")

    # Make sure sample values are 1-D
    x_row = np.ravel(X_sample_np)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": x_row,
        "shap_value": contrib,
        "abs_value": np.abs(contrib),
    }).sort_values("abs_value", ascending=False)

    print("Top SHAP feature contributions:")
    print(shap_df.head(10)[["feature", "feature_value", "shap_value"]])

    return shap_df


def collect_soft_path_details(model, X_sample_np, feature_names, top_k=3, max_depth=None):
    model.eval()
    x_t = torch.tensor(X_sample_np.reshape(1, -1), dtype=torch.float32).to(model.device)

    path_details = []
    # (node, node_id, path_prob_float)
    node_queue = deque([(model.root, 0, 1.0)])

    while node_queue:
        node, node_id, path_prob = node_queue.popleft()

        # Skip leaves/pruned nodes in the "split summary"
        if getattr(node, "is_leaf", False) or getattr(node, "pruned", False):
            continue

        # Optional cap on depth
        if (max_depth is not None) and (node.depth > max_depth):
            continue

        # This node's split probability for THIS sample
        with torch.no_grad():
            score = node.decision(x_t).squeeze(-1)  # shape (1,)
            p = torch.sigmoid(score / node.temperature).item()  # scalar in [0,1]

        # Top-|w| features at this split
        w = node.decision.weight.detach().cpu().numpy().ravel()
        take = min(top_k, w.size)
        idx = np.argsort(-np.abs(w))[:take]
        feats = [f"{feature_names[i]} ({w[i]:+.2f})" for i in idx]

        path_details.append({
            "node_id": node_id,
            "depth": int(getattr(node, "depth", 0)),
            "prob_go_left": round(p, 3),
            "prob_go_right": round(1.0 - p, 3),
            "path_prob_up_to_here": round(path_prob, 4),
            "top_features": feats
        })

        # Enqueue children if they exist and are internal
        if hasattr(node, "left") and node.left is not None and not node.left.is_leaf and not node.left.pruned:
            node_queue.append((node.left, 2 * node_id + 1, path_prob * p))
        if hasattr(node, "right") and node.right is not None and not node.right.is_leaf and not node.right.pruned:
            node_queue.append((node.right, 2 * node_id + 2, path_prob * (1.0 - p)))

    # ---- print AFTER traversal
    print("Soft-tree path summary:")
    for d in path_details:
        print(
            f"Depth {d['depth']}: p_left={d['prob_go_left']}, "
            f"p_right={d['prob_go_right']}, top={d['top_features']}, "
            f"path_prob={d['path_prob_up_to_here']}"
        )

    return path_details

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.num_bad = 0
            return False  # don't stop
        self.num_bad += 1
        return self.num_bad > self.patience  # True => stop

