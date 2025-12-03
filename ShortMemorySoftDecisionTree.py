import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import copy
import os

'''
Implementation for: "Soft Decision Tree classifier: explainable and extendable PyTorch implementation"
Author: Reuben R Shamir

In this file we implement:

1. Short-term Memory Soft Decision Tree node
2. Short-term Memory Soft Decision Tree
3. Helper functions 

'''

# SM-SDT node extends the original SDT node with a linear NN layer at the input level and the output of parent and grandparent of each node.
class ShortMemorySoftBinaryTreeNode(nn.Module):
    def __init__(self, input_dim, num_classes, depth, max_depth, temperature=1.0, effective_transform_output_dim = 16):
        super().__init__()
        self.depth = depth
        self.max_depth = max_depth
        self.temperature = temperature
        self.path_trace = None
        self.pruned = False
        self.is_leaf = (depth == max_depth)

        # Effective input dim = x + parent_out + grandparent_out
        self.effective_input_dim = input_dim + int(depth > 0) + int(depth > 1)

        if not self.is_leaf:
            self.feature_transform = nn.Linear(self.effective_input_dim, effective_transform_output_dim)
            self.decision = nn.Linear(effective_transform_output_dim, 1)  # decision happens on transformed features
            self.left = ShortMemorySoftBinaryTreeNode(
                input_dim, num_classes, depth + 1, max_depth, temperature, effective_transform_output_dim
            )
            self.right = ShortMemorySoftBinaryTreeNode(
                input_dim, num_classes, depth + 1, max_depth, temperature, effective_transform_output_dim
            )
        else:
            # self.output_logits = nn.Parameter(torch.randn(num_classes))
            self.output_logits = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, path_prob, parent_out=None, grandparent_out=None, root_input=None):
        if root_input is None:
            root_input = x  # save root input once

        if self.pruned or self.is_leaf:
            self.path_trace = path_prob.detach().clone()
            return path_prob * self.output_logits.unsqueeze(0)

        # Build input: root skip + ancestor outputs
        input_parts = [root_input]
        if parent_out is not None:
            input_parts.append(parent_out)
        if grandparent_out is not None:
            input_parts.append(grandparent_out)

        node_input = torch.cat(input_parts, dim=-1)
        transformed = torch.relu(self.feature_transform(node_input))

        decision_score = self.decision(transformed).squeeze(-1)
        decision_prob = torch.sigmoid(decision_score / self.temperature)

        self.path_trace = path_prob.detach().clone()
        self_output = decision_score.unsqueeze(-1)

        left_prob = path_prob * decision_prob.unsqueeze(1)
        right_prob = path_prob * (1 - decision_prob).unsqueeze(1)

        return self.left(x, left_prob, self_output, parent_out, root_input) + \
               self.right(x, right_prob, self_output, parent_out, root_input)

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



class ShortMemorySoftBinaryTree(nn.Module):
    def __init__(self, input_dim, num_classes, max_depth=3, temperature=1.0, effective_transform_output_dim = 16):
        super().__init__()
        self.root = ShortMemorySoftBinaryTreeNode(input_dim, num_classes, 0, max_depth, temperature, effective_transform_output_dim)
        self.trained_model_path = None
        self.trained = False
        self.device = 'cpu'

    def forward(self, x):
        batch_size = x.size(0)
        initial_prob = torch.ones(batch_size, 1, device=x.device)
        return self.root(x, initial_prob, None, None, root_input=x)

    def prune(self, threshold=1e-4):
        self.root.prune(threshold)

    def load_trained_model (self, trained_model_filename):
        self.load_state_dict(torch.load(trained_model_filename))
        self.trained = True
        self.to(self.device)
        self.eval()

    def fit(
            self,
            X_train,
            y_train,
            learning_rate=0.001,
            num_epochs=360,
            batch_size=300,
            stop_threshold=0.0001,  # used only if no val set is provided
            max_no_improvement_epochs=3,  # used only if no val set is provided
            model_prefix='SMSDT',
            output_folder=None,
            # NEW:
            X_val=None,
            y_val=None,
            val_patience=12,  # early-stop patience based on val loss
            val_min_delta=0.0,  # minimum improvement to reset patience
    ):
        best_model = None
        best_model_metric = float("inf")  # val_loss if available, else train_loss
        entropy_weight = 0.1
        l_weight = 0.001

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.device = device

        # ---- Data ----
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=max((os.cpu_count() or 2) // 2, 2),
            persistent_workers=True
        )

        val_loader = None
        use_val = (X_val is not None) and (y_val is not None)
        if use_val:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(4096, batch_size * 4),
                shuffle=False,
                pin_memory=(device.type == "cuda"),
                num_workers=max((os.cpu_count() or 2) // 2, 2),
                persistent_workers=True
            )

        # ---- Optim, sched, loss ----
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=5,
            verbose=True,
            min_lr=1e-5,
            threshold=1e-4
        )
        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs))
        loss_fn = torch.nn.CrossEntropyLoss()

        # AMP
        use_amp = device.type == "cuda"
        supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ---- Early stop state ----
        # If validation is provided, early stop on val_loss using (patience, min_delta).
        # Otherwise, use your original tiny-change-on-train-loss heuristic.
        prev_train_loss = -1.0
        train_no_improve = 0
        best_val = float("inf")
        val_bad_epochs = 0

        for epoch in range(num_epochs):
            # ===== Train =====
            self.train()
            total_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    output = self(batch_x)
                    tree_loss = loss_fn(output, batch_y)

                entropy_penalty = entropy_regularization_smsdt(self, batch_x)
                reg_penalty = collect_transform_weight_penalty(self.root, p=2, device=self.device)
                loss = tree_loss + entropy_weight * entropy_penalty + l_weight * reg_penalty

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * batch_x.size(0)

            train_loss = total_loss / len(train_loader.dataset)

            # ===== Validate (if provided) =====
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

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            # ===== Schedulers =====
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step(val_loss if use_val else train_loss)

            # ===== Track best & early stop =====
            current_metric = val_loss if use_val else train_loss
            improved = current_metric < best_model_metric - 1e-12
            if improved and (epoch > num_epochs / 3 or use_val):
                best_model_metric = current_metric
                best_model = copy.deepcopy(self.state_dict())

            if use_val:
                # Early stopping on val_loss
                if (val_loss is not None) and (val_loss < best_val - val_min_delta):
                    best_val = val_loss
                    val_bad_epochs = 0
                else:
                    val_bad_epochs += 1

                print(f"Epoch {epoch + 1}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")
                if val_bad_epochs >= val_patience:
                    print("Early stopping on validation loss.")
                    break
            else:
                # Fallback: tiny-change-on-train-loss heuristic (original behavior)
                print(f"Epoch {epoch + 1}/{num_epochs} | train {train_loss:.4f}")
                if (prev_train_loss > 0) and (abs(prev_train_loss - train_loss) < stop_threshold):
                    train_no_improve += 1
                    if train_no_improve > max_no_improvement_epochs:
                        print("Loss change is small - stop training")
                        break
                else:
                    prev_train_loss = train_loss
                    train_no_improve = 0

        # ===== Restore best =====
        if best_model is not None:
            self.load_state_dict(best_model)

        # ===== Prune after training =====
        self.prune(threshold=0.01)

        # ===== Save =====
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            self.trained_model_path = os.path.join(
                output_folder, f"{model_prefix}_short_memory_soft_decision_tree.pt"
            )
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
        # proba_soft = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        # If you want routing stats, compute on a representative subset:
        probs_dict = {}
        collect_decision_probs_smsdt(self.root, X_test[:batch_size].to(device), probs_dict)
        return proba, probs_dict

def entropy_regularization_smsdt(model, X):
    # keep current model mode; allow grads
    eps = 1e-4
    batch_size = X.size(0)
    initial_prob = torch.ones(batch_size, 1, device=X.device, dtype=X.dtype)

    node_stack = [(model.root, initial_prob, None, None)]
    entropies = []

    while node_stack:
        node, path_prob, parent_out, grandparent_out = node_stack.pop()
        if node.is_leaf or node.pruned:
            continue

        input_parts = [X]
        if parent_out is not None:      input_parts.append(parent_out)
        if grandparent_out is not None: input_parts.append(grandparent_out)
        node_input = torch.cat(input_parts, dim=-1)

        # FP32 block for stability
        with torch.cuda.amp.autocast(enabled=False):
            transformed   = torch.relu(node.feature_transform(node_input.float()))
            decision_score= node.decision(transformed).squeeze(-1)
            p             = torch.sigmoid(decision_score / node.temperature).clamp(eps, 1 - eps)
            entropy       = -p * torch.log(p) - (1 - p) * torch.log(1 - p)

        entropies.append(entropy.mean())

        node_output = decision_score.unsqueeze(-1)
        node_stack.append((node.left,  path_prob * p.unsqueeze(1), node_output, parent_out))
        node_stack.append((node.right, path_prob * (1 - p).unsqueeze(1), node_output, parent_out))

    return torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=X.device, dtype=X.dtype)


def collect_decision_probs_smsdt(node, x, probs_dict, node_id=0, parent_out=None, grandparent_out=None):
    if node.is_leaf or node.pruned:
        return

    # Build full input just like in forward()
    input_parts = [x]
    if parent_out is not None:
        input_parts.append(parent_out)
    if grandparent_out is not None:
        input_parts.append(grandparent_out)
    node_input = torch.cat(input_parts, dim=-1)

    with torch.no_grad():
        transformed = torch.relu(node.feature_transform(node_input))
        score = node.decision(transformed).squeeze(-1)
        decision_prob = torch.sigmoid(score / node.temperature)
        avg_prob = decision_prob.mean().item()
        probs_dict[node_id] = avg_prob

    # Pass current score as parent_out for child
    node_output = score.unsqueeze(-1)
    collect_decision_probs_smsdt(node.left, x, probs_dict, 2 * node_id + 1, node_output, parent_out)
    collect_decision_probs_smsdt(node.right, x, probs_dict, 2 * node_id + 2, node_output, parent_out)

def collect_transform_weight_penalty(node, p=2, device='cpu'):
    """
    Recursively collects Lp norms of transformation weights from all internal nodes.
    Returns the sum of all norms.
    """
    if node.is_leaf or node.pruned:
        return torch.tensor(0.0, device=device)
        # cast to float32 for stability
    w = node.feature_transform.weight
    norm = torch.norm(w.float(), p=p)
    return norm.to(device) \
        + collect_transform_weight_penalty(node.left, p, device) \
        + collect_transform_weight_penalty(node.right, p, device)