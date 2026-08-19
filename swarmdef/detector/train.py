"""Training and evaluation for the detector (Step 4).

Deliberately architecture-agnostic: the same `train_model` / `evaluate` pair is
reused by the centralised baseline (Step 4), by each federated client's local
update (Step 5), by adversarial retraining (Step 6) and under DP-SGD (Step 7).
Sharing one implementation is what makes those results comparable -- a separate
training loop per experiment would introduce differences that have nothing to
do with the mechanism under study.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from swarmdef.utils.logging import get_logger
from swarmdef.utils.metrics import (DetectionMetrics, attack_success_rate,
                                    compute_metrics, roc_auc)

log = get_logger("detector.train")


@dataclass
class TrainHistory:
    """Per-epoch record, used for the learning-curve figures."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)

    def best_epoch(self) -> int:
        return int(np.argmax(self.val_f1)) if self.val_f1 else -1


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, attack_score) over an entire loader."""
    model.eval()
    ys, preds, scores = [], [], []
    for x, edge_index, y in loader:
        x = x.to(device)
        ei = edge_index.to(device) if edge_index is not None else None
        logits = model(x, ei) if ei is not None else model(x)
        prob = torch.softmax(logits, dim=-1)
        # For the binary head this is P(attack); for the 7-way head it is
        # 1 - P(benign), which is the same quantity a SOC would alert on.
        score = prob[:, 1] if prob.shape[1] == 2 else 1.0 - prob[:, 0]
        ys.append(y.cpu().numpy())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        scores.append(score.cpu().numpy())
    if not ys:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(scores)


def evaluate(model: nn.Module, loader, device: torch.device,
             multiclass: bool = False) -> tuple[DetectionMetrics, dict]:
    """Binary detection metrics plus per-class recall and AUC."""
    y_true, y_pred, score = predict(model, loader, device)
    if y_true.size == 0:
        return compute_metrics([0], [0]), {}

    if multiclass:
        bin_true = (y_true != 0).astype(int)
        bin_pred = (y_pred != 0).astype(int)
    else:
        bin_true, bin_pred = y_true, y_pred

    metrics = compute_metrics(bin_true, bin_pred)
    extra = {
        "auc": roc_auc(bin_true, score),
        "evasion_rate": attack_success_rate(bin_true, bin_pred),
    }
    if multiclass:
        extra["per_class_recall"] = {
            int(c): float(np.mean(y_pred[y_true == c] == c))
            for c in np.unique(y_true)
        }
        extra["multiclass_accuracy"] = float(np.mean(y_pred == y_true))
    return metrics, extra


def train_model(
    model: nn.Module,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device | None = None,
    class_weight: torch.Tensor | None = None,
    multiclass: bool = False,
    early_stopping: int = 8,
    optimizer: torch.optim.Optimizer | None = None,
    verbose: bool = True,
    log_every: int = 5,
) -> TrainHistory:
    """Train in place; restore the best-validation weights before returning.

    `optimizer` may be supplied pre-built -- Step 7 passes an Opacus-wrapped
    optimizer so DP-SGD reuses this loop unchanged.
    """
    device = device or torch.device("cpu")
    model.to(device)
    opt = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight.to(device) if class_weight is not None else None
    )
    history = TrainHistory()
    best_f1, best_state, patience = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        total, n_batches = 0.0, 0
        for x, edge_index, y in train_loader:
            x, y = x.to(device), y.to(device)
            ei = edge_index.to(device) if edge_index is not None else None
            opt.zero_grad(set_to_none=True)
            logits = model(x, ei) if ei is not None else model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        history.train_loss.append(total / max(n_batches, 1))

        if val_loader is not None:
            metrics, _ = evaluate(model, val_loader, device, multiclass)
            history.val_acc.append(metrics.accuracy)
            history.val_f1.append(metrics.f1)
            history.val_loss.append(float("nan"))
            if metrics.f1 > best_f1 + 1e-5:
                best_f1 = metrics.f1
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
                log.info("  epoch %3d | loss %.4f | val acc %.4f f1 %.4f",
                         epoch, history.train_loss[-1], metrics.accuracy, metrics.f1)
            if early_stopping and patience >= early_stopping:
                if verbose:
                    log.info("  early stop at epoch %d (best val f1 %.4f)", epoch, best_f1)
                break
        elif verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            log.info("  epoch %3d | loss %.4f", epoch, history.train_loss[-1])

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
