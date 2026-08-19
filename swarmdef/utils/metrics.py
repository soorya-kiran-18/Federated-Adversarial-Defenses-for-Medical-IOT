"""Detection metrics used across every experiment.

Everything the report needs is computed here so that the centralised baseline,
the federated rounds and the adversarial evaluation all report *identical*
definitions -- otherwise the comparison plots would not be meaningful.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DetectionMetrics:
    """Binary intrusion-detection metrics (positive class = attack)."""

    accuracy: float
    precision: float
    recall: float           # a.k.a. detection rate / TPR
    f1: float
    fpr: float              # false positive rate -- critical in clinical settings
    tp: int
    fp: int
    tn: int
    fn: int
    n: int

    @property
    def detection_rate(self) -> float:
        return self.recall

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"acc={self.accuracy:.4f} f1={self.f1:.4f} "
            f"prec={self.precision:.4f} rec={self.recall:.4f} fpr={self.fpr:.4f}"
        )


def compute_metrics(y_true, y_pred) -> DetectionMetrics:
    """Confusion-matrix metrics from label arrays (no sklearn dependency)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = tp + fp + tn + fn

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return DetectionMetrics(accuracy, precision, recall, f1, fpr, tp, fp, tn, fn, n)


def attack_success_rate(y_true, y_pred) -> float:
    """Fraction of true attacks the detector let through (evasion rate).

    This is the headline number for the GAN experiment: the adversary "wins"
    every time a genuine attack sample is classified as benign.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    attacks = y_true == 1
    if not attacks.any():
        return 0.0
    return float(np.mean(y_pred[attacks] == 0))


def roc_auc(y_true, scores) -> float:
    """Rank-based AUC (Mann-Whitney U), robust to score scaling."""
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    # Average ranks over ties so the AUC is exact for discrete scores.
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum_pos = ranks[: pos.size].sum()
    return float((rank_sum_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
