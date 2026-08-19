"""One dataloader interface for all three detector architectures.

The GNN consumes batched graphs, the Transformer consumes per-device sequences
and the MLP consumes single windows. Rather than special-casing the architecture
at every call site -- centralised training, federated clients, adversarial
evaluation -- each loader yields the same 3-tuple:

    (inputs, edge_index_or_None, targets)

so downstream code is architecture-agnostic. This is what allows Steps 5-8 to
swap the detector without touching the federated, DP or aggregation code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from swarmdef.data.schema import FEATURE_NAMES
from swarmdef.detector.graph import build_graphs, build_sequences


class GraphBatchLoader:
    """Batches PyG graphs and yields (x, edge_index, y) with node-level targets."""

    def __init__(self, graphs, batch_size: int = 32, shuffle: bool = False,
                 multiclass: bool = False, seed: int = 42) -> None:
        self.graphs = graphs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.multiclass = multiclass
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return max(1, (len(self.graphs) + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        from torch_geometric.data import Batch

        order = np.arange(len(self.graphs))
        if self.shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            chunk = [self.graphs[i] for i in order[start:start + self.batch_size]]
            if not chunk:
                continue
            batch = Batch.from_data_list(chunk)
            y = batch.y_multi if self.multiclass else batch.y
            yield batch.x, batch.edge_index, y

    @property
    def n_samples(self) -> int:
        return sum(g.num_nodes for g in self.graphs)


class TensorBatchLoader:
    """Wraps a torch DataLoader so it yields (x, None, y)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64,
                 shuffle: bool = False) -> None:
        # np.ascontiguousarray guarantees a writable, owned buffer: pandas 3.0
        # hands back read-only views and torch.from_numpy on those emits a
        # non-writable-tensor warning and risks undefined behaviour.
        self.ds = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )
        self.dl = DataLoader(self.ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        self.n_samples = len(self.ds)

    def __len__(self) -> int:
        return len(self.dl)

    def __iter__(self):
        for x, y in self.dl:
            yield x, None, y


def make_loader(
    df: pd.DataFrame,
    arch: str,
    split: str | None,
    batch_size: int = 64,
    shuffle: bool = False,
    multiclass: bool = False,
    window: int = 8,
    topology: str = "star_peer",
    knn: int = 0,
    seed: int = 42,
):
    """Build the right loader for `arch`, restricted to one split."""
    target = "attack_type" if multiclass else "label"

    if arch == "gnn":
        graphs = build_graphs(df, topology=topology, knn=knn, split=split)
        return GraphBatchLoader(graphs, batch_size, shuffle, multiclass, seed)

    if arch == "transformer":
        X, y, y_multi = build_sequences(df, window=window, split=split)
        return TensorBatchLoader(X, y_multi if multiclass else y, batch_size, shuffle)

    sub = df if split is None else df[df["split"] == split]
    X = sub[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = sub[target].to_numpy(dtype=np.int64)
    return TensorBatchLoader(X, y, batch_size, shuffle)


def class_weights(df: pd.DataFrame, split: str = "train", multiclass: bool = False,
                  n_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency loss weights.

    Attacks are ~25% of windows and the rare families under 1%. Without
    reweighting the model can score 75% by predicting BENIGN everywhere, and
    the recall on Mirai and Recon -- the classes that matter most -- collapses.
    """
    sub = df[df["split"] == split] if split else df
    col = "attack_type" if multiclass else "label"
    counts = np.bincount(sub[col].to_numpy(dtype=int), minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)
