"""Device-graph construction: flow windows -> PyTorch Geometric graphs (Step 4).

Following Section 4.1.2 of the report, the MIoT segment is modelled as an
undirected graph G = (V, E) where V are medical devices and E are the MQTT
communication links between them. One graph is built per
(hospital, time-window); its nodes are that hospital's devices during that
second, each carrying its 30-D behavioural vector and its own label. The task
is therefore **node classification**: which devices are under attack right now.

Why the graph earns its place
-----------------------------
Several benign confounders are *segment-wide* while several attacks are
*device-local*, and telling them apart requires looking at a device's
neighbours:

    NetworkCongestion raises DUP flags on every device at once;
    an MITM interception proxy raises DUP on exactly one.
    ShiftHandover lifts traffic across the whole fleet;
    a DDoS lifts it on a single target.

A per-row model sees only "DUP is high" and must guess. A GNN sees whether the
neighbours agree, which is precisely the information that disambiguates them.

Topology
--------
    star       every clinical device <-> the ward gateway. This is the real
               MQTT path: devices do not talk peer-to-peer, they publish
               through the broker, which the gateway observes.
    peer       devices of the same type <-> each other. Two infusion pumps
               should behave alike; a divergence between them is evidence.
    knn        additional edges to the k most behaviourally similar devices,
               which lets the graph adapt when the physical topology is
               unknown (as it is for CIC-IoT2023 captures).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from swarmdef.data.schema import FEATURE_NAMES
from swarmdef.utils.logging import get_logger

log = get_logger("detector.graph")

GATEWAY_SUFFIX = "gateway"


def _device_type(device_id: str) -> str:
    """`H0-patient_monitor-01` -> `patient_monitor`."""
    parts = str(device_id).split("-")
    return parts[1] if len(parts) > 1 else str(device_id)


def build_edges(
    device_ids: list[str],
    features: np.ndarray,
    topology: str = "star_peer",
    knn: int = 0,
) -> torch.Tensor:
    """Return an undirected `edge_index` (2, E) for one window's devices."""
    n = len(device_ids)
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    types = [_device_type(d) for d in device_ids]
    gateways = [i for i, d in enumerate(device_ids) if GATEWAY_SUFFIX in str(d)]
    pairs: set[tuple[int, int]] = set()

    def link(a: int, b: int) -> None:
        if a != b:
            pairs.add((min(a, b), max(a, b)))

    if "star" in topology:
        # Everything routes through the broker, observed at the gateway. With no
        # gateway present (e.g. a CIC capture) fall back to node 0 as the hub so
        # the graph never degenerates into isolated nodes.
        hubs = gateways or [0]
        for hub in hubs:
            for i in range(n):
                link(hub, i)

    if "peer" in topology:
        for i in range(n):
            for j in range(i + 1, n):
                if types[i] == types[j]:
                    link(i, j)

    if "full" in topology:
        for i in range(n):
            for j in range(i + 1, n):
                link(i, j)

    if knn > 0 and n > 1:
        # Behavioural similarity in normalised feature space.
        d = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        for i in range(n):
            for j in np.argsort(d[i])[: min(knn, n - 1)]:
                link(i, int(j))

    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)

    src = [a for a, b in pairs] + [b for a, b in pairs]
    dst = [b for a, b in pairs] + [a for a, b in pairs]
    return torch.tensor([src, dst], dtype=torch.long)


def build_graphs(
    df: pd.DataFrame,
    topology: str = "star_peer",
    knn: int = 0,
    split: str | None = None,
) -> list[Data]:
    """Turn a flow-window table into one PyG graph per (hospital, time-window)."""
    if split is not None:
        df = df[df["split"] == split]
    if df.empty:
        return []

    graphs: list[Data] = []
    for (hospital_id, t_start), group in df.groupby(["hospital_id", "t_start"], sort=True):
        group = group.sort_values("device_id")
        x = group[FEATURE_NAMES].to_numpy(dtype=np.float32)
        device_ids = group["device_id"].astype(str).tolist()
        edge_index = build_edges(device_ids, x, topology, knn)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            y=torch.tensor(group["label"].to_numpy(), dtype=torch.long),
            y_multi=torch.tensor(group["attack_type"].to_numpy(), dtype=torch.long),
        )
        data.hospital_id = int(hospital_id)
        data.t_start = float(t_start)
        data.num_nodes = x.shape[0]
        graphs.append(data)

    return graphs


def graph_stats(graphs: list[Data]) -> dict:
    """Summary used to sanity-check topology choices."""
    if not graphs:
        return {"graphs": 0}
    nodes = np.array([g.num_nodes for g in graphs])
    edges = np.array([g.edge_index.shape[1] for g in graphs])
    labels = torch.cat([g.y for g in graphs]).numpy()
    return {
        "graphs": len(graphs),
        "nodes_total": int(nodes.sum()),
        "nodes_per_graph_mean": float(nodes.mean()),
        "edges_per_graph_mean": float(edges.mean()),
        "avg_degree": float(edges.sum() / max(nodes.sum(), 1)),
        "attack_node_fraction": float(labels.mean()),
    }


# ── sequence view, for the Transformer detector ──────────────────────────────
def build_sequences(
    df: pd.DataFrame, window: int = 8, split: str | None = None, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-device sliding windows of consecutive flow vectors.

    Returns (X, y, y_multi) where X is (N, window, 30). The label is that of the
    *final* step, so the model predicts the present from recent history -- the
    causal setting an online IDS actually faces.

    Sequences are built inside a single split, so a window never spans the
    train/test boundary.
    """
    if split is not None:
        df = df[df["split"] == split]
    if df.empty:
        return (np.zeros((0, window, len(FEATURE_NAMES)), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64))

    seqs, ys, ym = [], [], []
    for _, group in df.groupby(["hospital_id", "device_id"], sort=True):
        group = group.sort_values("t_start")
        feats = group[FEATURE_NAMES].to_numpy(dtype=np.float32)
        lab = group["label"].to_numpy()
        lab_m = group["attack_type"].to_numpy()
        n = len(group)
        if n == 0:
            continue
        # Left-pad the first windows by repeating the earliest observation, so
        # early traffic is not silently dropped from the evaluation.
        pad = np.repeat(feats[:1], window - 1, axis=0) if n >= 1 else feats
        padded = np.concatenate([pad, feats], axis=0)
        for end in range(window - 1, len(padded), stride):
            start = end - window + 1
            idx = end - (window - 1)
            if idx >= n:
                break
            seqs.append(padded[start:end + 1])
            ys.append(lab[idx])
            ym.append(lab_m[idx])

    if not seqs:
        return (np.zeros((0, window, len(FEATURE_NAMES)), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64))
    return (np.stack(seqs).astype(np.float32),
            np.array(ys, dtype=np.int64), np.array(ym, dtype=np.int64))
