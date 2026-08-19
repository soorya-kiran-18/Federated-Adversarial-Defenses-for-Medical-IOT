"""Detector architectures for the defence layer (Step 4).

Three models over the same 30-D flow features, so the architecture comparison
is controlled:

    MLPDetector          per-window baseline. No context at all.
    TransformerDetector  temporal context: attends over a device's recent windows.
    GNNDetector          spatial context: attends over a device's neighbours in
                         the same time window. This is the report's primary model.

Design constraints carried from later steps
-------------------------------------------
1. **LayerNorm, never BatchNorm.** Opacus (Step 7) cannot compute per-sample
   gradients through BatchNorm because it mixes information across the batch,
   and it is also the classic source of silent divergence in federated
   averaging when clients have non-IID batches. Using LayerNorm from the start
   means Steps 5 and 7 need no architectural rewrite.
2. **Flat, ordered parameter vectors.** `get_parameters`/`set_parameters`
   exchange plain NumPy arrays in a fixed order, which is exactly what Flower
   transmits and what the Byzantine aggregators (Step 8) operate on.
3. **No in-place activations.** In-place ops break Opacus' gradient hooks.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from swarmdef.data.schema import N_FEATURES


class Detector(nn.Module):
    """Base class providing the parameter exchange the FL layer relies on."""

    def get_parameters(self) -> list[np.ndarray]:
        """Model weights as an ordered list of NumPy arrays (Flower's format)."""
        return [p.detach().cpu().numpy() for p in self.state_dict().values()]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        keys = list(self.state_dict().keys())
        if len(keys) != len(params):
            raise ValueError(f"expected {len(keys)} tensors, got {len(params)}")
        state = OrderedDict(
            (k, torch.as_tensor(v)) for k, v in zip(keys, params)
        )
        self.load_state_dict(state, strict=True)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── baseline: no context ─────────────────────────────────────────────────────
class MLPDetector(Detector):
    """Per-window classifier. Deliberately context-free -- it is the control.

    Any accuracy the GNN or Transformer gains over this is attributable to
    context, which is the claim Section 4.1.2 of the report actually makes.
    """

    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 64,
                 n_layers: int = 2, n_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if x.dim() == 3:          # accept a sequence view; use the final step
            x = x[:, -1, :]
        return self.head(self.body(x))


# ── temporal context ─────────────────────────────────────────────────────────
class _SelfAttention(nn.Module):
    """Single-head-group self-attention built from plain Linear layers.

    Written out rather than using `nn.MultiheadAttention` because Opacus cannot
    attach per-sample gradient hooks to the fused packed-projection weight that
    PyTorch's implementation uses. Every parameter here lives in an ordinary
    Linear, which Opacus supports natively.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        def split(v):
            return v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att = self.drop(torch.softmax(att, dim=-1))
        ctx = torch.matmul(att, v).transpose(1, 2).reshape(b, t, d)
        return self.out(ctx)


class _EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x)))     # pre-norm: not in-place
        return x + self.drop(self.ff(self.n2(x)))


class TransformerDetector(Detector):
    """Attends over a device's recent flow windows to classify the current one."""

    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 64, n_layers: int = 2,
                 heads: int = 4, n_classes: int = 2, dropout: float = 0.2,
                 max_len: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        # Fixed sinusoidal positions: learned embeddings would add parameters
        # that differ in meaning across clients with different sequence lengths.
        pe = torch.zeros(max_len, hidden)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)
        self.blocks = nn.ModuleList([_EncoderBlock(hidden, heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if x.dim() == 2:                       # a lone window -> length-1 sequence
            x = x.unsqueeze(1)
        h = self.proj(x) + self.pe[: x.shape[1]].unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h)[:, -1, :])   # predict from the latest step


# ── spatial context: the report's primary detector ───────────────────────────
class GNNDetector(Detector):
    """GraphSAGE over the device graph, classifying each device-window node.

    SAGEConv keeps a separate weight for a node's own features and for the mean
    of its neighbours. That separation is what the task needs: the model must be
    able to say "my DUP rate is high *and my neighbours' is not*", which is the
    MITM-versus-congestion decision. A convolution that averaged self and
    neighbours together would blur exactly the contrast being tested.
    """

    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 64, n_layers: int = 2,
                 n_classes: int = 2, dropout: float = 0.2, conv: str = "sage",
                 heads: int = 4) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv

        self.conv_kind = conv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            if conv == "sage":
                layer = SAGEConv(d, hidden)
            elif conv == "gcn":
                layer = GCNConv(d, hidden)
            elif conv == "gat":
                layer = GATConv(d, hidden // heads, heads=heads, concat=True)
            else:
                raise ValueError(f"unknown conv {conv!r}")
            self.convs.append(layer)
            self.norms.append(nn.LayerNorm(hidden))
            d = hidden
        self.dropout = dropout
        # A residual path straight from the raw features. Without it a node with
        # no informative neighbours (an isolated gateway in a quiet window) would
        # depend entirely on message passing to be classified.
        self.skip = nn.Linear(in_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None,
                *args, **kwargs) -> torch.Tensor:
        # No edges means every node is isolated. That is a meaningful input, not
        # an error: adversarial samples (Step 6) are single windows the attacker
        # controls, with no ward context attached. The skip connection is what
        # keeps the model usable in that regime.
        if edge_index is None:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        h = None
        for conv, norm in zip(self.convs, self.norms):
            inp = x if h is None else h
            h = F.gelu(norm(conv(inp, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + self.skip(x)
        return self.head(h)


ARCHITECTURES = {"mlp": MLPDetector, "transformer": TransformerDetector, "gnn": GNNDetector}


def build_detector(arch: str = "gnn", n_classes: int = 2, **kw) -> Detector:
    """Factory used by every entry point so all steps build identical models."""
    if arch not in ARCHITECTURES:
        raise KeyError(f"unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
    cls = ARCHITECTURES[arch]
    valid = {
        "mlp": {"in_dim", "hidden", "n_layers", "n_classes", "dropout"},
        "transformer": {"in_dim", "hidden", "n_layers", "heads", "n_classes", "dropout", "max_len"},
        "gnn": {"in_dim", "hidden", "n_layers", "n_classes", "dropout", "conv", "heads"},
    }[arch]
    return cls(n_classes=n_classes, **{k: v for k, v in kw.items() if k in valid})
