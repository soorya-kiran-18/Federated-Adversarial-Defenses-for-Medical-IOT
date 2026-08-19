"""A hospital participating in federated training (Steps 5-8).

One `HospitalClient` owns one shard of the dataset and never exposes it. Each
round it receives the global parameters, trains locally, and returns only the
updated parameters plus a sample count.

The class also hosts the two adversarial behaviours the project studies, so
that a "malicious hospital" is a configuration of a normal client rather than a
separate code path -- the server cannot tell them apart, which is the point:

    byzantine_attack  -- how a compromised hospital corrupts what it uploads
    dp / gan hooks    -- injected by Steps 6 and 7 without changing this file
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from swarmdef.config import Config
from swarmdef.data.schema import LABEL_CLASSES
from swarmdef.detector.data import class_weights, make_loader
from swarmdef.detector.models import Detector, build_detector
from swarmdef.detector.train import evaluate, train_model
from swarmdef.utils.logging import get_logger

log = get_logger("federated.client")

Update = list[np.ndarray]

BYZANTINE_ATTACKS = ("sign_flip", "gauss", "scale", "label_flip", "none")


@dataclass
class ClientReport:
    """What a client returns to the server after local training."""

    client_id: int
    parameters: Update
    n_samples: int
    train_loss: float = 0.0
    local_accuracy: float = 0.0
    epsilon: float = 0.0
    is_byzantine: bool = False
    extra: dict = field(default_factory=dict)


class HospitalClient:
    """Local training at one hospital."""

    def __init__(
        self,
        client_id: int,
        shard: pd.DataFrame,
        cfg: Config,
        device: torch.device,
        multiclass: bool = False,
        byzantine: bool = False,
    ) -> None:
        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        self.multiclass = multiclass
        self.byzantine = byzantine
        self.n_classes = len(LABEL_CLASSES) if multiclass else 2
        self.arch = cfg.detector.arch

        self.shard = shard
        if byzantine and cfg.federated.byzantine_attack == "label_flip":
            # Data poisoning: the client trains on inverted labels, so its
            # update is a *plausible* gradient pointing the wrong way. This is
            # far harder for a distance-based rule to reject than a scaled
            # update, because its norm looks entirely normal.
            self.shard = shard.copy()
            self.shard["label"] = 1 - self.shard["label"]

        self.train_loader = make_loader(
            self.shard, self.arch, "train", batch_size=cfg.detector.batch_size,
            shuffle=True, multiclass=multiclass, window=cfg.data.window_size,
            knn=cfg.detector.graph_knn if self.arch == "gnn" else 0, seed=cfg.seed + client_id,
        )
        self.val_loader = make_loader(
            self.shard, self.arch, "val", batch_size=cfg.detector.batch_size,
            multiclass=multiclass, window=cfg.data.window_size,
            knn=cfg.detector.graph_knn if self.arch == "gnn" else 0, seed=cfg.seed + client_id,
        )
        self.class_weight = class_weights(self.shard, "train", multiclass, self.n_classes)
        self.model = self._build()
        self.n_samples = self.train_loader.n_samples
        # Hooks: Step 6 sets `adversary`, Step 7 sets `privacy_engine`.
        self.adversary = None
        self.privacy_engine = None

    def _build(self) -> Detector:
        d = self.cfg.detector
        return build_detector(
            self.arch, n_classes=self.n_classes, hidden=d.hidden_dim, n_layers=d.n_layers,
            dropout=d.dropout, heads=d.heads, conv="sage", max_len=self.cfg.data.window_size,
        ).to(self.device)

    # ── federated interface ──────────────────────────────────────────────────
    def get_parameters(self) -> Update:
        return self.model.get_parameters()

    def set_parameters(self, params: Update) -> None:
        self.model.set_parameters(params)

    def fit(self, global_params: Update, round_idx: int = 0) -> ClientReport:
        """One round of local training, starting from the global model."""
        self.set_parameters(global_params)
        loader = self.train_loader

        # Step 6 hook: replace the loader with one that mixes in GAN-generated
        # evasive samples for adversarial training.
        if self.adversary is not None:
            loader = self.adversary.augment(self, round_idx)

        history = train_model(
            self.model, loader, None,
            epochs=self.cfg.detector.local_epochs, lr=self.cfg.detector.lr,
            weight_decay=self.cfg.detector.weight_decay, device=self.device,
            class_weight=self.class_weight, multiclass=self.multiclass,
            early_stopping=0, verbose=False,
            optimizer=getattr(self, "_dp_optimizer", None),
        )

        params = self.get_parameters()
        epsilon = 0.0
        if self.privacy_engine is not None:
            # Step 7 hook: clip + noise the update before it leaves the hospital.
            params = self.privacy_engine.privatise(global_params, params)
            epsilon = self.privacy_engine.epsilon()

        if self.byzantine:
            params = self.corrupt(params, global_params)

        metrics, _ = evaluate(self.model, self.val_loader, self.device, self.multiclass)
        return ClientReport(
            client_id=self.client_id, parameters=params, n_samples=self.n_samples,
            train_loss=history.train_loss[-1] if history.train_loss else 0.0,
            local_accuracy=metrics.accuracy, epsilon=epsilon, is_byzantine=self.byzantine,
        )

    # ── malicious behaviour (Step 8) ─────────────────────────────────────────
    def corrupt(self, params: Update, global_params: Update) -> Update:
        """Apply this client's model-poisoning strategy to its upload."""
        mode = self.cfg.federated.byzantine_attack
        scale = self.cfg.federated.byzantine_scale
        rng = np.random.default_rng(self.cfg.seed * 977 + self.client_id)

        if mode in ("label_flip", "none"):
            # Already poisoned at the data level (or not malicious at all).
            return params

        if mode == "sign_flip":
            # Send the negation of the honest update: pull the global model
            # away from the optimum at every step.
            return [g - scale * (p - g) for p, g in zip(params, global_params)]

        if mode == "scale":
            # Norm-scaling / model-replacement: amplify the update so it
            # dominates the average and effectively replaces the global model.
            return [g + scale * (p - g) for p, g in zip(params, global_params)]

        if mode == "gauss":
            # Random noise of comparable magnitude to a real update.
            return [
                (g + rng.normal(0, scale * (np.std(p - g) + 1e-3), size=np.shape(p))).astype(p.dtype)
                for p, g in zip(params, global_params)
            ]

        raise ValueError(f"unknown byzantine_attack {mode!r}; choose from {BYZANTINE_ATTACKS}")

    def evaluate_global(self, params: Update, split: str = "test"):
        """Score the global model on this hospital's own held-out data."""
        self.set_parameters(params)
        loader = make_loader(
            self.shard, self.arch, split, batch_size=self.cfg.detector.batch_size,
            multiclass=self.multiclass, window=self.cfg.data.window_size,
            knn=self.cfg.detector.graph_knn if self.arch == "gnn" else 0, seed=self.cfg.seed,
        )
        return evaluate(self.model, loader, self.device, self.multiclass)

    def __repr__(self) -> str:
        tag = " BYZANTINE" if self.byzantine else ""
        return f"<HospitalClient {self.client_id} n={self.n_samples}{tag}>"
