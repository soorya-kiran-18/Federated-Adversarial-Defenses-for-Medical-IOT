"""Adversarial training: hardening the detector against the GAN (Step 6).

Each hospital runs its own generator against its own copy of the detector, then
retrains on a mixture of real traffic and the evasive samples that generator
found. This is the closed loop the report describes: the attacker searches for
blind spots, the defender learns them, and the improvement propagates to every
other hospital through the next federated aggregation.

The federated angle matters and is worth stating plainly: a blind spot
discovered by *one* hospital's adversary is patched for *all* of them after
aggregation, without any hospital sharing traffic. That is the project's actual
security claim, and Step 6 is where it becomes measurable.
"""
from __future__ import annotations

import numpy as np
import torch

from swarmdef.data.schema import FEATURE_NAMES
from swarmdef.detector.data import TensorBatchLoader
from swarmdef.gan.engine import AdversarialEngine
from swarmdef.utils.logging import get_logger

log = get_logger("gan.adv_train")


class ClientAdversary:
    """The per-hospital adversary, plugged into `HospitalClient.adversary`."""

    def __init__(self, cfg, device: torch.device, client_id: int = 0) -> None:
        self.cfg = cfg
        self.device = device
        self.client_id = client_id
        self.engine = AdversarialEngine(cfg, device, seed=cfg.seed + client_id)
        self.last_report: dict = {}

    def _attack_features(self, client) -> np.ndarray:
        """The hospital's own attack windows, in normalised feature space."""
        train = client.shard[client.shard["split"] == "train"]
        attacks = train[train["label"] == 1]
        return attacks[FEATURE_NAMES].to_numpy(dtype=np.float32)

    def augment(self, client, round_idx: int):
        """Return a loader mixing real traffic with fresh adversarial samples.

        Before `warmup_rounds` the loader is returned untouched: attacking a
        model that has not yet converged measures nothing about the defence,
        and it destabilises early federated rounds.
        """
        g = self.cfg.gan
        if not g.enabled or round_idx <= g.warmup_rounds:
            return client.train_loader

        attack_x = self._attack_features(client)
        if len(attack_x) < 8:
            return client.train_loader

        # 1. The adversary adapts to the *current* global detector.
        self.last_report = self.engine.fit(client.model, attack_x, verbose=False)

        # 2. Generate evasive variants and label them as what they really are:
        #    attacks. This is what teaches the detector the blind spot.
        n_adv = max(1, int(len(attack_x) * g.adv_train_ratio))
        idx = np.random.default_rng(self.cfg.seed + round_idx).choice(
            len(attack_x), size=n_adv, replace=n_adv > len(attack_x))
        x_adv = self.engine.generate(attack_x[idx])

        # 3. Mix with the hospital's real training data.
        train = client.shard[client.shard["split"] == "train"]
        X_real = train[FEATURE_NAMES].to_numpy(dtype=np.float32)
        y_real = train["label"].to_numpy(dtype=np.int64)
        X = np.concatenate([X_real, x_adv]).astype(np.float32)
        y = np.concatenate([y_real, np.ones(len(x_adv), dtype=np.int64)])

        # Adversarial samples are single windows with no graph or history, so
        # the augmented loader is the tabular one regardless of architecture.
        # The detector still sees them through its own input path.
        return TensorBatchLoader(X, y, batch_size=self.cfg.detector.batch_size, shuffle=True)


def evaluate_under_attack(detector, cfg, device, df, split: str = "test",
                          engine: AdversarialEngine | None = None,
                          fit_epochs: int | None = None) -> dict:
    """Measure a detector's accuracy on adversarially perturbed attack traffic.

    Benign windows are left untouched -- an attacker has no reason to perturb
    traffic it does not control -- so the reported accuracy is on a test set
    where only the attack half has been made evasive. That is the operationally
    meaningful number: how much of the attack traffic still gets caught.
    """
    sub = df[df["split"] == split]
    X = sub[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = sub["label"].to_numpy(dtype=np.int64)
    attack_mask = y == 1
    if attack_mask.sum() == 0:
        return {}

    engine = engine or AdversarialEngine(cfg, device, seed=cfg.seed)
    if fit_epochs:
        engine.fit(detector, X[attack_mask], epochs=fit_epochs)

    X_adv = X.copy()
    X_adv[attack_mask] = engine.generate(X[attack_mask])

    from swarmdef.detector.train import evaluate

    clean_m, _ = evaluate(detector, TensorBatchLoader(X, y, cfg.detector.batch_size), device)
    adv_m, _ = evaluate(detector, TensorBatchLoader(X_adv, y, cfg.detector.batch_size), device)
    report = engine.evaluate(detector, X[attack_mask])

    return {
        "clean_accuracy": clean_m.accuracy, "clean_f1": clean_m.f1, "clean_recall": clean_m.recall,
        "adv_accuracy": adv_m.accuracy, "adv_f1": adv_m.f1, "adv_recall": adv_m.recall,
        "evasion_rate": report["evasion"], "evasion_before": report["evasion_before"],
        "linf": report["linf"], "within_budget": report["within_budget"],
        "frozen_ok": report["frozen_ok"],
        "n_attack_samples": int(attack_mask.sum()),
    }
