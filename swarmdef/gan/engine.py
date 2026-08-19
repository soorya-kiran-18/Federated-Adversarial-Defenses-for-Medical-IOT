"""Training the adversarial generator against the live detector (Step 6).

The generator is trained to *minimise* the detector's confidence that a sample
is malicious. The detector plays the discriminator role from the report's
min-max objective; we do not train a separate discriminator network, because
the quantity of interest is evasion of the deployed detector, not sample
realism against an auxiliary critic.

    L_G = CE( D(x_adv), BENIGN )  +  lambda * ||x_adv - x||_2

The second term keeps the perturbation small inside the allowed budget: an
attack that spends its whole budget is easier for adversarial training to
learn, and a real adversary prefers the smallest change that works.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from swarmdef.gan.generator import PerturbationGenerator
from swarmdef.utils.logging import get_logger
from swarmdef.utils.metrics import attack_success_rate

log = get_logger("gan.engine")


class AdversarialEngine:
    """Trains a generator against a detector, and reports how well it evades."""

    def __init__(self, cfg, device: torch.device, seed: int = 42) -> None:
        self.cfg = cfg
        self.device = device
        g = cfg.gan
        self.generator = PerturbationGenerator(
            latent_dim=g.latent_dim, hidden=g.hidden_dim, epsilon=g.epsilon,
            mutable_fraction=g.mutable_fraction, seed=seed,
        ).to(device)
        self.opt = torch.optim.Adam(self.generator.parameters(), lr=g.lr_g, betas=(0.5, 0.999))
        self.sparsity_lambda = 0.05

    # ── training ─────────────────────────────────────────────────────────────
    def fit(self, detector: nn.Module, attack_x: np.ndarray, epochs: int | None = None,
            batch_size: int | None = None, verbose: bool = False) -> dict:
        """Train the generator to make `attack_x` look benign to `detector`."""
        epochs = epochs or self.cfg.gan.epochs
        batch_size = batch_size or self.cfg.gan.batch_size
        if len(attack_x) == 0:
            return {"loss": float("nan"), "evasion": 0.0}

        X = torch.as_tensor(attack_x, dtype=torch.float32, device=self.device)
        # The detector is the adversary's target, not its student: freeze it so
        # only the generator learns during this phase.
        detector.eval()
        for p in detector.parameters():
            p.requires_grad_(False)

        benign = torch.zeros(len(X), dtype=torch.long, device=self.device)
        criterion = nn.CrossEntropyLoss()
        last_loss = float("nan")

        for _ in range(epochs):
            perm = torch.randperm(len(X), device=self.device)
            for start in range(0, len(X), batch_size):
                idx = perm[start:start + batch_size]
                x = X[idx]
                self.opt.zero_grad(set_to_none=True)
                x_adv = self.generator(x)
                logits = self._detector_logits(detector, x_adv)
                # Push the detector toward calling these BENIGN, while keeping
                # the perturbation as small as the budget allows.
                loss = criterion(logits, benign[idx]) \
                    + self.sparsity_lambda * (x_adv - x).norm(dim=-1).mean()
                loss.backward()
                self.opt.step()
                last_loss = float(loss.detach())

        for p in detector.parameters():
            p.requires_grad_(True)

        report = self.evaluate(detector, attack_x)
        if verbose:
            log.info("  GAN: loss %.4f | evasion %.1f%% | Linf %.3f",
                     last_loss, 100 * report["evasion"], report["linf"])
        return {"loss": last_loss, **report}

    def _detector_logits(self, detector: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Score perturbed feature vectors, whatever the detector's input form.

        A GNN expects an edge_index and a Transformer expects a sequence. The
        adversary perturbs a single window's features, so for those models the
        sample is presented as an isolated node / a length-1 sequence. That is
        the honest reading of the threat model: the attacker controls its own
        traffic, not the behaviour of the ward around it.
        """
        from swarmdef.detector.models import GNNDetector, TransformerDetector

        if isinstance(detector, GNNDetector):
            empty = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            return detector(x, empty)
        if isinstance(detector, TransformerDetector):
            return detector(x.unsqueeze(1))
        return detector(x)

    # ── evaluation ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, detector: nn.Module, attack_x: np.ndarray) -> dict:
        """Fraction of true attacks the generator gets past the detector."""
        if len(attack_x) == 0:
            return {"evasion": 0.0, "evasion_before": 0.0, "linf": 0.0}
        detector.eval()
        X = torch.as_tensor(attack_x, dtype=torch.float32, device=self.device)
        x_adv = self.generator.perturb(X)

        pred_clean = self._detector_logits(detector, X).argmax(-1).cpu().numpy()
        pred_adv = self._detector_logits(detector, x_adv).argmax(-1).cpu().numpy()
        truth = np.ones(len(X), dtype=int)

        budget = self.generator.budget_report(X, x_adv)
        return {
            "evasion": attack_success_rate(truth, (pred_adv != 0).astype(int)),
            "evasion_before": attack_success_rate(truth, (pred_clean != 0).astype(int)),
            "linf": budget["linf"],
            "within_budget": budget["within_budget"],
            "frozen_ok": budget["frozen_features_untouched"],
        }

    @torch.no_grad()
    def generate(self, attack_x: np.ndarray) -> np.ndarray:
        """Produce adversarial versions of the given attack samples."""
        if len(attack_x) == 0:
            return attack_x
        X = torch.as_tensor(attack_x, dtype=torch.float32, device=self.device)
        return self.generator.perturb(X).cpu().numpy()
