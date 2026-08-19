"""The adversarial engine: a GAN that invents evasive attack traffic (Step 6).

Threat model
------------
The adversary already controls an attacking host inside the ward and wants its
traffic classified BENIGN. It may reshape *how* the attack looks on the wire --
pad packets, slow a flood down, spread it over more connections -- but it may
not change what the attack fundamentally does. So the generator produces a
bounded perturbation, not an arbitrary sample:

    x_adv = clip( x + eps * tanh(G(x, z)) * mutable_mask )

Three constraints make the result a *functional* attack rather than a merely
adversarial vector, and each one is a claim the report has to be able to defend:

1. **Bounded (L-inf <= eps).** An unbounded perturbation could simply overwrite
   the sample with a benign one, which proves nothing about the detector.
2. **Masked.** Only features in `schema.mutable_mask()` may move. An attacker
   cannot un-tamper a vital it deliberately altered, nor hide the connections
   it must open, so `vital_zscore_max`, `conn_attempt_rate`, `n_unique_dst`
   and `flag_rst_ratio` are frozen.
3. **Directionally honest.** The perturbation is applied in the same normalised
   space the detector consumes, so the reported evasion rate is measured
   against exactly the features the model actually sees.

This is a white-box attack: the generator differentiates through the current
detector. That is the strongest reasonable adversary for evaluating a defence,
and it is what "zero-day-style" means operationally -- the attacker adapts to
*this* detector, not to a fixed signature list.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from swarmdef.data.schema import N_FEATURES, mutable_mask
from swarmdef.utils.logging import get_logger

log = get_logger("gan.generator")


class PerturbationGenerator(nn.Module):
    """Maps (attack sample, noise) to a bounded, masked perturbation."""

    def __init__(self, n_features: int = N_FEATURES, latent_dim: int = 16,
                 hidden: int = 64, epsilon: float = 0.15,
                 mutable_fraction: float = 1.0, seed: int = 42) -> None:
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.epsilon = epsilon

        self.net = nn.Sequential(
            nn.Linear(n_features + latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, n_features),
        )

        # Schema-level immutability, then an optional further restriction so the
        # attacker is limited to a subset of the features it could touch.
        mask = np.array(mutable_mask(), dtype=np.float32)
        if mutable_fraction < 1.0:
            rng = np.random.default_rng(seed)
            allowed = np.flatnonzero(mask)
            keep = rng.choice(allowed, size=max(1, int(len(allowed) * mutable_fraction)),
                              replace=False)
            mask = np.zeros_like(mask)
            mask[keep] = 1.0
        self.register_buffer("mask", torch.from_numpy(mask))

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = torch.randn(x.shape[0], self.latent_dim, device=x.device)
        delta = torch.tanh(self.net(torch.cat([x, z], dim=-1)))   # in [-1, 1]
        return x + self.epsilon * delta * self.mask               # bounded + masked

    @torch.no_grad()
    def perturb(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

    def budget_report(self, x: torch.Tensor, x_adv: torch.Tensor) -> dict:
        """Confirm the generated samples respect the stated threat model."""
        delta = (x_adv - x).detach()
        frozen = (self.mask == 0)
        return {
            "linf": float(delta.abs().max()),
            "l2_mean": float(delta.norm(dim=-1).mean()),
            "epsilon": self.epsilon,
            "within_budget": bool(delta.abs().max() <= self.epsilon + 1e-5),
            "frozen_features_untouched": bool(delta[:, frozen].abs().max() < 1e-6),
            "n_mutable": int(self.mask.sum()),
        }
