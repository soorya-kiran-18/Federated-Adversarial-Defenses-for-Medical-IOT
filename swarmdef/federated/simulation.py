"""The federated training loop -- the closed loop of Section 3 of the report.

    for each round:
        broadcast theta_global to the selected hospitals
        each hospital trains locally on data that never leaves it   (Step 5)
          ... optionally hardened against GAN-generated evasions    (Step 6)
          ... optionally clipped + noised before upload             (Step 7)
        the server aggregates the uploads, robustly                 (Step 8)
        the new global model is evaluated on a held-out test set

This orchestrator is deliberately framework-independent so that the mechanism
under study is the only thing that changes between experiments. A genuine
Flower deployment of the same loop lives in `swarmdef/federated/flower_app.py`;
it reuses these clients and aggregators unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from swarmdef.config import Config
from swarmdef.data.build import load_hospital, load_pooled
from swarmdef.data.schema import LABEL_CLASSES
from swarmdef.detector.data import make_loader
from swarmdef.detector.models import build_detector
from swarmdef.detector.train import evaluate
from swarmdef.eval.logger import ExperimentLogger, RoundRecord
from swarmdef.federated.aggregators import aggregate
from swarmdef.federated.client import HospitalClient
from swarmdef.utils.logging import get_logger
from swarmdef.utils.metrics import DetectionMetrics
from swarmdef.utils.seed import set_seed

log = get_logger("federated.sim")


@dataclass
class RoundResult:
    round: int
    metrics: DetectionMetrics
    extra: dict
    mean_epsilon: float = 0.0
    client_reports: list = field(default_factory=list)
    seconds: float = 0.0


class FederatedSimulation:
    """Runs K hospitals through N rounds of federated training."""

    def __init__(self, cfg: Config, device: torch.device, multiclass: bool = False,
                 run_name: str | None = None, adversarial: bool = False,
                 track_adversarial: bool = False) -> None:
        self.cfg = cfg
        self.device = device
        self.multiclass = multiclass
        # adversarial       -> clients adversarially retrain on GAN samples (defence)
        # track_adversarial ->each round, also score the global model under attack
        self.adversarial = adversarial
        self.track_adversarial = track_adversarial or adversarial
        self.n_classes = len(LABEL_CLASSES) if multiclass else 2
        self.run_name = run_name or cfg.eval.run_name
        set_seed(cfg.seed)

        # Which hospitals are compromised. Fixed up front so the same nodes are
        # malicious in every round -- a hospital does not get re-compromised at
        # random, and holding it fixed keeps rounds comparable.
        n_byz = cfg.federated.byzantine_clients
        self.byzantine_ids = set(range(cfg.federated.n_clients - n_byz, cfg.federated.n_clients))

        self.clients = [
            HospitalClient(i, load_hospital(cfg, i), cfg, device, multiclass,
                           byzantine=(i in self.byzantine_ids))
            for i in range(cfg.federated.n_clients)
        ]

        # The server's held-out evaluation set: the pooled test split. Every
        # experiment is scored on exactly this data, so federated, DP and
        # Byzantine runs are directly comparable to the Step 4 baseline.
        pooled = load_pooled(cfg)
        self.test_loader = make_loader(
            pooled, cfg.detector.arch, "test", batch_size=cfg.detector.batch_size,
            multiclass=multiclass, window=cfg.data.window_size,
            knn=cfg.detector.graph_knn if cfg.detector.arch == "gnn" else 0, seed=cfg.seed,
        )

        self.pooled = pooled
        if self.adversarial:
            from swarmdef.gan.adv_train import ClientAdversary
            for c in self.clients:
                c.adversary = ClientAdversary(cfg, device, c.client_id)
            log.info("Adversarial training enabled:each hospital runs its own GAN "
                     "(eps=%.2f, warmup=%d rounds)", cfg.gan.epsilon, cfg.gan.warmup_rounds)

        self.global_model = self._build()
        self.global_params = self.global_model.get_parameters()
        self.logger = ExperimentLogger(self.run_name, cfg.eval.log_dir)
        self.logger.set_meta(
            aggregator=cfg.federated.aggregator, n_clients=cfg.federated.n_clients,
            byzantine_clients=n_byz, byzantine_attack=cfg.federated.byzantine_attack,
            byzantine_ids=sorted(self.byzantine_ids), arch=cfg.detector.arch,
            dp_enabled=cfg.privacy.enabled, gan_enabled=cfg.gan.enabled,
            multiclass=multiclass, seed=cfg.seed,
        )
        self.history: list[RoundResult] = []

    def _build(self):
        d = self.cfg.detector
        return build_detector(
            d.arch, n_classes=self.n_classes, hidden=d.hidden_dim, n_layers=d.n_layers,
            dropout=d.dropout, heads=d.heads, conv="sage", max_len=self.cfg.data.window_size,
        ).to(self.device)

    # ── one round ────────────────────────────────────────────────────────────
    def select_clients(self, round_idx: int) -> list[HospitalClient]:
        frac = self.cfg.federated.clients_per_round
        if frac >= 1.0:
            return self.clients
        rng = np.random.default_rng(self.cfg.seed * 31 + round_idx)
        k = max(1, int(round(frac * len(self.clients))))
        return [self.clients[i] for i in sorted(rng.choice(len(self.clients), k, replace=False))]

    def run_round(self, round_idx: int) -> RoundResult:
        t0 = time.time()
        selected = self.select_clients(round_idx)

        reports = [c.fit(self.global_params, round_idx) for c in selected]
        updates = [r.parameters for r in reports]
        # Weight by local sample count: a hospital with more traffic should
        # carry proportionally more of the average (standard FedAvg weighting).
        weights = [r.n_samples for r in reports]

        f = self.cfg.federated.krum_f
        self.global_params = aggregate(
            self.cfg.federated.aggregator, updates, weights,
            f=f if f is not None else self.cfg.federated.byzantine_clients,
            trim_ratio=self.cfg.federated.trim_ratio,
        )

        self.global_model.set_parameters(self.global_params)
        metrics, extra = evaluate(self.global_model, self.test_loader, self.device, self.multiclass)
        mean_eps = float(np.mean([r.epsilon for r in reports])) if reports else 0.0

        if self.track_adversarial:
            # Score the *current* global model against an adversary trained
            # fresh on it. Reusing an older generator would understate the
            # threat: a real attacker re-optimises against whatever is deployed.
            from swarmdef.gan.adv_train import evaluate_under_attack
            adv = evaluate_under_attack(
                self.global_model, self.cfg, self.device, self.pooled, "test",
                fit_epochs=self.cfg.gan.epochs,
            )
            extra.update({f"adv_{k}": v for k, v in adv.items()})

        result = RoundResult(round_idx, metrics, extra, mean_eps, reports, time.time() - t0)
        self.history.append(result)

        self.logger.log(RoundRecord(
            run=self.run_name, round=round_idx, accuracy=metrics.accuracy, f1=metrics.f1,
            precision=metrics.precision, recall=metrics.recall, fpr=metrics.fpr,
            auc=extra.get("auc", 0.0), evasion_rate=extra.get("evasion_rate", 0.0),
            loss=float(np.mean([r.train_loss for r in reports])) if reports else 0.0,
            epsilon=mean_eps, n_clients=len(selected),
            n_byzantine=sum(r.is_byzantine for r in reports),
            aggregator=self.cfg.federated.aggregator,
        ))
        return result

    # ── full run ─────────────────────────────────────────────────────────────
    def run(self, n_rounds: int | None = None, verbose: bool = True) -> list[RoundResult]:
        n_rounds = n_rounds or self.cfg.federated.n_rounds
        if verbose:
            byz = f", {len(self.byzantine_ids)} byzantine ({self.cfg.federated.byzantine_attack})" \
                  if self.byzantine_ids else ""
            log.info("Federated run '%s': %d clients%s, %s aggregation, %d rounds",
                     self.run_name, len(self.clients), byz,
                     self.cfg.federated.aggregator, n_rounds)

        for r in range(1, n_rounds + 1):
            res = self.run_round(r)
            if verbose:
                eps = f" eps={res.mean_epsilon:.2f}" if res.mean_epsilon else ""
                adv = ""
                if "adv_adv_accuracy" in res.extra:
                    adv = (f" | under attack: acc {res.extra['adv_adv_accuracy']:.4f}"
                           f" evasion {100 * res.extra['adv_evasion_rate']:.1f}%")
                log.info("  round %2d/%d | acc %.4f f1 %.4f fpr %.4f%s%s | %.1fs",
                         r, n_rounds, res.metrics.accuracy, res.metrics.f1,
                         res.metrics.fpr, eps, adv, res.seconds)
        self.logger.save()
        return self.history

    # ── reporting ────────────────────────────────────────────────────────────
    def curve(self, key: str = "accuracy") -> tuple[list[int], list[float]]:
        rounds = [h.round for h in self.history]
        if key in ("auc", "evasion_rate"):
            return rounds, [h.extra.get(key, 0.0) for h in self.history]
        return rounds, [getattr(h.metrics, key) for h in self.history]

    def final(self) -> RoundResult:
        return self.history[-1]

    def best(self, key: str = "f1") -> RoundResult:
        return max(self.history, key=lambda h: getattr(h.metrics, key))

    def per_hospital_report(self) -> pd.DataFrame:
        """How the final global model performs at each hospital individually.

        This is the federated-specific question a pooled baseline cannot answer:
        does the shared model actually help the hospital that contributed the
        least data, or does it only serve the majority?
        """
        rows = []
        for c in self.clients:
            m, extra = c.evaluate_global(self.global_params, "test")
            rows.append({
                "hospital": c.client_id, "byzantine": c.byzantine,
                "train_samples": c.n_samples, "accuracy": round(m.accuracy, 4),
                "f1": round(m.f1, 4), "recall": round(m.recall, 4), "fpr": round(m.fpr, 4),
            })
        return pd.DataFrame(rows).set_index("hospital")
