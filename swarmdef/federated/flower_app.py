"""Genuine Flower (flwr) deployment of the same federated loop.

`swarmdef/federated/simulation.py` runs the experiments: it is deterministic,
fast and easy to instrument, which is what the ablations in Steps 6-8 need.
This module proves the same system runs under a real federated framework, with
Flower's client/server protocol, serialisation and strategy machinery -- the
path a production deployment would actually take.

Both share the identical `HospitalClient` and aggregation rules, so a result
obtained in simulation is a result about this system, not about the harness.

    python scripts/train_federated_flower.py --rounds 10
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from swarmdef.config import Config
from swarmdef.data.build import load_hospital, load_pooled
from swarmdef.data.schema import LABEL_CLASSES
from swarmdef.detector.data import make_loader
from swarmdef.detector.models import build_detector
from swarmdef.detector.train import evaluate
from swarmdef.federated.aggregators import aggregate
from swarmdef.federated.client import HospitalClient
from swarmdef.utils.logging import get_logger
from swarmdef.utils.seed import resolve_device

log = get_logger("federated.flower")


def _flwr():
    import flwr
    return flwr


class FlowerHospitalClient:
    """Adapts `HospitalClient` to Flower's NumPyClient interface."""

    def __init__(self, client: HospitalClient) -> None:
        self.client = client

    def get_parameters(self, config: dict | None = None) -> list[np.ndarray]:
        return self.client.get_parameters()

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
        report = self.client.fit(parameters, int(config.get("server_round", 0)))
        return report.parameters, report.n_samples, {
            "train_loss": float(report.train_loss),
            "local_accuracy": float(report.local_accuracy),
            "epsilon": float(report.epsilon),
            "byzantine": bool(report.is_byzantine),
        }

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
        metrics, _ = self.client.evaluate_global(parameters, "val")
        n = max(self.client.n_samples, 1)
        return float(1.0 - metrics.accuracy), n, {"accuracy": metrics.accuracy, "f1": metrics.f1}


def build_strategy(cfg: Config, evaluate_fn=None):
    """A Flower Strategy whose aggregation is our own robust rule.

    Flower ships FedAvg, FedMedian and a few others, but not Krum with the
    parameterisation this project needs, and not the same code path the
    simulation uses. Subclassing FedAvg and overriding only `aggregate_fit`
    means the Byzantine defence under test is *literally the same function* in
    both harnesses -- there is no second implementation to diverge.
    """
    from flwr.server.strategy import FedAvg

    class RobustStrategy(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

            if not results:
                return None, {}
            updates = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
            weights = [fr.num_examples for _, fr in results]
            merged = aggregate(
                cfg.federated.aggregator, updates, weights,
                f=cfg.federated.krum_f if cfg.federated.krum_f is not None
                  else cfg.federated.byzantine_clients,
                trim_ratio=cfg.federated.trim_ratio,
            )
            n_byz = sum(int(fr.metrics.get("byzantine", False)) for _, fr in results)
            eps = float(np.mean([fr.metrics.get("epsilon", 0.0) for _, fr in results]))
            log.info("  [flower] round %d: aggregated %d updates with %s (%d byzantine)",
                     server_round, len(updates), cfg.federated.aggregator, n_byz)
            return ndarrays_to_parameters(merged), {"epsilon": eps, "n_byzantine": n_byz}

    return RobustStrategy(
        fraction_fit=cfg.federated.clients_per_round,
        fraction_evaluate=0.0,
        min_fit_clients=cfg.federated.n_clients,
        min_available_clients=cfg.federated.n_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
    )


def run_flower_simulation(cfg: Config, multiclass: bool = False, n_rounds: int | None = None):
    """Run the federated loop through Flower's in-process simulation engine.

    Flower's Ray backend executes each client in a separate worker process, so
    everything `client_fn` touches must be constructible *inside* that worker.
    An earlier version closed over pre-built `HospitalClient` objects holding
    torch models and DataFrames; Ray could not ship them, the workers died on
    startup, and the run silently produced a global model frozen at its
    initialisation (majority-class accuracy every round). Clients are therefore
    built on demand from the plain config -- which is also how a real
    deployment works, since each hospital constructs its own client locally.
    """
    import flwr as fl
    from flwr.common import ndarrays_to_parameters

    device = resolve_device(cfg.device)
    n_classes = len(LABEL_CLASSES) if multiclass else 2
    n_rounds = n_rounds or cfg.federated.n_rounds
    n_byz = cfg.federated.byzantine_clients
    byz_ids = set(range(cfg.federated.n_clients - n_byz, cfg.federated.n_clients))

    pooled = load_pooled(cfg)
    test_loader = make_loader(
        pooled, cfg.detector.arch, "test", batch_size=cfg.detector.batch_size,
        multiclass=multiclass, window=cfg.data.window_size,
        knn=cfg.detector.graph_knn if cfg.detector.arch == "gnn" else 0, seed=cfg.seed,
    )
    d = cfg.detector
    global_model = build_detector(
        d.arch, n_classes=n_classes, hidden=d.hidden_dim, n_layers=d.n_layers,
        dropout=d.dropout, heads=d.heads, conv="sage", max_len=cfg.data.window_size,
    ).to(device)

    history: list[dict] = []

    def server_evaluate(server_round: int, parameters, config):
        global_model.set_parameters(parameters)
        metrics, extra = evaluate(global_model, test_loader, device, multiclass)
        history.append({"round": server_round, "accuracy": metrics.accuracy,
                        "f1": metrics.f1, "fpr": metrics.fpr, "auc": extra.get("auc", 0.0)})
        if server_round > 0:
            log.info("  [flower] round %d: acc %.4f f1 %.4f",
                     server_round, metrics.accuracy, metrics.f1)
        return 1.0 - metrics.accuracy, {"accuracy": metrics.accuracy, "f1": metrics.f1}

    class _NumPyClient(fl.client.NumPyClient):
        """Constructed inside the Ray worker; owns exactly one hospital's data."""

        def __init__(self, partition_id: int) -> None:
            self.inner = HospitalClient(
                partition_id, load_hospital(cfg, partition_id), cfg,
                resolve_device(cfg.device), multiclass,
                byzantine=(partition_id in byz_ids),
            )

        def get_parameters(self, config):
            return self.inner.get_parameters()

        def fit(self, parameters, config):
            report = self.inner.fit(parameters, int(config.get("server_round", 0)))
            return report.parameters, report.n_samples, {
                "train_loss": float(report.train_loss),
                "local_accuracy": float(report.local_accuracy),
                "epsilon": float(report.epsilon),
                "byzantine": bool(report.is_byzantine),
            }

        def evaluate(self, parameters, config):
            metrics, _ = self.inner.evaluate_global(parameters, "val")
            return float(1.0 - metrics.accuracy), max(self.inner.n_samples, 1), \
                {"accuracy": metrics.accuracy, "f1": metrics.f1}

    def client_fn(context) -> fl.client.Client:
        pid = int(context.node_config.get("partition-id", 0)) % cfg.federated.n_clients
        return _NumPyClient(pid).to_client()

    strategy = build_strategy(cfg, evaluate_fn=server_evaluate)
    strategy.initial_parameters = ndarrays_to_parameters(global_model.get_parameters())

    app_client = fl.client.ClientApp(client_fn=client_fn)
    app_server = fl.server.ServerApp(
        server_fn=lambda ctx: fl.server.ServerAppComponents(
            strategy=strategy, config=fl.server.ServerConfig(num_rounds=n_rounds)
        )
    )

    log.info("Starting Flower simulation: %d clients, %d rounds, %s aggregation",
             cfg.federated.n_clients, n_rounds, cfg.federated.aggregator)
    fl.simulation.run_simulation(
        server_app=app_server, client_app=app_client,
        num_supernodes=cfg.federated.n_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0},
                        "init_args": {"log_to_driver": False, "include_dashboard": False}},
    )

    if len(history) > 1 and len({round(h["accuracy"], 6) for h in history}) == 1:
        log.error("Flower global model never changed across rounds -- the client "
                  "workers are not returning updates. Check the Ray worker logs.")
    return history, global_model
