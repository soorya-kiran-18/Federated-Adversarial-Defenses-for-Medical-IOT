#!/usr/bin/env python3
"""Step 5 -- federate the detector.

Trains the same detector across K simulated hospitals with FedAvg, where raw
data never leaves a hospital, and compares the result against the Step 4
centralised baseline (which had every hospital's data pooled in one place).

    python scripts/train_federated.py
    python scripts/train_federated.py --rounds 20 --set federated.aggregator=krum
    python scripts/train_federated.py --compare-local   # + isolated-hospital baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarmdef.config import Config, ensure_dirs
from swarmdef.data.build import build_dataset
from swarmdef.federated.simulation import FederatedSimulation
from swarmdef.utils.logging import banner, get_logger
from swarmdef.utils.seed import resolve_device, set_seed

log = get_logger("train_federated")


def centralised_reference(cfg: Config, multiclass: bool) -> dict | None:
    """Load the Step 4 baseline for the same task, if it has been run."""
    suffix = "multiclass" if multiclass else "binary"
    path = Path(cfg.eval.log_dir) / f"step4_baseline_{suffix}.json"
    if not path.exists():
        log.warning("No Step 4 baseline at %s -- run scripts/train_baseline.py first", path)
        return None
    results = json.loads(path.read_text())
    want = f"{cfg.detector.arch}/sage" if cfg.detector.arch == "gnn" else cfg.detector.arch
    for r in results:
        if r["arch"] == want:
            return r
    return max(results, key=lambda r: r["f1"])


def isolated_baseline(cfg: Config, device, multiclass: bool) -> pd.DataFrame:
    """Each hospital trained alone on its own shard -- the no-collaboration case.

    This is the comparison that actually justifies federation. Beating a pooled
    baseline is not the goal (pooling is illegal here); what matters is whether
    a hospital does better by joining the swarm than by training in isolation
    on the narrow slice of the threat landscape it happens to observe.
    """
    from swarmdef.data.build import load_hospital, load_pooled
    from swarmdef.detector.data import class_weights, make_loader
    from swarmdef.detector.models import build_detector
    from swarmdef.detector.train import evaluate, train_model
    from swarmdef.data.schema import LABEL_CLASSES

    n_classes = len(LABEL_CLASSES) if multiclass else 2
    d = cfg.detector
    pooled = load_pooled(cfg)
    global_test = make_loader(pooled, d.arch, "test", batch_size=d.batch_size,
                              multiclass=multiclass, window=cfg.data.window_size,
                              knn=d.graph_knn if d.arch == "gnn" else 0, seed=cfg.seed)
    rows = []
    for h in range(cfg.federated.n_clients):
        set_seed(cfg.seed)
        shard = load_hospital(cfg, h)
        tr = make_loader(shard, d.arch, "train", batch_size=d.batch_size, shuffle=True,
                         multiclass=multiclass, window=cfg.data.window_size,
                         knn=d.graph_knn if d.arch == "gnn" else 0, seed=cfg.seed + h)
        va = make_loader(shard, d.arch, "val", batch_size=d.batch_size,
                         multiclass=multiclass, window=cfg.data.window_size,
                         knn=d.graph_knn if d.arch == "gnn" else 0, seed=cfg.seed + h)
        model = build_detector(d.arch, n_classes=n_classes, hidden=d.hidden_dim,
                               n_layers=d.n_layers, dropout=d.dropout, heads=d.heads,
                               conv="sage", max_len=cfg.data.window_size).to(device)
        train_model(model, tr, va, epochs=d.central_epochs, lr=d.lr, device=device,
                    class_weight=class_weights(shard, "train", multiclass, n_classes),
                    multiclass=multiclass, early_stopping=10, verbose=False)
        m, _ = evaluate(model, global_test, device, multiclass)
        rows.append({"hospital": h, "train_samples": tr.n_samples,
                     "accuracy": round(m.accuracy, 4), "f1": round(m.f1, 4),
                     "recall": round(m.recall, 4), "fpr": round(m.fpr, 4)})
    return pd.DataFrame(rows).set_index("hospital")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--multiclass", action="store_true")
    p.add_argument("--compare-local", action="store_true",
                   help="also train each hospital in isolation (slower)")
    p.add_argument("--run-name", default=None)
    a = p.parse_args()

    cfg = Config.from_yaml(a.config) if Path(a.config).exists() else Config()
    for o in a.overrides:
        cfg.override(o)
    if a.rounds:
        cfg.federated.n_rounds = a.rounds
    cfg.propagate_seed()
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    print(banner("STEP 5 -- FEDERATED LEARNING (FedAvg)"))
    build_dataset(cfg)

    run_name = a.run_name or f"step5_{cfg.federated.aggregator}_{cfg.detector.arch}"
    sim = FederatedSimulation(cfg, device, a.multiclass, run_name)
    log.info("Clients: %s", sim.clients)
    sim.run()

    rounds, acc = sim.curve("accuracy")
    _, f1 = sim.curve("f1")
    final, best = sim.final(), sim.best("f1")

    print(banner("STEP 5 RESULTS"))
    ref = centralised_reference(cfg, a.multiclass)
    table = [
        {"setting": "federated (final round)", "accuracy": round(final.metrics.accuracy, 4),
         "f1": round(final.metrics.f1, 4), "recall": round(final.metrics.recall, 4),
         "fpr": round(final.metrics.fpr, 4)},
        {"setting": f"federated (best, round {best.round})", "accuracy": round(best.metrics.accuracy, 4),
         "f1": round(best.metrics.f1, 4), "recall": round(best.metrics.recall, 4),
         "fpr": round(best.metrics.fpr, 4)},
    ]
    if ref:
        table.append({"setting": f"centralised baseline ({ref['arch']})",
                      "accuracy": round(ref["accuracy"], 4), "f1": round(ref["f1"], 4),
                      "recall": round(ref["recall"], 4), "fpr": round(ref["fpr"], 4)})
    print(pd.DataFrame(table).to_string(index=False))

    if ref:
        gap = ref["accuracy"] - best.metrics.accuracy
        print(f"\n  privacy cost of federation: {gap:+.4f} accuracy "
              f"({'federated is behind' if gap > 0 else 'federated matches/beats'} the pooled baseline)")

    print(banner("GLOBAL MODEL EVALUATED AT EACH HOSPITAL"))
    print(sim.per_hospital_report().to_string())

    local_df = None
    if a.compare_local:
        print(banner("ISOLATED HOSPITALS (no collaboration) -- scored on the GLOBAL test set"))
        local_df = isolated_baseline(cfg, device, a.multiclass)
        print(local_df.to_string())
        print(f"\n  mean isolated accuracy : {local_df['accuracy'].mean():.4f}")
        print(f"  federated accuracy     : {best.metrics.accuracy:.4f}")
        print(f"  gain from federation   : {best.metrics.accuracy - local_df['accuracy'].mean():+.4f}")

    if cfg.eval.plot:
        from swarmdef.eval.plots import federated_vs_central
        refs = {}
        if local_df is not None:
            refs["mean isolated hospital"] = float(local_df["accuracy"].mean())
        federated_vs_central(
            rounds, acc, ref["accuracy"] if ref else max(acc),
            Path(cfg.eval.figure_dir) / f"step5_federated_vs_central_{cfg.detector.arch}.png",
            reference_lines=refs or None,
        )
        log.info("Figure -> %s", cfg.eval.figure_dir)


if __name__ == "__main__":
    main()
