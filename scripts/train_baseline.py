#!/usr/bin/env python3
"""Step 4 -- centralised baseline detector.

Trains each architecture on the POOLED dataset (all hospitals' data in one
place). This is the number federated learning must live up to: it is what a
hospital consortium could achieve if privacy law allowed them to pool raw
telemetry. Every later step is reported against it.

    python scripts/train_baseline.py                     # all architectures
    python scripts/train_baseline.py --arch gnn          # just the GNN
    python scripts/train_baseline.py --multiclass        # 7-way instead of binary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarmdef.config import Config, ensure_dirs
from swarmdef.data.build import build_dataset, load_pooled
from swarmdef.data.schema import ID_TO_LABEL, LABEL_CLASSES
from swarmdef.detector.data import class_weights, make_loader
from swarmdef.detector.models import build_detector
from swarmdef.detector.train import evaluate, train_model
from swarmdef.utils.logging import banner, get_logger
from swarmdef.utils.seed import resolve_device, set_seed

log = get_logger("train_baseline")


def run_one(arch: str, df: pd.DataFrame, cfg: Config, device, multiclass: bool,
            conv: str = "sage") -> dict:
    set_seed(cfg.seed)
    n_classes = len(LABEL_CLASSES) if multiclass else 2
    d = cfg.detector

    loaders = {
        s: make_loader(df, arch, s, batch_size=d.batch_size, shuffle=(s == "train"),
                       multiclass=multiclass, window=cfg.data.window_size,
                       knn=d.graph_knn if arch == "gnn" else 0, seed=cfg.seed)
        for s in ("train", "val", "test")
    }

    model = build_detector(arch, n_classes=n_classes, hidden=d.hidden_dim,
                           n_layers=d.n_layers, dropout=d.dropout, heads=d.heads,
                           conv=conv, max_len=cfg.data.window_size)
    label = f"{arch}/{conv}" if arch == "gnn" else arch
    log.info("%s: %d parameters, %d train samples",
             label, model.n_parameters(), loaders["train"].n_samples)

    t0 = time.time()
    history = train_model(
        model, loaders["train"], loaders["val"],
        epochs=d.central_epochs, lr=d.lr, weight_decay=d.weight_decay, device=device,
        class_weight=class_weights(df, "train", multiclass, n_classes),
        multiclass=multiclass, early_stopping=10,
    )
    train_s = time.time() - t0

    metrics, extra = evaluate(model, loaders["test"], device, multiclass)
    log.info("%s TEST: %s auc=%.4f", label, metrics, extra.get("auc", float("nan")))

    out = {
        "arch": label, "params": model.n_parameters(), "train_seconds": round(train_s, 1),
        "_history": history,
        "epochs_run": len(history.train_loss), "best_epoch": history.best_epoch(),
        **metrics.as_dict(), **{k: v for k, v in extra.items() if k != "per_class_recall"},
    }
    if multiclass and "per_class_recall" in extra:
        out["per_class_recall"] = {ID_TO_LABEL[c]: round(r, 4)
                                   for c, r in extra["per_class_recall"].items()}

    if cfg.eval.save_models:
        path = Path(cfg.eval.model_dir) / f"baseline_{label.replace('/', '_')}.pt"
        torch.save({"state_dict": model.state_dict(), "arch": arch, "conv": conv,
                    "n_classes": n_classes, "config": cfg.to_dict()}, path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--arch", default="all", choices=["all", "mlp", "transformer", "gnn"])
    p.add_argument("--multiclass", action="store_true", help="7-way attack typing")
    a = p.parse_args()

    cfg = Config.from_yaml(a.config) if Path(a.config).exists() else Config()
    for o in a.overrides:
        cfg.override(o)
    cfg.propagate_seed()
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    print(banner("STEP 4 -- CENTRALISED BASELINE DETECTOR"))
    build_dataset(cfg)
    df = load_pooled(cfg)
    log.info("Pooled dataset: %d rows | device=%s | task=%s",
             len(df), device, "7-way" if a.multiclass else "binary")

    jobs = ([("mlp", "sage"), ("transformer", "sage"), ("gnn", "sage"), ("gnn", "gat")]
            if a.arch == "all" else
            [(a.arch, "sage")] + ([("gnn", "gat")] if a.arch == "gnn" else []))

    results = []
    for arch, conv in jobs:
        print(banner(f"TRAINING {arch.upper()}" + (f" ({conv})" if arch == "gnn" else "")))
        results.append(run_one(arch, df, cfg, device, a.multiclass, conv))

    print(banner("STEP 4 RESULTS -- CENTRALISED BASELINE (test split)"))
    histories = {r["arch"]: r.pop("_history") for r in results}
    table = pd.DataFrame(results)
    cols = ["arch", "params", "accuracy", "f1", "precision", "recall", "fpr", "auc",
            "evasion_rate", "train_seconds"]
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))

    if a.multiclass and "per_class_recall" in table.columns:
        print(banner("PER-CLASS RECALL"))
        print(pd.DataFrame([r["per_class_recall"] for r in results],
                           index=[r["arch"] for r in results]).to_string())

    suffix = "multiclass" if a.multiclass else "binary"
    if cfg.eval.plot:
        from swarmdef.eval.plots import architecture_comparison, learning_curves
        fig_dir = Path(cfg.eval.figure_dir)
        for metric in ("f1", "accuracy"):
            architecture_comparison(results, fig_dir / f"step4_architectures_{metric}_{suffix}.png", metric)
        learning_curves(histories, fig_dir / f"step4_learning_curves_{suffix}.png")
        log.info("Figures -> %s", fig_dir)

    out = Path(cfg.eval.log_dir) / f"step4_baseline_{suffix}.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Results -> %s", out)

    best = max(results, key=lambda r: r["f1"])
    print(banner(f"BEST: {best['arch']}  acc={best['accuracy']:.4f}  f1={best['f1']:.4f}"))


if __name__ == "__main__":
    main()
