#!/usr/bin/env python3
"""Step 5 (deployment path) -- run the federated loop under the real Flower framework.

Proves the same clients, model and aggregation rules run under flwr's actual
client/server protocol, not only in our own simulation harness.

    python scripts/train_federated_flower.py --rounds 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarmdef.config import Config, ensure_dirs
from swarmdef.data.build import build_dataset
from swarmdef.federated.flower_app import run_flower_simulation
from swarmdef.utils.logging import banner, get_logger
from swarmdef.utils.seed import set_seed

log = get_logger("flower")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--multiclass", action="store_true")
    a = p.parse_args()

    cfg = Config.from_yaml(a.config) if Path(a.config).exists() else Config()
    for o in a.overrides:
        cfg.override(o)
    cfg.propagate_seed()
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    print(banner("STEP 5 (DEPLOYMENT PATH) -- FEDERATED LEARNING VIA FLOWER"))
    build_dataset(cfg)
    history, _ = run_flower_simulation(cfg, a.multiclass, a.rounds)

    df = pd.DataFrame([h for h in history if h["round"] > 0])
    print(banner("FLOWER RUN -- PER-ROUND GLOBAL MODEL"))
    print(df.round(4).to_string(index=False))
    out = Path(cfg.eval.log_dir) / "step5_flower.csv"
    df.to_csv(out, index=False)
    log.info("Results -> %s", out)
    if len(df):
        best = df.loc[df["f1"].idxmax()]
        print(banner(f"FLOWER BEST: round {int(best['round'])}  "
                     f"acc={best['accuracy']:.4f}  f1={best['f1']:.4f}"))


if __name__ == "__main__":
    main()
