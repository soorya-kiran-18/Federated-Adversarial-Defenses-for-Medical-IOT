#!/usr/bin/env python3
"""Step 3 -- build the federated dataset (hospital shards + pooled baseline set).

    python scripts/build_dataset.py                       # default config
    python scripts/build_dataset.py --set data.n_hospitals=6 --force
    python scripts/build_dataset.py --set data.partition=dirichlet --set data.dirichlet_alpha=0.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarmdef.config import Config, ensure_dirs
from swarmdef.data.build import build_dataset
from swarmdef.utils.logging import banner, get_logger
from swarmdef.utils.seed import set_seed

log = get_logger("build_dataset")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="dotted override, e.g. --set data.n_samples=50000")
    p.add_argument("--force", action="store_true", help="rebuild even if a dataset exists")
    a = p.parse_args()

    cfg = Config.from_yaml(a.config) if Path(a.config).exists() else Config()
    for o in a.overrides:
        cfg.override(o)
    cfg.propagate_seed()
    ensure_dirs(cfg)
    set_seed(cfg.seed)

    print(banner("STEP 3 -- FEDERATED DATASET CONSTRUCTION"))
    meta = build_dataset(cfg, force=a.force)
    print(banner("SUMMARY"))
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("skew_table", "features", "hospital_paths")}, indent=2))


if __name__ == "__main__":
    main()
