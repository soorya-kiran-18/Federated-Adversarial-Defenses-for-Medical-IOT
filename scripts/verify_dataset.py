#!/usr/bin/env python3
"""Step 3 verification -- prove the dataset is well-formed, non-IID and learnable.

Runs six checks and prints a pass/fail table:
  1. schema        -- all 30 features + metadata columns present, no NaN/inf
  2. splits        -- train/val/test disjoint, stratified, all classes in test
  3. leakage       -- scaler was fitted on train only (test stats differ)
  4. non-IID       -- per-hospital threat mix genuinely differs
  5. separability  -- a simple linear probe beats the majority-class baseline
  6. shards        -- every hospital shard is trainable (both classes present)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarmdef.config import Config
from swarmdef.data.build import SPLIT_COL, load_hospital, load_pooled, split_xy
from swarmdef.data.partition import earth_mover_skew, skew_report
from swarmdef.data.schema import FEATURE_NAMES, ID_TO_LABEL, META_COLUMNS
from swarmdef.utils.logging import banner

GREEN, RED, RESET, BOLD = "\033[38;5;42m", "\033[38;5;196m", "\033[0m", "\033[1m"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name:<34} {detail}")


def main() -> None:
    cfg = Config.from_yaml("configs/default.yaml")
    cfg.propagate_seed()
    meta = json.loads((Path(cfg.data.processed_dir) / "dataset_meta.json").read_text())
    pooled = load_pooled(cfg)
    shards = [load_hospital(cfg, h) for h in range(cfg.data.n_hospitals)]

    print(banner("STEP 3 VERIFICATION"))
    print(f"{BOLD}source={meta['source']}  rows={meta['n_rows']}  "
          f"features={meta['n_features']}  hospitals={meta['n_hospitals']}{RESET}\n")

    # 1. schema ---------------------------------------------------------------
    missing = [c for c in FEATURE_NAMES + META_COLUMNS if c not in pooled.columns]
    finite = np.isfinite(pooled[FEATURE_NAMES].to_numpy()).all()
    check("schema: 30 features + metadata", not missing and finite,
          f"missing={missing or 'none'}, all finite={finite}")

    # 2. splits ---------------------------------------------------------------
    counts = pooled[SPLIT_COL].value_counts()
    test_classes = set(pooled[pooled[SPLIT_COL] == "test"]["attack_type"].unique())
    all_classes = set(pooled["attack_type"].unique())
    check("splits: stratified, all classes in test",
          test_classes == all_classes and len(counts) == 3,
          f"train={counts.get('train',0)} val={counts.get('val',0)} "
          f"test={counts.get('test',0)}, classes in test={len(test_classes)}/{len(all_classes)}")

    # 3. leakage --------------------------------------------------------------
    # The scaler was fitted on train only, so the train split should be centred
    # near 0 while test is merely close -- identical medians would be suspicious.
    tr = pooled[pooled[SPLIT_COL] == "train"][FEATURE_NAMES].to_numpy()
    te = pooled[pooled[SPLIT_COL] == "test"][FEATURE_NAMES].to_numpy()
    train_med = float(np.abs(np.median(tr, axis=0)).mean())
    test_med = float(np.abs(np.median(te, axis=0)).mean())
    check("leakage: scaler fitted on train only",
          train_med < 0.15 and not np.allclose(np.median(tr, axis=0), np.median(te, axis=0)),
          f"|median| train={train_med:.4f} test={test_med:.4f} (train centred, test independent)")

    # 4. non-IID --------------------------------------------------------------
    tv_all = earth_mover_skew(shards)
    tv_atk = earth_mover_skew(shards, attacks_only=True)
    dominant = []
    for h, s in enumerate(shards):
        atk = s[s["label"] == 1]["attack_type"].value_counts()
        dominant.append(f"H{h}:{ID_TO_LABEL[int(atk.index[0])]}" if len(atk) else f"H{h}:-")
    check("non-IID: threat mix differs per hospital", tv_atk > 0.15,
          f"TV(attacks)={tv_atk:.3f} TV(all)={tv_all:.3f} | dominant {' '.join(dominant)}")

    # 5. separability ---------------------------------------------------------
    # A linear probe is deliberately weak: if it already beats the majority
    # baseline by a wide margin the features carry real signal, and any failure
    # of the GNN later is a modelling bug, not a broken dataset.
    from sklearn.linear_model import LogisticRegression

    Xtr, ytr, _ = split_xy(pooled, "train")
    Xte, yte, mte = split_xy(pooled, "test")
    probe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    acc = float(probe.score(Xte, yte))
    majority = float(max(np.mean(yte), 1 - np.mean(yte)))
    check("separability: linear probe > majority", acc > majority + 0.05,
          f"probe acc={acc:.4f} vs majority={majority:.4f} (+{acc - majority:.4f})")

    # 6. shards ---------------------------------------------------------------
    bad = [h for h, s in enumerate(shards)
           if s["label"].nunique() < 2 or (s[SPLIT_COL] == "train").sum() < 50]
    check("shards: every hospital is trainable", not bad,
          f"{len(shards)} shards, rows={[len(s) for s in shards]}, degenerate={bad or 'none'}")

    # ── detail tables ────────────────────────────────────────────────────────
    print(banner("PER-HOSPITAL THREAT MIX (the non-IID evidence)"))
    print(skew_report(shards).to_string())

    print(banner("PER-CLASS DETECTABILITY (linear probe, test split)"))
    pred = probe.predict(Xte)
    rows = []
    for cls_id, name in ID_TO_LABEL.items():
        m = mte == cls_id
        if not m.any():
            continue
        want = 0 if cls_id == 0 else 1
        rows.append({"class": name, "test_rows": int(m.sum()),
                     "recall": round(float(np.mean(pred[m] == want)), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    n_pass = sum(ok for _, ok, _ in results)
    print(banner(f"RESULT: {n_pass}/{len(results)} CHECKS PASSED"))
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
