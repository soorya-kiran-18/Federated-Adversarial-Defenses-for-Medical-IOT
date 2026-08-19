"""Dataset builder: raw source -> normalised, split, non-IID hospital shards.

This is the Step 3 deliverable. One call produces everything the federated
experiments consume:

    data/hospitals/hospital_0.csv ... hospital_{K-1}.csv   (train/val/test tagged)
    data/processed/pooled.csv                              (centralised baseline)
    data/processed/scaler.json                             (frozen normalisation)
    data/processed/dataset_meta.json                       (provenance + skew stats)

Split discipline
----------------
The split is made *before* normalisation is fitted and before sharding, and it
is stratified on the 7-way attack class. Test rows never touch the scaler, so
the reported numbers carry no leakage. Every hospital holds its own slice of
all three splits, which is what lets a client evaluate the global model on data
that hospital never trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from swarmdef.config import Config
from swarmdef.data import cic
from swarmdef.data.partition import earth_mover_skew, partition, skew_report
from swarmdef.data.scaler import FlowScaler
from swarmdef.data.schema import ALL_COLUMNS, FEATURE_NAMES, ID_TO_LABEL
from swarmdef.utils.logging import get_logger

log = get_logger("data.build")

SPLIT_COL = "split"


# ── source resolution ────────────────────────────────────────────────────────
def resolve_source(cfg: Config) -> str:
    """Decide which raw source to use, honouring `data.source`."""
    want = cfg.data.source
    if want == "auto":
        if cic.available(cfg.data.raw_dir):
            log.info("CIC-IoT2023 detected -> using it as the data source")
            return "cic"
        log.info("No CIC-IoT2023 CSVs under %s -> falling back to the digital twin",
                 cfg.data.raw_dir)
        return "twin"
    return want


def load_raw(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Produce a raw, unnormalised frame in the project schema."""
    source = resolve_source(cfg)
    if source == "cic":
        return cic.load(cfg.data.raw_dir, cfg.data.n_samples, cfg.seed), "cic_iot2023"

    from swarmdef.twin.runner import capture_offline

    result = capture_offline(
        n_hospitals=cfg.data.n_hospitals,
        devices_per_hospital=cfg.twin.devices_per_hospital,
        duration_s=cfg.data.twin_duration_s,
        publish_hz=cfg.twin.publish_hz,
        window_s=1.0,
        attacks_per_device=cfg.data.twin_attacks_per_device,
        attack_fraction=cfg.data.twin_attack_fraction,
        events_per_device=cfg.data.twin_events_per_device,
        non_iid=True,
        seed=cfg.seed,
    )
    log.info("Digital twin capture: %s", result.summary())
    return result.frame, "digital_twin"


# ── splitting ────────────────────────────────────────────────────────────────
def stratified_group_split(
    df: pd.DataFrame, test_fraction: float, val_fraction: float, seed: int
) -> pd.DataFrame:
    """Split at the *graph* level: a whole (hospital, time-window) goes to one split.

    Why not split individual rows
    -----------------------------
    The GNN classifies device-windows as nodes of a per-hospital graph, so
    neighbouring devices in the same time window exchange messages. If those
    neighbours straddled a split boundary, a test node's features would reach
    the model during training through message passing -- a transductive setup
    that quietly inflates the reported score.

    Splitting whole time-windows makes the evaluation *inductive*: the model is
    tested on graphs it has never seen any part of. It is also the honest
    setting for an IDS, which is deployed against future traffic.

    Groups are stratified by their dominant attack class so rare families
    (Mirai, Recon) still appear in every split.
    """
    rng = np.random.default_rng(seed)
    keys = df.groupby(["hospital_id", "t_start"], sort=True).ngroup()
    df = df.assign(_group=keys)

    # Stratify each group by the rarest attack class it contains: a window that
    # holds the only Mirai sample must not be lumped in with the benign bulk.
    class_freq = df["attack_type"].value_counts()
    def group_stratum(g: pd.Series) -> int:
        attacks = [c for c in g.unique() if c != 0]
        if not attacks:
            return 0
        return int(min(attacks, key=lambda c: class_freq.get(c, 0)))

    strata = df.groupby("_group")["attack_type"].apply(group_stratum)

    split_of_group: dict[int, str] = {}
    for _, members in strata.groupby(strata):
        gids = np.array(members.index.to_numpy(), copy=True)
        rng.shuffle(gids)
        n = len(gids)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        if n >= 3:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        for gid in gids[:n_test]:
            split_of_group[int(gid)] = "test"
        for gid in gids[n_test:n_test + n_val]:
            split_of_group[int(gid)] = "val"
        for gid in gids[n_test + n_val:]:
            split_of_group[int(gid)] = "train"

    out = df.copy()
    out[SPLIT_COL] = out["_group"].map(split_of_group)
    return out.drop(columns=["_group"])


def stratified_split(
    df: pd.DataFrame, test_fraction: float, val_fraction: float, seed: int
) -> pd.DataFrame:
    """Tag every row train/val/test, stratified on the 7-way attack class."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)

    for _, group in df.groupby("attack_type"):
        # pandas >=3.0 hands back a read-only view; copy before shuffling.
        idx = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        # Guarantee at least one row per split for very rare classes, so the
        # test set never silently loses an entire attack family.
        if n >= 3:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        split[df.index.get_indexer(idx[:n_test])] = "test"
        split[df.index.get_indexer(idx[n_test:n_test + n_val])] = "val"
        split[df.index.get_indexer(idx[n_test + n_val:])] = "train"

    out = df.copy()
    out[SPLIT_COL] = split
    return out


# ── main entry point ─────────────────────────────────────────────────────────
def build_dataset(cfg: Config, force: bool = False) -> dict:
    """Build (or reuse) the full federated dataset. Returns a metadata dict."""
    hospital_dir = Path(cfg.data.hospital_dir)
    processed_dir = Path(cfg.data.processed_dir)
    hospital_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    meta_path = processed_dir / "dataset_meta.json"

    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("n_hospitals") == cfg.data.n_hospitals and meta.get("seed") == cfg.seed:
            log.info("Reusing existing dataset (%d rows, source=%s). Use force=True to rebuild.",
                     meta["n_rows"], meta["source"])
            return meta

    # 1. Raw source ----------------------------------------------------------
    raw, source = load_raw(cfg)
    raw = raw[ALL_COLUMNS].reset_index(drop=True)
    if cfg.data.n_samples and len(raw) > cfg.data.n_samples:
        rng = np.random.default_rng(cfg.seed)
        keep = np.sort(rng.choice(len(raw), cfg.data.n_samples, replace=False))
        raw = raw.iloc[keep].reset_index(drop=True)
    log.info("Raw dataset: %d rows, %.1f%% attack, %d classes",
             len(raw), 100 * raw["label"].mean(), raw["attack_type"].nunique())

    # 2. Split BEFORE fitting the scaler (no leakage) -------------------------
    # Group-level by default so the GNN's evaluation is inductive; every model
    # (MLP, Transformer, GNN) then trains and tests on identical splits, which
    # is what makes the Step 4 architecture comparison fair.
    splitter = (stratified_group_split if cfg.data.split_mode == "group"
                else stratified_split)
    tagged = splitter(raw, cfg.data.test_fraction, cfg.data.val_fraction, cfg.seed)
    counts = tagged[SPLIT_COL].value_counts().to_dict()
    log.info("Split: train=%d val=%d test=%d",
             counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))

    # 3. Fit normalisation on the TRAIN split only ---------------------------
    train_mask = tagged[SPLIT_COL] == "train"
    scaler = FlowScaler().fit(tagged.loc[train_mask, FEATURE_NAMES].to_numpy())
    scaled = tagged.copy()
    scaled[FEATURE_NAMES] = scaler.transform(tagged[FEATURE_NAMES].to_numpy())
    scaler_path = scaler.save(processed_dir / "scaler.json")
    log.info("Fitted FlowScaler on %d train rows -> %s", int(train_mask.sum()), scaler_path.name)

    # 4. Non-IID partition into hospital shards ------------------------------
    shards = partition(
        scaled, cfg.data.n_hospitals,
        method=cfg.data.partition, alpha=cfg.data.dirichlet_alpha, seed=cfg.seed,
    )
    report = skew_report(shards)
    skew = earth_mover_skew(shards)
    skew_attacks = earth_mover_skew(shards, attacks_only=True)
    log.info("Non-IID partition (%s): TV-distance from pooled = %.3f overall, "
             "%.3f across attack classes only",
             cfg.data.partition, skew, skew_attacks)
    log.info("Per-hospital class distribution:\n%s", report.to_string())

    # 5. Persist -------------------------------------------------------------
    pooled_path = processed_dir / "pooled.csv"
    scaled.to_csv(pooled_path, index=False)
    shard_paths = []
    for h, shard in enumerate(shards):
        path = hospital_dir / f"hospital_{h}.csv"
        shard.to_csv(path, index=False)
        shard_paths.append(str(path))
        log.info("  hospital_%d.csv: %5d rows (%4.1f%% attack, %d classes present)",
                 h, len(shard), 100 * shard["label"].mean() if len(shard) else 0,
                 shard["attack_type"].nunique())

    meta = {
        "source": source,
        "n_rows": int(len(scaled)),
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "n_hospitals": cfg.data.n_hospitals,
        "partition": cfg.data.partition,
        "dirichlet_alpha": cfg.data.dirichlet_alpha,
        "seed": cfg.seed,
        "attack_fraction": float(scaled["label"].mean()),
        "class_counts": {ID_TO_LABEL[int(k)]: int(v)
                         for k, v in scaled["attack_type"].value_counts().items()},
        "split_counts": {k: int(v) for k, v in counts.items()},
        "split_mode": cfg.data.split_mode,
        "non_iid_tv_distance": skew,
        "non_iid_tv_distance_attacks_only": skew_attacks,
        "skew_table": json.loads(report.reset_index().to_json(orient="records")),
        "pooled_path": str(pooled_path),
        "scaler_path": str(scaler_path),
        "hospital_paths": shard_paths,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Dataset metadata -> %s", meta_path)
    return meta


# ── loading helpers used by the training code ────────────────────────────────
def load_hospital(cfg: Config, hospital_id: int) -> pd.DataFrame:
    path = Path(cfg.data.hospital_dir) / f"hospital_{hospital_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run build_dataset(cfg) first")
    return pd.read_csv(path)


def load_pooled(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.data.processed_dir) / "pooled.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run build_dataset(cfg) first")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, split: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, binary label, multi-class label) for one split."""
    sub = df if split is None else df[df[SPLIT_COL] == split]
    X = sub[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = sub["label"].to_numpy(dtype=np.int64)
    y_multi = sub["attack_type"].to_numpy(dtype=np.int64)
    return X, y, y_multi
