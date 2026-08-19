"""Non-IID partitioning of the pooled dataset into hospital shards (Step 3).

Federated learning is only interesting when clients disagree. If every hospital
saw an identical sample of the threat landscape, FedAvg would converge to the
centralised solution immediately and there would be nothing to study. Real
hospitals differ: a paediatric ward's device mix, patient load and exposure are
nothing like a regional trauma centre's.

Two partitioners are provided:

    native    -- keep the digital twin's own hospital assignment. Skew arises
                 organically from each hospital's device fleet and its attack
                 schedule, so the shards differ in *device population* as well
                 as label mix. This is the more realistic setting.
    dirichlet -- the standard synthetic benchmark. For each attack class, draw
                 a Dirichlet(alpha) vector over hospitals and split that class's
                 rows accordingly. Lower alpha => sharper skew. Required for
                 CIC-IoT2023, which carries no hospital structure of its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swarmdef.data.schema import ID_TO_LABEL
from swarmdef.utils.logging import get_logger

log = get_logger("data.partition")


def dirichlet_partition(
    df: pd.DataFrame,
    n_hospitals: int,
    alpha: float = 0.4,
    seed: int = 42,
    min_rows: int = 32,
) -> list[pd.DataFrame]:
    """Split `df` into `n_hospitals` shards with a Dirichlet label skew."""
    rng = np.random.default_rng(seed)
    shards: list[list[np.ndarray]] = [[] for _ in range(n_hospitals)]

    for cls, group in df.groupby("attack_type"):
        idx = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * n_hospitals)
        # Cut points along the shuffled class indices.
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for h, part in enumerate(np.split(idx, cuts)):
            if len(part):
                shards[h].append(part)

    out: list[pd.DataFrame] = []
    for h, parts in enumerate(shards):
        index = np.concatenate(parts) if parts else np.array([], dtype=int)
        shard = df.loc[index].copy()
        shard["hospital_id"] = h
        out.append(shard.sort_index().reset_index(drop=True))

    # A shard with no data (or only one class) would break local training, so
    # top it up from the largest shard rather than silently producing a client
    # that cannot compute a gradient.
    for h, shard in enumerate(out):
        if len(shard) >= min_rows and shard["label"].nunique() > 1:
            continue
        donor = int(np.argmax([len(s) for s in out]))
        if donor == h:
            continue
        need = max(min_rows - len(shard), 0) + 16
        take = out[donor].sample(min(need, len(out[donor]) // 3), random_state=seed)
        out[donor] = out[donor].drop(take.index)
        take = take.copy()
        take["hospital_id"] = h
        out[h] = pd.concat([shard, take], ignore_index=True)
        log.warning("Shard %d was degenerate; topped up with %d rows from shard %d",
                    h, len(take), donor)
    return out


def native_partition(df: pd.DataFrame, n_hospitals: int) -> list[pd.DataFrame]:
    """Use the hospital_id the digital twin already assigned."""
    return [
        df[df["hospital_id"] == h].reset_index(drop=True)
        for h in range(n_hospitals)
    ]


def partition(
    df: pd.DataFrame,
    n_hospitals: int,
    method: str = "native",
    alpha: float = 0.4,
    seed: int = 42,
) -> list[pd.DataFrame]:
    if method == "native" and df["hospital_id"].nunique() >= n_hospitals:
        shards = native_partition(df, n_hospitals)
    else:
        if method == "native":
            log.info("Source has no usable hospital structure; using Dirichlet(alpha=%.2f)", alpha)
        shards = dirichlet_partition(df, n_hospitals, alpha, seed)
    return shards


# ── diagnostics ──────────────────────────────────────────────────────────────
def skew_report(shards: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-hospital class counts -- the table that proves the split is non-IID."""
    rows = []
    for h, shard in enumerate(shards):
        counts = shard["attack_type"].value_counts()
        row = {"hospital": h, "rows": len(shard),
               "attack_%": round(100 * shard["label"].mean(), 1) if len(shard) else 0.0}
        for cls_id, name in ID_TO_LABEL.items():
            row[name] = int(counts.get(cls_id, 0))
        rows.append(row)
    return pd.DataFrame(rows).set_index("hospital")


def earth_mover_skew(
    shards: list[pd.DataFrame], n_classes: int = 7, attacks_only: bool = False
) -> float:
    """Mean total-variation distance between each shard's class distribution
    and the pooled distribution. 0 = perfectly IID, 1 = maximally skewed.

    This single number is what makes "our split is non-IID" a measured claim
    rather than an assertion.

    With `attacks_only`, the benign class is excluded. That is the more
    informative figure here: benign traffic is ~70% of every shard, so leaving
    it in dilutes the very skew we are trying to quantify. The attack-only
    distance answers the question that actually matters -- does each hospital
    see a different *threat mix*?
    """
    start = 1 if attacks_only else 0
    pooled = np.zeros(n_classes)
    per_shard = []
    for shard in shards:
        counts = np.zeros(n_classes)
        if len(shard):
            vc = shard["attack_type"].value_counts()
            for cls_id, n in vc.items():
                counts[int(cls_id)] = n
        pooled += counts
        per_shard.append(counts)

    pooled_p = pooled[start:] / max(pooled[start:].sum(), 1)
    tvs = []
    for counts in per_shard:
        sub = counts[start:]
        if sub.sum() == 0:
            continue
        tvs.append(0.5 * np.abs(sub / sub.sum() - pooled_p).sum())
    return float(np.mean(tvs)) if tvs else 0.0
