"""Aggregation rules for the central server (Steps 5 and 8).

Every rule maps a list of client updates to one global update:

    aggregate(updates, weights, **kw) -> list[np.ndarray]

`fedavg` is the standard baseline and is what Step 5 uses. The remaining rules
are Byzantine-robust and are what Step 8 compares against it.

The central insight, and the reason plain averaging fails
---------------------------------------------------------
FedAvg computes a *mean*, and a mean has an asymptotic breakdown point of zero:
one client sending sufficiently large values can move the average anywhere it
likes, no matter how many honest clients there are. Every robust rule below
replaces the mean with an estimator that ignores extreme values -- either by
selecting a representative update (Krum) or by discarding the tails coordinate
by coordinate (trimmed mean, median).
"""
from __future__ import annotations

import numpy as np

from swarmdef.utils.logging import get_logger

log = get_logger("federated.aggregate")

Update = list[np.ndarray]


# ── helpers ──────────────────────────────────────────────────────────────────
def flatten(update: Update) -> np.ndarray:
    """Concatenate a parameter list into one vector (for distance maths)."""
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in update])


def unflatten(vec: np.ndarray, template: Update) -> Update:
    """Inverse of `flatten`, using `template` for shapes and dtypes."""
    out, i = [], 0
    for a in template:
        a = np.asarray(a)
        n = a.size
        out.append(vec[i:i + n].reshape(a.shape).astype(a.dtype, copy=False))
        i += n
    return out


def pairwise_sq_distances(updates: list[Update]) -> np.ndarray:
    """Squared Euclidean distances between every pair of flattened updates."""
    flat = np.stack([flatten(u) for u in updates])
    sq = np.sum(flat ** 2, axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T)
    return np.maximum(d, 0.0)


def _normalised_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    return np.full(n, 1.0 / n) if total <= 0 else w / total


# ── the rules ────────────────────────────────────────────────────────────────
def fedavg(updates: list[Update], weights=None, **_) -> Update:
    """Sample-weighted mean of client updates (McMahan et al., 2017).

    Optimal when every client is honest; provides no protection when one is not.
    """
    w = _normalised_weights(weights, len(updates))
    return [
        sum(w[i] * np.asarray(updates[i][j], dtype=np.float64) for i in range(len(updates)))
        .astype(updates[0][j].dtype, copy=False)
        for j in range(len(updates[0]))
    ]


def krum(updates: list[Update], weights=None, f: int = 1, multi: int = 1, **_) -> Update:
    """Krum / Multi-Krum (Blanchard et al., 2017).

    Scores each client by the sum of squared distances to its n-f-2 nearest
    neighbours, then keeps the `multi` lowest-scoring updates and averages them.

    The intuition: honest updates cluster together because they are estimates of
    the same gradient, so an honest client has many close neighbours. A poisoned
    update must sit far from that cluster to change the global model, and that
    distance is exactly what the score measures. An attacker close enough to
    score well is, by construction, too close to do damage.

    Requires n > 2f + 2 to guarantee that the neighbourhood of an honest client
    contains only honest clients.
    """
    n = len(updates)
    if n == 1:
        return [np.array(a) for a in updates[0]]

    n_neighbours = max(n - f - 2, 1)
    d = pairwise_sq_distances(updates)
    np.fill_diagonal(d, np.inf)
    scores = np.array([np.sort(d[i])[:n_neighbours].sum() for i in range(n)])

    multi = max(1, min(multi, n - f if n > f else 1))
    chosen = np.argsort(scores)[:multi]
    log.debug("Krum scores=%s -> selected %s", np.round(scores, 3), chosen.tolist())

    selected = [updates[i] for i in chosen]
    sel_w = None if weights is None else [weights[i] for i in chosen]
    return fedavg(selected, sel_w)


def multikrum(updates: list[Update], weights=None, f: int = 1, **kw) -> Update:
    """Multi-Krum: average the n-f best-scoring updates instead of just one.

    Retains Krum's filtering while recovering most of the variance reduction of
    averaging, which single-Krum throws away by keeping exactly one client.
    """
    n = len(updates)
    return krum(updates, weights, f=f, multi=max(1, n - f))


def trimmed_mean(updates: list[Update], weights=None, trim_ratio: float = 0.25, **_) -> Update:
    """Coordinate-wise trimmed mean (Yin et al., 2018).

    For each parameter independently, discard the highest and lowest
    `trim_ratio` of client values and average the rest. Robust as long as the
    trimmed fraction exceeds the fraction of malicious clients, and unlike Krum
    it never discards an entire honest client -- only its outlying coordinates.
    """
    n = len(updates)
    k = int(np.floor(trim_ratio * n))
    if n <= 2 * k or k == 0:
        # Too few clients to trim both tails: fall back to the median, which is
        # the maximally robust coordinate-wise estimator.
        return coordinate_median(updates)

    out = []
    for j in range(len(updates[0])):
        stack = np.stack([np.asarray(u[j], dtype=np.float64) for u in updates])
        ordered = np.sort(stack, axis=0)
        out.append(ordered[k:n - k].mean(axis=0).astype(updates[0][j].dtype, copy=False))
    return out


def coordinate_median(updates: list[Update], weights=None, **_) -> Update:
    """Coordinate-wise median.

    The maximally robust of these rules -- its breakdown point is 50% -- at the
    cost of discarding magnitude information from every honest client.
    """
    return [
        np.median(np.stack([np.asarray(u[j], dtype=np.float64) for u in updates]), axis=0)
        .astype(updates[0][j].dtype, copy=False)
        for j in range(len(updates[0]))
    ]


AGGREGATORS = {
    "fedavg": fedavg,
    "krum": krum,
    "multikrum": multikrum,
    "trimmed_mean": trimmed_mean,
    "median": coordinate_median,
}


def get_aggregator(name: str):
    if name not in AGGREGATORS:
        raise KeyError(f"unknown aggregator {name!r}; choose from {sorted(AGGREGATORS)}")
    return AGGREGATORS[name]


def aggregate(name: str, updates: list[Update], weights=None, **kw) -> Update:
    """Dispatch to a named rule, passing only the kwargs it accepts."""
    fn = get_aggregator(name)
    allowed = {
        "fedavg": set(), "krum": {"f", "multi"}, "multikrum": {"f"},
        "trimmed_mean": {"trim_ratio"}, "median": set(),
    }[name]
    return fn(updates, weights, **{k: v for k, v in kw.items() if k in allowed})
