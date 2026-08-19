"""Result figures for Steps 4-8.

Every function returns a matplotlib Figure and writes a PNG. Conventions held
across all of them, so the report reads as one system:

* categorical hues assigned in fixed order, never cycled;
* a legend whenever two or more series are drawn, plus direct end-labels when
  there are four or fewer -- identity is never colour-alone;
* one y-axis per figure, never two scales;
* recessive grid and axes; the data is the only prominent ink.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from swarmdef.eval.style import STATUS, apply_style, ink, palette, save


# Fixed slot per entity, so a series keeps its colour no matter how the
# results are sorted or which subset is plotted. Colour follows the entity,
# never its rank -- otherwise re-ranking silently repaints the chart.
ENTITY_SLOT: dict[str, int] = {
    "mlp": 0, "transformer": 1, "gnn/sage": 2, "gnn/gat": 3, "gnn/gcn": 6, "gnn": 2,
    "fedavg": 0, "krum": 1, "multikrum": 2, "trimmed_mean": 3, "median": 6,
}


def entity_color(name: str, cols: list[str], fallback: int = 0) -> str:
    """Colour for a named series, stable across figures and orderings."""
    key = str(name).strip().lower()
    return cols[ENTITY_SLOT.get(key, fallback) % len(cols)]


def _end_labels(ax, x, ys, names, colors, tone) -> None:
    """Direct-label the right-hand end of each line (<=4 series)."""
    if len(names) > 4:
        return
    for y, name, c in zip(ys, names, colors):
        if len(y) == 0:
            continue
        ax.annotate(name, xy=(x[len(y) - 1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=tone["secondary"], clip_on=False)


def architecture_comparison(results: list[dict], path: str | Path,
                            metric: str = "f1", mode: str = "light"):
    """Step 4: horizontal bars comparing detector architectures.

    Bars are horizontal because the category labels are words, not a scale --
    horizontal keeps them readable without rotation.
    """
    import matplotlib.pyplot as plt

    apply_style(mode)
    tone, cols = ink(mode), palette(mode)
    results = sorted(results, key=lambda r: r[metric])
    names = [r["arch"] for r in results]
    vals = [r[metric] for r in results]

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(names) + 1.9))
    ax.barh(names, vals, height=0.58,
            color=[entity_color(n, cols, i) for i, n in enumerate(names)])
    for i, v in enumerate(vals):
        ax.text(v - 0.004, i, f"{v:.4f}", va="center", ha="right",
                fontsize=9, color="#ffffff", fontweight="bold")

    lo = min(vals) - (max(vals) - min(vals) + 1e-6) * 0.7
    ax.set_xlim(max(0.0, lo), min(1.0, max(vals) + 0.004))
    ax.set_xlabel(f"test {metric}")
    ax.set_title(f"Step 4 — centralised baseline: detector architectures ({metric})")
    ax.grid(axis="y", visible=False)
    save(fig, path)
    return fig


def learning_curves(histories: dict[str, object], path: str | Path, mode: str = "light"):
    """Step 4: validation F1 per epoch, one line per architecture."""
    import matplotlib.pyplot as plt

    apply_style(mode)
    tone, cols = ink(mode), palette(mode)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    names, ys = [], []
    for i, (name, h) in enumerate(histories.items()):
        y = np.asarray(getattr(h, "val_f1", h))
        if y.size == 0:
            continue
        ax.plot(np.arange(len(y)), y, color=entity_color(name, cols, i), label=name)
        names.append(name)
        ys.append(y)

    longest = max((len(y) for y in ys), default=1)
    _end_labels(ax, np.arange(longest), ys, names, cols, tone)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation F1")
    ax.set_title("Step 4 — training convergence")
    if len(names) >= 2:
        ax.legend(loc="lower right")
    save(fig, path)
    return fig


def federated_vs_central(rounds, fed_acc, central_acc: float, path: str | Path,
                         label: str = "federated (FedAvg)", mode: str = "light",
                         extra_series: dict | None = None, ylabel: str = "test accuracy",
                         title: str = "Step 5 — federated learning vs centralised baseline",
                         reference_lines: dict | None = None):
    """Step 5: per-round federated accuracy against the pooled-data reference."""
    import matplotlib.pyplot as plt

    apply_style(mode)
    tone, cols = ink(mode), palette(mode)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))

    # The centralised baseline is a reference level, not a series: drawn as a
    # recessive dashed rule so it never competes with the curves.
    ax.axhline(central_acc, color=tone["muted"], linestyle=(0, (5, 4)), linewidth=1.4)
    ax.annotate(f"centralised baseline  {central_acc:.4f}",
                xy=(rounds[0], central_acc), xytext=(0, 6), textcoords="offset points",
                fontsize=9, color=tone["secondary"])

    # Constant references are drawn as rules, not as series: putting per-round
    # markers on a value that was measured once would imply it varies by round.
    for i, (name, level) in enumerate((reference_lines or {}).items()):
        c = cols[(i + 1) % len(cols)]
        ax.axhline(level, color=c, linestyle=(0, (3, 3)), linewidth=1.6)
        ax.annotate(f"{name}  {level:.4f}", xy=(rounds[0], level), xytext=(0, 6),
                    textcoords="offset points", fontsize=9, color=c)

    names, ys = [label], [np.asarray(fed_acc)]
    ax.plot(rounds, fed_acc, color=cols[0], marker="o", label=label)
    for i, (name, y) in enumerate((extra_series or {}).items(), start=1):
        ax.plot(rounds[: len(y)], y, color=cols[i % len(cols)], marker="o", label=name)
        names.append(name)
        ys.append(np.asarray(y))

    ax.set_xlabel("federated round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(names) >= 2:
        ax.legend(loc="lower right")
    save(fig, path)
    return fig


def adversarial_impact(rounds, clean, under_attack, after_retrain, path: str | Path,
                       mode: str = "light"):
    """Step 6: detector accuracy clean vs under GAN attack vs after retraining."""
    import matplotlib.pyplot as plt

    apply_style(mode)
    tone, cols = ink(mode), palette(mode)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))

    series = [("clean traffic", clean, cols[0]),
              ("under GAN attack", under_attack, STATUS["critical"]),
              ("after adversarial retraining", after_retrain, STATUS["good"])]
    names, ys, colors = [], [], []
    for name, y, c in series:
        if y is None:
            continue
        y = np.asarray(y, dtype=float)
        ax.plot(rounds[: len(y)], y, color=c, marker="o", label=name)
        names.append(name); ys.append(y); colors.append(c)

    ax.set_xlabel("federated round")
    ax.set_ylabel("detection accuracy")
    ax.set_title("Step 6 — adversarial degradation and recovery")
    ax.legend(loc="lower left")
    save(fig, path)
    return fig


def privacy_tradeoff(epsilons, accuracies, path: str | Path, baseline: float | None = None,
                     f1s=None, mode: str = "light"):
    """Step 7: privacy budget against utility.

    Epsilon is plotted on a log axis because privacy budgets are compared
    multiplicatively -- the gap between eps=1 and eps=2 is the same kind of step
    as between eps=8 and eps=16.
    """
    import matplotlib.pyplot as plt

    apply_style(mode)
    tone, cols = ink(mode), palette(mode)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    if baseline is not None:
        ax.axhline(baseline, color=tone["muted"], linestyle=(0, (5, 4)), linewidth=1.4)
        ax.annotate(f"no privacy  {baseline:.4f}", xy=(min(epsilons), baseline),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=9, color=tone["secondary"])

    ax.plot(epsilons, accuracies, color=cols[0], marker="o", label="accuracy")
    if f1s is not None:
        ax.plot(epsilons, f1s, color=cols[1], marker="s", label="F1")
    for x, y in zip(epsilons, accuracies):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=tone["secondary"])

    ax.set_xscale("log")
    ax.set_xlabel("privacy budget ε  (lower = stronger privacy)")
    ax.set_ylabel("test performance")
    ax.set_title("Step 7 — privacy / utility trade-off (DP-SGD)")
    if f1s is not None:
        ax.legend(loc="lower right")
    save(fig, path)
    return fig


def byzantine_comparison(results: dict[str, dict[int, float]], path: str | Path,
                         mode: str = "light", ylabel: str = "final test accuracy"):
    """Step 8: grouped bars — aggregator against number of malicious hospitals."""
    import matplotlib.pyplot as plt

    apply_style(mode)
    tone, cols = ink(mode), palette(mode)
    aggregators = list(results.keys())
    attack_counts = sorted({k for v in results.values() for k in v})

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.8 / max(len(aggregators), 1)
    x = np.arange(len(attack_counts))

    for i, agg in enumerate(aggregators):
        vals = [results[agg].get(n, np.nan) for n in attack_counts]
        pos = x + (i - (len(aggregators) - 1) / 2) * width
        # 2px surface gap between adjacent bars keeps groups legible.
        ax.bar(pos, vals, width * 0.88, label=agg, color=entity_color(agg, cols, i))
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v + 0.012, f"{v:.3f}", ha="center", fontsize=8,
                        color=tone["secondary"], rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} malicious" for n in attack_counts])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.14)
    ax.set_title("Step 8 — Byzantine robustness: aggregation rules under poisoning")
    ax.legend(loc="lower left", ncol=min(len(aggregators), 4))
    ax.grid(axis="x", visible=False)
    save(fig, path)
    return fig


def non_iid_heatmap(skew_table, path: str | Path, mode: str = "light"):
    """Step 3 evidence: per-hospital threat mix as a sequential heatmap.

    Sequential single hue, light to dark: the quantity is a magnitude (counts),
    so it takes one ramp, never a rainbow.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    apply_style(mode)
    tone = ink(mode)
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#2a78d6", "#16407a"])

    data = skew_table.to_numpy(dtype=float)
    # Row-normalise: each hospital's mix as a share, which is the comparison
    # being made (absolute shard sizes differ and would dominate otherwise).
    shares = data / np.maximum(data.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(skew_table) + 2.2))
    im = ax.imshow(shares, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(skew_table.columns)))
    ax.set_xticklabels(skew_table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(skew_table)))
    ax.set_yticklabels([f"hospital {i}" for i in skew_table.index])
    for i in range(shares.shape[0]):
        for j in range(shares.shape[1]):
            ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if shares[i, j] > 0.45 else tone["secondary"])
    ax.set_title("Step 3 — non-IID threat mix per hospital (cell = window count)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of that hospital's windows", pad=0.02)
    save(fig, path)
    return fig
