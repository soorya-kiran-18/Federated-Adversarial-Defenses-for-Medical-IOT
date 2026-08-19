"""Shared figure styling so every result plot reads as one system.

Colour values are the validated reference categorical palette: eight hues in a
fixed order that clears the colourblind-separation and normal-vision floors on
the adjacent pairlist (the pairlist that applies to bar and line charts).

Two rules are load-bearing and are enforced by the helpers below:

* **Hues are assigned in fixed order and never cycled.** A series keeps its
  colour when other series are filtered out, so "the blue line" means the same
  thing in every figure in the report.
* **Identity is never carried by colour alone.** Every figure with two or more
  series carries a legend, and the numeric results are always written next to
  the figure as a table -- three of the light-mode slots sit below 3:1 contrast
  on a white surface, so the table is the required relief.
"""
from __future__ import annotations

from pathlib import Path

# ── categorical slots, in fixed assignment order ─────────────────────────────
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

# ── status slots, reserved: never reused as "series 5" ───────────────────────
STATUS = {"good": "#008300", "warning": "#eda100",
          "serious": "#eb6834", "critical": "#e34948"}

SURFACE_LIGHT, SURFACE_DARK = "#fcfcfb", "#1a1a19"
INK_LIGHT = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8880"}
INK_DARK = {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#8a8880"}


def palette(mode: str = "light") -> list[str]:
    return SERIES_DARK if mode == "dark" else SERIES_LIGHT


def ink(mode: str = "light") -> dict[str, str]:
    return INK_DARK if mode == "dark" else INK_LIGHT


def surface(mode: str = "light") -> str:
    return SURFACE_DARK if mode == "dark" else SURFACE_LIGHT


def apply_style(mode: str = "light") -> None:
    """Install the shared rcParams: recessive axes, thin marks, no chart junk."""
    import matplotlib as mpl

    tone, bg = ink(mode), surface(mode)
    mpl.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "axes.edgecolor": tone["muted"],
        "axes.labelcolor": tone["secondary"],
        "axes.titlecolor": tone["primary"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": tone["muted"],
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "xtick.color": tone["secondary"],
        "ytick.color": tone["secondary"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": tone["primary"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def save(fig, path: str | Path, also_dark: bool = False) -> Path:
    """Write a figure, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
