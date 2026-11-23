"""Figure style of the paper: fonts, colors and layout helpers, in absolute inches."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 372 / 72.27  # text width of the Springer Nature template, inches

OI = {"black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7"}
ARM_STYLE = {
    "BASE": {"color": OI["black"], "marker": "o", "linestyle": "-"},
    "IER": {"color": OI["blue"], "marker": "^", "linestyle": "-"},
    "IER3": {"color": OI["vermillion"], "marker": "s", "linestyle": "-."},
    "IER_REL": {"color": OI["sky_blue"], "marker": "v", "linestyle": "--"},
    "IER3_REL": {"color": OI["orange"], "marker": "D", "linestyle": "--"},
    "DUP": {"color": OI["bluish_green"], "marker": "P", "linestyle": ":"},
}
TASK_STYLE = {
    "HalfCheetah-v4": {"color": OI["orange"], "marker": "o"},
    "Hopper-v4": {"color": OI["sky_blue"], "marker": "^"},
    "walker-walk": {"color": OI["bluish_green"], "marker": "s"},
    "finger-turn_hard": {"color": OI["reddish_purple"], "marker": "D"},
}
# Conditions (fig_stress): more noise is a darker blue, the dynamics change stands apart.
COND_STYLE = {"default": OI["black"], "noise5": OI["sky_blue"], "noise10": OI["blue"], "dyn20": OI["vermillion"]}
GRAY = "#9a9a9a"
GRID = "#e3e3e3"
LW = 1.2
BAND_ALPHA = 0.2

# ---------------------------------------------------------------- fonts and rcParams

LM_DIRS = [ROOT / "paper_assets" / "fonts", Path("/usr/share/texmf/fonts/opentype/public/lm"),
           Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm")]


def apply_style() -> None:
    # Latin Modern Roman in its 8 pt optical design (what lmodern uses for 8 pt text), named explicitly so the choice
    # does not depend on which optical sizes the font cache lists first.
    for name in ("lmroman8-regular", "lmroman8-italic", "lmroman8-bold"):
        for folder in LM_DIRS:
            path = folder / f"{name}.otf"
            if path.exists():
                fm.fontManager.addfont(str(path))
                break
    family = "LM Roman 8"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [family], "mathtext.fontset": "cm",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.5,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": LW, "lines.markersize": 3.5,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.0, "legend.borderpad": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
        "axes.unicode_minus": True, "axes.formatter.use_mathtext": False,
    })


# ---------------------------------------------------------------- layout helpers (absolute inches)

MARGIN = {"left": 0.50, "right": 0.14, "bottom": 0.34, "top_legend": 0.40, "top": 0.20}
GAP = 0.36  # horizontal gap between panels in a strip, inches


def strip(height: float, ratios, legend: bool = True, gaps=None, left=None, right=None, bottom=None, top=None):
    """A full-width strip of panels with fixed margins in inches, so every figure shares them."""
    fig = plt.figure(figsize=(WIDTH, height))
    left = MARGIN["left"] if left is None else left
    right = MARGIN["right"] if right is None else right
    bottom = MARGIN["bottom"] if bottom is None else bottom
    top = (MARGIN["top_legend"] if legend else MARGIN["top"]) if top is None else top
    n = len(ratios)
    gaps = [GAP] * (n - 1) if gaps is None else list(gaps)
    usable = WIDTH - left - right - sum(gaps)
    widths = [usable * r / sum(ratios) for r in ratios]
    axes, x = [], left
    for i, w in enumerate(widths):
        ax = fig.add_axes([x / WIDTH, bottom / height, w / WIDTH, (height - bottom - top) / height])
        axes.append(ax)
        x += w + (gaps[i] if i < n - 1 else 0)
    return fig, axes


def grid(rows: int, cols: int, height: float, legend: bool = True, wgap: float = 0.50, hgap: float = 0.46,
         left=None, right=None, bottom=None, top=None):
    """A full-width grid of equal panels with the same fixed margins in inches as strip(), filled row by row."""
    fig = plt.figure(figsize=(WIDTH, height))
    left = MARGIN["left"] if left is None else left
    right = MARGIN["right"] if right is None else right
    bottom = MARGIN["bottom"] if bottom is None else bottom
    top = (MARGIN["top_legend"] if legend else MARGIN["top"]) if top is None else top
    pw = (WIDTH - left - right - (cols - 1) * wgap) / cols
    ph = (height - bottom - top - (rows - 1) * hgap) / rows
    axes = []
    for r in range(rows):
        for c in range(cols):
            x = left + c * (pw + wgap)
            y = height - top - (r + 1) * ph - r * hgap
            axes.append(fig.add_axes([x / WIDTH, y / height, pw / WIDTH, ph / height]))
    return fig, axes


def panel_label(ax, letter: str, text: str = "", dx: float = 0.0) -> None:
    """'(a) Task' just above the top-left corner of the axes, left-aligned with the y tick labels."""
    label = f"({letter})" + (f" {text}" if text else "")
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-14 + dx, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=8)


def top_legend(fig, handles, labels, ncol=None, y=None, height=None, fontsize=7):
    height = fig.get_figheight() if height is None else height
    y = 1 - 0.05 / height if y is None else y
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5 + (MARGIN["left"] - MARGIN["right"]) / 2 / WIDTH, y),
                      ncol=ncol or len(handles), frameon=False, fontsize=fontsize, handlelength=1.8,
                      columnspacing=1.2, handletextpad=0.4)


def steps_formatter(x, _=None) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    return f"{x / 1e3:g}k"
