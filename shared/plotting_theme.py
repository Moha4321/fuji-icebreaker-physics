"""
Plotting theme for the Fuji Icebreaker Physics Suite.

Consistent visual identity across all three modules.
Polar/oceanic color palette — dark background, cold accent colors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────────────────────────
COLORS = {
    # primary accent colors
    "ice_blue"    : "#A8D8EA",
    "deep_ocean"  : "#0A2342",
    "snow_white"  : "#F0F4F8",
    "fuji_orange" : "#E8723A",   # the ship's hull color
    "jare_pink"   : "#D4857A",   # the tractor's paint
    "crack_red"   : "#C0392B",
    "safe_green"  : "#27AE60",
    "vrs_yellow"  : "#F39C12",

    # backgrounds
    "bg_dark"     : "#0D1B2A",
    "bg_panel"    : "#1A2B3C",
    "grid_line"   : "#1E3A5F",
    "text_light"  : "#CDD9E5",
    "text_dim"    : "#6B8CAE",
}

# Custom colormaps
def make_ice_cmap():
    """Deep blue → cyan → white. For pressure/stress fields."""
    return LinearSegmentedColormap.from_list(
        "ice_pressure",
        [(0, COLORS["deep_ocean"]),
         (0.4, "#1E6B9E"),
         (0.7, COLORS["ice_blue"]),
         (1.0, COLORS["snow_white"])]
    )

def make_fracture_cmap():
    """Dark → orange → white. For stress near fracture."""
    return LinearSegmentedColormap.from_list(
        "fracture_stress",
        [(0, COLORS["bg_dark"]),
         (0.5, COLORS["fuji_orange"]),
         (0.85, "#F5CBA7"),
         (1.0, COLORS["snow_white"])]
    )

def make_velocity_cmap():
    """Blue → green → yellow. For velocity magnitude fields."""
    return LinearSegmentedColormap.from_list(
        "velocity_field",
        [(0, COLORS["deep_ocean"]),
         (0.3, "#1ABC9C"),
         (0.7, "#F1C40F"),
         (1.0, "#E74C3C")]
    )


# ─────────────────────────────────────────────────────────────
#  GLOBAL THEME APPLICATION
# ─────────────────────────────────────────────────────────────
def apply_theme():
    """Call once at notebook top. Sets matplotlib global style."""
    plt.rcParams.update({
        # figure
        "figure.facecolor"      : COLORS["bg_dark"],
        "figure.dpi"            : 120,
        "figure.figsize"        : (10, 6),
        # axes
        "axes.facecolor"        : COLORS["bg_panel"],
        "axes.edgecolor"        : COLORS["grid_line"],
        "axes.labelcolor"       : COLORS["text_light"],
        "axes.titlecolor"       : COLORS["snow_white"],
        "axes.titlesize"        : 14,
        "axes.labelsize"        : 11,
        "axes.titleweight"      : "bold",
        "axes.spines.top"       : False,
        "axes.spines.right"     : False,
        # grid
        "grid.color"            : COLORS["grid_line"],
        "grid.linewidth"        : 0.5,
        "grid.alpha"            : 0.6,
        "axes.grid"             : True,
        # ticks
        "xtick.color"           : COLORS["text_dim"],
        "ytick.color"           : COLORS["text_dim"],
        "xtick.labelsize"       : 9,
        "ytick.labelsize"       : 9,
        # legend
        "legend.facecolor"      : COLORS["bg_panel"],
        "legend.edgecolor"      : COLORS["grid_line"],
        "legend.labelcolor"     : COLORS["text_light"],
        "legend.fontsize"       : 9,
        # lines
        "lines.linewidth"       : 2.0,
        "lines.antialiased"     : True,
        # font
        "font.family"           : "monospace",
        "text.color"            : COLORS["text_light"],
        # save
        "savefig.facecolor"     : COLORS["bg_dark"],
        "savefig.dpi"           : 150,
        "savefig.bbox"          : "tight",
    })


def styled_fig(nrows=1, ncols=1, figsize=None, title=None):
    """Create a styled figure + axes. Returns (fig, ax)."""
    apply_theme()
    figsize = figsize or (10 * ncols, 6 * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if title:
        fig.suptitle(title, color=COLORS["snow_white"],
                     fontsize=15, fontweight="bold", y=1.02)
    return fig, ax


def add_photo_credit(ax, text="JS Fuji, Nagoya Port Museum"):
    """Add a subtle photo credit watermark."""
    ax.text(0.99, 0.01, text,
            transform=ax.transAxes,
            fontsize=7, color=COLORS["text_dim"],
            ha="right", va="bottom", style="italic")


def annotate_key_point(ax, x, y, label, color=None):
    """Drop a labeled marker on a key result point."""
    color = color or COLORS["fuji_orange"]
    ax.axvline(x, color=color, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.scatter([x], [y], color=color, s=80, zorder=5)
    ax.annotate(label,
                xy=(x, y), xytext=(x + 0.02, y * 1.05),
                color=color, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0))