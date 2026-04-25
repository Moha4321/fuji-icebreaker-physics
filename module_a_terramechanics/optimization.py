"""
Module A — optimization.py

The optimization surface: energy cost vs (track_width, vehicle_mass).
This is the 3D landscape where the saddle point lives.

Run this to generate the key plot: the U-shaped resistance curve
with the JARE vehicle's actual design marked on it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.constants import JAREKomatsu, G
from module_a_terramechanics.bekker_model import (
    sweep_track_widths, optimal_track_width,
    optimization_surface, evaluate_track
)
from shared.plotting_theme import apply_theme, COLORS, styled_fig, annotate_key_point


def plot_resistance_curve(vehicle_mass: float = JAREKomatsu.mass,
                          save_path: str = None):
    """
    The key Module A plot.

    Shows Rc, Rb, and Rc+Rb as functions of track width.
    Marks the mathematical optimum and the JARE actual design.

    This is the visualization that makes the non-obvious realization land:
    there is a U-shaped curve with a clear minimum.
    """
    apply_theme()
    b_vals = np.linspace(0.10, 1.40, 300)
    results = sweep_track_widths(vehicle_mass, b_vals)

    Rc = np.array([r.R_compaction for r in results])
    Rb = np.array([r.R_bulldozing for r in results])
    Rt = Rc + Rb
    z  = np.array([r.sinkage * 100 for r in results])  # cm

    b_opt, _ = optimal_track_width(vehicle_mass)
    perf_opt = evaluate_track(b_opt, vehicle_mass)
    perf_actual = evaluate_track(JAREKomatsu.track_width, vehicle_mass)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(COLORS["bg_dark"])

    # ── Top plot: resistance components ──
    ax1.plot(b_vals, Rc / 1000, color=COLORS["ice_blue"],
             label="Compaction resistance $R_c$ (sinkage cost)", lw=2)
    ax1.plot(b_vals, Rb / 1000, color=COLORS["fuji_orange"],
             label="Bulldozing resistance $R_b$ (bow-wave cost)", lw=2)
    ax1.plot(b_vals, Rt / 1000, color=COLORS["snow_white"],
             label="Total resistance $R_c + R_b$ ← minimize this", lw=2.5)

    # Mark optimum
    ax1.axvline(b_opt, color=COLORS["safe_green"], lw=1.5, ls="--", alpha=0.8)
    ax1.scatter([b_opt], [perf_opt.R_total / 1000],
                color=COLORS["safe_green"], s=100, zorder=6,
                label=f"Mathematical optimum: b = {b_opt:.3f} m")

    # Mark actual JARE design
    ax1.axvline(JAREKomatsu.track_width, color=COLORS["jare_pink"],
                lw=1.5, ls=":", alpha=0.9)
    ax1.scatter([JAREKomatsu.track_width], [perf_actual.R_total / 1000],
                color=COLORS["jare_pink"], s=100, zorder=6, marker="D",
                label=f"Actual JARE design: b = {JAREKomatsu.track_width:.2f} m")

    ax1.set_ylabel("Resistance [kN]")
    ax1.set_title(
        f"Track Width Optimization — JARE Komatsu ({vehicle_mass:.0f} kg)\n"
        f"Vehicle mass: {vehicle_mass:.0f} kg  |  "
        f"JARE deviation from optimum: "
        f"{abs(b_opt-JAREKomatsu.track_width)/b_opt*100:.1f}%",
        pad=12
    )
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.set_xlim(0.10, 1.40)
    ax1.set_ylim(0, Rt.max() / 1000 * 1.15)

    # ── Bottom plot: sinkage depth ──
    ax2.fill_between(b_vals, z, alpha=0.3, color=COLORS["ice_blue"])
    ax2.plot(b_vals, z, color=COLORS["ice_blue"], lw=2)
    ax2.axvline(b_opt, color=COLORS["safe_green"], lw=1.5, ls="--", alpha=0.8)
    ax2.axvline(JAREKomatsu.track_width, color=COLORS["jare_pink"],
                lw=1.5, ls=":", alpha=0.9)
    ax2.set_xlabel("Track width  b  [m]")
    ax2.set_ylabel("Sinkage [cm]")
    ax2.set_title("Sinkage depth — narrows with wider tracks (but bulldozing cost rises)")
    ax2.set_xlim(0.10, 1.40)

    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def plot_optimization_surface(save_path: str = None):
    """
    3D surface: energy_per_meter vs (track_width, vehicle_mass).

    The valley in this surface is the locus of optimal track widths
    for each possible vehicle mass. Mark the JARE vehicle on it.
    """
    apply_theme()

    mass_range  = np.linspace(1_000, 8_000, 60)
    width_range = np.linspace(0.10, 1.40, 60)
    energy = optimization_surface(mass_range, width_range)

    # Clip for visualization (log scale implied)
    energy_log = np.log10(np.clip(energy, 1, None))

    M_grid, B_grid = np.meshgrid(mass_range / 1000, width_range, indexing='ij')

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(COLORS["bg_panel"])

    surf = ax.plot_surface(M_grid, B_grid, energy_log,
                           cmap="plasma", alpha=0.85,
                           linewidth=0, antialiased=True)

    # Mark JARE vehicle
    perf_jare = evaluate_track(JAREKomatsu.track_width, JAREKomatsu.mass)
    ax.scatter(
        [JAREKomatsu.mass / 1000],
        [JAREKomatsu.track_width],
        [np.log10(perf_jare.energy_per_m)],
        color=COLORS["jare_pink"], s=200, zorder=10,
        label="JARE Komatsu (actual)"
    )

    # Mark the valley of optima
    opt_widths = [optimal_track_width(m)[0] for m in mass_range]
    opt_energy = [optimal_track_width(m)[1] for m in mass_range]
    ax.plot(mass_range / 1000, opt_widths,
            [np.log10(max(e, 1)) for e in opt_energy],
            color=COLORS["safe_green"], lw=2.5, label="Optimal width curve")

    ax.set_xlabel("Vehicle mass [tonnes]", labelpad=8)
    ax.set_ylabel("Track width [m]", labelpad=8)
    ax.set_zlabel("log₁₀(Energy / meter) [J/m]", labelpad=8)
    ax.set_title("Energy Cost Surface: (mass, track width) → resistance",
                 color=COLORS["snow_white"], pad=14)
    ax.tick_params(colors=COLORS["text_dim"], labelsize=7)
    ax.legend(loc="upper left", fontsize=8)

    fig.colorbar(surf, ax=ax, shrink=0.4, label="log₁₀(Resistance [N])")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


if __name__ == "__main__":
    print("Generating Module A optimization plots...")
    plot_resistance_curve(save_path="results/module_a/resistance_curve.png")
    plot_optimization_surface(save_path="results/module_a/optimization_surface.png")