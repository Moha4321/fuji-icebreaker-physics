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


def plot_resistance_curve(vehicle_mass: float = JAREKomatsu.mass, save_path: str = None):
    apply_theme()
    b_vals = np.linspace(0.10, 1.40, 300)
    results = sweep_track_widths(vehicle_mass, b_vals)

    Rc = np.array([r.R_compaction for r in results])
    Rb = np.array([r.R_bulldozing for r in results])
    Rt = np.array([r.R_turning for r in results])
    R_total = np.array([r.R_total for r in results])
    z  = np.array([r.sinkage * 100 for r in results]) 

    b_opt, _ = optimal_track_width(vehicle_mass)
    perf_opt = evaluate_track(b_opt, vehicle_mass)
    perf_actual = evaluate_track(JAREKomatsu.track_width, vehicle_mass)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(COLORS["bg_dark"])

    # Plot all components
    ax1.plot(b_vals, Rc / 1000, color=COLORS["ice_blue"], label="Compaction (Sinkage)", lw=2, ls="--")
    ax1.plot(b_vals, Rb / 1000, color=COLORS["fuji_orange"], label="Bulldozing (Bow-wave)", lw=2, ls="--")
    ax1.plot(b_vals, Rt / 1000, color=COLORS["jare_pink"], label="Skid-Steering (Turning)", lw=2, ls="--")
    ax1.plot(b_vals, R_total / 1000, color=COLORS["snow_white"], label="Total Resistance ← Minimize this", lw=3)

    ax1.axvline(b_opt, color=COLORS["safe_green"], lw=1.5)
    ax1.scatter([b_opt],[perf_opt.R_total / 1000], color=COLORS["safe_green"], s=100, zorder=6, label=f"Math Optimum: {b_opt:.3f} m")

    ax1.axvline(JAREKomatsu.track_width, color=COLORS["jare_pink"], lw=1.5, ls=":")
    ax1.scatter([JAREKomatsu.track_width],[perf_actual.R_total / 1000], color=COLORS["jare_pink"], s=100, zorder=6, marker="D", label=f"JARE Actual: {JAREKomatsu.track_width:.2f} m")

    ax1.set_ylabel("Resistance [kN]")
    ax1.set_title(f"Track Width Optimization ({vehicle_mass:.0f} kg)\nJARE deviation: {abs(b_opt-JAREKomatsu.track_width)/b_opt*100:.1f}%", pad=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(0.10, 1.40)

    ax2.fill_between(b_vals, z, alpha=0.3, color=COLORS["ice_blue"])
    ax2.plot(b_vals, z, color=COLORS["ice_blue"], lw=2)
    ax2.axvline(b_opt, color=COLORS["safe_green"], lw=1.5)
    ax2.set_xlabel("Track width b [m]")
    ax2.set_ylabel("Sinkage [cm]")
    ax2.set_xlim(0.10, 1.40)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig) # close to prevent double rendering

def plot_optimization_surface(save_path: str = None):
    apply_theme()
    mass_range  = np.linspace(1000, 8000, 60)
    width_range = np.linspace(0.10, 1.40, 60)
    energy = optimization_surface(mass_range, width_range)
    
    energy_log = np.log10(np.clip(energy, 1, None))
    M_grid, B_grid = np.meshgrid(mass_range / 1000, width_range, indexing='ij')

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(COLORS["bg_panel"])

    surf = ax.plot_surface(M_grid, B_grid, energy_log, cmap="plasma", alpha=0.85, linewidth=0)

    perf_jare = evaluate_track(JAREKomatsu.track_width, JAREKomatsu.mass)
    ax.scatter([JAREKomatsu.mass / 1000], [JAREKomatsu.track_width],[np.log10(perf_jare.R_total)],
               color=COLORS["jare_pink"], s=200, zorder=10, label="JARE Komatsu (actual)")

    opt_widths = [optimal_track_width(m)[0] for m in mass_range]
    opt_energy = [optimal_track_width(m)[1] for m in mass_range]
    ax.plot(mass_range / 1000, opt_widths,[np.log10(max(e, 1)) for e in opt_energy],
            color=COLORS["safe_green"], lw=2.5, label="Optimal width locus")

    ax.set_xlabel("Vehicle mass [tonnes]")
    ax.set_ylabel("Track width [m]")
    ax.set_zlabel("log10(Resistance) [N]")
    ax.legend()
    if save_path: 
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Module A optimization plots...")
    plot_resistance_curve(save_path="resistance_curve.png")
    plot_optimization_surface(save_path="optimization_surface.png")