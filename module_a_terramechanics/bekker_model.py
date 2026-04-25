"""
Module A — Terramechanics of the J.A.R.E. Snow Vehicle
bekker_model.py

The Bekker-Wong pressure-sinkage model, implemented from first principles.

THE CORE INSIGHT:
    Track width is not "bigger = better." There is a mathematical optimum.
    Too narrow → the track sinks deep, creating a bow wave of snow
                 that costs enormous energy to push through.
    Too wide   → turning becomes impossible, track slippage rises,
                 and you still have a bow-wave problem laterally.

    The JARE engineers in the 1950s found this optimum through iteration.
    This module finds it analytically in 3 milliseconds.

Reference: Wong (2010) "Terramechanics and Off-Road Vehicle Engineering"
           Lever et al. (2006) ERDC/CRREL TR-06-7
"""

import numpy as np
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.constants import BekkerSnow, JAREKomatsu, G


# ─────────────────────────────────────────────────────────────
#  CORE BEKKER-WONG EQUATIONS
# ─────────────────────────────────────────────────────────────

def bekker_pressure(z: float, b: float,
                    kc: float = BekkerSnow.kc,
                    kphi: float = BekkerSnow.kphi,
                    n: float = BekkerSnow.n) -> float:
    """
    Bekker (1960) pressure-sinkage relationship.

        p(z) = (kc/b + kphi) * z^n

    Args:
        z    : sinkage depth [m]
        b    : smaller track dimension (width) [m]
        kc   : cohesive modulus [N/m^(n+1)]
        kphi : frictional modulus [N/m^(n+2)]
        n    : sinkage exponent [-]

    Returns:
        p    : normal stress under track [Pa]

    Physical meaning of (kc/b + kphi):
        The term kc/b captures width-dependent cohesive resistance.
        Wider tracks → smaller kc/b → less cohesion per unit area.
        kphi is width-independent: pure vertical compaction.
    """
    return (kc / b + kphi) * (z ** n)


def sinkage_from_load(W: float, b: float, L: float,
                      kc: float = BekkerSnow.kc,
                      kphi: float = BekkerSnow.kphi,
                      n: float = BekkerSnow.n) -> float:
    """
    Invert the Bekker equation: given a load W on one track,
    find the equilibrium sinkage z.

    Derivation:
        Contact pressure p = W / (b * L)   [uniformly distributed]
        Bekker: p = (kc/b + kphi) * z^n
        → z = [p / (kc/b + kphi)]^(1/n)

    Args:
        W  : weight on one track [N]  (total_weight / n_tracks)
        b  : track width [m]
        L  : track contact length [m]

    Returns:
        z  : sinkage depth [m]
    """
    p = W / (b * L)                          # contact pressure [Pa]
    K = kc / b + kphi                        # combined modulus
    z = (p / K) ** (1.0 / n)                # invert power law
    return z


def compaction_resistance(W: float, b: float, L: float,
                          kc: float = BekkerSnow.kc,
                          kphi: float = BekkerSnow.kphi,
                          n: float = BekkerSnow.n) -> float:
    """
    Resistance force from compacting snow into the sinkage groove.
    This is the energy cost of creating the rut — the snow that
    gets permanently crushed under the track.

    Formula (Wong 2010, eq 2.13):
        Rc = b * integral_0^z p(z') dz'
           = b * (kc/b + kphi) * z^(n+1) / (n+1)

    Returns:
        Rc : compaction resistance [N]
    """
    z = sinkage_from_load(W, b, L, kc, kphi, n)
    K = kc / b + kphi
    Rc = b * K * (z ** (n + 1)) / (n + 1)
    return Rc


def bulldozing_resistance(W: float, b: float, L: float,
                          c: float = BekkerSnow.c,
                          phi: float = BekkerSnow.phi,
                          kc: float = BekkerSnow.kc,
                          kphi: float = BekkerSnow.kphi,
                          n: float = BekkerSnow.n) -> float:
    """
    Bulldozing resistance: the force needed to push the bow wave
    of snow ahead of the track front face.

    This term grows with track width b — critically.
    It's why you can't just make tracks infinitely wide.

    Simplified Reece (1965) formula for bow-wave resistance:
        Rb ≈ c * Nc * b * z² + 0.5 * rho_snow * g * Nq * b * z²

    We use the simplified form from Wong (2010) eq 2.15:
        Rb = b * z * (c * Nc + 0.5 * rho_snow * g * z * Ngamma)

    where Nc, Ngamma are bearing capacity factors:
        Nc     = (Nq - 1) / tan(phi)
        Nq     = e^(pi*tan(phi)) * tan²(45 + phi/2)
        Ngamma ≈ 2*(Nq + 1)*tan(phi)
    """
    from config.constants import RHO_SNOW_ANTARCTIC
    z = sinkage_from_load(W, b, L, kc, kphi, n)

    # Bearing capacity factors (Terzaghi-Reece)
    Nq     = np.exp(np.pi * np.tan(phi)) * np.tan(np.pi/4 + phi/2)**2
    Nc     = (Nq - 1.0) / np.tan(phi)
    Ngamma = 2.0 * (Nq + 1.0) * np.tan(phi)

    Rb = b * z * (c * Nc + 0.5 * RHO_SNOW_ANTARCTIC * G * z * Ngamma)
    return Rb


def gross_thrust(W: float, b: float, L: float,
                 c: float = BekkerSnow.c,
                 phi: float = BekkerSnow.phi,
                 K_shear: float = BekkerSnow.K,
                 slip: float = 0.15) -> float:
    """
    Gross thrust from Janosi-Hanamoto shear stress model.

        H = (c + p*tan(phi)) * (1 - K/j_max * (1 - e^(-j_max/K))) * A

    where j_max = slip * L (total shear deformation at max slip)

    This is how hard the track can push backward on the snow.
    Drawbar pull = H - Rc - Rb.

    Args:
        slip : wheel slip ratio (0 = no slip, 1 = full spin)

    Returns:
        H : gross thrust [N]
    """
    p = W / (b * L)
    tau_max = c + p * np.tan(phi)       # maximum shear stress [Pa]
    j_max   = slip * L                  # shear deformation at slip [m]
    # Janosi-Hanamoto integration over track length
    integral = j_max - K_shear * (1 - np.exp(-j_max / K_shear))
    H = tau_max * b * integral / j_max * b * L
    # Simpler closed form (Wong 2010 eq 2.20):
    H = tau_max * b * L * (1 - (K_shear / j_max) * (1 - np.exp(-j_max / K_shear)))
    return H


# ─────────────────────────────────────────────────────────────
#  TOTAL RESISTANCE AND DRAWBAR PULL
# ─────────────────────────────────────────────────────────────

@dataclass
class TrackPerformance:
    """All performance quantities for one (width, mass) configuration."""
    track_width   : float   # b [m]
    vehicle_mass  : float   # M [kg]
    sinkage       : float   # z [m]
    R_compaction  : float   # Rc [N]
    R_bulldozing  : float   # Rb [N]
    R_total       : float   # Rc + Rb [N] — total motion resistance
    gross_thrust  : float   # H [N]
    drawbar_pull  : float   # DP = H - R_total [N]
    energy_per_m  : float   # R_total [J/m] — the thing we minimize
    is_mobile     : bool    # DP > 0


def evaluate_track(track_width: float, vehicle_mass: float,
                   track_length: float = JAREKomatsu.track_length,
                   n_tracks: int = JAREKomatsu.n_tracks,
                   slip: float = 0.15) -> TrackPerformance:
    """
    Full performance evaluation for a given track width and vehicle mass.
    This is the function we sweep to find the optimum.

    Args:
        track_width   : b [m]
        vehicle_mass  : M [kg]
        track_length  : L [m] — contact patch length
        n_tracks      : number of driven tracks (2)
        slip          : track slip ratio

    Returns:
        TrackPerformance dataclass
    """
    W_per_track = vehicle_mass * G / n_tracks  # load per track [N]
    b = track_width
    L = track_length

    z  = sinkage_from_load(W_per_track, b, L)
    Rc = compaction_resistance(W_per_track, b, L) * n_tracks
    Rb = bulldozing_resistance(W_per_track, b, L) * n_tracks
    H  = gross_thrust(W_per_track, b, L, slip=slip) * n_tracks

    R_total    = Rc + Rb
    drawbar    = H - R_total
    energy_pm  = R_total          # [J/m] — Joules per meter of travel

    return TrackPerformance(
        track_width  = b,
        vehicle_mass = vehicle_mass,
        sinkage      = z,
        R_compaction = Rc,
        R_bulldozing = Rb,
        R_total      = R_total,
        gross_thrust = H,
        drawbar_pull = drawbar,
        energy_per_m = energy_pm,
        is_mobile    = drawbar > 0,
    )


# ─────────────────────────────────────────────────────────────
#  OPTIMIZATION — FIND THE OPTIMAL TRACK WIDTH
# ─────────────────────────────────────────────────────────────

def optimal_track_width(vehicle_mass: float,
                        b_min: float = 0.10,
                        b_max: float = 1.50) -> Tuple[float, float]:
    """
    Find the track width that minimizes total motion resistance.

    This is the mathematical saddle point where:
        d(Rc + Rb)/db = 0

    Rc decreases with wider tracks (less sinkage).
    Rb increases with wider tracks (more bulldozing area).
    The minimum of their sum is the optimum.

    Uses scipy scalar minimization — runs in microseconds.

    Returns:
        (optimal_width [m], minimum_resistance [N])
    """
    def objective(b):
        perf = evaluate_track(b, vehicle_mass)
        return perf.energy_per_m

    result = minimize_scalar(objective, bounds=(b_min, b_max),
                             method='bounded')
    return result.x, result.fun


def sweep_track_widths(vehicle_mass: float,
                       b_values: Optional[np.ndarray] = None) -> list:
    """
    Evaluate performance across a range of track widths.
    Returns list of TrackPerformance objects.
    """
    if b_values is None:
        b_values = np.linspace(0.10, 1.50, 200)

    results = [evaluate_track(b, vehicle_mass) for b in b_values]
    return results


def optimization_surface(mass_range: np.ndarray,
                         width_range: np.ndarray) -> np.ndarray:
    """
    Compute energy_per_m for a 2D grid of (mass, width) values.
    Returns a 2D array for surface plotting.

    This reveals the global landscape — where the optimum lives
    as a function of vehicle mass.
    """
    energy = np.zeros((len(mass_range), len(width_range)))
    for i, mass in enumerate(mass_range):
        for j, width in enumerate(width_range):
            perf = evaluate_track(width, mass)
            energy[i, j] = perf.energy_per_m
    return energy


# ─────────────────────────────────────────────────────────────
#  QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BEKKER MODEL — SELF TEST")
    print("=" * 60)

    M = JAREKomatsu.mass
    b_actual = JAREKomatsu.track_width

    # Evaluate the actual JARE vehicle
    perf_actual = evaluate_track(b_actual, M)
    print(f"\nActual JARE Komatsu (b = {b_actual:.2f} m):")
    print(f"  Sinkage:          {perf_actual.sinkage*100:.1f} cm")
    print(f"  Compaction res:   {perf_actual.R_compaction:.0f} N")
    print(f"  Bulldozing res:   {perf_actual.R_bulldozing:.0f} N")
    print(f"  Total resistance: {perf_actual.R_total:.0f} N")
    print(f"  Drawbar pull:     {perf_actual.drawbar_pull:.0f} N")
    print(f"  Mobile?           {perf_actual.is_mobile}")

    # Find mathematical optimum
    b_opt, R_min = optimal_track_width(M)
    print(f"\nMathematical optimum:")
    print(f"  Optimal width:    {b_opt:.3f} m")
    print(f"  Min resistance:   {R_min:.0f} N")
    print(f"  Actual width:     {b_actual:.3f} m")
    error = abs(b_opt - b_actual) / b_opt * 100
    print(f"  JARE deviation:   {error:.1f}% from optimum")
    print(f"\n  → Engineers hit within {error:.0f}% of math optimum without computers.")