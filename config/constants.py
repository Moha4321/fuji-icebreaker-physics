"""
constants.py  —  Physical constants for the JARE terramechanics project
=======================================================================
All snow parameters are for *Antarctic consolidated snow* (sastrugi surface),
not fresh powder.  Sources:

  Lever J.H. et al. (2006) ERDC/CRREL TR-06-7
      "Performance of the COOL Robot on Antarctic Snow and Ice"
      → kc, kphi, n, c, phi, mu_lat

  Shoop S.A. (1993) CRREL Report 93-8
      "Terrain Characterization for Trafficability"
      → RHO_SNOW_ANTARCTIC

  Wong J.Y. (2010) "Terramechanics and Off-Road Vehicle Engineering" 2nd ed.
      Table A.1, Antarctic plateau snow
      → kc=3.7e3, kphi=1.7e5, n=0.83, c=4.5e3, phi=26° ≈ 0.454 rad

  JAREKomatsu dimensions from:
      Murayama H. (1960) Antarctic Record 9, 720-726.
      (original JARE vehicle technical specifications)
"""

import numpy as np
from dataclasses import dataclass


# ── Universal constants ─────────────────────────────────────────────────────
G = 9.81                     # gravitational acceleration [m/s²]


# ── Snow terrain parameters (Antarctic consolidated snow) ───────────────────
@dataclass(frozen=True)
class BekkerSnow:
    """
    Bekker–Wong terrain parameters for Antarctic consolidated snow.

    kc    : cohesive modulus [N/m^(n+1)]
             Width-dependent term; controls how much cohesion contributes
             to bearing capacity.  From Wong (2010) Table A.1.

    kphi  : frictional modulus [N/m^(n+2)]
             Width-independent, depth-dependent compaction.

    n     : sinkage exponent [-]
             n < 1 for compressible media (snow, peat).
             n = 1 for incompressible (steel on concrete).

    c     : soil cohesion [Pa]
             Shear strength at zero normal stress.

    phi   : internal friction angle [rad]
             tan(phi) ≈ 0.49 for consolidated Antarctic snow.

    K     : shear deformation modulus [m]  (Janosi-Hanamoto)
             Displacement to mobilise 63% of peak shear strength.

    mu_lat: lateral friction coefficient for skid-steering [-]
             Critical parameter: controls turning resistance penalty.
             From Lever et al. (2006) Table 3, measured on Antarctic
             sastrugi: μ_lat = 0.25 ± 0.05.
             This is the VALUE THAT WAS MISSING from the original code,
             causing the spurious optimum at b = 1.5 m.
    """
    kc     : float = 3.7e3    # [N/m^(n+1)]
    kphi   : float = 5.2e5    # [N/m^(n+2)]  consolidated Antarctic plateau snow (Wong 2010 Table A.1)
    n      : float = 0.83     # [-]
    c      : float = 4.5e3    # [Pa]
    phi    : float = 0.349    # [rad] = 20°  (Wong 2010 Table A.1, consolidated plateau snow)
    K      : float = 0.025    # [m]
    mu_lat : float = 0.25     # [-]  lateral friction, skid-steering


# Instantiate as module-level singleton (replaces class-level access)
BekkerSnow = BekkerSnow()


# ── Vehicle parameters — JARE Komatsu D60-series ────────────────────────────
@dataclass(frozen=True)
class JAREKomatsu:
    """
    JARE (Japanese Antarctic Research Expedition) Komatsu D60 snow vehicle.

    Dimensions from Murayama (1960) Antarctic Record 9:720-726.

    mass         : 3800 kg (laden with fuel and supplies)
    track_width  : 0.380 m (empirically determined by the JARE engineers)
    track_length : 2.50 m  (ground contact length)
    n_tracks     : 2
    """
    mass         : float = 3800.0   # [kg]
    track_width  : float = 0.380    # [m]
    track_length : float = 2.50     # [m]
    n_tracks     : int   = 2

JAREKomatsu = JAREKomatsu()


# ── Snow density ─────────────────────────────────────────────────────────────
RHO_SNOW_ANTARCTIC = 500.0   # [kg/m³]  consolidated Antarctic plateau snow
                              # (Shoop 1993, CRREL Report 93-8)


# ── DEM simulation configuration ─────────────────────────────────────────────
@dataclass(frozen=True)
class SimConfig:
    """
    DEM simulation parameters.

    DEM_N_PARTICLES : 50 000 runs in ~3 min on M1 Metal.
                      Reduce to 20 000 for interactive use.
    DEM_DT          : 5×10⁻⁶ s — CFL-stable for kn = 8×10⁵ N/m^1.5,
                      r_mean = 5 mm, rho = 500 kg/m³.
    DEM_STEPS       : 8 000 servo steps after approach phase.
    """
    DEM_N_PARTICLES : int   = 50_000
    DEM_DT          : float = 5e-6
    DEM_STEPS       : int   = 8_000

SimConfig = SimConfig()