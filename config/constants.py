"""
Physical constants for the Fuji Icebreaker Physics Suite.

Every value here is traceable to a published source.
We don't approximate — we parameterize, so everything
can be changed and the whole simulation updates.

Sources:
  [W10]  Wong, J.Y. (2010). Terramechanics and Off-Road Vehicle Engineering. Elsevier.
  [L06]  Lever, J.H. et al. (2006). Performance of CRREL Lightweight Dozer. ERDC/CRREL TR-06-7.
  [P99]  Picu, R.C. & Gupta, V. (1999). Crack healing in ice. Acta Materialia.
  [T73]  Timco, G.W. & Weeks, W.F. (2010). A review of the engineering properties of sea ice. Cold Regions S&T.
  [S61]  Sikorsky Aircraft Corporation HSS-2 Flight Manual (1961).
  [B60]  Bekker, M.G. (1960). Off-the-Road Locomotion. Univ. of Michigan Press.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
#  UNIVERSAL
# ─────────────────────────────────────────────────────────────
G = 9.81          # gravitational acceleration [m/s²]
RHO_WATER = 1025  # seawater density [kg/m³]
RHO_ICE   = 917   # pure ice density [kg/m³]
RHO_SNOW_ANTARCTIC = 400  # Antarctic compacted snow [kg/m³] — [L06] Table 2

# ─────────────────────────────────────────────────────────────
#  MODULE A — BEKKER TERRAMECHANICS (Antarctic Snow)
#  Source: [W10] Table 3.1, Antarctic compacted snow regime
# ─────────────────────────────────────────────────────────────
class BekkerSnow:
    """
    Bekker-Wong parameters for Antarctic compacted snow.
    These are the numbers that go into p = (kc/b + kphi) * z^n.

    kc    — cohesive modulus of terrain deformation [N/m^(n+1)]
            Represents how strongly the snow resists being pushed
            sideways (shear). Higher = snow is more "sticky."

    kphi  — frictional modulus [N/m^(n+2)]
            How strongly snow resists pure compression. Denser
            snow has higher kphi.

    n     — sinkage exponent [dimensionless]
            If n=1, pressure grows linearly with sinkage.
            Real snow has n≈0.5 — it gets stiffer as you push deeper
            (because compaction raises density).

    c     — cohesion [Pa] — shear strength at zero normal pressure
    phi   — internal friction angle [rad]
    K     — shear deformation modulus [m] — from Janosi-Hanamoto
    """
    kc   = 4_530    # [N/m^(n+1)]     [W10] Table 3.1
    kphi = 196_440  # [N/m^(n+2)]     [W10] Table 3.1
    n    = 0.50     # [dimensionless]  [W10] Table 3.1
    c    = 1_020    # [Pa]             [W10] Table 3.1
    phi  = np.radians(20.7)  # [rad]  [W10] Table 3.1
    K    = 0.025    # [m]              [W10] Table 3.1

# JARE Komatsu snow tractor (Image 3 in your photos)
# Dimensions reverse-engineered from photo + known comparable vehicles
class JAREKomatsu:
    mass          = 3_800    # [kg] — estimated, comparable to Tucker Sno-Cat
    track_width   = 0.38     # [m]  — estimated from photo aspect ratio
    track_length  = 2.10     # [m]  — estimated from photo
    n_tracks      = 2        # two tracks
    max_speed     = 5.0      # [km/h] in deep snow

# ─────────────────────────────────────────────────────────────
#  MODULE B — ICE MECHANICS (Sea Ice / Antarctic Pack Ice)
#  Source: [T73] review paper — the definitive reference
# ─────────────────────────────────────────────────────────────
class SeaIce:
    """
    Mechanical properties of first-year Antarctic sea ice.
    These control how the ice plate bends and fractures.

    E        — Young's modulus: stiffness. Sea ice is ~5 GPa,
               about 10x weaker than steel (200 GPa).

    nu       — Poisson's ratio: when you compress in one direction,
               how much does it expand sideways? Ice ≈ 0.3.

    K_IC     — Mode I fracture toughness: how much stress the ice can
               concentrate at a crack tip before the crack runs.
               Ice is shockingly brittle: 120 kPa√m vs steel's 50,000 kPa√m.

    sigma_t  — Tensile strength: direct tension failure.
               Ice fails in tension much more easily than compression.

    rho      — density of sea ice (slightly less than pure ice, salt pockets)
    h_typical — typical first-year pack ice thickness [m]
    """
    E         = 5e9     # [Pa]       Young's modulus  [T73]
    nu        = 0.3     # [-]        Poisson's ratio  [T73]
    K_IC      = 120e3   # [Pa√m]     fracture toughness [T73]
    sigma_t   = 500e3   # [Pa]       tensile strength [T73]
    sigma_c   = 5e6     # [Pa]       compressive strength [T73]
    rho       = 900     # [kg/m³]    sea ice density [T73]
    h_typical = 1.5     # [m]        first-year pack ice [T73]

# JS Fuji icebreaker specs (historical, publicly documented)
class JSFuji:
    displacement  = 8_000  # [tonnes] full load
    mass          = 8e6    # [kg]
    bow_angle     = 22.0   # [degrees] — the famous sloped bow
    length        = 100.0  # [m]
    beam          = 22.0   # [m]
    max_ice_thickness = 1.2  # [m] at 3 knots (from JMSDF records)
    speed_knots   = 3.0    # [knots] in ice
    speed_ms      = 3.0 * 0.514  # [m/s]

# ─────────────────────────────────────────────────────────────
#  MODULE C — HELICOPTER AERODYNAMICS
#  Source: [S61] and standard rotor aerodynamics references
# ─────────────────────────────────────────────────────────────
class HSS2SeaKing:
    """
    Sikorsky HSS-2 Sea King — the helicopter on Fuji's deck.

    The key non-dimensionalization is h/D: the ratio of
    hover height to rotor diameter. Ground effect becomes
    significant below h/D ≈ 1.0, strong below h/D ≈ 0.5.

    v_h   — hover-induced velocity (from momentum theory):
            v_h = sqrt(T / (2 * rho_air * A_disk))
            This is the fundamental velocity scale. VRS onset
            occurs when descent rate ≈ 0.71 * v_h.
    """
    mass            = 9_300   # [kg] max gross weight [S61]
    rotor_diameter  = 18.9    # [m]  main rotor [S61]
    rotor_radius    = 18.9/2  # [m]
    rotor_area      = np.pi * (18.9/2)**2  # [m²]
    n_blades        = 5       # main rotor blades
    blade_chord     = 0.46    # [m]
    engine_power    = 2 * 1_050 * 746  # [W] — two 1050hp engines

    # Derived hover quantities
    T_hover = mass * G               # [N] hover thrust ≈ weight
    rho_air = 1.225                  # [kg/m³] sea level air
    v_h     = np.sqrt(
        T_hover / (2 * rho_air * rotor_area)
    )  # [m/s] hover induced velocity ≈ 8.2 m/s

    # VRS onset threshold (from Leishman 2006, Principles of Helicopter Aerodynamics)
    VRS_descent_ratio = 0.71  # descent_rate / v_h > this → VRS risk zone

# ─────────────────────────────────────────────────────────────
#  SIMULATION SETTINGS (tuned for M1 16GB + Colab T4)
# ─────────────────────────────────────────────────────────────
class SimConfig:
    # Module A — DEM
    DEM_N_PARTICLES   = 50_000   # particle count — fits in 4GB
    DEM_DT            = 1e-5     # [s] timestep for stable DEM contact
    DEM_STEPS         = 8_000    # total steps per run

    # Module B — Lattice
    LATTICE_NX        = 200      # grid points in x
    LATTICE_NY        = 200      # grid points in y
    FRACTURE_STEPS    = 300      # ship advance steps

    # Module C — PINN
    PINN_HIDDEN_LAYERS = 5
    PINN_NEURONS       = 128
    PINN_OMEGA_0       = 30.0    # SIREN first-layer frequency
    PINN_N_COLLOC      = 10_000  # collocation points
    PINN_EPOCHS        = 40_000  # for Colab T4 — ~90min
    PINN_LR            = 1e-4    # learning rate