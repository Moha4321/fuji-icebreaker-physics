"""
Module B — ice_plate.py

Kirchhoff-Love plate theory: the ice sheet bending under the ship.

THE CORE INSIGHT:
    Sea ice is not a rigid slab — it behaves like a thin elastic plate
    floating on an elastic foundation (the water beneath it).

    When the ship climbs onto the ice, the ice bends. The bending stress
    peaks NOT directly under the ship, but in a ring some distance AHEAD.
    That's where the crack initiates. The ring propagates faster than sound.

    Equation:
        D∇⁴w + ρ_w * g * w = q(x,y)

    Where:
        D  = E*h³ / [12*(1-ν²)]  — flexural rigidity [N·m]
        w  — vertical deflection field [m]
        ∇⁴ — biharmonic operator (two applications of Laplacian)
        ρ_w*g*w — hydrostatic restoring force from water (Winkler foundation)
        q  — applied load from ship's bow [Pa]

    This is solved with finite differences on a 2D grid.

Reference:
    Timoshenko & Woinowsky-Krieger (1959) "Theory of Plates and Shells"
    Kerr (1976) "On the determination of the elastic modulus of floating ice"
"""

import numpy as np
from scipy.sparse import diags, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.constants import SeaIce, JSFuji, RHO_WATER, G


# ─────────────────────────────────────────────────────────────
#  PHYSICAL PROPERTIES
# ─────────────────────────────────────────────────────────────

def flexural_rigidity(h: float = SeaIce.h_typical,
                      E: float = SeaIce.E,
                      nu: float = SeaIce.nu) -> float:
    """
    D = E*h³ / [12*(1-ν²)]

    This is the single most important quantity for ice plate mechanics.
    It controls the "characteristic length" of the bending problem.

    Physical meaning:
        High D → stiff plate → load spreads over large area → small deflection
        Low D  → flexible plate → load concentrates → deep bending → cracks

    For 1.5m ice: D ≈ 2.4 × 10⁹ N·m  (stiff, but orders below steel structure)
    """
    return E * h**3 / (12.0 * (1.0 - nu**2))


def characteristic_length(h: float = SeaIce.h_typical,
                           E: float = SeaIce.E,
                           nu: float = SeaIce.nu) -> float:
    """
    l_c = [D / (ρ_w * g)]^(1/4)

    The characteristic bending length: the distance from the load
    center to the first zero-crossing of the deflection.

    This is physically the "sphere of influence" of the ship's weight
    on the ice. Beyond l_c, the ice is essentially undisturbed.

    For 1.5m sea ice: l_c ≈ 13–18 m
    The Fuji's bow half-width ≈ 11 m → the load pattern overlaps
    with the characteristic length → maximum bending efficiency.
    Not a coincidence.
    """
    D = flexural_rigidity(h, E, nu)
    return (D / (RHO_WATER * G))**0.25


def principal_stresses(sigma_xx: np.ndarray,
                       sigma_yy: np.ndarray,
                       sigma_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Principal stresses from plane stress tensor.

    σ₁,₂ = (σ_xx + σ_yy)/2 ± √[((σ_xx - σ_yy)/2)² + σ_xy²]

    The maximum principal stress σ₁ is what causes tensile failure in ice.
    Tensile strength of ice ≈ 500 kPa — much weaker than steel.
    The crack initiates where σ₁ first exceeds this threshold.
    """
    mean   = (sigma_xx + sigma_yy) / 2.0
    radius = np.sqrt(((sigma_xx - sigma_yy) / 2.0)**2 + sigma_xy**2)
    sigma1 = mean + radius  # max principal
    sigma2 = mean - radius  # min principal
    return sigma1, sigma2


# ─────────────────────────────────────────────────────────────
#  FINITE DIFFERENCE SOLVER FOR ∇⁴w + β²w = q/D
# ─────────────────────────────────────────────────────────────

class IcePlateSolver:
    """
    Finite difference solver for the floating ice plate equation:

        D * ∇⁴w + ρ_w * g * w = q(x, y)

    Discretized on a uniform NxN grid.
    The biharmonic operator ∇⁴ = ∇²(∇²) is the key challenge —
    it involves a 13-point stencil and requires care at boundaries.

    Grid convention:
        x: ship's direction of travel (columns)
        y: cross-ship direction (rows)
        Origin at center of domain.
    """

    def __init__(self,
                 nx: int = 200,
                 ny: int = 200,
                 Lx: float = 100.0,   # domain size [m]
                 Ly: float = 100.0,
                 h_ice: float = SeaIce.h_typical):
        """
        Args:
            nx, ny : grid resolution
            Lx, Ly : domain size [m]
            h_ice  : ice thickness [m]
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.h  = h_ice
        self.dx = Lx / nx
        self.dy = Ly / ny

        self.D   = flexural_rigidity(h_ice)
        self.l_c = characteristic_length(h_ice)
        self.beta_sq = RHO_WATER * G / self.D   # Winkler foundation stiffness

        # Physical grid (centered at 0,0)
        self.x_arr = np.linspace(-Lx/2, Lx/2, nx)
        self.y_arr = np.linspace(-Ly/2, Ly/2, ny)
        self.X, self.Y = np.meshgrid(self.x_arr, self.y_arr, indexing='ij')

        print(f"Ice plate solver initialized:")
        print(f"  Grid: {nx}×{ny}, dx={self.dx:.2f}m")
        print(f"  Ice thickness: {h_ice:.2f}m")
        print(f"  Flexural rigidity D: {self.D:.3e} N·m")
        print(f"  Characteristic length l_c: {self.l_c:.2f}m")

    def _build_biharmonic_matrix(self) -> csr_matrix:
        """
        Build the sparse matrix for D*∇⁴w + ρ_w*g*w on a 2D grid.

        The biharmonic finite difference stencil at (i,j) uses:
            - The point (i,j) itself
            - 4 nearest neighbors (±1 in x or y)
            - 4 next-nearest neighbors (±2 in x or y)
            - 4 diagonal neighbors (±1 in both x and y)
        Total: 13 points.

        Boundary: simply supported (w=0, ∇²w=0 at boundaries).
        This means the ice is allowed to deflect right up to the
        domain edge — a reasonable far-field approximation.
        """
        nx, ny = self.nx, self.ny
        N_total = nx * ny
        dx, dy  = self.dx, self.dy
        D       = self.D
        beta_sq = self.beta_sq

        # Index mapping: (i,j) → i*ny + j
        def idx(i, j):
            return i * ny + j

        # Build in lil format (efficient insertion), convert to csr for solving
        A = lil_matrix((N_total, N_total), dtype=np.float64)

        # Precompute stencil coefficients
        # For ∇²: (f_{i+1,j} - 2f_{i,j} + f_{i-1,j})/dx² + (f_{i,j+1} - 2f_{i,j} + f_{i,j-1})/dy²
        # For ∇⁴ = ∇²(∇²), apply twice using 5-point stencil composition
        # Result is the classic 13-point biharmonic stencil:

        idx2x = 1.0 / dx**4    # coefficient for ±2 in x
        idx2y = 1.0 / dy**4    # coefficient for ±2 in y
        c_xx = 1.0 / dx**2     # for cross terms
        c_yy = 1.0 / dy**2

        # Biharmonic stencil weights
        w_center = D * (6.0/dx**4 + 6.0/dy**4 + 8.0/(dx**2*dy**2)) + beta_sq
        w_x1     = D * (-4.0/dx**4 - 2.0/(dx**2*dy**2))   # (i±1, j)
        w_y1     = D * (-4.0/dy**4 - 2.0/(dx**2*dy**2))   # (i, j±1)
        w_x2     = D * (1.0/dx**4)                          # (i±2, j)
        w_y2     = D * (1.0/dy**4)                          # (i, j±2)
        w_xy     = D * (2.0/(dx**2*dy**2))                  # (i±1, j±1)

        for i in range(nx):
            for j in range(ny):
                k = idx(i, j)
                is_boundary = (i < 2 or i >= nx-2 or j < 2 or j >= ny-2)

                if is_boundary:
                    # Simply supported: w = 0
                    A[k, k] = 1.0
                else:
                    A[k, k] = w_center

                    # ±1 in x
                    A[k, idx(i+1, j)] += w_x1
                    A[k, idx(i-1, j)] += w_x1
                    # ±1 in y
                    A[k, idx(i, j+1)] += w_y1
                    A[k, idx(i, j-1)] += w_y1
                    # ±2 in x
                    A[k, idx(i+2, j)] += w_x2
                    A[k, idx(i-2, j)] += w_x2
                    # ±2 in y
                    A[k, idx(i, j+2)] += w_y2
                    A[k, idx(i, j-2)] += w_y2
                    # diagonals
                    A[k, idx(i+1, j+1)] += w_xy
                    A[k, idx(i+1, j-1)] += w_xy
                    A[k, idx(i-1, j+1)] += w_xy
                    A[k, idx(i-1, j-1)] += w_xy

        print(f"  Biharmonic matrix: {N_total}×{N_total}, "
              f"nnz={A.nnz}")
        return csr_matrix(A)

    def solve(self, load_field: np.ndarray) -> np.ndarray:
        """
        Solve D*∇⁴w + ρ_w*g*w = q for deflection w.

        Args:
            load_field : 2D array [nx, ny] of applied pressure [Pa]
                         Positive = downward load on ice

        Returns:
            w : 2D deflection field [nx, ny] in meters
        """
        A = self._build_biharmonic_matrix()
        q_flat = load_field.ravel() / self.D
        w_flat = spsolve(A, q_flat)
        return w_flat.reshape(self.nx, self.ny)

    def ship_load_field(self, ship_x: float = 0.0,
                        ship_mass: float = JSFuji.mass,
                        bow_angle: float = JSFuji.bow_angle) -> np.ndarray:
        """
        Compute the pressure field the ship's bow exerts on the ice.

        The bow is modeled as an elliptical pressure distribution:
        concentrated near the bow tip, spread over the hull waterline.

        The vertical component of the hull-ice reaction force is
        what bends the ice — the horizontal component is what
        propels the ship (icebreaker propulsion via ice reaction!).

        Args:
            ship_x     : x-position of bow tip [m]
            ship_mass  : [kg]
            bow_angle  : bow slope [degrees]

        Returns:
            q : 2D pressure field [Pa], positive downward
        """
        # Effective footprint of the bow on ice
        # The contact zone is approximately elliptical
        a = 8.0    # half-length of ellipse in x [m] — hull contact
        b = JSFuji.beam / 2.0  # half-width [m]

        # Load distribution: Hertzian ellipse
        # q(x,y) = q_max * sqrt(max(0, 1 - ((x-x0)/a)² - (y/b)²))
        # The vertical force component from the bow inclination
        # F_vertical = F_reaction * sin(bow_angle)
        # For quasi-static: F_reaction ≈ ship_mass * g / sin(bow_angle) * sin(bow_angle) = ship_mass * g
        # But the pressure distribution spans the contact zone

        x0 = ship_x
        ellipse_term = 1.0 - ((self.X - x0)/a)**2 - (self.Y/b)**2
        q_raw = np.where(ellipse_term > 0, np.sqrt(np.maximum(0, ellipse_term)), 0.0)

        # Normalize so total force = ship_mass * g
        total_area = np.sum(q_raw) * self.dx * self.dy
        if total_area > 0:
            q_max = ship_mass * G / total_area
            q = q_raw * q_max
        else:
            q = q_raw

        return q

    def compute_stress_field(self, w: np.ndarray, h: float = SeaIce.h_typical):
        """
        From deflection w, compute bending stresses.

        In Kirchhoff plate theory, the bending stress at the bottom fiber
        (most critical — tensile side when top is loaded):

            σ_xx = -E*h/(2*(1-ν²)) * (∂²w/∂x² + ν*∂²w/∂y²)
            σ_yy = -E*h/(2*(1-ν²)) * (∂²w/∂y² + ν*∂²w/∂x²)
            σ_xy = -E*h/(2*(1+ν))  * ∂²w/∂x∂y

        The factor h/2 comes from the distance from the neutral axis
        to the bottom fiber (where tension is maximum).

        Returns:
            sigma_xx, sigma_yy, sigma_xy, sigma1 (max principal)
        """
        E  = SeaIce.E
        nu = SeaIce.nu
        dx, dy = self.dx, self.dy

        # Second derivatives (finite differences, interior points)
        w2x = np.gradient(np.gradient(w, dx, axis=0), dx, axis=0)
        w2y = np.gradient(np.gradient(w, dy, axis=1), dy, axis=1)
        w2xy = np.gradient(np.gradient(w, dx, axis=0), dy, axis=1)

        factor = E * h / (2.0 * (1.0 - nu**2))
        sigma_xx = factor * (w2x + nu * w2y)
        sigma_yy = factor * (w2y + nu * w2x)
        sigma_xy = E * h / (2.0 * (1.0 + nu)) * w2xy

        sigma1, sigma2 = principal_stresses(sigma_xx, sigma_yy, sigma_xy)

        return sigma_xx, sigma_yy, sigma_xy, sigma1

    def find_fracture_ring(self, sigma1: np.ndarray,
                           sigma_t: float = SeaIce.sigma_t) -> np.ndarray:
        """
        Find where max principal stress exceeds tensile strength.
        This is the fracture initiation zone.

        Returns boolean mask of fracture locations.
        """
        return sigma1 > sigma_t


# ─────────────────────────────────────────────────────────────
#  CONVENIENCE WRAPPER — ONE-SHOT ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_fuji_on_ice(h_ice: float = SeaIce.h_typical,
                        ship_mass: float = JSFuji.mass,
                        nx: int = 150, ny: int = 150) -> dict:
    """
    Full plate analysis for the Fuji icebreaker on given ice.

    Returns dict with all fields needed for visualization.
    """
    solver = IcePlateSolver(nx=nx, ny=ny, Lx=120.0, Ly=120.0, h_ice=h_ice)

    # Ship bow at x = +15m from center (ship approaching from right)
    q = solver.ship_load_field(ship_x=15.0, ship_mass=ship_mass)

    print("\nSolving plate equation...")
    w = solver.solve(q)
    print(f"  Max deflection: {w.max()*100:.1f} cm  "
          f"(at x={solver.x_arr[w.argmax()//ny]:.1f}m)")

    sx, sy, sxy, s1 = solver.compute_stress_field(w, h_ice)
    fracture = solver.find_fracture_ring(s1)

    n_fracture = fracture.sum()
    total_cells = nx * ny
    print(f"  Fracture zone: {n_fracture} cells "
          f"({n_fracture/total_cells*100:.1f}% of domain)")
    print(f"  Max principal stress: {s1.max()/1e6:.2f} MPa "
          f"(ice tensile strength: {SeaIce.sigma_t/1e6:.2f} MPa)")
    can_break = s1.max() > SeaIce.sigma_t
    print(f"  Can Fuji break {h_ice:.1f}m ice? {'YES' if can_break else 'NO'}")

    return {
        "solver"   : solver,
        "q"        : q,
        "w"        : w,
        "sigma_xx" : sx,
        "sigma_yy" : sy,
        "sigma_xy" : sxy,
        "sigma1"   : s1,
        "fracture" : fracture,
        "l_c"      : solver.l_c,
        "D"        : solver.D,
    }


if __name__ == "__main__":
    result = analyze_fuji_on_ice()
    print("\nCharacteristic length:", f"{result['l_c']:.2f}m")
    print("Fuji beam / 2:", f"{JSFuji.beam/2:.1f}m")
    print("→ Ship width ≈ characteristic length: maximum bending leverage")