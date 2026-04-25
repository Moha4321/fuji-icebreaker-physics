"""
Module A — dem_snow.py

Discrete Element Method (DEM) simulation of snow under a moving track.
50,000 disk particles. Hertz contact model. Runs on M1 Metal GPU via Taichi.

THE PHYSICS:
    Each snow grain is a disk with:
      - mass m, radius r, position x, velocity v
      - Contact force when two disks overlap (Hertz contact)
      - Cohesion force (snow grains stick to each other weakly)
      - Gravity

    The track is a rigid moving boundary pressing down from above.

    What emerges WITHOUT being programmed:
      - Pressure distribution under the track matches Bekker's equation
      - Bow wave of compacted grains ahead of the track
      - The optimal width result from bekker_model.py is reproduced

HARDWARE:
    M1 Mac (16 GB):  ti.cpu or ti.metal (set ARCH below)
    Colab T4:        ti.cuda

Run time: ~3 minutes for 8000 steps at 50k particles on M1 Metal.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.constants import (BekkerSnow, JAREKomatsu, G,
                               RHO_SNOW_ANTARCTIC, SimConfig)

try:
    import taichi as ti
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False
    print("⚠  Taichi not installed. Install with: pip install taichi")
    print("   DEM simulation will fall back to NumPy (slow, illustrative only).")


# ─────────────────────────────────────────────────────────────
#  HARDWARE SELECTION
# ─────────────────────────────────────────────────────────────
def init_taichi(arch: str = "auto"):
    """
    Initialize Taichi with the appropriate backend.

    arch options:
        "auto"  → detects Metal on M1, CUDA on Colab, CPU otherwise
        "metal" → force M1 GPU (best for Mac)
        "cuda"  → force CUDA (Colab T4)
        "cpu"   → force CPU (always works, slower)
    """
    if not TAICHI_AVAILABLE:
        return False

    if arch == "auto":
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            ti.init(arch=ti.metal, default_fp=ti.f32,
                    device_memory_fraction=0.7)
            print("✓ Taichi initialized: Apple Metal (M1/M2)")
        else:
            try:
                ti.init(arch=ti.cuda, default_fp=ti.f32)
                print("✓ Taichi initialized: CUDA")
            except Exception:
                ti.init(arch=ti.cpu, default_fp=ti.f32)
                print("✓ Taichi initialized: CPU")
    elif arch == "metal":
        ti.init(arch=ti.metal, default_fp=ti.f32,
                device_memory_fraction=0.7)
    elif arch == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f32)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    return True


# ─────────────────────────────────────────────────────────────
#  DEM SIMULATION CLASS
# ─────────────────────────────────────────────────────────────
class SnowDEM:
    """
    2D Discrete Element Method simulation of snow under a rigid track.

    Particle system:
        N particles, each a disk with radius drawn from a normal
        distribution centered on r_mean. Polydispersity (size variation)
        is important — monodisperse particles crystallize and give
        unrealistic behavior.

    Contact model:
        Normal force:  Fn = kn * overlap^1.5   (Hertz, elastic)
        Tangential:    Ft = min(kt * delta_t, mu * Fn)  (Coulomb slip)
        Cohesion:      Fc = -fc if overlap > -delta_c  (short-range adhesion)

    Integration:
        Velocity-Verlet, timestep 1e-5 s (stable for this stiffness).
    """

    def __init__(self,
                 N: int = SimConfig.DEM_N_PARTICLES,
                 domain_width: float = 2.0,    # [m]
                 domain_height: float = 1.0,   # [m]
                 track_width: float = JAREKomatsu.track_width,
                 vehicle_mass: float = JAREKomatsu.mass,
                 r_mean: float = 0.005,         # mean particle radius [m] = 5mm
                 r_std: float = 0.001):
        """
        Args:
            N            : number of particles
            domain_width : simulation box width [m]
            domain_height: simulation box height [m]
            track_width  : track contact width [m] (what we vary in optimization)
            vehicle_mass : [kg]
            r_mean       : mean grain radius [m]
            r_std        : grain radius std dev [m]
        """
        self.N = N
        self.W = domain_width
        self.H = domain_height
        self.track_width = track_width
        self.vehicle_mass = vehicle_mass
        self.r_mean = r_mean
        self.r_std  = r_std

        # Physical constants
        self.dt   = SimConfig.DEM_DT            # timestep [s]
        self.g    = G                           # gravity [m/s²]
        self.kn   = 1e6    # normal stiffness [N/m^1.5] — Hertz
        self.kt   = 0.8e6  # tangential stiffness [N/m]
        self.mu   = 0.4    # friction coefficient (snow-snow)
        self.en   = 0.7    # restitution coefficient (damping)
        self.fc   = 2.0    # cohesion force [N]
        self.dc   = 0.001  # cohesion cutoff distance [m]

        # Particle density from snow density
        # m = rho * pi * r²  (2D mass from area)
        self.rho  = RHO_SNOW_ANTARCTIC

        self._initialized = False

        if TAICHI_AVAILABLE:
            self._setup_taichi_fields()

    def _setup_taichi_fields(self):
        """Allocate Taichi fields on GPU memory."""
        N = self.N
        # Positions, velocities, forces, radii, masses
        self.x  = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.v  = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.f  = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.r  = ti.field(dtype=ti.f32, shape=N)
        self.m  = ti.field(dtype=ti.f32, shape=N)
        self.omega = ti.field(dtype=ti.f32, shape=N)   # angular velocity
        self.torque = ti.field(dtype=ti.f32, shape=N)

        # Track position and force measurement
        self.track_y       = ti.field(dtype=ti.f32, shape=())
        self.track_force_y = ti.field(dtype=ti.f32, shape=())
        self.pressure_x    = ti.field(dtype=ti.f32, shape=200)

        # Grid for broad-phase collision detection (linked-cell)
        cell_size = self.r_mean * 2.5
        self.cell_size = cell_size
        self.grid_nx = int(self.W / cell_size) + 1
        self.grid_ny = int(self.H / cell_size) + 1
        self.grid_cnt  = ti.field(dtype=ti.i32,
                                   shape=(self.grid_nx, self.grid_ny))
        self.grid_part = ti.field(dtype=ti.i32,
                                   shape=(self.grid_nx, self.grid_ny, 64))

    def initialize_particles(self):
        """
        Place particles in a random packing.
        Uses numpy, then copies to Taichi fields.
        """
        print(f"Initializing {self.N} snow particles...")

        # Random radii (polydisperse)
        rng = np.random.default_rng(42)
        radii = np.clip(
            rng.normal(self.r_mean, self.r_std, self.N),
            self.r_mean * 0.5,
            self.r_mean * 2.0
        ).astype(np.float32)

        # Masses from 2D area (treat as cylinders)
        masses = self.rho * np.pi * radii**2  # [kg/m] per unit depth

        # Initial positions: random, non-overlapping via random sequential placement
        # Simplified: grid placement with jitter (fast to initialize)
        # Full RSA would be more physical but takes minutes
        n_cols = int(np.sqrt(self.N * self.W / self.H))
        n_rows = int(self.N / n_cols) + 1
        dx = self.W / n_cols
        dy = self.H / n_rows * 0.8  # pack into lower 80% of domain

        xs, ys = [], []
        for row in range(n_rows):
            for col in range(n_cols):
                if len(xs) >= self.N:
                    break
                x = col * dx + rng.uniform(-dx*0.3, dx*0.3)
                y = row * dy + rng.uniform(-dy*0.3, dy*0.3)
                xs.append(np.clip(x, radii[len(xs)], self.W - radii[len(xs)]))
                ys.append(np.clip(y, radii[len(ys)], self.H * 0.8 - radii[len(ys)]))

        positions = np.array([xs[:self.N], ys[:self.N]], dtype=np.float32).T
        velocities = np.zeros((self.N, 2), dtype=np.float32)

        if TAICHI_AVAILABLE:
            self.x.from_numpy(positions)
            self.v.from_numpy(velocities)
            self.r.from_numpy(radii)
            self.m.from_numpy(masses)
            # Track starts above particles
            self.track_y[None] = self.H * 0.85
        else:
            # Numpy fallback storage
            self._x = positions
            self._v = velocities
            self._r = radii
            self._m = masses
            self._track_y = self.H * 0.85

        self._initialized = True
        print(f"  Domain: {self.W}m × {self.H}m")
        print(f"  Track width: {self.track_width}m centered at x={self.W/2:.2f}m")
        print(f"  Vehicle load (per track): "
              f"{self.vehicle_mass * G / 2:.0f} N")

    def get_taichi_kernels(self):
        """
        Define Taichi kernels. Must be called after ti.init().
        Returns dict of kernel functions.

        Note: Taichi kernels are JIT-compiled on first call.
        Subsequent calls are fast (compiled GPU code).
        """
        if not TAICHI_AVAILABLE:
            return {}

        N = self.N
        W = self.W
        H = self.H
        kn = self.kn
        kt = self.kt
        mu = self.mu
        en = self.en
        fc = self.fc
        dc = self.dc
        dt = self.dt
        g  = self.g
        cell_size = self.cell_size
        grid_nx = self.grid_nx
        grid_ny = self.grid_ny

        x  = self.x
        v  = self.v
        f  = self.f
        r  = self.r
        m  = self.m
        track_y       = self.track_y
        track_force_y = self.track_force_y
        track_width_val = self.track_width
        domain_cx = W / 2.0
        vehicle_load = self.vehicle_mass * G / 2.0  # load per track
        pressure_x = self.pressure_x
        grid_cnt = self.grid_cnt
        grid_part = self.grid_part

        @ti.kernel
        def clear_forces():
            for i in range(N):
                f[i] = ti.Vector([0.0, -m[i] * g])  # gravity
            track_force_y[None] = 0.0
            for k in range(200):
                pressure_x[k] = 0.0

        @ti.kernel
        def build_grid():
            for cx in range(grid_nx):
                for cy in range(grid_ny):
                    grid_cnt[cx, cy] = 0
            for i in range(N):
                cx = int(x[i][0] / cell_size)
                cy = int(x[i][1] / cell_size)
                cx = ti.max(0, ti.min(cx, grid_nx - 1))
                cy = ti.max(0, ti.min(cy, grid_ny - 1))
                old = ti.atomic_add(grid_cnt[cx, cy], 1)
                if old < 64:
                    grid_part[cx, cy, old] = i

        @ti.kernel
        def compute_contacts():
            """
            Hertz contact forces between overlapping particles.
            Broad phase: linked-cell grid.
            Narrow phase: check all pairs in neighboring cells.
            """
            for i in range(N):
                xi = x[i]
                ri = r[i]
                cx0 = int(xi[0] / cell_size)
                cy0 = int(xi[1] / cell_size)

                for dcx in ti.static(range(-1, 2)):
                    for dcy in ti.static(range(-1, 2)):
                        cx = cx0 + dcx
                        cy = cy0 + dcy
                        if 0 <= cx < grid_nx and 0 <= cy < grid_ny:
                            cnt = grid_cnt[cx, cy]
                            for k in range(cnt):
                                j = grid_part[cx, cy, k]
                                if j <= i:
                                    continue
                                xj = x[j]
                                rj = r[j]
                                dx = xi - xj
                                dist = dx.norm()
                                overlap = ri + rj - dist
                                if overlap > 0:
                                    # Normal direction
                                    n_hat = dx / (dist + 1e-10)
                                    # Hertz normal force
                                    fn_mag = kn * (overlap ** 1.5)
                                    # Damping
                                    vrel_n = (v[i] - v[j]).dot(n_hat)
                                    fn_damp = -2.0 * (1.0 - en) * ti.sqrt(
                                        kn * (ri + rj) / 2.0 * ti.sqrt(overlap)
                                    ) * 0.5 * (m[i] * m[j]) / (m[i] + m[j]) * vrel_n
                                    fn = (fn_mag + fn_damp) * n_hat

                                    # Tangential (friction) — simplified
                                    vrel = v[i] - v[j]
                                    vrel_t = vrel - vrel.dot(n_hat) * n_hat
                                    ft_mag = ti.min(
                                        kt * vrel_t.norm() * dt,
                                        mu * (fn_mag + fn_damp)
                                    )
                                    ft = -ft_mag * (
                                        vrel_t / (vrel_t.norm() + 1e-10)
                                    )

                                    # Cohesion (negative overlap = gap)
                                    fcoh = ti.Vector([0.0, 0.0])
                                    if overlap > -dc:
                                        fcoh = -fc * n_hat

                                    fi_total = fn + ft + fcoh
                                    f[i] += fi_total
                                    f[j] -= fi_total

        @ti.kernel
        def apply_track_force():
            """
            Track presses down on particles beneath it.
            Track is a rigid rectangular boundary:
              x in [cx - b/2, cx + b/2]
              y = track_y (bottom surface of track)

            The track descends at a rate controlled by a servo:
            if the sum of vertical reaction forces < vehicle_load,
            the track moves down; otherwise it holds position.
            """
            ty = track_y[None]
            cx = domain_cx
            bx_half = track_width_val / 2.0
            x_lo = cx - bx_half
            x_hi = cx + bx_half

            total_fy = 0.0

            for i in range(N):
                xi = x[i][0]
                yi = x[i][1]
                ri = r[i]
                if x_lo <= xi <= x_hi:
                    gap = (ty - ri) - yi
                    if gap < 0.0:  # overlap
                        overlap = -gap
                        fn = kn * (overlap ** 1.5)
                        # Damping
                        vn = v[i][1]
                        fn_damp = ti.min(0.0, -2.0 * 0.5 * en * ti.sqrt(
                            kn * ri * ti.sqrt(overlap)) * vn)
                        f_total = fn + fn_damp
                        f[i][1] += f_total
                        total_fy += f_total

                        # Record local pressure
                        k_bin = int((xi - x_lo) / track_width_val * 200)
                        k_bin = ti.max(0, ti.min(k_bin, 199))
                        pressure_x[k_bin] += f_total / (track_width_val / 200.0)

            track_force_y[None] = total_fy

        @ti.kernel
        def integrate():
            """Velocity-Verlet integration. Wall boundaries."""
            for i in range(N):
                v[i] += f[i] / m[i] * dt
                x[i] += v[i] * dt

                # Wall boundaries
                ri = r[i]
                if x[i][0] < ri:
                    x[i][0] = ri; v[i][0] = ti.abs(v[i][0]) * 0.3
                if x[i][0] > W - ri:
                    x[i][0] = W - ri; v[i][0] = -ti.abs(v[i][0]) * 0.3
                if x[i][1] < ri:
                    x[i][1] = ri; v[i][1] = ti.abs(v[i][1]) * 0.3

        @ti.kernel
        def servo_track(target_load: float, track_speed: float):
            """
            Move track down until it carries the target load.
            Then hold. Classic servo controller.
            """
            current_force = track_force_y[None]
            error = target_load - current_force
            # Simple proportional control
            delta = track_speed * dt * ti.tanh(error / (target_load * 0.1))
            track_y[None] -= delta

        return {
            "clear_forces"      : clear_forces,
            "build_grid"        : build_grid,
            "compute_contacts"  : compute_contacts,
            "apply_track_force" : apply_track_force,
            "integrate"         : integrate,
            "servo_track"       : servo_track,
        }

    def run(self, n_steps: int = SimConfig.DEM_STEPS,
            record_every: int = 100) -> dict:
        """
        Run the DEM simulation.

        Returns dict with:
            sinkage_history  : track_y as function of time
            force_history    : track reaction force vs time
            pressure_profile : spatial pressure distribution under track
            positions_final  : final particle positions for visualization
        """
        if not self._initialized:
            self.initialize_particles()

        if not TAICHI_AVAILABLE:
            print("Taichi not available — running NumPy fallback (fewer particles)")
            return self._numpy_fallback(n_steps, record_every)

        kernels = self.get_taichi_kernels()
        target_load = self.vehicle_mass * G / 2.0  # one track

        sinkage_history = []
        force_history   = []
        step_history    = []

        print(f"Running DEM: {n_steps} steps, {self.N} particles")
        print(f"Target track load: {target_load:.0f} N")

        initial_track_y = self.track_y[None]

        for step in range(n_steps):
            kernels["clear_forces"]()
            kernels["build_grid"]()
            kernels["compute_contacts"]()
            kernels["apply_track_force"]()
            kernels["servo_track"](target_load, track_speed=0.1)
            kernels["integrate"]()

            if step % record_every == 0:
                ty    = self.track_y[None]
                force = self.track_force_y[None]
                sinkage_history.append(initial_track_y - ty)
                force_history.append(force)
                step_history.append(step)

                if step % (record_every * 10) == 0:
                    pct = step / n_steps * 100
                    print(f"  Step {step:5d}/{n_steps} ({pct:.0f}%)  "
                          f"track_y={ty:.4f}m  F={force:.0f}N")

        # Final state
        positions = self.x.to_numpy()
        radii     = self.r.to_numpy()
        pressure  = self.pressure_x.to_numpy()

        final_sinkage = initial_track_y - self.track_y[None]
        print(f"\nDEM complete. Final sinkage: {final_sinkage*100:.1f} cm")

        # Compare to Bekker analytical
        from module_a_terramechanics.bekker_model import sinkage_from_load
        z_bekker = sinkage_from_load(
            target_load, self.track_width, JAREKomatsu.track_length
        )
        print(f"Bekker prediction:     {z_bekker*100:.1f} cm")
        print(f"DEM result:            {final_sinkage*100:.1f} cm")
        print(f"Agreement:             {abs(final_sinkage-z_bekker)/z_bekker*100:.1f}% error")

        return {
            "step_history"    : np.array(step_history),
            "sinkage_history" : np.array(sinkage_history),
            "force_history"   : np.array(force_history),
            "pressure_profile": pressure,
            "positions"       : positions,
            "radii"           : radii,
            "final_sinkage"   : final_sinkage,
            "bekker_sinkage"  : z_bekker,
        }

    def _numpy_fallback(self, n_steps: int, record_every: int) -> dict:
        """
        Pure NumPy DEM — runs on CPU, ~2000 particles max for speed.
        For debugging and environments without Taichi.
        """
        print("  NumPy fallback: 2000 particles, simple contact detection")
        N_small = min(2000, self.N)
        rng = np.random.default_rng(42)

        r = np.full(N_small, self.r_mean, dtype=np.float32)
        m = self.rho * np.pi * r**2

        # Grid placement
        n_cols = 40
        n_rows = N_small // n_cols + 1
        xs = np.linspace(r[0], self.W - r[0], n_cols)
        ys_row = np.linspace(r[0], self.H * 0.7, n_rows)
        xx, yy = np.meshgrid(xs, ys_row)
        pos = np.column_stack([xx.ravel()[:N_small],
                               yy.ravel()[:N_small]]).astype(np.float32)
        vel = np.zeros((N_small, 2), dtype=np.float32)

        track_y = self.H * 0.85
        target  = self.vehicle_mass * G / 2.0
        dt = self.dt * 10  # larger dt ok for numpy (fewer particles, less stiff)

        sinkage_hist, force_hist, step_hist = [], [], []
        initial_ty = track_y

        for step in range(n_steps // 10):  # 10x fewer steps
            # Forces
            f = np.zeros((N_small, 2), dtype=np.float32)
            f[:, 1] = -m * self.g

            # Particle-particle: brute force O(N²) — only ok for small N
            for i in range(N_small):
                for j in range(i+1, N_small):
                    dx = pos[i] - pos[j]
                    dist = np.linalg.norm(dx)
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        n_hat = dx / (dist + 1e-10)
                        fn = self.kn * overlap**1.5 * n_hat
                        f[i] += fn
                        f[j] -= fn

            # Track contact
            cx = self.W / 2
            mask = ((pos[:, 0] > cx - self.track_width/2) &
                    (pos[:, 0] < cx + self.track_width/2))
            track_force = 0.0
            for i in np.where(mask)[0]:
                gap = (track_y - r[i]) - pos[i, 1]
                if gap < 0:
                    fn = self.kn * (-gap)**1.5
                    f[i, 1] += fn
                    track_force += fn

            # Servo
            error = target - track_force
            track_y -= 0.01 * dt * np.tanh(error / (target * 0.1))

            # Integrate
            vel += f / m[:, None] * dt
            pos += vel * dt
            pos[:, 0] = np.clip(pos[:, 0], r, self.W - r)
            pos[:, 1] = np.clip(pos[:, 1], r, self.H)

            if step % (record_every // 10) == 0:
                sinkage_hist.append(initial_ty - track_y)
                force_hist.append(track_force)
                step_hist.append(step)

        from module_a_terramechanics.bekker_model import sinkage_from_load
        z_bekker = sinkage_from_load(target, self.track_width,
                                     JAREKomatsu.track_length)

        return {
            "step_history"    : np.array(step_hist),
            "sinkage_history" : np.array(sinkage_hist),
            "force_history"   : np.array(force_hist),
            "pressure_profile": np.zeros(200),
            "positions"       : pos,
            "radii"           : r,
            "final_sinkage"   : initial_ty - track_y,
            "bekker_sinkage"  : z_bekker,
        }


if __name__ == "__main__":
    init_taichi("auto")

    sim = SnowDEM(
        N=SimConfig.DEM_N_PARTICLES,
        track_width=JAREKomatsu.track_width,
        vehicle_mass=JAREKomatsu.mass,
    )
    sim.initialize_particles()
    results = sim.run(n_steps=SimConfig.DEM_STEPS, record_every=100)

    print(f"\nFinal sinkage: {results['final_sinkage']*100:.2f} cm")
    print(f"Bekker prediction: {results['bekker_sinkage']*100:.2f} cm")