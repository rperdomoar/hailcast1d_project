# hail_growth_stochastic.py
import numpy as np
import time

from .hail_growth import simulate_trajectory
from .hail_analysis import beard_terminal_velocity   # === NUEVO ===


def generate_initial_conditions_mc(
        n_particles,
        D0_mean=5.0,
        D0_std=2.0,
        z_birth_min=4000.0,
        z_birth_max=6000.0,
        T0_list=(263.15, 265.15),
        seed=None):
    """
    Genera condiciones iniciales Monte Carlo coherentes con el modelo determinista.
    """
    rng = np.random.default_rng(seed)

    # Tamaño inicial (mm)
    D0 = rng.normal(D0_mean, D0_std, n_particles)
    D0 = np.clip(D0, 0.1, None)

    # Altura de nacimiento (m)
    z0 = rng.uniform(z_birth_min, z_birth_max, n_particles)

    # Temperatura inicial del grano
    T0 = rng.choice(T0_list, size=n_particles)

    return D0, z0, T0


def run_montecarlo(profile,
                   n_particles=100,
                   seed=None,
                   progress_every=10,          # === NUEVO ===
                   warn_time=5.0,              # === NUEVO (segundos)
                   **trajectory_kwargs):
    """
    Corre un ensamble Monte Carlo usando exactamente la misma dinámica
    que el caso determinista, con monitoreo de avance.
    """
    D0, z0, T0 = generate_initial_conditions_mc(
        n_particles=n_particles,
        seed=seed
    )

    results = []

    t_start_total = time.time()

    for i in range(n_particles):

        t0 = time.time()

        out = simulate_trajectory(
            profile=profile,
            D0_mm=D0[i],
            z0=z0[i],
            T0_K=T0[i],
            **trajectory_kwargs
        )

        dt = time.time() - t0

        # -------------------------
        # Metadata MC
        # -------------------------
        out["z_birth"] = z0[i]
        out["T0_init"] = T0[i]

        # -------------------------
        # v_terminal MC (Beard)
        # -------------------------
        D_final_mm = 2.0 * out["radius"][-1] * 1000.0  # mm
        out["D_final_mm"] = D_final_mm
        out["v_terminal_beard"] = beard_terminal_velocity(D_final_mm)

        results.append(out)

        # -------------------------
        # PROGRESO
        # -------------------------
        if (i + 1) % progress_every == 0 or i == 0:
            elapsed = time.time() - t_start_total
            mean_t = elapsed / (i + 1)
            pct = 100.0 * (i + 1) / n_particles

            print(
                f"[MC] {i+1:4d}/{n_particles} "
                f"({pct:5.1f}%) | "
                f"⟨t⟩={mean_t:5.2f}s/part"
            )

        # -------------------------
        # WARNING si tarda demasiado
        # -------------------------
        if dt > warn_time:
            print(
                f"⚠️  Partícula {i} tardó {dt:.2f}s "
                f"(D0={D0[i]:.2f} mm, z0={z0[i]:.0f} m)"
            )

    total = time.time() - t_start_total
    print(f"[MC] Finalizado: {n_particles} partículas en {total:.1f}s")

    return results
