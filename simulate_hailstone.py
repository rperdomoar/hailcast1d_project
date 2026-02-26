import numpy as np
from hail_growth import simulate_trajectory
from cloud_model import CloudProfile


def simulate_hailstone(D0_mm=5.0, T0_K_list=None, updraft_type='strong',
                        montecarlo=False, n_runs=50, seed=42, n_bins=5):
    """
    Simula trayectorias de granizos individuales o con Monte Carlo,
    devolviendo trayectorias y tamaño medio por clase.

    Parámetros:
    - D0_mm: diámetro medio del embrión (mm)
    - T0_K_list: lista de temperaturas iniciales en Kelvin
    - updraft_type: tipo de corriente ascendente ('weak', 'dry', 'strong')
    - montecarlo: True para múltiples simulaciones
    - n_runs: número de simulaciones Monte Carlo
    - seed: semilla para reproducibilidad
    - n_bins: número de clases en el histograma
    """

    if T0_K_list is None:
        T0_K_list = [265.15, 260.15]  # -8°C y -13°C

    rho_ice = 900.0
    np.random.seed(seed)

    if montecarlo:
        montecarlo_results = []

        # Distribución lognormal de tamaños iniciales (mm)
        mu = np.log(D0_mm) - 0.5 * 0.5**2
        sigma = 0.5
        diameters_init = np.random.lognormal(mean=mu, sigma=sigma, size=n_runs)
        diameters_init = np.clip(diameters_init, 0.9, 10.0)

        # Definir bins y asegurar índices correctos
        bins = np.linspace(min(diameters_init), max(diameters_init), n_bins+1)
        bin_indices = np.digitize(diameters_init, bins, right=True) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins-1)  # evitar índice fuera de rango

        # Inicializar clases
        class_sizes = {i: [] for i in range(n_bins)}

        for T0_K in T0_K_list:
            for i, D0_i in enumerate(diameters_init):
                D0_m = D0_i / 1000.0  # mm → m
                r0 = D0_m / 2.0
                m0 = 4/3 * np.pi * rho_ice * r0**3
                z0 = 10000.0
                v0 = 0.0
                t_max = 200.0
                dt = 0.1

                profile = CloudProfile(
                    updraft_type=updraft_type,
                    lwc_peak=None,
                    w_peak=None,
                    z_min=0.0,
                    z_max=12000.0,
                    z_center=8000.0
                )

                times, masses, radios, heights, velocities = simulate_trajectory(
                    profile, m0, z0, v0, t_max, dt, T0_K
                )

                montecarlo_results.append({
                    "time": times,
                    "height": heights / 1000.0,
                    "density": np.full_like(heights, rho_ice),
                    "diameter": 2 * radios * 1000.0,
                    "temperature": T0_K - 0.0065 * (z0 - heights),
                    "D0_init": D0_i,
                    "T0_init": T0_K
                })

                class_sizes[bin_indices[i]].append(D0_i)

        # Tamaño medio de cada clase
        avg_size_per_class = {
            i: np.mean(class_sizes[i]) if len(class_sizes[i]) > 0 else None
            for i in range(n_bins)
        }

        return montecarlo_results, diameters_init, avg_size_per_class

    else:
        determinista_results = {}
        embryo_sizes_mm = [0.9, 5.0, 7.5, 10.0]  # Diámetros específicos

        for D0_mm in embryo_sizes_mm:
            for T0_K in T0_K_list:
                D0_m = D0_mm / 1000.0
                r0 = D0_m / 2.0
                m0 = 4/3 * np.pi * rho_ice * r0**3
                z0 = 10000.0
                v0 = 0.0
                t_max = 2000.0
                dt = 1.0

                profile = CloudProfile(
                    updraft_type=updraft_type,
                    lwc_peak=None,
                    w_peak=None,
                    z_min=0.0,
                    z_max=12000.0,
                    z_center=8000.0
                )

                times, masses, radios, heights, velocities = simulate_trajectory(
                    profile, m0, z0, v0, t_max, dt, T0_K
                )

                determinista_results[f"D0_{D0_mm}_T0_{int(T0_K)}"] = {
                    "time": times,
                    "height": heights / 1000.0,
                    "density": np.full_like(heights, rho_ice),
                    "diameter": 2 * radios * 1000.0,
                    "temperature": T0_K - 0.0065 * (z0 - heights),
                    "D0_init": D0_mm,
                    "T0_init": T0_K
                }

        return determinista_results
