# ================================================================
# hailcast1d.py (modular y final con v_terminal y caída realista)
# ================================================================
from .cloud_model import CloudProfile
from .hail_growth import simulate_trajectory
import numpy as np

# ================================================================
# 1️⃣ Simulación determinista de un solo grano
# ================================================================
def run_deterministic_simulation(profile, D0_mm, T0_K, z0, rho_init, **kwargs):
    """
    Simula un solo granizo determinista usando trayectoria realista.
    Retorna todas las variables, incluyendo velocidad terminal.
    """
    traj = simulate_trajectory(profile, D0_mm=D0_mm, z0=z0, T0_K=T0_K, v0=0.0, rho_init=rho_init, **kwargs)
    return {
        "diameter": traj["radius"]*2*1e3,  # mm
        "time": traj["time"],
        "height": traj["height"],
        "velocity": traj["velocity"],
        "v_terminal": traj["v_terminal"],
        "mass": traj["mass"],
        "temperature": traj["temperature"],
        "density": traj["density"],
        "D0_init": D0_mm
    }

# ================================================================
# 2️⃣ Simulación Monte Carlo de múltiples granos
# ================================================================
def run_montecarlo_simulations(profile, D0_mean, D0_std, n_particles, z0, rho_init, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    mc_results = []

    for _ in range(n_particles):
        D0 = max(rng.normal(D0_mean, D0_std), 0.1)
        traj = simulate_trajectory(profile, D0_mm=D0, z0=z0, T0_K=kwargs.get("T0_K", 263.15), v0=0.0, rho_init=rho_init, **kwargs)
        mc_results.append({
            "diameter": traj["radius"]*2*1e3,
            "time": traj["time"],
            "height": traj["height"],
            "velocity": traj["velocity"],
            "v_terminal": traj["v_terminal"],
            "mass": traj["mass"],
            "temperature": traj["temperature"],
            "density": traj["density"],
            "D0_init": D0
        })
    return mc_results

# ================================================================
# 3️⃣ Simulación de un caso específico
# ================================================================
def run_case(profile, D0_mm, T0_K, z0, rho_init, **kwargs):
    """
    Ejecuta un solo caso con parámetros específicos.
    """
    traj = simulate_trajectory(profile, D0_mm=D0_mm, z0=z0, T0_K=T0_K, v0=0.0, rho_init=rho_init, **kwargs)
    return {
        "D0_init": D0_mm,
        "time": traj["time"],
        "height": traj["height"],
        "diameter": traj["radius"]*2*1e3,
        "velocity": traj["velocity"],
        "v_terminal": traj["v_terminal"],
        "mass": traj["mass"],
        "temperature": traj["temperature"],
        "density": traj["density"]
    }

# ================================================================
# 4️⃣ Ejecutar un set de casos para un tipo de updraft
# ================================================================
def run_updraft_set(updraft_type, D0_list, T0_list, z0, rho_init, profile=None, **kwargs):
    """
    Corre un conjunto de simulaciones deterministas para un tipo de updraft.
    """
    if profile is None:
        profile = CloudProfile(
            updraft_type=updraft_type,
            z_center=4000.0,
            lwc_peak=None,
            w_peak=None,
            z_min=0.0,
            z_max=8000.0
        )

    det_results = {}
    for D0 in D0_list:
        for T0 in T0_list:
            res = run_case(profile, D0, T0, z0, rho_init=rho_init, **kwargs)
            det_results[f"D0_{D0:.1f}_T{int(T0)}"] = res
    return det_results

# ================================================================
# 5️⃣ Función principal para ejecutar todos los updrafts
# ================================================================
def run_all_cases(D0_list, T0_list, z0, rho_init, **kwargs):
    """
    Ejecuta todos los casos para los tres tipos de updraft.
    """
    updraft_types = ["strong", "dry", "weak"]
    results_by_updraft = {}

    for up in updraft_types:
        det_results = run_updraft_set(up, D0_list, T0_list, z0, rho_init, **kwargs)
        results_by_updraft[up] = det_results

    return results_by_updraft
