# ================================================================
# analyze_profiles.py
# Comparación Monte Carlo por perfil de nube
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from hail_analysis import beard_terminal_velocity


# ================================================================
# 1️⃣ Estadísticas por perfil
# ================================================================
def compute_profile_statistics(mc_results):

    D_final = np.array([r["diameter"][-1] for r in mc_results])
    H_max   = np.array([np.max(r["height"]) for r in mc_results]) / 1000
    v_mc    = beard_terminal_velocity(D_final)

    stats = {
        "D_mean": np.mean(D_final),
        "D_std": np.std(D_final),
        "D_skew": skew(D_final),

        "H_mean": np.mean(H_max),
        "H_std": np.std(H_max),

        "v_mean": np.mean(v_mc),
        "bias_beard": np.mean(v_mc - beard_terminal_velocity(D_final)),
        "rmse_beard": np.sqrt(np.mean((v_mc - beard_terminal_velocity(D_final))**2))
    }

    return stats, D_final, H_max


# ================================================================
# 2️⃣ Histograma comparado de tamaños
# ================================================================
def plot_diameter_comparison(mc_results_by_profile):

    plt.figure(figsize=(9,6))

    for profile, results in mc_results_by_profile.items():
        D = np.array([r["diameter"][-1] for r in results])
        plt.hist(D, bins=20, density=True, alpha=0.4, label=profile)

    plt.xlabel("Diámetro final (mm)")
    plt.ylabel("Densidad")
    plt.title("Distribución de tamaños finales por perfil")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ================================================================
# 3️⃣ Distribución de alturas máximas
# ================================================================
def plot_height_comparison(mc_results_by_profile):

    plt.figure(figsize=(9,6))

    for profile, results in mc_results_by_profile.items():
        H = np.array([np.max(r["height"]) for r in results]) / 1000
        plt.hist(H, bins=20, density=True, alpha=0.4, label=profile)

    plt.xlabel("Altura máxima (km)")
    plt.ylabel("Densidad")
    plt.title("Distribución de altura máxima por perfil")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ================================================================
# 4️⃣ Pluma de trayectorias por perfil
# ================================================================
def plot_plumes(mc_results_by_profile):

    for profile, results in mc_results_by_profile.items():

        plt.figure(figsize=(8,6))

        for r in results:
            plt.plot(r["time"],
                     np.array(r["height"])/1000,
                     alpha=0.2)

        plt.xlabel("Tiempo (s)")
        plt.ylabel("Altura (km)")
        plt.title(f"Pluma de trayectorias — {profile}")
        plt.grid(alpha=0.3)
        plt.show()


# ================================================================
# 5️⃣ Función principal
# ================================================================
def analyze_all_profiles(mc_results_by_profile):

    print("\n==============================")
    print("ESTADÍSTICAS POR PERFIL")
    print("==============================\n")

    for profile, results in mc_results_by_profile.items():

        stats, _, _ = compute_profile_statistics(results)

        print(f"Perfil: {profile}")
        print(f"  D_mean  = {stats['D_mean']:.2f} mm")
        print(f"  D_std   = {stats['D_std']:.2f}")
        print(f"  D_skew  = {stats['D_skew']:.2f}")
        print(f"  H_mean  = {stats['H_mean']:.2f} km")
        print(f"  H_std   = {stats['H_std']:.2f}")
        print(f"  v_mean  = {stats['v_mean']:.2f} m/s")
        print(f"  RMSE Beard = {stats['rmse_beard']:.3f}")
        print("----------------------------------")

    plot_diameter_comparison(mc_results_by_profile)
    plot_height_comparison(mc_results_by_profile)
    plot_plumes(mc_results_by_profile)
