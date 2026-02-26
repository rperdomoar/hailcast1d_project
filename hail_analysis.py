# ================================================================
# hail_analysis.py
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import os

# ================================================================
# Beard (1980) — velocidad terminal empírica
# ================================================================
def beard_terminal_velocity(D_mm):
    D_mm = np.asarray(D_mm)
    return 9.65 - 10.3 * np.exp(-0.6 * D_mm)


# ================================================================
# 1️⃣ Estadísticas comparativas determinista vs MC
# ================================================================
def analyze_hailstone_results(determinista_results, montecarlo_results):

    det_diam = np.array([res["diameter"][-1]
                         for res in determinista_results.values()])

    mc_diam = np.array([res["D_final_mm"]
                        for res in montecarlo_results])

    metrics = {
        "D_mean_det": np.mean(det_diam),
        "D_mean_MC": np.mean(mc_diam),
        "D_std_MC": np.std(mc_diam),
        "Delta_D_mean": np.mean(mc_diam) - np.mean(det_diam),
    }

    return metrics


# ================================================================
# 2️⃣ Velocidad terminal vs tamaño
# ================================================================
def plot_terminal_velocity_comparison(results_by_updraft,
                                      montecarlo_results,
                                      D0_list,
                                      save_path=None):

    plt.figure(figsize=(8, 6))

    for updraft, res_dict in results_by_updraft.items():

        D_det = []
        v_det = []

        for key, res in res_dict.items():
            D_det.append(res["diameter"][-1])
            v_det.append(res["v_terminal"][-1])

        plt.scatter(D_det, v_det, alpha=0.7, label=f"{updraft} (det)")

    D_mc = np.array([r["D_final_mm"] for r in montecarlo_results])
    v_mc = np.array([r["v_terminal"][-1] for r in montecarlo_results])

    plt.scatter(D_mc, v_mc,
                color="black",
                s=15,
                alpha=0.3,
                label="Monte Carlo")

    plt.xlabel("Diámetro final (mm)")
    plt.ylabel("Velocidad terminal (m/s)")
    plt.title("Velocidad terminal vs diámetro")
    plt.grid(alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


# ================================================================
# 3️⃣ Estadísticas por perfil (Monte Carlo)
# ================================================================
def compute_profile_statistics(mc_results):

    D_final = np.array([r["D_final_mm"] for r in mc_results])

    v_sim = np.array([r["v_terminal_sim"] for r in mc_results])
    v_beard = np.array([r["v_terminal_beard"] for r in mc_results])

    if "height_max_m" in mc_results[0]:
        H_max = np.array([r["height_max_m"] for r in mc_results]) / 1000
    else:
        H_max = np.zeros_like(D_final)

    stats = {
        "D_mean": np.mean(D_final),
        "D_std": np.std(D_final),
        "D_skew": skew(D_final),
        "H_mean": np.mean(H_max),
        "H_std": np.std(H_max),
        "v_mean": np.mean(v_sim),
        "bias_beard": np.mean(v_sim - v_beard),
        "rmse_beard": np.sqrt(np.mean((v_sim - v_beard)**2))
    }

    return stats, D_final, H_max


# ================================================================
# 4️⃣ Histogramas
# ================================================================
def plot_diameter_comparison(mc_results_by_profile):

    plt.figure(figsize=(9, 6))

    for profile, results in mc_results_by_profile.items():
        D = np.array([r["D_final_mm"] for r in results])
        plt.hist(D, bins=20, density=True,
                 alpha=0.4, label=profile)

    plt.xlabel("Diámetro final (mm)")
    plt.ylabel("Densidad")
    plt.title("Distribución de tamaños finales por perfil")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_height_comparison(mc_results_by_profile):

    plt.figure(figsize=(9, 6))

    for profile, results in mc_results_by_profile.items():

        if "height_max_m" not in results[0]:
            continue

        H = np.array([r["height_max_m"]
                      for r in results]) / 1000

        plt.hist(H, bins=20, density=True,
                 alpha=0.4, label=profile)

    plt.xlabel("Altura máxima (km)")
    plt.ylabel("Densidad")
    plt.title("Distribución de altura máxima por perfil")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ================================================================
# 5️⃣ Pluma de trayectorias
# ================================================================
def plot_plumes(mc_results_by_profile):

    for profile, results in mc_results_by_profile.items():

        plt.figure(figsize=(8, 6))

        for r in results:
            if "time" not in r or "height" not in r:
                continue

            plt.plot(r["time"],
                     np.array(r["height"]) / 1000,
                     alpha=0.2)

        plt.xlabel("Tiempo (s)")
        plt.ylabel("Altura (km)")
        plt.title(f"Pluma de trayectorias — {profile}")
        plt.grid(alpha=0.3)
        plt.show()


# ================================================================
# DATOS DIGITALIZADOS (STRONG)
# ================================================================
datos_strong = { "T-13_D10": [ [66.32, 6.50], [189.96, 6.56], [313.54, 6.69], [436.95, 6.99], [560.31, 7.35], [683.71, 7.67], [807.16, 7.93], [930.91, 7.88], [1054.80, 7.67], [1178.86, 7.29], [1303.03, 6.79], [1427.43, 6.05], [1535.05, 5.21], [1620.20, 4.35], [1694.11, 3.49], [1756.78, 2.62], [1808.21, 1.75], [1854.05, 0.84], [1879.06, 0.16] ], "T-13_D7.5": [ [54.96, 6.63], [178.37, 6.93], [301.81, 7.20], [425.17, 7.56], [548.56, 7.89], [672.05, 8.10], [795.72, 8.14], [913.87, 8.06], [1054.97, 7.49], [1179.05, 7.08], [1303.30, 6.51], [1422.07, 5.76], [1524.07, 4.92], [1614.95, 3.96], [1688.89, 3.06], [1745.94, 2.19], [1797.41, 1.28] ], "T-13_D5": [ [65.91, 6.94], [189.16, 7.41], [315.01, 7.69], [435.86, 8.15], [559.40, 8.31], [683.08, 8.34], [750.56, 8.33], [976.16, 7.58], [1100.24, 7.18], [1224.42, 6.67], [1348.78, 5.97], [1462.03, 5.13], [1552.81, 4.26], [1626.72, 3.40], [1689.39, 2.54], [1746.53, 1.56], [1798.08, 0.56] ], "T-8_D5": [ [89.12, 6.17], [216.76, 6.76], [342.15, 7.35], [458.59, 7.89], [575.71, 8.25], [705.42, 8.50], [784.14, 8.49], [975.81, 7.95], [1100.24, 7.18], [1224.42, 6.67], [1348.78, 5.97], [1462.03, 5.13], [1552.81, 4.26], [1626.72, 3.40], [1689.39, 2.54], [1746.53, 1.56], [1798.08, 0.56] ], "T-8_D7.5": [ [66.94, 5.84], [190.39, 6.11], [313.73, 6.49], [459.20, 7.25], [582.44, 7.73], [818.00, 8.35], [941.84, 8.21], [1065.86, 7.87], [1189.96, 7.44], [1291.59, 6.99], [1348.78, 5.97], [1462.03, 5.13], [1552.81, 4.26], [1626.72, 3.40], [1689.39, 2.54], [1746.53, 1.56], [1798.08, 0.56] ] }


# ================================================================
# COMPARACIÓN COMPLETA (FIX FINAL)
# ================================================================
def plot_comparacion_completa(sim_res, ref_data, label="", save_path=None):

    t_sim = np.array(sim_res["time"])
    h_sim = np.array(sim_res["height"]) / 1000.0  # m → km

    ref_data = np.array(ref_data)
    t_ref = ref_data[:, 0]
    h_ref = ref_data[:, 1]  # ya en km

    # 🔥 Ordenar por seguridad
    order = np.argsort(t_ref)
    t_ref = t_ref[order]
    h_ref = h_ref[order]

    # Interpolación
    h_interp = np.interp(t_sim, t_ref, h_ref)
    diff = h_sim - h_interp
    rmse = np.sqrt(np.mean(diff**2))

    # Plot
    fig, ax1 = plt.subplots(figsize=(8,5))

    ax1.plot(t_sim, h_sim, label="Simulación", linewidth=2)
    ax1.scatter(t_ref, h_ref, label="Paper", s=25)

    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Altura (km)")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t_sim, diff, linestyle="--", alpha=0.8,
             label=f"Δ Altura (RMSE={rmse:.3f} km)")
    ax2.set_ylabel("Δ Altura (km)")
    ax2.legend(loc="upper right")

    plt.title(f"Simulación vs Paper — {label}")

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

    return rmse


# ================================================================
# EJECUCIÓN COMPLETA
# ================================================================
def ejecutar_comparacion_completa(determinista_results, save_dir="graphs"):

    os.makedirs(save_dir, exist_ok=True)

    print("\n===== COMPARACIÓN STRONG (VALIDACIÓN) =====")

    mapping = {
        "T-13_D10": "D0_10.0_T260",
        "T-13_D7.5": "D0_7.5_T260",
        "T-13_D5": "D0_5.0_T260",
        "T-8_D5": "D0_5.0_T265",
        "T-8_D7.5": "D0_7.5_T265",
    }

    resultados_rmse = {}

    for paper_key, sim_key in mapping.items():

        if sim_key not in determinista_results:
            print(f"[WARNING] Falta {sim_key}")
            continue

        print(f"\nProcesando: {paper_key}")

        sim_res = determinista_results[sim_key]
        ref_data = datos_strong[paper_key]

        rmse = plot_comparacion_completa(
            sim_res,
            ref_data,
            label=paper_key,
            save_path=f"{save_dir}/comparacion_{paper_key}.png"
        )

        resultados_rmse[paper_key] = rmse

    print("\n===== RESUMEN RMSE =====")
    for k, v in resultados_rmse.items():
        print(f"{k:12s} : {v:.3f} km")

    return resultados_rmse

def analyze_all_profiles(mc_results_by_profile):

    print("\n===== ESTADÍSTICAS MONTE CARLO =====")

    summary = {}

    for profile, results in mc_results_by_profile.items():

        stats, D, H = compute_profile_statistics(results)

        summary[profile] = stats

        print(f"\n--- {profile.upper()} ---")
        print(f"D_mean     = {stats['D_mean']:.3f} mm")
        print(f"D_std      = {stats['D_std']:.3f} mm")
        print(f"D_skew     = {stats['D_skew']:.3f}")
        print(f"H_mean     = {stats['H_mean']:.3f} km")
        print(f"H_std      = {stats['H_std']:.3f} km")
        print(f"v_mean     = {stats['v_mean']:.3f} m/s")
        print(f"Bias Beard = {stats['bias_beard']:.3f} m/s")
        print(f"RMSE Beard = {stats['rmse_beard']:.3f} m/s")

    return summary