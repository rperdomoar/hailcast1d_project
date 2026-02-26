import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FUNCIÓN ORIGINAL (sin cambios)
# ============================================================
def plot_hailstone_trajectories(determinista_results, montecarlo_results=None,
                                 title='Trayectorias de granizo', save_path=None):

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(determinista_results)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    for i, (label, res) in enumerate(determinista_results.items()):
        plt.plot(res["time"], res["height"],
                 linewidth=2,
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)],
                 markerfacecolor='none',
                 markeredgecolor=colors[i % len(colors)],
                 markersize=6,
                 markevery=max(len(res["time"]) // 25, 1),
                 label=f"{label} — D0={res['D0_init']:.1f} mm")

    if montecarlo_results is not None:
        for res in montecarlo_results:
            plt.plot(res["time"], res["height"],
                     alpha=0.4,
                     linewidth=0.8,
                     color="gray")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Altura (km)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    if montecarlo_results is not None:
        plt.legend(handles[:len(determinista_results)], labels[:len(determinista_results)])
    else:
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

# ============================================================
# DATOS DIGITALIZADOS (STRONG)
# ============================================================
datos_strong = {
    "T-13_D10": [
        [66.32, 6500], [189.96, 6560], [313.54, 6690], [436.95, 6990],
        [560.31, 7350], [683.71, 7670], [807.16, 7930], [930.91, 7880],
        [1054.80, 7670], [1178.86, 7290], [1303.03, 6790], [1427.43, 6050],
        [1535.05, 5210], [1620.20, 4350], [1694.11, 3490], [1756.78, 2620],
        [1808.21, 1750], [1854.05, 84], [1879.06, 16]
    ],

    "T-13_D7.5": [
        [54.96, 6630], [178.37, 6930], [301.81, 7200], [425.17, 7560],
        [548.56, 7890], [672.05, 8100], [795.72, 8140], [913.87, 8060],
        [1054.97, 7490], [1179.05, 7080], [1303.30, 6510], [1422.07, 5760],
        [1524.07, 4920], [1614.95, 3960], [1688.89, 3060], [1745.94, 219],
        [1797.41, 128]
    ],

    "T-13_D5": [
        [65.91, 6940], [189.16, 7410], [315.01, 7690], [435.86, 8150],
        [559.40, 8310], [683.08, 8340], [750.56, 8330], [976.16, 7580],
        [1100.24, 7180], [1224.42, 6670], [1348.78, 5970], [1462.03, 5130],
        [1552.81, 4260], [1626.72, 3400], [1689.39, 2540], [1746.53, 1560],
        [1798.08, 56]
    ],

    "T-8_D5": [
        [89.12, 6170], [216.76, 6760], [342.15, 7350], [458.59, 7890],
        [575.71, 8250], [705.42, 8500], [784.14, 8490], [975.81, 7950],
        [1100.24, 7180], [1224.42, 6670], [1348.78, 5970], [1462.03, 5130],
        [1552.81, 4260], [1626.72, 3400], [1689.39, 2540], [1746.53, 1560],
        [1798.08, 56]
    ],

    "T-8_D7.5": [
        [66.94, 5840], [190.39, 6110], [313.73, 6490], [459.20, 7250],
        [582.44, 7730], [818.00, 8350], [941.84, 8210], [1065.86, 7870],
        [1189.96, 7440], [1291.59, 6990], [1348.78, 5970], [1462.03, 5130],
        [1552.81, 4260], [1626.72, 3400], [1689.39, 2540], [1746.53, 1560],
        [1798.08, 56]
    ]
}

# ============================================================
# FUNCIONES DE COMPARACIÓN
# ============================================================
def interpolar_referencia(t_sim, t_ref, h_ref):
    return np.interp(t_sim, t_ref, h_ref)

def calcular_rmse(diff):
    return np.sqrt(np.mean(diff**2))

def comparar_altura(sim_res, ref_data, label="", save_path=None):

    t_sim = np.array(sim_res["time"])
    h_sim = np.array(sim_res["height"])

    ref_data = np.array(ref_data)
    t_ref = ref_data[:, 0]
    h_ref = ref_data[:, 1]

    h_interp = interpolar_referencia(t_sim, t_ref, h_ref)
    diff = h_sim - h_interp

    rmse = calcular_rmse(diff)

    plt.figure(figsize=(8,5))
    plt.plot(t_sim, diff, label=f"{label} (RMSE={rmse:.3f} km)")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Δ Altura (km)")
    plt.title("Diferencia Simulación - Referencia")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

    return rmse

# ============================================================
# FUNCIÓN AUTOMÁTICA (LO IMPORTANTE 🔥)
# ============================================================
def ejecutar_comparacion_completa(determinista_results, save_dir="graphs"):

    print("\n===== COMPARACIÓN STRONG =====")

    resultados_rmse = {}

    for key, ref_data in datos_strong.items():

        if key not in determinista_results:
            print(f"[WARNING] No existe {key} en simulación")
            continue

        print(f"\nProcesando: {key}")

        rmse = comparar_altura(
            determinista_results[key],
            ref_data,
            label=key,
            save_path=f"{save_dir}/diff_{key}.png"
        )

        resultados_rmse[key] = rmse

    print("\n===== RESUMEN RMSE =====")
    for k, v in resultados_rmse.items():
        print(f"{k}: {v:.3f} km")

    return resultados_rmse