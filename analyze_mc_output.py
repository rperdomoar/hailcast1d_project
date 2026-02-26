import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

DATA_DIR = "./hailcast_output"
OUTPUT_DIR = "./mc_analysis"
PROFILES = ["strong", "dry", "weak"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Extraer datos
# ============================================================
def extract_final_diameters(profile):
    pattern = os.path.join(DATA_DIR, f"MC_{profile}_*.txt")
    files = glob(pattern)

    diam = []

    for f in files:
        data = np.loadtxt(f)
        if data.ndim == 2 and len(data) > 0:
            diam.append(data[-1, 2])  # columna diámetro final [mm]

    return np.array(diam)


# ============================================================
# Estadísticas
# ============================================================
def summarize(arr, profile):

    stats = {
        "N": len(arr),
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "q1": np.percentile(arr, 25),
        "q3": np.percentile(arr, 75),
        "max": np.max(arr),
    }

    print(f"\nProfile: {profile}")
    for k, v in stats.items():
        print(f"{k:>6} = {v:.3f}" if k != "N" else f"{k:>6} = {v}")

    return stats


# ============================================================
# Histogramas
# ============================================================
def plot_all_histograms(results_dict):

    plt.figure(figsize=(8,6))

    for profile, diam in results_dict.items():
        if len(diam) > 0:
            plt.hist(diam, bins=20, alpha=0.5, label=profile)

    plt.xlabel("Final diameter [mm]")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo - Final Diameter Distribution")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "hist_final_diameter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📊 Histograma guardado en {save_path}")


# ============================================================
# Boxplots comparativos
# ============================================================
def plot_boxplot(results_dict):

    profiles = []
    data = []

    for profile, values in results_dict.items():
        if len(values) > 0:
            profiles.append(profile)
            data.append(values)

    if len(data) == 0:
        return

    plt.figure(figsize=(8,6))
    plt.boxplot(data, labels=profiles, showfliers=True)

    plt.ylabel("Final diameter [mm]")
    plt.title("Monte Carlo - Final Diameter (Boxplot)")
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "boxplot_final_diameter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📦 Boxplot guardado en {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():

    results = {}
    stats_all = {}

    for profile in PROFILES:
        diam = extract_final_diameters(profile)

        if len(diam) == 0:
            print(f"No data for {profile}")
            continue

        stats_all[profile] = summarize(diam, profile)
        results[profile] = diam

    if len(results) == 0:
        print("No Monte Carlo data found.")
        return

    plot_all_histograms(results)
    plot_boxplot(results)

    print("\n✅ Análisis finalizado\n")


if __name__ == "__main__":
    main()
