#!/usr/bin/env python3
# ================================================================
# run_cordoba_20210127.py
# ================================================================
"""
Simulación HAILCAST 2D del caso de granizo de Córdoba del 27/01/2021.

El programa lee el radiosondeo de Córdoba Aero de las 12 UTC desde el
archivo histórico NOAA/NCEI IGRA, construye el perfil de nube observado,
ejecuta un ensamble reproducible de trayectorias x-z y genera figuras
listas para una presentación.

La simulación 2D usa la componente zonal ``u`` del viento. Las condiciones
iniciales del granizo conservan las utilizadas en los experimentos previos
del proyecto y deben interpretarse como una primera configuración de caso,
no como una calibración definitiva.

Autor: Richard Perdomo Arcila
Fecha de desarrollo: 4 de agosto de 2026
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hailcast2d.diagnostics import summarize_ensemble
from hailcast2d.ensemble import run_ensemble
from hailcast2d.sounding_cloud import SoundingCloudProfile
from hailcast2d.sounding_reader import read_igra_sounding


CASE_DATE = "2021-01-27 12 UTC"
STATION_ID = "ARM00087344"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Simula trayectorias de granizo con el radiosondeo de "
            "Córdoba Aero del 27/01/2021 a las 12 UTC."
        )
    )
    parser.add_argument(
        "--igra-file",
        type=Path,
        default=Path("data/igra/ARM00087344-data.txt"),
    )
    parser.add_argument("--n-members", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diameter-mm", type=float, default=7.5)
    parser.add_argument("--diameter-std-mm", type=float, default=0.5)
    parser.add_argument("--z0-m", type=float, default=6000.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=3600.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cordoba_20210127_observed"),
    )
    parser.add_argument("--max-plotted-trajectories", type=int, default=30)
    return parser.parse_args()


def load_observed_case(path):
    sounding = read_igra_sounding(
        path,
        station_id=STATION_ID,
        year=2021,
        month=1,
        day=27,
        hour=12,
    )
    cloud = SoundingCloudProfile(sounding)
    wind = cloud.wind_profile(component="u")
    return sounding, cloud, wind


def run_case(args, cloud, wind):
    if args.n_members <= 0:
        raise ValueError("--n-members debe ser mayor que cero.")
    if args.z0_m <= cloud.z_min or args.z0_m >= cloud.z_max:
        raise ValueError(
            "--z0-m debe encontrarse dentro de la nube observada."
        )

    # La temperatura inicial se toma directamente del ambiente observado
    # en la altura de inicio, en lugar de imponer un valor independiente.
    initial_temperature_k = cloud.temperature(args.z0_m)

    ensemble = run_ensemble(
        profile=cloud,
        wind_profile=wind,
        n_members=args.n_members,
        D0_mm=args.diameter_mm,
        z0=args.z0_m,
        T0_K=initial_temperature_k,
        diameter_std_mm=args.diameter_std_mm,
        x0_std_m=100.0,
        vx0_std_ms=0.5,
        wind_std_ms=1.0,
        seed=args.seed,
        dt=args.dt,
        t_max=args.t_max,
    )

    return ensemble, initial_temperature_k


def reference_levels_agl(sounding, cloud):
    surface = cloud.surface_elevation_m
    return {
        "LCL": sounding.lcl_height_m - surface,
        "LFC": sounding.lfc_height_m - surface,
        "0 °C": sounding.freezing_level_m - surface,
        "EL": sounding.el_height_m - surface,
    }


def apply_presentation_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.titlesize": 17,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def add_reference_levels(axis, levels, show_labels=False):
    colors = {
        "LCL": "#8c8c8c",
        "LFC": "#e67e22",
        "0 °C": "#2980b9",
        "EL": "#8e44ad",
    }

    for name, height_m in levels.items():
        axis.axhline(
            height_m / 1000.0,
            color=colors[name],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=name if show_labels else None,
        )


def plot_cloud_profiles(sounding, cloud, output_dir):
    apply_presentation_style()
    levels = reference_levels_agl(sounding, cloud)
    height_m = np.linspace(0.0, cloud.z_max, 500)
    height_km = height_m / 1000.0

    environment_c = np.asarray(cloud.temperature(height_m)) - 273.15
    parcel_c = np.interp(
        height_m,
        cloud.height_agl_m,
        sounding.parcel_temperature_c,
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 7.2), sharey=True)

    axes[0].plot(environment_c, height_km, label="Ambiente", linewidth=2.2)
    axes[0].plot(parcel_c, height_km, label="Parcela", linewidth=2.2)
    axes[0].axvline(0.0, color="#2980b9", linewidth=1.3, alpha=0.8)
    axes[0].set_xlabel("Temperatura (°C)")
    axes[0].set_ylabel("Altura AGL (km)")
    axes[0].set_title("Estructura térmica")
    axes[0].legend(loc="best")

    axes[1].plot(
        cloud.w(height_m),
        height_km,
        color="#c0392b",
        linewidth=2.4,
    )
    axes[1].axvline(0.0, color="0.35", linewidth=1.0)
    axes[1].set_xlabel("Ascenso efectivo (m/s)")
    axes[1].set_title("Corriente ascendente")

    axes[2].plot(
        cloud.lwc(height_m) * 1000.0,
        height_km,
        color="#16a085",
        linewidth=2.4,
    )
    axes[2].set_xlabel("Agua líquida (g/m³)")
    axes[2].set_title("Agua disponible")

    for index, axis in enumerate(axes):
        add_reference_levels(axis, levels, show_labels=index == 2)
        axis.grid(alpha=0.22)
        axis.set_ylim(0.0, cloud.z_max / 1000.0 + 0.3)

    axes[2].legend(loc="best")
    figure.suptitle(
        "Nube derivada del radiosondeo — Córdoba Aero, 27/01/2021"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "01_observed_cloud_profiles.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_trajectories(ensemble, sounding, cloud, args):
    apply_presentation_style()
    levels = reference_levels_agl(sounding, cloud)
    trajectories = ensemble["trajectories"]
    plotted = trajectories[:args.max_plotted_trajectories]

    figure, axis = plt.subplots(figsize=(12.8, 7.2))

    for trajectory in plotted:
        axis.plot(
            np.asarray(trajectory["x"]) / 1000.0,
            np.asarray(trajectory["height"]) / 1000.0,
            color="#2874a6",
            alpha=0.28,
            linewidth=1.0,
        )

    # Se destaca una trayectoria central para explicar el ciclo vertical.
    representative = trajectories[len(trajectories) // 2]
    axis.plot(
        np.asarray(representative["x"]) / 1000.0,
        np.asarray(representative["height"]) / 1000.0,
        color="#17202a",
        linewidth=2.2,
        label="Trayectoria representativa",
    )

    add_reference_levels(axis, levels, show_labels=True)
    axis.scatter(
        [0.0],
        [args.z0_m / 1000.0],
        marker="o",
        s=70,
        color="#f1c40f",
        edgecolor="black",
        zorder=5,
        label="Inicio",
    )
    axis.set_xlabel("Desplazamiento zonal (km)")
    axis.set_ylabel("Altura AGL (km)")
    axis.set_title(
        f"Trayectorias HAILCAST 2D — {args.n_members} miembros"
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="best", ncol=2)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "02_hail_trajectories_xz.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_impacts(ensemble, args):
    apply_presentation_style()
    impacts = ensemble["impacts"]

    if not impacts:
        return

    diameter = np.asarray([item["diameter_mm"] for item in impacts])
    energy = np.asarray([item["kinetic_energy_J"] for item in impacts])
    distance_km = np.asarray([item["x_m"] for item in impacts]) / 1000.0

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))

    axes[0].hist(
        diameter,
        bins="auto",
        color="#2874a6",
        edgecolor="white",
        alpha=0.88,
    )
    axes[0].axvline(
        np.mean(diameter),
        color="#c0392b",
        linestyle="--",
        linewidth=2.0,
        label=f"Media: {np.mean(diameter):.1f} mm",
    )
    axes[0].set_xlabel("Diámetro al impactar (mm)")
    axes[0].set_ylabel("Número de impactos")
    axes[0].set_title("Distribución de tamaños")
    axes[0].legend()

    scatter = axes[1].scatter(
        distance_km,
        energy,
        c=diameter,
        cmap="viridis",
        s=38,
        alpha=0.78,
        edgecolor="none",
    )
    axes[1].set_xlabel("Desplazamiento zonal al impactar (km)")
    axes[1].set_ylabel("Energía cinética (J)")
    axes[1].set_title("Propiedades de impacto")
    colorbar = figure.colorbar(scatter, ax=axes[1])
    colorbar.set_label("Diámetro (mm)")

    for axis in axes:
        axis.grid(alpha=0.22)

    figure.suptitle("Impactos simulados — caso Córdoba 27/01/2021")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "03_impact_properties.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_impacts_csv(ensemble, output_dir):
    path = output_dir / "individual_impacts.csv"
    fields = [
        "member_id",
        "time_s",
        "x_m",
        "diameter_mm",
        "speed_ms",
        "angle_from_horizontal_deg",
        "kinetic_energy_J",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for impact in ensemble["impacts"]:
            writer.writerow({field: impact[field] for field in fields})


def write_summary(
    ensemble,
    sounding,
    cloud,
    initial_temperature_k,
    args,
):
    ensemble_summary = summarize_ensemble(ensemble)
    payload = {
        "case": {
            "station_id": STATION_ID,
            "sounding_datetime": CASE_DATE,
            "event_datetime": "2021-01-27 17-18 UTC",
            "model_dimension": "2D_xz",
            "wind_component": "u",
        },
        "configuration": {
            "n_members": args.n_members,
            "seed": args.seed,
            "diameter_mm": args.diameter_mm,
            "diameter_std_mm": args.diameter_std_mm,
            "z0_m_agl": args.z0_m,
            "initial_temperature_k": initial_temperature_k,
            "dt_s": args.dt,
            "t_max_s": args.t_max,
        },
        "sounding": sounding.summary(),
        "cloud": cloud.summary(),
        "ensemble": ensemble_summary,
    }

    with (args.output_dir / "case_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main():
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sounding, cloud, wind = load_observed_case(args.igra_file)
    ensemble, initial_temperature_k = run_case(args, cloud, wind)

    plot_cloud_profiles(sounding, cloud, args.output_dir)
    plot_trajectories(ensemble, sounding, cloud, args)
    plot_impacts(ensemble, args)
    write_impacts_csv(ensemble, args.output_dir)
    write_summary(
        ensemble,
        sounding,
        cloud,
        initial_temperature_k,
        args,
    )

    print("\nCaso Córdoba 27/01/2021 completado")
    print(
        f"Impactos: {ensemble['n_impacted']}/"
        f"{ensemble['n_simulated']}"
    )
    print(f"Resultados: {args.output_dir}")


if __name__ == "__main__":
    main()