#!/usr/bin/env python3
"""Compara ambientes strong, dry y weak con HAILCAST 2D."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cloud_model import CloudProfile
from hailcast2d.diagnostics import summarize_ensemble, summarize_grid
from hailcast2d.ensemble import run_ensemble
from hailcast2d.impact_grid import build_impact_grid
from hailcast2d.wind_profile import HorizontalWindProfile


ENVIRONMENTS = ("strong", "dry", "weak")
COLORS = {
    "strong": "tab:red",
    "dry": "tab:orange",
    "weak": "tab:blue",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compara tres ambientes de nube con HAILCAST 2D."
    )
    parser.add_argument("--n-members", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=3600.0)
    parser.add_argument("--cell-size-m", type=float, default=1000.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hailcast2d_comparison"),
    )
    parser.add_argument("--max-plotted-trajectories", type=int, default=30)
    return parser.parse_args()


def create_cloud(environment):
    return CloudProfile(
        updraft_type=environment,
        z_min=0.0,
        z_max=12000.0,
        z_center=6000.0,
    )


def create_wind():
    return HorizontalWindProfile(
        heights_m=[0.0, 3000.0, 6000.0, 9000.0, 12000.0],
        speeds_ms=[3.0, 8.0, 15.0, 22.0, 28.0],
    )


def run_experiments(args):
    results = {}
    wind = create_wind()

    for environment in ENVIRONMENTS:
        print(f"Simulando ambiente {environment}...")

        ensemble = run_ensemble(
            profile=create_cloud(environment),
            wind_profile=wind,
            n_members=args.n_members,
            D0_mm=7.5,
            z0=6000.0,
            T0_K=260.15,
            diameter_std_mm=0.5,
            x0_std_m=100.0,
            vx0_std_ms=0.5,
            wind_std_ms=1.0,
            seed=args.seed,
            dt=args.dt,
            t_max=args.t_max,
        )

        results[environment] = {
            "ensemble": ensemble,
            "ensemble_summary": summarize_ensemble(ensemble),
        }

        print(
            f"  Impactos: {ensemble['n_impacted']}/"
            f"{ensemble['n_simulated']}"
        )

    return results


def common_grid_bounds(results, cell_size_m):
    x_values = [
        impact["x_m"]
        for environment in ENVIRONMENTS
        for impact in results[environment]["ensemble"]["impacts"]
    ]

    if not x_values:
        raise RuntimeError(
            "Ninguna piedra alcanzó el suelo; no puede crearse la grilla."
        )

    x_min = np.floor(min(x_values) / cell_size_m) * cell_size_m
    x_max = np.ceil(max(x_values) / cell_size_m) * cell_size_m

    if x_max <= x_min:
        x_max = x_min + cell_size_m

    return float(x_min), float(x_max)


def build_grids(results, args):
    x_min, x_max = common_grid_bounds(results, args.cell_size_m)

    for environment in ENVIRONMENTS:
        ensemble = results[environment]["ensemble"]

        grid = build_impact_grid(
            impacts=ensemble["impacts"],
            cell_size_m=args.cell_size_m,
            x_min_m=x_min,
            x_max_m=x_max,
            n_simulated=ensemble["n_simulated"],
        )

        results[environment]["grid"] = grid
        results[environment]["grid_summary"] = summarize_grid(
            grid,
            expected_n_impacts=ensemble["n_impacted"],
        )


def write_impacts_csv(results, output_dir):
    path = output_dir / "individual_impacts.csv"
    fields = [
        "environment",
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

        for environment in ENVIRONMENTS:
            for impact in results[environment]["ensemble"]["impacts"]:
                writer.writerow({
                    field: (
                        environment
                        if field == "environment"
                        else impact[field]
                    )
                    for field in fields
                })


def write_summary_csv(results, output_dir):
    path = output_dir / "environment_summary.csv"
    fields = [
        "environment",
        "n_simulated",
        "n_impacted",
        "impact_fraction",
        "mean_flight_time_s",
        "mean_x_m",
        "mean_diameter_mm",
        "max_diameter_mm",
        "mean_speed_ms",
        "mean_energy_J",
        "max_energy_J",
        "occupied_cells",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for environment in ENVIRONMENTS:
            summary = results[environment]["ensemble_summary"]
            stats = summary["statistics"]
            grid_summary = results[environment]["grid_summary"]

            writer.writerow({
                "environment": environment,
                "n_simulated": summary["n_simulated"],
                "n_impacted": summary["n_impacted"],
                "impact_fraction": summary["impact_fraction"],
                "mean_flight_time_s":
                    stats["flight_time_s"]["mean"],
                "mean_x_m": stats["impact_x_m"]["mean"],
                "mean_diameter_mm":
                    stats["impact_diameter_mm"]["mean"],
                "max_diameter_mm":
                    stats["impact_diameter_mm"]["max"],
                "mean_speed_ms":
                    stats["impact_speed_ms"]["mean"],
                "mean_energy_J":
                    stats["impact_energy_J"]["mean"],
                "max_energy_J":
                    stats["impact_energy_J"]["max"],
                "occupied_cells":
                    grid_summary["n_occupied_cells"],
            })


def write_diagnostics_json(results, args):
    path = args.output_dir / "diagnostics.json"

    serializable = {
        "configuration": {
            "n_members": args.n_members,
            "seed": args.seed,
            "dt_s": args.dt,
            "t_max_s": args.t_max,
            "cell_size_m": args.cell_size_m,
            "D0_mm": 7.5,
            "z0_m": 6000.0,
            "T0_K": 260.15,
        },
        "environments": {
            environment: {
                "ensemble": results[environment]["ensemble_summary"],
                "grid": results[environment]["grid_summary"],
            }
            for environment in ENVIRONMENTS
        },
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, ensure_ascii=False)


def plot_trajectories(results, args):
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 5), sharex=True, sharey=True
    )

    for axis, environment in zip(axes, ENVIRONMENTS):
        trajectories = results[environment]["ensemble"]["trajectories"]
        plotted = trajectories[:args.max_plotted_trajectories]

        for trajectory in plotted:
            axis.plot(
                np.asarray(trajectory["x"]) / 1000.0,
                np.asarray(trajectory["height"]) / 1000.0,
                color=COLORS[environment],
                alpha=0.35,
                linewidth=0.8,
            )

        axis.set_title(environment.capitalize())
        axis.set_xlabel("Distancia horizontal (km)")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Altura (km)")
    figure.suptitle("Trayectorias de granizo en el plano x-z")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "trajectories_xz.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_impact_counts(results, args):
    figure, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True, sharey=True
    )

    for axis, environment in zip(axes, ENVIRONMENTS):
        grid = results[environment]["grid"]

        axis.bar(
            np.asarray(grid["cell_centers_m"]) / 1000.0,
            grid["impact_count"],
            width=grid["cell_size_m"] / 1000.0,
            color=COLORS[environment],
            alpha=0.75,
        )
        axis.set_ylabel(environment.capitalize())
        axis.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("Distancia horizontal de impacto (km)")
    figure.suptitle("Distribución horizontal de impactos")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "impact_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_impact_properties(results, args):
    diameters = [
        [
            impact["diameter_mm"]
            for impact in results[environment]["ensemble"]["impacts"]
        ]
        for environment in ENVIRONMENTS
    ]
    energies = [
        [
            impact["kinetic_energy_J"]
            for impact in results[environment]["ensemble"]["impacts"]
        ]
        for environment in ENVIRONMENTS
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].boxplot(diameters, labels=ENVIRONMENTS, showmeans=True)
    axes[0].set_ylabel("Diámetro de impacto (mm)")
    axes[0].set_title("Diámetro final")

    axes[1].boxplot(energies, labels=ENVIRONMENTS, showmeans=True)
    axes[1].set_ylabel("Energía cinética (J)")
    axes[1].set_title("Energía de impacto")

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        args.output_dir / "impact_properties.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    args = parse_arguments()

    if args.n_members <= 0:
        raise ValueError("--n-members debe ser mayor que cero.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiments(args)
    build_grids(results, args)

    write_impacts_csv(results, args.output_dir)
    write_summary_csv(results, args.output_dir)
    write_diagnostics_json(results, args)

    plot_trajectories(results, args)
    plot_impact_counts(results, args)
    plot_impact_properties(results, args)

    print(f"\nResultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()