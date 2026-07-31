# ================================================================
# hailcast2d/impact_grid.py
# ================================================================
"""Agregación espacial de impactos de HAILCAST 2D."""

from __future__ import annotations

import numpy as np


_REQUIRED_FIELDS = ("x_m", "diameter_mm", "kinetic_energy_J")


def _validate_impacts(impacts):
    """Valida y convierte los impactos en arreglos numéricos."""

    if impacts is None:
        raise TypeError("impacts no puede ser None.")

    impacts = list(impacts)

    x_values = []
    diameters = []
    energies = []

    for index, impact in enumerate(impacts):
        if not isinstance(impact, dict):
            raise TypeError(
                f"El impacto {index} debe representarse mediante un diccionario."
            )

        missing = [
            field for field in _REQUIRED_FIELDS if field not in impact
        ]

        if missing:
            raise ValueError(
                f"El impacto {index} no contiene los campos: "
                + ", ".join(missing)
            )

        values = [float(impact[field]) for field in _REQUIRED_FIELDS]

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"El impacto {index} contiene valores no finitos."
            )

        if values[1] < 0.0:
            raise ValueError("El diámetro de impacto no puede ser negativo.")

        if values[2] < 0.0:
            raise ValueError(
                "La energía cinética de impacto no puede ser negativa."
            )

        x_values.append(values[0])
        diameters.append(values[1])
        energies.append(values[2])

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(diameters, dtype=float),
        np.asarray(energies, dtype=float),
    )


def _resolve_bounds(x, cell_size_m, x_min_m, x_max_m):
    """Determina los límites del dominio horizontal."""

    if x_min_m is None:
        if x.size == 0:
            raise ValueError(
                "x_min_m es obligatorio cuando no existen impactos."
            )
        x_min_m = np.floor(np.min(x) / cell_size_m) * cell_size_m

    if x_max_m is None:
        if x.size == 0:
            raise ValueError(
                "x_max_m es obligatorio cuando no existen impactos."
            )
        x_max_m = np.ceil(np.max(x) / cell_size_m) * cell_size_m

        if x_max_m <= x_min_m:
            x_max_m = x_min_m + cell_size_m

    x_min_m = float(x_min_m)
    x_max_m = float(x_max_m)

    if not np.isfinite(x_min_m) or not np.isfinite(x_max_m):
        raise ValueError("Los límites del dominio deben ser finitos.")

    if x_max_m <= x_min_m:
        raise ValueError("x_max_m debe ser mayor que x_min_m.")

    n_cells_float = (x_max_m - x_min_m) / cell_size_m
    n_cells = int(np.ceil(n_cells_float))

    edges = x_min_m + np.arange(n_cells + 1) * cell_size_m

    return x_min_m, float(edges[-1]), edges


def build_impact_grid(
    impacts,
    cell_size_m: float,
    x_min_m: float | None = None,
    x_max_m: float | None = None,
    n_simulated: int | None = None,
):
    """
    Agrega impactos sobre celdas horizontales regulares.

    Los intervalos son cerrados a la izquierda y abiertos a la derecha,
    excepto la última celda, que incluye también el borde final.
    """

    cell_size_m = float(cell_size_m)

    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m debe ser finito y mayor que cero.")

    x, diameters, energies = _validate_impacts(impacts)

    x_min_m, x_max_m, edges = _resolve_bounds(
        x=x,
        cell_size_m=cell_size_m,
        x_min_m=x_min_m,
        x_max_m=x_max_m,
    )

    if n_simulated is None:
        n_simulated = int(x.size)

    if (
        not isinstance(n_simulated, (int, np.integer))
        or n_simulated <= 0
    ):
        raise ValueError("n_simulated debe ser un entero mayor que cero.")

    if n_simulated < x.size:
        raise ValueError(
            "n_simulated no puede ser menor que el número de impactos."
        )

    inside = (x >= x_min_m) & (x <= x_max_m)
    excluded_impacts = int(np.count_nonzero(~inside))

    x = x[inside]
    diameters = diameters[inside]
    energies = energies[inside]

    n_cells = edges.size - 1
    cell_index = np.searchsorted(edges, x, side="right") - 1
    cell_index = np.clip(cell_index, 0, n_cells - 1)

    impact_count = np.zeros(n_cells, dtype=int)
    diameter_sum = np.zeros(n_cells, dtype=float)
    diameter_max = np.zeros(n_cells, dtype=float)
    energy_sum = np.zeros(n_cells, dtype=float)
    energy_max = np.zeros(n_cells, dtype=float)

    for index, diameter, energy in zip(
        cell_index, diameters, energies
    ):
        impact_count[index] += 1
        diameter_sum[index] += diameter
        diameter_max[index] = max(diameter_max[index], diameter)
        energy_sum[index] += energy
        energy_max[index] = max(energy_max[index], energy)

    occupied = impact_count > 0

    diameter_mean = np.zeros(n_cells, dtype=float)
    energy_mean = np.zeros(n_cells, dtype=float)

    diameter_mean[occupied] = (
        diameter_sum[occupied] / impact_count[occupied]
    )
    energy_mean[occupied] = (
        energy_sum[occupied] / impact_count[occupied]
    )

    return {
        "cell_edges_m": edges,
        "cell_centers_m": 0.5 * (edges[:-1] + edges[1:]),
        "cell_size_m": cell_size_m,
        "impact_count": impact_count,
        "impact_fraction": impact_count / n_simulated,
        "diameter_mean_mm": diameter_mean,
        "diameter_max_mm": diameter_max,
        "kinetic_energy_mean_J": energy_mean,
        "kinetic_energy_max_J": energy_max,
        "kinetic_energy_sum_J": energy_sum,
        "n_simulated": int(n_simulated),
        "n_impacts_inside": int(x.size),
        "n_impacts_excluded": excluded_impacts,
        "x_min_m": x_min_m,
        "x_max_m": x_max_m,
        "grid_dimension": "1D_ground_transect",
    }
