# ================================================================
# hailcast2d/diagnostics.py
# ================================================================
"""Diagnósticos estadísticos para ensambles HAILCAST 2D."""

from __future__ import annotations

import numpy as np


IMPACT_VARIABLES = {
    "time_s": "flight_time_s",
    "x_m": "impact_x_m",
    "diameter_mm": "impact_diameter_mm",
    "speed_ms": "impact_speed_ms",
    "angle_from_horizontal_deg": "impact_angle_deg",
    "kinetic_energy_J": "impact_energy_J",
}


NON_IMPACT_VARIABLES = {
    "time_s": "final_time_s",
    "x_m": "final_x_m",
    "height_m": "final_height_m",
    "diameter_mm": "final_diameter_mm",
    "speed_ms": "final_speed_ms",
}


def descriptive_statistics(values):
    """Calcula estadísticos descriptivos para una variable."""

    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "Los valores deben formar un arreglo 1D."
        )

    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "p90": None,
            "max": None,
        }

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Los valores contienen elementos no finitos."
        )

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(
            np.percentile(values, 90.0)
        ),
        "max": float(np.max(values)),
    }


def _extreme_member(impacts, variable):
    """Identifica el miembro con el valor máximo de una variable."""

    if not impacts:
        return None

    values = np.asarray(
        [
            float(impact[variable])
            for impact in impacts
        ],
        dtype=float,
    )

    index = int(np.argmax(values))
    impact = impacts[index]

    return {
        "member_id": int(impact["member_id"]),
        "value": float(values[index]),
    }


def summarize_ensemble(ensemble):
    """
    Resume un resultado producido por ``run_ensemble``.

    Incluye:

    - conteos y fracción de impacto;
    - estadísticas de los miembros que impactaron;
    - estado final de los miembros que no impactaron;
    - motivos de terminación;
    - identificación de los impactos extremos.
    """

    if not isinstance(ensemble, dict):
        raise TypeError(
            "ensemble debe ser un diccionario."
        )

    required = {
        "trajectories",
        "impacts",
        "n_simulated",
        "n_impacted",
        "impact_fraction",
    }
    missing = required.difference(ensemble)

    if missing:
        raise ValueError(
            "Faltan campos del ensamble: "
            + ", ".join(sorted(missing))
        )

    trajectories = list(
        ensemble["trajectories"]
    )
    impacts = list(
        ensemble["impacts"]
    )
    n_simulated = int(
        ensemble["n_simulated"]
    )
    n_impacted = int(
        ensemble["n_impacted"]
    )

    if len(trajectories) != n_simulated:
        raise ValueError(
            "El número de trayectorias no coincide "
            "con n_simulated."
        )

    if len(impacts) != n_impacted:
        raise ValueError(
            "El número de impactos no coincide "
            "con n_impacted."
        )

    expected_fraction = (
        n_impacted / n_simulated
        if n_simulated > 0
        else 0.0
    )

    if not np.isclose(
        float(ensemble["impact_fraction"]),
        expected_fraction,
    ):
        raise ValueError(
            "impact_fraction no coincide con los conteos."
        )

    # ------------------------------------------------------------
    # Estadísticas de los miembros que impactaron
    # ------------------------------------------------------------
    statistics = {}

    for source_name, output_name in IMPACT_VARIABLES.items():
        values = []

        for index, impact in enumerate(impacts):
            if source_name not in impact:
                raise ValueError(
                    f"El impacto {index} no contiene "
                    f"{source_name}."
                )

            values.append(
                float(impact[source_name])
            )

        statistics[output_name] = (
            descriptive_statistics(values)
        )

    # ------------------------------------------------------------
    # Estados finales y motivos de terminación
    # ------------------------------------------------------------
    non_impacts = []
    termination_reasons = {}

    expected_not_impacted = (
        n_simulated - n_impacted
    )

    terminal_states_available = all(
        "impact" in trajectory
        for trajectory in trajectories
    )

    if terminal_states_available:
        for index, trajectory in enumerate(trajectories):
            terminal_state = trajectory["impact"]

            if "impacted_ground" not in terminal_state:
                raise ValueError(
                    f"La trayectoria {index} no declara "
                    "impacted_ground."
                )

            reason = terminal_state.get(
                "termination_reason",
                "unknown",
            )

            termination_reasons[reason] = (
                termination_reasons.get(reason, 0) + 1
            )

            if not terminal_state["impacted_ground"]:
                non_impacts.append(terminal_state)

        if len(non_impacts) != expected_not_impacted:
            raise ValueError(
                "El número de miembros sin impacto no coincide "
                "con los conteos del ensamble."
            )

    else:
        # Compatibilidad con ensambles antiguos o artificiales que
        # solamente incluyen las listas de trayectorias e impactos.
        if n_impacted > 0:
            termination_reasons["ground_impact"] = n_impacted

        if expected_not_impacted > 0:
            termination_reasons[
                "terminal_state_unavailable"
            ] = expected_not_impacted
    # ------------------------------------------------------------
    # Estadísticas de los miembros que no impactaron
    # ------------------------------------------------------------
    non_impact_statistics = {}

    for (
        source_name,
        output_name,
    ) in NON_IMPACT_VARIABLES.items():
        values = []

        for index, terminal_state in enumerate(non_impacts):
            if source_name not in terminal_state:
                raise ValueError(
                    "El miembro sin impacto "
                    f"{index} no contiene {source_name}."
                )

            values.append(
                float(terminal_state[source_name])
            )

        non_impact_statistics[output_name] = (
            descriptive_statistics(values)
        )

    return {
        "n_simulated": n_simulated,
        "n_impacted": n_impacted,
        "n_not_impacted": expected_not_impacted,
        "impact_fraction": expected_fraction,
        "statistics": statistics,
        "non_impact_statistics": non_impact_statistics,
        "termination_reasons": termination_reasons,
        "extreme_members": {
            "largest_displacement": _extreme_member(
                impacts,
                "x_m",
            ),
            "largest_diameter": _extreme_member(
                impacts,
                "diameter_mm",
            ),
            "largest_energy": _extreme_member(
                impacts,
                "kinetic_energy_J",
            ),
            "longest_flight": _extreme_member(
                impacts,
                "time_s",
            ),
        },
    }


def summarize_grid(
    grid,
    expected_n_impacts=None,
):
    """
    Resume una grilla creada por ``build_impact_grid``.

    Si se proporciona ``expected_n_impacts``, verifica la conservación
    entre el ensamble y la suma de impactos dentro y fuera del dominio.
    """

    if not isinstance(grid, dict):
        raise TypeError(
            "grid debe ser un diccionario."
        )

    required = {
        "impact_count",
        "diameter_max_mm",
        "kinetic_energy_sum_J",
        "n_impacts_inside",
        "n_impacts_excluded",
        "cell_size_m",
    }
    missing = required.difference(grid)

    if missing:
        raise ValueError(
            "Faltan campos de la grilla: "
            + ", ".join(sorted(missing))
        )

    counts = np.asarray(
        grid["impact_count"],
        dtype=int,
    )
    diameter_max = np.asarray(
        grid["diameter_max_mm"],
        dtype=float,
    )
    energy_sum = np.asarray(
        grid["kinetic_energy_sum_J"],
        dtype=float,
    )

    if counts.ndim != 1:
        raise ValueError(
            "impact_count debe ser un arreglo 1D."
        )

    if diameter_max.shape != counts.shape:
        raise ValueError(
            "diameter_max_mm no coincide "
            "con la forma de la grilla."
        )

    if energy_sum.shape != counts.shape:
        raise ValueError(
            "kinetic_energy_sum_J no coincide "
            "con la forma de la grilla."
        )

    n_inside = int(
        grid["n_impacts_inside"]
    )
    n_excluded = int(
        grid["n_impacts_excluded"]
    )
    counted = int(
        np.sum(counts)
    )

    if counted != n_inside:
        raise ValueError(
            "La suma de impact_count no coincide "
            "con n_impacts_inside."
        )

    conservation_ok = True

    if expected_n_impacts is not None:
        expected_n_impacts = int(
            expected_n_impacts
        )
        conservation_ok = (
            n_inside + n_excluded
            == expected_n_impacts
        )

        if not conservation_ok:
            raise ValueError(
                "La grilla no conserva el número "
                "esperado de impactos."
            )

    occupied = counts > 0

    return {
        "n_cells": int(counts.size),
        "n_occupied_cells": int(
            np.count_nonzero(occupied)
        ),
        "occupied_fraction": (
            float(np.mean(occupied))
            if counts.size
            else 0.0
        ),
        "n_impacts_inside": n_inside,
        "n_impacts_excluded": n_excluded,
        "maximum_cell_count": (
            int(np.max(counts))
            if counts.size
            else 0
        ),
        "maximum_diameter_mm": (
            float(np.max(diameter_max))
            if counts.size
            else 0.0
        ),
        "total_kinetic_energy_J": float(
            np.sum(energy_sum)
        ),
        "cell_size_m": float(
            grid["cell_size_m"]
        ),
        "impact_conservation_ok": conservation_ok,
    }