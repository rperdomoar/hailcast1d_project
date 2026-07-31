# ================================================================
# hailcast2d/ensemble.py
# ================================================================
"""Simulaciones de ensamble para HAILCAST 2D."""

from __future__ import annotations

import numpy as np

from hailcast2d.trajectory import simulate_trajectory_2d
from hailcast2d.wind_profile import HorizontalWindProfile


def _validate_non_negative(name: str, value: float) -> float:
    """Valida parámetros escalares que no pueden ser negativos."""

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} debe ser finito.")

    if value < 0.0:
        raise ValueError(f"{name} no puede ser negativo.")

    return value


def run_ensemble(
    profile,
    wind_profile: HorizontalWindProfile,
    n_members: int,
    D0_mm: float,
    z0: float,
    T0_K: float,
    x0: float = 0.0,
    vx0: float = 0.0,
    diameter_std_mm: float = 0.0,
    x0_std_m: float = 0.0,
    vx0_std_ms: float = 0.0,
    wind_std_ms: float = 0.0,
    seed: int | None = None,
    **trajectory_kwargs,
):
    """
    Ejecuta un ensamble de trayectorias bidimensionales.

    Cada miembro puede variar en:

    - diámetro inicial;
    - posición horizontal inicial;
    - velocidad horizontal inicial;
    - intensidad del perfil completo de viento.

    La perturbación del viento es un desplazamiento constante aplicado
    a todos los niveles del perfil. De esta manera se conserva la
    estructura vertical y la cizalladura del perfil original.

    Parameters
    ----------
    profile:
        Perfil de nube usado por HAILCAST 1D.
    wind_profile:
        Perfil vertical base del viento horizontal.
    n_members:
        Número de piedras simuladas.
    D0_mm, z0, T0_K:
        Condiciones iniciales centrales.
    diameter_std_mm:
        Desvío estándar del diámetro inicial.
    x0_std_m:
        Desvío estándar de la posición inicial.
    vx0_std_ms:
        Desvío estándar de la velocidad horizontal inicial.
    wind_std_ms:
        Desvío estándar de la perturbación del viento.
    seed:
        Semilla del generador aleatorio.
    trajectory_kwargs:
        Argumentos adicionales para ``simulate_trajectory_2d``.

    Returns
    -------
    dict
        Trayectorias, impactos, condiciones iniciales y resumen.
    """

    if not isinstance(n_members, (int, np.integer)) or n_members <= 0:
        raise ValueError("n_members debe ser un entero mayor que cero.")

    if not isinstance(wind_profile, HorizontalWindProfile):
        raise TypeError(
            "wind_profile debe ser una instancia de HorizontalWindProfile."
        )

    D0_mm = float(D0_mm)
    z0 = float(z0)
    T0_K = float(T0_K)
    x0 = float(x0)
    vx0 = float(vx0)

    central_values = {
        "D0_mm": D0_mm,
        "z0": z0,
        "T0_K": T0_K,
        "x0": x0,
        "vx0": vx0,
    }

    if not all(np.isfinite(value) for value in central_values.values()):
        raise ValueError("Las condiciones iniciales deben ser finitas.")

    if D0_mm <= 0.0:
        raise ValueError("D0_mm debe ser mayor que cero.")

    diameter_std_mm = _validate_non_negative(
        "diameter_std_mm", diameter_std_mm
    )
    x0_std_m = _validate_non_negative("x0_std_m", x0_std_m)
    vx0_std_ms = _validate_non_negative("vx0_std_ms", vx0_std_ms)
    wind_std_ms = _validate_non_negative("wind_std_ms", wind_std_ms)

    rng = np.random.default_rng(seed)

    trajectories = []
    impacts = []
    initial_conditions = []

    for member_id in range(n_members):
        sampled_diameter = float(
            rng.normal(D0_mm, diameter_std_mm)
            if diameter_std_mm > 0.0
            else D0_mm
        )

        # Evita diámetros nulos o negativos en realizaciones extremas.
        sampled_diameter = max(sampled_diameter, 1e-6)

        sampled_x0 = float(
            rng.normal(x0, x0_std_m) if x0_std_m > 0.0 else x0
        )
        sampled_vx0 = float(
            rng.normal(vx0, vx0_std_ms) if vx0_std_ms > 0.0 else vx0
        )
        wind_offset = float(
            rng.normal(0.0, wind_std_ms)
            if wind_std_ms > 0.0
            else 0.0
        )

        member_wind = HorizontalWindProfile(
            heights_m=wind_profile.heights_m,
            speeds_ms=wind_profile.speeds_ms + wind_offset,
        )

        conditions = {
            "member_id": member_id,
            "D0_mm": sampled_diameter,
            "z0_m": z0,
            "T0_K": T0_K,
            "x0_m": sampled_x0,
            "vx0_ms": sampled_vx0,
            "wind_offset_ms": wind_offset,
        }

        result = simulate_trajectory_2d(
            profile=profile,
            wind_profile=member_wind,
            D0_mm=sampled_diameter,
            z0=z0,
            T0_K=T0_K,
            x0=sampled_x0,
            vx0=sampled_vx0,
            **trajectory_kwargs,
        )

        result["member_id"] = member_id
        result["initial_conditions"] = conditions.copy()

        trajectories.append(result)
        initial_conditions.append(conditions)

        if result["impact"]["impacted_ground"]:
            impact = dict(result["impact"])
            impact["member_id"] = member_id
            impact["initial_conditions"] = conditions.copy()
            impacts.append(impact)

    n_impacted = len(impacts)

    return {
        "trajectories": trajectories,
        "impacts": impacts,
        "initial_conditions": initial_conditions,
        "n_simulated": n_members,
        "n_impacted": n_impacted,
        "impact_fraction": n_impacted / n_members,
        "seed": seed,
        "perturbation_std": {
            "diameter_mm": diameter_std_mm,
            "x0_m": x0_std_m,
            "vx0_ms": vx0_std_ms,
            "wind_ms": wind_std_ms,
        },
    }
