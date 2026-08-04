# ================================================================
# hailcast2d/trajectory.py
# ================================================================
"""
Extensión bidimensional de HAILCAST 1D.

Esta versión utiliza acoplamiento unidireccional: HAILCAST 1D calcula
el crecimiento y la dinámica vertical, mientras este módulo integra
el movimiento horizontal causado por el viento y el arrastre.

La integración horizontal emplea subpasos automáticos para evitar
inestabilidades numéricas con embriones de granizo pequeños.
"""

from __future__ import annotations

import numpy as np

from hail_growth import simulate_trajectory
from hailcast2d.wind_profile import HorizontalWindProfile


def drag_coefficient(
    reynolds_number: float,
    eps: float = 1e-12,
) -> float:
    """Calcula el coeficiente de arrastre usado por HAILCAST 1D."""

    re = max(float(reynolds_number), 0.0)

    if re < 1.0:
        return 24.0 / (re + eps)

    if re < 1000.0:
        return (
            24.0 / (re + eps)
            + 6.0 / (1.0 + np.sqrt(re))
            + 0.4
        )

    return 0.44


def _horizontal_acceleration(
    vx: float,
    wind_ms: float,
    vertical_velocity: float,
    vertical_air_velocity: float,
    mass_kg: float,
    radius_m: float,
    air_density: float,
    dynamic_viscosity: float = 1.8e-5,
    eps: float = 1e-12,
) -> float:
    """
    Calcula la aceleración horizontal producida por el arrastre.

    El módulo de la velocidad relativa considera conjuntamente las
    componentes horizontal y vertical del movimiento.
    """

    relative_x = vx - wind_ms
    relative_z = vertical_velocity - vertical_air_velocity

    relative_speed = np.hypot(relative_x, relative_z)
    diameter_m = 2.0 * radius_m

    reynolds = (
        air_density
        * max(relative_speed, 1e-3)
        * diameter_m
        / dynamic_viscosity
    )

    cd = drag_coefficient(reynolds)
    projected_area = np.pi * radius_m**2

    return (
        -0.5
        * cd
        * air_density
        * projected_area
        * relative_x
        * relative_speed
        / (mass_kg + eps)
    )


def _rk4_horizontal_step(
    x: float,
    vx: float,
    dt: float,
    acceleration,
) -> np.ndarray:
    """Avanza el estado horizontal ``[x, vx]`` mediante RK4."""

    def derivatives(state):
        _, vx_state = state

        return np.array(
            [
                vx_state,
                acceleration(vx_state),
            ],
            dtype=float,
        )

    state = np.array([x, vx], dtype=float)

    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)

    return state + (dt / 6.0) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


def _integrate_horizontal_with_substeps(
    x: float,
    vx: float,
    dt: float,
    acceleration,
    max_substep_s: float = 0.05,
) -> np.ndarray:
    """
    Integra el movimiento horizontal mediante subpasos automáticos.

    El paso temporal usado por la dinámica vertical puede ser mayor
    que el intervalo numéricamente estable para la dinámica horizontal.
    Esto ocurre principalmente con embriones pequeños, cuya masa genera
    tiempos de respuesta aerodinámica muy cortos.

    El intervalo vertical se divide automáticamente en pasos menores o
    iguales a ``max_substep_s``.

    Parameters
    ----------
    x:
        Posición horizontal inicial del intervalo, en metros.
    vx:
        Velocidad horizontal inicial, en m/s.
    dt:
        Duración total del intervalo, en segundos.
    acceleration:
        Función que calcula la aceleración horizontal a partir de ``vx``.
    max_substep_s:
        Duración máxima permitida para cada subpaso horizontal.

    Returns
    -------
    numpy.ndarray
        Estado final ``[x, vx]`` del intervalo.

    Raises
    ------
    ValueError
        Si los pasos temporales no son positivos.
    FloatingPointError
        Si la integración genera valores no finitos.
    """

    if dt <= 0.0:
        raise ValueError(
            "El paso temporal debe ser positivo."
        )

    if max_substep_s <= 0.0:
        raise ValueError(
            "El subpaso horizontal máximo debe ser positivo."
        )

    n_substeps = max(
        1,
        int(np.ceil(dt / max_substep_s)),
    )
    substep = dt / n_substeps

    state = np.array([x, vx], dtype=float)

    for _ in range(n_substeps):
        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            next_state = _rk4_horizontal_step(
                x=float(state[0]),
                vx=float(state[1]),
                dt=substep,
                acceleration=acceleration,
            )

        if not np.all(np.isfinite(next_state)):
            raise FloatingPointError(
                "La integración horizontal produjo valores no finitos. "
                "Reduzca max_horizontal_substep_s."
            )

        state = next_state

    return state


def _linear_interpolation(
    a: float,
    b: float,
    fraction: float,
) -> float:
    """Interpola linealmente entre dos valores."""

    return a + fraction * (b - a)


def _calculate_impact(result, profile) -> dict:
    """
    Localiza el primer cruce descendente con el suelo.

    Las propiedades del granizo se interpolan linealmente entre los dos
    pasos temporales que rodean el cruce.
    """

    height = result["height"]
    ground = profile.z_min

    crossing = np.where(
        (height[:-1] > ground)
        & (height[1:] <= ground)
    )[0]

    if crossing.size == 0:
        final_mass = float(result["mass"][-1])
        final_radius = float(result["radius"][-1])
        final_vx = float(result["vx"][-1])
        final_vz = float(result["velocity"][-1])

        return {
            "impacted_ground": False,
            "termination_reason": (
                "t_max_or_no_ground_crossing"
            ),
            "time_s": float(result["time"][-1]),
            "x_m": float(result["x"][-1]),
            "height_m": float(result["height"][-1]),
            "vx_ms": final_vx,
            "vz_ms": final_vz,
            "speed_ms": float(
                np.hypot(final_vx, final_vz)
            ),
            "mass_kg": final_mass,
            "diameter_mm": float(
                2.0 * final_radius * 1000.0
            ),
        }

    i = int(crossing[0])

    denominator = height[i] - height[i + 1]

    if abs(denominator) < 1e-12:
        fraction = 1.0
    else:
        fraction = (
            (height[i] - ground)
            / denominator
        )

    fraction = float(
        np.clip(fraction, 0.0, 1.0)
    )

    impact_time = _linear_interpolation(
        result["time"][i],
        result["time"][i + 1],
        fraction,
    )
    impact_x = _linear_interpolation(
        result["x"][i],
        result["x"][i + 1],
        fraction,
    )
    impact_vx = _linear_interpolation(
        result["vx"][i],
        result["vx"][i + 1],
        fraction,
    )
    impact_vz = _linear_interpolation(
        result["velocity"][i],
        result["velocity"][i + 1],
        fraction,
    )
    impact_mass = _linear_interpolation(
        result["mass"][i],
        result["mass"][i + 1],
        fraction,
    )
    impact_radius = _linear_interpolation(
        result["radius"][i],
        result["radius"][i + 1],
        fraction,
    )

    impact_speed = float(
        np.hypot(impact_vx, impact_vz)
    )
    impact_energy = (
        0.5
        * impact_mass
        * impact_speed**2
    )

    # Ángulo medido respecto de la horizontal:
    # 0° representa un impacto rasante.
    # 90° representa un impacto vertical.
    impact_angle = float(
        np.degrees(
            np.arctan2(
                abs(impact_vz),
                abs(impact_vx),
            )
        )
    )

    return {
        "impacted_ground": True,
        "termination_reason": "ground_impact",
        "time_s": float(impact_time),
        "x_m": float(impact_x),
        "vx_ms": float(impact_vx),
        "vz_ms": float(impact_vz),
        "speed_ms": impact_speed,
        "angle_from_horizontal_deg": impact_angle,
        "mass_kg": float(impact_mass),
        "diameter_mm": float(
            2.0 * impact_radius * 1000.0
        ),
        "kinetic_energy_J": float(
            impact_energy
        ),
    }


def simulate_trajectory_2d(
    profile,
    wind_profile: HorizontalWindProfile,
    D0_mm: float,
    z0: float,
    T0_K: float,
    x0: float = 0.0,
    vx0: float = 0.0,
    max_horizontal_substep_s: float = 0.05,
    **trajectory_kwargs,
) -> dict:
    """
    Simula una trayectoria de granizo en el plano horizontal-vertical.

    Primero ejecuta HAILCAST 1D para obtener el crecimiento y movimiento
    vertical. Después integra la componente horizontal sobre los mismos
    intervalos temporales.

    La componente horizontal utiliza subpasos automáticos. Por ejemplo,
    si HAILCAST 1D usa ``dt=0.5 s`` y el máximo subpaso horizontal es
    ``0.05 s``, se realizan diez pasos horizontales dentro de cada paso
    vertical.

    Parameters
    ----------
    profile:
        Perfil de nube compatible con HAILCAST 1D.
    wind_profile:
        Perfil vertical del viento horizontal.
    D0_mm:
        Diámetro inicial del granizo, en milímetros.
    z0:
        Altura inicial AGL, en metros.
    T0_K:
        Temperatura inicial del granizo, en kelvin.
    x0:
        Posición horizontal inicial, en metros.
    vx0:
        Velocidad horizontal inicial, en m/s.
    max_horizontal_substep_s:
        Paso máximo de la integración horizontal, en segundos.
    **trajectory_kwargs:
        Parámetros adicionales enviados a ``simulate_trajectory``.

    Returns
    -------
    dict
        Trayectoria vertical y horizontal, junto con las propiedades
        calculadas del impacto.
    """

    if max_horizontal_substep_s <= 0.0:
        raise ValueError(
            "max_horizontal_substep_s debe ser positivo."
        )

    result_1d = simulate_trajectory(
        profile=profile,
        D0_mm=D0_mm,
        z0=z0,
        T0_K=T0_K,
        **trajectory_kwargs,
    )

    times = np.asarray(
        result_1d["time"],
        dtype=float,
    )
    heights = np.asarray(
        result_1d["height"],
        dtype=float,
    )
    vertical_velocities = np.asarray(
        result_1d["velocity"],
        dtype=float,
    )
    masses = np.asarray(
        result_1d["mass"],
        dtype=float,
    )
    radii = np.asarray(
        result_1d["radius"],
        dtype=float,
    )

    n_steps = len(times)

    x = np.empty(n_steps, dtype=float)
    vx = np.empty(n_steps, dtype=float)

    wind_values = np.asarray(
        wind_profile.velocity(heights),
        dtype=float,
    )

    x[0] = float(x0)
    vx[0] = float(vx0)

    for i in range(n_steps - 1):
        dt_step = float(
            times[i + 1] - times[i]
        )

        z_mid = 0.5 * (
            heights[i] + heights[i + 1]
        )
        vz_mid = 0.5 * (
            vertical_velocities[i]
            + vertical_velocities[i + 1]
        )
        mass_mid = 0.5 * (
            masses[i] + masses[i + 1]
        )
        radius_mid = 0.5 * (
            radii[i] + radii[i + 1]
        )

        wind_mid = float(
            wind_profile.velocity(z_mid)
        )
        vertical_air_mid = float(
            profile.w(z_mid)
        )
        air_density_mid = float(
            profile.air_density(z_mid)
        )

        def acceleration(vx_state):
            return _horizontal_acceleration(
                vx=vx_state,
                wind_ms=wind_mid,
                vertical_velocity=vz_mid,
                vertical_air_velocity=vertical_air_mid,
                mass_kg=mass_mid,
                radius_m=radius_mid,
                air_density=air_density_mid,
            )

        next_state = _integrate_horizontal_with_substeps(
            x=x[i],
            vx=vx[i],
            dt=dt_step,
            acceleration=acceleration,
            max_substep_s=max_horizontal_substep_s,
        )

        x[i + 1] = next_state[0]
        vx[i + 1] = next_state[1]

    result_2d = dict(result_1d)
    result_2d["x"] = x
    result_2d["vx"] = vx
    result_2d["horizontal_wind"] = wind_values
    result_2d["impact"] = _calculate_impact(
        result_2d,
        profile,
    )
    result_2d["model_dimension"] = "2D_xz"
    result_2d["coupling"] = "one_way"
    result_2d["max_horizontal_substep_s"] = float(
        max_horizontal_substep_s
    )

    return result_2d