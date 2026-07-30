# ================================================================
# hailcast2d/trajectory.py
# ================================================================
"""
Extensión bidimensional de HAILCAST 1D.

La versión inicial utiliza acoplamiento unidireccional:
HAILCAST 1D calcula crecimiento y dinámica vertical, mientras este
módulo integra el movimiento horizontal debido al viento y al arrastre.
"""

from __future__ import annotations

import numpy as np

from hail_growth import simulate_trajectory
from hailcast2d.wind_profile import HorizontalWindProfile


def drag_coefficient(reynolds_number: float, eps: float = 1e-12) -> float:
    """Coeficiente de arrastre compatible con el modelo HAILCAST 1D."""

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

    El módulo de la velocidad relativa incluye las componentes
    horizontal y vertical.
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
):
    """Avanza [x, vx] un paso mediante RK4."""

    def derivatives(state):
        x_state, vx_state = state
        return np.array([vx_state, acceleration(vx_state)], dtype=float)

    state = np.array([x, vx], dtype=float)

    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _linear_interpolation(a, b, fraction):
    """Interpolación lineal entre dos valores."""
    return a + fraction * (b - a)


def _calculate_impact(result, profile):
    """
    Localiza el primer cruce descendente con z_min e interpola
    las propiedades de impacto.
    """

    height = result["height"]
    ground = profile.z_min

    crossing = np.where(
        (height[:-1] > ground) & (height[1:] <= ground)
    )[0]

    if crossing.size == 0:
        return {
            "impacted_ground": False,
            "termination_reason": "t_max_or_no_ground_crossing",
        }

    i = int(crossing[0])

    denominator = height[i] - height[i + 1]

    if abs(denominator) < 1e-12:
        fraction = 1.0
    else:
        fraction = (height[i] - ground) / denominator

    fraction = float(np.clip(fraction, 0.0, 1.0))

    impact_time = _linear_interpolation(
        result["time"][i], result["time"][i + 1], fraction
    )
    impact_x = _linear_interpolation(
        result["x"][i], result["x"][i + 1], fraction
    )
    impact_vx = _linear_interpolation(
        result["vx"][i], result["vx"][i + 1], fraction
    )
    impact_vz = _linear_interpolation(
        result["velocity"][i], result["velocity"][i + 1], fraction
    )
    impact_mass = _linear_interpolation(
        result["mass"][i], result["mass"][i + 1], fraction
    )
    impact_radius = _linear_interpolation(
        result["radius"][i], result["radius"][i + 1], fraction
    )

    impact_speed = float(np.hypot(impact_vx, impact_vz))
    impact_energy = 0.5 * impact_mass * impact_speed**2

    # Ángulo medido respecto de la horizontal:
    # 0° = impacto rasante; 90° = impacto vertical.
    impact_angle = float(
        np.degrees(np.arctan2(abs(impact_vz), abs(impact_vx)))
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
        "diameter_mm": float(2.0 * impact_radius * 1000.0),
        "kinetic_energy_J": float(impact_energy),
    }


def simulate_trajectory_2d(
    profile,
    wind_profile: HorizontalWindProfile,
    D0_mm: float,
    z0: float,
    T0_K: float,
    x0: float = 0.0,
    vx0: float = 0.0,
    **trajectory_kwargs,
):
    """
    Simula una trayectoria en el plano x-z.

    Primero ejecuta HAILCAST 1D y después integra la componente
    horizontal sobre los mismos pasos temporales.
    """

    result_1d = simulate_trajectory(
        profile=profile,
        D0_mm=D0_mm,
        z0=z0,
        T0_K=T0_K,
        **trajectory_kwargs,
    )

    times = result_1d["time"]
    heights = result_1d["height"]
    vertical_velocities = result_1d["velocity"]
    masses = result_1d["mass"]
    radii = result_1d["radius"]

    n_steps = len(times)

    x = np.empty(n_steps, dtype=float)
    vx = np.empty(n_steps, dtype=float)
    wind_values = np.asarray(wind_profile.velocity(heights), dtype=float)

    x[0] = float(x0)
    vx[0] = float(vx0)

    for i in range(n_steps - 1):
        dt_step = float(times[i + 1] - times[i])

        z_mid = 0.5 * (heights[i] + heights[i + 1])
        vz_mid = 0.5 * (
            vertical_velocities[i] + vertical_velocities[i + 1]
        )
        mass_mid = 0.5 * (masses[i] + masses[i + 1])
        radius_mid = 0.5 * (radii[i] + radii[i + 1])

        wind_mid = float(wind_profile.velocity(z_mid))
        vertical_air_mid = float(profile.w(z_mid))
        air_density_mid = float(profile.air_density(z_mid))

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

        next_state = _rk4_horizontal_step(
            x=x[i],
            vx=vx[i],
            dt=dt_step,
            acceleration=acceleration,
        )

        x[i + 1] = next_state[0]
        vx[i + 1] = next_state[1]

    result_2d = dict(result_1d)
    result_2d["x"] = x
    result_2d["vx"] = vx
    result_2d["horizontal_wind"] = wind_values
    result_2d["impact"] = _calculate_impact(result_2d, profile)
    result_2d["model_dimension"] = "2D_xz"
    result_2d["coupling"] = "one_way"

    return result_2d