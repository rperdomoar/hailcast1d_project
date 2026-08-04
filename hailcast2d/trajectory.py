"""Trayectorias HAILCAST 2D con modos unidireccional y acoplado."""

from __future__ import annotations

import numpy as np

from hail_growth import simulate_trajectory
from hailcast2d.wind_profile import HorizontalWindProfile


def drag_coefficient(reynolds_number: float, eps: float = 1e-12) -> float:
    """Coeficiente de arrastre compatible con HAILCAST 1D."""
    re = max(float(reynolds_number), 0.0)
    if re < 1.0:
        return 24.0 / (re + eps)
    if re < 1000.0:
        return 24.0 / (re + eps) + 6.0 / (1.0 + np.sqrt(re)) + 0.4
    return 0.44


def gaussian_updraft_factor(x_m: float, center_x_m: float, radius_m: float) -> float:
    """Fracción gaussiana de corriente; el radio corresponde a exp(-1)."""
    radius_m = float(radius_m)
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("updraft_radius_m debe ser positivo y finito.")
    normalized = (float(x_m) - float(center_x_m)) / radius_m
    return float(np.exp(-(normalized**2)))


def _horizontal_acceleration(
    vx, wind_ms, vertical_velocity, vertical_air_velocity,
    mass_kg, radius_m, air_density, dynamic_viscosity=1.8e-5,
    eps=1e-12,
):
    relative_x = vx - wind_ms
    relative_z = vertical_velocity - vertical_air_velocity
    relative_speed = np.hypot(relative_x, relative_z)
    reynolds = (
        air_density * max(relative_speed, 1e-3) * 2.0 * radius_m
        / dynamic_viscosity
    )
    cd = drag_coefficient(reynolds)
    area = np.pi * radius_m**2
    return (
        -0.5 * cd * air_density * area * relative_x * relative_speed
        / (mass_kg + eps)
    )


def _rk4_horizontal_step(x, vx, dt, acceleration):
    def derivatives(state):
        return np.array([state[1], acceleration(state[1])], dtype=float)

    state = np.array([x, vx], dtype=float)
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def _integrate_horizontal_with_substeps(
    x, vx, dt, acceleration, max_substep_s=0.05,
):
    if dt <= 0.0:
        raise ValueError("El paso temporal debe ser positivo.")
    if max_substep_s <= 0.0:
        raise ValueError("El subpaso horizontal máximo debe ser positivo.")
    n_substeps = max(1, int(np.ceil(dt / max_substep_s)))
    substep = dt / n_substeps
    state = np.array([x, vx], dtype=float)
    for _ in range(n_substeps):
        with np.errstate(over="ignore", invalid="ignore"):
            state = _rk4_horizontal_step(
                float(state[0]), float(state[1]), substep, acceleration,
            )
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                "La integración horizontal produjo valores no finitos. "
                "Reduzca max_horizontal_substep_s."
            )
    return state


def _linear_interpolation(a, b, fraction):
    return a + fraction * (b - a)


def _calculate_impact(result, profile):
    height = result["height"]
    ground = profile.z_min
    crossing = np.where((height[:-1] > ground) & (height[1:] <= ground))[0]
    if crossing.size == 0:
        final_vx = float(result["vx"][-1])
        final_vz = float(result["velocity"][-1])
        return {
            "impacted_ground": False,
            "termination_reason": "t_max_or_no_ground_crossing",
            "time_s": float(result["time"][-1]),
            "x_m": float(result["x"][-1]),
            "height_m": float(result["height"][-1]),
            "vx_ms": final_vx,
            "vz_ms": final_vz,
            "speed_ms": float(np.hypot(final_vx, final_vz)),
            "mass_kg": float(result["mass"][-1]),
            "diameter_mm": float(2.0 * result["radius"][-1] * 1000.0),
        }
    i = int(crossing[0])
    denominator = height[i] - height[i + 1]
    fraction = 1.0 if abs(denominator) < 1e-12 else (
        (height[i] - ground) / denominator
    )
    fraction = float(np.clip(fraction, 0.0, 1.0))
    def interp(name):
        return _linear_interpolation(result[name][i], result[name][i + 1], fraction)
    impact_vx = interp("vx")
    impact_vz = interp("velocity")
    impact_mass = interp("mass")
    impact_radius = interp("radius")
    speed = float(np.hypot(impact_vx, impact_vz))
    return {
        "impacted_ground": True,
        "termination_reason": "ground_impact",
        "time_s": float(interp("time")),
        "x_m": float(interp("x")),
        "vx_ms": float(impact_vx),
        "vz_ms": float(impact_vz),
        "speed_ms": speed,
        "angle_from_horizontal_deg": float(
            np.degrees(np.arctan2(abs(impact_vz), abs(impact_vx)))
        ),
        "mass_kg": float(impact_mass),
        "diameter_mm": float(2.0 * impact_radius * 1000.0),
        "kinetic_energy_J": float(0.5 * impact_mass * speed**2),
    }


def _simulate_one_way(
    profile, wind_profile, D0_mm, z0, T0_K, x0, vx0,
    max_horizontal_substep_s, trajectory_kwargs,
):
    result_1d = simulate_trajectory(
        profile=profile, D0_mm=D0_mm, z0=z0, T0_K=T0_K,
        **trajectory_kwargs,
    )
    times = np.asarray(result_1d["time"], dtype=float)
    heights = np.asarray(result_1d["height"], dtype=float)
    vertical_velocities = np.asarray(result_1d["velocity"], dtype=float)
    masses = np.asarray(result_1d["mass"], dtype=float)
    radii = np.asarray(result_1d["radius"], dtype=float)
    x = np.empty(len(times), dtype=float)
    vx = np.empty(len(times), dtype=float)
    x[0], vx[0] = float(x0), float(vx0)
    for i in range(len(times) - 1):
        dt_step = float(times[i + 1] - times[i])
        z_mid = 0.5 * (heights[i] + heights[i + 1])
        vz_mid = 0.5 * (vertical_velocities[i] + vertical_velocities[i + 1])
        mass_mid = 0.5 * (masses[i] + masses[i + 1])
        radius_mid = 0.5 * (radii[i] + radii[i + 1])
        wind_mid = float(wind_profile.velocity(z_mid))
        w_mid = float(profile.w(z_mid))
        rho_air_mid = float(profile.air_density(z_mid))
        def acceleration(vx_state):
            return _horizontal_acceleration(
                vx_state, wind_mid, vz_mid, w_mid, mass_mid,
                radius_mid, rho_air_mid,
            )
        state = _integrate_horizontal_with_substeps(
            x[i], vx[i], dt_step, acceleration, max_horizontal_substep_s,
        )
        x[i + 1], vx[i + 1] = state
    result = dict(result_1d)
    result["x"] = x
    result["vx"] = vx
    result["horizontal_wind"] = np.asarray(wind_profile.velocity(heights))
    result["updraft_factor"] = np.ones_like(heights)
    result["vertical_air_velocity"] = np.asarray(
        [
            float(profile.w(float(height)))
            for height in heights
        ],
        dtype=float,
    )
    result["coupling"] = "one_way"
    return result


def _simulate_coupled(
    profile, wind_profile, D0_mm, z0, T0_K, x0, vx0,
    updraft_center_x_m, updraft_radius_m, storm_motion_ms,
    max_substep_s, trajectory_kwargs,
):
    defaults = {
        "v0": 0.0, "t_max": 3600.0, "dt": 0.1,
        "rho_init": 500.0, "rho_ice": 900.0, "rho_water": 1000.0,
        "alpha_rim": 0.06, "v_scale": 12.0, "beta_melt": 2e-3,
        "tau_base": 800.0, "p_min": 0.02, "p_max": 0.9,
        "C_d": 0.5, "g": 9.81, "k_wind": 1e-4,
        "k_melt": 5e-6, "h_conv": 30.0, "c_h": 2100.0,
        "tau_updraft": 900.0,
    }
    unknown = set(trajectory_kwargs) - set(defaults)
    if unknown:
        raise TypeError("Argumentos de trayectoria no reconocidos: " + ", ".join(sorted(unknown)))
    pms = {**defaults, **trajectory_kwargs}
    dt = float(pms["dt"])
    t_max = float(pms["t_max"])
    if dt <= 0.0 or t_max <= 0.0 or max_substep_s <= 0.0:
        raise ValueError("dt, t_max y max_horizontal_substep_s deben ser positivos.")
    eps, mu, L_f = 1e-12, 1.8e-5, 3.34e5
    m_scale, C_lift, erosion_factor = 1e-4, 0.2, 0.25
    rho_ice, rho_water = pms["rho_ice"], pms["rho_water"]
    m0 = (4.0/3.0) * np.pi * (D0_mm * 1e-3 / 2.0)**3 * rho_ice
    p0 = np.clip(
        (rho_water - pms["rho_init"]) / (rho_water - rho_ice),
        pms["p_min"], pms["p_max"],
    )
    state = np.array([
        m0, pms["v0"], z0, T0_K, p0, pms["rho_init"], 0.0, x0, vx0,
    ], dtype=float)

    def radius_from(state_local):
        mass = max(float(state_local[0]), eps)
        porosity = float(np.clip(state_local[4], pms["p_min"], pms["p_max"]))
        rho_eff = (1.0 - porosity) * rho_ice + porosity * rho_water
        return (3.0 * mass / (4.0 * np.pi * rho_eff + eps))**(1.0/3.0)

    def derivatives(s, _time):
        m, vz, z, temp, porosity, density, memory, x, vx = s
        m_safe = max(float(m), eps)
        p_safe = float(np.clip(porosity, pms["p_min"], pms["p_max"]))
        z_env = max(float(z), float(profile.z_min))
        moving_center = (
            updraft_center_x_m
            + storm_motion_ms * _time
        )
        factor = gaussian_updraft_factor(
            x,
            moving_center,
            updraft_radius_m,
        )
        w_air = float(profile.w(z_env)) * factor
        lwc = float(profile.lwc(z_env)) * factor
        temp_air = float(profile.temperature(z_env))
        rho_air = float(profile.air_density(z_env))
        wind = float(wind_profile.velocity(z_env))
        rho_eff = (1.0 - p_safe) * rho_ice + p_safe * rho_water
        radius = (3.0 * m_safe / (4.0 * np.pi * rho_eff + eps))**(1.0/3.0)
        area = np.pi * radius**2
        surface = 4.0 * np.pi * radius**2
        relative_z = vz - w_air
        r_mm = radius * 1000.0
        collection = np.clip(0.15 + 0.5*r_mm/(r_mm + 1.0), 0.05, 0.95)
        if temp <= 273.15:
            accretion = collection * area * lwc * abs(relative_z)
        else:
            accretion = -pms["k_melt"] * (temp - 273.15) * m_safe
        erosion = pms["k_wind"] * rho_air * area * abs(relative_z) * erosion_factor
        dm = max(accretion - erosion, -m_safe)
        if dm > 0.0:
            compaction = pms["alpha_rim"] * (dm/(m_safe + eps)) * (
                1.0 + abs(relative_z)/pms["v_scale"]
            )
            dp = -compaction
        else:
            dp = pms["beta_melt"] * abs(dm)/(m_safe + eps)
        tau = pms["tau_base"] * (1.0 + np.exp((273.15 - temp)/10.0))
        dp += (pms["p_min"] - p_safe) / max(tau, eps)
        dtemp = (
            pms["h_conv"] * surface * (temp_air - temp) + L_f * accretion
        ) / ((m_safe + eps) * pms["c_h"])
        if abs(w_air) > 0.05:
            dmemory = (1.0 - memory) / max(pms["tau_updraft"], 1.0)
        else:
            dmemory = -memory / max(pms["tau_updraft"], 1.0)
        updraft_mass_factor = 1.0 / (1.0 + m_safe/max(m_scale, eps))
        effective_updraft = w_air * updraft_mass_factor * (memory + 0.1)
        re = rho_air * max(abs(relative_z), 1e-3) * 2.0*radius / mu
        cd = drag_coefficient(re)
        drag = 0.5 * cd * rho_air * area * relative_z * abs(relative_z)
        lift = C_lift * rho_air * area * w_air**2
        dvz = -pms["g"] + effective_updraft - drag/(m_safe + eps) + lift/(m_safe + eps)
        target_density = (1.0 - p_safe)*rho_ice + p_safe*rho_water
        drho = 0.3 * (target_density - density)
        dvx = _horizontal_acceleration(
            vx, wind, vz, w_air, m_safe, radius, rho_air,
        )
        return np.array([dm, dvz, vz, dtemp, dp, drho, dmemory, vx, dvx])

    records = {
        key: []
        for key in (
            "time",
            "mass",
            "radius",
            "height",
            "velocity",
            "temperature",
            "density",
            "porosity",
            "v_terminal",
            "x",
            "vx",
            "horizontal_wind",
            "updraft_factor",
            "vertical_air_velocity",
            "updraft_center_x",
        )
    }

    def record(time_value):
        radius = radius_from(state)
        z_env = max(
            float(state[2]),
            float(profile.z_min),
        )

        moving_center = (
            updraft_center_x_m
            + storm_motion_ms * time_value
        )
        factor = gaussian_updraft_factor(
            state[7],
            moving_center,
            updraft_radius_m,
        )

        w_air = float(profile.w(z_env)) * factor
        rho_air = float(profile.air_density(z_env))
        relative_vertical_velocity = (
            state[1] - w_air
        )

        reynolds = (
            rho_air
            * max(
                abs(relative_vertical_velocity),
                1e-3,
            )
            * 2.0
            * radius
            / mu
        )
        cd = drag_coefficient(reynolds)
        terminal = np.sqrt(
            2.0
            * max(state[0], eps)
            * pms["g"]
            / (
                rho_air
                * np.pi
                * radius**2
                * cd
                + eps
            )
        )
        cd = drag_coefficient(reynolds)
        values = {
            "time": time_value, "mass": state[0], "radius": radius,
            "height": state[2], "velocity": state[1], "temperature": state[3],
            "density": state[5], "porosity": state[4], "v_terminal": terminal,
            "x": state[7], "vx": state[8],
            "horizontal_wind": wind_profile.velocity(z_env),
                        "updraft_factor": factor,
            "vertical_air_velocity": w_air,
            "updraft_center_x": moving_center,
        }
        for key, value in values.items():
            records[key].append(float(value))

    time = 0.0
    record(time)
    while state[2] > profile.z_min and time < t_max:
        outer = min(dt, t_max - time)
        n_substeps = max(1, int(np.ceil(outer/max_substep_s)))
        substep = outer/n_substeps
        for _ in range(n_substeps):
            k1 = derivatives(state, time)
            k2 = derivatives(state + 0.5*substep*k1, time + 0.5*substep)
            k3 = derivatives(state + 0.5*substep*k2, time + 0.5*substep)
            k4 = derivatives(state + substep*k3, time + substep)
            state = state + (substep/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
            if not np.all(np.isfinite(state)):
                raise FloatingPointError(
                    "La integración acoplada produjo valores no finitos. "
                    "Reduzca max_horizontal_substep_s."
                )
            state[0] = max(state[0], eps)
            state[4] = np.clip(state[4], pms["p_min"], pms["p_max"])
            state[5] = max(state[5], 1.0)
            state[6] = np.clip(state[6], 0.0, 1.0)
            time += substep
            if state[2] <= profile.z_min:
                break
        record(time)
    result = {key: np.asarray(value, dtype=float) for key, value in records.items()}
    result["D0_init"] = D0_mm
    result["coupling"] = "two_way_finite_updraft"
    result["updraft_center_x_m"] = float(
        updraft_center_x_m
    )
    result["updraft_radius_m"] = float(
        updraft_radius_m
    )
    result["storm_motion_ms"] = float(
        storm_motion_ms
    )
    return result


def simulate_trajectory_2d(
    profile,
    wind_profile: HorizontalWindProfile,
    D0_mm: float,
    z0: float,
    T0_K: float,
    x0: float = 0.0,
    vx0: float = 0.0,
    max_horizontal_substep_s: float = 0.05,
    updraft_center_x_m: float = 0.0,
    updraft_radius_m: float | None = None,
    storm_motion_ms: float = 0.0,
    **trajectory_kwargs,
):
    """Simula una trayectoria 2D; un radio activa acoplamiento completo."""
    if not isinstance(wind_profile, HorizontalWindProfile):
        raise TypeError("wind_profile debe ser HorizontalWindProfile.")
    if not np.isfinite(storm_motion_ms):
        raise ValueError(
            "storm_motion_ms debe ser finito."
        )
    if updraft_radius_m is None:
        result = _simulate_one_way(
            profile, wind_profile, D0_mm, z0, T0_K, x0, vx0,
            max_horizontal_substep_s, trajectory_kwargs,
        )
    else:
        result = _simulate_coupled(
            profile,
            wind_profile,
            D0_mm,
            z0,
            T0_K,
            x0,
            vx0,
            updraft_center_x_m,
            updraft_radius_m,
            storm_motion_ms,
            max_horizontal_substep_s,
            trajectory_kwargs,
        )
    result["storm_motion_ms"] = float(storm_motion_ms)
    result["impact"] = _calculate_impact(result, profile)
    result["model_dimension"] = "2D_xz"
    result["max_horizontal_substep_s"] = float(max_horizontal_substep_s)
    return result