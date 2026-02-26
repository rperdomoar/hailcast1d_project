# ================================================================
# hail_growth.py (dinámica ajustada para caída realista con RK4)
# ================================================================
from .cloud_model import CloudProfile
import numpy as np

def simulate_trajectory(profile: CloudProfile,
                        D0_mm,
                        z0,
                        T0_K,
                        v0=0.0,
                        t_max=3600.0,
                        dt=0.1,
                        rho_init=500.0,
                        rho_ice=900.0,
                        rho_water=1000.0,
                        alpha_rim=0.06,
                        v_scale=12.0,
                        beta_melt=2e-3,
                        tau_base=800.0,
                        p_min=0.02,
                        p_max=0.9,
                        C_d=0.5,
                        g=9.81,
                        k_wind=1e-4,
                        k_melt=5e-6,
                        h_conv=30.0,
                        c_h=2100.0,
                        tau_updraft=900.0):
    """
    Simula la trayectoria de un grano de granizo (RK4 para las ecuaciones ODE).
    Mantiene notación y nombres originales.
    Ajusta dinámica de caída y suspensión de partículas pequeñas.
    """

    # --- Inicialización ---
    m0 = (4/3) * np.pi * ((D0_mm * 1e-3 / 2.0) ** 3) * rho_ice
    p0 = np.clip((rho_water - rho_init) / (rho_water - rho_ice), p_min, p_max)

    times = [0.0]
    masses = [m0]
    radii = [(3*m0/(4*np.pi*rho_init))**(1/3)]
    heights = [z0]
    velocities = [v0]
    temps = [T0_K]
    densities = [rho_init]
    porosities = [p0]
    v_terminal_list = [0.0]

    # estado inicial
    m = float(m0)
    z = float(z0)
    v = float(v0)
    t = 0.0
    T_grain = float(T0_K)
    p = float(p0)

    # --- Constantes internas ---
    mu = 1.8e-5        # viscosidad dinámica aire
    eps = 1e-12
    m_scale = 1e-4
    m_susp_threshold = 5e-6
    L_f = 3.34e5       # calor latente fusión
    C_lift = 0.2       # coeficiente de sustentación
    erosion_factor = 0.25
    v_fall_max_small = 0.5

    def compute_Cd_from_Re(Re):
        if Re < 1.0:
            return 24.0 / (Re + eps)
        elif Re < 1000.0:
            return 24.0 / (Re + eps) + 6.0 / (1.0 + np.sqrt(Re)) + 0.4
        else:
            return 0.44

    def derivatives(state, t_local):
        """
        state = [m, v, z, T_grain, p, rho, u_mem]
        devuelve: [dm_dt, dv_dt, dz_dt, dT_dt, dp_dt, drho_dt, du_mem_dt]
        """
        m_s, v_s, z_s, T_s, p_s, rho_s, u_mem_s = state

        lwc = profile.lwc(z_s)
        w_air = profile.w(z_s)
        T_air = profile.temperature(z_s)
        rho_air = profile.air_density(z_s)

        # --- Radio y áreas ---
        rho_eff_local = (1.0 - p_s) * rho_ice + p_s * rho_water
        r_s = (3.0 * m_s / (4.0 * np.pi * (rho_eff_local + eps))) ** (1/3)
        A_proj = np.pi * r_s**2
        A_surf = 4.0 * np.pi * r_s**2

        # --- Velocidad relativa ---
        v_rel_s = v_s - w_air

        # --- Eficiencia de colección dependiente del radio ---
        r_mm = r_s * 1000.0
        E_c = np.clip(0.15 + 0.5 * (r_mm / (r_mm + 1.0)), 0.05, 0.95)

        # --- Crecimiento y erosión ---
        if T_s <= 273.15:
            dm_dt_accretion = E_c * A_proj * lwc * abs(v_rel_s)
        else:
            dm_dt_accretion = -k_melt * (T_s - 273.15) * m_s

        dm_dt_erosion = k_wind * rho_air * A_proj * abs(v_rel_s) * erosion_factor
        dm_dt_net = dm_dt_accretion - dm_dt_erosion
        dm_dt_net = max(dm_dt_net, -m_s)

        # --- Porosidad ---
        if dm_dt_net > 0:
            riming_flux = dm_dt_net
            compaction_rate = alpha_rim * (riming_flux / (m_s + eps)) * (1.0 + abs(v_rel_s) / v_scale)
            dp_dt = -compaction_rate
        else:
            dp_dt = beta_melt * (abs(dm_dt_net) / (m_s + eps))

        tau = tau_base * (1.0 + np.exp((273.15 - T_s) / 10.0))
        dp_dt += (p_min - p_s) / max(tau, eps)

        # --- Térmica ---
        Q_conv = h_conv * A_surf * (T_air - T_s)
        Q_latent = L_f * dm_dt_accretion
        dT_dt = (Q_conv + Q_latent) / ((m_s + eps) * c_h)

        # --- Sustentación con memoria ---
        if abs(w_air) > 0.05:
            du_mem_dt = (1.0 - u_mem_s) / max(tau_updraft, 1.0)
        else:
            du_mem_dt = -u_mem_s / max(tau_updraft, 1.0)

        updraft_factor = 1.0 / (1.0 + (m_s / max(m_scale, 1e-12)))
        effective_updraft = w_air * updraft_factor * (1.0 * u_mem_s + 0.1)

        # --- Drag y lift ---
        Re = rho_air * max(abs(v_rel_s), 1e-3) * 2.0 * r_s / mu
        C_d_local = compute_Cd_from_Re(Re)
        F_drag = 0.5 * C_d_local * rho_air * A_proj * v_rel_s * abs(v_rel_s)
        a_drag = -F_drag / (m_s + eps)

        F_lift = C_lift * rho_air * A_proj * (w_air ** 2)
        a_lift = F_lift / (m_s + eps)

        # --- Movimiento vertical ---
        dv_dt = -g + effective_updraft + a_drag + a_lift
        dz_dt = v_s

        # --- Densidad progresiva ---
        target_density = (1.0 - p_s) * rho_ice + p_s * rho_water
        drho_dt = 0.3 * (target_density - rho_s)

        return np.array([dm_dt_net, dv_dt, dz_dt, dT_dt, dp_dt, drho_dt, du_mem_dt], dtype=float)

    # --- Integración RK4 ---
    u_mem0 = 0.0
    state = np.array([m, v, z, T_grain, p, densities[-1], u_mem0], dtype=float)

    while state[2] > profile.z_min and t < t_max:
        k1 = derivatives(state, t)
        k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = derivatives(state + dt * k3, t + dt)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Seguridad física
        state[0] = max(state[0], 1e-12)
        state[4] = float(np.clip(state[4], p_min, p_max))
        state[5] = max(state[5], 1.0)
        state[6] = float(np.clip(state[6], 0.0, 1.0))

        m, v, z, T_grain, p, rho_current, u_mem = state

        # --- Suspensión suave para masas pequeñas ---
        if m < m_susp_threshold:
            if v < -v_fall_max_small:
                v = -v_fall_max_small
                state[1] = v

        # Registro
        t += dt
        times.append(t)
        masses.append(m)
        r_now = (3.0 * m / (4.0 * np.pi * ((1.0 - p) * rho_ice + p * rho_water) + eps)) ** (1/3)
        radii.append(r_now)
        heights.append(z)
        velocities.append(v)
        temps.append(T_grain)
        porosities.append(p)
        densities.append(rho_current)

        # Velocidad terminal instantánea
        rho_air_tmp = profile.air_density(z)
        Re_tmp = rho_air_tmp * max(abs(v - profile.w(z)), 1e-3) * 2.0 * r_now / mu
        Cd_tmp = compute_Cd_from_Re(Re_tmp)
        v_term = np.sqrt(2.0 * m * g / (rho_air_tmp * np.pi * r_now**2 * Cd_tmp + eps))
        v_terminal_list.append(v_term)

        if m <= 0:
            break

    return {
        "time": np.array(times),
        "mass": np.array(masses),
        "radius": np.array(radii),
        "height": np.array(heights),
        "velocity": np.array(velocities),
        "temperature": np.array(temps),
        "density": np.array(densities),
        "porosity": np.array(porosities),
        "v_terminal": np.array(v_terminal_list),
        "D0_init": D0_mm
    }
