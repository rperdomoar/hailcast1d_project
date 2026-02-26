# cloud_model.py
import numpy as np

class CloudProfile:
    def __init__(self, updraft_type="strong", lwc_peak=None, w_peak=None,
                 z_min=0.0, z_max=10000.0, z_center=4000.0):
        """
        Define el perfil de nube estacionario para HAILCAST 1D.

        Parámetros:
        - updraft_type: "strong", "dry" o "weak"
        - lwc_peak: contenido máximo de agua líquida (kg/m³)
        - w_peak: velocidad máxima de la corriente ascendente (m/s)
        - z_min, z_max: límites verticales (m)
        - z_center: altura del máximo ascenso (m)
        """
        self.z_min = z_min
        self.z_max = z_max
        self.z_center = z_center

        # Parámetros predeterminados según tipo de corriente
        defaults = {
            "strong": {"w_peak": 15.0, "lwc_peak": 1.2e-3},
            "dry":    {"w_peak": 8.0,  "lwc_peak": 0.6e-3},
            "weak":   {"w_peak": 3.0,  "lwc_peak": 0.3e-3}
        }
        params = defaults.get(updraft_type, defaults["weak"])
        self.w_peak = w_peak if w_peak is not None else params["w_peak"]
        self.lwc_peak = lwc_peak if lwc_peak is not None else params["lwc_peak"]

        # Gradiente térmico atmosférico (aprox estándar)
        self.lapse_rate = 0.0065  # K/m
        self.T_surface = 288.15   # Temperatura a nivel del suelo (15°C)

    # -------------------------------------------------------------
    # Contenido de agua líquida
    # -------------------------------------------------------------
    def lwc(self, z):
        """
        Contenido de agua líquida (kg/m³) según altura.
        Se modela con una campana gaussiana centrada en z_center.
        """
        sigma = 1500.0
        lwc_value = self.lwc_peak * np.exp(-((z - self.z_center)**2) / (2 * sigma**2))
        return np.maximum(lwc_value, 0.0)

    # -------------------------------------------------------------
    # Velocidad vertical
    # -------------------------------------------------------------
    def w(self, z):
        """
        Velocidad vertical (m/s) según altura.
        Perfil gaussiano con recirculación en base y tope de nube.
        """
        sigma = 2000.0
        w_val = self.w_peak * np.exp(-((z - self.z_center)**2) / (2 * sigma**2))

        # Regiones de recirculación descendente
        if z > self.z_center + 3000:
            w_val = -5.0  # Corriente descendente fuerte en la parte alta
        elif z < self.z_center - 3000:
            w_val = -2.0  # Corriente descendente leve en la base
        return w_val

    # -------------------------------------------------------------
    # Temperatura ambiental
    # -------------------------------------------------------------
    def temperature(self, z):
        """
        Temperatura ambiental (K) en función de la altura.
        """
        T = self.T_surface - self.lapse_rate * z
        return max(T, 180.0)  # límite inferior físico para evitar valores negativos
    

    # -------------------------------------------------------------
    # Densidad del aire 
    # -------------------------------------------------------------
    def air_density(self, z):
        """
        Densidad del aire aproximada usando ley barométrica simplificada.
        Devuelve rho [kg/m^3].
        """
        T = self.temperature(z)
        p0 = 101325.0
        lapse = 0.0065
        # Presión aproximada con gradiente estándar
        p = p0 * (1 - lapse * z / 288.15) ** 5.256
        R = 287.05
        rho = p / (R * (T if T>0 else 180.0))
        return rho
