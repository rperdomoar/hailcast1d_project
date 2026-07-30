# ================================================================
# hailcast2d/wind_profile.py
# ================================================================
"""Perfiles de viento horizontal para trayectorias en el plano x-z."""

from __future__ import annotations

import numpy as np


class HorizontalWindProfile:
    """
    Representa la componente horizontal del viento u(z).

    Parámetros
    ----------
    heights_m:
        Alturas en metros, estrictamente crecientes.
    speeds_ms:
        Velocidades horizontales correspondientes, en m/s.

    Notas
    -----
    Fuera del intervalo definido se mantiene constante el valor
    del extremo más próximo.
    """

    def __init__(self, heights_m, speeds_ms):
        heights = np.asarray(heights_m, dtype=float)
        speeds = np.asarray(speeds_ms, dtype=float)

        if heights.ndim != 1 or speeds.ndim != 1:
            raise ValueError("Las alturas y velocidades deben ser arreglos 1D.")

        if heights.size < 2:
            raise ValueError("El perfil necesita al menos dos niveles.")

        if heights.size != speeds.size:
            raise ValueError(
                "Las alturas y velocidades deben tener igual longitud."
            )

        if not np.all(np.isfinite(heights)):
            raise ValueError("Las alturas contienen valores no finitos.")

        if not np.all(np.isfinite(speeds)):
            raise ValueError("Las velocidades contienen valores no finitos.")

        if np.any(heights < 0.0):
            raise ValueError("Las alturas no pueden ser negativas.")

        if np.any(np.diff(heights) <= 0.0):
            raise ValueError(
                "Las alturas deben estar ordenadas de forma estrictamente creciente."
            )

        self.heights_m = heights.copy()
        self.speeds_ms = speeds.copy()

    @classmethod
    def constant(
        cls,
        speed_ms: float,
        z_min_m: float = 0.0,
        z_max_m: float = 12000.0,
    ) -> "HorizontalWindProfile":
        """Construye un perfil de viento horizontal constante."""

        if not np.isfinite(speed_ms):
            raise ValueError("La velocidad debe ser finita.")

        if z_min_m < 0.0 or z_max_m <= z_min_m:
            raise ValueError("Los límites verticales no son válidos.")

        return cls(
            heights_m=[z_min_m, z_max_m],
            speeds_ms=[speed_ms, speed_ms],
        )

    @classmethod
    def linear(
        cls,
        u_surface_ms: float,
        u_top_ms: float,
        z_top_m: float = 12000.0,
    ) -> "HorizontalWindProfile":
        """Construye un perfil lineal desde el suelo hasta z_top_m."""

        if z_top_m <= 0.0:
            raise ValueError("z_top_m debe ser mayor que cero.")

        return cls(
            heights_m=[0.0, z_top_m],
            speeds_ms=[u_surface_ms, u_top_ms],
        )

    def velocity(self, z_m):
        """
        Devuelve u(z) en m/s mediante interpolación lineal.

        Acepta una altura escalar o un arreglo de alturas.
        """

        z = np.asarray(z_m, dtype=float)

        if not np.all(np.isfinite(z)):
            raise ValueError("La altura contiene valores no finitos.")

        values = np.interp(
            z,
            self.heights_m,
            self.speeds_ms,
            left=self.speeds_ms[0],
            right=self.speeds_ms[-1],
        )

        if np.ndim(z_m) == 0:
            return float(values)

        return values

    def __call__(self, z_m):
        """Permite evaluar el perfil como wind(z)."""
        return self.velocity(z_m)

    def __repr__(self):
        return (
            f"HorizontalWindProfile(levels={self.heights_m.size}, "
            f"z_min={self.heights_m[0]:.1f} m, "
            f"z_max={self.heights_m[-1]:.1f} m)"
        )