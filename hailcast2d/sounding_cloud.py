# ================================================================
# hailcast2d/sounding_cloud.py
# ================================================================
"""Perfil de nube para HAILCAST derivado de un radiosondeo."""

from __future__ import annotations

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

from hailcast2d.sounding_profile import SoundingProfile
from hailcast2d.wind_profile import HorizontalWindProfile


class SoundingCloudProfile:
    """
    Construye perfiles de nube a partir de un radiosondeo.

    Parameters
    ----------
    sounding:
        Instancia de ``SoundingProfile``.
    updraft_efficiency:
        Fracción de la velocidad ideal de parcela conservada después
        de representar mezcla, carga de agua y otros efectos.
    lwc_efficiency:
        Fracción del agua condensada adiabática que permanece como
        agua líquida disponible para acreción.
    """

    def __init__(
        self,
        sounding: SoundingProfile,
        updraft_efficiency: float = 0.25,
        lwc_efficiency: float = 0.20,
    ):
        if not isinstance(sounding, SoundingProfile):
            raise TypeError(
                "sounding debe ser una instancia de SoundingProfile."
            )
        if sounding.el_height_m is None:
            raise ValueError(
                "El radiosondeo no alcanza un nivel de equilibrio; "
                "no puede cerrarse físicamente el perfil de ascenso."
            )

        self.updraft_efficiency = self._validate_efficiency(
            "updraft_efficiency",
            updraft_efficiency,
        )
        self.lwc_efficiency = self._validate_efficiency(
            "lwc_efficiency",
            lwc_efficiency,
        )

        self.sounding = sounding
        self.surface_elevation_m = float(sounding.height_m[0])

        self.height_agl_m = (
            sounding.height_m - self.surface_elevation_m
        )

        self.z_min = 0.0
        # El radiosondeo continúa por encima de la nube hasta la
        # estratosfera. Se conservan ambos límites por separado:
        # ``profile_top_m`` describe la observación atmosférica completa,
        # mientras que ``z_max`` representa el techo físico de la nube.
        self.profile_top_m = float(self.height_agl_m[-1])
        self.z_max = float(
            sounding.el_height_m - self.surface_elevation_m
        )

        if self.z_max <= self.z_min:
            raise ValueError(
                "El nivel de equilibrio debe encontrarse sobre la "
                "superficie."
            )
        if self.z_max > self.profile_top_m:
            raise ValueError(
                "El nivel de equilibrio supera el techo observado "
                "del radiosondeo."
            )

        self._build_pressure_profile()
        self._build_buoyancy_and_updraft()
        self._build_liquid_water_profile()

        maximum_index = int(np.argmax(self.updraft_ms))
        self.z_center = float(self.height_agl_m[maximum_index])
        self.w_peak = float(np.max(self.updraft_ms))
        self.lwc_peak = float(np.max(self.lwc_kgm3))

    @staticmethod
    def _validate_efficiency(name, value):
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(f"{name} debe ser finito.")

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{name} debe encontrarse entre 0 y 1."
            )

        return value

    def _build_pressure_profile(self):
        """
        Prepara la presión para interpolación logarítmica con altura.
        """

        self.log_pressure = np.log(self.sounding.pressure_hpa)

    def _build_buoyancy_and_updraft(self):
        """
        Calcula flotabilidad y velocidad vertical acumulada.

        La velocidad ideal se obtiene integrando únicamente la
        flotabilidad positiva. Después se aplica la eficiencia
        dinámica configurada.
        """

        pressure = self.sounding.pressure_hpa * units.hPa
        environment_temperature = (
            self.sounding.temperature_c * units.degC
        )
        dewpoint = self.sounding.dewpoint_c * units.degC
        parcel_temperature = (
            self.sounding.parcel_temperature_c * units.degC
        )

        environment_mixing_ratio = (
            mpcalc.saturation_mixing_ratio(pressure, dewpoint)
        )

        surface_mixing_ratio = mpcalc.saturation_mixing_ratio(
            pressure[0],
            dewpoint[0],
        )

        parcel_mixing_ratio = np.empty(
            pressure.size,
            dtype=float,
        ) * units.dimensionless

        lcl_pressure = self.sounding.lcl_pressure_hpa

        below_lcl = self.sounding.pressure_hpa >= lcl_pressure
        above_lcl = ~below_lcl

        parcel_mixing_ratio[below_lcl] = surface_mixing_ratio

        if np.any(above_lcl):
            parcel_mixing_ratio[above_lcl] = (
                mpcalc.saturation_mixing_ratio(
                    pressure[above_lcl],
                    parcel_temperature[above_lcl],
                )
            )

        environment_virtual_temperature = mpcalc.virtual_temperature(
            environment_temperature.to("kelvin"),
            environment_mixing_ratio,
        )
        parcel_virtual_temperature = mpcalc.virtual_temperature(
            parcel_temperature.to("kelvin"),
            parcel_mixing_ratio,
        )

        gravity = 9.80665 * units.meter / units.second**2

        buoyancy = (
            gravity
            * (
                parcel_virtual_temperature
                - environment_virtual_temperature
            )
            / environment_virtual_temperature
        ).to("meter / second ** 2")

        self.buoyancy_ms2 = np.asarray(
            buoyancy.magnitude,
            dtype=float,
        )

              # La integración comienza en el LFC. Se conserva el signo de
        # la flotabilidad para representar aceleración y frenado.
        lfc_agl = (
            self.sounding.lfc_height_m
            - self.surface_elevation_m
        )

        start_index = int(
            np.searchsorted(
                self.height_agl_m,
                lfc_agl,
                side="left",
            )
        )

        accumulated_energy = np.zeros_like(
            self.height_agl_m,
            dtype=float,
        )

        for index in range(start_index + 1, self.height_agl_m.size):
            dz = (
                self.height_agl_m[index]
                - self.height_agl_m[index - 1]
            )
            mean_buoyancy = 0.5 * (
                self.buoyancy_ms2[index - 1]
                + self.buoyancy_ms2[index]
            )

            accumulated_energy[index] = max(
                accumulated_energy[index - 1]
                + mean_buoyancy * dz,
                0.0,
            )

        ideal_updraft = np.sqrt(
            2.0 * accumulated_energy
        )

        self.ideal_updraft_ms = ideal_updraft
        self.updraft_ms = (
            self.updraft_efficiency * ideal_updraft
        )

        # Por encima del nivel de equilibrio la parcela deja de definir
        # la corriente ascendente convectiva utilizada por HAILCAST.
        self.updraft_ms[self.height_agl_m > self.z_max] = 0.0

    def _build_liquid_water_profile(self):
        """
        Estima agua líquida mediante condensación adiabática.

        Se calcula la reducción de la razón de mezcla de saturación
        de la parcela y se aplica una fracción líquida dependiente
        de la temperatura.
        """

        pressure = self.sounding.pressure_hpa * units.hPa
        parcel_temperature = (
            self.sounding.parcel_temperature_c * units.degC
        )
        dewpoint_surface = self.sounding.dewpoint_c[0] * units.degC

        initial_mixing_ratio = mpcalc.saturation_mixing_ratio(
            pressure[0],
            dewpoint_surface,
        ).to("dimensionless")

        saturation_profile = mpcalc.saturation_mixing_ratio(
            pressure,
            parcel_temperature,
        ).to("dimensionless")

        condensed_kgkg = np.maximum(
            initial_mixing_ratio.magnitude
            - saturation_profile.magnitude,
            0.0,
        )

        parcel_temperature_c = self.sounding.parcel_temperature_c

        # Fracción líquida inspirada en la glaciación progresiva
        # utilizada por WRF-HAILCAST: líquida sobre -31 °C,
        # completamente glaciada a -38 °C.
        liquid_fraction = np.ones_like(parcel_temperature_c)

        transition = (
            (parcel_temperature_c < -31.0)
            & (parcel_temperature_c > -38.0)
        )
        liquid_fraction[parcel_temperature_c <= -38.0] = 0.0
        liquid_fraction[transition] = (
            parcel_temperature_c[transition] + 38.0
        ) / 7.0

        density = np.asarray(
            [self.air_density(z) for z in self.height_agl_m],
            dtype=float,
        )

        lwc = (
            condensed_kgkg
            * density
            * liquid_fraction
            * self.lwc_efficiency
        )

        # No hay agua de nube por debajo del LCL.
        lcl_agl = (
            self.sounding.lcl_height_m
            - self.surface_elevation_m
        )
        lwc[self.height_agl_m < lcl_agl] = 0.0

        # El agua condensada calculada por la parcela no se considera
        # disponible para acreción fuera del límite superior de la nube.
        lwc[self.height_agl_m > self.z_max] = 0.0

        self.liquid_fraction = liquid_fraction
        self.lwc_kgm3 = np.maximum(lwc, 0.0)

    def pressure(self, z):
        """Presión interpolada en hPa."""

        values = np.exp(
            np.interp(
                z,
                self.height_agl_m,
                self.log_pressure,
                left=self.log_pressure[0],
                right=self.log_pressure[-1],
            )
        )

        return self._scalar_or_array(z, values)

    def temperature(self, z):
        """Temperatura ambiental interpolada en kelvin."""

        values = np.interp(
            z,
            self.height_agl_m,
            self.sounding.temperature_c,
            left=self.sounding.temperature_c[0],
            right=self.sounding.temperature_c[-1],
        ) + 273.15

        return self._scalar_or_array(z, values)

    def air_density(self, z):
        """Densidad del aire mediante la ecuación de gas ideal."""

        pressure_pa = np.asarray(self.pressure(z)) * 100.0
        temperature_k = np.asarray(self.temperature(z))

        values = pressure_pa / (287.05 * temperature_k)

        return self._scalar_or_array(z, values)

    def w(self, z):
        """Velocidad vertical efectiva en m/s."""

        z_array = np.asarray(z)
        values = np.interp(
            z,
            self.height_agl_m,
            self.updraft_ms,
            left=0.0,
            right=0.0,
        )
        values = np.where(
            (z_array < self.z_min) | (z_array >= self.z_max),
            0.0,
            values,
        )

        return self._scalar_or_array(z, values)

    def lwc(self, z):
        """Contenido de agua líquida disponible en kg/m³."""

        z_array = np.asarray(z)
        values = np.interp(
            z,
            self.height_agl_m,
            self.lwc_kgm3,
            left=0.0,
            right=0.0,
        )
        values = np.where(
            (z_array < self.z_min) | (z_array >= self.z_max),
            0.0,
            values,
        )

        return self._scalar_or_array(z, values)

    def wind_profile(self, component="u"):
        """Devuelve el viento observado usando alturas AGL."""

        if component == "u":
            values = self.sounding.u_wind_ms
        elif component == "v":
            values = self.sounding.v_wind_ms
        elif component == "speed":
            values = np.hypot(
                self.sounding.u_wind_ms,
                self.sounding.v_wind_ms,
            )
        else:
            raise ValueError(
                "component debe ser 'u', 'v' o 'speed'."
            )

        return HorizontalWindProfile(
            heights_m=self.height_agl_m,
            speeds_ms=values,
        )

    @staticmethod
    def _scalar_or_array(original, values):
        if np.ndim(original) == 0:
            return float(np.asarray(values))

        return np.asarray(values, dtype=float)

    def summary(self):
        """Resume los parámetros derivados para HAILCAST."""

        return {
            "surface_elevation_m": self.surface_elevation_m,
            "z_min_m": self.z_min,
            "z_max_m": self.z_max,
            "profile_top_m": self.profile_top_m,
            "z_center_m": self.z_center,
            "updraft_efficiency": self.updraft_efficiency,
            "lwc_efficiency": self.lwc_efficiency,
            "ideal_wmax_ms": float(
                np.max(self.ideal_updraft_ms)
            ),
            "effective_wmax_ms": self.w_peak,
            "lwc_peak_kgm3": self.lwc_peak,
            "cape_jkg": self.sounding.cape_jkg,
            "bulk_shear_0_6km_ms":
                self.sounding.bulk_shear_0_6km_ms,
            "energy_shear_index":
                self.sounding.energy_shear_index,
        }