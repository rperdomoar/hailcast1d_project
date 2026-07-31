# ================================================================
# hailcast2d/sounding_profile.py
# ================================================================
"""Representación y diagnóstico de radiosondeos para HAILCAST 2D."""

from __future__ import annotations

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

from hailcast2d.wind_profile import HorizontalWindProfile


class SoundingProfile:
    """
    Radiosondeo vertical con diagnósticos termodinámicos.

    Parameters
    ----------
    pressure_hpa:
        Presión en hPa, estrictamente decreciente.
    height_m:
        Altura en metros, estrictamente creciente.
    temperature_c:
        Temperatura ambiental en grados Celsius.
    dewpoint_c:
        Temperatura de punto de rocío en grados Celsius.
    u_wind_ms, v_wind_ms:
        Componentes zonal y meridional del viento en m/s.
    """

    def __init__(
        self,
        pressure_hpa,
        height_m,
        temperature_c,
        dewpoint_c,
        u_wind_ms,
        v_wind_ms,
    ):
        arrays = {
            "pressure_hpa": np.asarray(pressure_hpa, dtype=float),
            "height_m": np.asarray(height_m, dtype=float),
            "temperature_c": np.asarray(temperature_c, dtype=float),
            "dewpoint_c": np.asarray(dewpoint_c, dtype=float),
            "u_wind_ms": np.asarray(u_wind_ms, dtype=float),
            "v_wind_ms": np.asarray(v_wind_ms, dtype=float),
        }

        self._validate(arrays)

        self.pressure_hpa = arrays["pressure_hpa"].copy()
        self.height_m = arrays["height_m"].copy()
        self.temperature_c = arrays["temperature_c"].copy()

        # Pequeñas supersaturaciones pueden aparecer por redondeo.
        self.dewpoint_c = np.minimum(
            arrays["dewpoint_c"],
            arrays["temperature_c"],
        )

        self.u_wind_ms = arrays["u_wind_ms"].copy()
        self.v_wind_ms = arrays["v_wind_ms"].copy()

        self._calculate_thermodynamics()
        self._calculate_kinematics()

    @staticmethod
    def _validate(arrays):
        lengths = {array.size for array in arrays.values()}

        if len(lengths) != 1:
            raise ValueError(
                "Todas las variables del radiosondeo deben tener "
                "igual longitud."
            )

        n_levels = next(iter(lengths))

        if n_levels < 6:
            raise ValueError(
                "El radiosondeo debe contener al menos seis niveles."
            )

        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"{name} debe ser un arreglo 1D.")

            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contiene valores no finitos.")

        if np.any(arrays["pressure_hpa"] <= 0.0):
            raise ValueError("La presión debe ser mayor que cero.")

        if np.any(np.diff(arrays["pressure_hpa"]) >= 0.0):
            raise ValueError(
                "La presión debe ser estrictamente decreciente."
            )

        if np.any(np.diff(arrays["height_m"]) <= 0.0):
            raise ValueError(
                "La altura debe ser estrictamente creciente."
            )

        supersaturation = (
            arrays["dewpoint_c"] - arrays["temperature_c"]
        )

        if np.any(supersaturation > 0.5):
            raise ValueError(
                "El punto de rocío supera la temperatura en más de 0.5 °C."
            )

        if arrays["height_m"][-1] - arrays["height_m"][0] < 6000.0:
            raise ValueError(
                "El radiosondeo debe cubrir al menos 6 km verticales."
            )

    def _calculate_thermodynamics(self):
        pressure = self.pressure_hpa * units.hPa
        temperature = self.temperature_c * units.degC
        dewpoint = self.dewpoint_c * units.degC

        parcel = mpcalc.parcel_profile(
            pressure,
            temperature[0],
            dewpoint[0],
        ).to("degC")

        cape, cin = mpcalc.cape_cin(
            pressure,
            temperature,
            dewpoint,
            parcel,
        )

        lcl_pressure, lcl_temperature = mpcalc.lcl(
            pressure[0],
            temperature[0],
            dewpoint[0],
        )

        lfc_pressure, lfc_temperature = mpcalc.lfc(
            pressure,
            temperature,
            dewpoint,
            parcel_temperature_profile=parcel,
        )

        el_pressure, el_temperature = mpcalc.el(
            pressure,
            temperature,
            dewpoint,
            parcel_temperature_profile=parcel,
        )

        self.parcel_temperature_c = np.asarray(
            parcel.magnitude,
            dtype=float,
        )

        self.cape_jkg = float(cape.to("joule / kilogram").magnitude)
        self.cin_jkg = float(cin.to("joule / kilogram").magnitude)

        self.lcl_pressure_hpa = float(
            lcl_pressure.to("hPa").magnitude
        )
        self.lcl_temperature_c = float(
            lcl_temperature.to("degC").magnitude
        )

        self.lfc_pressure_hpa = self._quantity_or_none(
            lfc_pressure,
            "hPa",
        )
        self.lfc_temperature_c = self._quantity_or_none(
            lfc_temperature,
            "degC",
        )
        self.el_pressure_hpa = self._quantity_or_none(
            el_pressure,
            "hPa",
        )
        self.el_temperature_c = self._quantity_or_none(
            el_temperature,
            "degC",
        )

        self.lcl_height_m = self.height_at_pressure(
            self.lcl_pressure_hpa
        )
        self.lfc_height_m = (
            self.height_at_pressure(self.lfc_pressure_hpa)
            if self.lfc_pressure_hpa is not None
            else None
        )
        self.el_height_m = (
            self.height_at_pressure(self.el_pressure_hpa)
            if self.el_pressure_hpa is not None
            else None
        )

        self.freezing_level_m = self._first_zero_crossing_height()

        self.ideal_wmax_ms = float(
            np.sqrt(max(2.0 * self.cape_jkg, 0.0))
        )

    @staticmethod
    def _quantity_or_none(quantity, target_unit):
        value = float(quantity.to(target_unit).magnitude)

        if not np.isfinite(value):
            return None

        return value

    def _calculate_kinematics(self):
        base_height = self.height_m[0]
        top_height = base_height + 6000.0

        u_base = float(np.interp(base_height, self.height_m, self.u_wind_ms))
        v_base = float(np.interp(base_height, self.height_m, self.v_wind_ms))
        u_top = float(np.interp(top_height, self.height_m, self.u_wind_ms))
        v_top = float(np.interp(top_height, self.height_m, self.v_wind_ms))

        self.shear_u_0_6km_ms = u_top - u_base
        self.shear_v_0_6km_ms = v_top - v_base
        self.bulk_shear_0_6km_ms = float(
            np.hypot(
                self.shear_u_0_6km_ms,
                self.shear_v_0_6km_ms,
            )
        )

        # Unidades: J kg-1 multiplicado por s-1.
        self.energy_shear_index = (
            self.cape_jkg
            * self.bulk_shear_0_6km_ms
            / 6000.0
        )

    def height_at_pressure(self, pressure_hpa):
        """Interpola la altura correspondiente a una presión."""

        pressure_hpa = float(pressure_hpa)

        return float(
            np.interp(
                pressure_hpa,
                self.pressure_hpa[::-1],
                self.height_m[::-1],
            )
        )

    def _first_zero_crossing_height(self):
        """Obtiene el primer nivel de congelación sobre superficie."""

        temperature = self.temperature_c
        height = self.height_m

        crossing = np.where(
            (temperature[:-1] > 0.0)
            & (temperature[1:] <= 0.0)
        )[0]

        if crossing.size == 0:
            return None

        index = int(crossing[0])
        fraction = (
            temperature[index]
            / (temperature[index] - temperature[index + 1])
        )

        return float(
            height[index]
            + fraction * (height[index + 1] - height[index])
        )

    def wind_profile(self, component="u"):
        """
        Construye el perfil horizontal para el modelo x-z.

        ``component='u'`` usa viento zonal;
        ``component='v'`` usa viento meridional;
        ``component='speed'`` usa el módulo del viento.
        """

        if component == "u":
            values = self.u_wind_ms
        elif component == "v":
            values = self.v_wind_ms
        elif component == "speed":
            values = np.hypot(self.u_wind_ms, self.v_wind_ms)
        else:
            raise ValueError(
                "component debe ser 'u', 'v' o 'speed'."
            )

        return HorizontalWindProfile(
            heights_m=self.height_m,
            speeds_ms=values,
        )

    def summary(self):
        """Devuelve los principales diagnósticos del radiosondeo."""

        return {
            "n_levels": int(self.pressure_hpa.size),
            "cape_jkg": self.cape_jkg,
            "cin_jkg": self.cin_jkg,
            "lcl_pressure_hpa": self.lcl_pressure_hpa,
            "lcl_height_m": self.lcl_height_m,
            "lfc_pressure_hpa": self.lfc_pressure_hpa,
            "lfc_height_m": self.lfc_height_m,
            "el_pressure_hpa": self.el_pressure_hpa,
            "el_height_m": self.el_height_m,
            "freezing_level_m": self.freezing_level_m,
            "ideal_wmax_ms": self.ideal_wmax_ms,
            "bulk_shear_0_6km_ms": self.bulk_shear_0_6km_ms,
            "energy_shear_index": self.energy_shear_index,
        }