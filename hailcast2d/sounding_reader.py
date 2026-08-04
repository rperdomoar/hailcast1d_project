# ================================================================
# hailcast2d/sounding_reader.py
# ================================================================
"""
Lectura y normalización de radiosondeos métricos de Wyoming Weather Web.

Este módulo convierte un radiosondeo publicado como ``Text listing`` en
un objeto ``SoundingProfile`` compatible con HAILCAST 2D.

El procesamiento realiza las siguientes operaciones:

1. Localiza la tabla meteorológica dentro del archivo de texto.
2. Verifica que la altura esté expresada en metros y el viento en m/s.
3. Extrae presión, altura, temperatura, punto de rocío y viento.
4. Descarta filas incompletas, no numéricas o no finitas.
5. Elimina niveles incompatibles con un perfil vertical ordenado.
6. Convierte dirección y rapidez del viento en componentes ``u`` y ``v``.
7. Construye un objeto ``SoundingProfile`` con diagnósticos
   termodinámicos y cinemáticos.

Formato admitido
----------------
PRES:
    Presión atmosférica en hPa.
HGHT:
    Altura geopotencial en metros.
TEMP:
    Temperatura ambiental en grados Celsius.
DWPT:
    Temperatura de punto de rocío en grados Celsius.
DRCT:
    Dirección meteorológica del viento en grados.
SPED:
    Rapidez del viento en m/s.

Por decisión de diseño, esta primera versión solo admite unidades métricas.
El formato aeronáutico, con altura en pies y viento en nudos, se rechaza
explícitamente para evitar conversiones implícitas o ambiguas.

Autor: Richard Perdomo Arcila
Fecha de desarrollo: 4 de agosto de 2026
Proyecto: Extensión HAILCAST 2D para trayectorias y riesgo en cultivos
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from hailcast2d.sounding_profile import SoundingProfile


def read_uwyo_metric_sounding(path: str | Path) -> SoundingProfile:
    """
    Lee un radiosondeo métrico de Wyoming Weather Web.

    Parameters
    ----------
    path:
        Ruta del archivo descargado mediante la opción ``Text listing``.
        El archivo debe usar altura en metros y viento en m/s.

    Returns
    -------
    SoundingProfile
        Perfil vertical normalizado con presión, altura, temperatura,
        punto de rocío y componentes horizontales del viento.

    Raises
    ------
    FileNotFoundError
        Si el archivo indicado no existe.
    ValueError
        Si no se encuentra la tabla, las unidades no son métricas o
        quedan menos de seis niveles válidos después del control de calidad.
    """

    # Se conserva la lectura del archivo separada del análisis para que
    # el parser pueda probarse directamente mediante cadenas de texto.
    text = Path(path).read_text(encoding="utf-8")
    columns = _parse_uwyo_metric_text(text)

    # SoundingProfile calcula posteriormente CAPE, CIN, LCL, LFC,
    # nivel de equilibrio, nivel de congelación y cizalladura.
    return SoundingProfile(
        pressure_hpa=columns["pressure_hpa"],
        height_m=columns["height_m"],
        temperature_c=columns["temperature_c"],
        dewpoint_c=columns["dewpoint_c"],
        u_wind_ms=columns["u_wind_ms"],
        v_wind_ms=columns["v_wind_ms"],
    )


def _parse_uwyo_metric_text(text: str) -> dict[str, np.ndarray]:
    """
    Extrae y normaliza las variables necesarias del listado métrico.

    Los niveles se conservan únicamente cuando la presión disminuye y
    la altura aumenta respecto del último nivel aceptado. Esta condición
    elimina presiones repetidas y pequeñas inconsistencias verticales
    presentes en algunos radiosondeos de alta resolución.

    Parameters
    ----------
    text:
        Contenido completo del archivo de Wyoming.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arreglos de presión, altura, temperatura, punto de rocío y
        componentes ``u`` y ``v`` del viento.
    """

    lines = text.splitlines()

    # La tabla puede estar precedida por metadatos de estación, fecha,
    # ubicación geográfica y enlaces del sitio web.
    header_index = _find_table_header(lines)
    _validate_metric_units(lines, header_index)

    rows = []

    # La línea inmediatamente posterior a la cabecera contiene unidades.
    # El recorrido comienza dos líneas después; los separadores se omiten.
    for line in lines[header_index + 2:]:
        stripped = line.strip()

        # Omite las líneas horizontales que delimitan la tabla.
        if stripped and set(stripped) == {"-"}:
            continue

        # Esta leyenda marca el final de la sección de observaciones.
        if stripped.startswith("Description of the columns"):
            break

        if not stripped:
            continue

        fields = stripped.split()

        # Se necesitan al menos las ocho columnas básicas:
        # PRES, HGHT, TEMP, DWPT, RELH, MIXR, DRCT y SPED.
        if len(fields) < 8:
            continue

        try:
            pressure, height, temperature, dewpoint = map(
                float,
                fields[:4],
            )
            direction = float(fields[6])
            speed = float(fields[7])
        except ValueError:
            # Las filas con indicadores no numéricos o valores faltantes
            # no pueden alimentar los cálculos de SoundingProfile.
            continue

        meteorological_values = (
            pressure,
            height,
            temperature,
            dewpoint,
            direction,
            speed,
        )

        if not all(
            np.isfinite(value)
            for value in meteorological_values
        ):
            continue

        if rows:
            previous_pressure, previous_height = rows[-1][:2]

            # SoundingProfile exige presión estrictamente decreciente y
            # altura estrictamente creciente. Se conserva la primera fila
            # válida cuando aparece una presión repetida.
            if (
                pressure >= previous_pressure
                or height <= previous_height
            ):
                continue

        rows.append(meteorological_values)

    # SoundingProfile requiere al menos seis niveles para calcular
    # diagnósticos termodinámicos y verificar la cobertura vertical.
    if len(rows) < 6:
        raise ValueError(
            "El listado de Wyoming no contiene al menos seis niveles "
            "métricos válidos."
        )

    values = np.asarray(rows, dtype=float)

    direction_radians = np.deg2rad(values[:, 4])
    speed_ms = values[:, 5]

    # La dirección meteorológica indica desde dónde sopla el viento.
    # Por ello se emplean signos negativos al convertirla en:
    #
    # u > 0: flujo dirigido hacia el este.
    # v > 0: flujo dirigido hacia el norte.
    u_wind_ms = -speed_ms * np.sin(direction_radians)
    v_wind_ms = -speed_ms * np.cos(direction_radians)

    return {
        "pressure_hpa": values[:, 0],
        "height_m": values[:, 1],
        "temperature_c": values[:, 2],
        "dewpoint_c": values[:, 3],
        "u_wind_ms": u_wind_ms,
        "v_wind_ms": v_wind_ms,
    }


def _find_table_header(lines: Sequence[str]) -> int:
    """
    Localiza la cabecera de la tabla meteorológica.

    La búsqueda se basa en las seis variables indispensables para
    HAILCAST y no depende de una posición fija dentro del archivo.
    """

    required = {
        "PRES",
        "HGHT",
        "TEMP",
        "DWPT",
        "DRCT",
        "SPED",
    }

    for index, line in enumerate(lines):
        available_columns = set(line.split())

        if required.issubset(available_columns):
            return index

    raise ValueError(
        "No se encontró la cabecera esperada del listado de Wyoming."
    )


def _validate_metric_units(
    lines: Sequence[str],
    header_index: int,
) -> None:
    """
    Verifica que el listado utilice exclusivamente unidades métricas.

    Esta validación evita interpretar accidentalmente pies como metros
    o nudos como metros por segundo.
    """

    if header_index + 1 >= len(lines):
        raise ValueError(
            "El listado de Wyoming no declara sus unidades."
        )

    units_line = lines[header_index + 1].split()

    expected = [
        "hPa",
        "m",
        "C",
        "C",
        "%",
        "g/kg",
        "deg",
        "m/s",
    ]

    if units_line[:8] != expected:
        raise ValueError(
            "Solo se admite el Text listing métrico de Wyoming "
            "(altura en m y viento en m/s)."
        )