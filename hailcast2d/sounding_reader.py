# ================================================================
# hailcast2d/sounding_reader.py
# ================================================================
"""
Lectura y normalización de radiosondeos para HAILCAST 2D.

Este módulo convierte observaciones verticales de Wyoming Weather Web y
NOAA/NCEI IGRA 2.2 en objetos ``SoundingProfile`` compatibles con
HAILCAST 2D.

El procesamiento realiza las siguientes operaciones:

1. Localiza la tabla meteorológica dentro del archivo de texto.
2. Verifica que la altura esté expresada en metros y el viento en m/s.
3. Extrae presión, altura, temperatura, punto de rocío y viento.
4. Descarta filas incompletas, no numéricas o no finitas.
5. Elimina niveles incompatibles con un perfil vertical ordenado.
6. Convierte dirección y rapidez del viento en componentes ``u`` y ``v``.
7. Construye un objeto ``SoundingProfile`` con diagnósticos
   termodinámicos y cinemáticos.

Formatos admitidos
------------------
Wyoming Weather Web:
    ``Text listing`` con unidades métricas.
NOAA/NCEI IGRA 2.2:
    Archivo histórico ``*-data.txt`` con registros de ancho fijo.

Variables normalizadas
----------------------
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

Para Wyoming solo se admiten unidades métricas. El formato aeronáutico,
con altura en pies y viento en nudos, se rechaza explícitamente para evitar
conversiones implícitas o ambiguas. En IGRA se aplican las escalas y códigos
de datos faltantes definidos oficialmente por NOAA/NCEI.

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


def read_igra_sounding(
    path,
    station_id,
    year,
    month,
    day,
    hour,
):
    """Lee un lanzamiento individual de un archivo histórico IGRA 2.2.

    Parameters
    ----------
    path:
        Archivo ``*-data.txt`` descomprimido de NOAA/NCEI.
    station_id:
        Identificador IGRA de once caracteres, por ejemplo
        ``ARM00087344`` para Córdoba Aero.
    year, month, day, hour:
        Fecha y hora nominales UTC del lanzamiento.

    Returns
    -------
    SoundingProfile
        Perfil vertical normalizado en hPa, m, grados Celsius y m/s.

    Notes
    -----
    IGRA almacena presión en Pa, temperatura y depresión del punto de
    rocío en décimas de grado Celsius, y rapidez del viento en décimas
    de m/s. El valor ``-9999`` representa un dato faltante.
    """

    text = Path(path).read_text(encoding="utf-8")
    columns = _parse_igra_text(
        text,
        station_id=station_id,
        year=year,
        month=month,
        day=day,
        hour=hour,
    )

    return SoundingProfile(
        pressure_hpa=columns["pressure_hpa"],
        height_m=columns["height_m"],
        temperature_c=columns["temperature_c"],
        dewpoint_c=columns["dewpoint_c"],
        u_wind_ms=columns["u_wind_ms"],
        v_wind_ms=columns["v_wind_ms"],
    )


def _parse_igra_text(text, station_id, year, month, day, hour):
    """Localiza, decodifica y normaliza un lanzamiento IGRA."""

    lines = text.splitlines()
    level_lines = _find_igra_sounding(
        lines,
        station_id=str(station_id),
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
    )
    records = [_parse_igra_level(line) for line in level_lines]

    # Pares observados de presión-altura para reconstruir las alturas
    # faltantes de los niveles termodinámicos.
    pressure_height = {
        record["pressure_pa"]: record["height_m"]
        for record in records
        if record["pressure_pa"] is not None
        and record["height_m"] is not None
    }

    if len(pressure_height) < 2:
        raise ValueError(
            "El sondeo IGRA no contiene suficientes pares presión-altura."
        )

    reference_pressure = np.asarray(
        sorted(pressure_height),
        dtype=float,
    )
    reference_log_pressure = np.log(reference_pressure)
    reference_height = np.asarray(
        [pressure_height[p] for p in reference_pressure],
        dtype=float,
    )

    # Se aprovechan tanto los vientos informados sobre superficies de
    # presión como los informados sobre alturas fijas.
    wind_by_pressure = {}
    for record in records:
        if (
            record["wind_direction_deg"] is None
            or record["wind_speed_ms"] is None
        ):
            continue

        pressure_pa = record["pressure_pa"]

        if pressure_pa is None and record["height_m"] is not None:
            pressure_pa = float(
                np.exp(
                    _interpolate_with_extrapolation(
                        record["height_m"],
                        reference_height[::-1],
                        reference_log_pressure[::-1],
                    )
                )
            )

        if pressure_pa is None:
            continue

        direction = np.deg2rad(record["wind_direction_deg"])
        speed = record["wind_speed_ms"]
        wind_by_pressure[pressure_pa] = (
            -speed * np.sin(direction),
            -speed * np.cos(direction),
        )

    if len(wind_by_pressure) < 2:
        raise ValueError(
            "El sondeo IGRA no contiene suficientes niveles de viento."
        )

    wind_pressure = np.asarray(sorted(wind_by_pressure), dtype=float)
    wind_log_pressure = np.log(wind_pressure)
    wind_u = np.asarray(
        [wind_by_pressure[p][0] for p in wind_pressure],
        dtype=float,
    )
    wind_v = np.asarray(
        [wind_by_pressure[p][1] for p in wind_pressure],
        dtype=float,
    )

    rows = []
    for record in records:
        pressure_pa = record["pressure_pa"]
        temperature_c = record["temperature_c"]
        dewpoint_depression_c = record["dewpoint_depression_c"]

        # El perfil termodinámico necesita simultáneamente presión,
        # temperatura y humedad.
        if (
            pressure_pa is None
            or temperature_c is None
            or dewpoint_depression_c is None
        ):
            continue

        log_pressure = np.log(pressure_pa)
        height_m = float(
            _interpolate_with_extrapolation(
                log_pressure,
                reference_log_pressure,
                reference_height,
            )
        )
        u_wind_ms = float(
            _interpolate_with_extrapolation(
                log_pressure,
                wind_log_pressure,
                wind_u,
            )
        )
        v_wind_ms = float(
            _interpolate_with_extrapolation(
                log_pressure,
                wind_log_pressure,
                wind_v,
            )
        )

        rows.append(
            (
                pressure_pa / 100.0,
                height_m,
                temperature_c,
                temperature_c - dewpoint_depression_c,
                u_wind_ms,
                v_wind_ms,
            )
        )

    rows.sort(key=lambda row: row[0], reverse=True)
    ordered_rows = []

    for row in rows:
        if ordered_rows:
            previous_pressure, previous_height = ordered_rows[-1][:2]
            if row[0] >= previous_pressure or row[1] <= previous_height:
                continue
        ordered_rows.append(row)

    if len(ordered_rows) < 6:
        raise ValueError(
            "El sondeo IGRA no contiene al menos seis niveles "
            "termodinámicos completos."
        )

    values = np.asarray(ordered_rows, dtype=float)

    return {
        "pressure_hpa": values[:, 0],
        "height_m": values[:, 1],
        "temperature_c": values[:, 2],
        "dewpoint_c": values[:, 3],
        "u_wind_ms": values[:, 4],
        "v_wind_ms": values[:, 5],
    }


def _find_igra_sounding(
    lines,
    station_id,
    year,
    month,
    day,
    hour,
):
    """Devuelve las líneas de niveles del lanzamiento solicitado."""

    target = (
        station_id,
        year,
        month,
        day,
        hour,
    )

    for index, line in enumerate(lines):
        if not line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) < 7:
            continue

        try:
            current = (
                fields[0][1:],
                int(fields[1]),
                int(fields[2]),
                int(fields[3]),
                int(fields[4]),
            )
            number_of_levels = int(fields[6])
        except ValueError:
            continue

        if current != target:
            continue

        end_index = index + 1 + number_of_levels
        level_lines = lines[index + 1:end_index]

        if len(level_lines) != number_of_levels:
            raise ValueError(
                "El sondeo IGRA está truncado: la cabecera declara "
                f"{number_of_levels} niveles y se encontraron "
                f"{len(level_lines)}."
            )

        return level_lines

    raise ValueError(
        "No se encontró el sondeo IGRA solicitado: "
        f"{station_id} {year:04d}-{month:02d}-{day:02d} "
        f"{hour:02d} UTC."
    )


def _parse_igra_level(line):
    """Decodifica una línea de nivel según el formato fijo IGRA 2.2."""

    if len(line) < 51:
        line = line.ljust(52)

    pressure_pa = _igra_value(line[9:15])
    height_m = _igra_value(line[16:21])
    temperature_tenths = _igra_value(line[22:27])
    dewpoint_tenths = _igra_value(line[34:39])
    wind_direction = _igra_value(line[40:45])
    wind_speed_tenths = _igra_value(line[46:51])

    return {
        "pressure_pa": pressure_pa,
        "height_m": height_m,
        "temperature_c": (
            temperature_tenths / 10.0
            if temperature_tenths is not None
            else None
        ),
        "dewpoint_depression_c": (
            dewpoint_tenths / 10.0
            if dewpoint_tenths is not None
            else None
        ),
        "wind_direction_deg": wind_direction,
        "wind_speed_ms": (
            wind_speed_tenths / 10.0
            if wind_speed_tenths is not None
            else None
        ),
    }


def _igra_value(field):
    """Convierte un campo entero IGRA y traduce ``-9999`` a ``None``."""

    try:
        value = int(field)
    except ValueError:
        return None

    if value == -9999:
        return None

    return float(value)


def _interpolate_with_extrapolation(value, x, y):
    """Interpola linealmente y prolonga la pendiente en los extremos."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    value_array = np.asarray(value, dtype=float)
    result = np.interp(value_array, x, y)

    lower_slope = (y[1] - y[0]) / (x[1] - x[0])
    upper_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])

    result = np.where(
        value_array < x[0],
        y[0] + lower_slope * (value_array - x[0]),
        result,
    )
    result = np.where(
        value_array > x[-1],
        y[-1] + upper_slope * (value_array - x[-1]),
        result,
    )

    if np.ndim(value) == 0:
        return float(np.asarray(result))

    return np.asarray(result, dtype=float)

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