import tempfile
import unittest
from pathlib import Path

import numpy as np

from hailcast2d.sounding_reader import (
    _parse_igra_level,
    _parse_igra_text,
    read_igra_sounding,
)


def igra_level(
    pressure_pa,
    height_m,
    temperature_tenths,
    dewpoint_depression_tenths,
    wind_direction_deg,
    wind_speed_tenths,
):
    """Construye una línea IGRA de 52 columnas para las pruebas."""

    line = [" "] * 52
    line[0:2] = "10"
    line[3:8] = f"{-9999:5d}"
    line[9:15] = f"{pressure_pa:6d}"
    line[16:21] = f"{height_m:5d}"
    line[21] = "B"
    line[22:27] = f"{temperature_tenths:5d}"
    line[27] = "B"
    line[28:33] = f"{-9999:5d}"
    line[34:39] = f"{dewpoint_depression_tenths:5d}"
    line[40:45] = f"{wind_direction_deg:5d}"
    line[46:51] = f"{wind_speed_tenths:5d}"

    return "".join(line)


PRESSURE_PA = [
    100000,
    92500,
    85000,
    70000,
    50000,
    30000,
    20000,
]
HEIGHT_M = [100, 750, 1500, 3000, 5600, 9200, 11800]
TEMPERATURE_TENTHS = [300, 240, 180, 40, -120, -400, -560]
DEWPOINT_DEPRESSION_TENTHS = [80, 60, 40, 60, 80, 80, 90]
WIND_DIRECTION_DEG = [180, 200, 225, 250, 270, 285, 300]
WIND_SPEED_TENTHS = [50, 80, 100, 140, 180, 240, 300]

LEVELS = [
    igra_level(*values)
    for values in zip(
        PRESSURE_PA,
        HEIGHT_M,
        TEMPERATURE_TENTHS,
        DEWPOINT_DEPRESSION_TENTHS,
        WIND_DIRECTION_DEG,
        WIND_SPEED_TENTHS,
    )
]

IGRA_SOUNDING = "\n".join(
    [
        "#ARM00087344 2021 01 27 12 1130    7 "
        "ncdc-gts ncdc-gts -312966  -642119",
        *LEVELS,
    ]
)


class TestIgraReader(unittest.TestCase):

    def test_decodes_real_fixed_width_example(self):
        record = _parse_igra_level(
            "10 -9999  92500   761B  230B-9999    70   200    72 "
        )

        self.assertEqual(record["pressure_pa"], 92500.0)
        self.assertEqual(record["height_m"], 761.0)
        self.assertEqual(record["temperature_c"], 23.0)
        self.assertEqual(record["dewpoint_depression_c"], 7.0)
        self.assertEqual(record["wind_direction_deg"], 200.0)
        self.assertEqual(record["wind_speed_ms"], 7.2)

    def test_parses_requested_sounding_and_converts_units(self):
        columns = _parse_igra_text(
            IGRA_SOUNDING,
            station_id="ARM00087344",
            year=2021,
            month=1,
            day=27,
            hour=12,
        )

        np.testing.assert_allclose(
            columns["pressure_hpa"],
            np.asarray(PRESSURE_PA) / 100.0,
        )
        np.testing.assert_allclose(columns["height_m"], HEIGHT_M)
        np.testing.assert_allclose(
            columns["dewpoint_c"],
            np.asarray(TEMPERATURE_TENTHS) / 10.0
            - np.asarray(DEWPOINT_DEPRESSION_TENTHS) / 10.0,
        )

    def test_rejects_missing_requested_sounding(self):
        with self.assertRaisesRegex(ValueError, "No se encontró"):
            _parse_igra_text(
                IGRA_SOUNDING,
                station_id="ARM00087344",
                year=2021,
                month=1,
                day=28,
                hour=12,
            )

    def test_builds_sounding_profile_from_igra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ARM00087344-data.txt"
            path.write_text(IGRA_SOUNDING, encoding="utf-8")

            sounding = read_igra_sounding(
                path,
                station_id="ARM00087344",
                year=2021,
                month=1,
                day=27,
                hour=12,
            )

        self.assertEqual(sounding.pressure_hpa.size, 7)
        self.assertAlmostEqual(sounding.height_m[0], 100.0)
        self.assertGreater(sounding.bulk_shear_0_6km_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
