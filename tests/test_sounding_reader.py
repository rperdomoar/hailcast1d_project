import tempfile
import unittest
from pathlib import Path

import numpy as np

from hailcast2d.sounding_reader import (
    _parse_uwyo_metric_text,
    read_uwyo_metric_sounding,
)


METRIC_SOUNDING = """
Wyoming Weather Web
Observations for Station 87585 at 12 UTC 20 Jul 2026
--------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SPED
    hPa      m      C      C      %   g/kg    deg    m/s
--------------------------------------------------------
 1000.0     100   30.0   22.0     62  16.88    180    5.0
  925.0     750   24.0   18.0     69  14.01    225   10.0
  925.0     755   24.0   18.0     69  14.01    225   10.0
  850.0    1500   18.0   14.0     77  11.81    270   15.0
  700.0    3000    4.0   -2.0     65   4.68    315   20.0
  500.0    5600  -12.0  -20.0     51   1.56      0   25.0
  300.0    9200  -40.0  -48.0     39   0.23     90   30.0
  200.0   11800  -56.0  -65.0     30   0.04    135   35.0
Description of the columns
"""


class TestUwyoMetricSoundingReader(unittest.TestCase):

    def test_parses_metric_columns_and_removes_duplicate_pressure(self):
        columns = _parse_uwyo_metric_text(METRIC_SOUNDING)

        np.testing.assert_allclose(
            columns["pressure_hpa"],
            [1000, 925, 850, 700, 500, 300, 200],
        )
        np.testing.assert_allclose(
            columns["height_m"],
            [100, 750, 1500, 3000, 5600, 9200, 11800],
        )

    def test_converts_meteorological_wind_to_uv_components(self):
        columns = _parse_uwyo_metric_text(METRIC_SOUNDING)

        self.assertAlmostEqual(columns["u_wind_ms"][0], 0.0)
        self.assertAlmostEqual(columns["v_wind_ms"][0], 5.0)
        self.assertAlmostEqual(columns["u_wind_ms"][2], 15.0)
        self.assertAlmostEqual(columns["v_wind_ms"][2], 0.0)

    def test_rejects_aviation_units(self):
        aviation = METRIC_SOUNDING.replace(
            "hPa      m      C      C      %   g/kg    deg    m/s",
            "hPa     ft      C      C      %   g/kg    deg   knot",
        )

        with self.assertRaisesRegex(ValueError, "Text listing métrico"):
            _parse_uwyo_metric_text(aviation)

    def test_builds_sounding_profile_from_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sounding.txt"
            path.write_text(METRIC_SOUNDING, encoding="utf-8")

            sounding = read_uwyo_metric_sounding(path)

        self.assertEqual(sounding.pressure_hpa.size, 7)
        self.assertAlmostEqual(sounding.height_m[0], 100.0)
        self.assertGreater(sounding.bulk_shear_0_6km_ms, 0.0)


if __name__ == "__main__":
    unittest.main()