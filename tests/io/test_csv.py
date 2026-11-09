from io import BytesIO

import numpy as np
import pytest

from ramappy.io.csv import read_csv


@pytest.mark.parametrize("fmt", ["wide", "long"])
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("raster", [[0, 2, 4], [1, 3, 5]]),
        ("bidirectional", [[0, 3, 4], [1, 2, 5]]),
    ],
)
def test_csv_non_square_acquisition_order(fmt, pattern, expected):
    rows = np.arange(12, dtype=float).reshape(6, 2)
    buffer = BytesIO()
    np.savetxt(buffer, rows.T if fmt == "wide" else rows, delimiter=",")
    buffer.seek(0)
    m = read_csv(buffer, csv_format=fmt, has_x_axis=False, img_width=3, img_height=2, scan_pattern=pattern)
    np.testing.assert_array_equal(m.cube[:, :, 0], np.array(expected) * 2)
    np.testing.assert_array_equal(m.cube[:, :, 1], np.array(expected) * 2 + 1)


@pytest.mark.parametrize("pattern", ["raster", "bidirectional"])
def test_csv_square_unchanged(pattern):
    rows = np.arange(18, dtype=float).reshape(9, 2)
    buffer = BytesIO()
    np.savetxt(buffer, rows.T, delimiter=",")
    buffer.seek(0)
    m = read_csv(buffer, has_x_axis=False, img_width=3, img_height=3, scan_pattern=pattern)
    old = rows.reshape(3, 3, 2)
    if pattern == "bidirectional":
        old[1::2] = old[1::2, ::-1]
    np.testing.assert_array_equal(m.cube, np.flipud(np.rot90(old, axes=(0, 1))))
