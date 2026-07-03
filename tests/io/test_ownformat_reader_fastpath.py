from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ramappy.io.ownformat import read_ownformat


def test_read_ownformat_full_raster_fastpath_no_reorder():
    # 2x3 image -> 6 pixels, already in raster order.
    wn = ["100.0", "110.0", "120.0", "130.0"]
    data = np.arange(6 * 4, dtype=float).reshape(6, 4)

    df = pl.DataFrame(
        {
            wn[0]: data[:, 0],
            wn[1]: data[:, 1],
            wn[2]: data[:, 2],
            wn[3]: data[:, 3],
            "x_pos": [0, 1, 2, 0, 1, 2],
            "y_pos": [0, 0, 0, 1, 1, 1],
        }
    )

    hsi = read_ownformat(df, reorder_using_xy_pos=False, as_hsi=True)

    assert hsi.img_width == 3
    assert hsi.img_height == 2
    np.testing.assert_allclose(hsi.x, np.array([100.0, 110.0, 120.0, 130.0]))
    np.testing.assert_allclose(hsi.data, data)


def test_read_ownformat_no_reorder_requires_full_raster_or_pixel_index():
    # Missing one row (5 instead of 6) -> must error when reorder_using_xy_pos=False
    wn = ["100.0", "110.0"]
    data = np.arange(5 * 2, dtype=float).reshape(5, 2)

    df = pl.DataFrame(
        {
            wn[0]: data[:, 0],
            wn[1]: data[:, 1],
            "x_pos": [0, 1, 2, 0, 1],
            "y_pos": [0, 0, 0, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="not a full raster"):
        read_ownformat(df, reorder_using_xy_pos=False, as_hsi=True)
