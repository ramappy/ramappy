from __future__ import annotations

import numpy as np

from ramappy.core.spectral_map import SpectralMap
from ramappy.io.matlab import read_matlab_witec


def test_read_matlab_witec_coerces_numpy_dimension_scalars(monkeypatch):
    x_data = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
    fake_payload = {
        "fake_export": {
            "imagesize": np.array([2, 3], dtype=np.int64),
            "data": np.arange(24, dtype=np.float32).reshape(6, 4),
            "axisscale": [None, [x_data, "nm"]],
        }
    }

    monkeypatch.setattr("ramappy.io.matlab.scipy.io.loadmat", lambda *_args, **_kwargs: fake_payload)

    out = read_matlab_witec("dummy.mat", as_hsi=True)

    assert isinstance(out, SpectralMap)
    assert out.img_height == 2
    assert out.img_width == 3
    assert isinstance(out.img_height, int)
    assert isinstance(out.img_width, int)
    assert out.data.shape == (6, 4)
    assert out.x_axis_unit is not None
    assert out.x_axis_unit.quantity == "wavelength"
    assert out.x_axis_unit.unit == "nm"
