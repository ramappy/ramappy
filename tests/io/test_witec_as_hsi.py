from __future__ import annotations

import numpy as np

from ramappy.core.images2d import SpatialGrid
from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum
from ramappy.io.witec import read_witec
from ramappy.units import Unit


class _FakeWITec:
    def __init__(self, _filepath_or_buffer):
        self.x = np.array([100.0, 110.0, 120.0], dtype=np.float32)
        self.x_unit = "1/cm"
        self.y_unit = "CCD cts"
        # shape: (height, width, spectral)
        self.data = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
        self.name = "fake-witec"
        self.laser_wavelength = 532.0
        self.root_name = "WITec Project"
        self.data_classes = ["TDGraph", "TDSpectralInterpretation", "TDSpaceTransformation"]
        self.spatial_transform = {
            "scale": [0.168, 0.0, 0.0, 0.0, -0.188, 0.0, 0.0, 0.0, 1.0],
            "model_origin": [0.0, 0.0, 0.0],
            "world_origin": [1.0, 2.0, 0.0],
        }

        self.spatial_grid = SpatialGrid(
            pixel_size_y=0.188,
            pixel_size_x=0.168,
            spatial_unit=Unit.MICROMETER,
        )


def test_read_witec_supports_as_hsi_false(monkeypatch):
    monkeypatch.setattr("ramappy.io.witec.WITec", _FakeWITec)

    out = read_witec("dummy.wip", as_hsi=False)

    assert isinstance(out, Spectrum)
    assert out.data.shape == (6, 3)
    assert out.name == "fake-witec"


def test_read_witec_supports_as_hsi_true(monkeypatch):
    monkeypatch.setattr("ramappy.io.witec.WITec", _FakeWITec)

    out = read_witec("dummy.wip", as_hsi=True)

    assert isinstance(out, SpectralMap)
    assert out.img_height == 2
    assert out.img_width == 3
    assert out.data.shape == (6, 3)
    assert out.spatial_grid.pixel_size_y == 0.188
    assert out.spatial_grid.pixel_size_x == 0.168
    assert out.spatial_grid.spatial_unit == Unit.MICROMETER
    assert out.smap_metadata.instrument.laser_wavelength_nm == 532.0
    assert out.metadata["witec_root_type"] == "WITec Project"
    assert out.metadata["witec_spatial_transform"]["world_origin"] == [1.0, 2.0, 0.0]
