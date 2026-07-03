from __future__ import annotations

import numpy as np

from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum
from ramappy.io.horiba import read_horiba5
from ramappy.units import Unit


class _FakeHoriba:
    def __init__(self, _filepath_or_buffer, *, read_tables: bool = True):
        self.read_tables = read_tables
        self.comment = "fake-horiba"
        self.ndim = 3
        self.Spectr = np.array([100.0, 110.0, 120.0], dtype=np.float32)
        self.X = np.array([10.0, 10.5, 11.0], dtype=np.float32)
        self.Y = np.array([20.0, 21.0], dtype=np.float32)
        self.Intens = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
        self.axes_names = ["Y", "X", "Spectr", "Intens"]
        self.axes_units = {
            "Y": "µm",
            "X": "µm",
            "Spectr": "1/cm",
            "Intens": "Cnt/sec",
        }
        self.tables = {
            "Info": {
                "first": {"Value": "alpha"},
                "second": {"Value": "beta"},
            },
            "Params": {
                "laser": {"Value": 532.0, "Unit": "nm"},
            },
        }

    @property
    def img_shape(self):
        return self.Intens.shape[:2]


def test_read_horiba5_supports_as_hsi_false(monkeypatch):
    monkeypatch.setattr("ramappy.io.horiba.Horiba5", _FakeHoriba)

    out = read_horiba5("dummy.ngs", as_hsi=False)

    assert isinstance(out, Spectrum)
    assert out.data.shape == (2, 3, 3)
    assert out.name == "fake-horiba"
    assert out.spectral_metadata.description == "fake-horiba"
    assert out.spectral_metadata.original_filename == "dummy.ngs"
    assert "input_filename" not in out.metadata
    assert "pixel_physical_size_x" not in out.metadata
    assert "pixel_physical_size_y" not in out.metadata


def test_read_horiba5_supports_as_hsi_true(monkeypatch):
    monkeypatch.setattr("ramappy.io.horiba.Horiba5", _FakeHoriba)

    out = read_horiba5("dummy.ngs", as_hsi=True)

    assert isinstance(out, SpectralMap)
    assert out.img_height == 2
    assert out.img_width == 3
    assert out.data.shape == (6, 3)
    assert out.spatial_grid.pixel_size_x == 0.5
    assert out.spatial_grid.pixel_size_y == 1.0
    assert out.spatial_grid.origin_x == 10.0
    assert out.spatial_grid.origin_y == 20.0
    assert out.spatial_grid.spatial_unit == Unit.MICROMETER
    assert out.smap_metadata.description == "fake-horiba"
    assert "vendor" not in out.metadata
    assert out.smap_metadata.original_filename == "dummy.ngs"
    assert "input_filename" not in out.metadata
    assert out.metadata["Info"] == {"first": "alpha", "second": "beta"}
    assert out.metadata["Params"]["laser"] == {"value": 532.0, "unit": "nm"}
    assert "tables" not in out.metadata
