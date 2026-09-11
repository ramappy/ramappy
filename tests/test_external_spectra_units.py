"""Tests for unit propagation in the add_spectrum pipeline step.

Verifies that external spectra correctly inherit the parent SpectralMap's
axis units when their own unit metadata is absent (e.g. a plain CSV with no
header information), and that spectra carrying explicit units are kept
unchanged.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum
from ramappy.pipeline.steps.external_spectra import add_spectrum
from ramappy.units import Quantity, QuantityUnit, Unit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raman_map() -> SpectralMap:
    """A 2x2 Raman map with explicit x_axis_unit and data_unit."""
    x = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    data = np.ones((4, 4), dtype=np.float32)
    return SpectralMap(
        x=x,
        data=data,
        img_width=2,
        img_height=2,
        x_axis_unit=QuantityUnit(Quantity.RAMANSHIFT, Unit.CM_1),
        data_unit=QuantityUnit(Quantity.INTENSITY, Unit.COUNTS),
    )


def _make_unitless_spectrum(x=None) -> Spectrum:
    """Return a Spectrum with no units (simulates a bare CSV import)."""
    if x is None:
        x = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    return Spectrum(x=x, data=np.ones((1, len(x)), dtype=np.float32), x_axis_unit=None, data_unit=None)


def _make_spectrum_with_units(x_unit: QuantityUnit, d_unit: QuantityUnit) -> Spectrum:
    """Return a Spectrum carrying explicit units."""
    x = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    return Spectrum(x=x, data=np.ones((1, len(x)), dtype=np.float32), x_axis_unit=x_unit, data_unit=d_unit)


# ---------------------------------------------------------------------------
# Helper: invoke add_spectrum with a mocked reader
# ---------------------------------------------------------------------------


def _run_add_spectrum(
    spectral_map: SpectralMap,
    returned_spectrum: Spectrum,
    *,
    res_id: str = "ext_sp",
    format_name: str = "csv",
    file_path: Path | None = Path("/fake/file.csv"),
    format_params: dict | None = None,
):
    """Call add_spectrum with a mocked IO reader that returns *returned_spectrum*."""
    if format_params is None:
        format_params = {}

    # Build a fake InputFormat whose .read() always returns the given spectrum.
    class FakeFormat:
        def read(self, _path, _params, *, as_hsi=None):
            return returned_spectrum

    with patch(
        "ramappy.pipeline.steps.external_spectra.InputFormatRegistry.get_format",
        return_value=FakeFormat(),
    ):
        add_spectrum(
            spectral_map,
            file_path=file_path,
            format_name=format_name,
            res_id=res_id,
            as_hsi=False,
            format_params=format_params,
        )


# ---------------------------------------------------------------------------
# Tests: unit inheritance when the imported spectrum has no units
# ---------------------------------------------------------------------------


def test_unitless_spectrum_inherits_x_axis_unit_from_map(raman_map: SpectralMap):
    """x_axis_unit must be inherited from the parent map when None."""
    sp = _make_unitless_spectrum()
    assert sp.x_axis_unit is None  # pre-condition

    _run_add_spectrum(raman_map, sp)

    assert raman_map.spectra["ext_sp"].x_axis_unit == raman_map.x_axis_unit


def test_unitless_spectrum_inherits_data_unit_from_map(raman_map: SpectralMap):
    """data_unit must be inherited from the parent map when None."""
    sp = _make_unitless_spectrum()
    assert sp.data_unit is None  # pre-condition

    _run_add_spectrum(raman_map, sp)

    assert raman_map.spectra["ext_sp"].data_unit == raman_map.data_unit


def test_unitless_spectrum_inherits_both_units(raman_map: SpectralMap):
    """Both units are inherited in a single call."""
    sp = _make_unitless_spectrum()

    _run_add_spectrum(raman_map, sp)

    stored = raman_map.spectra["ext_sp"]
    assert stored.x_axis_unit == QuantityUnit(Quantity.RAMANSHIFT, Unit.CM_1)
    assert stored.data_unit == QuantityUnit(Quantity.INTENSITY, Unit.COUNTS)


# ---------------------------------------------------------------------------
# Tests: explicit units on the imported spectrum are NOT overridden
# ---------------------------------------------------------------------------


def test_explicit_x_axis_unit_is_not_overridden(raman_map: SpectralMap):
    """If the imported spectrum already has x_axis_unit, keep it."""
    explicit_unit = QuantityUnit(Quantity.WAVELENGTH, Unit.NANOMETER)
    sp = _make_spectrum_with_units(explicit_unit, QuantityUnit(Quantity.INTENSITY, Unit.ARBITRARY_UNITS))

    _run_add_spectrum(raman_map, sp)

    assert raman_map.spectra["ext_sp"].x_axis_unit == explicit_unit


def test_explicit_data_unit_is_not_overridden(raman_map: SpectralMap):
    """If the imported spectrum already has data_unit, keep it."""
    explicit_data_unit = QuantityUnit(Quantity.RELATIVE_INTENSITY, Unit.PERCENT)
    sp = _make_spectrum_with_units(QuantityUnit(Quantity.RAMANSHIFT, Unit.CM_1), explicit_data_unit)

    _run_add_spectrum(raman_map, sp)

    assert raman_map.spectra["ext_sp"].data_unit == explicit_data_unit


# ---------------------------------------------------------------------------
# Tests: edge cases — map has no units itself
# ---------------------------------------------------------------------------


def test_unitless_map_does_not_crash_for_unitless_spectrum():
    """If the map also has no units, no inheritance occurs and no error is raised."""
    x = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    data = np.ones((4, 4), dtype=np.float32)
    unitless_map = SpectralMap(
        x=x,
        data=data,
        img_width=2,
        img_height=2,
        x_axis_unit=None,
        data_unit=None,
    )
    sp = _make_unitless_spectrum()

    _run_add_spectrum(unitless_map, sp)

    stored = unitless_map.spectra["ext_sp"]
    # Both remain None — nothing to inherit.
    assert stored.x_axis_unit is None
    assert stored.data_unit is None


def test_map_without_data_unit_does_not_overwrite_spectrum_data_unit():
    """x_axis_unit is inherited but data_unit stays None when map has none."""
    x = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    data = np.ones((4, 4), dtype=np.float32)
    partial_map = SpectralMap(
        x=x,
        data=data,
        img_width=2,
        img_height=2,
        x_axis_unit=QuantityUnit(Quantity.RAMANSHIFT, Unit.CM_1),
        data_unit=None,
    )
    sp = _make_unitless_spectrum()

    _run_add_spectrum(partial_map, sp)

    stored = partial_map.spectra["ext_sp"]
    assert stored.x_axis_unit == QuantityUnit(Quantity.RAMANSHIFT, Unit.CM_1)
    assert stored.data_unit is None


# ---------------------------------------------------------------------------
# Tests: spectrum is actually stored in the map under the correct key
# ---------------------------------------------------------------------------


def test_spectrum_is_stored_under_correct_res_id(raman_map: SpectralMap):
    """The spectrum is accessible via the res_id key."""
    sp = _make_unitless_spectrum()

    _run_add_spectrum(raman_map, sp, res_id="my_reference")

    assert "my_reference" in raman_map.spectra


def test_multiple_spectra_independent_unit_inheritance(raman_map: SpectralMap):
    """Adding two unitless spectra in sequence assigns units to both."""
    sp1 = _make_unitless_spectrum()
    sp2 = _make_unitless_spectrum()

    _run_add_spectrum(raman_map, sp1, res_id="sp_a")
    _run_add_spectrum(raman_map, sp2, res_id="sp_b")

    assert raman_map.spectra["sp_a"].x_axis_unit == raman_map.x_axis_unit
    assert raman_map.spectra["sp_b"].x_axis_unit == raman_map.x_axis_unit


# ---------------------------------------------------------------------------
# Tests: error conditions
# ---------------------------------------------------------------------------


def test_missing_file_path_raises():
    """add_spectrum should raise ValueError when file_path is None."""
    x = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    data = np.ones((4, 4), dtype=np.float32)
    sm = SpectralMap(x=x, data=data, img_width=2, img_height=2)

    with pytest.raises(ValueError, match="file_path"):
        add_spectrum(sm, file_path=None, format_name="csv", res_id="x", as_hsi=False, format_params={})


def test_unknown_format_raises():
    """add_spectrum should raise ValueError for an unregistered format name."""
    x = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    data = np.ones((4, 4), dtype=np.float32)
    sm = SpectralMap(x=x, data=data, img_width=2, img_height=2)

    with pytest.raises(ValueError, match="not supported"):
        add_spectrum(
            sm,
            file_path=Path("/fake/file.xyz"),
            format_name="__nonexistent_format__",
            res_id="x",
            as_hsi=False,
            format_params={},
        )
