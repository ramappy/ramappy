import numpy as np
import pytest

from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum
from ramappy.processing.resample import StepResampleParams, resample


def test_resample_spectrum_step():
    x = np.linspace(100.0, 200.0, 11)  # step 10.0
    data = np.ones((2, 11), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=1)

    resample(smap, spectrum_step=5.0)
    assert len(smap.x) == 21
    np.testing.assert_allclose(smap.x[1] - smap.x[0], 5.0)


def test_resample_x_grid():
    x = np.linspace(100.0, 200.0, 10)
    data = np.ones((2, 10), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=1)

    grid = [120.0, 150.0, 180.0]
    resample(smap, x_grid=grid)
    assert len(smap.x) == 3
    np.testing.assert_allclose(smap.x, grid)


def test_resample_ref_spectrum_id_external():
    x1 = np.linspace(100.0, 200.0, 10)
    data1 = np.ones((2, 10), dtype=float)
    smap = SpectralMap(x=x1, data=data1, img_width=2, img_height=1)

    ref_x = np.array([100.0, 125.0, 150.0, 175.0, 200.0])
    ref_spec = Spectrum(x=ref_x, data=np.ones(5))
    ref_id = smap.add_spectra(ref_spec, key="ref1")

    resample(smap, ref_spectrum_id=ref_id)
    assert len(smap.x) == 5
    np.testing.assert_allclose(smap.x, ref_x)


def test_resample_ref_spectrum_id_multi_roi():
    x = np.linspace(100.0, 400.0, 31)
    data = np.ones((2, 31), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=1)

    ref_x = np.array([100.0, 150.0, 200.0, 300.0, 350.0, 400.0])
    ref_roi = np.array([[100.0, 200.0], [300.0, 400.0]])
    ref_spec = Spectrum(x=ref_x, data=np.ones(6), roi_x=ref_roi)
    ref_id = smap.add_spectra(ref_spec, key="ref_multi")

    resample(smap, ref_spectrum_id=ref_id)
    assert len(smap.x) == 6
    np.testing.assert_allclose(smap.x, ref_x)
    np.testing.assert_allclose(smap.roi_x, ref_roi)


def test_resample_rejects_non_external_reference():
    x = np.linspace(100.0, 200.0, 10)
    data = np.ones((4, 10), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=2)
    smap.masks["m1"] = smap.new_mask(idxs=[0, 1])

    # Mask, image and cursor spectra are derived from the map and always share its
    # spectral axis, so they cannot define a new target grid.
    np.testing.assert_allclose(smap.get_spectrum(mask="m1").x, smap.x)

    with pytest.raises(ValueError, match="Reference spectrum 'm1' not found"):
        resample(smap, ref_spectrum_id="m1")


def test_resample_mutually_exclusive_validation():
    x = np.linspace(100.0, 200.0, 10)
    data = np.ones((2, 10), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=1)

    # Missing all grid params
    with pytest.raises(ValueError, match="Exactly one of 'spectrum_step', 'ref_spectrum_id', or 'x_grid'"):
        resample(smap)

    # Providing both spectrum_step and ref_spectrum_id
    with pytest.raises(ValueError, match="Exactly one of 'spectrum_step', 'ref_spectrum_id', or 'x_grid'"):
        resample(smap, spectrum_step=5.0, ref_spectrum_id="ref1")

    # Providing both spectrum_step and x_grid
    with pytest.raises(ValueError, match="Exactly one of 'spectrum_step', 'ref_spectrum_id', or 'x_grid'"):
        resample(smap, spectrum_step=5.0, x_grid=[100.0, 150.0])

    # Providing both ref_spectrum_id and x_grid
    with pytest.raises(ValueError, match="Exactly one of 'spectrum_step', 'ref_spectrum_id', or 'x_grid'"):
        resample(smap, ref_spectrum_id="ref1", x_grid=[100.0, 150.0])


def test_resample_nonexistent_ref_spectrum_id():
    x = np.linspace(100.0, 200.0, 10)
    data = np.ones((2, 10), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=1)

    with pytest.raises(ValueError, match="Reference spectrum 'nonexistent' not found"):
        resample(smap, ref_spectrum_id="nonexistent")


def test_step_resample_params_pydantic_model():
    p1 = StepResampleParams(spectrum_step=2.5)
    assert p1.spectrum_step == 2.5

    p2 = StepResampleParams(ref_spectrum_id="my_ref")
    assert p2.ref_spectrum_id == "my_ref"

    with pytest.raises(ValueError, match="Exactly one of 'spectrum_step' or 'ref_spectrum_id'"):
        StepResampleParams()

    with pytest.raises(ValueError, match="Exactly one of 'spectrum_step' or 'ref_spectrum_id'"):
        StepResampleParams(spectrum_step=1.0, ref_spectrum_id="my_ref")


def test_resample_whittaker_and_spline_with_roi_x_1d_and_2d():
    x = np.linspace(100.0, 300.0, 21)
    data = np.ones((2, 21), dtype=float)

    # 1D roi_x with spectrum_step
    smap_spline = SpectralMap(x=x.copy(), data=data.copy(), img_width=2, img_height=1)
    resample(smap_spline, method="spline", spectrum_step=10.0, roi_x=[120.0, 200.0])
    assert len(smap_spline.x) == 9
    assert smap_spline.x[0] == pytest.approx(120.0)
    assert smap_spline.x[-1] == pytest.approx(200.0)

    smap_whittaker = SpectralMap(x=x.copy(), data=data.copy(), img_width=2, img_height=1)
    resample(smap_whittaker, method="whittaker", spectrum_step=10.0, roi_x=[120.0, 200.0])
    assert len(smap_whittaker.x) == 9
    assert smap_whittaker.x[0] == pytest.approx(120.0)
    assert smap_whittaker.x[-1] == pytest.approx(200.0)

    # 2D multi-region roi_x with whittaker
    smap_multi = SpectralMap(x=x.copy(), data=data.copy(), img_width=2, img_height=1)
    resample(smap_multi, method="whittaker", spectrum_step=10.0, roi_x=[[120.0, 160.0], [240.0, 280.0]])
    assert len(smap_multi.x) == 10
    np.testing.assert_allclose(smap_multi.roi_x, [[120.0, 160.0], [240.0, 280.0]])


def test_resample_flat_x_grid_is_restricted_by_roi_x():
    x = np.linspace(100.0, 300.0, 21)
    data = np.ones((2, 21), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=1)

    resample(smap, method="spline", x_grid=np.linspace(100.0, 300.0, 41), roi_x=[150.0, 250.0])
    assert smap.x[0] == pytest.approx(150.0)
    assert smap.x[-1] == pytest.approx(250.0)
    assert len(smap.x) == 21
    np.testing.assert_allclose(smap.roi_x, [[150.0, 250.0]])


def test_resample_regioned_x_grid_ignores_roi_x():
    """A grid given as a list of regions already defines its ROIs; roi_x is ignored."""
    x = np.linspace(100.0, 300.0, 21)
    data = np.ones((2, 21), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=1)

    grid = [np.linspace(100.0, 140.0, 5), np.linspace(240.0, 280.0, 5)]
    resample(smap, method="spline", x_grid=grid, roi_x=[150.0, 250.0])
    assert len(smap.x) == 10
    np.testing.assert_allclose(smap.roi_x, [[100.0, 140.0], [240.0, 280.0]])


def test_resample_roi_tol():
    x = np.linspace(100.0, 200.0, 11)  # step 10
    grid = np.arange(100.0, 205.1, 5.0)  # overshoots the data by 5

    smap = SpectralMap(x=x, data=np.ones((2, 11)), img_width=2, img_height=1)
    resample(smap, x_grid=grid)  # default tolerance is one spectral step
    assert smap.x[-1] == pytest.approx(205.0)

    smap = SpectralMap(x=x, data=np.ones((2, 11)), img_width=2, img_height=1)
    with pytest.raises(ValueError, match="do not fully match"):
        resample(smap, x_grid=grid, roi_tol=1.0)
