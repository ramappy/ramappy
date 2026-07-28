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


def test_resample_ref_spectrum_id_mask():
    x = np.linspace(100.0, 200.0, 10)
    data = np.ones((4, 10), dtype=float)
    smap = SpectralMap(x=x, data=data, img_width=2, img_height=2)
    smap.masks["m1"] = smap.new_mask(idxs=[0, 1])

    # Mask spectrum has same x axis initially
    ref_spec = smap.get_spectrum(mask="m1")
    assert ref_spec.x is not None

    resample(smap, ref_spectrum_id="m1")
    np.testing.assert_allclose(smap.x, ref_spec.x)


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
