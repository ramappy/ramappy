import numpy as np
import pytest

from ramappy.analysis.nfindr.step import nfindr
from ramappy.analysis.pca import principal_components
from ramappy.analysis.spectral_fitting import spectral_fitting
from ramappy.core.images2d import Image2D
from ramappy.core.pipeline import PipelineStepRegistry
from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum

H, W, N = 6, 6, 50


def _make_map() -> SpectralMap:
    rng = np.random.default_rng(0)
    x = np.linspace(400.0, 1800.0, N)
    endmembers = np.stack([np.exp(-0.5 * ((x - c) / 60.0) ** 2) for c in (700.0, 1100.0, 1500.0)])
    data = rng.dirichlet(np.ones(3), size=H * W) @ endmembers + 0.01 * rng.random((H * W, N))
    spectral_map = SpectralMap(x=x, data=data, img_width=W, img_height=H, ignore_sort=True)
    spectral_map.masks["top"] = spectral_map.new_mask(idxs=np.arange(H * W // 2), name="top")
    return spectral_map


def _outside_top() -> np.ndarray:
    outside = np.ones(H * W, dtype=bool)
    outside[: H * W // 2] = False
    return outside.reshape(H, W)


@pytest.mark.parametrize(
    ("run", "image_id"),
    [
        (lambda sm: nfindr(sm, n_endmembers=3, mask="top", res_id="nf"), "nf_0"),
        (lambda sm: principal_components(sm, pca_n_components=2, mask="top", res_id="pc"), "pc_0"),
        (
            lambda sm: spectral_fitting(sm, ref_spectra=Spectrum(x=sm.x, data=sm.data[:1]), mask="top", res_id="sf"),
            "sf",
        ),
    ],
)
def test_masked_analysis_images_are_nan_outside_the_mask(run, image_id):
    spectral_map = _make_map()

    run(spectral_map)

    raw = np.ma.getdata(spectral_map.images[image_id].data)
    assert np.isnan(raw[_outside_top()]).all()


def _map_with_masked_abundance(underlying: float) -> SpectralMap:
    spectral_map = _make_map()
    values = np.full((H, W), underlying)
    values[~_outside_top()] = np.linspace(0.0, 2.0, H * W // 2)
    abundance = np.ma.masked_array(values, mask=_outside_top())
    spectral_map.images["ab"] = Image2D(data=abundance, name="abundance")
    spectral_map.add_spectra(Spectrum(x=spectral_map.x, data=np.linspace(0.1, 0.5, N)[np.newaxis, :]), key="ref")
    return spectral_map


@pytest.mark.parametrize("underlying", [1e30, np.nan])
def test_subtract_scaled_spectrum_leaves_pixels_outside_the_abundance_mask_unchanged(underlying):
    spectral_map = _map_with_masked_abundance(underlying)
    before = spectral_map.cube.copy()
    reference = np.linspace(0.1, 0.5, N)

    PipelineStepRegistry.get_step("subtract_scaled_spectrum").run(
        spectral_map,
        {"abundance_image_id": "ab", "spectrum_id": "ref", "scale_abundance": True, "invert_abundance": False},
    )

    after = spectral_map.cube
    outside = _outside_top()
    assert np.isfinite(after).all()
    np.testing.assert_array_equal(after[outside], before[outside])
    # inside the mask the coefficient is scaled on the valid pixels only: 0 for the first, 1 for the last
    inside_rows, inside_cols = np.nonzero(~outside)
    np.testing.assert_allclose(after[inside_rows[0], inside_cols[0]], before[inside_rows[0], inside_cols[0]], atol=1e-6)
    np.testing.assert_allclose(
        after[inside_rows[-1], inside_cols[-1]], before[inside_rows[-1], inside_cols[-1]] - reference, atol=1e-5
    )
