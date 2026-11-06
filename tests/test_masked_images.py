import numpy as np
import pytest

from ramappy.analysis.nfindr.step import nfindr
from ramappy.analysis.pca import principal_components
from ramappy.analysis.spectral_fitting import spectral_fitting
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
def test_masked_analysis_images_are_nan_outside_the_mask(run, image_id, monkeypatch):
    # make uninitialised buffers deterministic garbage, as on a busy process
    monkeypatch.setattr(np, "empty", lambda shape, dtype=float, **kw: np.full(shape, 1e30, dtype=dtype))
    spectral_map = _make_map()

    run(spectral_map)

    raw = np.ma.getdata(spectral_map.images[image_id].data)
    assert np.isnan(raw[_outside_top()]).all()
