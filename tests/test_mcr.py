from inspect import signature

import numpy as np
import pytest

from ramappy.analysis.mcr import StepMCRParams, mcr, mcr_regressor_map
from ramappy.core.spectral_map import SpectralMap


def _make_mixed_hsi(*, img_side: int = 4, n_wn: int = 60) -> SpectralMap:
    x = np.linspace(400.0, 1800.0, n_wn, dtype=np.float32)

    # Two synthetic endmembers
    e1 = np.exp(-0.5 * ((x - 900.0) / 40.0) ** 2)
    e2 = np.exp(-0.5 * ((x - 1400.0) / 60.0) ** 2)

    n_pixels = img_side * img_side
    intensities = np.empty((n_pixels, n_wn), dtype=np.float32)

    # Spatially varying mixture weights
    for p in range(n_pixels):
        w = p / (n_pixels - 1)
        intensities[p] = (1.0 - w) * e1 + w * e2

    return SpectralMap(x=x, data=intensities, img_width=img_side, img_height=img_side, ignore_sort=True)


def test_mcr_creates_images_group():
    spectral_map = _make_mixed_hsi()
    mcr(
        spectral_map,
        initial_estimate_method="SVD",
        n_components=2,
        res_id="testmcr",
    )

    assert "testmcr" in spectral_map.images_group
    group = spectral_map.images_group["testmcr"]
    assert group.name == "MCR"
    assert len(group.elem_ids) == 2

    for img_id in group.elem_ids:
        assert img_id in spectral_map.images
        img = spectral_map.images[img_id]
        assert img.data.shape == spectral_map.map_shape
        assert img.spectrum is not None
        assert img.spectrum.x.shape[0] == spectral_map.x.shape[0]


def test_mcr_reference_init_requires_non_empty_reference_list():
    spectral_map = _make_mixed_hsi()

    with pytest.raises(ValueError, match="reference_spectra"):
        mcr(
            spectral_map,
            initial_estimate_method="reference_spectra",
            reference_spectra=[],
            res_id="testmcr_ref",
        )


def test_mcr_regressor_map_does_not_mutate_kwargs():
    kwargs = {"alpha": 0.5, "positive": True, "max_iter": 50}
    kwargs_before = dict(kwargs)

    _ = mcr_regressor_map("LASSO", kwargs)

    assert kwargs == kwargs_before


def test_mcr_supports_float32_work_dtype():
    spectral_map = _make_mixed_hsi()

    mcr(
        spectral_map,
        initial_estimate_method="SVD",
        n_components=2,
        work_dtype="float32",
        res_id="testmcr_f32",
    )

    assert "testmcr_f32" in spectral_map.images_group


def test_mcr_function_default_matches_registered_step_default():
    work_dtype_default = signature(mcr).parameters["work_dtype"].default

    assert work_dtype_default == StepMCRParams.model_fields["work_dtype"].default
