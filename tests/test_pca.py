from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from ramappy.analysis.pca import principal_components
from ramappy.core.spectral_map import SpectralMap
from ramappy.processing.normalize import StepNormalizeParams, _normalize_intensities


def _assert_allclose_up_to_sign(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-6, rtol: float = 1e-6):
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    if np.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    np.testing.assert_allclose(actual, -expected, atol=atol, rtol=rtol)


def test_pca_creates_masked_images_and_near_zero_residuals_for_rank1_data():
    x = np.linspace(200.0, 1800.0, 32)
    basis = np.exp(-0.5 * ((x - 900.0) / 80.0) ** 2).astype(np.float64)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    intensities = weights[:, None] * basis[None, :]

    spectral_map = SpectralMap(x=x, data=intensities, img_width=2, img_height=2, ignore_sort=True)
    spectral_map.masks["diag"] = spectral_map.new_mask(idxs=np.array([0, 3], dtype=int))

    principal_components(
        spectral_map,
        mask="diag",
        pca_n_components=1,
        compute_residuals=True,
        res_id="pcamasked",
    )

    assert "pcamasked" in spectral_map.images_group
    group = spectral_map.images_group["pcamasked"]
    assert group.name == "PCA"
    assert len(group.elem_ids) == 2

    residual_img = spectral_map.images["pcamasked_residuals"]
    pc_img = spectral_map.images["pcamasked_0"]

    expected_mask = np.array([[False, True], [True, False]])
    assert np.ma.isMaskedArray(residual_img.data)
    assert np.ma.isMaskedArray(pc_img.data)
    np.testing.assert_array_equal(np.asarray(residual_img.data.mask), expected_mask)
    np.testing.assert_array_equal(np.asarray(pc_img.data.mask), expected_mask)

    np.testing.assert_allclose(np.asarray(residual_img.data.compressed()), 0.0, atol=1e-6, rtol=1e-6)
    assert pc_img.spectrum is not None
    assert pc_img.spectrum.x.shape[0] == x.shape[0]
    assert pc_img.spectrum.metadata["explained_variance"] == pytest.approx(1.0)


def test_pca_with_roi_and_normalization_matches_sklearn_reference_up_to_sign():
    rng = np.random.default_rng(7)
    x = np.linspace(100.0, 1200.0, 12)
    intensities = np.abs(rng.normal(size=(6, x.size))).astype(np.float64) + 0.2

    spectral_map = SpectralMap(x=x, data=intensities.copy(), img_width=3, img_height=2, ignore_sort=True)
    roi_x = [[float(x[2]), float(x[8])]]
    normalize_kwargs = StepNormalizeParams(norm="l2", by_pixel=True)

    principal_components(
        spectral_map,
        roi_x=roi_x,
        pca_n_components=1,
        normalize_kwargs=normalize_kwargs,
        res_id="pcaroi",
    )

    group = spectral_map.images_group["pcaroi"]
    assert group.name == "PCA"
    assert group.elem_ids == ["pcaroi_0"]
    assert "pcaroi_residuals" not in spectral_map.images

    idxs, x_idx, _pixel_idx = spectral_map.get_indices(None, roi_x)
    adapted_roi_x = spectral_map.adapt_roi_x(roi_x)
    expected_intensities = _normalize_intensities(
        intensities[idxs],
        x[x_idx],
        roi_x=adapted_roi_x,
        **normalize_kwargs.model_dump(exclude_none=True),
    )

    pca = PCA(n_components=1, random_state=0)
    expected_scores = pca.fit_transform(expected_intensities)[:, 0]
    expected_loadings = (pca.components_.T * np.sqrt(pca.explained_variance_))[:, 0]

    pc_img = spectral_map.images["pcaroi_0"]
    actual_scores = np.asarray(pc_img.data).reshape(-1)
    _assert_allclose_up_to_sign(actual_scores, expected_scores)

    assert pc_img.spectrum is not None
    np.testing.assert_allclose(pc_img.spectrum.x, x[x_idx])
    _assert_allclose_up_to_sign(np.asarray(pc_img.spectrum.data).reshape(-1), expected_loadings)
    assert pc_img.spectrum.metadata["explained_variance"] == pytest.approx(float(pca.explained_variance_ratio_[0]))
